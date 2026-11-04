"""Regression checks for training signals; no downloads or GPU required."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from izumxai.distill.common import build_labeled_example
from izumxai.distill import train_orpo
from izumxai.distill import train_lora
from izumxai.distill import prepare_sft
from izumxai.eval import bench


class TrainingExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained("models/izumxai-2b", local_files_only=True)

    def test_answer_mask_excludes_empty_thinking_prefix(self):
        ex = build_labeled_example(self.tok, [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ], 128)
        answer = self.tok.decode([x for x in ex["labels"] if x != -100])
        self.assertEqual(answer, "4<|im_end|>\n")

    def test_overlength_answer_is_not_silently_trained_as_complete(self):
        ex = build_labeled_example(self.tok, [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello " * 100},
        ], 32)
        self.assertIsNone(ex)

    def test_reasoning_answer_keeps_its_thinking_trace(self):
        ex = build_labeled_example(self.tok, [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "<think>\nсчитаю\n</think>\n\n4"},
        ], 256)
        answer = self.tok.decode([x for x in ex["labels"] if x != -100])
        self.assertTrue(answer.startswith("<think>\nсчитаю\n</think>"))
        self.assertTrue(answer.endswith("4<|im_end|>\n"))


class PreferenceGradients(unittest.TestCase):
    def test_near_certain_responses_have_finite_gradients(self):
        chosen = torch.tensor(-1e-6, requires_grad=True)
        rejected = torch.tensor(-2e-6, requires_grad=True)
        loss, _, _ = train_orpo.orpo_loss(chosen, rejected, 0.1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(chosen.grad))
        self.assertTrue(torch.isfinite(rejected.grad))

    def test_loss_pushes_chosen_up_and_rejected_down(self):
        chosen = torch.tensor(math.log(0.25), requires_grad=True)
        rejected = torch.tensor(math.log(0.5), requires_grad=True)
        loss, _, _ = train_orpo.orpo_loss(chosen, rejected, 0.1)
        loss.backward()
        self.assertLess(chosen.grad.item(), 0)
        self.assertGreater(rejected.grad.item(), 0)

    def test_sequential_backward_matches_full_objective(self):
        torch.manual_seed(7)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, attention_dropout=0.0,
        ))
        pair = {
            "chosen": {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]},
            "rejected": {"input_ids": [1, 2, 5, 6, 7], "labels": [-100, -100, 5, 6, 7]},
        }
        lp_c = train_orpo.response_logp(model, pair["chosen"])
        lp_r = train_orpo.response_logp(model, pair["rejected"])
        loss, _, _ = train_orpo.orpo_loss(lp_c, lp_r, 0.1)
        (loss / 6).backward()
        expected = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        train_orpo.backward_pair(model, pair, lam=0.1, scale=6)
        for name, param in model.named_parameters():
            torch.testing.assert_close(param.grad, expected[name], atol=1e-6, rtol=2e-5)


class ValidationLoss(unittest.TestCase):
    def test_frozen_embedding_cast_preserves_ties_and_fp32_norms(self):
        from izumxai.distill.common import cast_frozen_embeddings
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, tie_word_embeddings=True,
        ))
        model.requires_grad_(False)
        cast_frozen_embeddings(model, torch.bfloat16)
        self.assertEqual(model.get_input_embeddings().weight.dtype, torch.bfloat16)
        self.assertIs(model.get_input_embeddings().weight, model.get_output_embeddings().weight)
        self.assertEqual(model.model.norm.weight.dtype, torch.float32)
        self.assertFalse(model.get_input_embeddings().weight.requires_grad)

    def test_response_projection_matches_full_loss_and_gradients(self):
        from izumxai.distill.common import response_loss
        torch.manual_seed(17)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, attention_dropout=0.0,
        ))
        ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 0]])
        labels = torch.tensor([[-100, -100, -100, 4, 5], [-100, -100, 6, 7, -100]])
        attn = ids.ne(0).long()
        expected_loss = model(input_ids=ids, labels=labels, attention_mask=attn).loss
        expected_loss.backward()
        expected_grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        projected = []
        hook = model.lm_head.register_forward_pre_hook(lambda _, args: projected.append(args[0].shape[1]))
        try:
            actual = response_loss(model, ids, labels, attn)
            actual.backward()
        finally:
            hook.remove()
        torch.testing.assert_close(actual, expected_loss)
        self.assertEqual(projected, [3])
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.grad, expected_grads[name], atol=1e-6, rtol=2e-5)

    def test_validation_is_token_weighted_and_restores_training_mode(self):
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=8, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
            head_dim=8,
        ))
        # Constant logits: P(7)=9/16, P(any other token)=1/16.
        model.lm_head = torch.nn.Linear(8, 8, bias=True)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
            model.lm_head.bias[7] = math.log(9)
        model.train()
        dataset = [
            {"input_ids": [1, 2, 3], "labels": [-100, -100, 3]},
            {"input_ids": [1, 7, 7, 7], "labels": [-100, 7, 7, 7]},
        ]
        result = train_lora.evaluate(model, dataset, pad_id=0, dtype=torch.float32)
        self.assertAlmostEqual(result, (math.log(16) + 3 * math.log(16 / 9)) / 4, places=5)
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


class DatasetSplit(unittest.TestCase):
    def test_duplicate_prompts_cannot_cross_train_validation_boundary(self):
        def item(prompt):
            return {"verified": True, "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "answer"},
            ]}
        rows = [item(" A "), item("a"), item("b"), item("c"), item("d")]
        rows.append({**item("unverified"), "verified": False})
        train, val = prepare_sft.split_records(rows, val_fraction=0.25, seed=42)
        prompts = [x["messages"][0]["content"].strip().casefold() for x in train + val]
        self.assertEqual(len(train), 3)
        self.assertEqual(len(val), 1)
        self.assertCountEqual(prompts, ["a", "b", "c", "d"])
        self.assertEqual((train, val), prepare_sft.split_records(rows, 0.25, 42))


class BenchmarkAccuracy(unittest.TestCase):
    def test_request_errors_and_zero_token_answers_count_in_accuracy(self):
        task = SimpleNamespace(problem="fixture", system="fixture", expected="4",
                               check=lambda text: text == "4")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "bench.json"
            argv = ["bench", "--models", "fixture", "--out", str(out)]
            with patch("sys.argv", argv), patch.object(bench, "build_tasks", return_value={
                "math": [("hard", task)] * 3,
            }), patch.object(bench, "chat", side_effect=[
                ("4", 1.0, 1), RuntimeError("test transport failure"), ("4", 1.0, 0),
            ]):
                bench.main()
            result = json.loads(out.read_text(encoding="utf-8"))["fixture"]
            self.assertAlmostEqual(result["acc"], 2 / 3)
            self.assertEqual(len(result["rows"]), 3)


class Stage4Assembly(unittest.TestCase):
    def test_held_think_rows_leave_training_and_land_in_validation(self):
        from izumxai.distill import assemble_stage4

        def item(prompt, assistant, verified=True):
            return {"verified": verified, "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant},
            ]}

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            think = td / "think.jsonl"
            plain = td / "plain.jsonl"
            plain_val = td / "plain_val.jsonl"
            think.write_text("".join(json.dumps(item(f"t{i}", "<think>\nx\n</think>\n4")) + "\n"
                                     for i in range(4)), encoding="utf-8")
            plain.write_text("".join(json.dumps(item(f"p{i}", "4")) + "\n" for i in range(3)),
                             encoding="utf-8")
            plain_val.write_text(json.dumps(item("p0", "4")) + "\n", encoding="utf-8")

            argv = ["assemble", "--think", str(think), "--plain", str(plain),
                    "--plain-val", str(plain_val), "--out", str(td / "out"),
                    "--repeat-think", "2", "--val-think", "2"]
            with patch("sys.argv", argv):
                assemble_stage4.main()

            train = [json.loads(l) for l in (td / "out/train.jsonl").read_text(encoding="utf-8").splitlines()]
            val = [json.loads(l) for l in (td / "out/val.jsonl").read_text(encoding="utf-8").splitlines()]
            train_prompts = [r["messages"][1]["content"] for r in train]
            val_prompts = [r["messages"][1]["content"] for r in val]

            self.assertEqual(len(val), 3)
            self.assertEqual(sum("<think>" in r["messages"][2]["content"] for r in val), 2)
            self.assertFalse(set(train_prompts) & set(val_prompts))
            # Оставшиеся рассуждения повторяются дважды, обычные — по одному разу.
            train_think = [p for p in train_prompts if p.startswith("t")]
            self.assertEqual(len(train_think), 4)
            self.assertEqual(sorted(set(train_think)), sorted(set(train_prompts) - {"p1", "p2"}))
            self.assertEqual(sorted(train_prompts).count("p1"), 1)
            self.assertEqual(sorted(train_prompts).count("p2"), 1)


class ModelfileExport(unittest.TestCase):
    def test_source_keeps_template_and_replaces_from(self):
        from izumxai.distill import merge_export

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source = td / "source.Modelfile"
            source.write_text(
                'FROM "old.gguf"\n\nTEMPLATE """{{ .Prompt }}"""\n\nPARAMETER stop "<|im_end|>"\n',
                encoding="utf-8",
            )
            gguf = td / "new.gguf"

            modelfile = merge_export.write_modelfile(td, gguf, str(source), "izumxai:test")
            text = modelfile.read_text(encoding="utf-8")

            self.assertIn(f'FROM "{gguf}"', text)
            self.assertNotIn("old.gguf", text)
            self.assertIn("TEMPLATE", text)
            self.assertIn('PARAMETER stop "<|im_end|>"', text)

    def test_without_source_uses_default_template(self):
        from izumxai.distill import merge_export

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            gguf = td / "new.gguf"

            modelfile = merge_export.write_modelfile(td, gguf, "", "izumxai:test")
            text = modelfile.read_text(encoding="utf-8")

            self.assertIn(f'FROM "{gguf}"', text)
            self.assertIn("TEMPLATE", text)


if __name__ == "__main__":
    unittest.main()
