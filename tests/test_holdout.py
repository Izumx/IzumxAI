import json
import tempfile
import unittest
from pathlib import Path

from izumxai.eval import holdout


class FrozenHoldout(unittest.TestCase):
    def test_external_check_executes_candidate_and_rejects_wrong_result(self):
        task = {"kind": "code", "function": "double", "prompt": "def double(x):\n    pass\n",
                "test_code": "def check(candidate):\n    assert candidate(3) == 6\n"}
        self.assertTrue(holdout.score(task, "def double(x):\n    return x * 2"))
        self.assertFalse(holdout.score(task, "def double(x):\n    return 0"))

    def test_reference_answers_pass_frozen_cases(self):
        rows = holdout.load_tasks("data/eval/stage2_holdout.jsonl", [])
        self.assertEqual(len(rows), 24)
        for row in rows:
            with self.subTest(task=row["id"]):
                answer = row.get("reference", "Ответ: " + str(row.get("expected", "")))
                self.assertTrue(holdout.score(row, answer))

    def test_training_prompt_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            task_path, train_path = Path(td)/"tasks.jsonl", Path(td)/"train.jsonl"
            task_path.write_text(json.dumps({"id":"test", "kind":"math", "prompt":"  SAME prompt ", "expected":"2"})+"\n", encoding="utf-8")
            train_path.write_text(json.dumps({"messages":[{"role":"user", "content":"same   PROMPT"}]})+"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                holdout.load_tasks(task_path, [train_path])


if __name__ == "__main__":
    unittest.main()
