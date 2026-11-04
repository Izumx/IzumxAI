import copy
import unittest

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from izumxai.distill import smel_dense
from izumxai.distill.common import response_loss


def tiny():
    torch.manual_seed(23)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, attention_dropout=0.0, tie_word_embeddings=True,
    ))
    model.requires_grad_(False)
    return model


class DenseSMEL(unittest.TestCase):
    def test_nonfinite_parity_never_allows_training(self):
        from izumxai.distill.train_smel import validate_parity
        normal={'full_loss':0.5,'cached_loss':0.5,'relative_lora_gradient_error':0.0}
        validate_parity(normal)
        for key in normal:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_parity({**normal,key:float('nan')})

    def test_cache_boundary_preserves_full_forward_at_multiple_cuts(self):
        ids = torch.tensor([[1, 2, 3, 4]])
        for cut in (0, 1, 3):
            with self.subTest(cut=cut):
                model = tiny().eval()
                with torch.no_grad():
                    expected = model(input_ids=ids).logits
                hidden = smel_dense.capture_prefix(model, ids, cut)
                smel_dense.drop_prefix(model, cut)
                with torch.no_grad():
                    actual = model(inputs_embeds=hidden).logits
                torch.testing.assert_close(actual, expected)

    def test_upper_lora_gradients_match_full_forward_with_padding(self):
        model = tiny()
        ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 0]])
        attention = ids.ne(0).long()
        labels = torch.tensor([[-100,-100,-100,4,5],[-100,-100,6,7,-100]])
        hidden = smel_dense.capture_prefix(model, ids, 2, attention)
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
            target_modules=['q_proj','v_proj'], layers_to_transform=[2,3],
            bias='none', task_type='CAUSAL_LM'))
        # Nonzero B exercises both LoRA factors, not only the zero-initialized path.
        with torch.no_grad():
            for name,p in model.named_parameters():
                if 'lora_B' in name: p.normal_(0,0.01)
        expected = response_loss(model, ids, labels, attention)
        (expected*0.7).backward()
        gradients = {n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        smel_dense.drop_prefix(model.get_base_model(), 2)
        actual = smel_dense.cached_response_loss(model, hidden.requires_grad_(), labels,
                                                  attention, vocab_chunk=11, token_chunk=2)
        (actual*0.7).backward()
        torch.testing.assert_close(actual, expected)
        for name,p in model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p.grad, gradients[name], atol=1e-6, rtol=2e-5)

    def test_cache_identity_includes_boundaries_labels_weights_and_cut(self):
        common = {'weights':'sha-A','template':'T','dtype':'float32','backend':'test'}
        first = [{'input_ids':[1,2], 'labels':[-100,2]}, {'input_ids':[3], 'labels':[3]}]
        changed_boundaries = [{'input_ids':[1], 'labels':[-100]}, {'input_ids':[2,3], 'labels':[2,3]}]
        original = smel_dense.cache_fingerprint(common, first, 2)
        self.assertNotEqual(original, smel_dense.cache_fingerprint(common, changed_boundaries, 2))
        changed = copy.deepcopy(first); changed[0]['labels'][1] = -100
        self.assertNotEqual(original, smel_dense.cache_fingerprint(common, changed, 2))
        self.assertNotEqual(original, smel_dense.cache_fingerprint({**common,'weights':'sha-B'}, first, 2))
        self.assertNotEqual(original, smel_dense.cache_fingerprint(common, first, 3))


if __name__ == '__main__':
    unittest.main()
