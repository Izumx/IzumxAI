"""Dense Qwen3 extension of SMEL FT's frozen-prefix and frozen-head loss path.

The MoE expert/router selection is not used for dense Qwen3. The actual loss
implementation is imported from the pinned upstream checkout, not reimplemented.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

SMEL_COMMIT = "74a4635d5f7733baae9915d291573fae94a0499b"
SMEL_PATH = Path(__file__).resolve().parents[2] / "tools" / "smel-ft"


def load_smel_loss():
    if not (SMEL_PATH / "smel" / "loss.py").is_file():
        raise RuntimeError("SMEL FT checkout missing; run scripts/09_smel_setup.ps1")
    if str(SMEL_PATH) not in sys.path:
        sys.path.insert(0, str(SMEL_PATH))
    from smel.loss import LMHead, chunked_ce
    return LMHead, chunked_ce


def cache_fingerprint(signature, examples, cut):
    # Nested sequences preserve causal boundaries; labels identify supervision.
    payload = {"format": 1, "signature": signature, "cut": cut, "examples": examples,
               "positions": "zero-based per sequence", "padding": "none in stored examples"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class _PrefixReady(Exception):
    def __init__(self, hidden):
        self.hidden = hidden


class _SkipLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


@torch.no_grad()
def capture_prefix(model, ids, cut, attention_mask=None):
    if not 0 <= cut < len(model.model.layers):
        raise ValueError("cut must leave at least one trainable suffix layer")
    was_training = model.training
    model.eval()
    def capture(module, args, kwargs):
        raise _PrefixReady(args[0].detach().clone())
    hook = model.model.layers[cut].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        model.model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
    except _PrefixReady as ready:
        return ready.hidden
    finally:
        hook.remove()
        model.train(was_training)
    raise RuntimeError("Qwen3 prefix boundary was not reached")


def drop_prefix(model, cut):
    if any(p.requires_grad for layer in model.model.layers[:cut] for p in layer.parameters()):
        raise ValueError("cannot drop trainable prefix parameters")
    for index in range(cut):
        model.model.layers[index] = _SkipLayer()


def cached_response_loss(model, hidden, labels, attention_mask=None, *,
                         vocab_chunk=32768, token_chunk=128, denominator=None):
    LMHead, chunked_ce = load_smel_loss()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    states = base.model(inputs_embeds=hidden, attention_mask=attention_mask,
                        use_cache=False).last_hidden_state
    mask = labels[:, 1:] != -100
    targets = labels[:, 1:][mask]
    if not targets.numel():
        raise ValueError("no supervised next-token targets")
    selected = states[:, :-1][mask]
    head = LMHead(weight=base.get_output_embeddings().weight, vocab_chunk=vocab_chunk)
    return chunked_ce(selected, head, targets,
                      denom=targets.numel() if denominator is None else denominator,
                      token_chunk=token_chunk,
                      need_grad=torch.is_grad_enabled() and selected.requires_grad)
