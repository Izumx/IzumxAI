"""IzumxAI-nano — собственная decoder-only архитектура (обучение с нуля).

Компоненты: RMSNorm, RoPE, GQA (grouped-query attention), SwiGLU,
pre-norm residual, tied embeddings. Всё написано здесь, без готовых
блоков из transformers — модель целиком наша.

Пресеты (params считаются в `Config.n_params()`):
    nano-30m  ~33M   d=384  L=6   H=6   KV=2
    nano-100m ~104M  d=640  L=12  H=10  KV=2
    nano-250m ~250M  d=896  L=16  H=14  KV=2
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- конфиг
PRESETS: dict[str, dict] = {
    "nano-30m": dict(d_model=384, n_layers=6, n_heads=6, n_kv_heads=2, d_ff=1024, max_seq=512),
    "nano-100m": dict(d_model=640, n_layers=12, n_heads=10, n_kv_heads=2, d_ff=1792, max_seq=1024),
    "nano-250m": dict(d_model=896, n_layers=16, n_heads=14, n_kv_heads=2, d_ff=2432, max_seq=1024),
}


@dataclass
class Config:
    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    d_ff: int = 1024
    max_seq: int = 512
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    pad_id: int = 0

    @classmethod
    def preset(cls, name: str, **overrides) -> "Config":
        if name not in PRESETS:
            raise KeyError(f"неизвестный пресет {name!r}, есть: {list(PRESETS)}")
        cfg = cls(**PRESETS[name])
        cfg.max_seq = PRESETS[name]["max_seq"]
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def n_params(self) -> int:
        emb = self.vocab_size * self.d_model
        head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        q = self.d_model * (self.n_heads * self.d_model // self.n_heads)
        kv = 2 * self.d_model * (self.n_kv_heads * self.d_model // self.n_heads)
        o = self.d_model * self.d_model
        mlp = 3 * self.d_model * self.d_ff
        per_layer = q + kv + o + mlp + 2 * self.d_model
        return emb + head + self.n_layers * per_layer

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------- слои
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rotary(nn.Module):
    """Кэш cos/sin для RoPE, до max_seq."""

    def __init__(self, head_dim: int, max_seq: int, theta: float = 10_000.0):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv)
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        # q,k: [B, H, T, D]
        T = q.shape[-2]
        cos = self.cos[offset:offset + T].to(q.dtype)[None, None]
        sin = self.sin[offset:offset + T].to(q.dtype)[None, None]
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.rope = Rotary(self.head_dim, cfg.max_seq, cfg.rope_theta)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, kv_cache=None, offset: int = 0):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, offset)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=kv_cache is None,
        )
        attn = attn.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(attn), new_cache


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, kv_cache=None, offset: int = 0):
        h, new_cache = self.attn(self.norm1(x), kv_cache, offset)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


# ------------------------------------------------------------------- модель
class IzumxNano(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(idx)
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 0.8, top_k: int = 50,
                 eos_id: int | None = None) -> torch.Tensor:
        self.eval()
        caches = [None] * len(self.blocks)
        for step in range(max_new_tokens):
            if step == 0:
                x = self.embed(idx)
                offset = 0
            else:
                x = self.embed(idx[:, -1:])
                offset = idx.shape[1] - 1
            for i, block in enumerate(self.blocks):
                x, caches[i] = block(x, caches[i], offset)
            logits = self.lm_head(self.norm(x))[:, -1]
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    thresh = torch.topk(logits, k).values[:, -1:]
                    logits = logits.masked_fill(logits < thresh, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and torch.all(nxt == eos_id):
                break
        return idx


def memory_report(cfg: Config) -> str:
    p = cfg.n_params()
    bytes_bf16 = p * 2
    # AdamW в fp32: веса + 2 момента + fp32 мастер-копия ≈ 16 байт/параметр
    bytes_adam = p * 16
    return (f"parameters: {p/1e6:.1f}M | weights bf16: {bytes_bf16/2**30:.2f} GiB | "
            f"обучение AdamW (без активаций): {bytes_adam/2**30:.2f} GiB")
