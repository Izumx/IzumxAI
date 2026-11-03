"""Данные для pretrain: сырой текст -> бинарный поток токенов -> батчи."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import BPETokenizer


def read_corpus(raw_dir: str | Path) -> list[str]:
    """Читает все .txt/.md/.jsonl из папки. jsonl: поле "text"."""
    docs: list[str] = []
    for path in sorted(Path(raw_dir).rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in {".txt", ".md"}:
            docs.append(path.read_text(encoding="utf-8", errors="replace"))
        elif path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and "text" in obj:
                    docs.append(str(obj["text"]))
    return docs


def build_bin(docs: list[str], tok: BPETokenizer, out_path: str | Path) -> int:
    """Токенизирует корпус и складывает id в .bin (uint16/uint32)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if tok.vocab_size <= 65535 else np.uint32
    total = 0
    with out_path.open("wb") as f:
        for i, doc in enumerate(docs):
            ids = tok.encode(doc, add_bos=True, add_eos=True)
            np.asarray(ids, dtype=dtype).tofile(f)
            total += len(ids)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(docs)} документов, {total / 1e6:.2f}M токенов", flush=True)
    meta = {"dtype": np.dtype(dtype).name, "tokens": total, "vocab_size": tok.vocab_size}
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return total


class TokenDataset(Dataset):
    def __init__(self, bin_path: str | Path, block_size: int, stride: int | None = None):
        meta = json.loads(Path(bin_path).with_suffix(".meta.json").read_text(encoding="utf-8"))
        dtype = np.dtype(meta["dtype"])
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        self.block = block_size
        self.stride = stride or block_size

    def __len__(self) -> int:
        return max(0, (len(self.data) - self.block - 1) // self.stride + 1)

    def __getitem__(self, i: int):
        start = i * self.stride
        chunk = torch.from_numpy(self.data[start:start + self.block + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]
