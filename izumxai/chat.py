"""Чат с обученной IzumxAI-nano (свой чекпоинт).

python -m izumxai.chat --ckpt runs/nano30m/latest.pt --tokenizer data/tokenizer/tokenizer.json
"""

from __future__ import annotations

import argparse

import torch

from .model import Config, IzumxNano
from .tokenizer import BPETokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/nano30m/latest.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--prompt", default="")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = Config(**ckpt["cfg"])
    model = IzumxNano(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(args.tokenizer)

    def reply(text: str) -> str:
        ids = tok.encode(text, add_bos=True)
        x = torch.tensor([ids], device=args.device)
        out = model.generate(x, max_new_tokens=args.max_new, temperature=args.temperature,
                             top_k=40, eos_id=tok.eos_id)
        return tok.decode(out[0].tolist()[len(ids):])

    if args.prompt:
        print(reply(args.prompt))
        return

    print("IzumxAI-nano. Пустая строка — выход.")
    while True:
        try:
            text = input("Ты> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        print("IzumxAI> " + reply(text).strip())


if __name__ == "__main__":
    main()
