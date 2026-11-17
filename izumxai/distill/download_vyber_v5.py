"""Download the large text sources of Vyber07/cyber-security for v5."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "Vyber07/cyber-security"
TARGET = Path(r"D:\IzumxAI\repo\data\external\vyber\_raw_v5")

FILES = [
    "vulnerability-intelligence/cybersec_master.jsonl",
    "220426-CyberSec-Dataset_escaped.jsonl",
    "CyberSec-Dataset_escaped.jsonl",
    "vulnerability-intelligence/Cybersecurity-ShareGPT.jsonl",
    "code_train.jsonl",
    "code_train_no_reasoning.jsonl",
]


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        print(f"[get] {name}", flush=True)
        path = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=name,
            local_dir=str(TARGET),
            token=token,
        )
        size = Path(path).stat().st_size / 2**20
        print(f"[ok ] {path} ({size:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
