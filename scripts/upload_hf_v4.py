"""Publish the v4 release to Hugging Face: repo, card, Modelfile, GGUFs."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "Izumx/IzumxAI-9B-v4"
ROOT = Path(__file__).resolve().parents[1]

FILES = [
    (ROOT / "release" / "hf-v4" / "README.md", "README.md"),
    (ROOT / "release" / "hf-v4" / "Modelfile", "Modelfile"),
    (
        ROOT / "models" / "izumxai-9b-v4-Q4_K_M" / "gguf"
        / "izumxai-9b-v4-Q4_K_M-Q4_K_M.gguf",
        "izumxai-9b-v4-Q4_K_M.gguf",
    ),
    (
        ROOT / "models" / "izumxai-9b-v4-Q5_K_M" / "gguf"
        / "izumxai-9b-v4-Q5_K_M-Q5_K_M.gguf",
        "izumxai-9b-v4-Q5_K_M.gguf",
    ),
]


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    api = HfApi(token=token)

    url = api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
    print(f"repo: {url}", flush=True)

    only = sys.argv[1:] or None
    for source, dest in FILES:
        if only and dest not in only:
            continue
        if not source.exists():
            print(f"missing: {source}", flush=True)
            continue
        size = source.stat().st_size / 1e9
        started = time.time()
        print(f"upload {dest} ({size:.2f} GB)...", flush=True)
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=dest,
            repo_id=REPO,
            repo_type="model",
        )
        minutes = (time.time() - started) / 60
        print(f"  done in {minutes:.1f} min", flush=True)


if __name__ == "__main__":
    main()
