"""Publish a release folder plus its GGUFs to Hugging Face.

Generic version of upload_hf_v4.py: the repo id, the release folder holding
README.md/Modelfile and the GGUF slug are all taken from the command line, so a
new version does not need a copy of this script.

    python scripts/upload_hf_release.py --repo Izumx/IzumxAI-9B-v51 --slug izumxai-9b-v51

HF_HUB_DISABLE_XET=1 is recommended while the xet backend is unreliable: it
fails large uploads with "xorb not found".
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. Izumx/IzumxAI-9B-v51")
    ap.add_argument("--slug", required=True, help="e.g. izumxai-9b-v51")
    ap.add_argument("--release-dir", default="", help="defaults to release/hf-<tag>")
    ap.add_argument("--quants", nargs="*", default=["Q4_K_M", "Q5_K_M"])
    ap.add_argument("--only", nargs="*", default=None, help="remote file names to upload")
    ap.add_argument("--overwrite", action="store_true", help="replace existing files")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    api = HfApi(token=token)

    tag = args.slug.replace("izumxai-9b-", "")
    release = Path(args.release_dir) if args.release_dir else ROOT / "release" / f"hf-{tag}"
    files = [
        (release / "README.md", "README.md"),
        (release / "Modelfile", "Modelfile"),
    ]
    for quant in args.quants:
        files.append((
            ROOT / "models" / f"{args.slug}-{quant}" / "gguf"
            / f"{args.slug}-{quant}-{quant}.gguf",
            f"{args.slug}-{quant}.gguf",
        ))

    url = api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)
    print(f"repo: {url}", flush=True)

    for source, dest in files:
        if args.only and dest not in args.only:
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
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"  done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
