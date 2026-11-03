"""Слияние LoRA + экспорт в GGUF/ollama.

python -m izumxai.distill.merge_export --adapter runs/izumxai-2b-lora/final \
       --out models/izumxai-2b --gguf
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

MODELFILE = """FROM "{gguf}"

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
\"\"\"

PARAMETER stop <|im_end|>
PARAMETER stop <|endoftext|>
PARAMETER num_ctx 8192
"""


def fetch_convert_script(dest: Path) -> Path:
    url = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"скачиваю {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def merge(adapter: str, out_dir: Path, base: str) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    meta_path = Path(adapter).parent / "train_meta.json"
    if meta_path.exists():
        base = json.loads(meta_path.read_text(encoding="utf-8")).get("base_model", base)

    print(f"база: {base}")
    tok = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(out_dir)
    print(f"слитая модель: {out_dir}")
    return out_dir


def to_gguf(model_dir: Path, gguf_dir: Path, quant: str | None) -> Path:
    script = Path("tools/llama_src/convert_hf_to_gguf.py")
    if not script.exists():
        script = Path("tools/llama.cpp/convert_hf_to_gguf.py")
    if not script.exists():
        fetch_convert_script(script)
    out = gguf_dir / "izumxai-2b-f16.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), str(model_dir), "--outfile", str(out), "--outtype", "f16"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    if not quant:
        return out
    exe = (shutil.which("llama-quantize")
           or next(iter(Path("tools/llama_bin").glob("llama-quantize.exe")), None))
    if not exe:
        print("llama-quantize не найден — оставляю f16 "
              "(скачать: https://github.com/ggml-org/llama.cpp/releases, "
              "llama-bXXXXX-bin-win-cpu-x64.zip -> tools/llama_bin)")
        return out
    q = gguf_dir / f"izumxai-2b-{quant}.gguf"
    subprocess.run([str(exe), str(out), str(q), quant], check=True)
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="runs/izumxai-2b-lora/final")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default="models/izumxai-2b")
    ap.add_argument("--gguf", action="store_true")
    ap.add_argument("--quant", default="Q4_K_M", help="Q8_0 / Q4_K_M / ... (пусто = без квантования)")
    ap.add_argument("--name", default="izumxai:2b")
    args = ap.parse_args()

    merged = merge(args.adapter, Path(args.out), args.base)

    if not args.gguf:
        # Ollama умеет импортировать safetensors напрямую (без конвертации)
        modelfile = Path(args.out) / "Modelfile"
        modelfile.write_text(MODELFILE.format(gguf=merged.resolve()), encoding="utf-8")
        print(f"Modelfile (safetensors): {modelfile}")
        print(f"\nдальше:\n  ollama create {args.name} -f \"{modelfile}\"\n  ollama run {args.name}")
        return

    gguf = to_gguf(merged, Path(args.out) / "gguf", args.quant or None)
    modelfile = Path(args.out) / "Modelfile"
    modelfile.write_text(MODELFILE.format(gguf=gguf), encoding="utf-8")
    print(f"Modelfile: {modelfile}")
    print(f"\nдальше:\n  ollama create {args.name} -f \"{modelfile}\"\n  ollama run {args.name}")


if __name__ == "__main__":
    main()
