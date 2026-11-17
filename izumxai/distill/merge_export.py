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


MODELFILE_MIN = """FROM "{gguf}"

PARAMETER num_ctx 8192
PARAMETER temperature 0.7
PARAMETER repeat_penalty 1.05
PARAMETER num_predict 1024
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
    from transformers import AutoTokenizer

    from .common import auto_model_class

    # transformers rejects Windows relative paths ("runs\...\final") as repo ids,
    # so any existing local directory must be handed over as an absolute path.
    if Path(adapter).is_dir():
        adapter = str(Path(adapter).resolve())

    meta_path = Path(adapter).parent / "train_meta.json"
    if meta_path.exists():
        base = json.loads(meta_path.read_text(encoding="utf-8")).get("base_model", base)

    print(f"база: {base}")
    tok = AutoTokenizer.from_pretrained(adapter)
    loader = auto_model_class(base)
    model = loader.from_pretrained(
        base, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(out_dir)
    print(f"слитая модель: {out_dir}")
    return out_dir


def to_gguf(model_dir: Path, gguf_dir: Path, quant: str | None, stem: str = "izumxai") -> Path:
    script = Path("tools/llama_src/convert_hf_to_gguf.py")
    if not script.exists():
        script = Path("tools/llama.cpp/convert_hf_to_gguf.py")
    if not script.exists():
        fetch_convert_script(script)
    out = gguf_dir / f"{stem}-f16.gguf"
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
    q = gguf_dir / f"{stem}-{quant}.gguf"
    subprocess.run([str(exe), str(out), str(q), quant], check=True)
    return q


def write_modelfile(out_dir: Path, gguf: Path, source: str = "", name: str = "izumxai:2b",
                    minimal: bool = False) -> Path:
    """Пишет Modelfile. С --modelfile-source берёт готовый шаблон (например, think)
    и заменяет в нём только строку FROM."""
    modelfile = out_dir / "Modelfile"
    # Ollama-сервер разворачивает относительный путь FROM от своего cwd, а не от
    # каталога вызова, и тогда принимает путь за имя модели («invalid model
    # name»). Поэтому в Modelfile всегда пишем абсолютный путь.
    gguf_abs = Path(gguf).resolve()
    if minimal:
        # Шаблон чата остаётся из GGUF (нужно для qwen3_5 с thinking).
        modelfile.write_text(MODELFILE_MIN.format(gguf=gguf_abs), encoding="utf-8")
        print(f"Modelfile (шаблон из GGUF): {modelfile}")
        print(f"\nдальше:\n  ollama create {name} -f \"{modelfile}\"\n  ollama run {name}")
        return modelfile
    if source:
        text = Path(source).read_text(encoding="utf-8")
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().upper().startswith("FROM "):
                lines[i] = f'FROM "{gguf_abs}"'
                break
        else:
            lines.insert(0, f'FROM "{gguf_abs}"')
        modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        modelfile.write_text(MODELFILE.format(gguf=gguf), encoding="utf-8")
    print(f"Modelfile: {modelfile}")
    print(f"\nдальше:\n  ollama create {name} -f \"{modelfile}\"\n  ollama run {name}")
    return modelfile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="runs/izumxai-2b-lora/final")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default="models/izumxai-2b")
    ap.add_argument("--gguf", action="store_true")
    ap.add_argument("--quant", default="Q4_K_M", help="Q8_0 / Q4_K_M / ... (пусто = без квантования)")
    ap.add_argument("--name", default="izumxai:2b")
    ap.add_argument("--modelfile-source", default="",
                    help="взять TEMPLATE/PARAMETER из готового Modelfile (например, models/izumxai-think/Modelfile)")
    ap.add_argument("--merged", action="store_true",
                    help="адаптер уже слит: --base указывает на готовую папку модели")
    ap.add_argument("--stem", default="", help="префикс имени GGUF (по умолчанию имя папки)")
    ap.add_argument("--no-template", action="store_true",
                    help="не переопределять TEMPLATE: взять шаблон чата из GGUF")
    args = ap.parse_args()

    merged = (Path(args.base) if args.merged
              else merge(args.adapter, Path(args.out), args.base))

    if args.gguf:
        gguf = to_gguf(merged, Path(args.out) / "gguf", args.quant or None,
                       args.stem or Path(args.out).name)
    else:
        gguf = merged.resolve()
    write_modelfile(Path(args.out), gguf, args.modelfile_source, args.name,
                    minimal=args.no_template)


if __name__ == "__main__":
    main()
