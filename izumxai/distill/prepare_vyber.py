"""Подготовка внешнего датасета Vyber07/cyber-security к SFT.

Скачивает только «мелкие» JSON/JSONL (<= --max-file-mb), один общий
ShareGPT-файл и первые N строк из cybersec_master.jsonl, приводит
ShareGPT / Instruction-IO / messages-записи к формату {"messages": [...]},
фильтрует мусор и дедуплицирует.

Нужен токен HF с принятыми условиями датасета:
    $env:HF_TOKEN = "hf_..."
    python -m izumxai.distill.prepare_vyber --out data/distill/vyber_sft.jsonl

Результат — JSONL, который потом подмешивается к stage5 в mix_v4.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

REPO_ID = "Vyber07/cyber-security"
SHAREGPT_FILE = "vulnerability-intelligence/Cybersecurity-ShareGPT.jsonl"
MASTER_FILE = "vulnerability-intelligence/cybersec_master.jsonl"

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "bot": "assistant",
}

REFUSAL_MARKERS = (
    "as an ai language model",
    "i cannot assist with",
    "i can't assist with",
    "i am unable to provide",
    "i'm unable to provide",
    "i cannot provide",
    "i can't provide",
    "i must decline",
    "against my guidelines",
)

# ```pythonimport x``` — сломанные блоки кода без переводов строки
MALFORMED_FENCE = re.compile(r"```[A-Za-z0-9_+#.-]{1,12}[A-Za-z0-9_(]")


def _clean(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def _unescape(text: str) -> str:
    """Часть файлов датасета хранит текст с литеральными \\n вместо переводов."""
    if "\\n" not in text and "\\\\" not in text and '\\"' not in text:
        return text
    placeholder = "\x00"
    text = text.replace("\\\\", placeholder)
    text = (
        text.replace("\\n", "\n")
        .replace("\\r", "")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\'", "'")
    )
    return text.replace(placeholder, "\\")


def _to_messages(record: dict[str, Any]) -> list[dict[str, str]] | None:
    """Приводит запись любого из поддерживаемых форматов к messages."""
    if isinstance(record.get("messages"), list):
        out = []
        for message in record["messages"]:
            if not isinstance(message, dict):
                return None
            role = ROLE_MAP.get(str(message.get("role", "")).lower())
            content = message.get("content")
            if role and isinstance(content, str) and content.strip():
                out.append({"role": role, "content": _clean(content)})
        return out or None

    # плоский формат {system, user, assistant} (файлы *_escaped.jsonl)
    if isinstance(record.get("user"), str) and isinstance(record.get("assistant"), str):
        messages = []
        system = record.get("system")
        if isinstance(system, str) and system.strip():
            messages.append({"role": "system", "content": _clean(_unescape(system))})
        messages.append({"role": "user", "content": _clean(_unescape(record["user"]))})
        messages.append(
            {"role": "assistant", "content": _clean(_unescape(record["assistant"]))}
        )
        return messages

    if isinstance(record.get("conversations"), list):
        out = []
        for turn in record["conversations"]:
            if not isinstance(turn, dict):
                return None
            role = ROLE_MAP.get(str(turn.get("from", "")).lower())
            value = turn.get("value")
            if role and isinstance(value, str) and value.strip():
                out.append({"role": role, "content": _clean(value)})
        return out or None

    instruction = record.get("Instruction") or record.get("instruction")
    output = record.get("Output") or record.get("output") or record.get("response")
    if isinstance(instruction, str) and isinstance(output, str) and output.strip():
        extra = record.get("Input") or record.get("input") or ""
        user = instruction.strip()
        if isinstance(extra, str) and extra.strip():
            user = f"{user}\n\n{extra.strip()}"
        messages = [{"role": "user", "content": _clean(user)}]
        system = record.get("System") or record.get("system")
        if isinstance(system, str) and system.strip():
            messages.insert(0, {"role": "system", "content": _clean(system)})
        messages.append({"role": "assistant", "content": _clean(output)})
        return messages

    prompt = record.get("prompt") or record.get("question")
    answer = record.get("response") or record.get("answer")
    if isinstance(prompt, str) and isinstance(answer, str) and answer.strip():
        return [
            {"role": "user", "content": _clean(prompt)},
            {"role": "assistant", "content": _clean(answer)},
        ]
    return None


def _normalize(messages: list[dict[str, str]]) -> list[dict[str, str]] | None:
    """Оставляет один system + чередование user/assistant, требует ответа."""
    system = [m for m in messages if m["role"] == "system"]
    turns = [m for m in messages if m["role"] != "system"]
    if not turns or turns[-1]["role"] != "assistant":
        return None
    # сливаем подряд идущие сообщения одной роли
    merged: list[dict[str, str]] = []
    for message in turns:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1]["content"] += "\n\n" + message["content"]
        else:
            merged.append(dict(message))
    if len(merged) < 2 or merged[0]["role"] != "user":
        return None
    result = ([system[-1]] if system else []) + merged
    user_chars = sum(len(m["content"]) for m in merged if m["role"] == "user")
    answer_chars = sum(len(m["content"]) for m in merged if m["role"] == "assistant")
    if not (20 <= user_chars <= 6000):
        return None
    if not (120 <= answer_chars <= 6000):
        return None
    if user_chars + answer_chars > 6500:
        return None
    answer_lower = result[-1]["content"].lower()
    if any(marker in answer_lower for marker in REFUSAL_MARKERS):
        return None
    # Внутренние reasoning-блоки внешнего датасета часто не совпадают с вопросом
    # (сгенерированы из общего пула) — такие цели учить нельзя.
    if any("<think>" in m["content"].lower() for m in result):
        return None
    if MALFORMED_FENCE.search(result[-1]["content"]):
        return None
    return result


def _iter_json_records(path: Path, max_records: int = 0) -> Iterable[dict[str, Any]]:
    try:
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for index, line in enumerate(handle):
                    if max_records and index >= max_records:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        yield record
                    elif isinstance(record, list):
                        for item in record:
                            if isinstance(item, dict):
                                yield item
            return
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return
    if isinstance(data, dict):
        data = data.get("data") or data.get("records") or [data]
    if isinstance(data, list):
        for item in data[:max_records] if max_records else data:
            if isinstance(item, dict):
                yield item


def _domain_of(name: str) -> str:
    stem = Path(name).name
    stem = stem.split("__")[-1]
    return stem.replace(".jsonl", "").replace(".json", "")


def _download(args: argparse.Namespace, dest: Path) -> list[Path]:
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN") or args.token
    api = HfApi(token=token)
    info = api.dataset_info(REPO_ID, files_metadata=True, token=token)
    files = []
    skipped_dupes = set()
    for sibling in info.siblings:
        name = sibling.rfilename
        size = sibling.size or 0
        if not name.endswith((".json", ".jsonl")) or size == 0:
            continue
        if size > args.max_file_mb * 1_000_000:
            continue
        # датасет содержит побайтовые дубли под разными именами
        key = (name.split("/")[-1], size)
        if key in skipped_dupes:
            continue
        skipped_dupes.add(key)
        files.append(name)
    if SHAREGPT_FILE not in files:
        files.append(SHAREGPT_FILE)
    print(f"[vyber] скачиваю {len(files)} файлов")
    paths = []
    for index, name in enumerate(files, 1):
        target = dest / name.replace("/", "__")
        if not target.exists():
            try:
                cached = hf_hub_download(
                    REPO_ID, name, repo_type="dataset", token=token, local_dir=str(dest / "_raw")
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [{index}/{len(files)}] FAIL {name}: {type(exc).__name__}")
                continue
            paths.append(Path(cached))
        else:
            paths.append(target)
        if index % 50 == 0:
            print(f"  [{index}/{len(files)}] ok")
    return paths


def _stream_master(args: argparse.Namespace, dest: Path) -> Path | None:
    from huggingface_hub import HfFileSystem

    token = os.environ.get("HF_TOKEN") or args.token
    target = dest / "cybersec_master_head.jsonl"
    if target.exists():
        return target
    if not args.master_lines:
        return None
    print(f"[vyber] читаю первые {args.master_lines} строк {MASTER_FILE}")
    fs = HfFileSystem(token=token)
    try:
        with fs.open(f"datasets/{REPO_ID}/{MASTER_FILE}", "r", encoding="utf-8") as handle:
            with open(target, "w", encoding="utf-8") as out:
                for index, line in enumerate(handle):
                    if index >= args.master_lines:
                        break
                    out.write(line)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        return None
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/distill/vyber_sft.jsonl")
    ap.add_argument("--work", default="data/external/vyber")
    ap.add_argument("--token", default="")
    ap.add_argument("--max-file-mb", type=float, default=40.0)
    ap.add_argument("--master-lines", type=int, default=40000)
    ap.add_argument("--max-per-domain", type=int, default=1200)
    ap.add_argument(
        "--exclude-domains",
        default="code_train,code_train_no_reasoning",
        help="через запятую: домены, которые не берём (офтоп/битые think-блоки)",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dest = Path(args.work)
    dest.mkdir(parents=True, exist_ok=True)
    paths = _download(args, dest)
    master = _stream_master(args, dest)
    if master is not None:
        paths.append(master)

    seen_user: set[str] = set()
    seen_pair: set[str] = set()
    per_domain: dict[str, int] = {}
    excluded = {name.strip() for name in args.exclude_domains.split(",") if name.strip()}
    kept: list[dict[str, Any]] = []
    stats = {"records": 0, "normalized": 0, "dupes": 0, "capped": 0, "excluded": 0}

    for path in paths:
        domain = _domain_of(path.name)
        if domain in excluded:
            stats["excluded"] += 1
            continue
        for record in _iter_json_records(path):
            stats["records"] += 1
            messages = _to_messages(record)
            if messages is None:
                continue
            normalized = _normalize(messages)
            if normalized is None:
                continue
            stats["normalized"] += 1
            user_text = "\n".join(m["content"] for m in normalized if m["role"] == "user")
            user_key = hashlib.sha1(user_text.strip().lower().encode()).hexdigest()
            pair_key = hashlib.sha1(
                (user_text + "\x00" + normalized[-1]["content"]).encode()
            ).hexdigest()
            if user_key in seen_user or pair_key in seen_pair:
                stats["dupes"] += 1
                continue
            if per_domain.get(domain, 0) >= args.max_per_domain:
                stats["capped"] += 1
                continue
            seen_user.add(user_key)
            seen_pair.add(pair_key)
            per_domain[domain] = per_domain.get(domain, 0) + 1
            kept.append(
                {
                    "messages": normalized,
                    "task_type": f"vyber:{domain}",
                    "source": f"{REPO_ID}/{path.name}",
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    top = sorted(per_domain.items(), key=lambda item: -item[1])[:25]
    print(json.dumps({**stats, "kept": len(kept), "domains": len(per_domain)}, indent=2))
    print("топ доменов:", ", ".join(f"{name}={count}" for name, count in top))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
