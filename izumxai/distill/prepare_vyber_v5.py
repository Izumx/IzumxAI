"""Отбор SFT-примеров из полного Vyber07/cyber-security (v5).

В отличие от `prepare_vyber.py` не качает ничего по списку, а читает уже
скачанные большие файлы (`download_vyber_v5.py`), стримит их построчно и
держит на каждый домен не первые N записей, а N с наименьшим хешем — так
выборка не смещена к началу файла.

    python -m izumxai.distill.prepare_vyber_v5 --out data/distill/vyber_v5.jsonl

Фильтры те же, что в v4 (без <think>, без отказов, без битых код-блоков,
лимит длины), плюс отдельный дедуп против данных v4 и более строгий минимум
длины ответа.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .prepare_vyber import _domain_of, _iter_json_records, _normalize, _to_messages

REFUSAL_MARKERS = (
    "as an ai language model",
    "as a language model",
    "i cannot assist with",
    "i can't assist with",
    "i cannot help with",
    "i can't help with",
    "i am unable to provide",
    "i'm unable to provide",
    "i cannot provide",
    "i can't provide",
    "i must decline",
    "against my guidelines",
    "i'm sorry, but i can",
    "i am sorry, but i can",
    "cannot fulfill this request",
    "can't fulfill this request",
)

MALFORMED_FENCE = re.compile(r"```[A-Za-z0-9_+#.-]{1,12}[A-Za-z0-9_(]")
ONLY_URLS = re.compile(r"^(?:https?://\S+\s*)+$")


def _hash_int(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")


def _user_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == "user")


def _quality_ok(
    messages: list[dict[str, str]],
    min_answer: int,
    max_total: int,
) -> str:
    """Возвращает причину отказа или пустую строку, если пример годен."""
    user_chars = sum(len(m["content"]) for m in messages if m["role"] == "user")
    answer_chars = sum(len(m["content"]) for m in messages if m["role"] == "assistant")
    if not (20 <= user_chars <= 4000):
        return "user_len"
    if answer_chars < min_answer:
        return "answer_short"
    if user_chars + answer_chars > max_total:
        return "too_long"
    answer = messages[-1]["content"]
    low = answer.lower()
    if any(marker in low for marker in REFUSAL_MARKERS):
        return "refusal"
    if any("<think>" in m["content"].lower() for m in messages):
        return "think"
    if MALFORMED_FENCE.search(answer):
        return "fence"
    if ONLY_URLS.match(answer.strip()):
        return "urls"
    if len(set(answer)) < 30:
        return "low_entropy"
    return ""


def _load_seen(paths: Iterable[Path]) -> tuple[set[str], set[str]]:
    seen_user: set[str] = set()
    seen_pair: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = row.get("messages")
                if not isinstance(messages, list):
                    continue
                user = "\n".join(
                    m.get("content", "") for m in messages if m.get("role") == "user"
                )
                answer = next(
                    (
                        m.get("content", "")
                        for m in reversed(messages)
                        if m.get("role") == "assistant"
                    ),
                    "",
                )
                seen_user.add(hashlib.sha1(user.strip().lower().encode()).hexdigest())
                seen_pair.add(
                    hashlib.sha1((user + "\x00" + answer).encode()).hexdigest()
                )
    return seen_user, seen_pair


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/external/vyber/_raw_v5")
    ap.add_argument("--out", default="data/distill/vyber_v5.jsonl")
    ap.add_argument(
        "--exclude-against",
        default="data/distill/stage5_v4_train.jsonl,data/distill/stage5_v4_val.jsonl",
        help="через запятую: файлы, примеры которых уже учили (дедуп по user/паре)",
    )
    ap.add_argument("--per-domain", type=int, default=1400)
    ap.add_argument(
        "--caps",
        default="",
        help="переопределить квоту домена: name=count,name=count",
    )
    ap.add_argument("--total-cap", type=int, default=14000)
    ap.add_argument("--min-answer", type=int, default=200)
    ap.add_argument("--max-total-chars", type=int, default=6500)
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    paths = sorted(p for p in raw_dir.rglob("*.jsonl"))
    if not paths:
        raise SystemExit(f"в {raw_dir} нет jsonl — сначала download_vyber_v5.py")

    excluded = [
        Path(item.strip())
        for item in args.exclude_against.split(",")
        if item.strip()
    ]
    seen_user, seen_pair = _load_seen(excluded)
    print(f"[v5] уже знакомых user: {len(seen_user)}, пар: {len(seen_pair)}")

    caps: dict[str, int] = {}
    for item in args.caps.split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            caps[key.strip()] = int(value)

    # per domain -> min-heap (score, seq, row)
    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {}
    stats = {
        "records": 0,
        "normalized": 0,
        "dupes": 0,
        "rejected": 0,
        "kept": 0,
    }
    reasons: dict[str, int] = {}
    seq = 0

    for path in paths:
        domain = _domain_of(path.name)
        print(f"[v5] {path.name} ({path.stat().st_size / 2**30:.2f} GB) домен={domain}")
        for record in _iter_json_records(path):
            stats["records"] += 1
            messages = _to_messages(record)
            if messages is None:
                continue
            normalized = _normalize(messages)
            if normalized is None:
                stats["normalized"] += 0  # _normalize уже отфильтровал
                continue
            stats["normalized"] += 1
            reason = _quality_ok(
                normalized, args.min_answer, args.max_total_chars
            )
            if reason:
                stats["rejected"] += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            user_text = _user_text(normalized)
            user_key = hashlib.sha1(user_text.strip().lower().encode()).hexdigest()
            pair_key = hashlib.sha1(
                (user_text + "\x00" + normalized[-1]["content"]).encode()
            ).hexdigest()
            if user_key in seen_user or pair_key in seen_pair:
                stats["dupes"] += 1
                continue
            seen_user.add(user_key)
            seen_pair.add(pair_key)
            score = _hash_int(pair_key)
            cap = caps.get(domain, args.per_domain)
            if cap <= 0:
                continue
            heap = heaps.setdefault(domain, [])
            seq += 1
            row = {
                "messages": normalized,
                "task_type": f"vyber5:{domain}",
                "source": f"Vyber07/cyber-security/{path.name}",
            }
            if len(heap) < cap:
                heapq.heappush(heap, (-score, seq, row))
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, (-score, seq, row))

    rows: list[dict[str, Any]] = []
    for heap in heaps.values():
        rows.extend(item[2] for item in heap)
    rows.sort(key=lambda row: _hash_int(_user_text(row["messages"])))
    if 0 < args.total_cap < len(rows):
        # берём одинаковую долю с каждого домена, чтобы не потерять редкие
        per_domain: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            per_domain.setdefault(row["task_type"], []).append(row)
        factor = args.total_cap / len(rows)
        sampled: list[dict[str, Any]] = []
        for group in per_domain.values():
            take = max(1, int(len(group) * factor))
            sampled.extend(group[:take])
        sampled.sort(key=lambda row: _hash_int(_user_text(row["messages"])))
        rows = sampled[: args.total_cap]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats["kept"] = len(rows)
    top = sorted(
        ((row["task_type"], 1) for row in rows), key=lambda item: item[0]
    )
    counts: dict[str, int] = {}
    for name, _ in top:
        counts[name] = counts.get(name, 0) + 1
    print(json.dumps({**stats, "reject_reasons": reasons}, indent=2, ensure_ascii=False))
    print(
        "по доменам:",
        ", ".join(f"{k.split(':', 1)[1]}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])),
    )
    print(f"-> {out} ({out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
