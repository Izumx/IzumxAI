"""Байтовый BPE-токенизатор IzumxAI: обучение с нуля, без внешних зависимостей.

Схема как в GPT-2/SentencePiece-BPE:
  1. текст -> UTF-8 байты -> предварительная нарезка регуляркой на "слова";
  2. внутри слов итеративно склеиваем самые частые пары символов;
  3. получаем словарь merge-правил, id 0..255 = байты.

Этого достаточно для nano-модели и для собственного токенизатора 2B-версии.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

def _split(text: str) -> list[str]:
    """Нарезка без регулярки на \\p{L}: буквы / цифры / пунктуация / пробелы.

    Один ведущий пробел приклеивается к следующему слову — как в GPT-2,
    так что " hello" и "hello" это разные токены.
    """
    raw: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            raw.append(text[i:j])
            i = j
        elif ch.isalpha():
            j = i
            while j < n and text[j].isalpha():
                j += 1
            raw.append(text[i:j])
            i = j
        elif ch.isdigit():
            j = i
            while j < n and text[j].isdigit():
                j += 1
            raw.append(text[i:j])
            i = j
        else:
            j = i
            while j < n and not text[j].isspace() and not text[j].isalpha() and not text[j].isdigit():
                j += 1
            raw.append(text[i:j])
            i = j

    # склейка одиночного пробела со следующим словом
    out: list[str] = []
    for piece in raw:
        if out and piece.isspace():
            out.append(piece)
        elif out and len(out[-1]) == 1 and out[-1].isspace():
            out[-1] = out[-1] + piece
        else:
            out.append(piece)
    return out


class BPETokenizer:
    def __init__(self, merges: dict[tuple[int, int], int] | None = None,
                 specials: dict[str, int] | None = None):
        self.merges = merges or {}
        self.specials = specials or {}
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        for tok, idx in self.specials.items():
            vocab[idx] = tok.encode("utf-8")
        self._decoder: dict[int, bytes] = vocab
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._cache: dict[str, tuple[int, ...]] = {}

    # ---------------------------------------------------------------- обучение
    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 8192,
              specials: tuple[str, ...] = ("<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"),
              min_freq: int = 2) -> "BPETokenizer":
        """Инкрементальный BPE: обновляем только слова, затронутые merge.

        Наивный вариант (пересчёт всех пар после каждого merge) на корпусе
        в несколько МБ работает часами; этот — секунды-минуты.
        """
        import heapq
        from collections import defaultdict

        word_freq: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            for chunk in _split(text):
                word_freq[tuple(chunk.encode("utf-8"))] += 1

        words: list[list[int]] = [list(w) for w in word_freq]
        freqs: list[int] = list(word_freq.values())
        del word_freq

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for idx, w in enumerate(words):
            f = freqs[idx]
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                pair_where[p].add(idx)

        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: dict[tuple[int, int], int] = {}
        next_id = 256
        target = vocab_size - len(specials)

        while next_id < target:
            best = None
            while heap:
                neg, pair = heapq.heappop(heap)
                if pair_counts.get(pair) == -neg:
                    best = pair
                    break
            if best is None or pair_counts[best] < min_freq:
                break

            merges[best] = next_id
            next_id += 1

            for idx in list(pair_where[best]):
                w, f = words[idx], freqs[idx]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    pair_where[p].discard(idx)
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)

                new: list[int] = []
                i = 0
                while i < len(w):
                    if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                        new.append(merges[best])
                        i += 2
                    else:
                        new.append(w[i])
                        i += 1
                words[idx] = new

                for p in set(zip(new, new[1:])):
                    pair_counts[p] += f
                    pair_where[p].add(idx)
                    heapq.heappush(heap, (-pair_counts[p], p))

            pair_where.pop(best, None)
            pair_counts.pop(best, None)

        special_ids = {tok: next_id + i for i, tok in enumerate(specials)}
        return cls(merges, special_ids)

    # ---------------------------------------------------------------- кодирование
    def _apply(self, ids: list[int]) -> list[int]:
        """Сливаем пары в порядке приоритета (жадно, по возрастанию rank)."""
        while len(ids) >= 2:
            best_rank = None
            best_pos = -1
            best_pair = None
            for pos, pair in enumerate(zip(ids, ids[1:])):
                rank = self._ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pos, best_pair = rank, pos, pair
            if best_pair is None:
                break
            new_id = self.merges[best_pair]
            ids = ids[:best_pos] + [new_id] + ids[best_pos + 2:]
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        out: list[int] = [self.specials["<|bos|>"]] if add_bos else []
        for chunk in _split(text):
            cached = self._cache.get(chunk)
            if cached is None:
                cached = tuple(self._apply(list(chunk.encode("utf-8"))))
                if len(self._cache) < 300_000:
                    self._cache[chunk] = cached
            out.extend(cached)
        if add_eos:
            out.append(self.specials["<|eos|>"])
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        special = set(self.specials.values())
        data = b"".join(self._decoder.get(i, b"") for i in ids
                        if not (skip_special_tokens and i in special))
        return data.decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- сериализация
    @property
    def vocab_size(self) -> int:
        return len(self._decoder)

    @property
    def bos_id(self) -> int:
        return self.specials["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.specials["<|eos|>"]

    @property
    def pad_id(self) -> int:
        return self.specials["<|pad|>"]

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "specials": self.specials,
            "merges": [[a, b, i] for (a, b), i in self.merges.items()],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = {(a, b): i for a, b, i in payload["merges"]}
        return cls(merges, payload["specials"])
