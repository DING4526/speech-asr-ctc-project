from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from dataprocess.vocab import read_pinyin_file


class PinyinNgramLM:
    """Small add-k smoothed pinyin n-gram language model."""

    def __init__(
        self,
        order: int = 4,
        delta: float = 0.1,
        backoff: float = 0.4,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
    ) -> None:
        if order < 1:
            raise ValueError(f"order must be positive: {order}")
        self.order = int(order)
        self.delta = float(delta)
        self.backoff = float(backoff)
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.vocab: set[str] = set()
        self.ngram_counts: dict[int, Counter[tuple[str, ...]]] = {
            n: Counter() for n in range(1, self.order + 1)
        }
        self.context_counts: dict[int, Counter[tuple[str, ...]]] = {
            n: Counter() for n in range(1, self.order)
        }
        self.total_unigrams = 0

    def fit(self, sentences: Iterable[list[str]]) -> None:
        for sent in sentences:
            tokens = [tok for tok in sent if tok]
            if not tokens:
                continue
            self.vocab.update(tokens)
            padded = [self.bos_token] * (self.order - 1) + tokens + [self.eos_token]
            for i in range(self.order - 1, len(padded)):
                for n in range(1, self.order + 1):
                    start = i - n + 1
                    if start < 0:
                        continue
                    gram = tuple(padded[start : i + 1])
                    self.ngram_counts[n][gram] += 1
                    if n > 1:
                        self.context_counts[n - 1][gram[:-1]] += 1

        self.vocab.add(self.eos_token)
        self.total_unigrams = sum(self.ngram_counts[1].values())
        if not self.total_unigrams:
            raise RuntimeError("Cannot train pinyin LM from empty text.")

    @classmethod
    def from_pinyin_file(cls, text_path: str | Path, order: int = 4, delta: float = 0.1, backoff: float = 0.4) -> "PinyinNgramLM":
        lm = cls(order=order, delta=delta, backoff=backoff)
        lm.fit(read_pinyin_file(text_path).values())
        return lm

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "order": self.order,
            "delta": self.delta,
            "backoff": self.backoff,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "vocab": sorted(self.vocab),
            "ngram_counts": {
                str(n): {"\t".join(k): v for k, v in counts.items()}
                for n, counts in self.ngram_counts.items()
            },
            "context_counts": {
                str(n): {"\t".join(k): v for k, v in counts.items()}
                for n, counts in self.context_counts.items()
            },
            "total_unigrams": self.total_unigrams,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PinyinNgramLM":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        lm = cls(
            order=int(payload["order"]),
            delta=float(payload["delta"]),
            backoff=float(payload["backoff"]),
            bos_token=payload.get("bos_token", "<s>"),
            eos_token=payload.get("eos_token", "</s>"),
        )
        lm.vocab = set(payload["vocab"])
        lm.ngram_counts = {
            int(n): Counter({tuple(k.split("\t")): int(v) for k, v in counts.items()})
            for n, counts in payload["ngram_counts"].items()
        }
        lm.context_counts = {
            int(n): Counter({tuple(k.split("\t")) if k else tuple(): int(v) for k, v in counts.items()})
            for n, counts in payload["context_counts"].items()
        }
        lm.total_unigrams = int(payload["total_unigrams"])
        return lm

    def score_next(self, prefix: tuple[str, ...] | list[str], token: str) -> float:
        """Return log P(token | prefix) with simple backoff."""
        prefix_tuple = tuple(prefix)
        vocab_size = max(1, len(self.vocab))
        backoff_penalty = 0.0
        max_context = min(self.order - 1, len(prefix_tuple))

        for context_len in range(max_context, -1, -1):
            context = prefix_tuple[-context_len:] if context_len else tuple()
            n = context_len + 1
            gram = context + (token,)
            count = self.ngram_counts[n].get(gram, 0)

            if context_len == 0:
                denom = self.total_unigrams + self.delta * vocab_size
                prob = (count + self.delta) / denom
                return backoff_penalty + math.log(prob)

            context_count = self.context_counts[context_len].get(context, 0)
            if context_count > 0:
                denom = context_count + self.delta * vocab_size
                prob = (count + self.delta) / denom
                return backoff_penalty + math.log(prob)
            backoff_penalty += math.log(self.backoff)

        raise AssertionError("unreachable")

    def score_sentence(self, tokens: list[str]) -> float:
        score = 0.0
        prefix: list[str] = []
        for token in tokens:
            score += self.score_next(prefix, token)
            prefix.append(token)
        score += self.score_next(prefix, self.eos_token)
        return score


class NullLM:
    def score_next(self, prefix: tuple[str, ...] | list[str], token: str) -> float:
        return 0.0
