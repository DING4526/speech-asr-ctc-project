from __future__ import annotations

import math
from typing import Dict, List, Protocol

import torch

from dataprocess.vocab import decode_ids


class TokenLM(Protocol):
    def score_next(self, prefix: tuple[str, ...] | list[str], token: str) -> float:
        ...


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def ctc_greedy_decode(logits: torch.Tensor, lengths: torch.Tensor, id2token: Dict[int, str], blank_id: int = 0) -> List[List[str]]:
    """Greedy CTC decoding.

    Args:
        logits: [B, T, V]
        lengths: [B]
    Returns:
        List of token lists.
    """
    pred_ids = logits.argmax(dim=-1).cpu().tolist()
    decoded: List[List[str]] = []
    for seq, length in zip(pred_ids, lengths.cpu().tolist()):
        collapsed = []
        last = None
        for idx in seq[:length]:
            if idx != last:
                if idx != blank_id:
                    collapsed.append(idx)
            last = idx
        decoded.append(decode_ids(collapsed, id2token, remove_special=True))
    return decoded


def ctc_prefix_beam_search_decode(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    id2token: Dict[int, str],
    blank_id: int = 0,
    beam_width: int = 25,
    topk: int = 20,
    lm: TokenLM | None = None,
    lm_alpha: float = 0.0,
    lm_beta: float = 0.0,
    skip_token_ids: set[int] | None = None,
) -> List[List[str]]:
    """CTC prefix beam search with optional shallow-fusion token LM.

    The decoder keeps CTC acoustic scores separate from LM scores and uses
    `ctc + lm_alpha * lm + lm_beta * token_count` only for pruning/final rank.
    """
    log_probs = logits.log_softmax(dim=-1).detach().cpu()
    lengths_cpu = lengths.cpu().tolist()
    vocab_size = log_probs.size(-1)
    if beam_width <= 0:
        raise ValueError(f"beam_width must be positive: {beam_width}")
    if topk <= 0:
        raise ValueError(f"topk must be positive: {topk}")
    if skip_token_ids is None:
        skip_token_ids = {blank_id, 1}
    else:
        skip_token_ids = set(skip_token_ids) | {blank_id}

    decoded: List[List[str]] = []
    for batch_idx, length in enumerate(lengths_cpu):
        beams: dict[tuple[int, ...], tuple[float, float, float]] = {tuple(): (0.0, -math.inf, 0.0)}
        frame_log_probs = log_probs[batch_idx, :length]
        for frame in frame_log_probs:
            k = min(vocab_size, max(topk, 1))
            values, indices = torch.topk(frame, k=k)
            frame_items = [(int(i), float(v)) for i, v in zip(indices.tolist(), values.tolist())]
            if blank_id not in {idx for idx, _ in frame_items}:
                frame_items.append((blank_id, float(frame[blank_id].item())))

            next_beams: dict[tuple[int, ...], tuple[float, float, float]] = {}

            def update(prefix: tuple[int, ...], pb_add: float, pnb_add: float, lm_score: float) -> None:
                old_pb, old_pnb, _ = next_beams.get(prefix, (-math.inf, -math.inf, lm_score))
                next_beams[prefix] = (_logaddexp(old_pb, pb_add), _logaddexp(old_pnb, pnb_add), lm_score)

            for prefix, (pb, pnb, lm_score) in beams.items():
                total = _logaddexp(pb, pnb)
                for token_id, token_log_prob in frame_items:
                    if token_id == blank_id:
                        update(prefix, total + token_log_prob, -math.inf, lm_score)
                        continue
                    if token_id in skip_token_ids:
                        continue

                    if prefix and token_id == prefix[-1]:
                        update(prefix, -math.inf, pnb + token_log_prob, lm_score)
                        acoustic = pb + token_log_prob
                    else:
                        acoustic = total + token_log_prob

                    if acoustic == -math.inf:
                        continue
                    new_prefix = prefix + (token_id,)
                    token = id2token.get(token_id, "")
                    lm_delta = lm.score_next(tuple(id2token.get(i, "") for i in prefix), token) if lm else 0.0
                    update(new_prefix, -math.inf, acoustic, lm_score + lm_delta)

            def rank(item: tuple[tuple[int, ...], tuple[float, float, float]]) -> float:
                prefix, (pb, pnb, lm_score) = item
                return _logaddexp(pb, pnb) + lm_alpha * lm_score + lm_beta * len(prefix)

            beams = dict(sorted(next_beams.items(), key=rank, reverse=True)[:beam_width])

        best_prefix, _ = max(
            beams.items(),
            key=lambda item: _logaddexp(item[1][0], item[1][1]) + lm_alpha * item[1][2] + lm_beta * len(item[0]),
        )
        decoded.append(decode_ids(list(best_prefix), id2token, remove_special=True))
    return decoded
