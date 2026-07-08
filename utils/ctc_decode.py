from __future__ import annotations

from typing import Dict, List

import torch

from dataprocess.vocab import decode_ids


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
