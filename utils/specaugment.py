from __future__ import annotations

import random
import torch


def specaugment(
    feats: torch.Tensor,
    lengths: torch.Tensor,
    time_mask_param: int = 24,
    freq_mask_param: int = 10,
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
) -> torch.Tensor:
    """Apply simple SpecAugment to padded log-Mel features.

    Args:
        feats: [B, T, F]
        lengths: valid frame lengths for each utterance.
    """
    feats = feats.clone()
    bsz, _, freq_dim = feats.shape
    for b in range(bsz):
        valid_t = int(lengths[b].item())
        if valid_t <= 1:
            continue
        for _ in range(num_time_masks):
            width = random.randint(0, min(time_mask_param, valid_t))
            if width > 0:
                start = random.randint(0, max(0, valid_t - width))
                feats[b, start : start + width, :] = 0.0
        for _ in range(num_freq_masks):
            width = random.randint(0, min(freq_mask_param, freq_dim))
            if width > 0:
                start = random.randint(0, max(0, freq_dim - width))
                feats[b, :valid_t, start : start + width] = 0.0
    return feats
