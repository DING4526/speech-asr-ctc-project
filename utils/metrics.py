from __future__ import annotations

from typing import List, Sequence, Tuple


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def token_error_rate(refs: List[List[str]], hyps: List[List[str]]) -> Tuple[float, int, int]:
    total_err = 0
    total_ref = 0
    for ref, hyp in zip(refs, hyps):
        total_err += edit_distance(ref, hyp)
        total_ref += len(ref)
    if total_ref == 0:
        return 0.0, total_err, total_ref
    return total_err / total_ref, total_err, total_ref
