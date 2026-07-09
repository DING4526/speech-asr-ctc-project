from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


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


def edit_distance_with_ops(ref: Sequence[str], hyp: Sequence[str]) -> Dict[str, int]:
    """Return edit distance and one optimal substitution/deletion/insertion split."""
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
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    substitutions = 0
    deletions = 0
    insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                substitutions += cost
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
            continue
        insertions += 1
        j -= 1

    return {
        "errors": dp[n][m],
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def token_error_rate(refs: List[List[str]], hyps: List[List[str]]) -> Tuple[float, int, int]:
    total_err = 0
    total_ref = 0
    for ref, hyp in zip(refs, hyps):
        total_err += edit_distance(ref, hyp)
        total_ref += len(ref)
    if total_ref == 0:
        return 0.0, total_err, total_ref
    return total_err / total_ref, total_err, total_ref


def utterance_error_summary(ref: Sequence[str], hyp: Sequence[str]) -> Dict[str, float | int]:
    ops = edit_distance_with_ops(ref, hyp)
    ref_len = len(ref)
    hyp_len = len(hyp)
    per = ops["errors"] / ref_len if ref_len else 0.0
    return {
        "ref_len": ref_len,
        "hyp_len": hyp_len,
        "errors": ops["errors"],
        "substitutions": ops["substitutions"],
        "deletions": ops["deletions"],
        "insertions": ops["insertions"],
        "per": per,
    }


def corpus_error_summary(refs: List[List[str]], hyps: List[List[str]]) -> Dict[str, float | int]:
    total_ref = 0
    total_hyp = 0
    total_errors = 0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    for ref, hyp in zip(refs, hyps):
        item = utterance_error_summary(ref, hyp)
        total_ref += int(item["ref_len"])
        total_hyp += int(item["hyp_len"])
        total_errors += int(item["errors"])
        total_substitutions += int(item["substitutions"])
        total_deletions += int(item["deletions"])
        total_insertions += int(item["insertions"])
    return {
        "num_utterances": len(refs),
        "ref_tokens": total_ref,
        "hyp_tokens": total_hyp,
        "errors": total_errors,
        "substitutions": total_substitutions,
        "deletions": total_deletions,
        "insertions": total_insertions,
        "per": total_errors / total_ref if total_ref else 0.0,
    }
