from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def align_ops(ref: list[str], hyp: list[str]) -> list[tuple[str, str, str]]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            choices = [
                (dp[i - 1][j - 1] + sub_cost, "M" if sub_cost == 0 else "S"),
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
            ]
            dp[i][j], bt[i][j] = min(choices, key=lambda item: item[0])

    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op in {"M", "S"}:
            ops.append((op, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            ops.append((op, ref[i - 1], ""))
            i -= 1
        elif op == "I":
            ops.append((op, "", hyp[j - 1]))
            j -= 1
        else:
            break
    ops.reverse()
    return ops


def main() -> None:
    parser = argparse.ArgumentParser(description="Count token-level substitutions from a predictions CSV.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    counter: Counter[tuple[str, str]] = Counter()
    with Path(args.predictions).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = row["ref"].split()
            hyp = row["hyp"].split()
            for op, ref_tok, hyp_tok in align_ops(ref, hyp):
                if op == "S":
                    counter[(ref_tok, hyp_tok)] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ref_token", "hyp_token", "count"])
        writer.writeheader()
        for (ref_tok, hyp_tok), count in counter.most_common(args.top):
            writer.writerow({"ref_token": ref_tok, "hyp_token": hyp_tok, "count": count})
    print(f"Wrote top substitutions to {out}")


if __name__ == "__main__":
    main()
