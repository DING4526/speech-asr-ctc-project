from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot robust dev checkpoint selection curves.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", default="results/figures/r2_robust_dev_selection.png")
    args = parser.parse_args()

    rows = []
    with Path(args.summary).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found in {args.summary}")

    by_epoch: dict[int, dict[str, float]] = defaultdict(dict)
    robust_scores: dict[int, float] = {}
    for row in rows:
        epoch = int(row["epoch"])
        by_epoch[epoch][row["condition"]] = float(row["per"]) * 100.0
        robust_scores[epoch] = float(row["robust_score"]) * 100.0

    epochs = sorted(robust_scores)
    best_epoch = min(epochs, key=lambda ep: robust_scores[ep])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(epochs, [robust_scores[ep] for ep in epochs], marker="o", linewidth=2.2, label="Robust dev score")
    for condition in ["clean", "noise20", "speed0.9", "speed1.1", "echo"]:
        values = [by_epoch[ep].get(condition) for ep in epochs]
        if all(value is not None for value in values):
            ax.plot(epochs, values, marker=".", alpha=0.55, linewidth=1.2, label=condition)
    ax.axvline(best_epoch, color="#E45756", linestyle="--", linewidth=1.5)
    ax.annotate(
        f"best epoch {best_epoch}\n{robust_scores[best_epoch]:.2f}%",
        (best_epoch, robust_scores[best_epoch]),
        textcoords="offset points",
        xytext=(-75, 15),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#E45756"},
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dev PER / score (%)")
    ax.set_title("R2 robust checkpoint selection")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote robust selection figure to {out}")


if __name__ == "__main__":
    main()
