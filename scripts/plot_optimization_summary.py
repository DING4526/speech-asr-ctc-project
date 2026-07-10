from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_decoder(decoder_matrix: str | Path, output_dir: Path) -> None:
    rows = read_csv(decoder_matrix)
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["experiment"] for row in rows]
    s_vals = [int(row["substitutions"]) for row in rows]
    d_vals = [int(row["deletions"]) for row in rows]
    i_vals = [int(row["insertions"]) for row in rows]
    per_vals = [float(row["per"]) * 100.0 for row in rows]
    seconds = [float(row["decode_seconds"]) for row in rows]

    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.bar(x, s_vals, label="Substitution", color="#4C78A8")
    ax.bar(x, d_vals, bottom=s_vals, label="Deletion", color="#F58518")
    bottoms = [s + d for s, d in zip(s_vals, d_vals)]
    ax.bar(x, i_vals, bottom=bottoms, label="Insertion", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Error count")
    ax.grid(True, axis="y", alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(x, per_vals, color="#E45756", marker="o", label="PER")
    ax2.set_ylabel("PER (%)")
    for idx, (per, sec) in enumerate(zip(per_vals, seconds)):
        ax2.annotate(f"{per:.2f}%\n{sec:.1f}s", (idx, per), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "decoder_error_breakdown.png")
    plt.close(fig)


def plot_robustness(robust_matrix: str | Path, output_dir: Path) -> None:
    rows = read_csv(robust_matrix)
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiments = list(dict.fromkeys(row["experiment"] for row in rows))
    bar_experiments = [exp for exp in experiments if exp != "R0_E2_no_specaugment"]
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    value = {(row["experiment"], row["condition"]): float(row["per"]) * 100.0 for row in rows}

    x = list(range(len(conditions)))
    width = 0.8 / max(1, len(bar_experiments))
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    for idx, exp in enumerate(bar_experiments):
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        ax.bar(offsets, [value.get((exp, cond), 0.0) for cond in conditions], width=width, label=exp)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=25, ha="right")
    ax.set_ylabel("PER (%)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "final_robustness_per.png")
    plt.close(fig)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["experiment"]].append(row)
    trade_rows = []
    for exp, items in grouped.items():
        clean = next((float(row["per"]) for row in items if row["condition"] == "clean"), None)
        perturbed = [float(row["per"]) for row in items if row["condition"] != "clean"]
        if clean is None or not perturbed:
            continue
        trade_rows.append((exp, clean * 100.0, sum(perturbed) / len(perturbed) * 100.0))

    display_names = {
        "R0_E2_no_specaugment": "R0 E2",
        "R0_E3_specaugment": "R0 E3",
        "R1_mild_audio_aug": "R1 mild",
        "R2_mixed_audio_aug": "R2 mixed",
    }
    colors = {
        "R0_E2_no_specaugment": "#B279A2",
        "R0_E3_specaugment": "#4C78A8",
        "R1_mild_audio_aug": "#F58518",
        "R2_mixed_audio_aug": "#54A24B",
    }
    label_offsets = {
        "R0_E2_no_specaugment": (-8, 7, "right"),
        "R0_E3_specaugment": (7, 6, "left"),
        "R1_mild_audio_aug": (7, 6, "left"),
        "R2_mixed_audio_aug": (7, -12, "left"),
    }

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
    for exp, clean, robust in trade_rows:
        xoff, yoff, align = label_offsets.get(exp, (8, 6, "left"))
        ax.scatter(clean, robust, s=80, color=colors.get(exp))
        ax.annotate(
            display_names.get(exp, exp),
            (clean, robust),
            textcoords="offset points",
            xytext=(xoff, yoff),
            ha=align,
            fontsize=9,
        )
    ax.set_xlabel("Clean PER (%)")
    ax.set_ylabel("Mean perturbed PER (%)")
    ax.set_title("Clean vs robustness trade-off")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0, right=max(row[1] for row in trade_rows) * 1.18)
    ax.set_ylim(bottom=0, top=max(row[2] for row in trade_rows) * 1.18)
    fig.tight_layout()
    fig.savefig(output_dir / "clean_vs_robust_tradeoff.png")
    plt.close(fig)


def plot_substitutions(paths: list[str], labels: list[str], output_dir: Path, top: int = 15) -> None:
    if not paths:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counters = []
    all_pairs = []
    for path in paths:
        rows = read_csv(path)
        counter = {f"{row['ref_token']}->{row['hyp_token']}": int(row["count"]) for row in rows}
        counters.append(counter)
        all_pairs.extend(counter.keys())
    top_pairs = []
    totals = defaultdict(int)
    for counter in counters:
        for pair, count in counter.items():
            totals[pair] += count
    for pair, _ in sorted(totals.items(), key=lambda item: item[1], reverse=True)[:top]:
        top_pairs.append(pair)

    x = list(range(len(top_pairs)))
    width = 0.8 / max(1, len(counters))
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    for idx, (label, counter) in enumerate(zip(labels, counters)):
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        ax.bar(offsets, [counter.get(pair, 0) for pair in top_pairs], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(top_pairs, rotation=35, ha="right")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "substitution_confusions.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot final decoder and robustness optimization summaries.")
    parser.add_argument("--decoder-matrix", default="results/decoder_matrix.csv")
    parser.add_argument("--robust-matrix", default="results/final_robustness_matrix.csv")
    parser.add_argument("--substitution", action="append", default=[], help="LABEL:CSV")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.decoder_matrix).exists():
        plot_decoder(args.decoder_matrix, output_dir)
    if Path(args.robust_matrix).exists():
        plot_robustness(args.robust_matrix, output_dir)
    labels, paths = [], []
    for item in args.substitution:
        label, path = item.split(":", 1)
        labels.append(label)
        paths.append(path)
    plot_substitutions(paths, labels, output_dir)
    print(f"Wrote optimization figures to {output_dir}")


if __name__ == "__main__":
    main()
