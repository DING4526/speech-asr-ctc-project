from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_matrix(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "condition",
        "per",
        "relative_degradation",
        "errors",
        "ref_tokens",
        "substitutions",
        "deletions",
        "insertions",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_matrix(rows: list[dict], out_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conditions = ["clean", "real_noise20", "real_noise10", "real_noise5"]
    bar_experiments = ["R0_E3_specaugment", "R1_mild_audio_aug", "R2_mixed_audio_aug"]
    scatter_experiments = [
        "R0_E2_no_specaugment",
        "R0_E3_specaugment",
        "R1_mild_audio_aug",
        "R2_mixed_audio_aug",
    ]
    labels = {
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
    value = {(row["experiment"], row["condition"]): float(row["per"]) * 100.0 for row in rows}

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5), dpi=150)
    x = list(range(len(conditions)))
    width = 0.24
    for idx, exp in enumerate(bar_experiments):
        offsets = [pos + (idx - 1) * width for pos in x]
        ys = [value.get((exp, cond), 0.0) for cond in conditions]
        axes[0].bar(offsets, ys, width=width, label=labels[exp], color=colors[exp])
        for xpos, y in zip(offsets, ys):
            axes[0].text(xpos, y + 0.7, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["clean", "20 dB", "10 dB", "5 dB"])
    axes[0].set_ylabel("PER (%)")
    axes[0].set_title("Real-noise mixed test")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    mean_points = []
    for exp in scatter_experiments:
        clean = value.get((exp, "clean"), 0.0)
        noisy = [value.get((exp, cond), 0.0) for cond in conditions if cond != "clean"]
        mean_noisy = sum(noisy) / max(1, len(noisy))
        mean_points.append((labels[exp], clean, mean_noisy, colors[exp]))
    for label, clean, mean_noisy, color in mean_points:
        axes[1].scatter(clean, mean_noisy, s=95, color=color)
        if label == "R0 E2":
            axes[1].annotate(label, (clean, mean_noisy), textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=9)
        elif label == "R0 E3":
            axes[1].annotate(label, (clean, mean_noisy), textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=9)
        elif label.startswith("R2"):
            axes[1].annotate(label, (clean, mean_noisy), textcoords="offset points", xytext=(8, -12), ha="left", fontsize=9)
        else:
            axes[1].annotate(label, (clean, mean_noisy), textcoords="offset points", xytext=(8, 6), ha="left", fontsize=9)
    axes[1].set_xlabel("Clean PER (%)")
    axes[1].set_ylabel("Mean real-noise PER (%)")
    axes[1].set_title("Clean vs real-noise robustness")
    axes[1].grid(alpha=0.3)
    clean_values = [point[1] for point in mean_points]
    noisy_values = [point[2] for point in mean_points]
    axes[1].set_xlim(left=0, right=max(clean_values) * 1.18)
    axes[1].set_ylim(bottom=0, top=max(noisy_values) * 1.18)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and plot the lightweight real-noise sanity-check matrix.")
    parser.add_argument("--e2-summary", default="results/real_noise_e2_summary.csv")
    parser.add_argument("--r0-summary", default="results/real_noise_r0_summary.csv")
    parser.add_argument("--r1-summary", default="results/real_noise_r1_summary.csv")
    parser.add_argument("--r2-summary", default="results/real_noise_r2_summary.csv")
    parser.add_argument("--matrix-out", default="results/real_noise_matrix.csv")
    parser.add_argument("--fig-out", default="results/figures/real_noise_sanity_check.png")
    args = parser.parse_args()

    rows = []
    for experiment, path in [
        ("R0_E2_no_specaugment", args.e2_summary),
        ("R0_E3_specaugment", args.r0_summary),
        ("R1_mild_audio_aug", args.r1_summary),
        ("R2_mixed_audio_aug", args.r2_summary),
    ]:
        for row in read_csv(path):
            item = dict(row)
            item["experiment"] = experiment
            rows.append(item)
    write_matrix(args.matrix_out, rows)
    plot_matrix(rows, args.fig_out)
    print(f"Wrote matrix to {args.matrix_out}")
    print(f"Wrote figure to {args.fig_out}")


if __name__ == "__main__":
    main()
