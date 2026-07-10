from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pct(value: str | float) -> float:
    return float(value) * 100.0


def plot_baseline(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = [
        ("E1 conv logmel", "checkpoints/e1_conv_logmel/metrics.csv", "#4C78A8"),
        ("E2 conv fbank", "checkpoints/e2_conv_fbank_strong/metrics.csv", "#F58518"),
        ("E3 + SpecAug", "checkpoints/e3_conv_fbank_weak_specaugment/metrics.csv", "#54A24B"),
    ]
    experiments = ["E0 baseline", "E1 conv logmel", "E2 conv fbank", "E3 + SpecAug"]
    test = [15.69, 11.0734, 5.0616, 4.3321]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), dpi=180, gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax = axes[0]
    for label, path, color in curves:
        rows = read_csv(path)
        epochs = [int(row["epoch"]) for row in rows]
        dev_per = [pct(row["dev_per"]) for row in rows]
        ax.plot(epochs, dev_per, marker="o", markersize=2.8, linewidth=1.7, label=label, color=color)
    ax.set_title("Dev PER convergence")
    ax.set_xlabel("epoch")
    ax.set_ylabel("PER (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    x = list(range(len(experiments)))
    colors = ["#B279A2", "#4C78A8", "#F58518", "#54A24B"]
    ax.bar(x, test, color=colors, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=22, ha="right")
    ax.set_title("Final test PER")
    ax.set_ylabel("PER (%)")
    ax.grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(test):
        ax.text(idx, value + 0.35, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Baseline convergence and greedy test performance", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "report_baseline_overview.png", bbox_inches="tight")
    plt.close(fig)


def plot_robustness(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_csv("results/final_robustness_matrix.csv")
    show_conditions = ["clean", "noise20", "noise10", "speed0.9", "speed1.1", "echo"]
    scatter_experiments = [
        "R0_E2_no_specaugment",
        "R0_E3_specaugment",
        "R1_mild_audio_aug",
        "R2_mixed_audio_aug",
    ]
    bar_experiments = scatter_experiments
    short_names = {
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
    values = {(row["experiment"], row["condition"]): pct(row["per"]) for row in rows}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=180, gridspec_kw={"width_ratios": [1.7, 1.0]})
    ax = axes[0]
    x = list(range(len(show_conditions)))
    width = 0.8 / len(bar_experiments)
    for idx, exp in enumerate(bar_experiments):
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        ax.bar(
            offsets,
            [values[(exp, cond)] for cond in show_conditions],
            width=width,
            label=short_names[exp],
            color=colors[exp],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(show_conditions, rotation=25, ha="right")
    ax.set_ylabel("PER (%)")
    ax.set_title("Robustness under acoustic perturbations")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1]
    scatter_points = []
    label_offsets = {
        "R0_E2_no_specaugment": (-8, 7, "right"),
        "R0_E3_specaugment": (7, 6, "left"),
        "R1_mild_audio_aug": (7, 6, "left"),
        "R2_mixed_audio_aug": (7, -12, "left"),
    }
    for exp in scatter_experiments:
        clean = values[(exp, "clean")]
        perturbed = [values[(exp, cond)] for cond in show_conditions if cond != "clean"]
        mean_perturbed = sum(perturbed) / len(perturbed)
        scatter_points.append((clean, mean_perturbed))
        xoff, yoff, align = label_offsets[exp]
        ax.scatter(clean, mean_perturbed, s=75, color=colors[exp])
        ax.annotate(
            short_names[exp],
            (clean, mean_perturbed),
            textcoords="offset points",
            xytext=(xoff, yoff),
            ha=align,
            fontsize=9,
        )
    ax.set_xlabel("Clean PER (%)")
    ax.set_ylabel("Mean perturbed PER (%)")
    ax.set_title("Clean vs robustness trade-off")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0, right=max(point[0] for point in scatter_points) * 1.18)
    ax.set_ylim(bottom=0, top=max(point[1] for point in scatter_points) * 1.18)
    fig.tight_layout()
    fig.savefig(output_dir / "report_robustness_overview.png", bbox_inches="tight")
    plt.close(fig)


def plot_decoder(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [
        "D0_E3_greedy",
        "D1_E3_beam_bw10",
        "D2_E3_beam_lm",
        "D3_R2_greedy",
        "D4_R2_fast_beam_lm",
        "D3_R2_beam_lm",
    ]
    rows = [row for row in read_csv("results/decoder_matrix.csv") if row["experiment"] in set(order)]
    rows.sort(key=lambda row: order.index(row["experiment"]))
    labels = ["D0 E3 greedy", "D1 E3 beam", "D2 E3 LM", "D3 R2 greedy", "D4 R2 fast LM", "D3 R2 LM"]

    sub_paths = {
        "E3 greedy": "results/d0_e3_greedy_substitutions.csv",
        "E3 LM": "results/d2_e3_beam_lm_substitutions.csv",
        "R2 greedy": "results/d3_r2_greedy_substitutions.csv",
        "R2 LM": "results/d4_r2_fast_beam_lm_substitutions.csv",
    }
    counters: dict[str, dict[str, int]] = {}
    totals: defaultdict[str, int] = defaultdict(int)
    for name, path in sub_paths.items():
        counter = {f"{row['ref_token']}->{row['hyp_token']}": int(row["count"]) for row in read_csv(path)}
        counters[name] = counter
        for pair, count in counter.items():
            totals[pair] += count
    top_pairs = [pair for pair, _ in sorted(totals.items(), key=lambda item: item[1], reverse=True)[:10]]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), dpi=180, gridspec_kw={"width_ratios": [1.25, 1.25]})
    ax = axes[0]
    x = list(range(len(rows)))
    s = [int(row["substitutions"]) for row in rows]
    d = [int(row["deletions"]) for row in rows]
    i = [int(row["insertions"]) for row in rows]
    per = [pct(row["per"]) for row in rows]
    ax.bar(x, s, label="Substitution", color="#4C78A8")
    ax.bar(x, d, bottom=s, label="Deletion", color="#F58518")
    ax.bar(x, i, bottom=[a + b for a, b in zip(s, d)], label="Insertion", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Error count")
    ax.set_title("Decoder error breakdown")
    ax.grid(True, axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(x, per, marker="o", color="#E45756", linewidth=1.8, label="PER")
    ax2.set_ylabel("PER (%)")
    for idx, value in enumerate(per):
        seconds = float(rows[idx]["decode_seconds"])
        ax2.text(idx, value + 0.12, f"{value:.2f}%\n{seconds:.1f}s", ha="center", fontsize=8, color="black")
    ax2.set_ylim(bottom=0, top=max(per) * 1.22)
    handles, handle_labels = ax.get_legend_handles_labels()
    line_handles, line_labels = ax2.get_legend_handles_labels()
    ax.legend(handles + line_handles, handle_labels + line_labels, loc="upper right", fontsize=8)

    ax = axes[1]
    bar_labels = ["E3 greedy", "E3 LM", "R2 greedy", "R2 LM"]
    width = 0.8 / len(bar_labels)
    x = list(range(len(top_pairs)))
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B"]
    for idx, name in enumerate(bar_labels):
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        ax.bar(offsets, [counters[name].get(pair, 0) for pair in top_pairs], width=width, label=name, color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(top_pairs, rotation=35, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Top substitution confusions")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(output_dir / "report_decoder_substitution.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_baseline(output_dir)
    plot_robustness(output_dir)
    plot_decoder(output_dir)
    print(f"Wrote report figures to {output_dir}")


if __name__ == "__main__":
    main()
