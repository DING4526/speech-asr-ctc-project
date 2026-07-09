from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_EXPERIMENTS = [
    ("E0_final_baseline", "checkpoints/lab6_clean_baseline_30/metrics.csv", "results/e0_final_baseline_analysis.json"),
    ("E1_conv_logmel", "checkpoints/e1_conv_logmel/metrics.csv", "results/e1_conv_logmel_analysis.json"),
    ("E2_conv_fbank_strong", "checkpoints/e2_conv_fbank_strong/metrics.csv", "results/e2_conv_fbank_strong_analysis.json"),
    (
        "E3_conv_fbank_weak_specaugment",
        "checkpoints/e3_conv_fbank_weak_specaugment/metrics.csv",
        "results/e3_conv_fbank_weak_specaugment_analysis.json",
    ),
]


def parse_experiment(value: str) -> tuple[str, str, str]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected NAME:METRICS_CSV:ANALYSIS_JSON")
    return parts[0], parts[1], parts[2]


def read_metrics(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "dev_loss": float(row["dev_loss"]),
                    "dev_per": float(row["dev_per"]),
                    "is_best": int(row.get("is_best") or 0),
                }
            )
    return rows


def read_analysis(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_metric(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return min(rows, key=lambda row: row["dev_per"])


def write_matrix(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "best_epoch",
        "best_dev_per",
        "test_per",
        "test_errors",
        "test_ref_tokens",
        "substitutions",
        "deletions",
        "insertions",
        "metrics_path",
        "analysis_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_training_curve(name: str, rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    dev_loss = [row["dev_loss"] for row in rows]
    dev_per = [row["dev_per"] * 100.0 for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    axes[0].plot(epochs, train_loss, marker="o", label="train loss")
    axes[0].plot(epochs, dev_loss, marker="o", label="dev loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("CTC loss")
    axes[0].set_title(f"{name} loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, dev_per, marker="o", color="#d62728", label="dev PER")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("PER (%)")
    axes[1].set_title(f"{name} dev PER")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_training_curves.png")
    plt.close(fig)


def plot_dev_compare(experiments: list[tuple[str, list[dict]]], output_dir: Path) -> None:
    valid = [(name, rows) for name, rows in experiments if rows]
    if not valid:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for name, rows in valid:
        ax.plot(
            [row["epoch"] for row in rows],
            [row["dev_per"] * 100.0 for row in rows],
            marker="o",
            label=name,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev PER (%)")
    ax.set_title("Dev PER Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "dev_per_comparison.png")
    plt.close(fig)


def plot_test_per(matrix_rows: list[dict], output_dir: Path) -> None:
    valid = [row for row in matrix_rows if row.get("test_per") not in {"", None}]
    if not valid:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["experiment"] for row in valid]
    values = [float(row["test_per"]) * 100.0 for row in valid]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.bar(names, values, color="#4c78a8")
    ax.set_ylabel("test PER (%)")
    ax.set_title("Test PER Comparison")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "test_per_comparison.png")
    plt.close(fig)


def plot_error_breakdown(matrix_rows: list[dict], output_dir: Path) -> None:
    valid = [row for row in matrix_rows if row.get("substitutions") not in {"", None}]
    if not valid:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["experiment"] for row in valid]
    substitutions = [int(row["substitutions"]) for row in valid]
    deletions = [int(row["deletions"]) for row in valid]
    insertions = [int(row["insertions"]) for row in valid]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.bar(names, substitutions, label="S")
    ax.bar(names, deletions, bottom=substitutions, label="D")
    bottoms = [s + d for s, d in zip(substitutions, deletions)]
    ax.bar(names, insertions, bottom=bottoms, label="I")
    ax.set_ylabel("error count")
    ax.set_title("Substitution / Deletion / Insertion")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "error_breakdown.png")
    plt.close(fig)


def plot_length_bins(analyses: list[tuple[str, dict | None]], output_dir: Path) -> None:
    valid = [(name, analysis) for name, analysis in analyses if analysis and analysis.get("by_reference_length")]
    if not valid:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for name, analysis in valid:
        rows = analysis["by_reference_length"]
        ax.plot(
            [row["length_bin"] for row in rows],
            [row["per"] * 100.0 for row in rows],
            marker="o",
            label=name,
        )
    ax.set_xlabel("reference length bin")
    ax.set_ylabel("PER (%)")
    ax.set_title("PER by Reference Length")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "length_bin_per.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ASR experiment metrics and analysis outputs")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--matrix-out", default="results/experiment_matrix.csv")
    parser.add_argument(
        "--experiment",
        action="append",
        type=parse_experiment,
        help="Experiment entry as NAME:METRICS_CSV:ANALYSIS_JSON. Can be repeated.",
    )
    args = parser.parse_args()

    experiments = args.experiment or DEFAULT_EXPERIMENTS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_rows = []
    metric_sets = []
    analyses = []
    for name, metrics_path, analysis_path in experiments:
        metrics = read_metrics(metrics_path)
        analysis = read_analysis(analysis_path)
        best = best_metric(metrics)
        summary = analysis.get("summary", {}) if analysis else {}
        matrix_rows.append(
            {
                "experiment": name,
                "best_epoch": best["epoch"] if best else "",
                "best_dev_per": best["dev_per"] if best else "",
                "test_per": summary.get("per", ""),
                "test_errors": summary.get("errors", ""),
                "test_ref_tokens": summary.get("ref_tokens", ""),
                "substitutions": summary.get("substitutions", ""),
                "deletions": summary.get("deletions", ""),
                "insertions": summary.get("insertions", ""),
                "metrics_path": metrics_path,
                "analysis_path": analysis_path,
            }
        )
        metric_sets.append((name, metrics))
        analyses.append((name, analysis))
        plot_training_curve(name, metrics, output_dir)

    write_matrix(matrix_rows, Path(args.matrix_out))
    plot_dev_compare(metric_sets, output_dir)
    plot_test_per(matrix_rows, output_dir)
    plot_error_breakdown(matrix_rows, output_dir)
    plot_length_bins(analyses, output_dir)

    print(f"Wrote matrix to {args.matrix_out}")
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
