from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUTS = [
    ("E2_strong", "results/e2_45_robustness_summary.csv"),
    ("E3_specaugment", "results/e3_45_robustness_summary.csv"),
]


def parse_input(value: str) -> tuple[str, str]:
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected NAME:SUMMARY_CSV")
    return parts[0], parts[1]


def read_summary(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "condition": row["condition"],
                    "per": float(row["per"]),
                    "relative_degradation": float(row.get("relative_degradation") or 0.0),
                }
            )
    return rows


def write_matrix(path: str | Path, all_rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "condition", "per", "relative_degradation"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)


def plot_metric(all_rows: list[dict], output_dir: Path, metric: str, ylabel: str, filename: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiments = list(dict.fromkeys(row["experiment"] for row in all_rows))
    conditions = list(dict.fromkeys(row["condition"] for row in all_rows))
    values = {
        (row["experiment"], row["condition"]): float(row[metric]) * 100.0
        for row in all_rows
    }

    x = list(range(len(conditions)))
    width = 0.8 / max(1, len(experiments))
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for idx, experiment in enumerate(experiments):
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        ax.bar(offsets, [values.get((experiment, cond), 0.0) for cond in conditions], width=width, label=experiment)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot robustness comparison from summary CSV files")
    parser.add_argument("--input", action="append", type=parse_input, help="Entry as NAME:SUMMARY_CSV")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--matrix-out", default="results/robustness_matrix.csv")
    args = parser.parse_args()

    inputs = args.input or DEFAULT_INPUTS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for name, path in inputs:
        if not Path(path).exists():
            continue
        for row in read_summary(path):
            row["experiment"] = name
            all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No robustness summary rows found.")

    write_matrix(args.matrix_out, all_rows)
    plot_metric(all_rows, output_dir, "per", "PER (%)", "robustness_per.png")
    plot_metric(all_rows, output_dir, "relative_degradation", "Relative degradation (%)", "robustness_relative_degradation.png")
    print(f"Wrote robustness matrix to {args.matrix_out}")
    print(f"Wrote robustness figures to {output_dir}")


if __name__ == "__main__":
    main()
