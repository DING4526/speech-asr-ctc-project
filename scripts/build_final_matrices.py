from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_ROBUST_INPUTS = [
    ("R0_E2_no_specaugment", "results/e2_90_robustness_summary.csv"),
    ("R0_E3_specaugment", "results/e3_90_robustness_summary.csv"),
    ("R1_mild_audio_aug", "results/r1_robustness_summary.csv"),
    ("R2_mixed_audio_aug", "results/r2_robustness_summary.csv"),
]


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: list[dict], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def parse_input(value: str) -> tuple[str, str]:
    name, path = value.split(":", 1)
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final robustness and ordered decoder matrices.")
    parser.add_argument("--robust-input", action="append", type=parse_input, default=[])
    parser.add_argument("--robust-out", default="results/final_robustness_matrix.csv")
    parser.add_argument("--decoder-in", default="results/decoder_matrix.csv")
    parser.add_argument("--decoder-out", default="results/decoder_matrix.csv")
    parser.add_argument(
        "--decoder-order",
        default="D0_E3_greedy,D1_E3_beam_bw10,D2_E3_beam_lm,D3_R2_greedy,D4_R2_fast_beam_lm,D3_R2_beam_lm",
    )
    args = parser.parse_args()

    robust_fields = [
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
    robust_rows = []
    robust_inputs = args.robust_input or DEFAULT_ROBUST_INPUTS
    for experiment, path in robust_inputs:
        for row in read_rows(path):
            item = {"experiment": experiment}
            item.update(row)
            robust_rows.append(item)
    if robust_rows:
        write_rows(args.robust_out, robust_rows, robust_fields)

    decoder_path = Path(args.decoder_in)
    if decoder_path.exists():
        decoder_rows = read_rows(decoder_path)
        order = [item.strip() for item in args.decoder_order.split(",") if item.strip()]
        decoder_rows.sort(key=lambda row: order.index(row["experiment"]) if row["experiment"] in order else len(order))
        if decoder_rows:
            write_rows(args.decoder_out, decoder_rows, list(decoder_rows[0].keys()))

    print("Wrote final matrices.")


if __name__ == "__main__":
    main()
