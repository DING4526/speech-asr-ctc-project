from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_decoder import evaluate_decoder_config


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def write_rows(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "split",
        "decoder",
        "beam_width",
        "topk",
        "lm_order",
        "lm_alpha",
        "lm_beta",
        "loss",
        "per",
        "errors",
        "ref_tokens",
        "substitutions",
        "deletions",
        "insertions",
        "decode_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune CTC beam and pinyin LM decoder on dev split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["dev"], default="dev")
    parser.add_argument("--beam-widths", default="10,25,50")
    parser.add_argument("--lm-alphas", default="0.1,0.3,0.5,0.8,1.0")
    parser.add_argument("--lm-betas", default="-0.1,0,0.1,0.2")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--lm-path", default="results/lm/pinyin_4gram.json")
    parser.add_argument("--lm-order", type=int, default=4)
    parser.add_argument("--matrix-out", default="results/decoder_matrix.csv")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    rows: list[dict] = []
    rows.append(
        evaluate_decoder_config(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            split=args.split,
            decoder="greedy",
            experiment="D0_greedy",
            max_batches=args.max_batches,
            batch_size=args.batch_size,
        )
    )

    for beam_width in parse_ints(args.beam_widths):
        rows.append(
            evaluate_decoder_config(
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                split=args.split,
                decoder="prefix_beam",
                experiment=f"D1_beam_bw{beam_width}",
                beam_width=beam_width,
                topk=args.topk,
                max_batches=args.max_batches,
                batch_size=args.batch_size,
            )
        )
        for lm_alpha in parse_floats(args.lm_alphas):
            for lm_beta in parse_floats(args.lm_betas):
                rows.append(
                    evaluate_decoder_config(
                        config_path=args.config,
                        checkpoint_path=args.checkpoint,
                        split=args.split,
                        decoder="prefix_beam_lm",
                        experiment=f"D2_lm_bw{beam_width}_a{lm_alpha:g}_b{lm_beta:g}",
                        beam_width=beam_width,
                        topk=args.topk,
                        lm_path=args.lm_path,
                        lm_order=args.lm_order,
                        lm_alpha=lm_alpha,
                        lm_beta=lm_beta,
                        max_batches=args.max_batches,
                        batch_size=args.batch_size,
                    )
                )
                latest = rows[-1]
                print(
                    f"{latest['experiment']}: PER={float(latest['per']) * 100:.2f}% "
                    f"S={latest['substitutions']} D={latest['deletions']} I={latest['insertions']}"
                )

    rows = sorted(rows, key=lambda row: (float(row["per"]), int(row["substitutions"])))
    write_rows(args.matrix_out, rows)
    best = rows[0]
    print(
        "Best dev decoder: "
        f"{best['experiment']} PER={float(best['per']) * 100:.2f}% "
        f"S={best['substitutions']} D={best['deletions']} I={best['insertions']}"
    )
    print(f"Wrote decoder matrix to {args.matrix_out}")


if __name__ == "__main__":
    main()
