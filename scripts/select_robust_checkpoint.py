from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataprocess.dataset import read_wav_scp
from dataprocess.feature import LogMelExtractor
from dataprocess.vocab import load_vocab, read_pinyin_file
from models.transformer_ctc import TransformerCTC
from scripts.evaluate_robustness import evaluate_condition
from utils.common import get_device, load_config


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1))
    return -1


def write_rows(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "checkpoint",
        "epoch",
        "condition",
        "per",
        "errors",
        "ref_tokens",
        "substitutions",
        "deletions",
        "insertions",
        "robust_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Select checkpoint by mean dev PER over robustness conditions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--pattern", default="epoch_*.pt")
    parser.add_argument("--checkpoints", default=None, help="Optional comma-separated checkpoint paths; overrides pattern.")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--conditions", default="clean,noise20,speed0.9,speed1.1,echo")
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--analysis-out", required=True)
    parser.add_argument("--selected-out", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["train"].get("device", "auto"))
    token2id, id2token = load_vocab(cfg["vocab_path"])
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    extractor = LogMelExtractor(**cfg["feature"])
    wavs = read_wav_scp(cfg["data"][f"{args.split}_wav_scp"])
    texts = read_pinyin_file(cfg["data"][f"{args.split}_text"])
    batch_size = int(cfg["train"]["batch_size"])
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]

    if args.checkpoints:
        checkpoints = [Path(item.strip()) for item in args.checkpoints.split(",") if item.strip()]
    else:
        checkpoints = sorted(Path(args.checkpoint_dir).glob(args.pattern), key=checkpoint_epoch)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints matched {args.checkpoint_dir}/{args.pattern}")

    all_rows = []
    checkpoint_summaries = []
    for checkpoint in checkpoints:
        ckpt = torch.load(checkpoint, map_location=device)
        model = TransformerCTC(vocab_size=len(token2id), **cfg["model"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        condition_summaries = []
        for condition in conditions:
            summary, _ = evaluate_condition(
                model=model,
                criterion=criterion,
                extractor=extractor,
                wavs=wavs,
                texts=texts,
                token2id=token2id,
                id2token=id2token,
                condition=condition,
                batch_size=batch_size,
                device=device,
                max_items=args.max_items,
            )
            condition_summaries.append(summary)
        robust_score = sum(row["per"] for row in condition_summaries) / max(1, len(condition_summaries))
        epoch = int(ckpt.get("epoch", checkpoint_epoch(checkpoint)))
        for row in condition_summaries:
            all_rows.append(
                {
                    "checkpoint": str(checkpoint),
                    "epoch": epoch,
                    "condition": row["condition"],
                    "per": row["per"],
                    "errors": row["errors"],
                    "ref_tokens": row["ref_tokens"],
                    "substitutions": row["substitutions"],
                    "deletions": row["deletions"],
                    "insertions": row["insertions"],
                    "robust_score": robust_score,
                }
            )
        checkpoint_summaries.append(
            {
                "checkpoint": str(checkpoint),
                "epoch": epoch,
                "robust_score": robust_score,
                "conditions": condition_summaries,
            }
        )
        print(f"{checkpoint.name}: robust_dev_score={robust_score * 100:.2f}%")

    best = min(checkpoint_summaries, key=lambda row: (row["robust_score"], row["epoch"]))
    if args.selected_out:
        selected_out = Path(args.selected_out)
        selected_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best["checkpoint"], selected_out)

    analysis = {
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "split": args.split,
        "conditions": conditions,
        "best": best,
        "checkpoints": checkpoint_summaries,
    }
    Path(args.analysis_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.analysis_out).write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_rows(args.summary_out, all_rows)
    print(
        f"Best robust checkpoint: epoch {best['epoch']} "
        f"score={best['robust_score'] * 100:.2f}% ({best['checkpoint']})"
    )
    print(f"Wrote robust dev summary to {args.summary_out}")


if __name__ == "__main__":
    main()
