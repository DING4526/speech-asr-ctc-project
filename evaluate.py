from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataprocess.dataset import SpeechDataset, collate_fn
from dataprocess.vocab import load_vocab
from models.transformer_ctc import TransformerCTC
from utils.common import get_device, load_config
from utils.ctc_decode import ctc_greedy_decode
from utils.metrics import corpus_error_summary, token_error_rate, utterance_error_summary


def length_bin(ref_len: int) -> str:
    if ref_len <= 10:
        return "001-010"
    if ref_len <= 20:
        return "011-020"
    if ref_len <= 30:
        return "021-030"
    if ref_len <= 40:
        return "031-040"
    return "041+"


def summarize_length_bins(records: list[dict]) -> list[dict]:
    buckets: dict[str, dict] = {}
    for record in records:
        label = length_bin(int(record["ref_len"]))
        bucket = buckets.setdefault(
            label,
            {
                "length_bin": label,
                "num_utterances": 0,
                "ref_tokens": 0,
                "hyp_tokens": 0,
                "errors": 0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
            },
        )
        bucket["num_utterances"] += 1
        bucket["ref_tokens"] += int(record["ref_len"])
        bucket["hyp_tokens"] += int(record["hyp_len"])
        bucket["errors"] += int(record["errors"])
        bucket["substitutions"] += int(record["substitutions"])
        bucket["deletions"] += int(record["deletions"])
        bucket["insertions"] += int(record["insertions"])

    summaries = []
    for label in sorted(buckets):
        bucket = buckets[label]
        bucket["per"] = bucket["errors"] / bucket["ref_tokens"] if bucket["ref_tokens"] else 0.0
        summaries.append(bucket)
    return summaries


def write_predictions_csv(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "utt_id",
        "ref",
        "hyp",
        "ref_len",
        "hyp_len",
        "errors",
        "substitutions",
        "deletions",
        "insertions",
        "per",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})


def write_analysis_json(path: str | Path, analysis: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer-CTC ASR checkpoint")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--show", type=int, default=10, help="Number of prediction examples to print")
    parser.add_argument("--predictions-out", default=None, help="Optional CSV path for per-utterance predictions")
    parser.add_argument("--analysis-out", default=None, help="Optional JSON path for aggregate error analysis")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional maximum number of batches to evaluate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["train"].get("device", "auto"))
    token2id, id2token = load_vocab(cfg["vocab_path"])

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = TransformerCTC(vocab_size=len(token2id), **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wav_scp_key = f"{args.split}_wav_scp"
    text_key = f"{args.split}_text"
    dataset = SpeechDataset(cfg["data"][wav_scp_key], cfg["data"][text_key], token2id, cfg["feature"])
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_fn)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    refs_all, hyps_all = [], []
    total_loss, num_batches = 0.0, 0
    examples = []
    records = []
    with torch.no_grad():
        for step, batch in enumerate(loader, 1):
            features = batch["features"].to(device)
            lengths = batch["feat_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            logits, out_lengths = model(features, lengths)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = criterion(log_probs, targets, out_lengths, target_lengths)
            total_loss += float(loss.item())
            num_batches += 1
            hyps = ctc_greedy_decode(logits, out_lengths, id2token)
            refs = batch["target_tokens"]
            hyps_all.extend(hyps)
            refs_all.extend(refs)
            for utt_id, ref, hyp in zip(batch["utt_ids"], refs, hyps):
                stats = utterance_error_summary(ref, hyp)
                records.append(
                    {
                        "utt_id": utt_id,
                        "ref": " ".join(ref),
                        "hyp": " ".join(hyp),
                        **stats,
                    }
                )
                if len(examples) < args.show:
                    examples.append((utt_id, ref, hyp))
            if args.max_batches is not None and step >= args.max_batches:
                break

    ter, err, ref_len = token_error_rate(refs_all, hyps_all)
    summary = corpus_error_summary(refs_all, hyps_all)
    by_length = summarize_length_bins(records)
    worst_by_errors = sorted(records, key=lambda x: (int(x["errors"]), float(x["per"])), reverse=True)[:10]
    analysis = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "loss": total_loss / max(1, num_batches),
        "summary": summary,
        "by_reference_length": by_length,
        "worst_examples": worst_by_errors,
    }

    print(f"{args.split} loss: {total_loss/max(1, num_batches):.4f}")
    print(f"{args.split} PER/WER-like token error rate: {ter*100:.2f}% ({err}/{ref_len})")
    print(
        "error breakdown: "
        f"S={summary['substitutions']} D={summary['deletions']} I={summary['insertions']}"
    )
    if by_length:
        print("\nPER by reference length:")
        for item in by_length:
            print(
                f"{item['length_bin']}: {item['per']*100:.2f}% "
                f"({item['errors']}/{item['ref_tokens']}, n={item['num_utterances']})"
            )
    print("\nPrediction examples:")
    for utt_id, ref, hyp in examples:
        print(f"[{utt_id}]")
        print("REF:", " ".join(ref))
        print("HYP:", " ".join(hyp))
        print()

    if args.predictions_out:
        write_predictions_csv(args.predictions_out, records)
        print(f"Wrote predictions CSV to {args.predictions_out}")
    if args.analysis_out:
        write_analysis_json(args.analysis_out, analysis)
        print(f"Wrote analysis JSON to {args.analysis_out}")


if __name__ == "__main__":
    main()
