from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataprocess.dataset import SpeechDataset, collate_fn
from dataprocess.vocab import load_vocab
from models.transformer_ctc import TransformerCTC
from utils.common import get_device, load_config
from utils.ctc_decode import ctc_greedy_decode, ctc_prefix_beam_search_decode
from utils.metrics import corpus_error_summary, token_error_rate, utterance_error_summary
from utils.ngram_lm import PinyinNgramLM


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
    out = []
    for label in sorted(buckets):
        bucket = buckets[label]
        bucket["per"] = bucket["errors"] / bucket["ref_tokens"] if bucket["ref_tokens"] else 0.0
        out.append(bucket)
    return out


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


def append_matrix_row(path: str | Path, row: dict) -> None:
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
        "predictions_path",
        "analysis_path",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def load_or_train_lm(
    lm_path: str | None,
    train_text: str,
    order: int,
    delta: float,
    backoff: float,
) -> PinyinNgramLM:
    if lm_path and Path(lm_path).exists():
        return PinyinNgramLM.load(lm_path)
    lm = PinyinNgramLM.from_pinyin_file(train_text, order=order, delta=delta, backoff=backoff)
    if lm_path:
        lm.save(lm_path)
    return lm


def decode_logits(
    logits: torch.Tensor,
    out_lengths: torch.Tensor,
    id2token: dict[int, str],
    decoder: str,
    beam_width: int,
    topk: int,
    lm: PinyinNgramLM | None,
    lm_alpha: float,
    lm_beta: float,
) -> list[list[str]]:
    if decoder == "greedy":
        return ctc_greedy_decode(logits, out_lengths, id2token)
    if decoder == "prefix_beam":
        return ctc_prefix_beam_search_decode(
            logits,
            out_lengths,
            id2token,
            beam_width=beam_width,
            topk=topk,
        )
    if decoder == "prefix_beam_lm":
        if lm is None:
            raise ValueError("prefix_beam_lm requires an LM.")
        return ctc_prefix_beam_search_decode(
            logits,
            out_lengths,
            id2token,
            beam_width=beam_width,
            topk=topk,
            lm=lm,
            lm_alpha=lm_alpha,
            lm_beta=lm_beta,
        )
    raise ValueError(f"Unsupported decoder: {decoder}")


@torch.no_grad()
def evaluate_decoder_config(
    config_path: str,
    checkpoint_path: str,
    split: str,
    decoder: str,
    experiment: str,
    beam_width: int = 25,
    topk: int = 20,
    lm_path: str | None = None,
    lm_order: int = 4,
    lm_alpha: float = 0.0,
    lm_beta: float = 0.0,
    lm_delta: float = 0.1,
    lm_backoff: float = 0.4,
    predictions_out: str | None = None,
    analysis_out: str | None = None,
    matrix_out: str | None = None,
    max_batches: int | None = None,
    batch_size: int | None = None,
) -> dict:
    cfg = load_config(config_path)
    device = get_device(cfg["train"].get("device", "auto"))
    token2id, id2token = load_vocab(cfg["vocab_path"])
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = TransformerCTC(vocab_size=len(token2id), **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    lm = None
    if decoder == "prefix_beam_lm":
        lm = load_or_train_lm(
            lm_path=lm_path,
            train_text=cfg["data"]["train_text"],
            order=lm_order,
            delta=lm_delta,
            backoff=lm_backoff,
        )

    wav_scp_key = f"{split}_wav_scp"
    text_key = f"{split}_text"
    dataset = SpeechDataset(cfg["data"][wav_scp_key], cfg["data"][text_key], token2id, cfg["feature"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size or int(cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn,
    )
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    refs_all, hyps_all = [], []
    total_loss, num_batches = 0.0, 0
    records = []
    decode_seconds = 0.0
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

        start = time.perf_counter()
        hyps = decode_logits(
            logits=logits,
            out_lengths=out_lengths,
            id2token=id2token,
            decoder=decoder,
            beam_width=beam_width,
            topk=topk,
            lm=lm,
            lm_alpha=lm_alpha,
            lm_beta=lm_beta,
        )
        decode_seconds += time.perf_counter() - start

        refs = batch["target_tokens"]
        refs_all.extend(refs)
        hyps_all.extend(hyps)
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
        if max_batches is not None and step >= max_batches:
            break

    ter, err, ref_len = token_error_rate(refs_all, hyps_all)
    summary = corpus_error_summary(refs_all, hyps_all)
    analysis = {
        "experiment": experiment,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "split": split,
        "decoder": decoder,
        "beam_width": beam_width,
        "topk": topk,
        "lm_order": lm_order if decoder == "prefix_beam_lm" else "",
        "lm_alpha": lm_alpha if decoder == "prefix_beam_lm" else "",
        "lm_beta": lm_beta if decoder == "prefix_beam_lm" else "",
        "loss": total_loss / max(1, num_batches),
        "summary": summary,
        "decode_seconds": decode_seconds,
        "by_reference_length": summarize_length_bins(records),
        "worst_examples": sorted(records, key=lambda x: (int(x["errors"]), float(x["per"])), reverse=True)[:10],
    }
    row = {
        "experiment": experiment,
        "split": split,
        "decoder": decoder,
        "beam_width": beam_width if decoder != "greedy" else "",
        "topk": topk if decoder != "greedy" else "",
        "lm_order": lm_order if decoder == "prefix_beam_lm" else "",
        "lm_alpha": lm_alpha if decoder == "prefix_beam_lm" else "",
        "lm_beta": lm_beta if decoder == "prefix_beam_lm" else "",
        "loss": analysis["loss"],
        "per": ter,
        "errors": err,
        "ref_tokens": ref_len,
        "substitutions": summary["substitutions"],
        "deletions": summary["deletions"],
        "insertions": summary["insertions"],
        "decode_seconds": decode_seconds,
        "predictions_path": predictions_out or "",
        "analysis_path": analysis_out or "",
    }

    if predictions_out:
        write_predictions_csv(predictions_out, records)
    if analysis_out:
        write_analysis_json(analysis_out, analysis)
    if matrix_out:
        append_matrix_row(matrix_out, row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with greedy, beam, or beam+LM decoding.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--experiment", default="decoder_eval")
    parser.add_argument("--decoder", choices=["greedy", "prefix_beam", "prefix_beam_lm"], default="greedy")
    parser.add_argument("--beam-width", type=int, default=25)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--lm-path", default="results/lm/pinyin_4gram.json")
    parser.add_argument("--lm-order", type=int, default=4)
    parser.add_argument("--lm-alpha", type=float, default=0.5)
    parser.add_argument("--lm-beta", type=float, default=0.0)
    parser.add_argument("--lm-delta", type=float, default=0.1)
    parser.add_argument("--lm-backoff", type=float, default=0.4)
    parser.add_argument("--predictions-out", default=None)
    parser.add_argument("--analysis-out", default=None)
    parser.add_argument("--matrix-out", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    row = evaluate_decoder_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        decoder=args.decoder,
        experiment=args.experiment,
        beam_width=args.beam_width,
        topk=args.topk,
        lm_path=args.lm_path,
        lm_order=args.lm_order,
        lm_alpha=args.lm_alpha,
        lm_beta=args.lm_beta,
        lm_delta=args.lm_delta,
        lm_backoff=args.lm_backoff,
        predictions_out=args.predictions_out,
        analysis_out=args.analysis_out,
        matrix_out=args.matrix_out,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
    )
    print(
        f"{row['experiment']} {row['split']} {row['decoder']}: "
        f"PER={float(row['per']) * 100:.2f}% "
        f"S={row['substitutions']} D={row['deletions']} I={row['insertions']} "
        f"decode_seconds={float(row['decode_seconds']):.2f}"
    )


if __name__ == "__main__":
    main()
