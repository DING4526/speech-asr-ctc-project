from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pad_sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataprocess.dataset import read_wav_scp
from dataprocess.feature import LogMelExtractor
from dataprocess.vocab import encode_tokens, load_vocab, read_pinyin_file
from models.transformer_ctc import TransformerCTC
from utils.common import get_device, load_config
from utils.ctc_decode import ctc_greedy_decode
from utils.metrics import corpus_error_summary, utterance_error_summary


def stable_seed(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_waveform(path: str) -> tuple[torch.Tensor, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(wav.T).contiguous()
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, int(sr)


def add_noise(waveform: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=waveform.device).manual_seed(seed)
    noise = torch.randn(waveform.shape, generator=generator, device=waveform.device, dtype=waveform.dtype)
    signal_rms = waveform.pow(2).mean().sqrt().clamp_min(1e-8)
    noise_rms = noise.pow(2).mean().sqrt().clamp_min(1e-8)
    target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    return (waveform + noise * (target_noise_rms / noise_rms)).clamp(-1.0, 1.0)


def speed_perturb(waveform: torch.Tensor, sample_rate: int, factor: float) -> torch.Tensor:
    if factor <= 0:
        raise ValueError(f"speed factor must be positive: {factor}")
    orig_freq = max(1, int(round(sample_rate * factor)))
    return torchaudio.functional.resample(waveform, orig_freq=orig_freq, new_freq=sample_rate)


def echo_perturb(waveform: torch.Tensor, sample_rate: int, delay_ms: float = 60.0, decay: float = 0.35) -> torch.Tensor:
    delay = max(1, int(sample_rate * delay_ms / 1000.0))
    echoed = torch.zeros_like(waveform)
    if waveform.size(1) > delay:
        echoed[:, delay:] = waveform[:, :-delay] * decay
    out = waveform + echoed
    peak = out.abs().max().clamp_min(1.0)
    return (out / peak).clamp(-1.0, 1.0)


def apply_condition(waveform: torch.Tensor, sample_rate: int, condition: str, utt_id: str) -> torch.Tensor:
    if condition == "clean":
        return waveform
    if condition.startswith("noise"):
        snr_db = float(condition.replace("noise", ""))
        return add_noise(waveform, snr_db=snr_db, seed=stable_seed(f"{utt_id}-{condition}"))
    if condition.startswith("speed"):
        factor = float(condition.replace("speed", ""))
        return speed_perturb(waveform, sample_rate=sample_rate, factor=factor)
    if condition.startswith("volume"):
        gain = float(condition.replace("volume", ""))
        return (waveform * gain).clamp(-1.0, 1.0)
    if condition == "echo":
        return echo_perturb(waveform, sample_rate=sample_rate)
    raise ValueError(f"Unsupported robustness condition: {condition}")


def batch_items(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        features = [item["features"] for item in chunk]
        targets = [item["target"] for item in chunk]
        yield {
            "utt_ids": [item["utt_id"] for item in chunk],
            "features": pad_sequence(features, batch_first=True, padding_value=0.0),
            "feat_lengths": torch.tensor([x.size(0) for x in features], dtype=torch.long),
            "targets": torch.cat(targets, dim=0),
            "target_lengths": torch.tensor([x.numel() for x in targets], dtype=torch.long),
            "target_tokens": [item["target_tokens"] for item in chunk],
        }


@torch.no_grad()
def evaluate_condition(
    model,
    criterion,
    extractor: LogMelExtractor,
    wavs: dict[str, str],
    texts: dict[str, list[str]],
    token2id: dict[str, int],
    id2token: dict[int, str],
    condition: str,
    batch_size: int,
    device: torch.device,
    max_items: int | None = None,
) -> tuple[dict, list[dict]]:
    model.eval()
    items = []
    for utt_id in wavs:
        if utt_id not in texts:
            continue
        waveform, sample_rate = load_waveform(wavs[utt_id])
        waveform = apply_condition(waveform, sample_rate, condition, utt_id)
        features = extractor.extract_waveform(waveform, sample_rate)
        target_tokens = texts[utt_id]
        target = torch.tensor(encode_tokens(target_tokens, token2id), dtype=torch.long)
        items.append(
            {
                "utt_id": utt_id,
                "features": features,
                "target": target,
                "target_tokens": target_tokens,
            }
        )
        if max_items is not None and len(items) >= max_items:
            break

    refs_all = []
    hyps_all = []
    records = []
    total_loss = 0.0
    num_batches = 0
    for batch in batch_items(items, batch_size):
        features = batch["features"].to(device)
        feat_lengths = batch["feat_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        logits, out_lengths = model(features, feat_lengths)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        loss = criterion(log_probs, targets, out_lengths, target_lengths)
        total_loss += float(loss.item())
        num_batches += 1

        hyps = ctc_greedy_decode(logits, out_lengths, id2token)
        refs = batch["target_tokens"]
        refs_all.extend(refs)
        hyps_all.extend(hyps)
        for utt_id, ref, hyp in zip(batch["utt_ids"], refs, hyps):
            stats = utterance_error_summary(ref, hyp)
            records.append(
                {
                    "condition": condition,
                    "utt_id": utt_id,
                    "ref": " ".join(ref),
                    "hyp": " ".join(hyp),
                    **stats,
                }
            )

    summary = corpus_error_summary(refs_all, hyps_all)
    summary["condition"] = condition
    summary["loss"] = total_loss / max(1, num_batches)
    return summary, records


def write_summary_csv(path: str | Path, summaries: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "condition",
        "loss",
        "num_utterances",
        "ref_tokens",
        "hyp_tokens",
        "errors",
        "substitutions",
        "deletions",
        "insertions",
        "per",
        "relative_degradation",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_records_csv(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "condition",
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
        for row in records:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASR robustness under deterministic waveform perturbations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--conditions", default="clean,noise20,noise10,noise5,speed0.9,speed1.1,volume0.5,volume1.5,echo")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--records-out", default=None)
    parser.add_argument("--analysis-out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["train"].get("device", "auto"))
    token2id, id2token = load_vocab(cfg["vocab_path"])
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = TransformerCTC(vocab_size=len(token2id), **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    extractor = LogMelExtractor(**cfg["feature"])

    wavs = read_wav_scp(cfg["data"][f"{args.split}_wav_scp"])
    texts = read_pinyin_file(cfg["data"][f"{args.split}_text"])
    batch_size = args.batch_size or int(cfg["train"]["batch_size"])

    summaries = []
    all_records = []
    for condition in [item.strip() for item in args.conditions.split(",") if item.strip()]:
        print(f"Evaluating condition: {condition}")
        summary, records = evaluate_condition(
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
        summaries.append(summary)
        all_records.extend(records)
        print(f"{condition}: PER={summary['per'] * 100:.2f}% ({summary['errors']}/{summary['ref_tokens']})")

    clean_per = next((row["per"] for row in summaries if row["condition"] == "clean"), None)
    if clean_per is not None:
        for row in summaries:
            row["relative_degradation"] = (row["per"] - clean_per) / clean_per if clean_per else 0.0

    analysis = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "conditions": summaries,
    }
    Path(args.analysis_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.analysis_out).write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(args.summary_out, summaries)
    if args.records_out:
        write_records_csv(args.records_out, all_records)

    print(f"Wrote summary CSV to {args.summary_out}")
    print(f"Wrote analysis JSON to {args.analysis_out}")


if __name__ == "__main__":
    main()
