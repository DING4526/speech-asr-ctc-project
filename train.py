from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataprocess.dataset import SpeechDataset, collate_fn
from dataprocess.vocab import build_vocab_from_files, load_vocab
from models.transformer_ctc import TransformerCTC
from utils.common import ensure_dir, get_device, load_config, set_seed
from utils.ctc_decode import ctc_greedy_decode
from utils.metrics import token_error_rate
from utils.specaugment import specaugment


def run_eval(model, loader, criterion, device, id2token, max_batches: int | None = None):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    refs_all = []
    hyps_all = []
    with torch.no_grad():
        for step, batch in enumerate(loader, 1):
            features = batch["features"].to(device)
            feat_lengths = batch["feat_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            logits, out_lengths = model(features, feat_lengths)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # [T, B, V]
            loss = criterion(log_probs, targets, out_lengths, target_lengths)
            total_loss += float(loss.item())
            num_batches += 1
            hyps_all.extend(ctc_greedy_decode(logits, out_lengths, id2token))
            refs_all.extend(batch["target_tokens"])
            if max_batches is not None and step >= max_batches:
                break
    ter, err, ref_len = token_error_rate(refs_all, hyps_all)
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, ter, err, ref_len


def main():
    parser = argparse.ArgumentParser(description="Train Transformer-CTC ASR baseline")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 2026)))
    if "num_threads" in cfg.get("train", {}):
        torch.set_num_threads(int(cfg["train"]["num_threads"]))
    device = get_device(cfg["train"].get("device", "auto"))
    print(f"Using device: {device}")

    vocab_path = Path(cfg["vocab_path"])
    if not vocab_path.exists():
        print(f"Vocabulary not found. Building vocabulary at {vocab_path} ...")
        build_vocab_from_files([cfg["data"]["train_text"]], vocab_path)
    token2id, id2token = load_vocab(vocab_path)
    vocab_size = len(token2id)
    print(f"Vocabulary size: {vocab_size}")

    feature_conf = cfg["feature"]
    train_set = SpeechDataset(cfg["data"]["train_wav_scp"], cfg["data"]["train_text"], token2id, feature_conf)
    dev_set = SpeechDataset(cfg["data"]["dev_wav_scp"], cfg["data"]["dev_text"], token2id, feature_conf)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
        collate_fn=collate_fn,
    )

    model = TransformerCTC(vocab_size=vocab_size, **cfg["model"]).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer_name = str(cfg["train"].get("optimizer", "adamw")).lower()
    optimizer_cls = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }.get(optimizer_name)
    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    optimizer = optimizer_cls(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    start_epoch = 1
    best_dev_ter = 1e9
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_dev_ter = float(ckpt.get("best_dev_per", ckpt.get("dev_per", best_dev_ter)))
        print(f"Resumed from {args.resume}: start_epoch={start_epoch}, best_dev_PER={best_dev_ter*100:.2f}%")

    output_dir = Path(cfg["output_dir"])
    ensure_dir(output_dir)
    metrics_path = output_dir / "metrics.csv"
    append_metrics = bool(args.resume and metrics_path.exists())
    max_train_batches = cfg["train"].get("max_train_batches")
    max_eval_batches = cfg["train"].get("max_eval_batches")
    if max_train_batches is not None:
        max_train_batches = int(max_train_batches)
    if max_eval_batches is not None:
        max_eval_batches = int(max_eval_batches)

    with metrics_path.open("a" if append_metrics else "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "dev_loss", "dev_per", "dev_errors", "dev_ref_tokens", "is_best"],
        )
        if not append_metrics:
            writer.writeheader()

        for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
            model.train()
            running_loss = 0.0
            train_batches = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            for step, batch in enumerate(pbar, 1):
                features = batch["features"].to(device)
                feat_lengths = batch["feat_lengths"].to(device)
                targets = batch["targets"].to(device)
                target_lengths = batch["target_lengths"].to(device)

                if cfg["train"].get("use_specaugment", False):
                    features = specaugment(
                        features,
                        feat_lengths,
                        time_mask_param=int(cfg["train"].get("time_mask_param", 24)),
                        freq_mask_param=int(cfg["train"].get("freq_mask_param", 10)),
                        num_time_masks=int(cfg["train"].get("num_time_masks", 2)),
                        num_freq_masks=int(cfg["train"].get("num_freq_masks", 2)),
                    )

                logits, out_lengths = model(features, feat_lengths)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                loss = criterion(log_probs, targets, out_lengths, target_lengths)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 5.0)))
                optimizer.step()

                running_loss += float(loss.item())
                train_batches += 1
                if step % int(cfg["train"].get("log_interval", 20)) == 0:
                    pbar.set_postfix(loss=running_loss / step)
                if max_train_batches is not None and step >= max_train_batches:
                    break

            train_loss = running_loss / max(1, train_batches)
            dev_loss, dev_ter, dev_err, dev_ref = run_eval(
                model,
                dev_loader,
                criterion,
                device,
                id2token,
                max_batches=max_eval_batches,
            )
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"dev_loss={dev_loss:.4f}, dev_PER={dev_ter*100:.2f}% "
                f"({dev_err}/{dev_ref})"
            )

            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                "token2id": token2id,
                "epoch": epoch,
                "dev_per": dev_ter,
                "best_dev_per": min(best_dev_ter, dev_ter),
            }
            torch.save(ckpt, output_dir / "last.pt")
            is_best = dev_ter < best_dev_ter
            if is_best:
                best_dev_ter = dev_ter
                torch.save(ckpt, output_dir / "best.pt")
                print(f"Saved new best checkpoint: dev_PER={best_dev_ter*100:.2f}%")

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "dev_loss": f"{dev_loss:.6f}",
                    "dev_per": f"{dev_ter:.6f}",
                    "dev_errors": dev_err,
                    "dev_ref_tokens": dev_ref,
                    "is_best": int(is_best),
                }
            )
            f.flush()
    print(f"Training metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
