from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataprocess.dataset import SpeechDataset, collate_fn
from dataprocess.vocab import load_vocab
from models.transformer_ctc import TransformerCTC
from utils.common import get_device, load_config
from utils.ctc_decode import ctc_greedy_decode
from utils.metrics import token_error_rate


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer-CTC ASR checkpoint")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--show", type=int, default=10, help="Number of prediction examples to print")
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
    with torch.no_grad():
        for batch in loader:
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
                if len(examples) < args.show:
                    examples.append((utt_id, ref, hyp))

    ter, err, ref_len = token_error_rate(refs_all, hyps_all)
    print(f"{args.split} loss: {total_loss/max(1, num_batches):.4f}")
    print(f"{args.split} PER/WER-like token error rate: {ter*100:.2f}% ({err}/{ref_len})")
    print("\nPrediction examples:")
    for utt_id, ref, hyp in examples:
        print(f"[{utt_id}]")
        print("REF:", " ".join(ref))
        print("HYP:", " ".join(hyp))
        print()


if __name__ == "__main__":
    main()
