from __future__ import annotations

import argparse

import torch

from dataprocess.feature import LogMelExtractor
from dataprocess.vocab import load_vocab
from models.transformer_ctc import TransformerCTC
from utils.common import get_device, load_config
from utils.ctc_decode import ctc_greedy_decode


def main():
    parser = argparse.ArgumentParser(description="Infer one wav file with Transformer-CTC ASR checkpoint")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--wav", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["train"].get("device", "auto"))
    token2id, id2token = load_vocab(cfg["vocab_path"])
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = TransformerCTC(vocab_size=len(token2id), **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    extractor = LogMelExtractor(**cfg["feature"])
    feat = extractor(args.wav).unsqueeze(0).to(device)  # [1, T, F]
    length = torch.tensor([feat.size(1)], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, out_lengths = model(feat, length)
        hyp = ctc_greedy_decode(logits, out_lengths, id2token)[0]
    print(" ".join(hyp))


if __name__ == "__main__":
    main()
