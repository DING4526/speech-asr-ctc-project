from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torchaudio


TOKENS = [
    "wo ai yu yin",
    "jin tian xue xi",
    "zhong wen shi bie",
    "ni hao nan kai",
    "ji qi xue xi",
    "shen du xue xi",
]


def synth_wave(index: int, sr: int = 16000, seconds: float = 0.35) -> torch.Tensor:
    t = torch.linspace(0, seconds, int(sr * seconds))
    freq = 220 + 35 * index
    wav = 0.25 * torch.sin(2 * math.pi * freq * t)
    wav += 0.08 * torch.sin(2 * math.pi * (freq * 2.1) * t)
    return wav.unsqueeze(0)


def write_split(root: Path, split: str, start_idx: int, count: int) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    wav_scp_lines = []
    text_lines = []
    for i in range(count):
        idx = start_idx + i
        utt_id = f"{split}_{idx:04d}"
        wav_path = split_dir / f"{utt_id}.wav"
        torchaudio.save(str(wav_path), synth_wave(idx), 16000)
        wav_scp_lines.append(f"{utt_id}\t{wav_path.resolve()}\n")
        text_lines.append(f"{utt_id}\t{TOKENS[idx % len(TOKENS)]}\n")
    (split_dir / "wav.scp").write_text("".join(wav_scp_lines), encoding="utf-8")
    (split_dir / "pinyin").write_text("".join(text_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create a tiny synthetic dataset only for code smoke testing")
    parser.add_argument("--root", default="data/toy")
    args = parser.parse_args()
    root = Path(args.root)
    write_split(root, "train", 0, 24)
    write_split(root, "dev", 100, 6)
    write_split(root, "test", 200, 6)
    print(f"Toy dataset written to {root.resolve()}")
    print("Edit configs/base.yaml data paths if you want to train on this toy dataset.")


if __name__ == "__main__":
    main()
