from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from dataprocess.feature import LogMelExtractor
from dataprocess.vocab import read_pinyin_file, encode_tokens
from utils.audio_augment import apply_random_augmentation, load_waveform


def read_wav_scp(path: str | Path) -> Dict[str, str]:
    """Read Kaldi-style wav.scp: utt_id<TAB>wav_path or utt_id wav_path."""
    path = Path(path)
    wavs: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                utt_id, wav_path = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"{path}:{lineno} should be 'utt_id wav_path'")
                utt_id, wav_path = parts
            wavs[utt_id] = wav_path.strip()
    return wavs


class SpeechDataset(Dataset):
    def __init__(
        self,
        wav_scp: str | Path,
        text_path: str | Path,
        token2id: Dict[str, int],
        feature_conf: dict,
        audio_augment_conf: dict | None = None,
    ) -> None:
        self.wavs = read_wav_scp(wav_scp)
        self.texts = read_pinyin_file(text_path)
        self.utt_ids = [utt for utt in self.wavs.keys() if utt in self.texts]
        missing_text = set(self.wavs.keys()) - set(self.texts.keys())
        if missing_text:
            print(f"[WARN] {len(missing_text)} utterances have wav but no pinyin text; ignored.")
        if not self.utt_ids:
            raise RuntimeError("No matched utterances found between wav.scp and pinyin file.")
        self.token2id = token2id
        self.extractor = LogMelExtractor(**feature_conf)
        self.audio_augment_conf = audio_augment_conf or {}
        self.rng = random.Random(int(self.audio_augment_conf.get("seed", 2026)))

    def __len__(self) -> int:
        return len(self.utt_ids)

    def __getitem__(self, index: int) -> dict:
        utt_id = self.utt_ids[index]
        wav_path = self.wavs[utt_id]
        if not Path(wav_path).exists():
            raise FileNotFoundError(f"Audio file for {utt_id} not found: {wav_path}")
        if self.audio_augment_conf.get("enabled", False):
            waveform, sample_rate = load_waveform(wav_path)
            waveform = apply_random_augmentation(waveform, sample_rate, self.audio_augment_conf, self.rng)
            feat = self.extractor.extract_waveform(waveform, sample_rate)
        else:
            feat = self.extractor(wav_path)
        target_tokens = self.texts[utt_id]
        target = torch.tensor(encode_tokens(target_tokens, self.token2id), dtype=torch.long)
        return {
            "utt_id": utt_id,
            "features": feat,
            "target": target,
            "target_tokens": target_tokens,
        }


def collate_fn(batch: List[dict]) -> dict:
    feats = [item["features"] for item in batch]
    targets = [item["target"] for item in batch]
    utt_ids = [item["utt_id"] for item in batch]
    target_tokens = [item["target_tokens"] for item in batch]

    feat_lengths = torch.tensor([x.size(0) for x in feats], dtype=torch.long)
    target_lengths = torch.tensor([x.numel() for x in targets], dtype=torch.long)
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
    flat_targets = torch.cat(targets, dim=0)
    return {
        "utt_ids": utt_ids,
        "features": padded_feats,          # [B, T, F]
        "feat_lengths": feat_lengths,     # [B]
        "targets": flat_targets,          # [sum target lengths]
        "target_lengths": target_lengths, # [B]
        "target_tokens": target_tokens,
    }
