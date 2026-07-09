from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

import soundfile as sf
import torch
import torchaudio


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


def _as_list(value, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(item) for item in value]
    return [float(value)]


def apply_random_augmentation(
    waveform: torch.Tensor,
    sample_rate: int,
    config: dict | None,
    rng: random.Random | None = None,
) -> torch.Tensor:
    if not config or not config.get("enabled", False):
        return waveform
    rng = rng or random
    probability = float(config.get("probability", 0.0))
    if probability <= 0.0 or rng.random() >= probability:
        return waveform

    operations = list(config.get("operations") or [])
    if not operations:
        for name, key in [
            ("noise", "noise_snr_db"),
            ("speed", "speed_factors"),
            ("volume", "volume_gains"),
            ("echo", "echo_decays"),
        ]:
            if key in config:
                operations.append(name)
    if not operations:
        return waveform

    operation = rng.choice(operations)
    if operation == "noise":
        snr = rng.choice(_as_list(config.get("noise_snr_db"), [20.0]))
        return add_noise(waveform, snr_db=snr, seed=rng.randrange(2**31))
    if operation == "speed":
        factor = rng.choice(_as_list(config.get("speed_factors"), [0.95, 1.05]))
        return speed_perturb(waveform, sample_rate=sample_rate, factor=factor)
    if operation == "volume":
        gain = rng.choice(_as_list(config.get("volume_gains"), [0.8, 1.2]))
        return (waveform * gain).clamp(-1.0, 1.0)
    if operation == "echo":
        delay_ms = float(config.get("echo_delay_ms", 60.0))
        decay = rng.choice(_as_list(config.get("echo_decays"), [0.35]))
        return echo_perturb(waveform, sample_rate=sample_rate, delay_ms=delay_ms, decay=decay)
    raise ValueError(f"Unsupported audio augmentation operation: {operation}")
