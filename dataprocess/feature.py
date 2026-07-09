from __future__ import annotations

import torch
import torchaudio
import soundfile as sf


class LogMelExtractor:
    """Extract acoustic features from a waveform.

    Output shape: [time, n_mels].
    This version uses soundfile to read wav files, avoiding TorchCodec issues
    on Windows.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: float = 20.0,
        f_max: float | None = 7600.0,
        normalize: bool = True,
        backend: str = "log_mel",
        normalize_mode: str | None = None,
    ) -> None:
        if backend not in {"log_mel", "kaldi_fbank"}:
            raise ValueError(f"Unsupported feature backend: {backend}")
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.normalize = normalize
        self.backend = backend
        if normalize_mode is None:
            normalize_mode = "mean" if backend == "kaldi_fbank" else "mean_std"
        if normalize_mode not in {"mean", "mean_std"}:
            raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")
        self.normalize_mode = normalize_mode

        self.melspec = None
        self.db = None
        if backend == "log_mel":
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mels,
                power=2.0,
            )
            self.db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __call__(self, wav_path: str) -> torch.Tensor:
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)

        # soundfile: [num_samples, channels] -> torch: [channels, num_samples]
        waveform = torch.from_numpy(wav.T).contiguous()

        # convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        if self.backend == "kaldi_fbank":
            frame_length_ms = self.win_length / self.sample_rate * 1000.0
            frame_shift_ms = self.hop_length / self.sample_rate * 1000.0
            feat = torchaudio.compliance.kaldi.fbank(
                waveform,
                sample_frequency=self.sample_rate,
                frame_length=frame_length_ms,
                frame_shift=frame_shift_ms,
                num_mel_bins=self.n_mels,
                low_freq=self.f_min,
                high_freq=self.f_max or 0.0,
                window_type="hamming",
                preemphasis_coefficient=0.97,
                use_energy=False,
            )
        else:
            assert self.melspec is not None and self.db is not None
            feat = self.db(self.melspec(waveform))          # [1, n_mels, time]
            feat = feat.squeeze(0).transpose(0, 1).contiguous()  # [time, n_mels]

        if self.normalize:
            mean = feat.mean(dim=0, keepdim=True)
            feat = feat - mean
            if self.normalize_mode == "mean_std":
                std = feat.std(dim=0, keepdim=True).clamp_min(1e-5)
                feat = feat / std

        return feat.float()
