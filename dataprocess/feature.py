from __future__ import annotations

import torch
import torchaudio
import soundfile as sf


class LogMelExtractor:
    """Extract log-Mel filterbank features from a waveform.

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
    ) -> None:
        self.sample_rate = sample_rate
        self.normalize = normalize
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

        feat = self.db(self.melspec(waveform))          # [1, n_mels, time]
        feat = feat.squeeze(0).transpose(0, 1).contiguous()  # [time, n_mels]

        if self.normalize:
            mean = feat.mean(dim=0, keepdim=True)
            std = feat.std(dim=0, keepdim=True).clamp_min(1e-5)
            feat = (feat - mean) / std

        return feat.float()
