from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Return mask with True at padded positions for TransformerEncoder."""
    device = lengths.device
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def conv_out_length(lengths: torch.Tensor) -> torch.Tensor:
    """Length after Conv2d(kernel=3, stride=2, padding=1)."""
    return (lengths + 1) // 2


class Conv2dSubsampling(nn.Module):
    """Two-layer 2D convolutional frontend with about 4x time subsampling."""

    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        hidden_channels = max(1, d_model // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        freq_after = int(conv_out_length(conv_out_length(torch.tensor(input_dim))).item())
        self.out = nn.Linear(d_model * freq_after, d_model)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [B, T, F] -> [B, 1, T, F] -> [B, C, T', F']
        x = self.conv(features.unsqueeze(1))
        batch_size, channels, time_steps, freq_bins = x.shape
        x = x.transpose(1, 2).contiguous().view(batch_size, time_steps, channels * freq_bins)
        x = self.out(x)
        out_lengths = conv_out_length(conv_out_length(lengths))
        return x, out_lengths


class TransformerCTC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        frontend: str = "linear",
    ) -> None:
        super().__init__()
        if frontend not in {"linear", "conv2d_subsample4"}:
            raise ValueError(f"Unsupported frontend: {frontend}")
        self.frontend_type = frontend
        if frontend == "linear":
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.subsampling = Conv2dSubsampling(input_dim=input_dim, d_model=d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            features: [B, T, F]
            lengths: [B]
        Returns:
            logits: [B, T, vocab_size]
            out_lengths: [B]
        """
        if self.frontend_type == "linear":
            x = self.input_proj(features)
            out_lengths = lengths
        else:
            x, out_lengths = self.subsampling(features, lengths)
        x = self.pos_encoding(x)
        padding_mask = make_padding_mask(out_lengths, x.size(1))
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        logits = self.classifier(x)
        return logits, out_lengths
