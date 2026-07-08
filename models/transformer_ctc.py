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
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
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
            out_lengths: [B], unchanged because no time subsampling is used.
        """
        x = self.input_proj(features)
        x = self.pos_encoding(x)
        padding_mask = make_padding_mask(lengths, features.size(1))
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        logits = self.classifier(x)
        return logits, lengths
