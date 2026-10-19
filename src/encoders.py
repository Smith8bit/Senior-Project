"""Transformer encoder stacks used by v2.xx models.

Two kinds (config model.encoder_norm):
  layernorm : PyTorch nn.TransformerEncoderLayer, pre-norm, ReLU feed-forward, attention dropout = dropout
              (what main models v1 / v2 and the first v2.01 build used).
  batchnorm : PatchTST's own encoder layer, written to match the official code (github.com/yuqinie98/PatchTST,
              layers/PatchTST_backbone.py `TSTEncoderLayer`, checked 2026-10-20): post-norm (normalise AFTER each
              residual add), BatchNorm1d over d_model via a transpose, GELU feed-forward
              Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model), attention dropout 0 (official default),
              dropout on both residual branches. The paper uses BatchNorm because "Zerveas et al. (2021) has shown
              that BatchNorm outperforms LayerNorm in time series Transformer" (Nie et al. 2023, sec. 3.1).

Both kinds have the same number of weights (BatchNorm and LayerNorm each hold 2 x d_model). BatchNorm also keeps
running statistics (buffers, not weights) that are used in eval mode, so a prediction never depends on the
other rows of its batch.
"""
from __future__ import annotations

import torch
from torch import nn

ENCODER_NORMS = ("layernorm", "batchnorm")


class BatchNormTokens(nn.Module):
    """BatchNorm1d over the d_model features of a [N, T, d] token tensor (PatchTST's Transpose-BatchNorm-Transpose)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.bn(z.transpose(1, 2)).transpose(1, 2)


class PatchTSTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, attn_dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = BatchNormTokens(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = BatchNormTokens(d_model)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src2 = self.self_attn(src, src, src, need_weights=False)[0]
        src = self.norm_attn(src + self.dropout_attn(src2))     # post-norm
        src = self.norm_ffn(src + self.dropout_ffn(self.ff(src)))
        return src


def make_encoder(kind: str, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float,
                 attn_dropout: float = 0.0) -> nn.Module:
    if kind == "layernorm":
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
    if kind == "batchnorm":
        return nn.Sequential(*[PatchTSTEncoderLayer(d_model, n_heads, d_ff, dropout, attn_dropout) for _ in range(n_layers)])
    raise ValueError(f"encoder_norm must be one of {ENCODER_NORMS}")
