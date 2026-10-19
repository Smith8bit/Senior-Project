"""Faithful PatchTST baseline (proposal objective 2: "PatchTST ต้นฉบับ"), adapted only where our task differs.

Follows Nie, Nguyen, Sinthong & Kalagnanam (2023, ICLR; arXiv 2211.14730) and the official code
(github.com/yuqinie98/PatchTST; checked 2026-10-20):
  * RevIN instance normalisation on the input (affine, subtract the mean, not the last value).
  * Channel independence: every input channel goes through the same patching, embedding and encoder.
  * Patching with end padding: S copies of the last value are appended (ReplicationPad1d((0, S))), so
    N = floor((L - P) / S) + 2 patches (paper sec. 3.1); the last patch always contains day t.
  * Linear patch embedding W_P (P -> d_model) + learnable position embedding, initialised uniform(-0.02, 0.02)
    (official pe='zeros', learn_pe=True), then dropout.
  * Encoder: PatchTST's TSTEncoderLayer (post-norm BatchNorm, GELU feed-forward; src/encoders.py).
  * Heads: the official ClassificationHead / RegressionHead (self-supervised code, models/patchTST.py): take each
    channel's LAST patch token, flatten all channels (C x d_model), dropout, one Linear layer.

Adapted (necessary for this task, to be stated in the report):
  * Two heads at once: classification (9 = 3 horizons x 3 classes, softmax per horizon) and regression
    (3 log returns), trained with the same multi-task loss as v2.01.
  * RevIN does not add the mean and std back, because the outputs are classes and future returns, not the
    input series.
  * Same 18 inputs, rows, labels, split and training protocol as v2.01, so the comparison isolates the architecture.
No stock or sector embedding (PatchTST has none).
"""
from __future__ import annotations

import torch
from torch import nn

from encoders import make_encoder
from multiscale_patchtst import RevIN


class FaithfulPatchTST(nn.Module):
    def __init__(self, *, n_channels: int, lookback: int, patch_len: int, stride: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, dropout: float, attn_dropout: float, head_dropout: float,
                 horizons=(1, 5, 10), n_classes: int = 3):
        super().__init__()
        if patch_len > lookback + stride:
            raise ValueError(f"patch length {patch_len} is longer than the padded look-back {lookback + stride}")
        self.n_channels, self.lookback, self.horizons, self.n_classes = n_channels, lookback, tuple(horizons), n_classes
        self.patch_len, self.stride = patch_len, stride
        self.n_tokens = (lookback - patch_len) // stride + 2                    # paper: N = floor((L-P)/S) + 2
        self.revin = RevIN(n_channels)
        self.pad = nn.ReplicationPad1d((0, stride))                             # S copies of the last value
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = nn.Parameter(torch.empty(self.n_tokens, d_model))
        nn.init.uniform_(self.W_pos, -0.02, 0.02)
        self.drop = nn.Dropout(dropout)
        self.encoder = make_encoder("batchnorm", d_model, n_heads, n_layers, d_ff, dropout, attn_dropout)
        h = len(self.horizons)
        self.trend_head = nn.Sequential(nn.Flatten(1), nn.Dropout(head_dropout), nn.Linear(n_channels * d_model, h * n_classes))
        self.price_head = nn.Sequential(nn.Flatten(1), nn.Dropout(head_dropout), nn.Linear(n_channels * d_model, h))

    def forward(self, x: torch.Tensor, stock: torch.Tensor | None = None, sector: torch.Tensor | None = None) -> dict:
        B, L, C = x.shape                                                       # ids accepted and ignored
        if L != self.lookback or C != self.n_channels:
            raise ValueError(f"expected [batch, {self.lookback}, {self.n_channels}], got {list(x.shape)}")
        x = self.revin(x)
        u = self.pad(x.permute(0, 2, 1)).reshape(B * C, L + self.stride)        # channel independence + end padding
        patches = u.unfold(-1, self.patch_len, self.stride)                     # [B*C, N, P]
        z = self.encoder(self.drop(self.W_P(patches) + self.W_pos))             # [B*C, N, d]
        last = z[:, -1, :].reshape(B, C, -1)                                    # last patch token of every channel
        return {"trend_logits": self.trend_head(last).reshape(B, len(self.horizons), self.n_classes),
                "returns": self.price_head(last)}
