"""Multi-Scale PatchTST for SET50: trend (up/flat/down) and log-return at 1, 5 and 10 trading days.

Design (proof-of-life version, 2026-10-06; choices recorded in the project log, Phase 1.11-1.14):
  * Channel independence and patching as in PatchTST (Nie et al., ICLR 2023): every input channel
    (price, indicators, sentiment) is cut into patches and encoded by the SAME weights.
  * Multi-scale branches as in MTST (Zhang et al., AISTATS 2024): parallel branches with different
    patch lengths; each branch's output is flattened, the branches are concatenated and a linear
    layer fuses them.
  * RevIN (Kim et al., ICLR 2022): every window is normalised with its own mean and std, per channel,
    so no statistic crosses windows (hard rule 3 holds by construction on this pathway).
  * Fusion across channels (Phase 1.12), option 1 = late concatenation: the per-channel vectors are
    concatenated and fed to the heads. (Option 2, TimeXer-style cross-attention from price tokens
    to exogenous variate tokens, is the planned comparison.)
  * Two heads (both kept, as in the approved proposal): trend logits [batch, horizon, 3] and
    returns [batch, horizon]. Loss weighting by learned uncertainty (Kendall, Gal & Cipolla, CVPR 2018).
Positional encoding is a learnable absolute embedding per branch here; MTST uses relative encoding.
"""
from __future__ import annotations

import torch
from torch import nn


class RevIN(nn.Module):
    """Per-window, per-channel normalisation (statistics from the window itself only)."""

    def __init__(self, n_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(n_channels))
            self.bias = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: [B, L, C]
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        x = (x - mean) / std
        return x * self.weight + self.bias if self.affine else x


class PatchBranch(nn.Module):
    """One resolution: patch length P, stride S -> N tokens -> Transformer encoder -> flat vector."""

    def __init__(self, lookback: int, patch_len: int, stride: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        if patch_len > lookback:
            raise ValueError(f"patch length {patch_len} is longer than the look-back {lookback}")
        self.patch_len, self.stride = patch_len, stride
        self.n_tokens = (lookback - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.out_dim = self.n_tokens * d_model

    def forward(self, u: torch.Tensor) -> torch.Tensor:          # u: [B*C, L] (one channel per row)
        patches = u.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # [B*C, N, P]
        z = self.drop(self.embed(patches) + self.pos)
        return self.encoder(z).flatten(1)                        # [B*C, N*d]


class MultiScalePatchTST(nn.Module):
    def __init__(self, n_channels: int, lookback: int = 60, patches=((5, 5), (20, 10)), horizons=(1, 5, 10),
                 n_classes: int = 3, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, d_ff: int = 128,
                 d_channel: int = 64, dropout: float = 0.1, revin: bool = True):
        super().__init__()
        self.n_channels, self.lookback, self.horizons = n_channels, lookback, tuple(horizons)
        self.revin = RevIN(n_channels) if revin else None
        self.branches = nn.ModuleList(PatchBranch(lookback, p, s, d_model, n_heads, n_layers, d_ff, dropout) for p, s in patches)
        # MTST-style fusion of the resolutions: flatten, concatenate, linear
        self.fuse_scales = nn.Sequential(nn.Linear(sum(b.out_dim for b in self.branches), d_channel), nn.GELU(), nn.Dropout(dropout))
        # Phase 1.12 option 1: late concatenation across channels
        h = len(self.horizons)
        self.trunk = nn.Sequential(nn.Linear(n_channels * d_channel, d_channel * 2), nn.GELU(), nn.Dropout(dropout))
        self.trend_head = nn.Linear(d_channel * 2, h * n_classes)
        self.return_head = nn.Linear(d_channel * 2, h)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> dict:                 # x: [B, L, C], window ending at day t
        B, L, C = x.shape
        if L != self.lookback or C != self.n_channels:
            raise ValueError(f"expected [batch, {self.lookback}, {self.n_channels}], got {list(x.shape)}")
        if self.revin is not None:
            x = self.revin(x)
        u = x.permute(0, 2, 1).reshape(B * C, L)                # channel independence: shared weights per channel
        per_channel = self.fuse_scales(torch.cat([b(u) for b in self.branches], dim=-1))   # [B*C, d_channel]
        z = self.trunk(per_channel.reshape(B, C * per_channel.shape[-1]))
        return {"trend_logits": self.trend_head(z).reshape(B, len(self.horizons), self.n_classes),
                "returns": self.return_head(z)}


class UncertaintyWeightedLoss(nn.Module):
    """L = sum_i exp(-s_i) * L_i + s_i with learned s_i (Kendall, Gal & Cipolla 2018), one s per task.
    Stops either task from dominating just because of its scale (cross-entropy vs MSE)."""

    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(2))
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.mse = nn.MSELoss()

    def forward(self, out: dict, trend: torch.Tensor, returns: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l_cls = self.ce(out["trend_logits"].reshape(-1, out["trend_logits"].shape[-1]), trend.reshape(-1))
        l_reg = self.mse(out["returns"], returns)
        s = self.log_var
        total = torch.exp(-s[0]) * l_cls + s[0] + torch.exp(-s[1]) * l_reg + s[1]
        with torch.no_grad():
            parts = {"ce": l_cls.item(), "mse": l_reg.item(), "w_ce": torch.exp(-s[0]).item(), "w_mse": torch.exp(-s[1]).item()}
        return total, parts
