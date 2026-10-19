"""Proposal-accurate Multi-Scale PatchTST (experiment v2.01): the architecture on presentation slides 10-14 and 19.

Slide map (each block below is marked with the slide it implements):
  [S11] Multi-scale patching: patch lengths 4 / 8 / 16 (strides from the config)
  [S11] Per scale: patch embedding + its own position embedding -> Transformer layer(s) -> mean pooling
        (patch lengths and strides come from the config; the slide's 4 / 8 / 16 were placeholders, Siwat 2026-10-19)
  [S19] Attention-based fusion of the scales: stack -> MLP -> softmax -> weights -> weighted sum
  [S12] Stock embedding (stock_id and sector_id, slide 9) concatenated -> one Linear layer (-> 128)
  [S12] Price head: Linear 128->128, GELU, Dropout, Linear 128->64, GELU, Linear 64->3 (log return per horizon)
  [S13] Trend head: Linear 128->128, GELU, Dropout, Linear 128->64, GELU, Linear 64->9, reshape (3, 3),
        softmax per horizon (the softmax is applied in the loss / at prediction time)

Encoder layers (config model.encoder_norm): "layernorm" (PyTorch pre-norm) or "batchnorm" (PatchTST's own
post-norm BatchNorm layer, src/encoders.py).

Two ways to feed the 14 input channels into the patch encoder (slide 10 shows "fused data"; config model.channel_mode):
  independent: every channel is patched and encoded on its own with shared weights (PatchTST, Nie et al. 2023);
               the per-channel vectors are concatenated before the Linear layer.
  mixed:       each patch holds all channels of its days (patch_len x C values) and becomes one token.

No number has a default here: every size comes from the run config (configs/v2_01.json), so nothing is guessed.
"""
from __future__ import annotations

import torch
from torch import nn

from encoders import make_encoder
from multiscale_patchtst import RevIN, UncertaintyWeightedLoss

CHANNEL_MODES = ("independent", "mixed")


# ---------------------------------------------------------------- [S11] one scale: patch -> encode -> mean pool
class ScaleBranch(nn.Module):
    def __init__(self, lookback: int, patch_len: int, stride: int, in_per_step: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, dropout: float, encoder_norm: str = "layernorm"):
        super().__init__()
        if patch_len > lookback:
            raise ValueError(f"patch length {patch_len} is longer than the look-back {lookback}")
        self.patch_len, self.stride = patch_len, stride
        self.n_tokens = (lookback - patch_len) // stride + 1
        self.offset = (lookback - patch_len) % stride                           # patches end exactly at day t;
        #                                                                         any leftover days are the OLDEST
        self.embed = nn.Linear(patch_len * in_per_step, d_model)               # patch -> token
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))        # position embedding (own per scale)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.encoder = make_encoder(encoder_norm, d_model, n_heads, n_layers, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [N, L] (independent: one channel per row) or [N, L, C] (mixed: all channels)
        if u.dim() == 2:
            patches = u[:, self.offset:].unfold(-1, self.patch_len, self.stride)              # [N, T, P]
        else:
            patches = u[:, self.offset:, :].unfold(1, self.patch_len, self.stride).flatten(2)  # [N, T, C*P]
        z = self.drop(self.embed(patches) + self.pos)                           # [N, T, d]
        return self.encoder(z).mean(dim=1)                                      # mean pooling over tokens -> [N, d]


# ---------------------------------------------------------------- [S19] attention-based fusion of the scales
class AttentionScaleFusion(nn.Module):
    """Stack the scale vectors, score each with a shared MLP, softmax over scales, weighted sum."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, vecs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        s = torch.stack(vecs, dim=1)                                            # stack   [N, S, d]
        w = torch.softmax(self.mlp(s).squeeze(-1), dim=1)                       # MLP -> softmax -> weights [N, S]
        return (w.unsqueeze(-1) * s).sum(dim=1), w                              # weighted sum [N, d]


def mlp_head(width: int, hidden: tuple[int, int], out: int, dropout: float) -> nn.Sequential:
    """[S12/S13] Linear(width->h1), GELU, Dropout, Linear(h1->h2), GELU, Linear(h2->out)."""
    h1, h2 = hidden
    return nn.Sequential(nn.Linear(width, h1), nn.GELU(), nn.Dropout(dropout), nn.Linear(h1, h2), nn.GELU(), nn.Linear(h2, out))


# ---------------------------------------------------------------- the full model
class ProposalMultiScalePatchTST(nn.Module):
    def __init__(self, *, n_channels: int, lookback: int, patch_lengths, patch_strides, channel_mode: str, revin: bool,
                 encoder_norm: str,
                 d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float, fusion_hidden: int,
                 n_stocks: int, d_stock: int, n_sectors: int, d_sector: int, trunk_width: int, head_hidden, head_dropout: float,
                 horizons=(1, 5, 10), n_classes: int = 3):
        super().__init__()
        if channel_mode not in CHANNEL_MODES:
            raise ValueError(f"channel_mode must be one of {CHANNEL_MODES}")
        if len(patch_lengths) != len(patch_strides) or len(patch_lengths) < 2:
            raise ValueError("multi-scale needs at least 2 scales, each with a length and a stride")
        self.n_channels, self.lookback, self.channel_mode = n_channels, lookback, channel_mode
        self.horizons, self.n_classes = tuple(horizons), n_classes
        self.revin = RevIN(n_channels) if revin else None                       # optional per-window normalisation
        per_step = 1 if channel_mode == "independent" else n_channels
        self.branches = nn.ModuleList(ScaleBranch(lookback, p, s, per_step, d_model, n_heads, n_layers, d_ff, dropout, encoder_norm)
                                      for p, s in zip(patch_lengths, patch_strides))
        self.fusion = AttentionScaleFusion(d_model, fusion_hidden)
        # [S12] stock embedding: stock_id and sector_id (index 0 = unknown in both)
        self.stock_emb = nn.Embedding(n_stocks + 1, d_stock)
        self.sector_emb = nn.Embedding(n_sectors + 1, d_sector)
        enc_out = d_model * (n_channels if channel_mode == "independent" else 1)
        self.linear = nn.Linear(enc_out + d_stock + d_sector, trunk_width)    # the single "Linear Layer" on S12/S13
        h = len(self.horizons)
        self.price_head = mlp_head(trunk_width, head_hidden, h, head_dropout)               # [S12] -> 3 log returns
        self.trend_head = mlp_head(trunk_width, head_hidden, h * n_classes, head_dropout)   # [S13] -> 9 -> (3, 3)

    def forward(self, x: torch.Tensor, stock: torch.Tensor, sector: torch.Tensor) -> dict:
        B, L, C = x.shape                                                       # window ending at day t
        if L != self.lookback or C != self.n_channels:
            raise ValueError(f"expected [batch, {self.lookback}, {self.n_channels}], got {list(x.shape)}")
        if self.revin is not None:
            x = self.revin(x)
        u = x.permute(0, 2, 1).reshape(B * C, L) if self.channel_mode == "independent" else x
        fused, w = self.fusion([b(u) for b in self.branches])                  # [S11] -> [S19]
        enc = fused.reshape(B, -1)                                              # independent: concat C vectors
        z = self.linear(torch.cat([enc, self.stock_emb(stock), self.sector_emb(sector)], dim=-1))
        return {"trend_logits": self.trend_head(z).reshape(B, len(self.horizons), self.n_classes),
                "returns": self.price_head(z),
                "scale_weights": w.reshape(B, -1, len(self.branches))}         # [B, C or 1, S]


# ---------------------------------------------------------------- multi-task loss (proposal scope 4)
class FixedWeightLoss(nn.Module):
    """w_trend * CE(trend) + w_price * price loss (MSE or Huber on the train-standardised log return)."""

    def __init__(self, w_trend: float, w_price: float, price_loss: str, huber_delta: float | None):
        super().__init__()
        self.w_trend, self.w_price = w_trend, w_price
        self.ce = nn.CrossEntropyLoss()
        self.reg = nn.MSELoss() if price_loss == "mse" else nn.HuberLoss(delta=huber_delta)

    def forward(self, out: dict, trend: torch.Tensor, returns: torch.Tensor):
        l_cls = self.ce(out["trend_logits"].reshape(-1, out["trend_logits"].shape[-1]), trend.reshape(-1))
        l_reg = self.reg(out["returns"], returns)
        return self.w_trend * l_cls + self.w_price * l_reg, {"ce": l_cls.item(), "price": l_reg.item()}


class KendallLoss(UncertaintyWeightedLoss):
    """Learned uncertainty weights (Kendall, Gal & Cipolla 2018), the 2026-10-15 design decision; MSE or Huber."""

    def __init__(self, price_loss: str, huber_delta: float | None):
        super().__init__()
        if price_loss == "huber":
            self.mse = nn.HuberLoss(delta=huber_delta)

    def forward(self, out: dict, trend: torch.Tensor, returns: torch.Tensor):
        total, parts = super().forward(out, trend, returns)
        return total, {"ce": parts["ce"], "price": parts["mse"], "w_ce": parts["w_ce"], "w_price": parts["w_mse"]}


def make_loss(cfg_loss: dict) -> nn.Module:
    if cfg_loss["weighting"] == "kendall":
        return KendallLoss(cfg_loss["price_loss"], cfg_loss.get("huber_delta"))
    return FixedWeightLoss(cfg_loss["w_trend"], cfg_loss["w_price"], cfg_loss["price_loss"], cfg_loss.get("huber_delta"))
