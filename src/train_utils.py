"""Training helpers for the v2.xx runs (v2.02 onwards; Siwat's choices of 2026-10-25). Tested in tests/test_v2xx_training.py.

* EmaWeights      : exponential moving average of the model's weights, updated after every optimiser step
                    (ema = decay * ema + (1 - decay) * weights). Buffers (BatchNorm running statistics) are NOT
                    averaged; they are recomputed for the averaged weights by `recompute_bn` before every evaluation,
                    as the curve-noise log asked (claude/log-entry-2026-10-20-v2-01-curve-noise.md).
* recompute_bn    : reset every BatchNorm layer and re-estimate its running mean and variance with a cumulative average
                    (momentum None) over the given batches, in train mode, without gradients. Same procedure as
                    torch.optim.swa_utils.update_bn (dropout is active during the pass, as there).
* make_scheduler  : learning-rate schedule per optimiser step. constant, or cosine from training.lr to training.lr_min
                    over all steps of the run (no early stopping with cosine, Siwat 2026-10-25).
* holdout_diagnostics : per-horizon hold-out cross-entropy, predicted-class shares and the share of calls unchanged
                    from the previous epoch.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn


class EmaWeights:
    def __init__(self, model: nn.Module, decay: float):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.model = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)


def batchnorm_layers(model: nn.Module) -> list:
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def recompute_bn(model: nn.Module, batches) -> int:
    """Re-estimate BatchNorm running statistics over `batches` (an iterable of argument tuples for model(*args)).
    Returns the number of batches used (0 and nothing done if the model has no BatchNorm)."""
    bns = batchnorm_layers(model)
    if not bns:
        return 0
    momenta = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None                                          # cumulative average over the pass
    was_training = model.training
    model.train()
    n = 0
    for args in batches:
        model(*args)
        n += 1
    for bn, m in zip(bns, momenta):
        bn.momentum = m
    model.train(was_training)
    return n


def make_scheduler(opt: torch.optim.Optimizer, training: dict, steps_per_epoch: int, epochs: int):
    """None for a constant learning rate; otherwise a per-step cosine schedule reaching lr_min at the last step."""
    if training["lr_schedule"] == "constant":
        return None
    if training["lr_schedule"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_per_epoch * epochs, eta_min=training["lr_min"])
    raise ValueError(f"unknown lr_schedule {training['lr_schedule']!r}")


def steps_per_epoch(n_rows: int, batch_size: int) -> int:
    return math.ceil(n_rows / batch_size)


def holdout_diagnostics(P: np.ndarray, Y: np.ndarray, prev_calls: np.ndarray | None) -> dict:
    """P [N, H, 3] probabilities, Y [N, H] labels. Cross-entropy per horizon (natural log; ln 3 = 1.0986 is the
    uniform guess), predicted-class shares per horizon (down, flat, up), and the share of rows whose predicted class is
    the same as in the previous epoch (None at the first epoch)."""
    N, H, _ = P.shape
    p_true = np.take_along_axis(P, Y[..., None], axis=-1)[..., 0]
    ce = (-np.log(np.clip(p_true, 1e-12, None))).mean(axis=0)
    calls = P.argmax(-1)
    shares = [(np.bincount(calls[:, j], minlength=3) / N).tolist() for j in range(H)]
    agree = None if prev_calls is None else (calls == prev_calls).mean(axis=0).tolist()
    return {"ce": ce.tolist(), "ce_mean": float(ce.mean()), "call_shares": shares, "agreement": agree, "calls": calls}


def blank_windows(X: np.ndarray, rows: np.ndarray, channels: list, values) -> np.ndarray:
    """v2.04 news-robust training (Siwat, 2026-10-27): a copy of the window batch X [N, L, C] in which the windows
    `rows` get the no-news value on every real day for each channel in `channels` (the standardised no-news fill).
    Padding positions (before a segment starts: every channel exactly 0) stay 0, as sequences.windows leaves them."""
    out = X.copy()
    if len(rows) == 0:
        return out
    real = ~(X[rows] == 0).all(-1)                                   # [n, L]
    for c, v in zip(channels, values):
        sub = out[rows, :, c]
        sub[real] = v
        out[rows, :, c] = sub
    return out
