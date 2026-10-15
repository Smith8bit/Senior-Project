"""Targets: forward h-day log returns (regression) and up / flat / down classes (trend).

flat if |r_h| < k * sigma_h  (López de Prado 2018, ch. 3 form; status doc §11c).
  * sigma_h: per-stock std of h-day forward log returns, measured on TRAINING rows only.
    Stocks with too little training history get the median training sigma_h. [Uncertainty Flag:
    the fallback is a judgement call; it affects stocks that joined the index late.]
  * k: chosen on TRAINING rows only, per horizon, to make the three classes as balanced as possible
    (Siwat, 2026-10-15). Both sigma_h and k are required arguments of `classify` -- no defaults.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10)
CLASSES = ("down", "flat", "up")      # encoded 0, 1, 2
MIN_TRAIN_ROWS_FOR_SIGMA = 250


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """fwd_h at t = sum of adjusted daily log returns t+1..t+h inside the same segment, else NaN."""
    out = df.copy()
    for h in HORIZONS:
        out[f"fwd_{h}"] = out.groupby("seg_key", sort=False)["ret"].transform(
            lambda r: r[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1))
    return out


def fit_sigma(train: pd.DataFrame, h: int) -> tuple[dict, float]:
    g = train.groupby("symbol")[f"fwd_{h}"]
    sd, n = g.std(), g.count()
    fallback = float(sd[n >= MIN_TRAIN_ROWS_FOR_SIGMA].median())
    per = {s: float(v) for s, v in sd.items() if n[s] >= MIN_TRAIN_ROWS_FOR_SIGMA}
    return per, fallback


def sigma_for(symbols: pd.Series, per: dict, fallback: float) -> np.ndarray:
    return symbols.map(per).fillna(fallback).to_numpy()


def classify(r: np.ndarray, sigma: np.ndarray, k: float) -> np.ndarray:
    z = np.asarray(r) / np.asarray(sigma)
    return np.where(np.abs(z) < k, 1, np.where(z > 0, 2, 0))


def fit_k(train_r: np.ndarray, train_sigma: np.ndarray, grid=np.arange(0.005, 2.0, 0.001)) -> tuple[float, dict]:
    """k minimising the squared distance of the train class shares from (1/3, 1/3, 1/3)."""
    z = np.abs(np.asarray(train_r) / np.asarray(train_sigma))
    best, best_loss = None, np.inf
    for k in grid:
        y = classify(train_r, train_sigma, k)
        shares = np.bincount(y, minlength=3) / len(y)
        loss = float(((shares - 1 / 3) ** 2).sum())
        if loss < best_loss:
            best, best_loss = float(round(k, 3)), loss
    y = classify(train_r, train_sigma, best)
    return best, {c: round(float((y == i).mean()), 4) for i, c in enumerate(CLASSES)}
