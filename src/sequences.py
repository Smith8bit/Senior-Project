"""Look-back windows for sequence models, cut from the full daily frame without crossing a segment.

Window for sample row t = feature rows t-L+1 .. t of the same segment (ticker changeover / break).
Days before the segment start, and indicator warm-up NaNs, are filled with 0 AFTER train-only
standardisation, i.e. with the training mean (hard rule 3: the scaler sees training rows only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def standardise_full(full: pd.DataFrame, cols: list[str], train_index: pd.Index) -> np.ndarray:
    """Scale every daily row with mean/std of TRAINING sample rows only; NaN -> 0 (= train mean)."""
    mu = full.loc[train_index, cols].mean()
    sd = full.loc[train_index, cols].std().replace(0, 1.0)
    z = ((full[cols] - mu) / sd).to_numpy(dtype=np.float32)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def windows(full: pd.DataFrame, z: np.ndarray, rows: pd.Index, lookback: int) -> np.ndarray:
    """[len(rows), lookback, C] windows ending at each sample row; positions outside the row's segment are 0.
    `full` must have a RangeIndex sorted by symbol/date (as returned by prices.load_prices)."""
    if not full.index.equals(pd.RangeIndex(len(full))):
        raise ValueError("full frame must keep the RangeIndex from load_prices")
    end = rows.to_numpy()
    idx = end[:, None] + np.arange(-lookback + 1, 1)[None, :]          # [N, L]
    valid = idx >= 0
    idx_c = np.clip(idx, 0, len(full) - 1)
    seg = full["seg_key"].to_numpy()
    valid &= seg[idx_c] == seg[end][:, None]                           # never cross a segment
    out = z[idx_c]                                                     # [N, L, C]
    out[~valid] = 0.0
    return out
