"""Chronological 70 / 15 / 15 split by trading date, with a purge gap so labels never cross a boundary.

A label at day t looks h trading days ahead, so the last `purge` trading days before each boundary are
dropped from the earlier block (López de Prado 2018, ch. 7, purging). purge = the longest horizon (10).
The test block is sealed until Phase 6: scripts report validation numbers only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHARES = (0.70, 0.15, 0.15)


def date_blocks(dates: pd.Series, purge: int = 10) -> dict:
    u = np.sort(pd.unique(dates))
    n = len(u)
    a, b = int(round(n * SHARES[0])), int(round(n * (SHARES[0] + SHARES[1])))
    return {"train": (u[0], u[a - 1 - purge]), "val": (u[a], u[b - 1 - purge]), "test": (u[b], u[-1]),
            "n_dates": n, "purge": purge}


def assign(df: pd.DataFrame, blocks: dict) -> pd.Series:
    s = pd.Series("purged", index=df.index)
    for name in ("train", "val", "test"):
        lo, hi = blocks[name]
        s[(df.date >= lo) & (df.date <= hi)] = name
    return s


class TrainOnlyScaler:
    """Standardiser that refuses to be fitted on anything but training rows (hard rule 3)."""
    def fit(self, X: pd.DataFrame, split: pd.Series):
        if set(split.loc[X.index].unique()) != {"train"}:
            raise ValueError("TrainOnlyScaler.fit: only training rows may be used")
        self.mu, self.sd = X.mean(), X.std().replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X - self.mu) / self.sd
