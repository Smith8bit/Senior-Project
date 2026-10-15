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


def walk_forward_blocks(dates: pd.Series, eval_start, eval_end, purge: int = 10) -> dict:
    """E05 walk-forward fold: train on every date before the evaluation window (minus a purge of `purge` trading
    days), evaluate on [eval_start, eval_end]. The evaluation window is returned under "val" so all scripts treat it
    as the scored block. It must end on or before the official validation end, so no label reaches into the sealed
    test block, which keeps its official dates and is never trained on or scored."""
    u = np.sort(pd.unique(dates))
    official = date_blocks(dates, purge)
    e = u[(u >= np.datetime64(pd.Timestamp(eval_start))) & (u <= np.datetime64(pd.Timestamp(eval_end)))]
    if len(e) == 0:
        raise ValueError(f"no trading dates in the evaluation window {eval_start} .. {eval_end}")
    if e[-1] > official["val"][1]:
        raise ValueError(f"evaluation window ends {pd.Timestamp(e[-1]).date()}, after the official validation end "
                         f"{pd.Timestamp(official['val'][1]).date()}: its labels would reach into the sealed test block")
    i0 = int(np.searchsorted(u, e[0]))
    if i0 - 1 - purge < 0:
        raise ValueError("evaluation window starts too early: no training dates left after the purge")
    return {"train": (u[0], u[i0 - 1 - purge]), "val": (e[0], e[-1]), "test": official["test"],
            "n_dates": len(u), "purge": purge, "fold": {"eval_start": str(pd.Timestamp(e[0]).date()),
                                                         "eval_end": str(pd.Timestamp(e[-1]).date())}}


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
