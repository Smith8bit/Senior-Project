"""E26 inputs: volatility and range measures from daily open, high, low and close.

Every value at day t uses only prices at or before t (checked by tests/test_new_features.py). Computed per series
segment on the same dividend- and split-adjusted prices as src/features.py, so no window crosses a ticker changeover.

Fixed 2026-10-16 (claude/batches-4-7-design-2026-10-16.md), before any E26 result:
  * range volatility over 20 days: Parkinson (1980), Garman & Klass (1980), Rogers & Satchell (1991);
  * realised volatility (std of daily log returns) over 5, 10 and 60 days (vol_20 is already a base input);
  * overnight gap log(open_t / close_t-1), intraday return log(close_t / open_t), daily range log(high_t / low_t).
  * "Close position in the day's range" was in the plan but is left out: I01 found the similar bb_pctb carries
    nothing for either model.
Bad bars (non-positive prices, or high below low) give NaN for that day; the deep model's scaler maps NaN to 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import features as F

VOL_FEATURES = ["park_vol_20", "gk_vol_20", "rs_vol_20", "vol_5", "vol_10", "vol_60", "gap_1", "intraday_1", "range_1"]
WINDOW = 20


def segment_vol_features(g: pd.DataFrame) -> pd.DataFrame:
    a = F.adjusted_ohlc(g)
    o, h, l, c = (a[k].astype(float) for k in ("open", "high", "low", "close"))
    bad = (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0) | (h < l) | o.isna() | h.isna() | l.isna()
    o, h, l = o.where(~bad), h.where(~bad), l.where(~bad)
    hl, co = np.log(h / l), np.log(c / o)
    hc, ho, lc, lo = np.log(h / c), np.log(h / o), np.log(l / c), np.log(l / o)
    out = pd.DataFrame(index=g.index)
    roll = lambda x: x.rolling(WINDOW, min_periods=WINDOW).mean()   # noqa: E731
    out["park_vol_20"] = np.sqrt(roll(hl ** 2) / (4 * np.log(2)))
    out["gk_vol_20"] = np.sqrt(roll(0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).clip(lower=0))
    out["rs_vol_20"] = np.sqrt(roll(hc * ho + lc * lo).clip(lower=0))
    for k in (5, 10, 60):
        out[f"vol_{k}"] = g["ret"].rolling(k, min_periods=k).std()
    out["gap_1"] = np.log(o / c.shift(1))
    out["intraday_1"] = co
    out["range_1"] = hl
    return out[VOL_FEATURES].replace([np.inf, -np.inf], np.nan)


def add_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = [segment_vol_features(g) for _, g in df.groupby("seg_key", sort=False)]
    return df.join(pd.concat(parts))
