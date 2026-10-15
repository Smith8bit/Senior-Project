"""E03 inputs: volatility-scaled past returns and returns relative to the market and the sector on the same day.

Every value at day t uses only closes at or before t (checked by tests/test_no_lookahead.py). Leak rules (E03, 2026-10-16):
  1. Volatility scale is TRAILING: std of daily adjusted log returns over the past 60 days of the same series segment,
     never the training-period sigma used for labels (that one is measured on forward returns).
  2. Market and sector averages use only stocks that are SET50 members ON DAY t (point-in-time `member` flag),
     never the 70-ticker union, which was assembled in hindsight.
  3. Sector average leaves the stock itself out; it needs >= 3 members of that sector on day t, otherwise NaN
     (XGBoost treats NaN as missing; the deep model's scaler maps NaN to 0, i.e. "no information").
Sector labels: data/sectors.csv (one snapshot, already used by the main model's sector embedding).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VOL_WINDOW = 60
MIN_MARKET_MEMBERS = 10
MIN_SECTOR_MEMBERS = 3
LAGS = (1, 5, 10, 20)
Z_LAGS = (1, 5, 10)

E03_FEATURES = ([f"zr_{k}" for k in Z_LAGS] + [f"mrel_r_{k}" for k in LAGS] + [f"srel_r_{k}" for k in LAGS])


def add_xs_features(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """df: daily frame from prices.load_prices + features.add_features (needs symbol, date, seg_key, ret, member, r_k)."""
    out = df.copy()
    # 1) trailing volatility within each series segment; past h-day return in units of sqrt(h) * daily vol
    vol = out.groupby("seg_key", sort=False)["ret"].transform(lambda r: r.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std())
    for k in Z_LAGS:
        out[f"zr_{k}"] = out[f"r_{k}"] / (vol * np.sqrt(k)).replace(0, np.nan)

    sector = out["symbol"].map(pd.read_csv(root / "data/sectors.csv").set_index("symbol")["sector"])
    for k in LAGS:
        r = out[f"r_{k}"]
        m = out["member"].astype(bool) & r.notna()               # members on day t with a defined return
        v = r.where(m)
        # 2) market: mean over members on day t
        n_mkt = m.groupby(out["date"]).transform("sum")
        mkt = v.groupby(out["date"]).transform("sum") / n_mkt
        out[f"mrel_r_{k}"] = (r - mkt).where(n_mkt >= MIN_MARKET_MEMBERS)
        # 3) sector: leave-one-out mean over the other members of the stock's sector on day t
        keys = [out["date"], sector]
        s_sum = v.groupby(keys).transform("sum")
        s_n = m.groupby(keys).transform("sum")
        own, own_n = v.fillna(0.0), m.astype(int)
        loo = (s_sum - own) / (s_n - own_n).replace(0, np.nan)
        out[f"srel_r_{k}"] = (r - loo).where((s_n >= MIN_SECTOR_MEMBERS) & sector.notna())
    return out
