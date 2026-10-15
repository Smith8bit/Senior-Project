"""Technical indicators and price features. Every value at day t uses only data at or before t.

Computed per series segment (no window crosses a ticker changeover or an unexplained break) on
dividend- and split-adjusted prices. Ratios to price keep features comparable across stocks, so one
pooled model can use them without global scaling (hard rule 3 is still respected where scaling is used).

Indicators (status doc §4): SMA, EMA, RSI, MACD, Bollinger %b, ADX. Standard parameters:
SMA 20/50, EMA 12/26, RSI 14 (Wilder), MACD 12/26/9, Bollinger 20/2, ADX 14 (Wilder).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["r_1", "r_5", "r_10", "r_20", "vol_20", "logvol_ratio_20",
            "sma20_gap", "sma50_gap", "ema12_gap", "ema26_gap", "rsi14",
            "macd", "macd_signal", "macd_hist", "bb_pctb", "adx14", "plus_di14", "minus_di14"]


def adjusted_ohlc(g: pd.DataFrame) -> pd.DataFrame:
    """Total-return price index within one segment, anchored at the segment's first raw close."""
    r = g["ret"].fillna(0.0).to_numpy()
    idx = g["close"].iloc[0] * np.exp(np.cumsum(r))
    f = idx / g["close"].to_numpy()
    return pd.DataFrame({"open": g["open"] * f, "high": g["high"] * f, "low": g["low"] * f, "close": idx,
                         "volume": g["volume"]}, index=g.index)


def _wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def segment_features(g: pd.DataFrame) -> pd.DataFrame:
    a = adjusted_ohlc(g)
    c, h, l = a["close"], a["high"], a["low"]
    lc = np.log(c)
    out = pd.DataFrame(index=g.index)
    for k in (1, 5, 10, 20):
        out[f"r_{k}"] = lc - lc.shift(k)
    out["vol_20"] = g["ret"].rolling(20, min_periods=20).std()
    lv = np.log1p(a["volume"].astype(float))
    out["logvol_ratio_20"] = lv - lv.rolling(20, min_periods=20).mean()
    out["sma20_gap"] = c / c.rolling(20, min_periods=20).mean() - 1
    out["sma50_gap"] = c / c.rolling(50, min_periods=50).mean() - 1
    e12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    e26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    out["ema12_gap"], out["ema26_gap"] = c / e12 - 1, c / e26 - 1
    d = c.diff()
    up, dn = _wilder(d.clip(lower=0), 14), _wilder((-d).clip(lower=0), 14)
    out["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    out.loc[(dn == 0) & up.notna(), "rsi14"] = 100.0
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd"], out["macd_signal"], out["macd_hist"] = macd / c, sig / c, (macd - sig) / c
    m20, s20 = c.rolling(20, min_periods=20).mean(), c.rolling(20, min_periods=20).std()
    out["bb_pctb"] = (c - (m20 - 2 * s20)) / (4 * s20).replace(0, np.nan)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    upm, dnm = h.diff(), -l.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=g.index)
    ndm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=g.index)
    atr = _wilder(tr, 14)
    pdi, ndi = 100 * _wilder(pdm, 14) / atr, 100 * _wilder(ndm, 14) / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    out["plus_di14"], out["minus_di14"], out["adx14"] = pdi, ndi, _wilder(dx, 14)
    return out[FEATURES]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = [segment_features(g) for _, g in df.groupby("seg_key", sort=False)]
    return df.join(pd.concat(parts))
