"""E29 inputs: an Alpha158-style factor set ported to SET daily OHLCV, plus per-day cross-sectional ranks.

Port of the factor definitions in Qlib's Alpha158 handler (Yang et al. 2020, "Qlib: An AI-oriented Quantitative
Investment Platform", arXiv 2009.11189; qlib/contrib/data/handler.py). Fixed 2026-10-16 before any E29 result
(claude/batches-4-7-design-2026-10-16.md):
  * 9 candlestick (KBAR) factors, OPEN0 / HIGH0 / LOW0 (VWAP0 dropped: SET daily data has no VWAP), and 29 rolling
    operators over 5, 10, 20, 30 and 60 days = 9 + 3 + 145 = 157 factors.
  * Prices are the dividend- and split-adjusted OHLC of src/features.py, per series segment (no window crosses a
    ticker changeover). Volume is the raw traded volume, as in the base inputs. [Uncertainty Flag: Qlib adjusts
    volume for splits; here a split day can move the volume factors.]
  * Differences from Qlib, kept on purpose: IMAX / IMIN use the position of the extreme inside the window
    (1 = oldest day) divided by the window length; RANK is the share of window closes at or below today's close.
  * Cross-sectional ranks: for every factor, the percentile rank among stocks that are SET50 members on day t
    (point-in-time `member` flag). Non-member stock-days get NaN.
Every value at day t uses only data at or before day t (tests/test_new_features.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import features as F

WINDOWS = (5, 10, 20, 30, 60)
ROLLING_OPS = ["ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD", "RANK", "RSV", "IMAX", "IMIN",
               "IMXD", "CORR", "CORD", "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP",
               "VSUMN", "VSUMD"]
KBAR = ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
PRICE0 = ["OPEN0", "HIGH0", "LOW0"]
FACTORS = KBAR + PRICE0 + [f"{op}{d}" for d in WINDOWS for op in ROLLING_OPS]
RANKED = [f"xr_{f}" for f in FACTORS]
EPS = 1e-12


def _argpos(x: np.ndarray, d: int, fn) -> np.ndarray:
    """1-based position of fn (argmax / argmin) inside each trailing window of length d; NaN before d values."""
    out = np.full(len(x), np.nan)
    if len(x) >= d:
        w = sliding_window_view(x, d)
        pos = fn(np.where(np.isnan(w), -np.inf if fn is np.argmax else np.inf, w), axis=1) + 1.0
        pos[np.isnan(w).any(axis=1)] = np.nan
        out[d - 1:] = pos
    return out


def segment_factors(g: pd.DataFrame) -> pd.DataFrame:
    a = F.adjusted_ohlc(g)
    # rounded to 8 decimals so equal raw prices give exactly equal adjusted prices: ties (RANK, IMAX/IMIN, zero-move days in
    # CNTP / WVMA) then resolve the same way on every machine instead of by floating-point noise
    o, h, l, c = (a[k].astype(float).round(8) for k in ("open", "high", "low", "close"))
    v = g["volume"].astype(float)
    bad = (o <= 0) | (h <= 0) | (l <= 0) | (h < l)
    o, h, l = o.where(~bad), h.where(~bad), l.where(~bad)
    out = {}
    hl = h - l + EPS
    out["KMID"], out["KLEN"], out["KMID2"] = (c - o) / o, (h - l) / o, (c - o) / hl
    top, bot = np.maximum(o, c), np.minimum(o, c)
    out["KUP"], out["KUP2"] = (h - top) / o, (h - top) / hl
    out["KLOW"], out["KLOW2"] = (bot - l) / o, (bot - l) / hl
    out["KSFT"], out["KSFT2"] = (2 * c - h - l) / o, (2 * c - h - l) / hl
    out["OPEN0"], out["HIGH0"], out["LOW0"] = o / c, h / c, l / c
    t = pd.Series(np.arange(len(c), dtype=float), index=c.index)
    dc = c.diff()
    ret = c / c.shift(1) - 1
    lv = np.log(v + 1)
    lvr = np.log(v / v.shift(1).replace(0, np.nan) + 1)
    dv = v.diff()
    hv, lw = h.to_numpy(), l.to_numpy()
    for d in WINDOWS:
        r = lambda x: x.rolling(d, min_periods=d)   # noqa: E731
        out[f"ROC{d}"] = c.shift(d) / c
        out[f"MA{d}"] = r(c).mean() / c
        out[f"STD{d}"] = r(c).std() / c
        slope = r(c).cov(t) / r(t).var()
        out[f"BETA{d}"] = slope / c
        flat = r(c).std() <= 1e-10 * c                      # no price change in the window: correlations undefined
        out[f"RSQR{d}"] = (r(c).corr(t) ** 2).mask(flat)
        out[f"RESI{d}"] = (c - r(c).mean() - slope * (d - 1) / 2) / c
        mx, mn = r(h).max(), r(l).min()
        out[f"MAX{d}"], out[f"MIN{d}"] = mx / c, mn / c
        out[f"QTLU{d}"] = r(c).quantile(0.8) / c
        out[f"QTLD{d}"] = r(c).quantile(0.2) / c
        cv = c.to_numpy()
        rank = np.full(len(cv), np.nan)
        if len(cv) >= d:
            w = sliding_window_view(cv, d)
            rank[d - 1:] = np.where(np.isnan(w).any(axis=1), np.nan, (w <= w[:, -1:]).mean(axis=1))
        out[f"RANK{d}"] = rank
        out[f"RSV{d}"] = (c - mn) / (mx - mn + EPS)
        imax, imin = _argpos(hv, d, np.argmax), _argpos(lw, d, np.argmin)
        out[f"IMAX{d}"], out[f"IMIN{d}"], out[f"IMXD{d}"] = imax / d, imin / d, (imax - imin) / d
        out[f"CORR{d}"] = r(c).corr(lv).mask(flat | (r(lv).std() <= 1e-10))
        cr = c / c.shift(1)
        out[f"CORD{d}"] = r(cr).corr(lvr).mask((r(cr).std() <= 1e-10) | (r(lvr).std() <= 1e-10))
        up, dn = (dc > 0).astype(float).where(dc.notna()), (dc < 0).astype(float).where(dc.notna())
        out[f"CNTP{d}"], out[f"CNTN{d}"] = r(up).mean(), r(dn).mean()
        out[f"CNTD{d}"] = out[f"CNTP{d}"] - out[f"CNTN{d}"]
        sp, sn, sa = r(dc.clip(lower=0)).sum(), r((-dc).clip(lower=0)).sum(), r(dc.abs()).sum() + EPS
        out[f"SUMP{d}"], out[f"SUMN{d}"], out[f"SUMD{d}"] = sp / sa, sn / sa, (sp - sn) / sa
        out[f"VMA{d}"] = r(v).mean() / (v + EPS)
        out[f"VSTD{d}"] = r(v).std() / (v + EPS)
        wv = ret.abs() * v
        out[f"WVMA{d}"] = (r(wv).std() / (r(wv).mean() + EPS)).mask(r(wv).max() == 0, 0.0)   # all-zero window -> 0
        vp, vn, va = r(dv.clip(lower=0)).sum(), r((-dv).clip(lower=0)).sum(), r(dv.abs()).sum() + EPS
        out[f"VSUMP{d}"], out[f"VSUMN{d}"], out[f"VSUMD{d}"] = vp / va, vn / va, (vp - vn) / va
    res = pd.DataFrame(out, index=g.index)[FACTORS]
    return res.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def add_factors(df: pd.DataFrame, ranks: bool = True) -> pd.DataFrame:
    """df: daily frame from prices.load_prices (needs symbol, date, seg_key, open/high/low/close, volume, ret, member)."""
    fac = pd.concat([segment_factors(g) for _, g in df.groupby("seg_key", sort=False)]).reindex(df.index)
    out = df.join(fac)
    if ranks:
        mem = out["member"].astype(bool)
        rk = fac[mem].groupby(out.loc[mem, "date"]).rank(pct=True)
        rk.columns = RANKED
        out = out.join(rk.astype(np.float32))
    return out
