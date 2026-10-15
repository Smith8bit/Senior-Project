"""Daily prices for the SET50 union universe: adjusted log returns, series breaks, point-in-time membership.

Decisions (claude/pre-registration-amendment-2026-10-15-exam-primary.md, "Other decisions"):
  * source: Siamchart raw EOD OHLCV + Yahoo split / cash-dividend events (Phase 1.22);
  * adjustment: splits AND dividends, so an ex-dividend drop is not a "down" move;
  * no return spans a ticker changeover (Phase 1.10) or an unexplained corporate-action jump.

Daily total log return on day t (raw close C, split ratio s_t = new/old shares, dividend D_t per old share):
    r_t = log(C_t * s_t + D_t) - log(C_{t-1})
Yahoo states dividends per share AFTER any later split, so D_raw = D_yahoo * product of later split ratios.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

# Phase 1.10: first trading day of the NEW entity; returns never cross these days.
CHANGEOVERS = {"TRUE": "2023-03-03", "SCB": "2022-04-27", "GULF": "2025-04-03", "TIDLOR": "2025-05-15"}
# Large raw moves with no Yahoo event (README, 2026-09-25). Causes unverified, so treated as breaks
# rather than guessed adjustment factors. [Uncertainty Flag: confirm against SET corporate-action records.]
UNEXPLAINED_BREAKS = {"GULF": ["2020-04-16"], "BANPU": ["2021-08-16"], "TOP": ["2024-12-20"]}
# Yahoo stitches TMB's history under TTB.BK; TMB is a separate Siamchart symbol until 2021-05-11.
YAHOO_ALIAS_BEFORE = {"TTB": ("TMB", "2021-05-12")}
STUDY_START, STUDY_END = "2020-01-01", "2025-12-31"


def _ratio(text: str) -> float:
    a, b = text.split(":")
    return float(Fraction(a) / Fraction(b)) if "." not in text else float(a) / float(b)


def load_events(path: Path) -> pd.DataFrame:
    ev = pd.read_csv(path, dtype={"value": str})
    ev["date"] = pd.to_datetime(ev["date"].astype(str), format="%Y%m%d")
    for ysym, (old, until) in YAHOO_ALIAS_BEFORE.items():
        m = (ev.symbol == ysym) & (ev.date < pd.Timestamp(until))
        ev.loc[m, "symbol"] = old
    ev["split"] = np.where(ev.event == "split", ev["value"].map(lambda v: _ratio(v) if ":" in str(v) else np.nan), np.nan)
    ev["div_yahoo"] = np.where(ev.event == "div", pd.to_numeric(ev["value"], errors="coerce"), np.nan)
    # Yahoo dividends are split-adjusted for splits AFTER the ex-date: undo that to get raw THB per share.
    raw = []
    for _, e in ev.iterrows():
        if e.event != "div":
            raw.append(np.nan); continue
        later = ev[(ev.symbol == e.symbol) & (ev.event == "split") & (ev.date > e.date)]["split"]
        raw.append(e.div_yahoo * float(np.prod(later)) if len(later) else e.div_yahoo)
    ev["div_raw"] = raw
    return ev


def load_prices(root: Path) -> pd.DataFrame:
    """One row per (symbol, trading day): raw OHLCV, adjusted log return `ret`, `segment` id, `member` flag."""
    raw = pd.read_csv(root / "data/raw/prices/siamchart/set50_union_2019_2025.csv")
    raw["date"] = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d")
    raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)
    ev = load_events(root / "data/raw/prices/yahoo/yahoo_events_2019_2025.csv")
    split = ev.dropna(subset=["split"]).groupby(["symbol", "date"])["split"].prod()
    div = ev.dropna(subset=["div_raw"]).groupby(["symbol", "date"])["div_raw"].sum()
    key = pd.MultiIndex.from_frame(raw[["symbol", "date"]])
    raw["split"] = split.reindex(key).fillna(1.0).to_numpy()
    raw["div"] = div.reindex(key).fillna(0.0).to_numpy()

    prev = raw.groupby("symbol")["close"].shift(1)
    raw["ret_raw"] = np.log(raw["close"] / prev)
    raw["ret"] = np.log((raw["close"] * raw["split"] + raw["div"]) / prev)

    # segments: a new segment starts at each changeover / unexplained break; the first return of a segment is undefined
    brk = pd.Series(False, index=raw.index)
    for sym, d in CHANGEOVERS.items():
        brk |= (raw.symbol == sym) & (raw.date == pd.Timestamp(d))
    for sym, ds in UNEXPLAINED_BREAKS.items():
        for d in ds:
            brk |= (raw.symbol == sym) & (raw.date == pd.Timestamp(d))
    missing = [(s, d) for s, d in list(CHANGEOVERS.items()) + [(s, d) for s, ds in UNEXPLAINED_BREAKS.items() for d in ds]
               if not ((raw.symbol == s) & (raw.date == pd.Timestamp(d))).any()]
    if missing:
        raise ValueError(f"break dates not found in the price file: {missing}")
    raw["segment"] = brk.groupby(raw.symbol).cumsum().astype(int)
    first_of_segment = raw.groupby(["symbol", "segment"]).cumcount() == 0
    raw.loc[first_of_segment, ["ret", "ret_raw"]] = np.nan
    raw["seg_key"] = raw.symbol + "#" + raw.segment.astype(str)

    raw["member"] = membership_mask(root, raw)
    return raw


def membership_mask(root: Path, df: pd.DataFrame) -> np.ndarray:
    """True where the symbol is a SET50 member on that date (17 sub-periods, set50_membership.csv)."""
    m = pd.read_csv(root / "data/set50_membership.csv", parse_dates=["effective_from", "effective_to"])
    out = np.zeros(len(df), dtype=bool)
    for _, r in m.iterrows():
        out |= ((df.symbol == r.symbol) & (df.date >= r.effective_from) & (df.date <= r.effective_to)).to_numpy()
    return out


def dividend_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per symbol: dividend events applied in the study window. Zero means XD drops stay in the returns."""
    w = df[(df.date >= STUDY_START) & (df.date <= STUDY_END)]
    return w.groupby("symbol").agg(div_events=("div", lambda s: int((s > 0).sum())),
                                   member_days=("member", "sum")).reset_index()
