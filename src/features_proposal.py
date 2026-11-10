"""Inputs of the proposal-accurate model (v2.01): the four groups on presentation slide 9.

  Stock price  : Open, High, Low, Close, Volume, each "normalised (log return)"            -> 5 channels
  Technical    : SMA, EMA, RSI, MACD, Bollinger %b, ADX (proposal scope 2), "normalised"  -> one channel per
                 SMA period and per EMA period, plus RSI, MACD, %b, ADX
  Sentiment    : probability vector [pos, neg, neu] from the fine-tuned WangchanBERTa     -> 3 channels
  Stock identity: stock_id, sector_id (embeddings inside the model, not channels here)

Every number (indicator periods, how a log return of Open/High/Low is taken, the no-news fill) comes from the
run config; there are no defaults, so a missing choice fails loudly instead of being guessed.

v2.11 options (Siwat, 2026-11-10; departures from the proposal's input list, recorded as an ablation of scope 2):
  inputs.extra_technical ["plus_di", "minus_di"]: +DI and -DI of the ADX calculation (same period)  -> 2 channels
  inputs.market "set50_index_ohlcv": the official SET50 index, OHLC as log returns against the previous index
                 close and volume as the change of log(1 + V), like the stock's own price channels   -> 5 channels
                 (Siamchart EOD archive, extracted to data/raw/prices/siamchart/set50_index_2019_2025.csv;
                 2026 rows from a separate forward file, see scripts/step0_v2_03beta.py prepare)
Channel order: price, technical, extra technical, market, sentiment. Absent keys = the v2.01-v2.10 inputs exactly.

Causality: every value at day t uses data at or before t only, computed per series segment on dividend- and
split-adjusted prices (features.adjusted_ohlc), so no value crosses a ticker changeover or a break.
Scaling to "normalised" happens later in sequences.standardise_full, with training rows only (hard rule 3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import features as F

ROOT = Path(__file__).resolve().parents[1]

PRICE_CH = ["lr_open", "lr_high", "lr_low", "lr_close", "lr_volume"]
EXTRA_TECH = ("plus_di", "minus_di")                        # v2.11
MARKET_CH = ["set50_lr_open", "set50_lr_high", "set50_lr_low", "set50_lr_close", "set50_lr_volume"]   # v2.11
MARKETS = ("set50_index_ohlcv",)
SET50_INDEX_FILE = "data/raw/prices/siamchart/set50_index_2019_2025.csv"   # sha256 2363de30... (extracted 2026-11-10)
SENT_CH = ["p_pos", "p_neg", "p_neu"]                      # slide 9 order: [pos, neg, neu]

OHLC_BASES = ("own_previous", "previous_close")
NO_NEWS_FILLS = ("zeros", "neutral", "uniform")
TECH_KEYS = ("sma_periods", "ema_periods", "rsi_period", "macd_fast", "macd_slow",
             "bb_period", "bb_width_k", "adx_period")


def tech_channels(cfg: dict) -> list[str]:
    """Technical channel names for a config's "inputs" block: one per SMA and EMA period, then RSI, MACD, %b, ADX."""
    return ([f"sma_gap_{n}" for n in cfg["sma_periods"]] + [f"ema_gap_{n}" for n in cfg["ema_periods"]]
            + ["rsi", "macd", "bb_pctb", "adx"])


def extra_tech_channels(cfg: dict) -> list[str]:
    ex = list(cfg.get("extra_technical") or [])
    bad = [c for c in ex if c not in EXTRA_TECH]
    if bad or len(set(ex)) != len(ex):
        raise ValueError(f"inputs.extra_technical must be distinct names from {EXTRA_TECH} (got {ex})")
    return ex


def market_channels(cfg: dict) -> list[str]:
    m = cfg.get("market")
    if m is None:
        return []
    if m not in MARKETS:
        raise ValueError(f"inputs.market must be one of {MARKETS} (got {m!r})")
    return list(MARKET_CH)


def channels(cfg: dict) -> list[str]:
    return PRICE_CH + tech_channels(cfg) + extra_tech_channels(cfg) + market_channels(cfg) + SENT_CH


def groups(cfg: dict) -> dict:
    g = {"price": PRICE_CH, "technical": tech_channels(cfg)}
    if extra_tech_channels(cfg):
        g["technical_extra"] = extra_tech_channels(cfg)
    if market_channels(cfg):
        g["market"] = market_channels(cfg)
    return g | {"sentiment": SENT_CH}


def _wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def segment_inputs(g: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Price and technical channels for one segment. cfg = the config's "inputs" block."""
    a = F.adjusted_ohlc(g)
    o, h, l, c, v = a["open"], a["high"], a["low"], a["close"], a["volume"].astype(float)
    out = pd.DataFrame(index=g.index)
    # --- price: log returns (slide 9) ---
    prev_c = c.shift(1)
    if cfg["ohlc_return_base"] == "own_previous":          # ln(X_t / X_{t-1}) for X in O, H, L, C
        for name, s in (("lr_open", o), ("lr_high", h), ("lr_low", l)):
            out[name] = np.log(s / s.shift(1))
    elif cfg["ohlc_return_base"] == "previous_close":      # ln(X_t / C_{t-1}): keeps the day's shape
        for name, s in (("lr_open", o), ("lr_high", h), ("lr_low", l)):
            out[name] = np.log(s / prev_c)
    else:
        raise ValueError(f"inputs.ohlc_return_base must be one of {OHLC_BASES}")
    out["lr_close"] = np.log(c / prev_c)                    # same under both bases
    lv = np.log1p(v)                                        # volume can be 0: log(1 + V) before differencing
    out["lr_volume"] = lv - lv.shift(1)
    # --- technical (proposal scope 2); price-level indicators are taken relative to the close ---
    for n in cfg["sma_periods"]:                            # one channel per period
        out[f"sma_gap_{n}"] = c / c.rolling(n, min_periods=n).mean() - 1
    for n in cfg["ema_periods"]:
        out[f"ema_gap_{n}"] = c / c.ewm(span=n, adjust=False, min_periods=n).mean() - 1
    n = cfg["rsi_period"]
    d = c.diff()
    up, dn = _wilder(d.clip(lower=0), n), _wilder((-d).clip(lower=0), n)
    out["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    out.loc[(dn == 0) & up.notna(), "rsi"] = 100.0
    ef = c.ewm(span=cfg["macd_fast"], adjust=False, min_periods=cfg["macd_fast"]).mean()
    es = c.ewm(span=cfg["macd_slow"], adjust=False, min_periods=cfg["macd_slow"]).mean()
    out["macd"] = (ef - es) / c                             # MACD line / close (one channel, as in the proposal)
    m, s = c.rolling(cfg["bb_period"], min_periods=cfg["bb_period"]).mean(), c.rolling(cfg["bb_period"], min_periods=cfg["bb_period"]).std()
    k = cfg["bb_width_k"]
    out["bb_pctb"] = (c - (m - k * s)) / (2 * k * s).replace(0, np.nan)
    n = cfg["adx_period"]
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    upm, dnm = h.diff(), -l.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=g.index)
    ndm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=g.index)
    atr = _wilder(tr, n)
    pdi, ndi = 100 * _wilder(pdm, n) / atr, 100 * _wilder(ndm, n) / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    out["adx"] = _wilder(dx, n)
    out["plus_di"], out["minus_di"] = pdi, ndi                # v2.11 channels when inputs.extra_technical asks for them
    return out.replace([np.inf, -np.inf], np.nan)[PRICE_CH + tech_channels(cfg) + extra_tech_channels(cfg)]


def load_set50_index(forward_file: Path | str | None = None) -> pd.DataFrame:
    """The official SET50 index (date, open, high, low, close, volume), frozen 2019-2025 file plus, for the 2026 world,
    the forward file's rows after the frozen last day."""
    idx = pd.read_csv(ROOT / SET50_INDEX_FILE)
    if forward_file is not None:
        fw = pd.read_csv(forward_file)
        fw = fw[fw.date > idx.date.max()]
        idx = pd.concat([idx, fw[idx.columns]], ignore_index=True)
    idx["date"] = pd.to_datetime(idx.date.astype(str), format="%Y%m%d")
    assert idx.date.is_unique and idx.date.is_monotonic_increasing, "SET50 index dates are not unique and sorted"
    return idx


def market_inputs(idx: pd.DataFrame) -> pd.DataFrame:
    """SET50 index channels per date, built like the stock's price channels with ohlc_return_base previous_close:
    ln(X_t / C_{t-1}) for X in O, H, L, C, and the change of ln(1 + V). The index has no breaks or adjustments."""
    pc = idx["close"].shift(1)
    out = pd.DataFrame({"date": idx["date"]})
    for name, col in (("set50_lr_open", "open"), ("set50_lr_high", "high"), ("set50_lr_low", "low"), ("set50_lr_close", "close")):
        out[name] = np.log(idx[col] / pc)
    lv = np.log1p(idx["volume"].astype(float))
    out["set50_lr_volume"] = lv - lv.shift(1)
    return out.replace([np.inf, -np.inf], np.nan)


def add_price_tech(df: pd.DataFrame, cfg: dict, market_forward_file: Path | str | None = None) -> pd.DataFrame:
    """Add the price and technical channels (and, v2.11, the extra technical and SET50 index channels when the config
    asks for them) to the full daily frame (index kept). market_forward_file: the 2026 SET50 index rows (forward world)."""
    for key in TECH_KEYS + ("ohlc_return_base",):
        if cfg.get(key) is None:
            raise ValueError(f"inputs.{key} is not set")
    parts = [segment_inputs(g, cfg) for _, g in df.groupby("seg_key", sort=False)]
    new = pd.concat(parts)
    out = df.drop(columns=[c for c in new.columns if c in df.columns]).join(new)
    if market_channels(cfg):
        mk = market_inputs(load_set50_index(market_forward_file))
        vals = mk.set_index("date")[MARKET_CH].reindex(pd.DatetimeIndex(out["date"])).to_numpy()
        for j, c in enumerate(MARKET_CH):                      # a stock day with no index row stays missing (-> 0 later)
            out[c] = vals[:, j]
    return out


def sentiment_probs(scores: pd.DataFrame, rows: pd.DataFrame, no_news: str) -> pd.DataFrame:
    """Daily [p_pos, p_neg, p_neu] per (symbol, date): mean over the stock's articles assigned to that trading day.

    Articles are assigned by `trading_day` (published after the 16:30 close -> next trading day; hard rule 4). An
    article whose trading day is not a trading day of that stock (suspension) rolls forward to the stock's next row,
    as in scripts/news_beyond_price.py. No-news days get the configured fill:
      zeros   -> [0, 0, 0]  (a no-news day is distinguishable from any real probability vector)
      neutral -> [0, 0, 1]
      uniform -> [1/3, 1/3, 1/3]  (Siwat, 2026-10-19: exactly one third each)
    """
    if no_news not in NO_NEWS_FILLS:
        raise ValueError(f"inputs.sentiment_no_news must be one of {NO_NEWS_FILLS}")
    sc = scores[["ticker", "trading_day", "p_pos", "p_neg", "p_neu"]].copy()
    sc["trading_day"] = pd.to_datetime(sc["trading_day"])
    days = rows[["symbol", "date"]].drop_duplicates().sort_values("date")
    sc = pd.merge_asof(sc.sort_values("trading_day"), days.rename(columns={"date": "row_date"}),
                       left_on="trading_day", right_on="row_date", left_by="ticker", right_by="symbol",
                       direction="forward").dropna(subset=["row_date"])
    daily = sc.groupby(["ticker", "row_date"])[SENT_CH].mean().reset_index()
    daily = daily.rename(columns={"ticker": "symbol", "row_date": "date"})
    out = rows[["symbol", "date"]].merge(daily, on=["symbol", "date"], how="left")
    has = out["p_pos"].notna()
    fill = {"zeros": (0.0, 0.0, 0.0), "neutral": (0.0, 0.0, 1.0), "uniform": (1 / 3, 1 / 3, 1 / 3)}[no_news]
    for c, v in zip(SENT_CH, fill):
        out[c] = out[c].fillna(v)
    out["has_news"] = has.astype(float)                     # bookkeeping only; not a model input
    return out.set_index(["symbol", "date"])
