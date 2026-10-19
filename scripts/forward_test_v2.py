"""Forward (out-of-sample) test of main model v2 on 2026 data. The 2025 test block stays sealed.

Main model v2 (configs/main_model_v2.json, commit 6eaa76f) is re-trained exactly as it was selected:
  * same code, same training block (2020-01-02 .. 2024-02-15), same inner early-stopping hold-out, same seeds;
  * deep part = E07 (multiscale_sector, patches 2,1 + 20,10, RevIN off, classification loss only, stop rule "every",
    lr 1e-4, dropout 0.2, patience 10, max 30 epochs), each horizon at its own best inner hold-out epoch;
  * output rule: 1 d = 0.5 deep + 0.5 LightGBM E29 (343 inputs); 5 d = 0.5 deep + 0.5 XGBoost (29 inputs), then log-prob
    biases (flat -0.05, up +0.05); 10 d = deep alone. Seed s is paired with seed s.
Nothing is fitted on 2025 or 2026: scalers, label sigma and k come from the training block only, as before.

Why the training data cannot move:
  * splits.date_blocks() splits by SHARE of dates, so adding 2026 dates would shift the blocks. The official block dates
    are pinned here instead, and `prepare` checks that every training and validation row (inputs, sigma, labels) is
    identical to the frozen build.
  * 2026 inputs only ADD rows: prices up to 2025-12-30 come from the frozen file (sha256-checked); new corporate-action
    events are applied only to 2026 rows and to tickers with no training rows; new breaks likewise.
  * The validation predictions of the re-trained models are compared with the stored batch 4 / batch 6 prediction files.

What is scored: member stock-days from 2026-01-01 whose 1, 5 and 10-day outcomes are all known ("fwd"). Later member
stock-days are predicted but not scored ("live"). No 2025 row is predicted or scored. News: the sentiment file ends
2025-12-30, so every 2026 day is a no-news day; `deep` also scores validation with news blanked to show what that costs.

  python scripts/forward_test_v2.py prepare --fwd-root /content/fwd_root --out-dir OUT --price-source siamchart --archive ZIP
  python scripts/forward_test_v2.py deep    --fwd-root /content/fwd_root --out-dir OUT
  python scripts/forward_test_v2.py trees   --fwd-root /content/fwd_root --out-dir OUT
  python scripts/forward_test_v2.py score   --fwd-root /content/fwd_root --out-dir OUT --ref-deep D --ref-xgb X --ref-lgbm L
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import baselines_tabular as BT  # noqa: E402
import features as F  # noqa: E402
import labels as L  # noqa: E402
import metrics as M  # noqa: E402
import posthoc as PH  # noqa: E402
import prices as P  # noqa: E402
import splits as SP  # noqa: E402

SMOKE = os.environ.get("FWD_SMOKE") == "1"           # local plumbing test only: tiny training, never for results

# ----------------------------------------------------------------------------------------------- frozen facts
CODE_COMMIT = "6eaa76f"
FROZEN_SHA = {"data/raw/prices/siamchart/set50_union_2019_2025.csv": "9d7ce4c3e72a974979566dc089d5b8819af54dc24f62b71240f5b4716b556e6f",
              "data/raw/prices/yahoo/yahoo_events_2019_2025.csv": "d021a58df95fa487f1e49eb50617f6b3187b025b6377235c6180166b1ec84d53"}
OFFICIAL_BLOCKS = {"train": ("2020-01-02", "2024-02-15"), "val": ("2024-03-04", "2025-01-08"), "test": ("2025-01-23", "2025-12-16")}
FROZEN_LAST_DAY = 20251230                           # last trading day in the frozen price file
FWD_START = "2026-01-01"
FWD_STUDY_END = "2026-12-31"
SEEDS = (20261015, 20261016, 20261017)
HS = (1, 5, 10)
DEEP = "multiscale_sector_ph"
# v2 output rule (configs/main_model_v2.json): partner model and (bias_flat, bias_up) on the log-probabilities
RULE = {1: ("lgbm", (0.0, 0.0)), 5: ("xgb", (-0.05, 0.05)), 10: (None, (0.0, 0.0))}
PASS_BAR = {1: 0.393, 5: 0.400, 10: 0.407}
MARGIN = 0.02
# reference numbers (official validation window, 3-seed mean macro-F1; Drive Senior/results/batch7/final_summary.json)
REF_VAL = {"v2": {1: 0.3871, 5: 0.3963, 10: 0.3775}, "deep": {1: 0.3535, 5: 0.3861, 10: 0.3775}, "xgb": {1: 0.3668, 5: 0.3842, 10: 0.3680}}
REF_WF_MEAN = {1: 0.3666, 5: 0.3660, 10: 0.3701}     # v2, mean of the 4 walk-forward windows
REPRO_TOL = 0.005                                    # |validation macro-F1 re-trained - stored| allowed per horizon

# SET50 constituents, official SET lists (media.set.or.th): H1 2026 "SET50_100_H1_2026.pdf" (update 2025-12-15, for
# 1 Jan - 30 Jun 2026); H2 2026 "SET50-SET100-H2-2026.pdf" (update 2026-06-17, for 1 Jul - 31 Dec 2026). The 2026-08-03
# revision "SET50_SET100_H2_2026_revise.pdf" lists the same 50 symbols.
SET50_2026H1 = ("ADVANC AOT AWC BANPU BBL BDMS BEM BH BJC BTS CBG CCET CENTEL COM7 CPALL CPF CPN CRC DELTA EGCO GPSC GULF "
                "HMPRO IVL KBANK KKP KTB KTC LH MINT MTC OR OSP PTT PTTEP PTTGC RATCH SAWAD SCB SCC SCGP TCAP TIDLOR TISCO "
                "TLI TOP TRUE TTB TU WHA").split()
SET50_2026H2 = ("ADVANC AOT AWC BANPU BBL BCP BDMS BEM BH BJC CCET COM7 CPALL CPF CPN CRC DELTA EGCO GPSC GULF HMPRO IVL "
                "KBANK KKP KTB KTC LH MINT MRDIYT MTC OR OSP PTT PTTEP PTTGC RATCH SCB SCC SCGP TCAP TFG THAI TIDLOR TISCO "
                "TLI TOP TRUE TTB TU WHA").split()
MEMBERSHIP_2026 = [
    ("2026H1", "2026-01-01", "2026-06-30", "2025-12-15", "PDF (semi-annual review) SET50_100_H1_2026.pdf", SET50_2026H1),
    ("2026H2", "2026-07-01", "2026-12-31", "2026-06-17", "PDF (semi-annual review) SET50-SET100-H2-2026.pdf; same 50 in the 2026-08-03 revision", SET50_2026H2)]
# Sectors of the 2026 joiners with no row in data/sectors.csv, from the H2 2026 list (all three sector names already exist,
# so the sector ids of the trained embedding do not move).
NEW_SECTORS = {"MRDIYT": "Commerce", "TFG": "Food & Beverage", "THAI": "Transportation & Logistics"}
AUTO_GAP_DAYS = 30                                   # trading gap (calendar days) treated as a series break
AUTO_JUMP = float(np.log(1.35))                      # |daily log return| beyond the +-30% SET price limit -> break
REVIEW_JUMP = 0.18                                   # listed for review, kept


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ymd(d) -> int:
    return int(pd.Timestamp(d).strftime("%Y%m%d"))


def say(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------------- price sources
def read_siamchart(src: Path, symbols: set, start: int) -> pd.DataFrame:
    """Siamchart EOD files (<TICKER>,<DTYYYYMMDD>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>): a zip of daily CSVs, a folder of
    them, or one CSV. Keeps `symbols` on or after `start` (yyyymmdd)."""
    out, s0 = [], str(start)

    def take(text):
        for line in text.splitlines():
            if not line or line[0] == "<":
                continue
            p = line.strip().split(",")
            if len(p) >= 7 and p[0] in symbols and p[1] >= s0:
                out.append(p[:7])

    def wanted(name):
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
        return m is None or m.group(1) + m.group(2) + m.group(3) >= s0

    src = Path(src)
    if src.is_dir():
        for f in sorted(src.rglob("*.csv")):
            if wanted(f.name):
                take(f.read_text(encoding="utf-8", errors="replace"))
    elif src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            for n in sorted(z.namelist()):
                if n.lower().endswith(".csv") and wanted(n):
                    try:
                        take(z.read(n).decode("utf-8", "replace"))
                    except (zipfile.BadZipFile, OSError) as e:      # one member is corrupt at source (2018-09-14)
                        say(f"  skipped unreadable member {n}: {e}")
    else:
        take(src.read_text(encoding="utf-8", errors="replace"))
    df = pd.DataFrame(out, columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    df["date"] = df["date"].astype(int)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].astype(float).round().astype("int64")
    return df.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)


def yahoo_chart(sym: str, start: str, end: str, tries: int = 4) -> dict:
    """Yahoo chart API for <sym>.BK: daily quotes (split-adjusted by Yahoo) and dividend / split events."""
    import requests
    p1 = int(pd.Timestamp(start, tz="Asia/Bangkok").timestamp())
    p2 = int((pd.Timestamp(end, tz="Asia/Bangkok") + pd.Timedelta(days=1)).timestamp())
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}.BK"
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params={"period1": p1, "period2": p2, "interval": "1d", "events": "div|split",
                                          "includeAdjustedClose": "true"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if r.status_code == 429:
                raise RuntimeError("HTTP 429")
            r.raise_for_status()
            res = r.json()["chart"]["result"]
            if not res:
                raise RuntimeError("empty result")
            return res[0]
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"Yahoo {sym}.BK failed: {last}")


def _bkk_ymd(ts) -> int:
    return int(pd.to_datetime(int(ts), unit="s", utc=True).tz_convert("Asia/Bangkok").strftime("%Y%m%d"))


def _num(x) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else repr(x)


def yahoo_events(res: dict, sym: str) -> pd.DataFrame:
    ev = res.get("events") or {}
    rows = [(sym, _bkk_ymd(v["date"]), "div", repr(float(v["amount"]))) for v in (ev.get("dividends") or {}).values()]
    rows += [(sym, _bkk_ymd(v["date"]), "split", f"{_num(v['numerator'])}:{_num(v['denominator'])}")
             for v in (ev.get("splits") or {}).values() if float(v["numerator"]) > 0 and float(v["denominator"]) > 0]
    return pd.DataFrame(rows, columns=["symbol", "date", "event", "value"])


def yahoo_prices(res: dict, sym: str, events: pd.DataFrame) -> pd.DataFrame:
    """Raw (un-adjusted) OHLCV from Yahoo's split-adjusted quotes: multiply prices (divide volume) by every later split
    ratio. Zero-volume filler rows are dropped (Siamchart has none)."""
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"symbol": sym, "date": [_bkk_ymd(t) for t in res.get("timestamp", [])],
                       **{k: q.get(k) for k in ("open", "high", "low", "close", "volume")}})
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[df.volume > 0].copy()
    sp = events[events.event == "split"]
    f = np.ones(len(df))
    for _, e in sp.iterrows():
        f *= np.where(df.date.to_numpy() < e.date, P._ratio(e.value), 1.0)
    for c in ("open", "high", "low", "close"):
        df[c] = (df[c].astype(float) * f).round(4)
    df["volume"] = (df["volume"].astype(float) / f).round().astype("int64")
    return df.drop_duplicates(["symbol", "date"], keep="last")


# ----------------------------------------------------------------------------------------------- forward world
def pinned_blocks(dates=None, purge: int = 10) -> dict:
    return {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in OFFICIAL_BLOCKS.items()} | {"n_dates": None, "purge": purge}


@contextlib.contextmanager
def forward_world(fwd_root: Path):
    """Point the project's loaders at the forward data: extra events file, 2026 breaks, pinned blocks, study end 2026."""
    cfg = json.loads((fwd_root / "forward_config.json").read_text(encoding="utf-8"))
    saved = (P.load_events, P.UNEXPLAINED_BREAKS, P.STUDY_END, SP.date_blocks)
    orig_events = P.load_events
    extra = fwd_root / "data/raw/prices/yahoo/yahoo_events_forward.csv"

    def load_events_fwd(path):
        ev = orig_events(path)
        if Path(path).resolve().parent == extra.resolve().parent and extra.exists():
            new = orig_events(extra)                 # later-split correction inside the new file only
            ev = pd.concat([ev, new], ignore_index=True)
        return ev

    brk = {k: list(v) for k, v in P.UNEXPLAINED_BREAKS.items()}
    for s, ds in cfg["breaks"].items():
        brk[s] = sorted(set(brk.get(s, []) + ds))
    P.load_events, P.UNEXPLAINED_BREAKS, P.STUDY_END, SP.date_blocks = load_events_fwd, brk, FWD_STUDY_END, pinned_blocks
    try:
        yield cfg
    finally:
        P.load_events, P.UNEXPLAINED_BREAKS, P.STUDY_END, SP.date_blocks = saved


def detect_breaks(raw: pd.DataFrame, events: pd.DataFrame, new_syms: set, extra_breaks: dict) -> tuple[dict, list]:
    """Breaks for rows the frozen study never used: tickers with no training rows (all dates) and 2026 rows (all tickers).
    A trading gap > AUTO_GAP_DAYS or an unexplained |log return| > AUTO_JUMP starts a new segment."""
    df = raw.copy()
    df["d"] = pd.to_datetime(df.date.astype(str), format="%Y%m%d")
    ev = events.copy()
    ev["split"] = np.where(ev.event == "split", ev.value.map(lambda v: P._ratio(v) if ":" in str(v) else np.nan), np.nan)
    ev["divv"] = np.where(ev.event == "div", pd.to_numeric(ev.value, errors="coerce"), np.nan)
    sp = ev.dropna(subset=["split"]).groupby(["symbol", "date"]).split.prod()
    dv = ev.dropna(subset=["divv"]).groupby(["symbol", "date"]).divv.sum()
    key = pd.MultiIndex.from_frame(df[["symbol", "date"]])
    df["split"] = sp.reindex(key).fillna(1.0).to_numpy()
    df["div"] = dv.reindex(key).fillna(0.0).to_numpy()
    g = df.groupby("symbol")
    df["gap"] = (df.d - g.d.shift(1)).dt.days
    df["r"] = np.log((df["close"] * df["split"] + df["div"]) / g["close"].shift(1))
    scope = df.symbol.isin(new_syms) | (df.date > FROZEN_LAST_DAY)
    brk, review = {}, []
    for _, row in df[scope & ((df.gap > AUTO_GAP_DAYS) | (df.r.abs() > AUTO_JUMP))].iterrows():
        brk.setdefault(row.symbol, []).append(str(row.d.date()))
    for _, row in df[scope & (df.r.abs() > REVIEW_JUMP) & (df.r.abs() <= AUTO_JUMP) & (df.gap <= AUTO_GAP_DAYS)].iterrows():
        review.append({"symbol": row.symbol, "date": str(row.d.date()), "log_return": round(float(row.r), 4)})
    for s, ds in (extra_breaks or {}).items():
        brk[s] = sorted(set(brk.get(s, []) + list(ds)))
    return {s: sorted(set(v)) for s, v in brk.items()}, review


def prepare(args):
    out_dir, fwd_root = Path(args.out_dir), Path(args.fwd_root)
    inp = out_dir / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    for rel, h in FROZEN_SHA.items():
        got = sha256(ROOT / rel)
        assert got == h, f"stop: {rel} differs from the frozen copy ({got[:12]} vs {h[:12]})"
    frozen = pd.read_csv(ROOT / "data/raw/prices/siamchart/set50_union_2019_2025.csv")
    assert frozen.date.max() == FROZEN_LAST_DAY
    old_syms = set(frozen.symbol)
    members26 = set(SET50_2026H1) | set(SET50_2026H2)
    assert len(set(SET50_2026H1)) == 50 and len(set(SET50_2026H2)) == 50
    new_syms = members26 - old_syms
    want = old_syms | members26
    say(f"2026 members: {len(members26)} symbols; with no frozen price history: {sorted(new_syms)}")

    fwd_px, fwd_ev = inp / "forward_prices.csv", inp / "forward_events.csv"
    manifest = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code_commit_expected": CODE_COMMIT,
                "price_source": args.price_source, "frozen_sha256": FROZEN_SHA}
    if fwd_px.exists() and fwd_ev.exists() and not args.refresh:
        say(f"reusing saved forward inputs in {inp} (pass --refresh to fetch again)")
        px, ev_new = pd.read_csv(fwd_px), pd.read_csv(fwd_ev, dtype={"value": str})
        manifest.update(json.loads((inp / "inputs_meta.json").read_text(encoding="utf-8")))
    else:
        meta = {}
        # --- events (Yahoo, as for the frozen file), unless a file is given
        if args.events_csv:
            ev_all = pd.read_csv(args.events_csv, dtype={"value": str})
            meta["events_source"] = f"file {Path(args.events_csv).name} sha256 {sha256(args.events_csv)[:16]}"
            yres = {}
        else:
            yres, ev_parts, failed = {}, [], []
            for s in sorted(want):
                try:
                    yres[s] = yahoo_chart(s, "2019-01-01" if s in new_syms else "2025-10-01",
                                          pd.Timestamp.now(tz="Asia/Bangkok").strftime("%Y-%m-%d"))
                    ev_parts.append(yahoo_events(yres[s], s))
                except Exception as e:               # noqa: BLE001
                    failed.append(s)
                    say(f"  events: {s} failed ({e})")
            ev_all = pd.concat(ev_parts, ignore_index=True) if ev_parts else pd.DataFrame(columns=["symbol", "date", "event", "value"])
            meta["events_source"] = "Yahoo chart API (events=div|split), fetched " + manifest["computed_at"]
            meta["events_failed"] = failed
            miss = sorted(set(failed) & members26)
            if miss and not args.allow_missing_events:
                raise SystemExit(f"stop: no Yahoo events for 2026 members {miss}; their ex-dividend drops would count as "
                                 "'down' moves. Re-run, give --events-csv, or pass --allow-missing-events to accept it.")
        ev_all["date"] = ev_all["date"].astype(int)
        ev_new = ev_all[(ev_all.symbol.isin(new_syms)) | ((ev_all.symbol.isin(want)) & (ev_all.date > 20251231))].copy()
        # --- prices
        if args.price_source == "siamchart":
            if not args.archive or not Path(args.archive).exists():
                raise SystemExit(f"stop: fresh Siamchart archive not found at {args.archive}")
            meta["archive"] = {"name": Path(args.archive).name, "sha256": sha256(args.archive) if Path(args.archive).is_file() else None}
            src = read_siamchart(Path(args.archive), want, 20190101)
            px = src[(src.symbol.isin(new_syms)) | (src.date > FROZEN_LAST_DAY)].copy()
            # overlap: the fresh archive should repeat the frozen 2019-2025 rows exactly
            ov = src[src.symbol.isin(old_syms) & (src.date <= FROZEN_LAST_DAY)].merge(frozen, on=["symbol", "date"], how="outer",
                                                                                      suffixes=("", "_f"), indicator=True)
            both = ov[ov._merge == "both"]
            diff = ~np.isclose(both[["open", "high", "low", "close", "volume"]].to_numpy(float),
                               both[["open_f", "high_f", "low_f", "close_f", "volume_f"]].to_numpy(float)).all(1)
            meta["overlap_with_frozen"] = {"rows_both": int(len(both)), "rows_differ": int(diff.sum()),
                                           "only_frozen": int((ov._merge == "right_only").sum()), "only_archive": int((ov._merge == "left_only").sum())}
            say(f"overlap with the frozen file: {meta['overlap_with_frozen']}")
            if len(both) and diff.mean() > 0.001:
                say("  WARNING: the fresh archive revises > 0.1% of the frozen rows; the frozen rows are kept for <= 2025-12-30.")
        else:                                        # Yahoo fallback: same events, prices un-adjusted for splits
            parts, overlap = [], []
            for s, res in yres.items():
                y = yahoo_prices(res, s, ev_all[ev_all.symbol == s])
                parts.append(y[(y.symbol.isin(new_syms)) | (y.date > FROZEN_LAST_DAY)])
                if s in old_syms:
                    o = y[(y.date >= 20251001) & (y.date <= FROZEN_LAST_DAY)].merge(frozen, on=["symbol", "date"], suffixes=("", "_f"))
                    if len(o):
                        overlap.append({"symbol": s, "days": len(o), "close_within_0.5pct": round(float((abs(o.close / o.close_f - 1) < 0.005).mean()), 3),
                                        "volume_ratio_median": round(float((o.volume / o.volume_f.replace(0, np.nan)).median()), 3)})
            px = pd.concat(parts, ignore_index=True)
            ovd = pd.DataFrame(overlap)
            meta["yahoo_overlap_oct_dec_2025"] = overlap
            bad = ovd[ovd["close_within_0.5pct"] < 0.95].symbol.tolist() if len(ovd) else []
            say(f"Yahoo vs frozen Siamchart, Oct-Dec 2025: median close match {ovd['close_within_0.5pct'].median() if len(ovd) else None}, "
                f"median volume ratio {ovd['volume_ratio_median'].median() if len(ovd) else None}; tickers below 95% close match: {bad}")
            meta["yahoo_tickers_close_mismatch"] = bad
        px = px[px.date >= 20190101].sort_values(["symbol", "date"]).reset_index(drop=True)
        px.to_csv(fwd_px, index=False)
        ev_new.sort_values(["symbol", "date", "event"]).to_csv(fwd_ev, index=False)
        meta["forward_prices_sha256"], meta["forward_events_sha256"] = sha256(fwd_px), sha256(fwd_ev)
        (inp / "inputs_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        manifest.update(meta)

    px26 = px[px.date > FROZEN_LAST_DAY]
    if px26.empty:
        raise SystemExit("stop: the price source has no rows after 2025-12-30")
    last = pd.to_datetime(str(px26.date.max()), format="%Y%m%d")
    say(f"forward price rows: {len(px26)} after 2025-12-30, last day {last.date()}, {px26.date.nunique()} trading days")
    cover = px26[px26.symbol.isin(members26)].groupby("symbol").size()
    manifest["coverage_2026"] = {"trading_days": int(px26.date.nunique()), "last_day": str(last.date()),
                                 "members_without_rows": sorted(members26 - set(cover.index)),
                                 "members_rows_min": int(cover.min()), "members_rows_median": float(cover.median())}
    if manifest["coverage_2026"]["members_without_rows"]:
        say(f"  WARNING: 2026 members with no 2026 prices: {manifest['coverage_2026']['members_without_rows']}")

    # --- forward root: same file names the loaders expect
    if fwd_root.exists():
        shutil.rmtree(fwd_root)
    (fwd_root / "data/raw/prices/siamchart").mkdir(parents=True)
    (fwd_root / "data/raw/prices/yahoo").mkdir(parents=True)
    raw = pd.concat([frozen, px], ignore_index=True)
    assert not raw.duplicated(["symbol", "date"]).any()
    raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)
    raw.to_csv(fwd_root / "data/raw/prices/siamchart/set50_union_2019_2025.csv", index=False)
    shutil.copy(ROOT / "data/raw/prices/yahoo/yahoo_events_2019_2025.csv", fwd_root / "data/raw/prices/yahoo/yahoo_events_2019_2025.csv")
    ev_new.to_csv(fwd_root / "data/raw/prices/yahoo/yahoo_events_forward.csv", index=False)
    mem = pd.read_csv(ROOT / "data/set50_membership.csv")
    add = [{"symbol": s, "period": p, "effective_from": a, "effective_to": b, "published": pub, "provenance": prov}
           for p, a, b, pub, prov, syms in MEMBERSHIP_2026 for s in syms]
    pd.concat([mem, pd.DataFrame(add)], ignore_index=True).to_csv(fwd_root / "data/set50_membership.csv", index=False)
    sec = pd.read_csv(ROOT / "data/sectors.csv")
    assert set(NEW_SECTORS.values()) <= set(sec.sector), "a new sector name would shift the trained sector ids"
    miss = sorted(members26 - set(sec.symbol) - set(NEW_SECTORS))
    assert not miss, f"2026 members without a sector: {miss}"
    newsec = pd.DataFrame([{"symbol": s, "sector": v, "source": "SET, SET50/SET100 constituents H2 2026 (media.set.or.th/set/Documents/2026/Jun/SET50-SET100-H2-2026.pdf)"}
                           for s, v in NEW_SECTORS.items() if s not in set(sec.symbol)])
    pd.concat([sec, newsec], ignore_index=True).to_csv(fwd_root / "data/sectors.csv", index=False)

    events_all = pd.concat([pd.read_csv(ROOT / "data/raw/prices/yahoo/yahoo_events_2019_2025.csv", dtype={"value": str}), ev_new], ignore_index=True)
    breaks, review = detect_breaks(raw, events_all, new_syms, json.loads(args.extra_breaks) if args.extra_breaks else {})
    cfg = {"breaks": breaks, "review_moves": review, "new_symbols": sorted(new_syms), "last_price_day": str(last.date())}
    (fwd_root / "forward_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    say(f"breaks added (new tickers, or 2026 rows): {breaks}")
    if review:
        say(f"large moves kept (|log r| {REVIEW_JUMP}-{AUTO_JUMP:.3f}), check against SET corporate actions: {review}")
    s26 = ev_new[ev_new.date > 20251231]
    manifest["events_2026"] = {"dividends": int((s26.event == "div").sum()), "splits": s26[s26.event == "split"].to_dict("records")}
    late_splits = s26[(s26.event == "split") & (s26.symbol.isin(old_syms)) & (s26.date < 20260925)]
    if len(late_splits):
        say(f"  note: 2026 splits fetched before the frozen events file was made (2026-09-25): {late_splits.to_dict('records')}. "
            "Yahoo restates older dividends for later splits, so a few frozen ex-dividend adjustments may be slightly off. "
            "Training data is left exactly as frozen.")

    # --- identity check: training and validation rows must not move
    say("building the frozen dataset and the forward dataset to compare them ...")
    d0, i0 = BT.build_dataset(root=ROOT, extra="e03")
    with forward_world(fwd_root):
        d1, i1 = BT.build_dataset(root=fwd_root, extra="e03")
    for k, (a, b) in OFFICIAL_BLOCKS.items():
        assert i0["blocks"][k] == [a, b], f"stop: frozen {k} block {i0['blocks'][k]} != pinned {a}..{b}"
    cols = F.FEATURES + BT.EXTRA["e03"] + [f"{p}_{h}" for h in HS for p in ("fwd", "sigma", "y", "r")]
    rep = {}
    for part in ("train", "val"):
        a = d0[d0.split == part].set_index(["symbol", "date"])[cols].sort_index()
        b = d1[d1.split == part].set_index(["symbol", "date"])[cols].sort_index()
        assert a.index.equals(b.index), f"stop: {part} rows differ ({len(a)} frozen vs {len(b)} forward)"
        dif = float(np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float))))
        same_nan = bool((a.isna().to_numpy() == b.isna().to_numpy()).all())
        assert same_nan and dif <= 1e-12, f"stop: {part} values differ (max {dif}, NaN pattern same {same_nan})"
        rep[part] = {"rows": int(len(a)), "max_abs_diff": dif}
    for h in HS:
        assert i0[f"h{h}"]["k"] == i1[f"h{h}"]["k"], f"stop: k changed at h={h}"
    manifest["identity_check"] = rep | {"k": {f"h{h}": i1[f"h{h}"]["k"] for h in HS}}
    fwd = d1[d1.date >= pd.Timestamp(FWD_START)]
    manifest["forward_window"] = {"first": str(fwd.date.min().date()), "last": str(fwd.date.max().date()), "rows": int(len(fwd)),
                                  "dates": int(fwd.date.nunique()), "symbols": int(fwd.symbol.nunique()),
                                  "members_per_day_median": float(fwd.groupby("date").size().median()),
                                  "class_shares": {f"h{h}": {c: round(float((fwd[f'y_{h}'] == i).mean()), 4) for i, c in enumerate(L.CLASSES)} for h in HS},
                                  "train_class_shares": {f"h{h}": i1[f"h{h}"]["train_class_shares"] for h in HS}}
    manifest["breaks"], manifest["review_moves"] = breaks, review
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    say(f"identity check passed: train {rep['train']['rows']} rows, val {rep['val']['rows']} rows unchanged; k = {manifest['identity_check']['k']}")
    say(f"forward window (all 3 outcomes known): {manifest['forward_window']['first']} .. {manifest['forward_window']['last']}, "
        f"{manifest['forward_window']['rows']} stock-days on {manifest['forward_window']['dates']} dates")


# ----------------------------------------------------------------------------------------------- shared dataset
def load_forward(fwd_root: Path, factors: bool = False):
    """Forward dataset: rows tagged train / val / fwd / live. 2025 rows are dropped before anything is predicted."""
    with forward_world(fwd_root):
        data, info, full = BT.build_dataset(root=fwd_root, return_full=True, extra="e03")
        fac = None
        if factors:
            import alpha158 as A
            px = P.load_prices(fwd_root)
            assert px[["symbol", "date"]].equals(full[["symbol", "date"]]), "factor frame not aligned with the daily frame"
            fac = A.add_factors(px)[A.FACTORS + A.RANKED]
    start = pd.Timestamp(FWD_START)
    data.loc[data.date >= start, "split"] = "fwd"
    data = data[data.split.isin(["train", "val", "fwd"])].copy()           # test (2025) and purged rows leave here
    live_mask = full.member & (full.date >= start) & full[F.FEATURES].notna().all(axis=1) & ~full.index.isin(data.index)
    live = full[live_mask].copy()
    live["split"] = "live"
    for h in HS:
        live[f"y_{h}"] = -1
    rows = pd.concat([data, live[[c for c in data.columns if c in live.columns]]])
    assert not ((rows.date >= pd.Timestamp(OFFICIAL_BLOCKS["test"][0])) & (rows.date < start)).any(), "a 2025 row would be used"
    return rows, info, full, fac


def key_frame(d: pd.DataFrame) -> pd.DataFrame:
    k = d[["symbol", "date", "split"]].copy()
    k["date"] = k.date.dt.strftime("%Y-%m-%d")
    return k.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- deep part
def deep(args):
    import torch
    import baselines_deep as BD
    import news_beyond_price as NB
    import sequences as SQ
    out_dir, fwd_root = Path(args.out_dir), Path(args.fwd_root)
    todo = [s for s in SEEDS if not (out_dir / f"deep_s{s}.csv.gz").exists()]
    if not todo:
        say("deep: all seeds done"); return
    BD.OPT.update(revin=False, cls_only=True, label_smoothing=0.0, stop_rule="every", revin_stats=False, patches=((2, 1), (20, 10)),
                  d_channel=64, trunk_width=None, chan_mask=0.0, context_min=0, recent_weight=1.0, recent_from="2022-01-01")
    BD.SET.update(max_epochs=30, lr=1e-4, dropout=0.2, patience=10)
    BD.LOOKBACK = 60
    if SMOKE:                                         # CPU plumbing test: one epoch, small prediction batches
        import functools
        BD.SET.update(max_epochs=1, patience=1, batch_size=32)
        BD.predict = functools.partial(BD.predict, bs=48)
    rows, info, full, _ = load_forward(fwd_root)
    channels = list(F.FEATURES) + BT.EXTRA["e03"]
    nf = NB.news_features(pd.read_csv(ROOT / "data/derived/sentiment_scores.csv"), full[["symbol", "date"]])
    key = pd.MultiIndex.from_frame(full[["symbol", "date"]])
    for c in ("sent_mean", "sent_n", "has_news"):
        full[c] = nf[c].reindex(key).to_numpy()
    channels += ["sent_mean", "sent_n", "has_news"]
    tr = rows[rows.split == "train"]
    tdates = np.sort(tr.date.unique()); cut = int(len(tdates) * (1 - BD.SET["inner_es_share"]))
    fit_end, es_start = tdates[cut - 1 - BD.SET["purge"]], tdates[cut]
    fit, es = tr[tr.date <= fit_end], tr[tr.date >= es_start]
    parts = {"fit": fit, "es": es, "val": rows[rows.split == "val"], "fwd": rows[rows.split == "fwd"], "live": rows[rows.split == "live"]}
    if SMOKE:
        parts["fit"] = fit.iloc[:: max(1, len(fit) // 1500)]
    z = SQ.standardise_full(full, channels, tr.index)
    X = {k: SQ.windows(full, z, d.index, BD.LOOKBACK) for k, d in parts.items()}
    # validation with news blanked from the first validation day on (what every 2026 window looks like)
    blank = full.copy()
    vmask = blank.date >= pd.Timestamp(OFFICIAL_BLOCKS["val"][0])
    blank.loc[vmask, ["sent_mean", "sent_n", "has_news"]] = 0.0
    X["val_nonews"] = SQ.windows(blank, SQ.standardise_full(blank, channels, tr.index), parts["val"].index, BD.LOOKBACK)
    news_share = {k: round(float((full.loc[d.index, "has_news"] > 0).mean()), 4) for k, d in parts.items() if len(d)}
    Y = {k: parts[k][[f"y_{h}" for h in HS]].to_numpy(np.int64) for k in ("fit", "es")}
    r_sd = tr[[f"fwd_{h}" for h in HS]].std().to_numpy()
    Rfit = (parts["fit"][[f"fwd_{h}" for h in HS]].to_numpy() / r_sd).astype(np.float32)
    sectors = pd.read_csv(fwd_root / "data/sectors.csv")
    sec_id = {s: i + 1 for i, s in enumerate(sorted(sectors.sector.unique()))}
    frozen_ids = {s: i + 1 for i, s in enumerate(sorted(pd.read_csv(ROOT / "data/sectors.csv").sector.unique()))}
    assert sec_id == frozen_ids, "sector ids moved"
    sym2g = dict(zip(sectors.symbol, sectors.sector.map(sec_id)))
    G = {k: d.symbol.map(sym2g).fillna(0).to_numpy(np.int64) for k, d in parts.items()}
    G["val_nonews"] = G["val"]
    say(f"rows: fit {len(parts['fit'])} / inner-ES {len(es)} / val {len(parts['val'])} / fwd {len(parts['fwd'])} / live {len(parts['live'])}; "
        f"window 60 x {len(channels)}; device {BD.DEVICE}; days with news: {news_share}")
    pred_parts = ("val", "val_nonews", "fwd", "live")
    for seed in todo:
        t0 = time.time()
        model, meta, state_h = BD.train_one("multiscale_sector", seed, X["fit"], Y["fit"], Rfit, X["es"], Y["es"], G["fit"], G["es"], len(sec_id), None)
        frames = []
        for part in pred_parts:
            src = "val" if part == "val_nonews" else part
            k = key_frame(parts[src]); k["split"] = part
            for j, h in enumerate(HS):
                model.load_state_dict(state_h[j])
                pj = BD.predict(model, X[part], G[part])[0][:, j] if len(k) else np.zeros((0, 3))
                for c, cls in enumerate(L.CLASSES):
                    k[f"{DEEP}_s{seed}_p{h}_{cls}"] = pj[:, c].round(6)
            frames.append(k)
        out = pd.concat(frames, ignore_index=True)
        n_params = sum(p.numel() for p in model.parameters())
        meta_out = {"seed": seed, "seconds": round(time.time() - t0, 1), "params": n_params,
                    **{k: meta[k] for k in ("best_epoch", "epochs_run", "best_epoch_per_horizon", "best_inner_es_per_horizon")},
                    "history": meta["history"], "torch": torch.__version__, "device": str(BD.DEVICE),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "inner_es": {"fit_end": str(pd.Timestamp(fit_end).date()), "es_start": str(pd.Timestamp(es_start).date())},
                    "news_day_share": news_share, "smoke": SMOKE}
        (out_dir / f"deep_s{seed}_meta.json").write_text(json.dumps(meta_out, indent=2, default=str), encoding="utf-8")
        out.to_csv(out_dir / f"deep_s{seed}.csv.gz", index=False)       # written last: its presence marks the seed done
        say(f"deep seed {seed}: {n_params} weights, {meta['epochs_run']} epochs, best per horizon {meta['best_epoch_per_horizon']} "
            f"({time.time() - t0:.0f}s)")


# ----------------------------------------------------------------------------------------------- tree partners
def trees(args):
    import factors_tabular as FT
    import alpha158 as A
    from xgboost import XGBClassifier
    out_dir, fwd_root = Path(args.out_dir), Path(args.fwd_root)
    if (out_dir / "trees.csv.gz").exists():
        say("trees: done"); return
    t0 = time.time()
    rows, info, full, fac = load_forward(fwd_root, factors=True)
    say(f"dataset and {fac.shape[1]} factor columns in {time.time() - t0:.0f}s")
    live = rows.split == "live"
    for c in BT.EXTRA["e03"]:                         # live rows came from the daily frame; make sure E03 inputs are there
        rows.loc[live, c] = full.loc[rows.index[live], c].to_numpy()
    rows = rows.join(fac)
    tr = rows[rows.split == "train"]
    pred = rows[rows.split.isin(["val", "fwd", "live"])]
    out = key_frame(pred)
    xgb_feats = F.FEATURES + BT.EXTRA["e03"]
    big = xgb_feats + A.FACTORS + A.RANKED
    xgb_params = dict(BT.XGB) | ({"n_estimators": 20} if SMOKE else {})
    lgbm_params = dict(FT.LGBM) | ({"n_estimators": 20} if SMOKE else {})
    import lightgbm
    import xgboost
    for h in HS:
        ytr = tr[f"y_{h}"].to_numpy()
        for seed in SEEDS:
            clf = XGBClassifier(objective="multi:softprob", num_class=3, random_state=seed, **xgb_params).fit(tr[xgb_feats], ytr)
            pr = clf.predict_proba(pred[xgb_feats])
            for c, cls in enumerate(L.CLASSES):
                out[f"xgb_s{seed}_p{h}_{cls}"] = pr[:, c].round(6)
            if h == 1:                                # LightGBM E29 is v2's partner at 1 day only
                from lightgbm import LGBMClassifier
                lg = LGBMClassifier(random_state=seed, **lgbm_params).fit(tr[big], ytr)
                pl = lg.predict_proba(pred[big])
                for c, cls in enumerate(L.CLASSES):
                    out[f"lgbm_s{seed}_p{h}_{cls}"] = pl[:, c].round(6)
            say(f"  h{h} seed {seed} done ({time.time() - t0:.0f}s)")
    (out_dir / "trees_meta.json").write_text(json.dumps({"xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
                                                         "xgb_params": xgb_params, "lgbm_params": lgbm_params, "n_inputs_lgbm": len(big),
                                                         "seconds": round(time.time() - t0, 1), "smoke": SMOKE}, indent=2), encoding="utf-8")
    out.to_csv(out_dir / "trees.csv.gz", index=False)


# ----------------------------------------------------------------------------------------------- scoring
def v2_probs(df, h, s, partner_prefix=True):
    p = PH.probs(df, DEEP, s, h)
    partner, _ = RULE[h]
    if partner:
        p = 0.5 * p + 0.5 * PH.probs(df, partner, s, h)
    return p


def v2_classes(df, h, seeds):
    return np.stack([PH.predict(v2_probs(df, h, s), RULE[h][1]) for s in seeds])


def f1s(y, preds):
    return [PH.macro_f1(y, p) for p in np.atleast_2d(preds)]


def ms(xs):
    return round(statistics.mean(xs), 4), (round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0)


def merge_preds(out_dir: Path) -> pd.DataFrame:
    deep_ = None
    for s in SEEDS:
        d = pd.read_csv(out_dir / f"deep_s{s}.csv.gz")
        deep_ = d if deep_ is None else deep_.merge(d, on=["symbol", "date", "split"], validate="one_to_one")
    t = pd.read_csv(out_dir / "trees.csv.gz")
    nn = deep_[deep_.split != "val_nonews"]
    df = nn.merge(t, on=["symbol", "date", "split"], how="left", validate="one_to_one")
    assert df.filter(like="xgb_s").notna().all().all(), "tree predictions missing for some rows"
    return df, deep_[deep_.split == "val_nonews"]


def repro_check(df_val: pd.DataFrame, refs: dict) -> dict:
    """Re-trained validation predictions vs the stored files (same rows, same columns)."""
    res = {}
    for name, (path, prefix, hs) in refs.items():
        if not path or not Path(path).exists():
            res[name] = {"status": "reference file not given or missing"}; continue
        ref = pd.read_csv(path)
        ref["date"] = ref["date"].astype(str).str[:10]
        m = df_val.merge(ref, on=["symbol", "date"], suffixes=("", "_ref"), validate="one_to_one")
        r = {"rows_matched": int(len(m)), "rows_val": int(len(df_val)), "rows_ref": int(len(ref))}
        for h in hs:
            y = m[f"y_{h}"].to_numpy(int)
            if f"y_{h}_ref" in m:
                assert (y == m[f"y_{h}_ref"].to_numpy(int)).all(), f"labels differ from {name}'s stored file at h={h}"
            mine = [PH.macro_f1(y, PH.probs(m, prefix, s, h).argmax(1)) for s in SEEDS]
            theirs = [PH.macro_f1(y, m[[f"{prefix}_s{s}_p{h}_{c}_ref" for c in L.CLASSES]].to_numpy(float).argmax(1)) for s in SEEDS]
            dp = max(float(np.abs(PH.probs(m, prefix, s, h) - m[[f"{prefix}_s{s}_p{h}_{c}_ref" for c in L.CLASSES]].to_numpy(float)).max()) for s in SEEDS)
            r[f"h{h}"] = {"retrained": ms(mine)[0], "stored": ms(theirs)[0], "diff": round(ms(mine)[0] - ms(theirs)[0], 4),
                          "max_abs_prob_diff": round(dp, 6)}
        res[name] = r
    return res


def score(args):
    out_dir = Path(args.out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    df, nonews = merge_preds(out_dir)
    rows, info, full, _ = load_forward(Path(args.fwd_root))
    lab = key_frame(rows)
    for h in HS:
        lab[f"y_{h}"] = rows[f"y_{h}"].to_numpy(); lab[f"fwd_{h}"] = rows[f"fwd_{h}"].to_numpy() if f"fwd_{h}" in rows else np.nan
        lab[f"r_{h}"] = rows[f"r_{h}"].to_numpy(); lab[f"sigma_{h}"] = rows[f"sigma_{h}"].to_numpy()
    df = df.merge(lab, on=["symbol", "date", "split"], how="left", validate="one_to_one")
    assert not df.split.isin(["test", "purged"]).any()
    val, fwd, live = df[df.split == "val"], df[df.split == "fwd"].reset_index(drop=True), df[df.split == "live"]
    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": "main model v2 (re-trained, commit 6eaa76f)",
           "scored_on": "2026 forward window; 2025 test block not used", "seeds": list(SEEDS), "smoke": SMOKE,
           "forward_window": manifest["forward_window"], "pass_bar": {f"h{h}": PASS_BAR[h] for h in HS}}

    # 1) reproduction on the official validation window
    refs = {"deep": (args.ref_deep, DEEP, HS), "xgb": (args.ref_xgb, "xgb", HS), "lgbm": (args.ref_lgbm, "lgbm", (1,))}
    rep = repro_check(val, refs)
    v2val = {}
    for h in HS:
        y = val[f"y_{h}"].to_numpy(int)
        nb = nonews.drop(columns="split").merge(val[["symbol", "date", f"y_{h}"]], on=["symbol", "date"], validate="one_to_one")
        v2val[f"h{h}"] = {"v2": ms(f1s(y, v2_classes(val, h, SEEDS)))[0], "stored_v2": REF_VAL["v2"][h],
                          "deep_alone": ms(f1s(y, [PH.probs(val, DEEP, s, h).argmax(1) for s in SEEDS]))[0],
                          "deep_news_blanked": ms(f1s(nb[f"y_{h}"].to_numpy(int), [PH.probs(nb, DEEP, s, h).argmax(1) for s in SEEDS]))[0]}
        v2val[f"h{h}"]["diff_vs_stored"] = round(v2val[f"h{h}"]["v2"] - REF_VAL["v2"][h], 4)
    ok = all(abs(v2val[f"h{h}"]["diff_vs_stored"]) <= REPRO_TOL for h in HS)
    res["reproduction"] = {"components": rep, "v2_validation": v2val, "tolerance": REPRO_TOL, "ok": ok}
    say("\nREPRODUCTION on the official validation window (re-trained vs stored v2):")
    for h in HS:
        v = v2val[f"h{h}"]
        say(f"  h{h}: v2 {v['v2']:.4f} vs stored {v['stored_v2']:.4f} ({v['diff_vs_stored']:+.4f}); deep alone {v['deep_alone']:.4f}"
            f" (stored {REF_VAL['deep'][h]:.4f}); deep with news blanked {v['deep_news_blanked']}")
    for k, v in rep.items():
        say(f"  {k}: {v}")
    if not ok and not args.allow_nonidentical and not SMOKE:
        (out_dir / "forward_summary.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        raise SystemExit("stop: the re-trained v2 does not reproduce the stored validation scores within "
                         f"{REPRO_TOL}. Forward results would not be v2's. Check GPU type and library versions "
                         "(or pass --allow-nonidentical to score anyway and report it).")

    # 2) forward window
    res["horizons"] = {}
    dates = fwd["date"].to_numpy()
    for h in HS:
        y = fwd[f"y_{h}"].to_numpy(int)
        k = info[f"h{h}"]["k"]
        past = L.classify(fwd[f"r_{h}"].to_numpy(), fwd[f"sigma_{h}"].to_numpy(), k)
        tr_sh = np.array([info[f"h{h}"]["train_class_shares"][c] for c in L.CLASSES])
        pv = np.bincount(y, minlength=3) / len(y)
        v2 = v2_classes(fwd, h, SEEDS)
        deep_ = np.stack([PH.probs(fwd, DEEP, s, h).argmax(1) for s in SEEDS])
        xgb = np.stack([PH.probs(fwd, "xgb", s, h).argmax(1) for s in SEEDS])
        comp = {"v2": ms(f1s(y, v2)), "deep_alone": ms(f1s(y, deep_)), "xgb": ms(f1s(y, xgb))}
        if h == 1:
            comp["lgbm"] = ms(f1s(y, np.stack([PH.probs(fwd, "lgbm", s, h).argmax(1) for s in SEEDS])))
        floors = {"trend_continues": round(PH.macro_f1(y, past), 4), "trend_reverses": round(PH.macro_f1(y, 2 - past), 4),
                  "stratified_random_expected": round(float(np.mean(2 * pv * tr_sh / (pv + tr_sh))), 4),
                  "majority_class": M.trend_metrics(y, np.full_like(y, int(np.argmax(tr_sh))))["macro_f1"]}
        best_floor = max(floors["trend_continues"], floors["trend_reverses"])
        window_bar = round(max(best_floor, comp["xgb"][0]) + MARGIN, 4)
        boot = PH.block_bootstrap(dates, y, {"v2": v2, "deep_alone": deep_, "xgb": xgb, "trend_continues": past[None, :],
                                             "trend_reverses": (2 - past)[None, :]},
                                  pairs=[("v2", "xgb"), ("v2", "deep_alone"), ("v2", "trend_continues"), ("v2", "trend_reverses")],
                                  bar=PASS_BAR[h])
        cm = np.zeros((3, 3), int)
        for p in v2:
            cm += np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
        spread = []
        for p in v2:
            t = pd.DataFrame({"d": dates, "r": fwd[f"fwd_{h}"].to_numpy(), "p": p}).groupby(["d", "p"]).r.mean().unstack()
            spread.append(float((t.get(2) - t.get(0)).mean()) if 2 in t and 0 in t else float("nan"))
        res["horizons"][f"h{h}"] = {
            "macro_f1": {k_: {"mean": v[0], "std": v[1]} for k_, v in comp.items()}, "floors": floors,
            "official_bar": PASS_BAR[h], "passes_official_bar": comp["v2"][0] >= PASS_BAR[h],
            "forward_window_bar": window_bar, "passes_forward_window_bar": comp["v2"][0] >= window_bar,
            "v2_minus_best_floor": round(comp["v2"][0] - best_floor, 4), "v2_minus_xgb": round(comp["v2"][0] - comp["xgb"][0], 4),
            "vs_validation": round(comp["v2"][0] - REF_VAL["v2"][h], 4), "vs_walk_forward_mean": round(comp["v2"][0] - REF_WF_MEAN[h], 4),
            "bootstrap": boot, "confusion_pooled_over_seeds": cm.tolist(), "confusion_layout": "rows actual down/flat/up, cols predicted",
            "class_shares_actual": {c: round(float(pv[i]), 4) for i, c in enumerate(L.CLASSES)},
            "class_shares_predicted": {c: round(float((v2 == i).mean()), 4) for i, c in enumerate(L.CLASSES)},
            "up_minus_down_mean_fwd_log_return": {"mean_over_seeds": round(float(np.nanmean(spread)), 6), "per_seed": [round(s, 6) for s in spread],
                                                 "note": "equal-weight daily average of h-day outcomes; overlapping, no costs; not a backtest"}}
        # monthly
        mon = pd.Series(pd.to_datetime(dates).strftime("%Y-%m"))
        res["horizons"][f"h{h}"]["monthly"] = {m: {"rows": int((mon == m).sum()), "v2": round(statistics.mean(PH.macro_f1(y[mon == m], p[mon == m]) for p in v2), 4),
                                                   "trend_continues": round(PH.macro_f1(y[mon == m], past[mon == m]), 4),
                                                   "trend_reverses": round(PH.macro_f1(y[mon == m], 2 - past[mon == m]), 4)}
                                               for m in sorted(mon.unique())}
        new = fwd.symbol.isin(json.loads((Path(args.fwd_root) / "forward_config.json").read_text(encoding="utf-8"))["new_symbols"]).to_numpy()
        if new.any():
            res["horizons"][f"h{h}"]["new_tickers"] = {"rows": int(new.sum()), "v2": round(statistics.mean(PH.macro_f1(y[new], p[new]) for p in v2), 4),
                                                       "other_rows_v2": round(statistics.mean(PH.macro_f1(y[~new], p[~new]) for p in v2), 4)}

    # 3) live calls (outcomes not known yet): 3-seed average of v2's blended probabilities, then the bias
    live_out = live[["symbol", "date"]].copy()
    for h in HS:
        p = np.mean([v2_probs(live, h, s) for s in SEEDS], axis=0)
        live_out[f"call_{h}d"] = [L.CLASSES[i] for i in PH.predict(p, RULE[h][1])]
        for c, cls in enumerate(L.CLASSES):
            live_out[f"p{h}_{cls}"] = p[:, c].round(4)
    res["live"] = {"rows": int(len(live)), "dates": sorted(live.date.unique().tolist())}

    # outputs
    keep = ["symbol", "date", "split"] + [c for c in df.columns if c.startswith(("y_", "fwd_"))]
    fo = fwd[keep].copy()
    for h in HS:
        for s in SEEDS:
            fo[f"v2_s{s}_class{h}"] = PH.predict(v2_probs(fwd, h, s), RULE[h][1])
    fo = pd.concat([fo, fwd[[c for c in fwd.columns if re.search(r"_s\d+_p\d+_", c)]]], axis=1)
    fo.to_csv(out_dir / "forward_preds.csv.gz", index=False)
    live_out.to_csv(out_dir / "live_calls.csv", index=False)
    (out_dir / "forward_summary.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    say(f"\nFORWARD WINDOW {manifest['forward_window']['first']} .. {manifest['forward_window']['last']} "
        f"({len(fwd)} stock-days, {fwd.date.nunique()} dates). 3-seed mean macro-F1:")
    say(f"{'h':4s} {'v2':>14s} {'deep':>7s} {'xgb':>7s} {'cont.':>7s} {'rev.':>7s} {'bar':>6s} {'fw-bar':>7s} {'95% interval':>17s} {'P>=bar':>7s} {'vs val':>7s} {'vs WF':>7s}")
    for h in HS:
        r = res["horizons"][f"h{h}"]
        b = r["bootstrap"]["values"]["v2"]
        say(f"h{h:<3d} {r['macro_f1']['v2']['mean']:.4f}±{r['macro_f1']['v2']['std']:.4f} {r['macro_f1']['deep_alone']['mean']:7.4f} "
            f"{r['macro_f1']['xgb']['mean']:7.4f} {r['floors']['trend_continues']:7.4f} {r['floors']['trend_reverses']:7.4f} "
            f"{PASS_BAR[h]:6.3f} {r['forward_window_bar']:7.4f} {str(b['ci95']):>17s} {b['share_of_resamples_at_or_above_bar']:7.3f} "
            f"{r['vs_validation']:+7.4f} {r['vs_walk_forward_mean']:+7.4f}")
    plots(res, out_dir)
    say(f"wrote {out_dir / 'forward_summary.json'}, forward_preds.csv.gz, live_calls.csv, forward_scores.png, forward_monthly.png")


def plots(res, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#1f2328", "#6e7781", "#e6e8eb"
    c_v2, c_ref, c_floor = "#2a6fdb", "#9aa4b1", "#d97706"
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for ax, h in zip(axs, HS):
        r = res["horizons"][f"h{h}"]
        b = r["bootstrap"]["values"]
        items = [("v2 (forward)", b["v2"]["point"], b["v2"]["ci95"], c_v2), ("deep alone", b["deep_alone"]["point"], b["deep_alone"]["ci95"], c_ref),
                 ("XGBoost", b["xgb"]["point"], b["xgb"]["ci95"], c_ref), ("trend continues", b["trend_continues"]["point"], b["trend_continues"]["ci95"], c_floor),
                 ("trend reverses", b["trend_reverses"]["point"], b["trend_reverses"]["ci95"], c_floor)]
        for i, (lab, pt, ci, col) in enumerate(items[::-1]):
            ax.plot(ci, [i, i], color=col, lw=2, solid_capstyle="round")
            ax.plot(pt, i, "o", color=col, ms=7)
        ax.axvline(PASS_BAR[h], color=ink, ls="--", lw=1)
        ax.axvline(REF_VAL["v2"][h], color=c_v2, ls=":", lw=1)
        ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items[::-1]] if h == 1 else [])
        ax.set_title(f"{h}-day", color=ink, fontsize=11, loc="left")
        ax.grid(axis="x", color=grid); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=muted)
        ax.set_xlabel("macro-F1 (dot) and 95% date-block interval", color=muted, fontsize=9)
    fig.text(0.01, 0.01, "dashed = pass bar; dotted blue = v2 on the official validation window", color=muted, fontsize=9)
    fig.suptitle("Main model v2 on the 2026 forward window", x=0.01, ha="left", color=ink, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(out_dir / "forward_scores.png", dpi=140); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
    for ax, h in zip(axs, HS):
        mo = res["horizons"][f"h{h}"]["monthly"]
        ks = list(mo)
        ax.plot(ks, [mo[k]["v2"] for k in ks], "-o", color=c_v2, ms=4, label="v2")
        ax.plot(ks, [max(mo[k]["trend_continues"], mo[k]["trend_reverses"]) for k in ks], "-", color=c_floor, lw=1.2, label="better one-rule floor")
        ax.axhline(PASS_BAR[h], color=ink, ls="--", lw=1)
        ax.set_title(f"{h}-day", color=ink, fontsize=11, loc="left")
        ax.tick_params(axis="x", rotation=45, colors=muted); ax.tick_params(axis="y", colors=muted)
        ax.grid(axis="y", color=grid); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(frameon=False, fontsize=9)
    axs[0].set_ylabel("macro-F1 by month", color=muted)
    fig.tight_layout(); fig.savefig(out_dir / "forward_monthly.png", dpi=140); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "deep", "trees", "score"])
    ap.add_argument("--fwd-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--price-source", choices=["siamchart", "yahoo"], default="siamchart")
    ap.add_argument("--archive", default=None, help="fresh Siamchart EOD archive (zip of daily CSVs, a folder, or a CSV)")
    ap.add_argument("--events-csv", default=None, help="corporate-action events file (symbol,date,event,value) instead of Yahoo")
    ap.add_argument("--allow-missing-events", action="store_true")
    ap.add_argument("--extra-breaks", default=None, help='JSON, e.g. {"XYZ": ["2026-05-04"]}')
    ap.add_argument("--refresh", action="store_true", help="fetch forward prices and events again")
    ap.add_argument("--ref-deep", default=None); ap.add_argument("--ref-xgb", default=None); ap.add_argument("--ref-lgbm", default=None)
    ap.add_argument("--allow-nonidentical", action="store_true")
    args = ap.parse_args(argv)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "deep": deep, "trees": trees, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
