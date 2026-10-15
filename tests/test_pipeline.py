"""Leak and correctness checks for the real-data pipeline (prices -> features -> labels -> split).

  python tests/test_pipeline.py        # needs data/raw/prices (local only; not in git)
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402
import labels as L  # noqa: E402
import prices as P  # noqa: E402
import splits as SP  # noqa: E402

FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def main():
    if not (ROOT / "data/raw/prices/siamchart/set50_union_2019_2025.csv").exists():
        print("skip: price data not present"); return
    df = P.load_prices(ROOT)

    # --- prices
    d = df[df["div"] > 0]
    check(d.ret_raw.mean() < -0.01 and abs(d.ret.mean()) < 0.005, f"dividend adjustment removes the XD drop (raw {d.ret_raw.mean():.4f} -> adj {d.ret.mean():.4f})")
    s = df[df.split != 1]
    check(s.ret.abs().max() < 0.2, f"split days give normal returns after adjustment (max |r| {s.ret.abs().max():.3f})")
    for sym, day in P.CHANGEOVERS.items():
        row = df[(df.symbol == sym) & (df.date == pd.Timestamp(day))]
        check(len(row) == 1 and np.isnan(row.ret.iloc[0]) and row.segment.iloc[0] >= 1, f"no return across the {sym} changeover")
    check(np.isnan(df[(df.symbol == "GULF") & (df.date == pd.Timestamp("2020-04-16"))].ret.iloc[0]), "unexplained GULF 2020-04-16 jump is a break")
    check(df[df.member & (df.date >= "2020-01-01")].groupby("date").symbol.count().between(40, 50).mean() > 0.95,
          "point-in-time membership: 40-50 members with prices on >95% of days")

    # --- features are causal: corrupt everything after t, recompute, values at t unchanged
    g = df[df.seg_key == "PTT#0"].copy()
    base = F.segment_features(g)
    t = g.index[400]
    g2 = g.copy()
    after = g2.index > t
    rng = np.random.default_rng(0)
    for c in ("open", "high", "low", "close"):
        g2.loc[after, c] = g2.loc[after, c] * rng.uniform(0.5, 1.5, after.sum())
    g2.loc[after, "volume"] = rng.integers(1, 10**8, after.sum())
    g2.loc[after, "ret"] = rng.normal(0, 0.1, after.sum())
    pert = F.segment_features(g2)
    check(np.allclose(base.loc[:t].to_numpy(), pert.loc[:t].to_numpy(), equal_nan=True), "indicators at t ignore all data after t")

    # --- labels
    lab = L.add_forward_returns(df)
    x = lab[lab.seg_key == "PTT#0"]
    i = 300
    check(np.isclose(x.fwd_5.iloc[i], x.ret.iloc[i + 1:i + 6].sum()), "fwd_5 at t = sum of returns t+1..t+5")
    tru = lab[lab.symbol == "TRUE"]
    last_old = tru[tru.segment == 0].index[-1]
    check(np.isnan(lab.loc[last_old, "fwd_1"]), "forward returns stop at a changeover")
    sig = inspect.signature(L.classify).parameters
    check(all(sig[p].default is inspect.Parameter.empty for p in ("sigma", "k")), "classify has no default sigma or k")
    k, shares = L.fit_k(np.random.default_rng(1).normal(size=20000), np.ones(20000))
    check(all(abs(v - 1 / 3) < 0.01 for v in shares.values()), f"fit_k balances classes (k={k}, {shares})")

    # --- sequence windows
    import sequences as SQ
    full = F.add_features(df)
    z = SQ.standardise_full(full, F.FEATURES, full.index[:5000])
    tru = full[full.symbol == "TRUE"]
    first_new = tru[tru.segment == 1].index[0]
    rows = pd.Index([first_new + 3, tru.index[500]])
    w = SQ.windows(full, z, rows, 60)
    check(np.allclose(w[1], z[tru.index[500] - 59:tru.index[500] + 1]), "window = the 60 rows ending at t (inclusive), nothing after t")
    check(np.all(w[0][:-4] == 0) and np.allclose(w[0][-4:], z[first_new:first_new + 4]), "window never reaches across a changeover")
    s = pd.read_csv(ROOT / "data/sectors.csv")
    check(set(s.symbol) == set(pd.read_csv(ROOT / "data/set50_membership.csv").symbol), "sector table covers all 70 union tickers")

    # --- split
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=1000))
    b = SP.date_blocks(dates, purge=10)
    check(b["train"][1] < b["val"][0] and b["val"][1] < b["test"][0], "blocks are chronological and disjoint")
    u = np.sort(dates.unique())
    check(int(np.searchsorted(u, b["val"][0]) - np.searchsorted(u, b["train"][1])) == 11, "10 purged dates between train and val")
    frame = pd.DataFrame({"date": dates, "x": np.arange(1000.0)})
    split = SP.assign(frame, b)
    try:
        SP.TrainOnlyScaler().fit(frame[["x"]], split); ok = False
    except ValueError:
        ok = True
    check(ok, "scaler refuses to fit on non-training rows")
    SP.TrainOnlyScaler().fit(frame.loc[split == "train", ["x"]], split)

    print("all pipeline tests passed" if not FAILS else f"{len(FAILS)} FAILED")
    if FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()
