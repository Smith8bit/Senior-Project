"""Look-ahead and sanity tests for the batch 4-5 inputs: E26 (src/features_vol.py) and E29 (src/alpha158.py).

  python tests/test_new_features.py        # needs data/raw/prices (local only; not in git)

Same method as tests/test_no_lookahead.py, which E03 passed:
1. Change the future: scramble every price, volume and return AFTER a cut-off date T; every new input on or before T
   must come out identical (two cut-offs).
2. Test of the test: a deliberately leaky input (tomorrow's return) must be caught.
3. Direct computation: Parkinson volatility, the overnight gap and three Alpha158 factors recomputed by hand on random
   rows; cross-sectional ranks recomputed over the members of that day.
4. Sample rows: the dataset built with the E26 inputs has the same rows, splits and labels as without them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import alpha158 as A  # noqa: E402
import features as F  # noqa: E402
import features_vol as V  # noqa: E402
import prices as P  # noqa: E402
from test_no_lookahead import same, scramble  # noqa: E402

FAILS = []
COLS = V.VOL_FEATURES + A.FACTORS + A.RANKED


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def build(df):
    return A.add_factors(V.add_vol_features(df))


def main():
    if not (ROOT / "data/raw/prices/siamchart/set50_union_2019_2025.csv").exists():
        print("skip: price data not present"); return
    df = P.load_prices(ROOT)
    base = build(df)
    mem = base.member.astype(bool)
    check(len(A.FACTORS) == 157 and len(set(A.FACTORS)) == 157, f"{len(A.FACTORS)} distinct Alpha158-style factors (158 minus VWAP0)")
    check(len(V.VOL_FEATURES) == 9, "9 E26 inputs")
    for c in V.VOL_FEATURES + ["KMID", "ROC60", "CORR60", "xr_MA20"]:
        share = base.loc[mem, c].notna().mean()
        check(share > 0.9, f"{c}: defined on {share:.0%} of member stock-days")

    # 1 + 2) change the future
    base["leaky_tomorrow"] = base.groupby("seg_key")["ret"].shift(-1)
    for T in ("2022-06-30", "2024-02-15"):
        T = pd.Timestamp(T)
        pert = build(scramble(df, df.date > T, seed=1))
        pert["leaky_tomorrow"] = pert.groupby("seg_key")["ret"].shift(-1)
        upto = base.date <= T
        bad = [c for c in COLS if not same(base.loc[upto, c], pert.loc[upto, c])]
        check(not bad, f"cut-off {T.date()}: all {len(COLS)} new inputs on or before the cut-off unchanged" + (f" (changed: {bad[:8]})" if bad else ""))
        # (range volatility is not in this list: the scramble scales a day's open/high/low/close by one factor, so
        # high/low ratios, and with them Parkinson volatility, are unchanged by design)
        moved = [c for c in ("gap_1", "vol_5", "MA20", "xr_MA20") if not same(base.loc[~upto, c], pert.loc[~upto, c])]
        check(len(moved) == 4, f"cut-off {T.date()}: inputs after the cut-off did change (scramble is effective)")
        check(not same(base.loc[upto, "leaky_tomorrow"], pert.loc[upto, "leaky_tomorrow"]), f"cut-off {T.date()}: a deliberately leaky input IS caught")

    # 3) direct computation on random rows
    rows = base[mem & base["CORR20"].notna() & base["park_vol_20"].notna()].sample(60, random_state=1)
    errs = {"park": [], "gap": [], "MA20": [], "RSV10": [], "SUMP5": [], "rank": []}
    for i, row in rows.iterrows():
        g = df[df.seg_key == row.seg_key]
        a = F.adjusted_ohlc(g)
        pos = g.index.get_loc(i)
        w = a.iloc[pos - 19:pos + 1]
        park = np.sqrt(np.mean(np.log(w.high / w.low) ** 2) / (4 * np.log(2)))
        errs["park"].append(abs(park - row.park_vol_20))
        errs["gap"].append(abs(np.log(a.open.iloc[pos] / a.close.iloc[pos - 1]) - row.gap_1))
        errs["MA20"].append(abs(w.close.mean() / w.close.iloc[-1] - row.MA20))
        w10 = a.iloc[pos - 9:pos + 1]
        rsv = (w10.close.iloc[-1] - w10.low.min()) / (w10.high.max() - w10.low.min() + 1e-12)
        errs["RSV10"].append(abs(rsv - row.RSV10))
        dc = a.close.diff().iloc[pos - 4:pos + 1]
        errs["SUMP5"].append(abs(dc.clip(lower=0).sum() / (dc.abs().sum() + 1e-12) - row.SUMP5))
        day = base[(base.date == row.date) & mem]
        errs["rank"].append(abs(day["MA20"].rank(pct=True).loc[i] - row.xr_MA20))
    tol = {"park": 1e-6, "gap": 1e-9, "MA20": 1e-5, "RSV10": 1e-5, "SUMP5": 1e-5, "rank": 1e-6}
    for k, e in errs.items():
        check(max(e) < tol[k], f"direct computation {k}: max error {max(e):.1e} on 60 random member rows")
    check(base.loc[~mem, A.RANKED[:5]].isna().all().all(), "cross-sectional ranks are NaN on non-member stock-days")

    # 4) same sample rows with and without E26
    import baselines_tabular as BT
    d0, _ = BT.build_dataset(extra="e03")
    d1, info1 = BT.build_dataset(extra="e03_e26")
    check(d0.index.equals(d1.index) and (d0.split == d1.split).all(), f"same {len(d0)} sample rows and splits with and without E26")
    check(all((d0[f"y_{h}"] == d1[f"y_{h}"]).all() for h in (1, 5, 10)), "same labels with and without E26")
    check(info1["extra_features"] == BT.EXTRA["e03_e26"] and len(BT.EXTRA["e03_e26"]) == 20, "dataset records the 11 + 9 extra inputs")
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
