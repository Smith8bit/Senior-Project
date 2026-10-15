"""Look-ahead test for every price-based input (the 18 base features and the E03 inputs).

  python tests/test_no_lookahead.py        # needs data/raw/prices (local only; not in git)

1. Change the future: scramble every price, volume and return AFTER a cut-off date T; every input on or before T
   must come out identical. Run at two cut-offs (mid-train and the train/validation boundary).
2. Test of the test: a deliberately leaky input (tomorrow's return) must be caught by the same comparison.
3. Point-in-time membership: scrambling NON-member stock-days must leave members' 1-day market/sector inputs unchanged.
   (Only 1-day inputs: a stock that just joined SET50 has a 5-20 day return that legitimately spans its own pre-joining
   days, so its longer-lag inputs move when those days are scrambled. Verified 2026-10-16: every such change fell on a
   date with a recent joiner; none unexplained.)
4. Direct computation, every lag: market average = mean over members on day t; sector average = mean over the OTHER
   members of the sector on day t.
5. Sample rows: building the dataset with and without the E03 inputs gives the same rows and labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import features as F  # noqa: E402
import features_xs as XS  # noqa: E402
import prices as P  # noqa: E402

FAILS = []
COLS = F.FEATURES + XS.E03_FEATURES


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def build(df):
    return XS.add_xs_features(F.add_features(df), ROOT)


def scramble(df, mask, seed):
    """Replace prices/volume/returns on `mask` rows with random values (shape of the data kept)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = int(mask.sum())
    f = np.exp(rng.normal(0, 0.2, n))
    for c in ("open", "high", "low", "close"):
        out.loc[mask, c] = out.loc[mask, c].to_numpy() * f
    out.loc[mask, "volume"] = rng.integers(1, 10**8, n)
    r = out.loc[mask, "ret"]
    out.loc[mask, "ret"] = np.where(r.isna(), np.nan, rng.normal(0, 0.05, n))
    return out


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=1e-12))


def main():
    if not (ROOT / "data/raw/prices/siamchart/set50_union_2019_2025.csv").exists():
        print("skip: price data not present"); return
    df = P.load_prices(ROOT)
    base = build(df)
    check(all(c in base for c in COLS), f"all {len(COLS)} inputs built")
    for c in XS.E03_FEATURES:
        share = base.loc[base.member, c].notna().mean()
        check(share > 0.5, f"{c}: defined on {share:.0%} of member stock-days")

    # 1 + 2) change the future
    base["leaky_tomorrow"] = base.groupby("seg_key")["ret"].shift(-1)
    for T in ("2022-06-30", "2024-02-15"):
        T = pd.Timestamp(T)
        after = df.date > T
        pert = build(scramble(df, after, seed=1))
        pert["leaky_tomorrow"] = pert.groupby("seg_key")["ret"].shift(-1)
        upto = base.date <= T
        bad = [c for c in COLS if not same(base.loc[upto, c], pert.loc[upto, c])]
        check(not bad, f"cut-off {T.date()}: all {len(COLS)} inputs on or before the cut-off unchanged" + (f" (changed: {bad})" if bad else ""))
        check(not same(base.loc[~upto, "mrel_r_1"], pert.loc[~upto, "mrel_r_1"]), f"cut-off {T.date()}: inputs after the cut-off did change (scramble is effective)")
        check(not same(base.loc[upto, "leaky_tomorrow"], pert.loc[upto, "leaky_tomorrow"]), f"cut-off {T.date()}: a deliberately leaky input IS caught")

    # 3) non-members must not move members' market/sector inputs
    pert = build(scramble(df, ~df.member.astype(bool), seed=2))
    mem = base.member.astype(bool)
    bad = [c for c in ("mrel_r_1", "srel_r_1") if not same(base.loc[mem, c], pert.loc[mem, c])]
    check(not bad, "scrambling non-member stock-days leaves members' 1-day market and sector inputs unchanged" + (f" (changed: {bad})" if bad else ""))

    # 4) direct computation at every lag (150 random member rows each)
    sec = base.symbol.map(pd.read_csv(ROOT / "data/sectors.csv").set_index("symbol")["sector"])
    for k in XS.LAGS:
        r, mr, sr = f"r_{k}", f"mrel_r_{k}", f"srel_r_{k}"
        rows = base[mem & base[sr].notna()].sample(150, random_state=k)
        e_s = [abs(row[r] - base[(base.date == row.date) & mem & (sec == sec[i]) & base[r].notna() & (base.index != i)][r].mean() - row[sr])
               for i, row in rows.iterrows()]
        rows = base[mem & base[mr].notna()].sample(150, random_state=100 + k)
        e_m = [abs(row[r] - base[(base.date == row.date) & mem & base[r].notna()][r].mean() - row[mr]) for _, row in rows.iterrows()]
        check(max(e_s) < 1e-12 and max(e_m) < 1e-12,
              f"lag {k}: market average = members on day t (max error {max(e_m):.0e}); sector average = other members (max error {max(e_s):.0e})")

    # 5) same sample rows with and without E03
    import baselines_tabular as BT
    d0, _ = BT.build_dataset()
    d1, info1 = BT.build_dataset(extra="e03")
    check(d0.index.equals(d1.index) and (d0.split == d1.split).all(), f"same {len(d0)} sample rows and splits with and without E03")
    check(all((d0[f"y_{h}"] == d1[f"y_{h}"]).all() for h in (1, 5, 10)), "same labels with and without E03")
    check(info1["extra_features"] == XS.E03_FEATURES, "dataset records which extra inputs it carries")

    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
