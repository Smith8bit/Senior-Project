"""Leak checks for the E05 walk-forward folds (scripts/baselines_tabular.build_dataset(fold=...)).

  python tests/test_walk_forward.py        # needs data/raw/prices (local only; not in git)

1. Default split unchanged: same rows per block and the same k as before E05.
2. Per fold: exactly `purge` trading dates between the last training date and the first scored date; the scored
   window ends on or before the official validation end; no fold row is in the sealed test block.
3. Change the future: scramble every price, volume and return from the fold's first scored date onward. The fold's
   training rows, their features, labels, sigma and k must be unchanged (nothing from the scored period reaches training).
4. A window that would reach into the test block is refused.
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
import baselines_tabular as BT  # noqa: E402
import features as F  # noqa: E402
import features_xs as XS  # noqa: E402
import prices as P  # noqa: E402
from test_no_lookahead import same, scramble  # noqa: E402

FOLDS = [("2022-07-01", "2022-12-31"), ("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"), ("2024-01-01", "2024-06-30")]
FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def main():
    if not (ROOT / "data/raw/prices/siamchart/set50_union_2019_2025.csv").exists():
        print("skip: price data not present"); return
    d0, i0 = BT.build_dataset(extra="e03")
    check(d0.split.value_counts().to_dict() == {"train": 49610, "test": 10719, "val": 10277, "purged": 990},
          f"default split unchanged: {d0.split.value_counts().to_dict()}")
    check([i0[f"h{h}"]["k"] for h in (1, 5, 10)] == [0.307, 0.318, 0.324], "default k unchanged (0.307 / 0.318 / 0.324)")
    official_val_end = pd.Timestamp(i0["blocks"]["val"][1])
    test_start = pd.Timestamp(i0["blocks"]["test"][0])
    u = np.sort(d0.date.unique())

    real_load = P.load_prices
    for es, ee in FOLDS:
        fold = {"eval_start": es, "eval_end": ee}
        d, info = BT.build_dataset(extra="e03", fold=fold)
        tr, va = d[d.split == "train"], d[d.split == "val"]
        gap = int(np.searchsorted(u, np.datetime64(va.date.min())) - np.searchsorted(u, np.datetime64(tr.date.max()))) - 1
        check(gap == 10, f"{es}..{ee}: {gap} purged trading dates between train end {tr.date.max().date()} and first scored date {va.date.min().date()}")
        check(va.date.max() <= official_val_end and va.date.max() < test_start, f"{es}..{ee}: scored window ends {va.date.max().date()} (official val end {official_val_end.date()})")
        check(not d.loc[d.date >= test_start, "split"].isin(["train", "val"]).any(), f"{es}..{ee}: no train/scored row in the sealed test block")
        print(f"     rows train {len(tr)} / scored {len(va)}; k = {[info[f'h{h}']['k'] for h in (1, 5, 10)]}")

        # change the future from the first scored date onward
        cut = va.date.min()
        P.load_prices = lambda root, _c=cut: (lambda df: scramble(df, df.date >= _c, seed=7))(real_load(root))
        try:
            dp, ip = BT.build_dataset(extra="e03", fold=fold)
        finally:
            P.load_prices = real_load
        trp = dp[dp.split == "train"]
        cols = F.FEATURES + XS.E03_FEATURES + [f"{p}_{h}" for p in ("y", "fwd", "sigma") for h in (1, 5, 10)]
        bad = [c for c in cols if not same(tr[c], trp[c])]
        check(tr.index.equals(trp.index) and not bad, f"{es}..{ee}: scrambling prices from {cut.date()} on leaves training rows, features, labels and sigma unchanged" + (f" (changed: {bad})" if bad else ""))
        check([info[f"h{h}"]["k"] for h in (1, 5, 10)] == [ip[f"h{h}"]["k"] for h in (1, 5, 10)], f"{es}..{ee}: k unchanged after scrambling the future")
        vp = dp[dp.split == "val"]
        check(not same(va["fwd_5"], vp["fwd_5"]), f"{es}..{ee}: scored-window outcomes did change (scramble is effective)")

    try:
        BT.build_dataset(extra="e03", fold={"eval_start": "2024-07-01", "eval_end": "2025-03-31"})
        check(False, "a window reaching into the test block is refused")
    except ValueError as e:
        check("sealed test block" in str(e), "a window reaching into the test block is refused")

    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
