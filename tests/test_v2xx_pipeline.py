"""Full-pipeline look-ahead test for the v2.xx training data (scripts/train_v2xx.build_data), on the real data.

  python tests/test_v2xx_pipeline.py        # about 3-5 minutes on a CPU; needs the price files and sentiment scores

Builds the exact arrays training sees (windows X, labels Y, price targets R, stock and sector ids) three times with
configs/v2_01.json:
  A. the real data;
  B. every price after a cut-off date scrambled (open/high/low/close/volume and returns);
  C. every news article dated after the cut-off scrambled (probabilities shuffled, extra fake articles added).
The cut-off (2024-06-28) lies inside the validation block.

Must hold:
  1. Training and inner hold-out arrays (X, Y, R, ids) are identical in A, B and C: nothing after the cut reaches
     training, the scaler, the labels' k / sigma, or the stock vocabulary.
  2. Every validation window that ends on or before the cut is identical in A, B and C: an input at day t never
     uses anything after t, for prices and for news.
  3. Sanity: validation windows that end after the cut DO change in B and in C (the scramble had an effect).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import baselines_tabular as BT  # noqa: E402
import prices as P  # noqa: E402
import train_v2xx as TV  # noqa: E402

FAILS = []
CUT = pd.Timestamp("2024-06-28")


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def build(cfg, news_df, prices_fn):
    real = P.load_prices
    BT.P.load_prices = prices_fn
    news_path = Path(ROOT / "data/derived/_scramble_news.csv")
    news_df.to_csv(news_path, index=False)
    try:
        return TV.build_data(cfg, news_path)
    finally:
        BT.P.load_prices = real
        news_path.unlink(missing_ok=True)


def main():
    cfg = json.loads((ROOT / "configs/v2_01.json").read_text(encoding="utf-8"))
    news = pd.read_csv(ROOT / "data/derived/sentiment_scores.csv")
    base_prices = P.load_prices(ROOT)

    def scrambled_prices(root):
        d = base_prices.copy(); d["volume"] = d["volume"].astype(float)
        after = d.date > CUT; rng = np.random.default_rng(7)
        for c in ("open", "high", "low", "close"):
            d.loc[after, c] = d.loc[after, c] * rng.uniform(0.6, 1.4, after.sum())
        d.loc[after, "volume"] = d.loc[after, "volume"] * rng.uniform(0.2, 5, after.sum())
        d.loc[after, "ret"] = rng.normal(0, 0.05, after.sum())
        return d

    n2 = news.copy(); rng = np.random.default_rng(8)
    late = pd.to_datetime(n2.trading_day) > CUT
    p = rng.dirichlet([1, 1, 1], late.sum())
    n2.loc[late, ["p_pos", "p_neg", "p_neu"]] = p
    fake = n2[late].sample(2000, random_state=1).copy()
    fake["url"] = fake["url"] + "#fake"
    fake["ticker"] = rng.choice(n2.ticker.unique(), len(fake))
    n2 = pd.concat([n2, fake], ignore_index=True)

    A = build(cfg, news, lambda root: base_prices.copy())
    B = build(cfg, news, scrambled_prices)
    Cn = build(cfg, n2, lambda root: base_prices.copy())

    for name, other in (("prices scrambled after the cut", B), ("news scrambled after the cut", Cn)):
        same_train = all(np.array_equal(A["D"][k][f], other["D"][k][f]) for k in ("fit", "es") for f in ("X", "Y", "S", "G"))
        check(same_train and A["stock_vocab"] == other["stock_vocab"],
              f"{name}: training and inner hold-out windows, labels and ids bit-identical; same stock vocabulary")
        # Price targets: labels.add_forward_returns uses pandas' rolling sum on the reversed series, a running sum whose
        # floating-point rounding carries in values from later in the series. Measured 2026-10-20: differences of at most
        # 3e-15 (returns have std ~0.02, so ~1e-13 relative), and no label changes (Y above is bit-identical). That is
        # rounding, not information, so the targets are compared to 1e-12.
        dR = max(float(np.abs(A["D"][k]["R"] - other["D"][k]["R"]).max()) for k in ("fit", "es"))
        check(dR < 1e-12 and np.allclose(A["r_sd"], other["r_sd"], rtol=0, atol=1e-15),
              f"{name}: training price targets equal up to float rounding (max diff {dR:.1e})")
        dates = A["D"]["val"]["date"]
        upto = dates <= np.datetime64(CUT)
        check(np.array_equal(A["D"]["val"]["date"], other["D"]["val"]["date"]), f"{name}: same validation rows")
        same_before = np.array_equal(A["D"]["val"]["X"][upto], other["D"]["val"]["X"][upto])
        check(same_before, f"{name}: all {int(upto.sum()):,} validation windows ending on or before {CUT.date()} identical")
        changed = (np.abs(A["D"]["val"]["X"][~upto] - other["D"]["val"]["X"][~upto]).reshape((~upto).sum(), -1).max(1) > 0).mean()
        check(changed > 0.5, f"{name}: sanity, {changed:.0%} of the windows ending after the cut changed")
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
