"""Does news sentiment add anything beyond price + indicators? (monkey 3 / register Phase 5.3)

Implements claude/pre-registration-amendment-2026-10-15-news-beyond-price.md exactly:
  Arm A  price + indicators (same rows, split, labels, XGBoost settings and seeds as baselines_tabular.py)
  Arm B  A + news features from data/derived/sentiment_scores.csv
  Arm C  B with the news features rebuilt WITHOUT price-restatement articles (restatement == 1)
Measure: validation macro-F1 per horizon from seed-averaged class probabilities; delta = B - A, C - A;
95% CI from a paired bootstrap over validation DATES (1,000 resamples, seed 20261015).
Decision: news adds information if the B - A CI is above 0 at >= 2 of 3 horizons AND the C - A CI is too.
Validation only; the test block stays sealed.

  python scripts/news_beyond_price.py [--scores data/derived/sentiment_scores.csv] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
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

NEWS = ["sent_mean", "sent_n", "has_news", "sent_mean_5", "sent_n_5"]
BOOT, BOOT_SEED = 1000, 20261015


def news_features(scores: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, row date): features known at the close of that day. News whose trading day is not a trading
    day of that stock (e.g. suspension) rolls forward to the stock's next row."""
    sc = scores.copy()
    sc["trading_day"] = pd.to_datetime(sc["trading_day"])
    days = rows[["symbol", "date"]].drop_duplicates().sort_values(["date"])
    sc = sc.sort_values("trading_day")
    sc = pd.merge_asof(sc, days.rename(columns={"date": "row_date"}), left_on="trading_day", right_on="row_date",
                       by=None, left_by="ticker", right_by="symbol", direction="forward").dropna(subset=["row_date"])
    daily = sc.groupby(["ticker", "row_date"]).agg(s_sum=("s", "sum"), n=("s", "size")).reset_index()
    daily = daily.rename(columns={"ticker": "symbol", "row_date": "date"})
    out = rows[["symbol", "date"]].merge(daily, on=["symbol", "date"], how="left").fillna({"s_sum": 0.0, "n": 0})
    out = out.sort_values(["symbol", "date"])
    g = out.groupby("symbol")
    s5, n5 = g["s_sum"].transform(lambda x: x.rolling(5, min_periods=1).sum()), g["n"].transform(lambda x: x.rolling(5, min_periods=1).sum())
    out["sent_mean"] = np.where(out.n > 0, out.s_sum / out.n.clip(lower=1), 0.0)
    out["sent_n"], out["has_news"] = np.log1p(out.n), (out.n > 0).astype(float)
    out["sent_mean_5"] = np.where(n5 > 0, s5 / n5.clip(lower=1), 0.0)
    out["sent_n_5"] = np.log1p(n5)
    return out.set_index(["symbol", "date"])[NEWS]


def fit_predict(Xtr, ytr, Xva):
    from xgboost import XGBClassifier
    ps = [XGBClassifier(objective="multi:softprob", num_class=3, random_state=s, **BT.XGB).fit(Xtr, ytr).predict_proba(Xva)
          for s in BT.SEEDS]
    return np.mean(ps, axis=0)


def boot_delta(y, pa, pb, dates, rng):
    """Paired bootstrap over dates of macro-F1(B) - macro-F1(A)."""
    from sklearn.metrics import f1_score
    u = np.unique(dates); pos = {d: np.flatnonzero(dates == d) for d in u}
    ya, yb = pa.argmax(1), pb.argmax(1)
    out = []
    for _ in range(BOOT):
        idx = np.concatenate([pos[d] for d in rng.choice(u, len(u), replace=True)])
        out.append(f1_score(y[idx], yb[idx], average="macro", labels=[0, 1, 2]) - f1_score(y[idx], ya[idx], average="macro", labels=[0, 1, 2]))
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/derived/sentiment_scores.csv")
    ap.add_argument("--out", default="data/derived/news_beyond_price.json")
    args = ap.parse_args(argv)
    scores = pd.read_csv(ROOT / args.scores)
    data, info = BT.build_dataset()
    nf_all = news_features(scores, data)
    nf_ctl = news_features(scores[scores.restatement == 0], data)
    key = pd.MultiIndex.from_frame(data[["symbol", "date"]])
    B_cols = F.FEATURES + NEWS
    data_b = data.copy(); data_b[NEWS] = nf_all.reindex(key).to_numpy()
    data_c = data.copy(); data_c[NEWS] = nf_ctl.reindex(key).to_numpy()
    tr, va = data.split == "train", data.split == "val"
    dates = data.loc[va, "date"].to_numpy()
    rng = np.random.default_rng(BOOT_SEED)
    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": BT.git_commit(),
           "scores_file": args.scores, "evaluated_on": "validation only (test sealed)",
           "news_coverage_val": {"B_share_rows_with_news": round(float(data_b.loc[va, "has_news"].mean()), 4),
                                 "C_share_rows_with_news": round(float(data_c.loc[va, "has_news"].mean()), 4)},
           "horizons": {}}
    passes_b = passes_c = 0
    for h in L.HORIZONS:
        y_tr, y_va = data.loc[tr, f"y_{h}"].to_numpy(), data.loc[va, f"y_{h}"].to_numpy()
        pa = fit_predict(data.loc[tr, F.FEATURES], y_tr, data.loc[va, F.FEATURES])
        pb = fit_predict(data_b.loc[tr, B_cols], y_tr, data_b.loc[va, B_cols])
        pc = fit_predict(data_c.loc[tr, B_cols], y_tr, data_c.loc[va, B_cols])
        fa, fb, fc = (M.trend_metrics(y_va, p.argmax(1))["macro_f1"] for p in (pa, pb, pc))
        ci_b, ci_c = boot_delta(y_va, pa, pb, dates, rng), boot_delta(y_va, pa, pc, dates, rng)
        passes_b += ci_b[0] > 0; passes_c += ci_c[0] > 0
        res["horizons"][f"h{h}"] = {"macro_f1": {"A": fa, "B": fb, "C": fc},
                                    "delta_B_minus_A": round(fb - fa, 4), "ci95_B_minus_A": ci_b,
                                    "delta_C_minus_A": round(fc - fa, 4), "ci95_C_minus_A": ci_c}
        print(f"h={h}: A {fa:.4f}  B {fb:.4f} (d {fb - fa:+.4f}, CI {ci_b})  C {fc:.4f} (d {fc - fa:+.4f}, CI {ci_c})", flush=True)
    if passes_b >= 2 and passes_c >= 2:
        verdict = "news adds information beyond price, not only by restating price moves"
    elif passes_b >= 2:
        verdict = "gain is mostly price restatement; the sentiment ablation uses arm C"
    else:
        verdict = "no measurable gain beyond price (negative-result framing)"
    res["decision"] = {"B_horizons_ci_above_0": int(passes_b), "C_horizons_ci_above_0": int(passes_c), "verdict": verdict}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(verdict); print(f"wrote {out}")


if __name__ == "__main__":
    main()
