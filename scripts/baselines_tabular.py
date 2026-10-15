"""First real-data baselines: naive floors + XGBoost on price and technical-indicator features.

Monkey 4 ("is there any predictable signal at 1/5/10 days beyond the naive floors?"), price+TA only.
Pre-registered expectation (status doc §11d): R²_OS on returns 0 to +1%, possibly negative at 1 day;
anything near 10% means a leak. Trend: the model should clear majority-class on macro-F1.

VALIDATION ONLY. The test block is sealed until Phase 6; this script never scores it.

  python scripts/baselines_tabular.py            # writes data/derived/baselines_tabular.json
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402
import labels as L  # noqa: E402
import metrics as M  # noqa: E402
import prices as P  # noqa: E402
import splits as SP  # noqa: E402

SEEDS = (20261015, 20261016, 20261017)
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=50, reg_lambda=1.0, tree_method="hist", n_jobs=4)   # fixed, untuned


def build_dataset(root: Path = ROOT, return_full: bool = False):
    """Sample rows (member stock-days with features and labels) + info; with return_full, also the full
    daily frame (all trading days, incl. non-member days) that sequence models cut their windows from."""
    df = P.load_prices(root)
    df = F.add_features(df)
    df = L.add_forward_returns(df)
    fwd = [f"fwd_{h}" for h in L.HORIZONS]
    keep = df.member & (df.date >= P.STUDY_START) & (df.date <= P.STUDY_END)
    keep &= df[F.FEATURES + fwd].notna().all(axis=1)
    data = df[keep].copy()
    blocks = SP.date_blocks(data.date)
    data["split"] = SP.assign(data, blocks)
    info = {"rows_total": int(len(data)), "blocks": {k: [str(pd.Timestamp(v[0]).date()), str(pd.Timestamp(v[1]).date())]
                                                     if isinstance(v, tuple) else v for k, v in blocks.items()},
            "rows_by_split": data.split.value_counts().to_dict(), "symbols": int(data.symbol.nunique()),
            "dividend_coverage_zero": P.dividend_coverage(df).query("div_events == 0").symbol.tolist()}
    train = data[data.split == "train"]
    for h in L.HORIZONS:
        per, fb = L.fit_sigma(train, h)
        data[f"sigma_{h}"] = L.sigma_for(data.symbol, per, fb)
        tr = data.split == "train"
        k, shares = L.fit_k(data.loc[tr, f"fwd_{h}"].to_numpy(), data.loc[tr, f"sigma_{h}"].to_numpy())
        data[f"y_{h}"] = L.classify(data[f"fwd_{h}"].to_numpy(), data[f"sigma_{h}"].to_numpy(), k)
        info[f"h{h}"] = {"k": k, "train_class_shares": shares, "sigma_fallback": round(fb, 6),
                         "stocks_on_fallback_sigma": int(data.loc[~data.symbol.isin(per), "symbol"].nunique()),
                         "val_class_shares": {c: round(float((data.loc[data.split == "val", f"y_{h}"] == i).mean()), 4)
                                              for i, c in enumerate(L.CLASSES)}}
    return (data, info, df) if return_full else (data, info)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def confusion(y_true, y_pred):
    """3x3 counts, rows = actual (down, flat, up), columns = predicted (down, flat, up)."""
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()


def main():
    from xgboost import XGBClassifier, XGBRegressor
    data, info = build_dataset()
    tr, va = data[data.split == "train"], data[data.split == "val"]
    assert not (data.split == "test").loc[va.index].any()
    Xtr, Xva = tr[F.FEATURES], va[F.FEATURES]
    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
           "evaluated_on": "validation only (test sealed until Phase 6)", "features": F.FEATURES, "xgb_params": XGB,
           "seeds": list(SEEDS), "data": info, "results": {}}
    for h in L.HORIZONS:
        ytr, yva = tr[f"y_{h}"].to_numpy(), va[f"y_{h}"].to_numpy()
        rtr, rva = tr[f"fwd_{h}"].to_numpy(), va[f"fwd_{h}"].to_numpy()
        maj = M.majority_class(ytr)
        k = info[f"h{h}"]["k"]
        past = L.classify(va[f"r_{h}"].to_numpy(), va[f"sigma_{h}"].to_numpy(), k)   # past h-day move, same k and sigma
        q = np.bincount(ytr, minlength=3) / len(ytr); pv = np.bincount(yva, minlength=3) / len(yva)
        floors = {"majority_class": M.trend_metrics(yva, np.full_like(yva, maj)) | {"class": L.CLASSES[maj]},
                  "stratified_random_expected": {"macro_f1": round(float(np.mean(2 * pv * q / (pv + q))), 4),
                                                 "accuracy": round(float((pv * q).sum()), 4)},
                  "trend_continues": M.trend_metrics(yva, past) | {"confusion": confusion(yva, past)},
                  "trend_reverses": M.trend_metrics(yva, 2 - past) | {"confusion": confusion(yva, 2 - past)},
                  "persistence_zero_return": M.return_metrics(rva, np.zeros_like(rva)),
                  "train_mean_return": M.return_metrics(rva, np.full_like(rva, rtr.mean()))}
        cls_runs, reg_runs = [], []
        for seed in SEEDS:
            clf = XGBClassifier(objective="multi:softprob", num_class=3, random_state=seed, **XGB).fit(Xtr, ytr)
            yhat = clf.predict(Xva)
            cls_runs.append({"seed": seed, **M.trend_metrics(yva, yhat), "confusion": confusion(yva, yhat)})
            reg = XGBRegressor(objective="reg:squarederror", random_state=seed, **XGB).fit(Xtr, rtr)
            pr = reg.predict(Xva)
            reg_runs.append({"seed": seed, **M.return_metrics(rva, pr), "corr_with_outcome": round(float(np.corrcoef(pr, rva)[0, 1]), 4)})

        def agg(runs, keys):
            return {k: {"mean": round(statistics.mean(r[k] for r in runs), 5), "std": round(statistics.stdev(r[k] for r in runs), 5)}
                    for k in keys}
        res["results"][f"h{h}"] = M.report(
            {"xgb_trend": agg(cls_runs, ["macro_f1", "accuracy"]), "xgb_return": agg(reg_runs, ["rmse", "mae", "r2_os_vs_zero", "direction_acc", "corr_with_outcome"]),
             "per_seed": {"trend": cls_runs, "return": reg_runs}}, floors)
        print(f"h={h}: k={info[f'h{h}']['k']}  trend macro-F1 xgb {res['results'][f'h{h}']['model']['xgb_trend']['macro_f1']} "
              f"vs majority {floors['majority_class']['macro_f1']} / random {floors['stratified_random_expected']['macro_f1']} / "
              f"continues {floors['trend_continues']['macro_f1']} / reverses {floors['trend_reverses']['macro_f1']}  |  "
              f"R2_OS xgb {res['results'][f'h{h}']['model']['xgb_return']['r2_os_vs_zero']}")
    out = ROOT / "data/derived/baselines_tabular.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
