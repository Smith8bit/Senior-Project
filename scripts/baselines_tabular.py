"""First real-data baselines: naive floors + XGBoost on price and technical-indicator features.

Monkey 4 ("is there any predictable signal at 1/5/10 days beyond the naive floors?"), price+TA only.
Pre-registered expectation (status doc §11d): R²_OS on returns 0 to +1%, possibly negative at 1 day;
anything near 10% means a leak. Trend: the model should clear majority-class on macro-F1.

VALIDATION ONLY. The test block is sealed until Phase 6; this script never scores it.

  python scripts/baselines_tabular.py            # writes data/derived/baselines_tabular.json
  python scripts/baselines_tabular.py --out results/xgb.json   # E02b: also writes results/xgb_preds.csv.gz

Every validation prediction (class probabilities, predicted class and return forecast, per seed) is written to
<out>_preds.csv.gz, keyed by symbol and date, so it can be blended with the deep models' prediction files.
"""
from __future__ import annotations

import argparse
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
import features_xs as XS  # noqa: E402
import labels as L  # noqa: E402
import metrics as M  # noqa: E402
import prices as P  # noqa: E402
import splits as SP  # noqa: E402

SEEDS = (20261015, 20261016, 20261017)
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=50, reg_lambda=1.0, tree_method="hist", n_jobs=4)   # fixed, untuned


EXTRA = {"none": [], "e03": XS.E03_FEATURES}


def build_dataset(root: Path = ROOT, return_full: bool = False, extra: str = "none", fold: dict | None = None):
    """Sample rows (member stock-days with features and labels) + info; with return_full, also the full
    daily frame (all trading days, incl. non-member days) that sequence models cut their windows from.
    extra="e03" adds the E03 inputs (src/features_xs.py). They are NOT part of the row filter, so the sample
    rows are identical with and without them.
    fold={"eval_start": ..., "eval_end": ...} (E05) replaces the official split by a walk-forward fold: train on
    everything before eval_start (minus the purge), score eval_start..eval_end. Labels (k, sigma) are refitted on the
    fold's training rows. Default None = the official 70/15/15 split, unchanged."""
    df = P.load_prices(root)
    df = F.add_features(df)
    if extra != "none":
        df = XS.add_xs_features(df, root)
    df = L.add_forward_returns(df)
    fwd = [f"fwd_{h}" for h in L.HORIZONS]
    keep = df.member & (df.date >= P.STUDY_START) & (df.date <= P.STUDY_END)
    keep &= df[F.FEATURES + fwd].notna().all(axis=1)
    data = df[keep].copy()
    blocks = SP.date_blocks(data.date) if fold is None else SP.walk_forward_blocks(data.date, **fold)
    data["split"] = SP.assign(data, blocks)
    info = {"rows_total": int(len(data)), "blocks": {k: [str(pd.Timestamp(v[0]).date()), str(pd.Timestamp(v[1]).date())]
                                                     if isinstance(v, tuple) else v for k, v in blocks.items()},
            "rows_by_split": data.split.value_counts().to_dict(), "symbols": int(data.symbol.nunique()),
            "dividend_coverage_zero": P.dividend_coverage(df).query("div_events == 0").symbol.tolist(),
            "extra_features": EXTRA[extra]}
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


def fold_arg(args):
    if (args.eval_start is None) != (args.eval_end is None):
        raise SystemExit("give both --eval-start and --eval-end, or neither")
    return None if args.eval_start is None else {"eval_start": args.eval_start, "eval_end": args.eval_end}


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


def main(argv=None):
    import xgboost
    from xgboost import XGBClassifier, XGBRegressor
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/derived/baselines_tabular.json")
    ap.add_argument("--preds", default=None, help="validation predictions file (default: <out>_preds.csv.gz)")
    ap.add_argument("--extra-features", default="none", choices=sorted(EXTRA), help="e03: add src/features_xs.py inputs")
    ap.add_argument("--eval-start", default=None, help="E05 walk-forward fold: first evaluation date (with --eval-end)")
    ap.add_argument("--eval-end", default=None, help="E05 walk-forward fold: last evaluation date")
    args = ap.parse_args(argv)
    out = ROOT / args.out
    pred_path = ROOT / args.preds if args.preds else out.with_name(out.stem + "_preds.csv.gz")
    data, info = build_dataset(extra=args.extra_features, fold=fold_arg(args))
    tr, va = data[data.split == "train"], data[data.split == "val"]
    assert not (data.split == "test").loc[va.index].any()
    feats = F.FEATURES + EXTRA[args.extra_features]
    Xtr, Xva = tr[feats], va[feats]
    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
           "xgboost": xgboost.__version__, "evaluated_on": "validation only (test sealed until Phase 6)",
           "features": feats, "xgb_params": XGB, "seeds": list(SEEDS), "data": info, "results": {},
           "preds_file": pred_path.name}
    preds = va[["symbol", "date"]].reset_index(drop=True).copy()      # one row per validation stock-day
    for h in L.HORIZONS:
        ytr, yva = tr[f"y_{h}"].to_numpy(), va[f"y_{h}"].to_numpy()
        rtr, rva = tr[f"fwd_{h}"].to_numpy(), va[f"fwd_{h}"].to_numpy()
        preds[f"y_{h}"] = yva; preds[f"fwd_{h}"] = rva
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
            prob = clf.predict_proba(Xva)
            assert (prob.argmax(1) == yhat).all(), "predicted class must equal the most likely class"
            preds[f"xgb_s{seed}_class{h}"] = yhat
            for c, cls in enumerate(L.CLASSES):
                preds[f"xgb_s{seed}_p{h}_{cls}"] = prob[:, c].round(6)
            cls_runs.append({"seed": seed, **M.trend_metrics(yva, yhat), "confusion": confusion(yva, yhat)})
            reg = XGBRegressor(objective="reg:squarederror", random_state=seed, **XGB).fit(Xtr, rtr)
            pr = reg.predict(Xva)
            preds[f"xgb_s{seed}_r{h}"] = pr.round(6)
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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    preds.to_csv(pred_path, index=False)
    print(f"wrote {out} and {pred_path}")


if __name__ == "__main__":
    main()
