"""E29: LightGBM on an Alpha158-style factor set (SET port) plus per-day cross-sectional ranks; XGBoost on the same
inputs as the check (claude/batches-4-7-design-2026-10-16.md, fixed before any E29 result).

Inputs (343): the 29 inputs of XGBoost + E03, the 157 factors of src/alpha158.py, and their 157 cross-sectional ranks.
Learners (fixed, untuned):
  lgbm : LightGBM, 500 trees, learning rate 0.03, 31 leaves, at least 100 rows per leaf, 80% rows / 50% inputs per
         tree, L2 1.0. Qlib's LightGBM benchmark uses similar sizes. [Uncertainty Flag: chosen, not searched]
  xgbf : XGBoost with the project's fixed settings (scripts/baselines_tabular.py XGB), on the same 343 inputs. Separates
         "more inputs" from "LightGBM instead of XGBoost". (The plan named CatBoost as the check; XGBoost is used
         instead because it isolates the input effect and needs no new library on Colab.)
Same rows, labels and splits as XGBoost; one process runs every window, so the factors are computed once.
VALIDATION ONLY (walk-forward windows sit inside train + validation). The test block stays sealed.

  python scripts/factors_tabular.py --out-dir results/batch4/e29 --window official \
      --window 2022H2 2022-07-01 2022-12-31 ...
Writes <out-dir>/<window>_factors.json and <window>_factors_preds.csv.gz (columns lgbm_s<seed>_p<h>_<class>, xgbf_...).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import alpha158 as A  # noqa: E402
import baselines_tabular as BT  # noqa: E402
import features as F  # noqa: E402
import labels as L  # noqa: E402
import metrics as M  # noqa: E402
import prices as P  # noqa: E402

LGBM = dict(objective="multiclass", n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_jobs=-1, verbose=-1)
LEARNERS = ("lgbm", "xgbf")


def make(learner, seed):
    if learner == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, **LGBM)
    from xgboost import XGBClassifier
    return XGBClassifier(objective="multi:softprob", num_class=3, random_state=seed, **BT.XGB)


def run_window(label, fold, fac, learners, seeds, out_dir):
    data, info = BT.build_dataset(extra="e03", fold=fold)
    assert data.index.isin(fac.index).all()
    data = data.join(fac)
    feats = F.FEATURES + BT.EXTRA["e03"] + A.FACTORS + A.RANKED
    tr, va = data[data.split == "train"], data[data.split == "val"]
    assert not (data.split == "test").loc[va.index].any()
    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": BT.git_commit(),
           "experiment": "E29", "window": label, "evaluated_on": "validation only (test sealed)", "n_inputs": len(feats),
           "learner_params": {"lgbm": LGBM, "xgbf": BT.XGB}, "seeds": list(seeds), "data": info, "results": {}}
    preds = va[["symbol", "date"]].reset_index(drop=True).copy()
    for h in L.HORIZONS:
        ytr, yva = tr[f"y_{h}"].to_numpy(), va[f"y_{h}"].to_numpy()
        preds[f"y_{h}"] = yva; preds[f"fwd_{h}"] = va[f"fwd_{h}"].to_numpy()
        past = L.classify(va[f"r_{h}"].to_numpy(), va[f"sigma_{h}"].to_numpy(), info[f"h{h}"]["k"])
        floors = {"trend_continues": M.trend_metrics(yva, past), "trend_reverses": M.trend_metrics(yva, 2 - past)}
        model = {}
        for learner in learners:
            runs = []
            for seed in seeds:
                t0 = time.time()
                clf = make(learner, seed).fit(tr[feats], ytr)
                prob = clf.predict_proba(va[feats])
                yhat = prob.argmax(1)
                for c, cls in enumerate(L.CLASSES):
                    preds[f"{learner}_s{seed}_p{h}_{cls}"] = prob[:, c].round(6)
                runs.append({"seed": seed, **M.trend_metrics(yva, yhat), "confusion": BT.confusion(yva, yhat),
                             "seconds": round(time.time() - t0, 1)})
                print(f"  {label} h{h} {learner} seed {seed}: macro-F1 {runs[-1]['macro_f1']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            f1 = [r["macro_f1"] for r in runs]
            model[f"{learner}_trend"] = {"macro_f1": {"mean": round(statistics.mean(f1), 5),
                                                      "std": round(statistics.stdev(f1), 5) if len(f1) > 1 else None},
                                         "per_seed": runs}
        res["results"][f"h{h}"] = M.report(model, floors)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}_factors.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    preds.to_csv(out_dir / f"{label}_factors_preds.csv.gz", index=False)
    print(f"wrote {out_dir / f'{label}_factors.json'}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", nargs="+", action="append", required=True, metavar="LABEL [START END]",
                    help="'official' or 'LABEL START END' for a walk-forward window; repeatable")
    ap.add_argument("--learners", nargs="+", default=list(LEARNERS), choices=LEARNERS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(BT.SEEDS))
    ap.add_argument("--skip-done", action="store_true", help="skip windows whose JSON already exists")
    args = ap.parse_args(argv)
    out_dir = ROOT / args.out_dir
    t0 = time.time()
    fac = A.add_factors(P.load_prices(ROOT))[A.FACTORS + A.RANKED]
    print(f"factors: {fac.shape[1]} columns built in {time.time() - t0:.0f}s", flush=True)
    for w in args.window:
        label = w[0]
        if args.skip_done and (out_dir / f"{label}_factors.json").exists():
            print(f"=== {label}: already done"); continue
        fold = None if label == "official" else {"eval_start": w[1], "eval_end": w[2]}
        print(f"=== E29 window {label}", flush=True)
        run_window(label, fold, fac, args.learners, args.seeds, out_dir)


if __name__ == "__main__":
    main()
