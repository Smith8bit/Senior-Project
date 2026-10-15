"""E02b: blend the main model's validation probabilities with XGBoost's, one seed pair at a time.

Rule fixed before any blend was scored (claude/experiment-ledger.md, E02b):
  p_blend = w * p_deep + (1 - w) * p_xgb, with w = 0.5 (not tuned), predicted class = most likely class.
  Seed pairs: deep seed s is blended with XGBoost seed s, giving 3 scores per horizon (mean +- std, as the pass bar asks).
Also re-scores each input alone from the same files, so the blend is compared like for like.
VALIDATION ONLY. The test block stays sealed.

  python scripts/blend_preds.py --deep results/main_model_v1_news_p41_preds.csv.gz --deep-model multiscale_sector \
      --xgb results/xgb_e02b_preds.csv.gz --out results/blend_e02b.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import labels as L  # noqa: E402
import metrics as M  # noqa: E402

PASS_BAR = {"h1": 0.393, "h5": 0.400, "h10": 0.407}      # claude/pre-registration-amendment-2026-10-16-success-bands.md


def confusion(y, yhat):
    g = np.zeros((3, 3), int)
    np.add.at(g, (y, yhat), 1)
    return g


def separation(grid):
    """Per class: share called X on days that really were X, minus share called X overall (points)."""
    g = np.asarray(grid, float)
    called = g.sum(0) / g.sum()
    return {c: round(float(100 * (g[i, i] / g[i].sum() - called[i])), 1) for i, c in enumerate(L.CLASSES)}


def probs(df, prefix, seed, h):
    return df[[f"{prefix}_s{seed}_p{h}_{c}" for c in L.CLASSES]].to_numpy(float)


def seeds_of(df, prefix):
    return sorted({int(m.group(1)) for c in df.columns if (m := re.fullmatch(rf"{re.escape(prefix)}_s(\d+)_p1_down", c))})


def score(y, yhat):
    return M.trend_metrics(y, yhat) | {"confusion": confusion(y, yhat).tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", required=True, help="deep model predictions file (<out>_preds.csv.gz from baselines_deep.py)")
    ap.add_argument("--deep-model", default="multiscale_sector")
    ap.add_argument("--xgb", required=True, help="XGBoost predictions file (<out>_preds.csv.gz from baselines_tabular.py)")
    ap.add_argument("--weight", type=float, default=0.5, help="weight on the deep model (fixed at 0.5 for E02b)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    deep, xgb = pd.read_csv(ROOT / args.deep), pd.read_csv(ROOT / args.xgb)
    key = ["symbol", "date"]
    assert not deep.duplicated(key).any() and not xgb.duplicated(key).any(), "duplicate stock-days"
    df = deep.merge(xgb, on=key, suffixes=("", "_x"), validate="one_to_one")
    assert len(df) == len(deep) == len(xgb), f"rows differ: deep {len(deep)}, xgb {len(xgb)}, matched {len(df)}"
    for h in L.HORIZONS:
        assert (df[f"y_{h}"] == df[f"y_{h}_x"]).all(), f"labels differ between files at h={h}"

    sd, sx = seeds_of(df, args.deep_model), seeds_of(df, "xgb")
    assert sd == sx and sd, f"seed lists differ: deep {sd}, xgb {sx}"
    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "experiment": "E02b",
           "rule": f"p = {args.weight} * deep + {1 - args.weight} * xgb, argmax; seed s with seed s",
           "inputs": {"deep": args.deep, "deep_model": args.deep_model, "xgb": args.xgb}, "rows": len(df), "seeds": sd,
           "evaluated_on": "validation only (test sealed)", "pass_bar": PASS_BAR, "horizons": {}}
    for h in L.HORIZONS:
        y = df[f"y_{h}"].to_numpy(int)
        out = {}
        for name in ("deep_alone", "xgb_alone", "blend"):
            runs, pooled = [], np.zeros((3, 3), int)
            for s in sd:
                pd_, px = probs(df, args.deep_model, s, h), probs(df, "xgb", s, h)
                p = {"deep_alone": pd_, "xgb_alone": px, "blend": args.weight * pd_ + (1 - args.weight) * px}[name]
                r = score(y, p.argmax(1)); runs.append({"seed": s, **r}); pooled += np.array(r["confusion"])
            f1 = [r["macro_f1"] for r in runs]
            out[name] = {"macro_f1_mean": round(statistics.mean(f1), 4), "macro_f1_std": round(statistics.stdev(f1), 4) if len(f1) > 1 else None,
                         "separation_pooled": separation(pooled), "pooled_confusion": pooled.tolist(), "runs": runs}
        b = out["blend"]["macro_f1_mean"]
        out["vs_pass_bar"] = {"bar": PASS_BAR[f"h{h}"], "blend": b, "gap": round(b - PASS_BAR[f"h{h}"], 4), "passes": b >= PASS_BAR[f"h{h}"]}
        res["horizons"][f"h{h}"] = out
        print(f"h={h}: deep {out['deep_alone']['macro_f1_mean']}  xgb {out['xgb_alone']['macro_f1_mean']}  "
              f"blend {b} +- {out['blend']['macro_f1_std']}  (bar {PASS_BAR[f'h{h}']}, gap {out['vs_pass_bar']['gap']:+.4f})  "
              f"separation {out['blend']['separation_pooled']}")
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
