"""E05: walk-forward summary. For each fold and horizon, the main model (mean +- std over seeds) against XGBoost, the
one-rule floors and random-by-class-shares, all recomputed on that fold's scored window with labels refitted on that
fold's training rows.

Per-fold bar (claude/pre-registration-amendment-2026-10-16-success-bands.md, confirmation rule 3):
  bar = max(best one-rule floor, XGBoost) + 0.02, computed per window.
Uses train + validation data only. The test block stays sealed.

  python scripts/summarize_walk_forward.py --model multiscale_sector \
      --fold 2022H2 results/e05/f1_deep.json results/e05/f1_xgb.json ... --out results/e05/summary.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HS = ("h1", "h5", "h10")
MARGIN = 0.02


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", nargs=3, action="append", required=True, metavar=("LABEL", "DEEP_JSON", "XGB_JSON"))
    ap.add_argument("--model", default="multiscale_sector")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-horizon-best", action="store_true", help="score the main model with each horizon's own best epoch (E12)")
    ap.add_argument("--experiment", default="E05")
    args = ap.parse_args(argv)
    key = "val_per_horizon_best" if args.per_horizon_best else "val"

    rows = []
    for label, dp, xp in args.fold:
        d, x = json.loads((ROOT / dp).read_text(encoding="utf-8")), json.loads((ROOT / xp).read_text(encoding="utf-8"))
        assert d["data"]["blocks"].get("fold") == x["data"]["blocks"].get("fold"), f"{label}: deep and XGBoost files are from different folds"
        assert d["data"]["rows_by_split"] == x["data"]["rows_by_split"], f"{label}: deep and XGBoost files use different rows"
        runs = d["models"][args.model]["runs"]
        fold = d["data"]["blocks"].get("fold") or {"eval_start": d["data"]["blocks"]["val"][0], "eval_end": d["data"]["blocks"]["val"][1]}
        row = {"fold": label, **fold, "rows_train": d["data"]["rows_by_split"]["train"], "rows_scored": d["data"]["rows_by_split"]["val"],
               "seeds": [r["seed"] for r in runs], "best_epochs": [r["best_epoch"] for r in runs],
               "best_inner_es": [r["best_inner_es_macro_f1"] for r in runs], "h": {}}
        for h in HS:
            if any(key not in r for r in runs):
                raise SystemExit(f"{label}: {dp} has no {key} scores")
            f1 = [r[key][h]["macro_f1"] for r in runs]
            fl = d["floors"][h]
            cont, rev = fl["trend_continues"]["macro_f1"], fl["trend_reverses"]["macro_f1"]
            rnd = fl["stratified_random_expected"]["macro_f1"]
            xgb = x["results"][h]["model"]["xgb_trend"]["macro_f1"]["mean"]
            best_floor = max(cont, rev)
            bar = round(max(best_floor, xgb) + MARGIN, 4)
            m = statistics.mean(f1)
            row["h"][h] = {"model_mean": round(m, 4), "model_std": round(statistics.stdev(f1), 4) if len(f1) > 1 else None,
                           "xgb": round(xgb, 4), "trend_continues": cont, "trend_reverses": rev, "random": rnd,
                           "model_minus_random": round(m - rnd, 4), "model_minus_best_floor": round(m - best_floor, 4),
                           "model_minus_xgb": round(m - xgb, 4), "fold_bar": bar, "passes_fold_bar": m >= bar,
                           "k": d["data"][h]["k"]}
        rows.append(row)

    print(f"{'fold':8s} {'h':4s} {'model':>13s} {'XGB':>6s} {'cont':>6s} {'rev':>6s} {'random':>6s} {'m-floor':>8s} {'m-XGB':>7s} {'bar':>6s}")
    for r in rows:
        for h in HS:
            v = r["h"][h]
            print(f"{r['fold']:8s} {h:4s} {v['model_mean']:.3f} ± {v['model_std'] or 0:.3f} {v['xgb']:6.3f} {v['trend_continues']:6.3f} "
                  f"{v['trend_reverses']:6.3f} {v['random']:6.3f} {v['model_minus_best_floor']:+8.3f} {v['model_minus_xgb']:+7.3f} {v['fold_bar']:6.3f}"
                  + ("  PASS" if v["passes_fold_bar"] else ""))
    summary = {h: {"model_mean_over_folds": round(statistics.mean(r["h"][h]["model_mean"] for r in rows), 4),
                   "xgb_mean_over_folds": round(statistics.mean(r["h"][h]["xgb"] for r in rows), 4),
                   "folds_model_beats_random": sum(r["h"][h]["model_minus_random"] > 0 for r in rows),
                   "folds_model_beats_best_floor": sum(r["h"][h]["model_minus_best_floor"] > 0 for r in rows),
                   "folds_model_beats_xgb": sum(r["h"][h]["model_minus_xgb"] > 0 for r in rows),
                   "folds_pass_bar": sum(r["h"][h]["passes_fold_bar"] for r in rows), "folds": len(rows)} for h in HS}
    print("\nacross folds:", json.dumps(summary))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"experiment": args.experiment, "model": args.model, "scored_on": key, "margin": MARGIN,
                               "folds": rows, "summary": summary,
                               "evaluated_on": "walk-forward windows inside train + validation; test sealed"}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
