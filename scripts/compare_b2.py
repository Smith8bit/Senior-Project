"""v2.08 (B2 "true data" label) against its pre-registered baselines, validation only (test sealed).

  python scripts/compare_b2.py --run results/v2xx/A100/v2_08.json --out results/v2_08_vs_b2_baselines.json

Adopt rule (Siwat, 2026-11-03, "beat XGBoost and random based on imbalance bias"; claude/log-entry-2026-11-03-v2-08-design.md),
raw model, validation, macro-F1 over the 2 classes (seed mean), paired 10-day date-block bootstrap (2,000 resamples,
rng 20261016, the same resamples for every comparison):
  (a) the 95% interval of v2.08 - XGBoost (B2) is entirely above 0;
  (b) the 95% interval of v2.08 - random (B2 imbalance) is entirely above 0.
Both -> adopted (as the B2 next-day model).

Baselines, all on exactly v2.08's validation rows and B2 labels:
  * XGBoost (B2): the tabular inputs used beside every v2.xx run (src/features.py + the E03 inputs, the fixed untuned
    settings in scripts/baselines_tabular.XGB), trained on v2.08's fit rows only (the inner hold-out and its purge are
    left out, as for the deep model), objective binary:logistic, one model per v2.08 seed (random_state = seed);
    call = not down if P(not down) >= 0.5.
  * random (B2 imbalance): calls drawn at random with P(not down) = the share of "not down" on the training rows,
    independent of the row; 100 draws (rng 20261103), averaged like seeds. Its expected macro-F1 is 0.5.
  * floors (reported only): majority class, direction continues, direction reverses (the B2 rule on the move t-1 -> t).
Reported beside it: accuracy (with the majority-class accuracy), the seed ensemble, per-class F1 and call shares.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import baselines_tabular as BT  # noqa: E402
import features as F  # noqa: E402
import labels as L  # noqa: E402
import posthoc as PH  # noqa: E402

RANDOM_DRAWS, RANDOM_RNG = 100, 20261103
CLS = L.B2_CLASSES


def per_class(y, C):
    """mean over rows of C [S, n]: per-class F1, accuracy and call shares."""
    f = np.mean([f1_score(y, c, labels=[0, 1], average=None, zero_division=0) for c in C], axis=0)
    return {"f1_per_class": dict(zip(CLS, np.round(f, 4).tolist())), "accuracy": round(float(np.mean([(c == y).mean() for c in C])), 4),
            "call_shares": dict(zip(CLS, np.round([float((C == k).mean()) for k in (0, 1)], 4).tolist()))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="v2.08 run JSON (its _preds.csv.gz must sit beside it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="plumbing check on a --smoke run: 20 trees, no row-count checks")
    a = ap.parse_args(argv)
    run_path = Path(a.run)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    cfg = run["config"]
    if cfg["protocol"].get("label_rule") != "b2_raw_close" or cfg["protocol"]["horizons"] != [1]:
        raise SystemExit("stop: this comparison is for a B2 (label_rule b2_raw_close, horizons [1]) run")
    if run.get("smoke") and not a.smoke:
        raise SystemExit("stop: this is a --smoke run of train_v2xx.py (pass --smoke for a plumbing check)")
    seeds = [r["seed"] for r in run["runs"]]
    preds = pd.read_csv(run_path.with_name(run_path.stem + "_preds.csv.gz"))
    preds["date"] = pd.to_datetime(preds["date"].astype(str).str[:10])

    # rows and labels exactly as training built them
    data, info, full = BT.build_dataset(return_full=True, extra="e03")
    data = data.copy()
    data["y_b2"], data["b2_prev"] = L.b2_next_day(full, data.index)
    tr = data[data.split == "train"]
    ie = run["inner_es"]
    fit = tr[tr.date <= pd.Timestamp(ie["fit_end"])]
    va = data[data.split == "val"]
    m = preds[["symbol", "date", "y_1"]].merge(va[["symbol", "date", "y_b2", "b2_prev"] + F.FEATURES + BT.EXTRA["e03"]],
                                               on=["symbol", "date"], how="left", validate="one_to_one")
    assert m["y_b2"].notna().all(), "validation rows of the run are missing from the rebuilt dataset"
    assert (m["y_1"].to_numpy() == m["y_b2"].to_numpy()).all(), "the run's labels differ from the B2 labels rebuilt here"
    if not a.smoke:
        assert len(fit) == ie["rows_fit"] and len(m) == len(va) == run["data"]["rows_by_split"]["val"], \
            "fit / validation rows differ from the run"
    y = m["y_b2"].to_numpy(int)
    dates = m["date"].to_numpy()
    q = float(run["label"]["train_class_shares"][1])                       # share of "not down" on the training rows

    calls = {"v2.08": np.stack([preds[f"v2xx_s{s}_class1"].to_numpy(int) for s in seeds])}
    if "v2xx_ens_class1" in preds:
        calls["v2.08_seed_ensemble"] = preds["v2xx_ens_class1"].to_numpy(int)[None, :]
    from xgboost import XGBClassifier
    import xgboost
    feats = F.FEATURES + BT.EXTRA["e03"]
    params = dict(BT.XGB) | ({"n_estimators": 20} if a.smoke else {})
    xp = []
    for s in seeds:
        clf = XGBClassifier(objective="binary:logistic", random_state=s, **params)
        clf.fit(fit[feats], fit["y_b2"].to_numpy(int))
        xp.append(clf.predict_proba(m[feats])[:, 1])
        print(f"  XGBoost (B2) seed {s}: trained on {len(fit)} fit rows", flush=True)
    xp = np.stack(xp)
    calls["xgboost_b2"] = (xp >= 0.5).astype(int)
    rng = np.random.default_rng(RANDOM_RNG)
    calls["random_b2_imbalance"] = (rng.random((RANDOM_DRAWS, len(y))) < q).astype(int)
    maj = int(np.argmax(np.bincount(tr["y_b2"].to_numpy(int), minlength=2)))
    calls["majority_class"] = np.full((1, len(y)), maj)
    prev = m["b2_prev"].to_numpy(int)
    cont = np.where(prev < 0, maj, prev)
    calls["direction_continues"] = cont[None, :]
    calls["direction_reverses"] = (1 - cont)[None, :]

    pairs = [("v2.08", "xgboost_b2"), ("v2.08", "random_b2_imbalance"), ("v2.08", "majority_class"),
             ("v2.08", "direction_continues"), ("v2.08", "direction_reverses"), ("xgboost_b2", "random_b2_imbalance")]
    boot = PH.block_bootstrap(dates, y, calls, pairs=pairs, n_classes=2)
    d_x, d_r = boot["differences"]["v2.08 - xgboost_b2"], boot["differences"]["v2.08 - random_b2_imbalance"]
    a1, a2 = d_x["ci95"][0] > 0, d_r["ci95"][0] > 0
    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": cfg["name"], "seeds": seeds,
           "run_git_commit": run.get("git_commit"), "rows_val": int(len(y)), "rows_fit_xgboost": int(len(fit)),
           "label": "B2: 0 = down if raw close(t+1) < raw close(t), else 1 = not down (claude/decision-2026-11-03-binary-label-b2.md)",
           "val_class_shares": dict(zip(CLS, np.round(np.bincount(y, minlength=2) / len(y), 4).tolist())),
           "train_not_down_share": round(q, 4), "random_draws": RANDOM_DRAWS, "random_rng": RANDOM_RNG,
           "xgboost": {"version": xgboost.__version__, "params": params, "features": len(feats), "objective": "binary:logistic",
                       "trained_on": "v2.08 fit rows only"},
           "rule": __doc__.split("Adopt rule ")[1].split("\nBaselines")[0].strip(),
           "macro_f1": boot["values"], "differences": boot["differences"],
           "details": {k: per_class(y, C) for k, C in calls.items()},
           "adopt": {"adopted": bool(a1 and a2), "a_above_xgboost": bool(a1), "b_above_random": bool(a2)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    xo = m[["symbol", "date"]].copy(); xo["date"] = xo.date.dt.strftime("%Y-%m-%d"); xo["y_b2"] = y
    for s, p in zip(seeds, xp):
        xo[f"xgb_s{s}_p1_not_down"] = p.round(6)
    xo.to_csv(Path(a.out).with_name(Path(a.out).stem + "_xgboost_preds.csv.gz"), index=False)

    print(f"\n{cfg['name']} on validation ({len(y)} rows; actual down / not down {res['val_class_shares']['down']:.1%} / "
          f"{res['val_class_shares']['not_down']:.1%}; train not-down share {q:.1%})")
    print(f"{'':24s}{'macro-F1':>9s}  {'95% interval':16s}{'accuracy':>9s}{'F1 down':>9s}{'F1 n-down':>10s}{'calls down':>11s}")
    for k in calls:
        v, dt = boot["values"][k], res["details"][k]
        print(f"{k:24s}{v['point']:9.4f}  {str(v['ci95']):16s}{dt['accuracy']:9.4f}{dt['f1_per_class']['down']:9.4f}"
              f"{dt['f1_per_class']['not_down']:10.4f}{dt['call_shares']['down']:11.1%}")
    print("\npaired differences (macro-F1, 95% interval):")
    for k, d in boot["differences"].items():
        print(f"  {k:42s}{d['point']:+.4f} {d['ci95']}")
    print(f"\nADOPT RULE: (a) above XGBoost (B2): interval {d_x['ci95']} entirely above 0: {a1}; "
          f"(b) above random (B2 imbalance): interval {d_r['ci95']} entirely above 0: {a2} -> "
          f"{'ADOPTED' if a1 and a2 else 'not adopted'}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
