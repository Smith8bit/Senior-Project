"""E02: seed ensembles of one model. Average class probabilities over a fixed group of seeds, predicted class = most
likely class. Several disjoint groups give several ensemble scores, so the result has a mean +- std like any other run.

Rule fixed before scoring (claude/experiment-ledger.md, E02): 9 seeds of the E03 configuration, groups in seed order
  A = 20261015/16/17 (the E03 run), B = 20261018/19/20, C = 20261021/22/23.
Also reports every single seed on its own, so the ensemble is compared with single models from the same 9 runs.
VALIDATION ONLY. The test block stays sealed.

  python scripts/seed_ensemble.py --preds results/main_model_e03_preds.csv.gz results/main_model_e03_s2_preds.csv.gz \
      --model multiscale_sector --groups 20261015,20261016,20261017 20261018,20261019,20261020 20261021,20261022,20261023 \
      --out results/ensemble_e02.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import labels as L  # noqa: E402
from blend_preds import PASS_BAR, confusion, probs, score, seeds_of, separation  # noqa: E402


def load(files, model):
    """Merge prediction files on (symbol, date); labels must agree; every seed column set must appear once."""
    key = ["symbol", "date"]
    df = None
    for f in files:
        d = pd.read_csv(ROOT / f)
        assert not d.duplicated(key).any(), f"duplicate stock-days in {f}"
        if df is None:
            df = d
            continue
        clash = set(seeds_of(df, model)) & set(seeds_of(d, model))
        assert not clash, f"seed(s) {sorted(clash)} appear in more than one file"
        n = len(df)
        df = df.merge(d, on=key, suffixes=("", "_dup"), validate="one_to_one")
        assert len(df) == n == len(d), f"rows differ: {n} vs {len(d)} in {f}, matched {len(df)}"
        for h in L.HORIZONS:
            assert (df[f"y_{h}"] == df[f"y_{h}_dup"]).all(), f"labels differ at h={h} in {f}"
        df = df.drop(columns=[c for c in df.columns if c.endswith("_dup")])
    return df


def summarize(runs):
    f1 = [r["macro_f1"] for r in runs]
    pooled = np.sum([np.array(r["confusion"]) for r in runs], axis=0)
    return {"macro_f1_mean": round(statistics.mean(f1), 4), "macro_f1_std": round(statistics.stdev(f1), 4) if len(f1) > 1 else None,
            "n": len(f1), "separation_pooled": separation(pooled), "pooled_confusion": pooled.tolist(), "runs": runs}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="one or more <out>_preds.csv.gz files from baselines_deep.py")
    ap.add_argument("--model", default="multiscale_sector")
    ap.add_argument("--groups", nargs="+", required=True, help="comma-separated seed groups, e.g. 1,2,3 4,5,6")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    df = load(args.preds, args.model)
    groups = [[int(s) for s in g.split(",")] for g in args.groups]
    flat = [s for g in groups for s in g]
    assert len(flat) == len(set(flat)), "a seed is in more than one group"
    have = set(seeds_of(df, args.model))
    assert set(flat) <= have, f"seeds missing from the files: {sorted(set(flat) - have)}"

    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "experiment": "E02",
           "rule": "mean of class probabilities over each seed group, argmax", "model": args.model,
           "inputs": args.preds, "groups": groups, "rows": len(df), "evaluated_on": "validation only (test sealed)",
           "pass_bar": PASS_BAR, "horizons": {}}
    for h in L.HORIZONS:
        y = df[f"y_{h}"].to_numpy(int)
        single = [{"seed": s, **score(y, probs(df, args.model, s, h).argmax(1))} for s in flat]
        ens = [{"group": g, **score(y, np.mean([probs(df, args.model, s, h) for s in g], axis=0).argmax(1))} for g in groups]
        out = {"single_seeds": summarize(single), "ensembles": summarize(ens)}
        e = out["ensembles"]["macro_f1_mean"]
        out["ensemble_minus_single"] = round(e - out["single_seeds"]["macro_f1_mean"], 4)
        out["vs_pass_bar"] = {"bar": PASS_BAR[f"h{h}"], "ensemble": e, "gap": round(e - PASS_BAR[f"h{h}"], 4), "passes": e >= PASS_BAR[f"h{h}"]}
        res["horizons"][f"h{h}"] = out
        print(f"h={h}: single seeds {out['single_seeds']['macro_f1_mean']} +- {out['single_seeds']['macro_f1_std']} (n={len(single)})  "
              f"ensembles {e} +- {out['ensembles']['macro_f1_std']} (n={len(ens)})  diff {out['ensemble_minus_single']:+.4f}  "
              f"bar {PASS_BAR[f'h{h}']} gap {out['vs_pass_bar']['gap']:+.4f}  separation {out['ensembles']['separation_pooled']}")
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
