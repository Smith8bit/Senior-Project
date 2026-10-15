"""Collect deep-model result files into one spreadsheet: scores, per-seed scores and confusion grids.

  python scripts/summarize_results.py --inputs main_model_v1_news.json main_model_v0_gpu.json --out results_summary.xlsx

Confusion grids: rows = actual (down, flat, up), columns = predicted (down, flat, up), validation block only.
"All seeds" adds the three seeds' grids together, so each validation stock-day is counted three times; the row
percentages are what to read. Also prints a plain-text version for the notebook.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("down", "flat", "up")
H = ("h1", "h5", "h10")
LABEL = {"multiscale_sector": "Multi-Scale PatchTST", "singlescale_sector": "PatchTST one scale",
         "lstm": "LSTM", "patchtst": "PatchTST (baseline)", "multiscale": "Multi-Scale PatchTST, no sector"}


def model_label(name, res):
    return LABEL.get(name, name) + (" + news" if res["settings"].get("news_file") else "")


def per_class(cm):
    cm = np.asarray(cm, float)
    out = {}
    for i, c in enumerate(CLASSES):
        tp, pred, act = cm[i, i], cm[:, i].sum(), cm[i, :].sum()
        p = tp / pred if pred else 0.0
        r = tp / act if act else 0.0
        out[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
                  "actual": int(act), "predicted": int(pred)}
    return out


def collect(paths):
    scores, seeds, conf, floors, meta = [], [], [], [], []
    for path in paths:
        res = json.loads(Path(path).read_text(encoding="utf-8"))
        f = Path(path).name
        meta.append({"file": f, "computed_at": res.get("computed_at"), "git_commit": res.get("git_commit"),
                     "script_sha256": res.get("script_sha256"), "device": res.get("device"), "gpu": res.get("gpu"),
                     "torch": res.get("torch"), "channels": len(res["settings"]["channels"]),
                     "news_file": res["settings"].get("news_file")})
        for h in H:
            for fl, v in res["floors"][h].items():
                if "macro_f1" in v:
                    floors.append({"file": f, "horizon": h, "floor": fl, "macro_f1": v["macro_f1"], "accuracy": v.get("accuracy")})
                if "confusion" in v:
                    for i, c in enumerate(CLASSES):
                        row = v["confusion"][i]
                        conf.append({"file": f, "model": fl, "model_label": "Rule: " + fl.replace("_", " "), "horizon": h,
                                     "seed": "rule", "actual": c, **{f"pred_{k}": row[j] for j, k in enumerate(CLASSES)},
                                     "row_total": sum(row), **{f"pct_{k}": round(100 * row[j] / max(sum(row), 1), 1) for j, k in enumerate(CLASSES)}})
        for name, m in res["models"].items():
            lab, runs = model_label(name, res), m["runs"]
            for r in runs:
                for h in H:
                    v = r["val"][h]
                    seeds.append({"file": f, "model": name, "model_label": lab, "seed": r["seed"], "horizon": h,
                                  "macro_f1": v["macro_f1"], "accuracy": v["accuracy"], "r2_os_vs_zero": v["r2_os_vs_zero"],
                                  "corr": v["corr_with_outcome"], "direction_acc": v.get("direction_acc"),
                                  "best_epoch": r["best_epoch"], "seconds": r["seconds"]})
            for h in H:
                row = {"file": f, "model": name, "model_label": lab, "horizon": h, "seeds": len(runs)}
                for k, src in (("macro_f1", "macro_f1"), ("accuracy", "accuracy"), ("r2_os_vs_zero", "r2_os_vs_zero"), ("corr", "corr_with_outcome")):
                    xs = [r["val"][h][src] for r in runs]
                    row[k + "_mean"] = round(statistics.mean(xs), 4)
                    row[k + "_std"] = round(statistics.stdev(xs), 4) if len(xs) > 1 else None
                scores.append(row)
                grids = [r["val"][h]["confusion"] for r in runs if "confusion" in r["val"][h]]
                if not grids:
                    continue
                for seed, g in [(r["seed"], r["val"][h]["confusion"]) for r in runs] + [("all seeds", np.sum(grids, 0).tolist())]:
                    for i, c in enumerate(CLASSES):
                        rw = g[i]
                        conf.append({"file": f, "model": name, "model_label": lab, "horizon": h, "seed": seed, "actual": c,
                                     **{f"pred_{k}": rw[j] for j, k in enumerate(CLASSES)}, "row_total": sum(rw),
                                     **{f"pct_{k}": round(100 * rw[j] / max(sum(rw), 1), 1) for j, k in enumerate(CLASSES)}})
                pc = per_class(np.sum(grids, 0))
                for c in CLASSES:
                    floors.append({"file": f, "horizon": h, "floor": f"{lab} | class {c}", **pc[c]})
    conf = pd.DataFrame(conf)
    conf = conf[~(conf.seed.astype(str) == "rule") | ~conf.duplicated(subset=["model", "horizon", "actual"])]   # rules repeat per file
    return (pd.DataFrame(meta), pd.DataFrame(scores), pd.DataFrame(seeds), conf,
            pd.DataFrame(floors).drop_duplicates(subset=["horizon", "floor"], keep="first"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    paths = [p for p in a.inputs if Path(p).exists()]
    meta, scores, seeds, conf, floors = collect(paths)
    readme = pd.DataFrame({"note": [
        "Validation block only (2024-03-04 to 2025-01-08). The test block is sealed until Phase 6.",
        "Confusion: rows = actual class, columns = predicted class. 'all seeds' = the three seeds' grids added together.",
        "pct_* columns are row percentages: of the days that were really <actual>, the share the model called down/flat/up.",
        "Floors / per-class sheet: rule floors (macro-F1) and, for each model, precision/recall/F1 per class from the all-seeds grid.",
    ]})
    with pd.ExcelWriter(a.out) as xw:
        readme.to_excel(xw, sheet_name="read_me", index=False)
        scores.to_excel(xw, sheet_name="scores", index=False)
        seeds.to_excel(xw, sheet_name="per_seed", index=False)
        conf.to_excel(xw, sheet_name="confusion", index=False)
        floors.to_excel(xw, sheet_name="floors_and_classes", index=False)
        meta.to_excel(xw, sheet_name="provenance", index=False)
    print(f"wrote {a.out}\n")
    print(scores[["model_label", "horizon", "seeds", "macro_f1_mean", "macro_f1_std", "accuracy_mean"]].to_string(index=False))
    allg = conf[conf.seed == "all seeds"]
    for (f, lab, h), g in allg.groupby(["file", "model_label", "horizon"], sort=False):
        print(f"\n{lab}  {h}  [{f}]  (rows = actual, % of row; all seeds)")
        print(g[["actual", "pct_down", "pct_flat", "pct_up", "row_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
