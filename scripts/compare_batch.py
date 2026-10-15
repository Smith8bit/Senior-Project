"""Phase 4 batch summary: each run against the same-session control and against the pass bar.

  python scripts/compare_batch.py --control results/b1/control.json --reference results/main_model_e03.json \
      --run E10 results/b1/e10.json --run E11 results/b1/e11.json ... --out results/b1/batch_summary.json

For every run (and the control): mean +- std macro-F1 per horizon, difference from the control, and the per-horizon
best-epoch score (E12). With --reference, the control's per-seed scores are compared with the earlier run of the same
configuration (run-to-run reproducibility on the GPU). Validation only; the test block stays sealed.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HS = ("h1", "h5", "h10")
PASS_BAR = {"h1": 0.393, "h5": 0.400, "h10": 0.407}


def load(path):
    r = json.loads((ROOT / path).read_text(encoding="utf-8"))
    (name, m), = r["models"].items()
    return r, name, m["runs"]


def stats(xs):
    return round(statistics.mean(xs), 4), (round(statistics.stdev(xs), 4) if len(xs) > 1 else None)


def describe(label, path):
    r, name, runs = load(path)
    out = {"label": label, "file": str(path), "model": name, "options": r["settings"].get("options"),
           "git_commit": r.get("git_commit"), "script_sha256": r.get("script_sha256"), "params": runs[0]["params"],
           "seeds": [x["seed"] for x in runs], "best_epochs": [x["best_epoch"] for x in runs],
           "best_epoch_per_horizon": [x.get("best_epoch_per_horizon") for x in runs], "h": {}}
    for h in HS:
        m, s = stats([x["val"][h]["macro_f1"] for x in runs])
        ph = [x["val_per_horizon_best"][h]["macro_f1"] for x in runs if "val_per_horizon_best" in x]
        pm, ps = stats(ph) if ph else (None, None)
        out["h"][h] = {"mean": m, "std": s, "per_horizon_best_mean": pm, "per_horizon_best_std": ps}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--run", nargs=2, action="append", default=[], metavar=("LABEL", "JSON"))
    ap.add_argument("--reference", default=None, help="earlier run of the control configuration (reproducibility check)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    ctrl = describe("control", args.control)
    rows = [ctrl] + [describe(lbl, p) for lbl, p in args.run]
    for r in rows:
        for h in HS:
            v = r["h"][h]
            v["minus_control"] = round(v["mean"] - ctrl["h"][h]["mean"], 4)
            v["control_std"] = ctrl["h"][h]["std"]
            v["counted_gain"] = r is not ctrl and v["minus_control"] > (ctrl["h"][h]["std"] or 0)
            v["gap_to_bar"] = round(v["mean"] - PASS_BAR[h], 4)
            v["passes_bar"] = v["mean"] >= PASS_BAR[h]

    print(f"{'run':10s} {'h':4s} {'macro-F1':>15s} {'vs control':>11s} {'per-horizon best (E12)':>23s} {'gap to bar':>11s}")
    for r in rows:
        for h in HS:
            v = r["h"][h]
            ph = f"{v['per_horizon_best_mean']:.3f} ± {v['per_horizon_best_std'] or 0:.3f}" if v["per_horizon_best_mean"] is not None else "n/a"
            print(f"{r['label']:10s} {h:4s} {v['mean']:.3f} ± {v['std'] or 0:.3f} {v['minus_control']:+11.3f} {ph:>23s} {v['gap_to_bar']:+11.3f}"
                  + ("  COUNTED GAIN" if v["counted_gain"] else "") + ("  PASSES BAR" if v["passes_bar"] else ""))

    repro = None
    if args.reference:
        _, _, ref = load(args.reference)
        _, _, cr = load(args.control)
        pairs = {x["seed"]: x for x in ref}
        diffs = [round(c["val"][h]["macro_f1"] - pairs[c["seed"]]["val"][h]["macro_f1"], 4) for c in cr if c["seed"] in pairs for h in HS]
        repro = {"max_abs_diff": max(abs(d) for d in diffs) if diffs else None, "diffs": diffs}
        print(f"\nreproducibility (control vs earlier run, same seeds): max |difference| {repro['max_abs_diff']}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "pass_bar": PASS_BAR, "reproducibility": repro,
                               "evaluated_on": "validation only (test sealed)"}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
