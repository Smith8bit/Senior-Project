"""Batch 7: assemble the final configuration from batches 5-6 by rules fixed before any of their results
(claude/batches-4-7-design-2026-10-16.md). Every step writes its decision to one JSON file, so the notebook can run
batch 7 unattended and the record shows why each change is in or out.

  python scripts/select_final.py select  --summary B5.json B6.json --out FINAL.json
  python scripts/select_final.py stacked --config FINAL.json --summary STACK_SUMMARY.json
  python scripts/select_final.py e13     --config FINAL.json --summary E13_SUMMARY.json

Rules
  * Kept: counted gain at >= 1 horizon (run - E21 > E21 seed std) and no counted loss (run - E21 < -E21 std).
  * Conflicts (gain g = mean over the 3 horizons of run - E21):
      patch scales: E15 (one scale) is used if kept and its g beats every other kept patch change; otherwise the short
      scale is 2,1 if E07 is kept, and the long scale is the kept one of E08 / E09 with the larger g;
      look-back: E27 and E06 together -> the one with the larger g.
  * 0 kept -> final = E21 (nothing rerun). 1 kept -> final = that run. 2+ kept -> the stacked run is trained; final = the
    best mean score among the stacked run (only if it has no counted loss against E21) and the kept single runs.
  * E13 (older rows at loss weight 0.5 before 2022-01-01) on the final configuration: kept if counted gain and no
    counted loss against the final configuration's own run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HS = ("h1", "h5", "h10")

E21 = ["--models", "multiscale_sector", "--news", "data/derived/sentiment_scores.csv", "--extra-features", "e03",
       "--lr", "1e-4", "--dropout", "0.2", "--max-epochs", "30", "--patience", "10", "--cls-only", "--stop-rule", "every",
       "--no-revin"]
CHANGES = {           # option -> value, applied on top of E21
    "E26": {"--extra-features": "e03_e26"},
    "E27": {"--lookback": "30"},
    "E28": {"--chan-mask": "0.15"},
    "E06": {"--context-min": "20"},
    "E07": {"--patches": "2,1;20,10"},
    "E08": {"--patches": "5,5;20,5"},
    "E09": {"--patches": "5,5;10,5"},
    "E15": {"--patches": "16,8"},
    "E17": {"--d-channel": "16", "--trunk-width": "128"},
    "E13": {"--recent-weight": "0.5", "--recent-from": "2022-01-01"},
}
PATCH = ("E07", "E08", "E09", "E15")


def with_options(base, opts):
    args = list(base)
    for k, v in opts.items():
        if k in args:
            args[args.index(k) + 1] = v
        else:
            args += [k, v]
    return args


def args_for(labels, patches=None):
    opts = {}
    for lb in labels:
        if lb not in PATCH:
            opts.update(CHANGES[lb])
    if patches:
        opts["--patches"] = patches
    return with_options(E21, opts)


def rows_of(paths):
    rows = {}
    for p in paths:
        for r in json.loads((ROOT / p).read_text(encoding="utf-8"))["rows"]:
            rows[r["label"]] = r
    return rows


def verdict(row):
    gain = [row["h"][h]["minus_control"] for h in HS]
    std = [row["h"][h]["control_std"] or 0 for h in HS]
    counted_gain = [g > s for g, s in zip(gain, std)]
    counted_loss = [g < -s for g, s in zip(gain, std)]
    return {"gain": gain, "mean_gain": round(sum(gain) / 3, 4), "counted_gain": counted_gain, "counted_loss": counted_loss,
            "kept": any(counted_gain) and not any(counted_loss)}


def select(args):
    rows = rows_of(args.summary)
    v = {lb: verdict(r) for lb, r in rows.items() if lb in CHANGES}
    kept = [lb for lb in v if v[lb]["kept"]]
    notes, drop = [], []
    g = {lb: v[lb]["mean_gain"] for lb in kept}
    patches = None
    kp = [lb for lb in kept if lb in PATCH]
    if kp:
        if "E15" in kp and all(g["E15"] > g[o] for o in kp if o != "E15"):
            use = ["E15"]; patches = CHANGES["E15"]["--patches"]
        else:
            short = "2,1" if "E07" in kp else "5,5"
            longs = [o for o in ("E08", "E09") if o in kp]
            lg = max(longs, key=lambda o: g[o]) if longs else None
            long_ = {"E08": "20,5", "E09": "10,5", None: "20,10"}[lg]
            use = [o for o in ("E07", lg) if o and o in kp]
            patches = f"{short};{long_}"
        drop += [o for o in kp if o not in use]
        notes.append(f"patch scales: kept {kp}, used {use} -> --patches {patches}")
    if "E27" in kept and "E06" in kept:
        loser = "E06" if g["E27"] >= g["E06"] else "E27"
        drop.append(loser); notes.append(f"look-back conflict: E27 vs E06 -> drop {loser}")
    use = [lb for lb in kept if lb not in drop]
    cfg = {"verdicts": v, "kept": kept, "dropped_by_conflict": drop, "used": use, "notes": notes,
           "stacked_args": args_for(use, patches) if len(use) >= 2 else None}
    if not use:
        cfg.update(final_label="E21", final_args=E21, final_run=None, needs_stacked_run=False)
    elif len(use) == 1:
        cfg.update(final_label=use[0], final_args=args_for(use, patches), final_run=rows[use[0]]["file"], needs_stacked_run=False)
    else:
        cfg.update(final_label=None, final_args=None, final_run=None, needs_stacked_run=True)
    save(cfg, args.out)
    print(json.dumps({k: cfg[k] for k in ("kept", "dropped_by_conflict", "used", "notes", "final_label", "needs_stacked_run")}, indent=1))


def stacked(args):
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    rows = rows_of(args.summary)          # control = E21; runs: STACK and the kept single runs
    mean = {lb: sum(r["h"][h]["mean"] for h in HS) / 3 for lb, r in rows.items()}
    cands = [lb for lb in cfg["used"] if lb in rows]
    st = verdict(rows["STACK"])
    if not any(st["counted_loss"]):
        cands.append("STACK")
    best = max(cands, key=lambda lb: mean[lb])
    cfg["stacked_verdict"] = st
    cfg["stack_choice"] = {"candidates": {lb: round(mean[lb], 4) for lb in cands}, "chosen": best}
    cfg["final_label"] = "STACK" if best == "STACK" else best
    cfg["final_args"] = cfg["stacked_args"] if best == "STACK" else args_for([best], CHANGES[best].get("--patches"))
    cfg["final_run"] = rows[best]["file"]
    save(cfg, args.config)
    print("final after stacking:", cfg["final_label"], cfg["stack_choice"])


def e13(args):
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    rows = rows_of(args.summary)          # control = the final configuration's run; run: E13
    v = verdict(rows["E13"])
    cfg["e13_verdict"] = v
    if v["kept"]:
        cfg["final_label"] = f"{cfg['final_label']}+E13"
        cfg["final_args"] = with_options(cfg["final_args"], CHANGES["E13"])
        cfg["final_run"] = rows["E13"]["file"]
    save(cfg, args.config)
    print("E13", "kept" if v["kept"] else "not kept", v, "-> final", cfg["final_label"])


def save(cfg, path):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "stacked", "e13"])
    ap.add_argument("--summary", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    {"select": select, "stacked": stacked, "e13": e13}[args.step](args)


if __name__ == "__main__":
    main()
