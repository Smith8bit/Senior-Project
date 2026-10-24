"""Merge per-seed parts of a parallel v2.xx run into the usual single result (Siwat, 2026-10-25: three seeds in parallel).

  python scripts/merge_v2xx.py --config configs/v2_03.json --parts DIR --out results/v2xx/A100/v2_03.json

Each part is `scripts/train_v2xx.py --part --seeds S --out DIR/seedS.json`. Every seed trains exactly as in a
sequential run (train_one re-seeds torch, numpy and its own random streams from the seed), so merging changes nothing
about training; it only collects the runs.

Checks before merging (stops otherwise): one part per seed in protocol.seeds, each finished (one run, its seed);
identical config, config / code hashes, git commit, data block, inner hold-out, floors, channels and stock vocabulary;
identical validation rows and labels in every predictions file. Then the summary and the seed ensemble are computed
by the same function as a sequential run (train_v2xx.finish), from the saved predictions.
Writes <out> and <out>_preds.csv.gz; weights stay where the parts wrote them.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import labels as L  # noqa: E402
import run_config as RC  # noqa: E402
import train_v2xx as TV  # noqa: E402

SAME = ("config", "config_sha256", "script_sha256", "model_sha256", "encoders_sha256", "features_sha256", "git_commit",
        "data", "inner_es", "floors", "channels", "stock_vocab", "evaluated_on", "smoke")
KEY = ["symbol", "date", "close_t"] + [f"{p}_{h}" for h in L.HORIZONS for p in ("y", "fwd")]


def merge(cfg_path: Path, parts_dir: Path, out_path: Path) -> dict:
    cfg = RC.load(cfg_path)
    seeds = cfg["protocol"]["seeds"]
    parts, preds = [], []
    for s in seeds:
        jp, pp = parts_dir / f"seed{s}.json", parts_dir / f"seed{s}_preds.csv.gz"
        if not jp.exists() or not pp.exists():
            raise SystemExit(f"missing part for seed {s}: {jp} / {pp}")
        r = json.loads(jp.read_text(encoding="utf-8"))
        if not r.get("part") or [x["seed"] for x in r["runs"]] != [s]:
            raise SystemExit(f"{jp} is not a finished single-seed part for seed {s}")
        parts.append(r); preds.append(pd.read_csv(pp))
    first = parts[0]
    for s, r in zip(seeds[1:], parts[1:]):
        bad = [k for k in SAME if json.dumps(r.get(k), sort_keys=True, default=str) != json.dumps(first.get(k), sort_keys=True, default=str)]
        if bad:
            raise SystemExit(f"part seed {s} differs from part seed {seeds[0]} in: {bad}")
    base = preds[0][KEY]
    for s, p in zip(seeds[1:], preds[1:]):
        if not p[KEY].equals(base):
            raise SystemExit(f"predictions file of seed {s} has different validation rows or labels")
    merged = base.copy()
    for s, p in zip(seeds, preds):
        for c in (c for c in p.columns if c.startswith(f"v2xx_s{s}_")):
            merged[c] = p[c].to_numpy()                                  # column by column keeps int class columns int
    res = {k: v for k, v in first.items() if k not in ("runs", "summary", "seed_ensemble", "part", "computed_at", "gpu")}
    res.update(computed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), seeds=seeds,
               gpu=sorted({r.get("gpu") for r in parts}, key=str), parallel=True,
               merged_from=[str(parts_dir / f"seed{s}.json") for s in seeds],
               part_computed_at=[r["computed_at"] for r in parts], runs=[r["runs"][0] for r in parts])
    TV.finish(res, merged, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    merged.to_csv(out_path.with_name(out_path.stem + "_preds.csv.gz"), index=False)
    print(f"merged {len(parts)} seeds into {out_path}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cp = Path(a.config) if Path(a.config).is_absolute() else ROOT / a.config
    merge(cp, Path(a.parts), Path(a.out))
