"""Compare two v2.xx models from their post-hoc folders (scripts/step0_v2_03beta.py outputs) and apply v2.04's adopt rule.

  python scripts/compare_v2xx_posthoc.py --base results/v2_03beta --new results/v2_04_posthoc --out results/v2_04_vs_v2_03.json

Both folders hold the same rows: inner hold-out, validation and 2026 predictions, news on (`val`, `fwd`) and news blanked
(`val_blank`, `fwd_blank`), plus XGBoost on the fit rows (`trees.csv.gz`) and `step0.json` (the blend weight each model
got on its own inner hold-out).

Adopt rule (pre-registered, Siwat 2026-10-27; claude/log-entry-2026-10-27-v2-04-design.md), judged on the RAW models
(the training change itself; ts + blend reported beside):
  (a) validation, news blanked: new 3-horizon mean macro-F1 (seed mean) > base's;
  (b) validation, news on: the paired 95% date-block interval of new - base (3-horizon mean) is not entirely below 0.
Intervals: 10-day date blocks, 2,000 resamples, one set of resamples shared by every horizon, model and condition
(the same scheme as src/posthoc.block_bootstrap). 2026 is a seen window: reported, never used.
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
import posthoc as PH  # noqa: E402
import step0_v2_03beta as S  # noqa: E402

HS = (1, 5, 10)
B_RESAMPLES, BLOCK, RNG = 2000, 10, 20261016


def load(folder: Path) -> dict:
    s = json.loads((folder / "step0.json").read_text(encoding="utf-8"))
    if not s["reproduction"]["ok"]:
        raise SystemExit(f"stop: {folder} did not pass its reproduction check")
    parts = {k: pd.read_csv(folder / "parts" / f"{k}.csv.gz") for k in ("es", "val", "fwd", "val_blank", "fwd_blank")}
    t = pd.read_csv(folder / "trees.csv.gz")
    for k in ("es", "val", "fwd"):
        m = parts[k][["symbol", "date"]].merge(t[t.part == k].drop(columns="part"), on=["symbol", "date"], how="left", validate="one_to_one")
        for c in m.columns:
            if c.startswith("xgb_s"):
                for target in (k, f"{k}_blank"):                       # XGBoost has no news input: same for blanked
                    if target in parts:
                        parts[target][c] = m[c].to_numpy()
    seeds = s["run"]["seeds"]
    w = {h: s["choices"]["blend_w"][f"h{h}"]["w_deep"] for h in HS}
    return {"step0": s, "parts": parts, "seeds": seeds, "w": w, "model": s["run"]["model"]}


def calls(M: dict, part: str, h: int, kind: str) -> np.ndarray:
    df = M["parts"][part]
    if kind == "raw":
        return np.stack([S.probs(df, "v2xx", s, h).argmax(1) for s in M["seeds"]])
    es, w = M["parts"]["es"], M["w"][h]                     # blend exactly as step0 method_calls: ts(deep), ts(xgb), w
    yf = es[f"y_{h}"].to_numpy(int)
    out = []
    for s in M["seeds"]:
        pf, xf = S.probs(es, "v2xx", s, h), S.probs(es, "xgb", s, h)
        q = w * S.apply_affine(S.probs(df, "v2xx", s, h), S.fit_affine(pf, yf, False)) + \
            (1 - w) * S.apply_affine(S.probs(df, "xgb", s, h), S.fit_affine(xf, yf, False))
        out.append(q.argmax(1))
    return np.stack(out)


def counts(dates_idx, nd, y, C):
    """[S, nd, 3, 3] confusion counts per seed and date."""
    c = np.zeros((C.shape[0], nd, 9))
    for s in range(C.shape[0]):
        np.add.at(c[s], (dates_idx, y * 3 + C[s]), 1)
    return c.reshape(C.shape[0], nd, 3, 3)


def boot_weights(nd: int) -> np.ndarray:
    rng = np.random.default_rng(RNG)
    nb = int(np.ceil(nd / BLOCK))
    starts = rng.integers(0, nd - BLOCK + 1, size=(B_RESAMPLES, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(B_RESAMPLES, -1)[:, :nd]
    w = np.zeros((B_RESAMPLES, nd))
    np.add.at(w, (np.repeat(np.arange(B_RESAMPLES), nd), idx.ravel()), 1)
    return w


def scores(base: dict, new: dict, period: str) -> dict:
    """period 'val' or 'fwd'. Returns point values and resampled 3-horizon means for every (model, condition, kind)."""
    ref = base["parts"][period]
    for M in (base, new):
        for cond in ("", "_blank"):
            d = M["parts"][period + cond]
            assert (d[["symbol", "date"]].to_numpy() == ref[["symbol", "date"]].to_numpy()).all(), "rows differ between folders"
    ud, di = np.unique(ref["date"].to_numpy(), return_inverse=True)
    W = boot_weights(len(ud))
    out = {}
    for name, M in (("base", base), ("new", new)):
        for cond in ("news_on", "news_blanked"):
            part = period + ("" if cond == "news_on" else "_blank")
            for kind in ("raw", "ts_blend"):
                per_h, res_h = {}, []
                for h in HS:
                    y = ref[f"y_{h}"].to_numpy(int)
                    C = calls(M, part, h, kind)
                    cn = counts(di, len(ud), y, C)
                    per_h[f"h{h}"] = round(float(PH.f1_from_counts(cn.sum(1)).mean()), 4)
                    res_h.append(PH.f1_from_counts(np.einsum("bd,sdij->bsij", W, cn)).mean(1))      # [B]
                out[(name, cond, kind)] = {"per_horizon": per_h, "mean3": round(statistics.mean(per_h.values()), 4),
                                           "resampled_mean3": np.mean(res_h, axis=0),
                                           "resampled_h": {f"h{h}": r for h, r in zip(HS, res_h)}}
    return out


def diff(sc, cond, kind) -> dict:
    a, b = sc[("new", cond, kind)], sc[("base", cond, kind)]
    d = a["resampled_mean3"] - b["resampled_mean3"]
    lo, hi = np.percentile(d, [2.5, 97.5])
    ph = {}
    for h in HS:
        dh = a["resampled_h"][f"h{h}"] - b["resampled_h"][f"h{h}"]
        l2, h2 = np.percentile(dh, [2.5, 97.5])
        ph[f"h{h}"] = {"point": round(a["per_horizon"][f"h{h}"] - b["per_horizon"][f"h{h}"], 4), "ci95": [round(float(l2), 4), round(float(h2), 4)]}
    return {"point": round(a["mean3"] - b["mean3"], 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "share_above_0": round(float((d > 0).mean()), 3), "per_horizon": ph}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="post-hoc folder of the reference model (e.g. results/v2_03beta)")
    ap.add_argument("--new", required=True, help="post-hoc folder of the new model (e.g. results/v2_04_posthoc)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    base, new = load(Path(a.base)), load(Path(a.new))
    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "base": base["model"], "new": new["model"],
           "blend_w": {"base": base["w"], "new": new["w"]}, "rule": __doc__.split("Adopt rule")[1].split("Intervals")[0].strip()}
    for period, label in (("val", "validation"), ("fwd", "forward_2026_seen")):
        sc = scores(base, new, period)
        res[label] = {f"{n}/{c}/{k}": {kk: v for kk, v in d.items() if not kk.startswith("resampled")} for (n, c, k), d in sc.items()}
        res[label]["new_minus_base"] = {f"{c}/{k}": diff(sc, c, k) for c in ("news_on", "news_blanked") for k in ("raw", "ts_blend")}
    v = res["validation"]
    a1 = v["new/news_blanked/raw"]["mean3"] > v["base/news_blanked/raw"]["mean3"]
    a2 = v["new_minus_base"]["news_on/raw"]["ci95"][1] >= 0
    res["adopt"] = {"adopted": bool(a1 and a2), "a_blanked_mean_up": bool(a1), "b_news_on_not_below": bool(a2)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"\n{new['model']} vs {base['model']} (validation macro-F1, seed mean; blend w {new['model']} {list(new['w'].values())}, "
          f"{base['model']} {list(base['w'].values())})")
    print(f"{'':28s}{'1 d':>8s}{'5 d':>8s}{'10 d':>8s}{'mean':>8s}   new - base, 3-horizon mean (95% interval)")
    for c in ("news_on", "news_blanked"):
        for k in ("raw", "ts_blend"):
            for n, M in (("base", base), ("new", new)):
                r = v[f"{n}/{c}/{k}"]
                tail = ""
                if n == "new":
                    d = v["new_minus_base"][f"{c}/{k}"]
                    tail = f"   {d['point']:+.4f} {d['ci95']}"
                print(f"{M['model'] + ' ' + c + ' ' + k:28s}" + "".join(f"{r['per_horizon'][f'h{h}']:8.4f}" for h in HS) + f"{r['mean3']:8.4f}{tail}")
    print(f"\nADOPT RULE: (a) blanked mean up: {a1}; (b) news-on not below (interval upper end >= 0): {a2} -> "
          f"{'ADOPTED' if a1 and a2 else 'not adopted'}")
    f = res["forward_2026_seen"]
    print("2026 (seen window, reported only), raw 3-horizon mean: " + "; ".join(
        f"{c} {base['model']} {f[f'base/{c}/raw']['mean3']:.4f} -> {new['model']} {f[f'new/{c}/raw']['mean3']:.4f}" for c in ("news_on", "news_blanked")))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
