"""Compare two v2.xx models from their post-hoc folders (scripts/step0_v2_03beta.py outputs) and apply an adopt rule.

  python scripts/compare_v2xx_posthoc.py --base results/v2_03beta --new results/v2_04_posthoc --out results/v2_04_vs_v2_03.json
  python scripts/compare_v2xx_posthoc.py --base results/v2_04_posthoc --new results/v2_05_posthoc --rule trend_up_price_held \
         --base-run results/v2xx/A100/v2_04.json --new-run results/v2xx/A100/v2_05.json --out results/v2_05_vs_v2_04.json

Both folders hold the same rows: inner hold-out, validation and 2026 predictions, news on (`val`, `fwd`) and news blanked
(`val_blank`, `fwd_blank`), plus XGBoost on the fit rows (`trees.csv.gz`) and `step0.json` (the blend weight each model
got on its own inner hold-out).

Adopt rules (pre-registered by Siwat; judged on the RAW models, the training change itself; ts + blend reported beside):
  blanked_up_news_on_held (v2.04, 2026-10-27; claude/log-entry-2026-10-27-v2-04-design.md), the default:
    (a) validation, news blanked: new 3-horizon mean macro-F1 (seed mean) > base's;
    (b) validation, news on: the paired 95% date-block interval of new - base (3-horizon mean) is not entirely below 0.
  trend_up_price_held (v2.05, 2026-10-27; claude/log-entry-2026-10-27-v2-05-design.md); needs --base-run / --new-run:
    (a) validation, news on: new 3-horizon mean macro-F1 (seed mean) > base's;
    (b) price R2_OS (seed mean from the run JSON, mean of the 3 horizons): new >= base;
    (c) validation, news blanked: the paired 95% interval of new - base (3-horizon mean) is not entirely below 0.
  mean_held_any_horizon_up (v2.11, 2026-11-10; claude/log-entry-2026-11-10-v2-11-design.md):
    validation, news on: (a) the paired 95% interval of new - base (mean over the horizons) is not entirely below 0
    (upper end >= 0), and (b) at least one horizon's paired 95% interval is entirely above 0.
  h1_up_significant (v2.07, 2026-11-02; claude/log-entry-2026-11-02-v2-07-design.md):
    (a) validation, news on, 1-day horizon: the paired 95% interval of new - base macro-F1 (seed mean) is entirely above 0.
  trend_up_significant (v2.06, 2026-11-02; claude/log-entry-2026-11-02-v2-06-design.md):
    (a) validation, news on: the paired 95% interval of new - base (3-horizon mean) is entirely above 0;
    (b) validation, news blanked: the paired 95% interval of new - base (3-horizon mean) is not entirely below 0.
Only the horizons both models predict are compared (v2.07 predicts the 1-day horizon only).
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
    ap.add_argument("--rule", default="blanked_up_news_on_held",
                    choices=("blanked_up_news_on_held", "trend_up_price_held", "trend_up_significant", "h1_up_significant",
                             "mean_held_any_horizon_up"))
    ap.add_argument("--base-run", default=None, help="run JSON of the reference model (price R2_OS; rule trend_up_price_held)")
    ap.add_argument("--new-run", default=None, help="run JSON of the new model")
    a = ap.parse_args(argv)
    if a.rule == "trend_up_price_held" and not (a.base_run and a.new_run):
        ap.error("--rule trend_up_price_held needs --base-run and --new-run (price R2_OS)")
    hb = [int(k[1:]) for k in json.loads((Path(a.base) / "step0.json").read_text(encoding="utf-8"))["validation"]]
    hn = [int(k[1:]) for k in json.loads((Path(a.new) / "step0.json").read_text(encoding="utf-8"))["validation"]]
    globals()["HS"] = tuple(h for h in (1, 5, 10) if h in hb and h in hn)
    if a.rule == "h1_up_significant" and 1 not in HS:
        ap.error("rule h1_up_significant needs the 1-day horizon in both models")
    base, new = load(Path(a.base)), load(Path(a.new))
    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "base": base["model"], "new": new["model"],
           "blend_w": {"base": base["w"], "new": new["w"]}, "rule_name": a.rule,
           "rule": __doc__.split(f"  {a.rule} ")[1].split("\n  blanked_up")[0].split("\n  trend_up")[0].split("\n  h1_up")[0].split("\n  mean_held")[0].split("\nOnly the horizons")[0].split("Intervals")[0].strip()}
    price = {}
    for name, path in (("base", a.base_run), ("new", a.new_run)):
        if path:
            r = json.loads(Path(path).read_text(encoding="utf-8"))
            assert r["config"]["name"] == {"base": base, "new": new}[name]["model"], f"{path} is not the {name} model's run"
            per = {f"h{h}": r["summary"]["model"][f"h{h}"]["r2_os_vs_zero"]["mean"] for h in HS}
            price[name] = {"r2_os_per_horizon": per, "r2_os_mean3": round(statistics.mean(per.values()), 5)}
    res["price"] = price
    for period, label in (("val", "validation"), ("fwd", "forward_2026_seen")):
        sc = scores(base, new, period)
        res[label] = {f"{n}/{c}/{k}": {kk: v for kk, v in d.items() if not kk.startswith("resampled")} for (n, c, k), d in sc.items()}
        res[label]["new_minus_base"] = {f"{c}/{k}": diff(sc, c, k) for c in ("news_on", "news_blanked") for k in ("raw", "ts_blend")}
    v = res["validation"]
    if a.rule == "blanked_up_news_on_held":
        a1 = v["new/news_blanked/raw"]["mean3"] > v["base/news_blanked/raw"]["mean3"]
        a2 = v["new_minus_base"]["news_on/raw"]["ci95"][1] >= 0
        res["adopt"] = {"adopted": bool(a1 and a2), "a_blanked_mean_up": bool(a1), "b_news_on_not_below": bool(a2)}
        verdict = f"(a) blanked mean up: {a1}; (b) news-on not below (interval upper end >= 0): {a2}"
    elif a.rule == "mean_held_any_horizon_up":
        d = v["new_minus_base"]["news_on/raw"]
        a1 = d["ci95"][1] >= 0
        up = [h for h in HS if d["per_horizon"][f"h{h}"]["ci95"][0] > 0]
        a2 = bool(up)
        res["adopt"] = {"adopted": bool(a1 and a2), "a_mean_not_below_0": bool(a1), "b_horizons_above_0": [f"h{h}" for h in up]}
        verdict = (f"(a) news-on mean {d['point']:+.4f}, interval {d['ci95']} not entirely below 0: {a1}; (b) horizons with the "
                   f"interval entirely above 0: {[f'{h} d' for h in up] or 'none'}: {a2}")
    elif a.rule == "h1_up_significant":
        d1 = v["new_minus_base"]["news_on/raw"]["per_horizon"]["h1"]
        a1 = d1["ci95"][0] > 0
        res["adopt"] = {"adopted": bool(a1), "a_h1_news_on_interval_above_0": bool(a1)}
        verdict = f"(a) 1-day news-on gain {d1['point']:+.4f}, interval {d1['ci95']} entirely above 0: {a1}"
    elif a.rule == "trend_up_significant":
        a1 = v["new_minus_base"]["news_on/raw"]["ci95"][0] > 0
        a2 = v["new_minus_base"]["news_blanked/raw"]["ci95"][1] >= 0
        res["adopt"] = {"adopted": bool(a1 and a2), "a_news_on_interval_above_0": bool(a1), "b_blanked_not_below": bool(a2)}
        verdict = (f"(a) news-on gain significant (interval {v['new_minus_base']['news_on/raw']['ci95']} entirely above 0): {a1}; "
                   f"(b) blanked not below (interval upper end >= 0): {a2}")
    else:
        a1 = v["new/news_on/raw"]["mean3"] > v["base/news_on/raw"]["mean3"]
        a2 = price["new"]["r2_os_mean3"] >= price["base"]["r2_os_mean3"]
        a3 = v["new_minus_base"]["news_blanked/raw"]["ci95"][1] >= 0
        res["adopt"] = {"adopted": bool(a1 and a2 and a3), "a_news_on_mean_up": bool(a1), "b_price_r2_not_worse": bool(a2),
                        "c_blanked_not_below": bool(a3)}
        verdict = (f"(a) news-on mean up: {a1}; (b) price R2_OS not worse ({price['new']['r2_os_mean3']:+.4f} vs "
                   f"{price['base']['r2_os_mean3']:+.4f}): {a2}; (c) blanked not below (interval upper end >= 0): {a3}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"\n{new['model']} vs {base['model']} (validation macro-F1, seed mean; blend w {new['model']} {list(new['w'].values())}, "
          f"{base['model']} {list(base['w'].values())})")
    print(f"{'':28s}" + "".join(f"{f'{h} d':>8s}" for h in HS) + f"{'mean':>8s}   new - base, mean over {len(HS)} horizon(s) (95% interval)")
    for c in ("news_on", "news_blanked"):
        for k in ("raw", "ts_blend"):
            for n, M in (("base", base), ("new", new)):
                r = v[f"{n}/{c}/{k}"]
                tail = ""
                if n == "new":
                    d = v["new_minus_base"][f"{c}/{k}"]
                    tail = f"   {d['point']:+.4f} {d['ci95']}"
                print(f"{M['model'] + ' ' + c + ' ' + k:28s}" + "".join(f"{r['per_horizon'][f'h{h}']:8.4f}" for h in HS) + f"{r['mean3']:8.4f}{tail}")
    for name, M in (("base", base), ("new", new)):
        if name in price:
            print(f"{M['model'] + ' price R2_OS':28s}" + "".join(f"{price[name]['r2_os_per_horizon'][f'h{h}']:+8.4f}" for h in HS)
                  + f"{price[name]['r2_os_mean3']:+8.4f}")
    print(f"\nADOPT RULE {a.rule}: {verdict} -> {'ADOPTED' if res['adopt']['adopted'] else 'not adopted'}")
    f = res["forward_2026_seen"]
    print(f"2026 (seen window, reported only), raw mean over {len(HS)} horizon(s): " + "; ".join(
        f"{c} {base['model']} {f[f'base/{c}/raw']['mean3']:.4f} -> {new['model']} {f[f'new/{c}/raw']['mean3']:.4f}" for c in ("news_on", "news_blanked")))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
