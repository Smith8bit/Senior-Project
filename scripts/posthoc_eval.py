"""Batch 4 (D01, D02, E24, E30, E29 verdict) and batch 7 (final blend / calibration / confirmation) from saved predictions.

Rules fixed 2026-10-16 before any batch 4-7 result was scored (claude/batches-4-7-design-2026-10-16.md):
  * Deep scores are per-horizon best epochs (prefix multiscale_sector_ph), 3 seeds; partner seed s pairs with deep seed s.
  * E24 blend kept at horizon h if (a) on the official window blend - deep > the deep model's seed std there, and
    (b) the mean over the 4 walk-forward windows of (blend - deep) is > 0.
  * E30 calibration kept at h if (a) official calibrated - deep > the deep std, with biases fitted on the 4 windows only,
    and (b) leave-one-window-out gain (fit on 3 windows, score the 4th) averaged over the 4 windows is > 0.
  * E29 "beats XGBoost" at h if official LightGBM - XGBoost(E03) > XGBoost's seed std and the 4-window mean of
    (LightGBM - XGBoost) is > 0. Only then may batch 7 try the E29 blend (E24b).
  * Batch 7, per horizon: blend candidates = none, XGBoost (if E24 kept), LightGBM E29 (if E29 beats XGBoost); the one
    with the highest 4-window mean is used (ties -> none). Then, if E30 was kept, biases are refitted on the 4 windows of
    that pipeline and used only if their leave-one-window-out gain is > 0. Window scores use the leave-one-window-out
    biases, so no window is scored with biases fitted on itself.
  * Confirmation (amendment 2026-10-16, rule 3): a horizon is confirmed if the official mean is at or above the pass
    bar AND the window mean is at or above max(best one-rule floor, XGBoost) + 0.02 in at least 3 of the 4 windows.
VALIDATION ONLY. The test block stays sealed.

  python scripts/posthoc_eval.py batch4 --deep-official E21_PREDS --deep-window 2022H2 PREDS ... --xgb-official XGB_PREDS \
      --xgb-window 2022H2 PREDS ... --e29-dir DIR --window-json 2022H2 DEEP_JSON ... --official-json E21_JSON --out OUT.json
  python scripts/posthoc_eval.py final  ... same inputs for the final deep config ... --batch4 BATCH4.json --out OUT.json
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
import posthoc as PH  # noqa: E402

HS = ("h1", "h5", "h10")
PASS_BAR = {"h1": 0.393, "h5": 0.400, "h10": 0.407}
DEEP = "multiscale_sector_ph"
MARGIN = 0.02


def ms(xs):
    return round(statistics.mean(xs), 4), (round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0)


def read(path):
    return pd.read_csv(ROOT / path)


def load_sets(args):
    """{window: merged predictions frame} with deep, xgb and (if given) lgbm/xgbf columns; 'official' + 4 windows."""
    sets = {}
    wins = [("official", args.deep_official, args.xgb_official)] + \
           [(w, d, dict(args.xgb_window)[w]) for w, d in args.deep_window]
    for w, dp, xp in wins:
        df = PH.merge(read(dp), read(xp))
        if args.e29_dir and (ROOT / args.e29_dir / f"{w}_factors_preds.csv.gz").exists():
            df = PH.merge(df, read(Path(args.e29_dir) / f"{w}_factors_preds.csv.gz"))
        sets[w] = df
    return sets


def pipe_probs(df, h, s, blend=None):
    p = PH.probs(df, DEEP, s, h)
    if blend:
        p = 0.5 * p + 0.5 * PH.probs(df, blend, s, h)
    return p


def seed_f1(df, h, seeds, blend=None, bias=None, prefix=None):
    y = df[f"y_{h}"].to_numpy(int)
    out = []
    for s in seeds:
        p = PH.probs(df, prefix, s, h) if prefix else pipe_probs(df, h, s, blend)
        out.append(PH.macro_f1(y, PH.predict(p, bias or (0.0, 0.0))))
    return out


def calibrate(sets, wins, h, seeds, blend=None):
    """Fit on all windows (for the official score) and leave-one-window-out (for the window scores)."""
    def groups(ws):
        return [(pipe_probs(sets[w], h, s, blend), sets[w][f"y_{h}"].to_numpy(int)) for w in ws for s in seeds]
    full = PH.fit_bias(groups(wins))
    lowo = {}
    for w in wins:
        fit = PH.fit_bias(groups([v for v in wins if v != w]))
        b = (fit["bias_flat"], fit["bias_up"])
        cal, raw = seed_f1(sets[w], h, seeds, blend, b), seed_f1(sets[w], h, seeds, blend)
        lowo[w] = {"bias": list(b), "calibrated": ms(cal)[0], "uncalibrated": ms(raw)[0], "gain": round(ms(cal)[0] - ms(raw)[0], 4)}
    return full, lowo


def class_shares(paths):
    out = {}
    for w, p in paths:
        d = json.loads((ROOT / p).read_text(encoding="utf-8"))["data"]
        out[w] = {h: {"train": d[h]["train_class_shares"], "val": d[h]["val_class_shares"],
                      "flat_drift": round(d[h]["val_class_shares"]["flat"] - d[h]["train_class_shares"]["flat"], 4),
                      "k": d[h]["k"]} for h in HS}
    return out


def floors_by_window(paths):
    """Per-window floors and XGBoost means for the per-window bar (from the deep JSON and the XGBoost JSON)."""
    out = {}
    for w, dp, xp in paths:
        d = json.loads((ROOT / dp).read_text(encoding="utf-8"))
        x = json.loads((ROOT / xp).read_text(encoding="utf-8"))
        out[w] = {}
        for h in HS:
            fl = d["floors"][h]
            best = max(fl["trend_continues"]["macro_f1"], fl["trend_reverses"]["macro_f1"])
            xgb = x["results"][h]["model"]["xgb_trend"]["macro_f1"]["mean"]
            out[w][h] = {"best_floor": best, "xgb": xgb, "bar": round(max(best, xgb) + MARGIN, 4)}
    return out


def past_classes(official_rows):
    """One-rule floor predictions (trend continues / reverses) for the official validation rows, rebuilt from data."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import baselines_tabular as BT
    import labels as L
    data, info = BT.build_dataset()
    va = data[data.split == "val"].copy()
    va["date"] = va["date"].astype(str).str[:10]
    out = official_rows[["symbol", "date"]].merge(va[["symbol", "date"] + [f"r_{h}" for h in L.HORIZONS] +
                                                  [f"sigma_{h}" for h in L.HORIZONS] + [f"y_{h}" for h in L.HORIZONS]],
                                                  on=["symbol", "date"], how="left", validate="one_to_one")
    assert out[["r_1"]].notna().all().all(), "official rows not found in the rebuilt dataset"
    for h in L.HORIZONS:
        assert (out[f"y_{h}"].to_numpy() == official_rows[f"y_{h}"].to_numpy()).all(), "labels differ from the rebuilt dataset"
        past = L.classify(out[f"r_{h}"].to_numpy(), out[f"sigma_{h}"].to_numpy(), info[f"h{h}"]["k"])
        out[f"cont_{h}"], out[f"rev_{h}"] = past, 2 - past
    return out


def bootstrap_official(df, seeds, extra_preds, pairs, floors=None):
    """D02 on the official window: dict per horizon."""
    res = {}
    dates = df["date"].to_numpy()
    for h in HS:
        hn = int(h[1:])
        y = df[f"y_{hn}"].to_numpy(int)
        preds = {"deep": np.stack([PH.predict(PH.probs(df, DEEP, s, hn)) for s in seeds])}
        for name, fn in extra_preds.items():
            preds[name] = fn(hn)
        if floors is not None:
            preds["trend_continues"] = floors[f"cont_{hn}"].to_numpy()[None, :]
            preds["trend_reverses"] = floors[f"rev_{hn}"].to_numpy()[None, :]
        prs = [p for p in pairs if p[0] in preds and p[1] in preds]
        res[h] = PH.block_bootstrap(dates, y, preds, pairs=prs, bar=PASS_BAR[h])
    return res


def batch4(args):
    sets = load_sets(args)
    wins = [w for w, _ in args.deep_window]
    seeds = PH.seeds_of(sets["official"], DEEP)
    assert seeds == PH.seeds_of(sets["official"], "xgb"), "deep and XGBoost seeds differ"
    have_e29 = all("lgbm_s%d_p1_down" % seeds[0] in sets[w] for w in ["official"] + wins)
    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "batch": 4, "seeds": seeds,
           "windows": wins, "evaluated_on": "validation only (test sealed)", "pass_bar": PASS_BAR,
           "d01_class_shares": class_shares([("official", args.official_json)] + list(args.window_json)),
           "e24": {}, "e30": {}, "e29": {}, "keep": {"E24": {}, "E30": {}, "E29_beats_xgb": {}}}
    for h in HS:
        hn = int(h[1:])
        off_deep = seed_f1(sets["official"], hn, seeds)
        dm, dsd = ms(off_deep)
        # E24 blend
        off_bl = seed_f1(sets["official"], hn, seeds, blend="xgb")
        win = {w: {"deep": ms(seed_f1(sets[w], hn, seeds))[0], "blend": ms(seed_f1(sets[w], hn, seeds, blend="xgb"))[0]} for w in wins}
        wgain = statistics.mean(v["blend"] - v["deep"] for v in win.values())
        res["e24"][h] = {"official_deep": [dm, dsd], "official_blend": list(ms(off_bl)), "official_gain": round(ms(off_bl)[0] - dm, 4),
                         "windows": win, "window_mean_gain": round(wgain, 4)}
        res["keep"]["E24"][h] = bool(ms(off_bl)[0] - dm > dsd and wgain > 0)
        # E30 calibration
        full, lowo = calibrate(sets, wins, hn, seeds)
        b = (full["bias_flat"], full["bias_up"])
        off_cal = seed_f1(sets["official"], hn, seeds, bias=b)
        lgain = statistics.mean(v["gain"] for v in lowo.values())
        res["e30"][h] = {"bias_all_windows": list(b), "fit": full, "official_calibrated": list(ms(off_cal)),
                         "official_gain": round(ms(off_cal)[0] - dm, 4), "lowo": lowo, "lowo_mean_gain": round(lgain, 4)}
        res["keep"]["E30"][h] = bool(ms(off_cal)[0] - dm > dsd and lgain > 0)
        # E29 against XGBoost (E03 inputs)
        if have_e29:
            off_x = seed_f1(sets["official"], hn, seeds, prefix="xgb")
            e = {"official_xgb": list(ms(off_x))}
            for lr in ("lgbm", "xgbf"):
                e[f"official_{lr}"] = list(ms(seed_f1(sets["official"], hn, seeds, prefix=lr)))
                e[f"windows_{lr}"] = {w: ms(seed_f1(sets[w], hn, seeds, prefix=lr))[0] for w in wins}
            e["windows_xgb"] = {w: ms(seed_f1(sets[w], hn, seeds, prefix="xgb"))[0] for w in wins}
            e["windows_deep"] = {w: v["deep"] for w, v in win.items()}
            wg = statistics.mean(e["windows_lgbm"][w] - e["windows_xgb"][w] for w in wins)
            e["lgbm_minus_xgb_official"] = round(e["official_lgbm"][0] - e["official_xgb"][0], 4)
            e["lgbm_minus_xgb_window_mean"] = round(wg, 4)
            res["e29"][h] = e
            res["keep"]["E29_beats_xgb"][h] = bool(e["lgbm_minus_xgb_official"] > e["official_xgb"][1] and wg > 0)
        else:
            res["keep"]["E29_beats_xgb"][h] = False
    # D02 bootstrap intervals on the official window
    floors = past_classes(sets["official"]) if not args.no_floors else None
    extra = {"xgb": lambda hn: np.stack([PH.predict(PH.probs(sets["official"], "xgb", s, hn)) for s in seeds]),
             "blend_xgb": lambda hn: np.stack([PH.predict(pipe_probs(sets["official"], hn, s, "xgb")) for s in seeds])}
    if have_e29:
        extra["lgbm"] = lambda hn: np.stack([PH.predict(PH.probs(sets["official"], "lgbm", s, hn)) for s in seeds])
    pairs = [("deep", "xgb"), ("deep", "trend_reverses"), ("deep", "trend_continues"), ("blend_xgb", "deep"),
             ("lgbm", "xgb"), ("lgbm", "deep")]
    res["d02_official"] = bootstrap_official(sets["official"], seeds, extra, pairs, floors)
    res["d02_windows"] = {w: {h: PH.block_bootstrap(sets[w]["date"].to_numpy(), sets[w][f"y_{h[1:]}"].to_numpy(int),
                                                    {"deep": np.stack([PH.predict(PH.probs(sets[w], DEEP, s, int(h[1:]))) for s in seeds])})["values"]["deep"]
                              for h in HS} for w in wins}
    write(res, args.out)
    show_batch4(res)


def show_batch4(r):
    print("\nD01 class shares (official): " + "; ".join(
        f"{h} flat train {v['train']['flat']:.3f} -> val {v['val']['flat']:.3f}" for h, v in r["d01_class_shares"]["official"].items()))
    print(f"\n{'h':4s} {'E21':>13s} {'E24 blend':>13s} {'gain':>7s} {'win gain':>9s} {'keep':>5s} | {'E30 cal':>13s} {'gain':>7s} {'LOWO':>7s} {'keep':>5s}")
    for h in HS:
        a, c = r["e24"][h], r["e30"][h]
        print(f"{h:4s} {a['official_deep'][0]:.4f}±{a['official_deep'][1]:.4f} {a['official_blend'][0]:.4f}±{a['official_blend'][1]:.4f} "
              f"{a['official_gain']:+7.4f} {a['window_mean_gain']:+9.4f} {str(r['keep']['E24'][h]):>5s} | "
              f"{c['official_calibrated'][0]:.4f}±{c['official_calibrated'][1]:.4f} {c['official_gain']:+7.4f} {c['lowo_mean_gain']:+7.4f} {str(r['keep']['E30'][h]):>5s}")
    if r["e29"]:
        print(f"\n{'h':4s} {'XGB(E03)':>13s} {'E29 LightGBM':>13s} {'E29 XGB-343':>13s} {'lgbm-xgb':>9s} {'win':>7s} {'beats':>6s}")
        for h in HS:
            e = r["e29"][h]
            print(f"{h:4s} {e['official_xgb'][0]:.4f}±{e['official_xgb'][1]:.4f} {e['official_lgbm'][0]:.4f}±{e['official_lgbm'][1]:.4f} "
                  f"{e['official_xgbf'][0]:.4f}±{e['official_xgbf'][1]:.4f} {e['lgbm_minus_xgb_official']:+9.4f} "
                  f"{e['lgbm_minus_xgb_window_mean']:+7.4f} {str(r['keep']['E29_beats_xgb'][h]):>6s}")
    print("\nD02 official, 95% intervals (date blocks of 10 days):")
    for h in HS:
        v = r["d02_official"][h]
        print(f"  {h}: E21 {v['values']['deep']['point']} {v['values']['deep']['ci95']} (P >= bar {v['values']['deep']['share_of_resamples_at_or_above_bar']}); "
              + "; ".join(f"{k} {d['point']:+.4f} {d['ci95']}" for k, d in v["differences"].items()))


def final(args):
    b4 = json.loads((ROOT / args.batch4).read_text(encoding="utf-8"))
    sets = load_sets(args)
    wins = [w for w, _ in args.deep_window]
    seeds = PH.seeds_of(sets["official"], DEEP)
    bars = floors_by_window([(w, dict(args.window_json)[w], dict(args.xgb_json)[w]) for w in wins])
    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "batch": 7, "seeds": seeds, "windows": wins,
           "label": args.label, "evaluated_on": "validation only (test sealed)", "pass_bar": PASS_BAR, "window_bars": bars,
           "batch4_keep": b4["keep"], "horizons": {}}
    for h in HS:
        hn = int(h[1:])
        cands = [None] + (["xgb"] if b4["keep"]["E24"][h] else []) + \
                (["lgbm"] if b4["keep"]["E29_beats_xgb"][h] and f"lgbm_s{seeds[0]}_p1_down" in sets["official"] else [])
        wmean = {str(c): round(statistics.mean(ms(seed_f1(sets[w], hn, seeds, blend=c))[0] for w in wins), 4) for c in cands}
        blend = None
        for c in cands[1:]:
            if wmean[str(c)] > wmean[str(blend)]:
                blend = c
        hres = {"blend_candidates_window_mean": wmean, "blend": blend, "calibration": None}
        bias_off, lowo = None, None
        if b4["keep"]["E30"][h]:
            full, lowo = calibrate(sets, wins, hn, seeds, blend)
            lg = statistics.mean(v["gain"] for v in lowo.values())
            hres["calibration_check"] = {"fit": full, "lowo": lowo, "lowo_mean_gain": round(lg, 4)}
            if lg > 0:
                bias_off = (full["bias_flat"], full["bias_up"])
                hres["calibration"] = list(bias_off)
        raw = ms(seed_f1(sets["official"], hn, seeds))
        off = ms(seed_f1(sets["official"], hn, seeds, blend, bias_off))
        wscore = {}
        for w in wins:
            b = tuple(lowo[w]["bias"]) if (bias_off is not None) else None
            wscore[w] = ms(seed_f1(sets[w], hn, seeds, blend, b))[0]
        passes = {w: wscore[w] >= bars[w][h]["bar"] for w in wins}
        hres.update({"official_deep_alone": list(raw), "official_final": list(off), "official_gap_to_bar": round(off[0] - PASS_BAR[h], 4),
                     "official_passes_bar": off[0] >= PASS_BAR[h], "windows_final": wscore,
                     "window_mean": round(statistics.mean(wscore.values()), 4), "windows_passing_bar": int(sum(passes.values())),
                     "confirmed": bool(off[0] >= PASS_BAR[h] and sum(passes.values()) >= 3)})
        y = sets["official"][f"y_{hn}"].to_numpy(int)
        fin = np.stack([PH.predict(pipe_probs(sets["official"], hn, s, blend), bias_off or (0.0, 0.0)) for s in seeds])
        alone = np.stack([PH.predict(PH.probs(sets["official"], DEEP, s, hn)) for s in seeds])
        xg = np.stack([PH.predict(PH.probs(sets["official"], "xgb", s, hn)) for s in seeds])
        hres["d02"] = PH.block_bootstrap(sets["official"]["date"].to_numpy(), y, {"final": fin, "deep_alone": alone, "xgb": xg},
                                         pairs=[("final", "deep_alone"), ("final", "xgb")], bar=PASS_BAR[h])
        res["horizons"][h] = hres
    res["candidate"] = all(res["horizons"][h]["official_passes_bar"] for h in HS)
    res["partial_pass"] = [h for h in HS if res["horizons"][h]["official_passes_bar"]]
    res["confirmed_horizons"] = [h for h in HS if res["horizons"][h]["confirmed"]]
    write(res, args.out)
    print(f"\nFINAL ({args.label}):")
    for h in HS:
        v = res["horizons"][h]
        print(f"  {h}: blend {v['blend']}, calibration {v['calibration']}; official {v['official_final'][0]:.4f} ± {v['official_final'][1]:.4f} "
              f"(deep alone {v['official_deep_alone'][0]:.4f}; bar {PASS_BAR[h]}; gap {v['official_gap_to_bar']:+.4f}); "
              f"windows {v['windows_final']} pass {v['windows_passing_bar']}/4; confirmed {v['confirmed']}; "
              f"95% interval {v['d02']['values']['final']['ci95']}")
    print(f"  candidate (all horizons pass the official bar): {res['candidate']}; partial: {res['partial_pass']}; confirmed: {res['confirmed_horizons']}")


def write(res, out):
    p = ROOT / out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"wrote {p}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["batch4", "final"])
    ap.add_argument("--deep-official", required=True)
    ap.add_argument("--deep-window", nargs=2, action="append", required=True, metavar=("LABEL", "PREDS"))
    ap.add_argument("--xgb-official", required=True)
    ap.add_argument("--xgb-window", nargs=2, action="append", required=True, metavar=("LABEL", "PREDS"))
    ap.add_argument("--e29-dir", default=None)
    ap.add_argument("--official-json", default=None, help="batch4: E21 run JSON (class shares)")
    ap.add_argument("--window-json", nargs=2, action="append", default=[], metavar=("LABEL", "DEEP_JSON"))
    ap.add_argument("--xgb-json", nargs=2, action="append", default=[], metavar=("LABEL", "XGB_JSON"), help="final: per-window XGBoost JSON (bars)")
    ap.add_argument("--batch4", default=None, help="final: batch 4 summary with the keep flags")
    ap.add_argument("--label", default="final")
    ap.add_argument("--no-floors", action="store_true", help="batch4: skip rebuilding the dataset for the floor intervals")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    batch4(args) if args.mode == "batch4" else final(args)


if __name__ == "__main__":
    main()
