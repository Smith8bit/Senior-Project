"""Forward (out-of-sample) test of a trained v2.xx model (default v2.03) on 2026 data, from its SAVED weights.

Nothing is re-trained. Each seed's kept weights (`<run>_weights/seed<seed>.pt`, written by scripts/train_v2xx.py) are
loaded and applied to 2026 member stock-days. The 2025 test block stays sealed: late-2025 prices are only look-back
inputs for January 2026; no 2025 row is predicted or scored.

Same 2026 world as the main model v2 forward test (scripts/forward_test_v2.py): the same saved Yahoo prices and
events, SET50 2026 lists, sectors for the 2026 joiners, series breaks, pinned official blocks and the identity check
(training and validation rows unchanged). So v2.xx and v2 are scored on exactly the same 2026 rows.

Checks before anything is scored (stops otherwise):
  * the loaded weights reproduce the run's stored validation predictions (max |probability difference| and per-seed
    macro-F1), so the inputs built here are the inputs the model was trained on;
  * the checkpoint's stock vocabulary, sector ids and return scale equal the ones rebuilt from the training rows;
  * the code files match the hashes recorded in the run JSON.

Stages:
  python scripts/forward_test_v2.py prepare --fwd-root /content/fwd_root --out-dir OUT --price-source yahoo   # 2026 data
  python scripts/forward_test_v2xx.py predict --fwd-root /content/fwd_root --out-dir OUT --run-json RUN.json --weights DIR
  python scripts/forward_test_v2xx.py score   --fwd-root /content/fwd_root --out-dir OUT --run-json RUN.json \
         [--v2-forward DIR_OF_V2_FORWARD_TEST]
News: the sentiment file ends 2025-12-30, so every 2026 day gets the config's no-news fill. `predict` also scores the
validation window with that fill on every day from the validation start ("news blanked"), to show what it costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import baselines_tabular as BT  # noqa: E402
import features as F  # noqa: E402
import features_proposal as FP  # noqa: E402
import forward_test_v2 as FT  # noqa: E402
import labels as L  # noqa: E402
import metrics as M  # noqa: E402
import posthoc as PH  # noqa: E402
import sequences as SQ  # noqa: E402

HS = L.HORIZONS
REPRO_PROB_TOL = 1e-3          # max |probability difference| vs the stored validation predictions
REPRO_F1_TOL = 0.002           # per-seed validation macro-F1 difference
NO_NEWS = {"zeros": (0.0, 0.0, 0.0), "neutral": (0.0, 0.0, 1.0), "uniform": (1 / 3, 1 / 3, 1 / 3)}
CODE_KEYS = {"script_sha256": "scripts/train_v2xx.py", "model_sha256": "src/proposal_model.py",
             "encoders_sha256": "src/encoders.py", "features_sha256": "src/features_proposal.py"}
say = FT.say


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_run(run_json: Path) -> dict:
    run = json.loads(Path(run_json).read_text(encoding="utf-8"))
    bad = {k: (run.get(k, "")[:12], sha(ROOT / f)[:12]) for k, f in CODE_KEYS.items() if run.get(k) and run[k] != sha(ROOT / f)}
    if bad:
        raise SystemExit(f"stop: code differs from the code the run used (run vs here): {bad}. Check out the run's commit "
                         f"({run.get('git_commit')}).")
    return run


def forward_news(path: Path, full: pd.DataFrame, frozen: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """2026 headline scores (scripts/sentiment_score_forward.py). Their trading_day was assigned on the PC with a
    Monday-Friday calendar; it is recomputed here from `published` with the real calendar (dates on which >= 30 symbols
    trade, as news_coverage_check.trading_calendar), by the same hard-rule-4 function."""
    from datetime import datetime
    from news_coverage_check import news_trading_day
    fw = pd.read_csv(path)
    n = full.groupby("date").symbol.size()
    cal = sorted(d.date() for d in n[n >= 30].index)
    td = [news_trading_day(datetime.fromisoformat(p), cal) for p in fw["published"]]
    fw["trading_day_pc"] = fw["trading_day"]
    fw["trading_day"] = [d.isoformat() if d else None for d in td]
    dropped = int(fw.trading_day.isna().sum())
    fw = fw.dropna(subset=["trading_day"])
    assert (fw.trading_day > "2025-12-30").all(), "a forward headline falls in the frozen file's period"
    dup = fw.merge(frozen[["url", "ticker"]], on=["url", "ticker"])
    assert dup.empty, f"{len(dup)} (url, ticker) pairs are in both news files"
    st = {"file": Path(path).name, "sha256": sha(path), "pairs": int(len(fw)), "dropped_after_last_trading_day": dropped,
          "trading_day_moved_by_real_calendar": int((fw.trading_day != fw.trading_day_pc).sum()),
          "first_day": fw.trading_day.min(), "last_day": fw.trading_day.max()}
    return fw.drop(columns=["trading_day_pc"]), st


def build(cfg: dict, fwd_root: Path, news_fwd: Path | None = None) -> dict:
    """Rows and inputs exactly as scripts/train_v2xx.build_data makes them, in the forward world (2026 added)."""
    with FT.forward_world(fwd_root):
        data, info, full = BT.build_dataset(root=fwd_root, return_full=True, extra="none")
    full = FP.add_price_tech(full, cfg["inputs"])
    scores = pd.read_csv(ROOT / "data/derived/sentiment_scores.csv")
    news_stats = None
    if news_fwd:
        fw, news_stats = forward_news(Path(news_fwd), full, scores)
        scores = pd.concat([scores, fw[scores.columns]], ignore_index=True)
    sent = FP.sentiment_probs(scores, full[["symbol", "date"]], cfg["inputs"]["sentiment_no_news"])
    key = pd.MultiIndex.from_frame(full[["symbol", "date"]])
    for c in FP.SENT_CH:
        full[c] = sent[c].reindex(key).to_numpy()
    full["has_news"] = sent["has_news"].reindex(key).to_numpy()
    start = pd.Timestamp(FT.FWD_START)
    tr_rows, va = data[data.split == "train"], data[data.split == "val"]
    fwd = data[data.date >= start]
    live = full[full.member & (full.date >= start) & full[F.FEATURES].notna().all(axis=1) & ~full.index.isin(data.index)]
    t0 = pd.Timestamp(FT.OFFICIAL_BLOCKS["test"][0])
    for d in (va, fwd, live):
        assert not ((d.date >= t0) & (d.date < start)).any(), "a 2025 row would be predicted"
    tdates = np.sort(tr_rows.date.unique()); cut = int(len(tdates) * (1 - cfg["protocol"]["inner_es_share"]))
    fit = tr_rows[tr_rows.date <= tdates[cut - 1 - cfg["protocol"]["purge"]]]
    return {"data": data, "info": info, "full": full, "tr_rows": tr_rows, "va": va, "fwd": fwd, "live": live,
            "fit_symbols": sorted(fit.symbol.unique()), "channels": FP.channels(cfg["inputs"]), "news_stats": news_stats}


def key_frame(d: pd.DataFrame, part: str) -> pd.DataFrame:
    k = d[["symbol", "date"]].copy()
    k["date"] = k.date.dt.strftime("%Y-%m-%d")
    k.insert(2, "split", part)
    return k.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------- predict
def predict(args):
    import torch
    import train_v2xx as TV
    out_dir, fwd_root, wdir = Path(args.out_dir), Path(args.fwd_root), Path(args.weights)
    out_file = out_dir / "v2xx_forward_preds.csv.gz"
    if out_file.exists() and not args.redo:
        say(f"predict: {out_file.name} exists (pass --redo to predict again)"); return
    run = load_run(Path(args.run_json))
    cfg, seeds = run["config"], [r["seed"] for r in run["runs"]]
    t0 = time.time()
    B = build(cfg, fwd_root, args.news_fwd)
    if B["news_stats"]:
        say(f"2026 news: {B['news_stats']}")
    full, tr_rows, channels = B["full"], B["tr_rows"], B["channels"]
    say(f"{cfg['name']}: seeds {seeds}; rows val {len(B['va'])} / 2026 scored {len(B['fwd'])} / 2026 not yet scorable "
        f"{len(B['live'])}; {len(channels)} channels; built in {time.time() - t0:.0f}s; device {TV.DEVICE}")
    assert channels == run["channels"], "channels differ from the run"
    if not run.get("smoke"):                       # a --smoke run builds its vocabulary from a subsample of the fit rows
        assert B["fit_symbols"] == run["stock_vocab"], "stock vocabulary rebuilt from the fit rows differs from the run"
    lookback, bs = cfg["model"]["lookback"], cfg["training"]["batch_size"]
    if args.smoke_rows:                                                    # local plumbing test only
        for k in ("va", "fwd", "live"):
            B[k] = B[k].iloc[: args.smoke_rows]
    z = SQ.standardise_full(full, channels, tr_rows.index)                 # training rows only, as in training
    blank = full.copy()                                                    # news blanked from the validation start
    fill = NO_NEWS[cfg["inputs"]["sentiment_no_news"]]
    vmask = blank.date >= pd.Timestamp(FT.OFFICIAL_BLOCKS["val"][0])
    for c, v in zip(FP.SENT_CH, fill):
        blank.loc[vmask, c] = v
    zb = SQ.standardise_full(blank, channels, tr_rows.index)
    parts = {"val": (B["va"], z), "val_nonews": (B["va"], zb), "fwd": (B["fwd"], z), "live": (B["live"], z)}
    sectors = pd.read_csv(fwd_root / "data/sectors.csv")
    news_share = {k: round(float(full.loc[d.index, "has_news"].mean()), 4) for k, (d, _) in parts.items() if len(d)}

    frames = {k: key_frame(d, k) for k, (d, _) in parts.items()}
    for k, (d, _) in parts.items():
        for h in HS:
            frames[k][f"y_{h}"] = d[f"y_{h}"].to_numpy() if k != "live" else -1
            for c in ("fwd", "r", "sigma"):
                frames[k][f"{c}_{h}"] = d[f"{c}_{h}"].to_numpy() if f"{c}_{h}" in d else np.nan
        frames[k]["close_t"] = d["close"].to_numpy(float)
    meta = {"model": cfg["name"], "seeds": seeds, "news_share": news_share, "torch": torch.__version__,
            "device": str(TV.DEVICE), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "run_json": str(args.run_json), "weights": {}, "news_fwd": B["news_stats"]}
    vocab0 = None
    for seed in seeds:
        wf = wdir / f"seed{seed}.pt"
        ck = torch.load(wf, map_location="cpu", weights_only=False)        # the project's own checkpoint file
        meta["weights"][str(seed)] = {"file": wf.name, "sha256": sha(wf)[:16], "epochs_used": ck["epochs_used"]}
        vocab0 = vocab0 or ck["stock_vocab"]
        assert ck["stock_vocab"] == run["stock_vocab"] == vocab0, "stock vocabulary differs between run and checkpoint"
        r_sd = np.asarray(ck["r_sd"])
        assert np.allclose(r_sd, tr_rows[[f"fwd_{h}" for h in HS]].std().to_numpy(), rtol=0, atol=1e-12), "return scale differs"
        sec_id = ck["sector_ids"]
        frozen = {s: i + 1 for i, s in enumerate(sorted(pd.read_csv(ROOT / "data/sectors.csv").sector.unique()))}
        assert sec_id == frozen, "sector ids differ from the frozen sector table"
        sym2s = {s: i + 1 for i, s in enumerate(ck["stock_vocab"])}
        sym2g = dict(zip(sectors.symbol, sectors.sector.map(sec_id)))
        model = TV.build_model(cfg, len(channels), len(ck["stock_vocab"]), len(sec_id)).to(TV.DEVICE)
        cache = {}
        for j, h in enumerate(HS):
            st = ck["states"][j]
            sid = id(st)
            if sid not in cache:
                model.load_state_dict(st)
                cache[sid] = {}
                for k, (d, zz) in parts.items():
                    if not len(d):
                        cache[sid][k] = (np.zeros((0, len(HS), 3)), np.zeros((0, len(HS))))
                        continue
                    X = SQ.windows(full, zz, d.index, lookback)
                    S = d.symbol.map(sym2s).fillna(0).to_numpy(np.int64)
                    G = d.symbol.map(sym2g).fillna(0).to_numpy(np.int64)
                    P, R, _ = TV.predict(model, X, S, G, bs)
                    cache[sid][k] = (P, R)
            for k in parts:
                P, R = cache[sid][k]
                for c, cls in enumerate(L.CLASSES):
                    frames[k][f"v2xx_s{seed}_p{h}_{cls}"] = P[:, j, c].round(6)
                frames[k][f"v2xx_s{seed}_r{h}"] = (R[:, j] * r_sd[j]).round(6)
        say(f"  seed {seed}: predicted ({time.time() - t0:.0f}s)")
    meta["unknown_stock_rows"] = {k: int((~d.symbol.isin(vocab0)).sum()) for k, (d, _) in parts.items()}
    meta["unknown_sector_rows"] = {k: int((~d.symbol.isin(set(sectors.symbol))).sum()) for k, (d, _) in parts.items()}
    (out_dir / "predict_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    pd.concat(frames.values(), ignore_index=True).to_csv(out_file, index=False)
    say(f"wrote {out_file.name}: stocks with no training rows get the 'unknown' stock id; rows: {meta['unknown_stock_rows']}")


# ----------------------------------------------------------------------------------------------------- scoring
def probs(df, seed, h):
    return df[[f"v2xx_s{seed}_p{h}_{c}" for c in L.CLASSES]].to_numpy(float)


def ens_probs(df, seeds, h):
    return np.mean([probs(df, s, h) for s in seeds], axis=0)


def ms(xs):
    xs = list(xs)
    return round(statistics.mean(xs), 4), (round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0)


def f1(y, p):
    return PH.macro_f1(np.asarray(y, int), np.asarray(p, int))


def price_block(d, seeds, h):
    """R2_OS of the log return vs zero, and RMSE / MAE of the implied close in baht vs persistence."""
    r = d[f"fwd_{h}"].to_numpy(float); c = d["close_t"].to_numpy(float)
    out = {}
    for name, rp in [(f"s{s}", d[f"v2xx_s{s}_r{h}"].to_numpy(float)) for s in seeds] + \
                    [("ensemble", np.mean([d[f"v2xx_s{s}_r{h}"].to_numpy(float) for s in seeds], axis=0))]:
        ph, pa = c * np.exp(rp), c * np.exp(r)
        out[name] = {"r2_os_vs_zero": round(1 - float(np.mean((r - rp) ** 2)) / float(np.mean(r ** 2)), 5),
                     "price_rmse_baht": round(float(np.sqrt(np.mean((ph - pa) ** 2))), 4),
                     "price_mae_baht": round(float(np.mean(np.abs(ph - pa))), 4)}
    pa = c * np.exp(r)
    out["persistence"] = {"price_rmse_baht": round(float(np.sqrt(np.mean((c - pa) ** 2))), 4),
                          "price_mae_baht": round(float(np.mean(np.abs(c - pa))), 4)}
    per = [out[f"s{s}"]["r2_os_vs_zero"] for s in seeds]
    out["seed_mean_r2_os"] = ms(per)
    return out


def score(args):
    out_dir, fwd_root = Path(args.out_dir), Path(args.fwd_root)
    run = load_run(Path(args.run_json))
    cfg, seeds = run["config"], [r["seed"] for r in run["runs"]]
    name = cfg["name"]
    df = pd.read_csv(out_dir / "v2xx_forward_preds.csv.gz")
    val, vnb = df[df.split == "val"].reset_index(drop=True), df[df.split == "val_nonews"].reset_index(drop=True)
    fwd, live = df[df.split == "fwd"].reset_index(drop=True), df[df.split == "live"].reset_index(drop=True)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    info_k = manifest["identity_check"]["k"]
    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": name, "seeds": seeds,
           "weights": "saved kept weights of the run (no re-training)", "run_git_commit": run.get("git_commit"),
           "scored_on": "2026 forward window; 2025 test block not used", "forward_window": manifest["forward_window"],
           "pass_bar": {f"h{h}": FT.PASS_BAR[h] for h in HS},
           "smoke": bool(args.smoke_rows) or bool(run.get("smoke"))}

    # 1) reproduction: loaded weights vs the run's stored validation predictions
    stored = pd.read_csv(args.run_preds)
    stored["date"] = stored["date"].astype(str).str[:10]
    m = val.merge(stored, on=["symbol", "date"], suffixes=("", "_ref"), validate="one_to_one")
    rep = {"rows_here": len(val), "rows_stored": len(stored), "rows_matched": len(m), "per_horizon": {}}
    smoke = bool(args.smoke_rows) or bool(run.get("smoke"))            # a --smoke run stores only a validation subsample
    ok = len(m) == len(val) == len(stored) or (smoke and len(m) == len(stored))
    for h in HS:
        y = m[f"y_{h}"].to_numpy(int)
        assert (y == m[f"y_{h}_ref"].to_numpy(int)).all(), f"labels differ from the stored predictions at h={h}"
        dp = max(float(np.abs(probs(m, s, h) - m[[f"v2xx_s{s}_p{h}_{c}_ref" for c in L.CLASSES]].to_numpy(float)).max()) for s in seeds)
        mine = [f1(y, probs(m, s, h).argmax(1)) for s in seeds]
        theirs = [r["val"][f"h{h}"]["macro_f1"] for r in run["runs"]]
        df1 = max(abs(a - b) for a, b in zip(mine, theirs))
        rep["per_horizon"][f"h{h}"] = {"max_abs_prob_diff": round(dp, 6), "f1_here": [round(x, 4) for x in mine],
                                       "f1_stored": theirs, "max_f1_diff": round(df1, 4)}
        ok &= dp <= REPRO_PROB_TOL and df1 <= REPRO_F1_TOL
    rep["ok"] = bool(ok)
    res["reproduction"] = rep
    say(f"\nREPRODUCTION ({name} saved weights vs its stored validation predictions):")
    for h in HS:
        v = rep["per_horizon"][f"h{h}"]
        say(f"  h{h}: max |prob diff| {v['max_abs_prob_diff']:.2e}; F1 here {v['f1_here']} vs stored {v['f1_stored']}")
    if not ok:
        (out_dir / "forward_summary.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        raise SystemExit("stop: the loaded weights do not reproduce the stored validation predictions; the 2026 inputs "
                         "would not be the ones the model was trained on. Check the commit and the weights folder.")

    # 2) validation with news blanked (the 2026 condition)
    nb = {}
    for h in HS:
        y = val[f"y_{h}"].to_numpy(int)
        a = [f1(y, probs(val, s, h).argmax(1)) for s in seeds]
        b = [f1(y, probs(vnb, s, h).argmax(1)) for s in seeds]
        sh = lambda d: [round(float(np.mean([(probs(d, s, h).argmax(1) == i).mean() for s in seeds])), 4) for i in range(3)]
        nb[f"h{h}"] = {"f1_news": ms(a)[0], "f1_news_blanked": ms(b)[0], "change": round(ms(b)[0] - ms(a)[0], 4),
                       "call_shares_news (down/flat/up)": sh(val), "call_shares_news_blanked": sh(vnb),
                       "actual_shares": [round(float((y == i).mean()), 4) for i in range(3)]}
    res["validation_news_blanked"] = nb
    pm = json.loads((out_dir / "predict_meta.json").read_text(encoding="utf-8"))
    res["news_share"], res["unknown_stock_rows"] = pm["news_share"], pm["unknown_stock_rows"]
    res["news_fwd"] = pm.get("news_fwd")
    res["news_2026"] = "scored 2026 headlines" if pm.get("news_fwd") else "none (every 2026 day gets the no-news fill)"
    say(f"share of stock-days with news: {pm['news_share']}" + (f"; 2026 news file: {pm['news_fwd']}" if pm.get("news_fwd") else ""))

    # 3) comparison models on the same 2026 rows (optional: the main model v2 forward test folder)
    comp = {}
    if args.v2_forward and (Path(args.v2_forward) / "forward_preds.csv.gz").exists():
        v2 = pd.read_csv(Path(args.v2_forward) / "forward_preds.csv.gz")
        v2s = sorted({int(x.group(1)) for c in v2.columns if (x := re.fullmatch(r"v2_s(\d+)_class1", c))})
        mm = fwd[["symbol", "date"]].merge(v2[["symbol", "date"] + [f"v2_s{s}_class{h}" for s in v2s for h in HS]],
                                          on=["symbol", "date"], how="left", validate="one_to_one")
        if mm.filter(like="v2_s").notna().all().all():
            comp["v2"] = {h: np.stack([mm[f"v2_s{s}_class{h}"].to_numpy(int) for s in v2s]) for h in HS}
        tr_path = Path(args.v2_forward) / "trees.csv.gz"
        if tr_path.exists():
            t = pd.read_csv(tr_path)
            t = t[t.split == "fwd"]
            xs = sorted({int(x.group(1)) for c in t.columns if (x := re.fullmatch(r"xgb_s(\d+)_p1_down", c))})
            tm = fwd[["symbol", "date"]].merge(t, on=["symbol", "date"], how="left", validate="one_to_one")
            if tm.filter(like="xgb_s").notna().all().all():
                comp["xgb"] = {h: np.stack([tm[[f"xgb_s{s}_p{h}_{c}" for c in L.CLASSES]].to_numpy(float).argmax(1) for s in xs]) for h in HS}
    res["comparisons_available"] = sorted(comp)

    # 4) 2026 scores
    res["horizons"] = {}
    dates = fwd["date"].to_numpy()
    tr_sh = manifest["forward_window"]["train_class_shares"]
    new_syms = json.loads((fwd_root / "forward_config.json").read_text(encoding="utf-8"))["new_symbols"]
    stored_val = run.get("summary", {}).get("model", {})
    stored_ens = run.get("seed_ensemble", {}).get("val", {})
    for h in HS:
        y = fwd[f"y_{h}"].to_numpy(int)
        past = L.classify(fwd[f"r_{h}"].to_numpy(), fwd[f"sigma_{h}"].to_numpy(), info_k[f"h{h}"])
        seed_cls = np.stack([probs(fwd, s, h).argmax(1) for s in seeds])
        ens_cls = ens_probs(fwd, seeds, h).argmax(1)
        q = np.array([tr_sh[f"h{h}"][c] for c in L.CLASSES]); pv = np.bincount(y, minlength=3) / len(y)
        floors = {"trend_continues": round(f1(y, past), 4), "trend_reverses": round(f1(y, 2 - past), 4),
                  "stratified_random_expected": round(float(np.mean(2 * pv * q / (pv + q))), 4)}
        sc = {"model": ms(f1(y, p) for p in seed_cls), "ensemble": round(f1(y, ens_cls), 4)}
        for k_, v in comp.items():
            sc[k_] = ms(f1(y, p) for p in v[h])
        best_floor = max(floors["trend_continues"], floors["trend_reverses"])
        fw_bar = round(max(best_floor, sc["xgb"][0]) + FT.MARGIN, 4) if "xgb" in sc else None
        boot_in = {"model": seed_cls, "ensemble": ens_cls[None, :], "trend_continues": past[None, :], "trend_reverses": (2 - past)[None, :]}
        boot_in.update({k_: v[h] for k_, v in comp.items()})
        pairs = [("model", k_) for k_ in ("v2", "xgb") if k_ in comp] + [("model", "trend_continues"), ("model", "trend_reverses"),
                                                                        ("ensemble", "model")]
        boot = PH.block_bootstrap(dates, y, boot_in, pairs=pairs, bar=FT.PASS_BAR[h])
        cm = sum(np.bincount(y * 3 + p, minlength=9).reshape(3, 3) for p in seed_cls)
        cm_v2 = sum(np.bincount(y * 3 + p, minlength=9).reshape(3, 3) for p in comp["v2"][h]) if "v2" in comp else None
        mon = pd.Series(pd.to_datetime(dates).strftime("%Y-%m"))
        monthly = {mo: {"rows": int((mon == mo).sum()),
                        "model": round(statistics.mean(f1(y[mon == mo], p[mon == mo]) for p in seed_cls), 4),
                        "trend_continues": round(f1(y[mon == mo], past[mon == mo]), 4),
                        "trend_reverses": round(f1(y[mon == mo], 2 - past[mon == mo]), 4)}
                   for mo in sorted(mon.unique())}
        new = fwd.symbol.isin(new_syms).to_numpy()
        hv = stored_val.get(f"h{h}", {}).get("macro_f1", {}).get("mean")
        res["horizons"][f"h{h}"] = {
            "macro_f1": {k_: ({"mean": v[0], "std": v[1]} if isinstance(v, tuple) else v) for k_, v in sc.items()},
            "floors": floors, "official_bar": FT.PASS_BAR[h], "passes_official_bar": sc["model"][0] >= FT.PASS_BAR[h],
            "forward_window_bar": fw_bar, "passes_forward_window_bar": (sc["model"][0] >= fw_bar) if fw_bar else None,
            "validation_stored": {"model": hv, "ensemble": stored_ens.get(f"h{h}", {}).get("macro_f1")},
            "model_minus_validation": round(sc["model"][0] - hv, 4) if hv is not None else None,
            "bootstrap": boot, "confusion_pooled_over_seeds": cm.tolist(),
            "confusion_v2_pooled_over_seeds": None if cm_v2 is None else cm_v2.tolist(),
            "confusion_layout": "rows actual down/flat/up, cols predicted",
            "class_shares_actual": {c: round(float(pv[i]), 4) for i, c in enumerate(L.CLASSES)},
            "class_shares_predicted": {c: round(float((seed_cls == i).mean()), 4) for i, c in enumerate(L.CLASSES)},
            "monthly": monthly, "price": price_block(fwd, seeds, h),
            "new_tickers": ({"rows": int(new.sum()), "model": round(statistics.mean(f1(y[new], p[new]) for p in seed_cls), 4)}
                            if new.any() else None)}

    # 5) live calls (outcomes not known yet): seed ensemble
    lo = live[["symbol", "date"]].copy()
    for h in HS:
        p = ens_probs(live, seeds, h)
        lo[f"call_{h}d"] = [L.CLASSES[i] for i in p.argmax(1)]
        for c, cls in enumerate(L.CLASSES):
            lo[f"p{h}_{cls}"] = p[:, c].round(4)
        lo[f"ret_{h}d"] = np.mean([live[f"v2xx_s{s}_r{h}"].to_numpy(float) for s in seeds], axis=0).round(5)
    lo.to_csv(out_dir / "live_calls.csv", index=False)
    res["live"] = {"rows": int(len(live)), "last_date": str(live.date.max()) if len(live) else None}
    (out_dir / "forward_summary.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    fw = manifest["forward_window"]
    say(f"\n{name} ON THE 2026 FORWARD WINDOW {fw['first']} .. {fw['last']} ({len(fwd)} stock-days, {fwd.date.nunique()} dates). macro-F1:")
    say(f"{'h':4s} {'model (3 seeds)':>16s} {'ensemble':>9s} {'v2':>7s} {'xgb':>7s} {'cont.':>7s} {'rev.':>7s} {'random':>7s} {'bar':>6s} "
        f"{'95% interval':>17s} {'vs val':>7s}")
    for h in HS:
        r = res["horizons"][f"h{h}"]
        mf = r["macro_f1"]
        g = lambda k_: f"{mf[k_]['mean']:7.4f}" if k_ in mf else f"{'-':>7s}"
        say(f"h{h:<3d} {mf['model']['mean']:9.4f}±{mf['model']['std']:.4f} {mf['ensemble']:9.4f} {g('v2')} {g('xgb')} "
            f"{r['floors']['trend_continues']:7.4f} {r['floors']['trend_reverses']:7.4f} {r['floors']['stratified_random_expected']:7.4f} "
            f"{FT.PASS_BAR[h]:6.3f} {str(r['bootstrap']['values']['model']['ci95']):>17s} "
            f"{(r['model_minus_validation'] if r['model_minus_validation'] is not None else float('nan')):+7.4f}")
    say("calls down/flat/up vs actual: " + "; ".join(
        f"h{h} {'/'.join(f'{v:.0%}' for v in res['horizons'][f'h{h}']['class_shares_predicted'].values())} vs "
        f"{'/'.join(f'{v:.0%}' for v in res['horizons'][f'h{h}']['class_shares_actual'].values())}" for h in HS))
    say("validation, news on -> blanked: " + "; ".join(f"h{h} {nb[f'h{h}']['f1_news']:.3f} -> {nb[f'h{h}']['f1_news_blanked']:.3f}" for h in HS))
    say("price R2_OS (seed mean): " + "; ".join(f"h{h} {res['horizons'][f'h{h}']['price']['seed_mean_r2_os'][0]:+.4f}" for h in HS))
    plots(res, out_dir, name)
    say(f"wrote {out_dir / 'forward_summary.json'}, live_calls.csv, forward_scores.png, forward_monthly.png, forward_confusion.png")


SEQ_BLUE = ["#f7fafe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # light -> dark


def confusion_heatmap(ax, cm, title: str, show_ylabels: bool = True):
    """3x3 confusion grid: colour = share of each ACTUAL class (row-normalised, 0-100%), text = share and count.
    The diagonal is each class's recall. Rows = actual down / flat / up, columns = called."""
    from matplotlib.colors import LinearSegmentedColormap
    cm = np.asarray(cm, float)
    share = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ink = "#ffffff" if share[i, j] > 0.55 else "#1f2328"
            ax.text(j, i, f"{share[i, j]:.0%}\n{int(cm[i, j]):,}", ha="center", va="center", fontsize=9, color=ink,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(3)); ax.set_xticklabels(["down", "flat", "up"], fontsize=9, color="#6e7781")
    ax.set_yticks(range(3)); ax.set_yticklabels(["down", "flat", "up"] if show_ylabels else [], fontsize=9, color="#6e7781")
    ax.set_xlabel("called", fontsize=9, color="#6e7781")
    if show_ylabels:
        ax.set_ylabel("actual", fontsize=9, color="#6e7781")
    ax.set_title(title, loc="left", fontsize=10, color="#1f2328")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True); ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=2); ax.tick_params(which="minor", length=0)


def confusion_figure(res, out_dir: Path, name: str):
    import matplotlib.pyplot as plt
    rows = [(f"{name}", "confusion_pooled_over_seeds")]
    if res["horizons"]["h1"].get("confusion_v2_pooled_over_seeds"):
        rows.append(("main model v2", "confusion_v2_pooled_over_seeds"))
    fig, axs = plt.subplots(len(rows), 3, figsize=(11, 3.6 * len(rows)), squeeze=False)
    for r, (lab, key) in enumerate(rows):
        for c, h in enumerate(HS):
            hz = res["horizons"][f"h{h}"]
            f1v = hz["macro_f1"]["model" if key.startswith("confusion_pooled") else "v2"]["mean"]
            confusion_heatmap(axs[r, c], hz[key], f"{lab}, {h}-day (macro-F1 {f1v:.3f})", show_ylabels=c == 0)
    fig.suptitle("2026 forward window: share of each actual class by call (3 seeds pooled; diagonal = recall)",
                 x=0.01, ha="left", fontsize=11, color="#1f2328")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out_dir / "forward_confusion.png", dpi=140); plt.close(fig)


def plots(res, out_dir: Path, name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#1f2328", "#6e7781", "#e6e8eb"
    c_main, c_ref, c_floor = "#2a6fdb", "#9aa4b1", "#d97706"
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ax, h in zip(axs, HS):
        r = res["horizons"][f"h{h}"]
        b = r["bootstrap"]["values"]
        items = [(f"{name} (3 seeds)", "model", c_main), (f"{name} ensemble", "ensemble", c_main), ("main model v2", "v2", c_ref),
                 ("XGBoost", "xgb", c_ref), ("trend continues", "trend_continues", c_floor), ("trend reverses", "trend_reverses", c_floor)]
        items = [x for x in items if x[1] in b]
        for i, (lab, k_, col) in enumerate(items[::-1]):
            ax.plot(b[k_]["ci95"], [i, i], color=col, lw=2, solid_capstyle="round")
            ax.plot(b[k_]["point"], i, "o", color=col, ms=7, mfc=col if k_ != "ensemble" else "white")
        ax.axvline(FT.PASS_BAR[h], color=ink, ls="--", lw=1)
        if r["validation_stored"]["model"]:
            ax.axvline(r["validation_stored"]["model"], color=c_main, ls=":", lw=1)
        ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items[::-1]] if h == 1 else [])
        ax.set_title(f"{h}-day", color=ink, fontsize=11, loc="left")
        ax.grid(axis="x", color=grid); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=muted)
        ax.set_xlabel("macro-F1 (dot) and 95% date-block interval", color=muted, fontsize=9)
    fig.text(0.01, 0.01, f"dashed = pass bar; dotted blue = {name} on the official validation window", color=muted, fontsize=9)
    fig.suptitle(f"{name} on the 2026 forward window", x=0.01, ha="left", color=ink, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(out_dir / "forward_scores.png", dpi=140); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
    for ax, h in zip(axs, HS):
        mo = res["horizons"][f"h{h}"]["monthly"]
        ks = list(mo)
        ax.plot(ks, [mo[k]["model"] for k in ks], "-o", color=c_main, ms=4, label=name)
        ax.plot(ks, [max(mo[k]["trend_continues"], mo[k]["trend_reverses"]) for k in ks], "-", color=c_floor, lw=1.2,
                label="better one-rule floor")
        ax.axhline(FT.PASS_BAR[h], color=ink, ls="--", lw=1)
        ax.set_title(f"{h}-day", color=ink, fontsize=11, loc="left")
        ax.tick_params(axis="x", rotation=45, colors=muted); ax.tick_params(axis="y", colors=muted)
        ax.grid(axis="y", color=grid); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(frameon=False, fontsize=9)
    axs[0].set_ylabel("macro-F1 by month", color=muted)
    fig.tight_layout(); fig.savefig(out_dir / "forward_monthly.png", dpi=140); plt.close(fig)
    confusion_figure(res, out_dir, name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["predict", "score"])
    ap.add_argument("--fwd-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-json", required=True, help="the run's result file, e.g. Senior/results/v2xx/A100/v2_03.json")
    ap.add_argument("--run-preds", default=None, help="the run's validation predictions (default: <run>_preds.csv.gz)")
    ap.add_argument("--weights", default=None, help="the run's weights folder (default: <run>_weights/)")
    ap.add_argument("--v2-forward", default=None, help="main model v2 forward-test folder, for v2 and XGBoost on the same rows")
    ap.add_argument("--news-fwd", default=None, help="2026 headline scores (scripts/sentiment_score_forward.py); default none")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--smoke-rows", type=int, default=0, help="local plumbing test: predict only this many rows per part")
    args = ap.parse_args(argv)
    rj = Path(args.run_json)
    args.run_preds = args.run_preds or str(rj.with_name(rj.stem + "_preds.csv.gz"))
    args.weights = args.weights or str(rj.with_name(rj.stem + "_weights"))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    {"predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
