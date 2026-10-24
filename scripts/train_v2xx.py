"""Train and score a v2.xx experiment, main model v3, or the faithful PatchTST baseline from one JSON config.

  python scripts/train_v2xx.py --config configs/v2_01.json --out results/v2xx/v2_01.json
  python scripts/train_v2xx.py --config configs/baseline_patchtst.json --out results/v2xx/baseline_patchtst.json
  python scripts/train_v2xx.py --config ... --out ... --eval-start 2022-07-01 --eval-end 2022-12-31   # walk-forward fold

* Models: model.family = proposal_msp (src/proposal_model.py, committee slides) or patchtst_faithful
  (src/patchtst_faithful.py, Nie et al. 2023). Inputs: src/features_proposal.py (slide 9).
* Same sample rows, labels (k x sigma_h fitted on train), split and 10-day purges as every earlier run, via
  scripts/baselines_tabular.build_dataset, so the scores are directly comparable with E00-E31 and main model v2.
* Validation block only; the test block stays sealed (no test-block value is read, not even inputs, except what
  build_dataset already holds in memory). Early stopping uses the inner hold-out, never the validation block.
* Trend: macro-F1, accuracy, confusion grid. Price (proposal scope 5): RMSE / MAE / R2_OS of the log return, and RMSE /
  MAE of the implied closing price C_t * exp(r_hat) in baht, each beside its zero-return (persistence) floor.
* v2.02 options: cosine learning rate, EMA weights (BatchNorm recomputed), selection "f1_up_to_ce_min", no early
  stopping, per-epoch hold-out diagnostics (cross-entropy, call shares, call agreement), seed ensemble.
* Saves the weights of every seed. On a GPU, reports the peak memory per epoch and stops with a clear message if
  memory runs out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import baselines_tabular as BT  # noqa: E402
import features_proposal as FP  # noqa: E402
import labels as L  # noqa: E402
import metrics as M  # noqa: E402
import run_config as RC  # noqa: E402
import sequences as SQ  # noqa: E402
import stopping as STOP  # noqa: E402
import train_utils as TU  # noqa: E402
from patchtst_faithful import FaithfulPatchTST  # noqa: E402
from proposal_model import ProposalMultiScalePatchTST, make_loss  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEWS_FILE = "data/derived/sentiment_scores.csv"
SMOKE = dict(fit=3000, es=800, val=1500, max_epochs=2)          # --smoke: pipeline check only, never a result


def build_model(cfg: dict, n_channels: int, n_stocks: int, n_sectors: int) -> nn.Module:
    m, hz = cfg["model"], tuple(cfg["protocol"]["horizons"])
    if m["family"] == "patchtst_faithful":
        return FaithfulPatchTST(n_channels=n_channels, lookback=m["lookback"], patch_len=m["patch_len"], stride=m["stride"],
                                d_model=m["d_model"], n_heads=m["n_heads"], n_layers=m["n_layers"], d_ff=m["d_ff"],
                                dropout=m["dropout"], attn_dropout=m["attn_dropout"], head_dropout=m["head_dropout"], horizons=hz)
    return ProposalMultiScalePatchTST(
        n_channels=n_channels, lookback=m["lookback"], patch_lengths=m["patch_lengths"], patch_strides=m["patch_strides"],
        channel_mode=m["channel_mode"], revin=m["revin"], encoder_norm=m["encoder_norm"], d_model=m["d_model"],
        n_heads=m["n_heads"], n_layers=m["n_layers"], d_ff=m["d_ff"], dropout=m["dropout"], fusion_hidden=m["fusion_hidden"],
        n_stocks=n_stocks, d_stock=m["d_stock"], n_sectors=n_sectors, d_sector=m["d_sector"], trunk_width=m["trunk_width"],
        head_hidden=tuple(m["head_hidden"]), head_dropout=m["head_dropout"], horizons=hz)


def param_breakdown(model: nn.Module) -> dict:
    if isinstance(model, FaithfulPatchTST):
        parts = {"revin": model.revin, "patch_embedding": model.W_P, "encoder": model.encoder,
                 "price_head": model.price_head, "trend_head": model.trend_head}
        out = {k: sum(p.numel() for p in v.parameters()) for k, v in parts.items()}
        out["position_embedding"] = model.W_pos.numel()
    else:
        parts = {"revin": model.revin, "scale_branches": model.branches, "scale_fusion": model.fusion,
                 "stock_embedding": model.stock_emb, "sector_embedding": model.sector_emb, "linear_layer": model.linear,
                 "price_head": model.price_head, "trend_head": model.trend_head}
        out = {k: sum(p.numel() for p in v.parameters()) if v is not None else 0 for k, v in parts.items()}
    out["total"] = sum(p.numel() for p in model.parameters())
    return out


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE)


def predict(model, X, S, G, bs: int):
    """bs = the training batch size, so prediction never needs more GPU memory than a training step. Eval mode: dropout off and BatchNorm uses its stored training statistics, so a row's prediction never
    depends on the other rows of its batch."""
    model.eval()
    P, R, W = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o = model(_t(X[i:i + bs]), _t(S[i:i + bs]), _t(G[i:i + bs]))
            P.append(torch.softmax(o["trend_logits"], -1).cpu().numpy())      # softmax per horizon (slide 13)
            R.append(o["returns"].cpu().numpy())
            if "scale_weights" in o:
                W.append(o["scale_weights"].cpu().numpy())
    return np.concatenate(P), np.concatenate(R), (np.concatenate(W) if W else None)


def macro_f1_each(Y, P):
    return [M.trend_metrics(Y[:, j], P[:, j].argmax(-1))["macro_f1"] for j in range(Y.shape[1])]


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()


def build_data(cfg: dict, news_path: Path, fold: dict | None = None, smoke: bool = False) -> dict:
    """Rows, inputs, labels and ids exactly as training sees them. Separate from main() so tests/test_v2xx_pipeline.py
    can rebuild it from scrambled prices or news and compare the windows."""
    data, info, full = BT.build_dataset(return_full=True, extra="none", fold=fold)
    full = FP.add_price_tech(full, cfg["inputs"])
    sent = FP.sentiment_probs(pd.read_csv(news_path), full[["symbol", "date"]], cfg["inputs"]["sentiment_no_news"])
    key = pd.MultiIndex.from_frame(full[["symbol", "date"]])
    for c in FP.SENT_CH:
        full[c] = sent[c].reindex(key).to_numpy()
    has_news = sent["has_news"].reindex(key).to_numpy()
    channels = FP.channels(cfg["inputs"])
    lookback = cfg["model"]["lookback"]

    tr_rows, va = data[data.split == "train"], data[data.split == "val"]
    tdates = np.sort(tr_rows.date.unique()); cut = int(len(tdates) * (1 - cfg["protocol"]["inner_es_share"]))
    fit_end, es_start = tdates[cut - 1 - cfg["protocol"]["purge"]], tdates[cut]
    fit, es = tr_rows[tr_rows.date <= fit_end], tr_rows[tr_rows.date >= es_start]
    if smoke:
        r = np.random.default_rng(0)
        fit, es, va = (d.loc[np.sort(r.choice(d.index, min(n, len(d)), replace=False))]
                       for d, n in ((fit, SMOKE["fit"]), (es, SMOKE["es"]), (va, SMOKE["val"])))

    z = SQ.standardise_full(full, channels, tr_rows.index)                    # "normalised": train rows only
    missing = {c: round(float(full.loc[tr_rows.index, c].isna().mean()), 4) for c in channels}   # training rows only
    r_sd = tr_rows[[f"fwd_{h}" for h in L.HORIZONS]].std().to_numpy()       # price target scale, train only
    stock_vocab = sorted(fit.symbol.unique())                                # ids 1..n; no fit rows -> 0 (unknown)
    sym2s = {s: i + 1 for i, s in enumerate(stock_vocab)}
    sectors = pd.read_csv(ROOT / "data/sectors.csv")
    sec_id = {s: i + 1 for i, s in enumerate(sorted(sectors.sector.unique()))}
    sym2g = dict(zip(sectors.symbol, sectors.sector.map(sec_id)))
    D = {}
    for k, d in (("fit", fit), ("es", es), ("val", va)):
        D[k] = {"X": SQ.windows(full, z, d.index, lookback),
                "Y": d[[f"y_{h}" for h in L.HORIZONS]].to_numpy(np.int64),
                "R": (d[[f"fwd_{h}" for h in L.HORIZONS]].to_numpy() / r_sd).astype(np.float32),
                "S": d.symbol.map(sym2s).fillna(0).to_numpy(np.int64),
                "G": d.symbol.map(sym2g).fillna(0).to_numpy(np.int64),
                "date": d.date.to_numpy(), "symbol": d.symbol.to_numpy()}
    return {"data": data, "info": info, "tr_rows": tr_rows, "va": va, "fit": fit, "es": es, "D": D,
            "channels": channels, "missing": missing, "r_sd": r_sd, "stock_vocab": stock_vocab, "sec_id": sec_id,
            "has_news": has_news, "fit_end": fit_end, "es_start": es_start}


def _cpu_state(st):
    return {k: v.detach().cpu().clone() for k, v in st.items()}


def train_one(cfg, seed, D, n_channels, n_stocks, n_sectors, max_epochs, save_path: Path | None = None):
    """Train one seed and return the kept weights.

    v2.01 behaviour: constant lr, raw weights, kept epoch = best 3-horizon mean macro-F1, early stopping.
    v2.02 options (config "training"): cosine lr per step; EMA weights (scored, kept and saved; BatchNorm statistics
    recomputed on the fit rows before every hold-out evaluation); selection f1_up_to_ce_min; no early stopping.
    Every time the kept weights change they are written at once to save_path (on Drive in Colab); the printout marks it."""
    tr = cfg["training"]
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    model = build_model(cfg, n_channels, n_stocks, n_sectors).to(DEVICE)
    loss_fn = make_loss(cfg["loss"]).to(DEVICE)
    opt = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=tr["lr"], weight_decay=tr["weight_decay"])
    f = D["fit"]
    spe = TU.steps_per_epoch(len(f["X"]), tr["batch_size"])
    sched = TU.make_scheduler(opt, tr, spe, max_epochs)
    ema = TU.EmaWeights(model, tr["ema_decay"]) if tr["weight_averaging"] == "ema" else None
    rng = np.random.default_rng(seed)
    drop_rng = np.random.default_rng(seed + 2)                      # stock_id dropout (own stream)
    p_drop = cfg["model"].get("stock_id_dropout", 0.0)
    nh = D["es"]["Y"].shape[1]
    sel = STOP.Selector(tr["selection"])
    shared = tr["checkpoint"] == "shared"
    early = tr["early_stopping"]
    prefix_state = kept_state = None
    bad, hist, prev_calls = 0, [], None
    best_h, best_ep_h, state_h, bad_h = [-1.0] * nh, [0] * nh, [None] * nh, [0] * nh

    def fit_batches(bs):                                            # BatchNorm recompute pass: real ids, fixed order
        for i in range(0, len(f["X"]), bs):
            yield _t(f["X"][i:i + bs]), _t(f["S"][i:i + bs]), _t(f["G"][i:i + bs])

    for ep in range(1, max_epochs + 1):
        model.train(); t0 = time.time()
        order = rng.permutation(len(f["X"])); tot, parts_sum = 0.0, {}
        for i in range(0, len(order), tr["batch_size"]):
            b = order[i:i + tr["batch_size"]]
            sb = f["S"][b].copy()
            if p_drop > 0:
                sb[drop_rng.random(len(b)) < p_drop] = 0
            out = model(_t(f["X"][b]), _t(sb), _t(f["G"][b]))
            loss, parts = loss_fn(out, _t(f["Y"][b]), _t(f["R"][b]))
            opt.zero_grad(); loss.backward()
            if tr["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                ema.update(model)
            tot += loss.item() * len(b)
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v * len(b)
        lr_now = opt.param_groups[0]["lr"]
        if ema is not None:                                          # score the averaged weights
            TU.recompute_bn(ema.model, fit_batches(tr["batch_size"]))
            eval_model = ema.model
        else:
            eval_model = model
        e = D["es"]
        P, R, _ = predict(eval_model, e["X"], e["S"], e["G"], tr["batch_size"])
        es_each = macro_f1_each(e["Y"], P)
        es = float(np.mean(es_each))
        diag = TU.holdout_diagnostics(P, e["Y"], prev_calls)
        prev_calls = diag["calls"]
        es_rmse = [float(np.sqrt(np.mean((R[:, j] - e["R"][:, j]) ** 2))) for j in range(nh)]   # standardised units
        improved = STOP.update_per_horizon(best_h, bad_h, es_each)
        for j in improved:
            best_ep_h[j] = ep
            state_h[j] = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}
        flags = sel.update(ep, es, diag["ce_mean"])
        if flags["prefix_improved"]:
            prefix_state = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if flags["kept_changed"]:
            kept_state = prefix_state
        kept_changed = flags["kept_changed"] if shared else bool(improved)
        if save_path is not None and kept_changed:
            torch.save({"seed": seed, "saved_at_epoch": ep, "checkpoint": tr["checkpoint"], "config": cfg,
                        "weights": "ema" if ema is not None else "raw", "best_epoch": sel.kept_epoch,
                        "best_inner_es_macro_f1": sel.kept_f1, "best_epoch_per_horizon": best_ep_h,
                        "best_inner_es_per_horizon": best_h,
                        "states": [_cpu_state(kept_state)] if shared else [_cpu_state(x) for x in state_h]}, save_path)
        row = {"epoch": ep, "lr_end": lr_now, "train_loss": round(tot / len(f["X"]), 5),
               **{f"train_{k}": round(v / len(f["X"]), 5) for k, v in parts_sum.items()},
               "es_macro_f1": round(es, 4), "es_per_horizon": [round(s, 4) for s in es_each],
               "es_ce_per_horizon": [round(c, 5) for c in diag["ce"]], "es_ce_mean": round(diag["ce_mean"], 5),
               "es_call_shares": [[round(x, 4) for x in s] for s in diag["call_shares"]],
               "es_call_agreement": None if diag["agreement"] is None else [round(a, 4) for a in diag["agreement"]],
               "es_return_rmse_std": [round(s, 4) for s in es_rmse], "new_best_f1": flags["prefix_improved"],
               "new_min_ce": flags["ce_improved"], "kept_epoch": sel.kept_epoch,
               "improved_horizons": [L.HORIZONS[j] for j in improved], "stall_per_horizon": list(bad_h),
               "sec": round(time.time() - t0, 1)}
        if torch.cuda.is_available():
            row["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        hist.append(row)
        per_h = " ".join(f"{h}d {sc:.3f}{'*' if j in improved else ' '}" for j, (h, sc) in enumerate(zip(L.HORIZONS, es_each)))
        up = "/".join(f"{s[2]:.0%}" for s in diag["call_shares"])
        agr = "-" if diag["agreement"] is None else "/".join(f"{a:.0%}" for a in diag["agreement"])
        notes = [x for x, on in (("best F1 so far", flags["prefix_improved"]), ("lowest CE so far", flags["ce_improved"])) if on]
        mark = (f"  <== KEPT = epoch {sel.kept_epoch} (F1 {sel.kept_f1:.4f}){' - saved' if save_path is not None else ''}"
                if flags["kept_changed"] else f"      kept still epoch {sel.kept_epoch}")
        stall = f" | stall {'/'.join(map(str, bad_h))} of {tr['patience']}" if early else ""
        print(f"    seed {seed} ep {ep:>2}: lr {lr_now:.1e} loss {tot / len(f['X']):.4f} | ES F1 {es:.4f} [{per_h}] "
              f"CE {diag['ce_mean']:.4f} | up calls {up} same as last {agr}{stall}"
              + (f" ({', '.join(notes)})" if notes else "") + f"{mark}  ({time.time() - t0:.0f}s"
              + (f", GPU peak {row['gpu_peak_gb']} GB" if "gpu_peak_gb" in row else "") + ")", flush=True)
        if early and STOP.should_stop(tr["stop_rule"], bad, bad_h, tr["patience"]):
            print(f"    seed {seed}: stopped after epoch {ep} - "
                  + ("every horizon" if tr["stop_rule"] == "every" else "the 3-horizon mean")
                  + f" went {tr['patience']} epochs without improving; weights kept from epoch {sel.kept_epoch}"
                  + ("" if shared else f" (per horizon: {best_ep_h})"), flush=True)
            break
    else:
        print(f"    seed {seed}: ran all {max_epochs} epochs; weights kept from epoch {sel.kept_epoch}"
              + (f" (lowest CE at epoch {sel.ce_min_epoch})" if tr["selection"] == "f1_up_to_ce_min" else "")
              + ("" if shared else f" (per horizon: {best_ep_h})"), flush=True)
    meta = {"best_epoch": sel.kept_epoch, "best_inner_es_macro_f1": round(sel.kept_f1, 4), "epochs_run": len(hist),
            "selection": tr["selection"], "weights_scored": "ema" if ema is not None else "raw",
            "best_f1_epoch_overall": sel.prefix_epoch, "best_f1_overall": round(sel.prefix_f1, 4),
            "es_actual_shares": [(np.bincount(D["es"]["Y"][:, j], minlength=3) / len(D["es"]["Y"])).round(4).tolist() for j in range(nh)],
            "history": hist, "best_epoch_per_horizon": best_ep_h, "best_inner_es_per_horizon": [round(b, 4) for b in best_h]}
    if tr["selection"] == "f1_up_to_ce_min":
        meta["ce_min_epoch"], meta["ce_min"] = sel.ce_min_epoch, round(sel.ce_min, 5)
    if cfg["loss"]["weighting"] == "kendall":
        meta["loss_weights"] = {"w_trend": round(float(torch.exp(-loss_fn.log_var[0].detach())), 4),
                                "w_price": round(float(torch.exp(-loss_fn.log_var[1].detach())), 4)}
    states = [kept_state] * nh if shared else state_h
    epochs_used = [sel.kept_epoch] * nh if shared else best_ep_h
    return model, meta, states, epochs_used


OOM_HELP = ("GPU memory ran out. Options (each is Siwat's call): a larger GPU (A100 80 GB), bf16 mixed precision, "
            "gradient accumulation (note: BatchNorm statistics are then per micro-batch), or a smaller batch size.")


def finish(res: dict, preds: pd.DataFrame, cfg: dict) -> None:
    """Per-seed summary (mean +- std) and, when configured, the seed ensemble, computed from the saved predictions
    (probabilities and return forecasts rounded to 6 decimals), so a sequential run and a merge of parallel per-seed
    runs (scripts/merge_v2xx.py) give the same numbers. Writes res["summary"], res["seed_ensemble"] and ensemble
    columns in preds."""
    runs = res["runs"]
    summ = {}
    for h in L.HORIZONS:
        summ[f"h{h}"] = {k: {"mean": round(statistics.mean(r["val"][f"h{h}"][k] for r in runs), 5),
                             "std": round(statistics.stdev(r["val"][f"h{h}"][k] for r in runs), 5) if len(runs) > 1 else None}
                         for k in ("macro_f1", "accuracy", "rmse", "mae", "r2_os_vs_zero", "price_rmse_baht", "price_mae_baht")}
    res["summary"] = M.report(summ, res["floors"])
    if not (cfg["report"]["seed_ensemble"] and len(runs) > 1):     # reported alongside the per-seed mean +- std
        return
    seeds = [r["seed"] for r in runs]
    close_t = preds["close_t"].to_numpy(float)
    ens = {"how": f"mean of the {len(runs)} seeds' kept probabilities (argmax) and return forecasts", "seeds": seeds, "val": {}}
    for h in L.HORIZONS:
        Y, Rv = preds[f"y_{h}"].to_numpy(), preds[f"fwd_{h}"].to_numpy(float)
        Pm = np.mean([preds[[f"v2xx_s{sd}_p{h}_{c}" for c in L.CLASSES]].to_numpy(float) for sd in seeds], axis=0)
        rm = np.mean([preds[f"v2xx_s{sd}_r{h}"].to_numpy(float) for sd in seeds], axis=0)
        yh = Pm.argmax(-1)
        ph, pa = close_t * np.exp(rm), close_t * np.exp(Rv)
        ens["val"][f"h{h}"] = {**M.trend_metrics(Y, yh), "confusion": confusion(Y, yh), **M.return_metrics(Rv, rm),
                               "price_rmse_baht": round(float(np.sqrt(np.mean((ph - pa) ** 2))), 4),
                               "price_mae_baht": round(float(np.mean(np.abs(ph - pa))), 4)}
        preds[f"v2xx_ens_class{h}"] = yh
        for c, cls in enumerate(L.CLASSES):
            preds[f"v2xx_ens_p{h}_{cls}"] = Pm[:, c].round(6)
        preds[f"v2xx_ens_r{h}"] = rm.round(6)
    res["seed_ensemble"] = ens
    print("  seed ensemble: " + "  ".join(f"h{h} F1 {ens['val'][f'h{h}']['macro_f1']:.3f}" for h in L.HORIZONS), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--news", default=NEWS_FILE)
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="override protocol.seeds (still >= 3 unless --smoke)")
    ap.add_argument("--eval-start", default=None, help="walk-forward fold: first evaluation date (with --eval-end)")
    ap.add_argument("--eval-end", default=None)
    ap.add_argument("--save-weights", default=None, help="directory for the weights (default: <out>_weights/)")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 2 epochs: checks the pipeline, never a result")
    ap.add_argument("--part", action="store_true",
                    help="one seed of a parallel run (fewer than 3 seeds allowed); merge the parts with scripts/merge_v2xx.py")
    args = ap.parse_args(argv)
    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = RC.load(cfg_path)
    seeds = args.seeds or cfg["protocol"]["seeds"]
    if len(seeds) < 3 and not (args.smoke or args.part):
        ap.error("at least 3 seeds (hard rule 5); for one seed of a parallel run use --part")
    max_epochs = SMOKE["max_epochs"] if args.smoke else cfg["training"]["max_epochs"]
    print(f"config {cfg['name']}: {json.dumps({k: cfg[k] for k in ('inputs', 'model', 'loss', 'training')})}", flush=True)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU {p.name}, {p.total_memory / 2 ** 30:.1f} GB", flush=True)

    B = build_data(cfg, ROOT / args.news, BT.fold_arg(args), args.smoke)
    D, channels, info, va, tr_rows, r_sd = B["D"], B["channels"], B["info"], B["va"], B["tr_rows"], B["r_sd"]
    fit, es, stock_vocab, sec_id = B["fit"], B["es"], B["stock_vocab"], B["sec_id"]
    Rva = va[[f"fwd_{h}" for h in L.HORIZONS]].to_numpy()
    close_t = va["close"].to_numpy(float)                                    # raw close on day t
    print(f"rows: fit {len(fit)} / inner-ES {len(es)} / val {len(va)}; window {cfg['model']['lookback']} x {len(channels)}; "
          f"stocks {len(stock_vocab)} (+unknown), sectors {len(sec_id)} (+unknown)", flush=True)

    floors = {}
    for j, h in enumerate(L.HORIZONS):
        ytr, yva = tr_rows[f"y_{h}"].to_numpy(), D["val"]["Y"][:, j]
        past = L.classify(va[f"r_{h}"].to_numpy(), va[f"sigma_{h}"].to_numpy(), info[f"h{h}"]["k"])
        q, pv = np.bincount(ytr, minlength=3) / len(ytr), np.bincount(yva, minlength=3) / len(yva)
        p0 = close_t; p1 = close_t * np.exp(Rva[:, j])
        floors[f"h{h}"] = {"majority_class": M.trend_metrics(yva, np.full_like(yva, M.majority_class(ytr))),
                           "stratified_random_expected": {"macro_f1": round(float(np.mean(2 * pv * q / (pv + q))), 4)},
                           "trend_continues": M.trend_metrics(yva, past) | {"confusion": confusion(yva, past)},
                           "trend_reverses": M.trend_metrics(yva, 2 - past) | {"confusion": confusion(yva, 2 - past)},
                           "persistence_zero_return": M.return_metrics(Rva[:, j], np.zeros(len(Rva))),
                           "persistence_price_baht": {"rmse": round(float(np.sqrt(np.mean((p1 - p0) ** 2))), 4),
                                                      "mae": round(float(np.mean(np.abs(p1 - p0))), 4)}}

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wdir = Path(args.save_weights) if args.save_weights else out_path.with_name(out_path.stem + "_weights")
    wdir.mkdir(parents=True, exist_ok=True)
    pred_path = out_path.with_name(out_path.stem + "_preds.csv.gz")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": BT.git_commit(),
           "script_sha256": sha(__file__), "model_sha256": sha(ROOT / "src/proposal_model.py"),
           "faithful_sha256": sha(ROOT / "src/patchtst_faithful.py"), "encoders_sha256": sha(ROOT / "src/encoders.py"),
           "features_sha256": sha(ROOT / "src/features_proposal.py"), "config_sha256": sha(cfg_path), "config": cfg,
           "seeds": seeds, "smoke": bool(args.smoke), "torch": torch.__version__, "device": str(DEVICE),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "evaluated_on": "validation only (test sealed)" if args.eval_start is None else f"walk-forward fold {args.eval_start}..{args.eval_end}",
           "channels": channels, "channel_missing_share_on_training_rows": B["missing"],
           "news_share": {k: round(float(B["has_news"][d.index].mean()), 4) for k, d in (("fit", fit), ("es", es), ("val", va))},
           "stock_vocab": stock_vocab, "val_rows_unknown_stock": int((D["val"]["S"] == 0).sum()),
           "inner_es": {"fit_end": str(pd.Timestamp(B["fit_end"]).date()), "es_start": str(pd.Timestamp(B["es_start"]).date()),
                        "rows_fit": len(fit), "rows_es": len(es)},
           "data": info, "floors": floors, "part": bool(args.part), "runs": []}
    preds = va[["symbol", "date"]].reset_index(drop=True).copy()
    for j, h in enumerate(L.HORIZONS):
        preds[f"y_{h}"] = D["val"]["Y"][:, j]; preds[f"fwd_{h}"] = Rva[:, j]
    preds["close_t"] = close_t
    groups = FP.groups(cfg["inputs"])

    for seed in seeds:
        t0 = time.time()
        try:
            model, meta, states, epochs_used = train_one(cfg, seed, D, len(channels), len(stock_vocab), len(sec_id), max_epochs,
                                                       save_path=wdir / f"seed{seed}_best_so_far.pt")
        except torch.cuda.OutOfMemoryError as e:                             # clear stop instead of a long traceback
            raise SystemExit(f"{OOM_HELP}\n(seed {seed}; {e})")
        torch.save({"config": cfg, "stock_vocab": stock_vocab, "sector_ids": sec_id, "r_sd": r_sd.tolist(),
                    "epochs_used": epochs_used, "states": [{k: v.cpu() for k, v in s.items()} for s in states]},
                   wdir / f"seed{seed}.pt")
        per_h, cache, weights_by_h = {}, {}, {}
        for j, h in enumerate(L.HORIZONS):
            sid = id(states[j])
            if sid not in cache:
                model.load_state_dict(states[j]); cache[sid] = predict(model, D["val"]["X"], D["val"]["S"], D["val"]["G"], cfg["training"]["batch_size"])
            P, R, W = cache[sid]
            yhat = P[:, j].argmax(-1)
            rp = R[:, j] * r_sd[j]
            ph, pa = close_t * np.exp(rp), close_t * np.exp(Rva[:, j])
            per_h[f"h{h}"] = {**M.trend_metrics(D["val"]["Y"][:, j], yhat), "confusion": confusion(D["val"]["Y"][:, j], yhat),
                              **M.return_metrics(Rva[:, j], rp),
                              "corr_with_outcome": round(float(np.corrcoef(rp, Rva[:, j])[0, 1]), 4) if np.std(rp) > 0 else None,
                              "price_rmse_baht": round(float(np.sqrt(np.mean((ph - pa) ** 2))), 4),
                              "price_mae_baht": round(float(np.mean(np.abs(ph - pa))), 4),
                              "epoch_used": epochs_used[j]}
            weights_by_h[f"h{h}"] = W
            preds[f"v2xx_s{seed}_class{h}"] = yhat
            for c, cls in enumerate(L.CLASSES):
                preds[f"v2xx_s{seed}_p{h}_{cls}"] = P[:, j, c].round(6)
            preds[f"v2xx_s{seed}_r{h}"] = rp.round(6)
        run = {"seed": seed, "seconds": round(time.time() - t0, 1), "params": param_breakdown(model), **meta, "val": per_h}
        W = weights_by_h["h1"]                                       # scale weights of the horizon-1 checkpoint
        if W is not None:
            scale_w = {"overall": [round(float(v), 4) for v in W.mean(axis=(0, 1))]}
            if cfg["model"].get("channel_mode") == "independent":
                for g, chs in groups.items():
                    idx = [channels.index(c) for c in chs]
                    scale_w[g] = [round(float(v), 4) for v in W[:, idx, :].mean(axis=(0, 1))]
            run[f"scale_weights_mean (patch {cfg['model']['patch_lengths']}; horizon-1 checkpoint)"] = scale_w
        res["runs"].append(run)
        print(f"  seed {seed}: " + "  ".join(f"h{h} F1 {per_h[f'h{h}']['macro_f1']:.3f} R2 {per_h[f'h{h}']['r2_os_vs_zero']:+.4f}"
                                          for h in L.HORIZONS), flush=True)
        out_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        preds.to_csv(pred_path, index=False)

    finish(res, preds, cfg)
    if res.get("seed_ensemble"):
        preds.to_csv(pred_path, index=False)
    out_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path}, {pred_path} and weights in {wdir}")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as e:                                 # anywhere, incl. prediction
        raise SystemExit(f"{OOM_HELP}\n({e})")
