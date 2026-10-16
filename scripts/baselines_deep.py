"""Deep baselines on real data: LSTM and single-scale PatchTST, same rows / split / labels as baselines_tabular.

Both get the same inputs as XGBoost (price + 6 indicators, 18 daily channels, look-back 60) and the same
two heads (trend up/flat/down and log return, at 1/5/10 days) trained with learned uncertainty weights
(Kendall et al. 2018) -- the decisions of 2026-10-15. The PatchTST baseline is the project's own
MultiScalePatchTST with ONE branch, patch 16 / stride 8 (Nie et al. 2023 defaults), so the later
multi-scale comparison differs in exactly one thing. RevIN on (hard rule 8).

Fixed settings, no tuning. Early stopping uses an inner hold-out = the last 10% of TRAINING dates
(after a 10-day purge); the validation block is never used for any choice. Test stays sealed.

  python scripts/baselines_deep.py [--models lstm patchtst] [--seeds ...] [--max-epochs 15] [--lr 1e-3] [--dropout 0.1] [--patience 3]
  python scripts/baselines_deep.py --models multiscale_sector --news data/derived/sentiment_scores.csv \
      --lr 1e-4 --dropout 0.2 --max-epochs 30 --patience 5 --out results/main_model_v1_news_p41.json   # Phase 4.1
  python scripts/baselines_deep.py --models multiscale_sector --out data/derived/main_model_v0.json   # main model, no sentiment yet

Each run also stores a 3x3 confusion grid per horizon (rows = actual down/flat/up, columns = predicted) and writes
every validation prediction (class probabilities and return forecasts, per seed) to <out>_preds.csv.gz.
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
import features as F  # noqa: E402
import labels as L  # noqa: E402
import metrics as M  # noqa: E402
import sequences as SQ  # noqa: E402
import stopping as STOP  # noqa: E402
from multiscale_patchtst import MultiScalePatchTST, UncertaintyWeightedLoss  # noqa: E402

LOOKBACK = 60
SEEDS = (20261015, 20261016, 20261017)
# Defaults = the v1 run (commit 0da8473), so the logged v0/v1 commands still reproduce. Phase 4.1 overrides
# lr / dropout / max-epochs / patience on the command line; every run records the values it used under "settings".
SET = dict(lr=1e-3, dropout=0.1, weight_decay=0.01, batch_size=256, max_epochs=15, patience=3, inner_es_share=0.10, purge=10)
# Phase 4 switches (experiment ledger E10, E11, E16). Defaults reproduce every earlier run exactly.
OPT = dict(revin=True, cls_only=False, label_smoothing=0.0, stop_rule="mean",   # stop_rule: E19 (src/stopping.py)
           revin_stats=False)                                                     # E23: window mean / log-std re-injected


class LSTMBaseline(nn.Module):
    """Two-layer LSTM over the window; last hidden state -> the same two heads as the PatchTST models."""
    def __init__(self, n_channels, horizons=(1, 5, 10), n_classes=3, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, layers, batch_first=True, dropout=dropout)
        self.trunk = nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(dropout))
        self.h, self.k = len(horizons), n_classes
        self.trend_head, self.return_head = nn.Linear(128, self.h * n_classes), nn.Linear(128, self.h)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        z = self.trunk(hn[-1])
        return {"trend_logits": self.trend_head(z).reshape(len(x), self.h, self.k), "returns": self.return_head(z)}


MAIN_PATCHES = ((5, 5), (20, 10))      # Phase 1.11 decision, 2026-10-15
SMALL = dict(d_model=32, n_heads=4, n_layers=1, d_ff=64, d_channel=16)   # E04a (experiment ledger), vs default 64/4/2/128/64


def make_model(name, n_channels, n_groups=0, dropout=0.1):
    m = _make_model(name, n_channels, n_groups, dropout)
    if isinstance(m, MultiScalePatchTST) and not OPT["revin"]:
        m.revin = None                           # E10: no per-window normalisation (inputs stay train-scaled)
    return m


def _make_model(name, n_channels, n_groups=0, dropout=0.1):
    if name == "lstm":
        return LSTMBaseline(n_channels, dropout=dropout)
    if name == "patchtst":                       # baseline: single scale, PatchTST defaults
        return MultiScalePatchTST(n_channels, revin_stats=OPT["revin_stats"], lookback=LOOKBACK, patches=((16, 8),), dropout=dropout)
    if name == "multiscale":                     # main architecture without the sector label
        return MultiScalePatchTST(n_channels, revin_stats=OPT["revin_stats"], lookback=LOOKBACK, patches=MAIN_PATCHES, dropout=dropout)
    if name == "multiscale_sector":              # main model (Phase 1.11-1.14 decisions)
        return MultiScalePatchTST(n_channels, revin_stats=OPT["revin_stats"], lookback=LOOKBACK, patches=MAIN_PATCHES, n_groups=n_groups, dropout=dropout)
    if name == "multiscale_sector_small":        # E04a: same architecture, ~1/10 of the weights (fixed, not searched)
        return MultiScalePatchTST(n_channels, revin_stats=OPT["revin_stats"], lookback=LOOKBACK, patches=MAIN_PATCHES, n_groups=n_groups, dropout=dropout,
                                  **SMALL)
    if name == "singlescale_sector":             # ablation partner: only the multi-scale part removed
        return MultiScalePatchTST(n_channels, revin_stats=OPT["revin_stats"], lookback=LOOKBACK, patches=((16, 8),), n_groups=n_groups, dropout=dropout)
    raise ValueError(name)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Colab GPU when present, else CPU


class ClsOnlyLoss(nn.Module):
    """E11: cross-entropy on the trend classes only; the return head gets no gradient. Same interface as
    UncertaintyWeightedLoss (log_var kept, frozen at 0, so the run JSON's loss_weights read 1.0 / 1.0)."""
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out, trend, returns):
        l_cls = self.ce(out["trend_logits"].reshape(-1, out["trend_logits"].shape[-1]), trend.reshape(-1))
        return l_cls, {"ce": l_cls.item()}


def make_loss():
    if OPT["cls_only"]:
        return ClsOnlyLoss(OPT["label_smoothing"])
    loss_fn = UncertaintyWeightedLoss()
    if OPT["label_smoothing"]:
        loss_fn.ce = nn.CrossEntropyLoss(label_smoothing=OPT["label_smoothing"])   # E16
    return loss_fn


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE)


def _call(model, xb, gb):
    return model(xb, gb) if getattr(model, "group_emb", None) is not None else model(xb)


def predict(model, X, G=None, bs=2048):
    model.eval()
    outs_p, outs_r = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o = _call(model, _t(X[i:i + bs]), None if G is None else _t(G[i:i + bs]))
            outs_p.append(torch.softmax(o["trend_logits"], -1).cpu().numpy()); outs_r.append(o["returns"].cpu().numpy())
    return np.concatenate(outs_p), np.concatenate(outs_r)


def confusion(y_true, y_pred):
    """3x3 counts, rows = actual (down, flat, up), columns = predicted (down, flat, up)."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()


def macro_f1_mean(Y, P):
    return float(np.mean(macro_f1_each(Y, P)))


def macro_f1_each(Y, P):
    return [M.trend_metrics(Y[:, j], P[:, j].argmax(-1))["macro_f1"] for j in range(Y.shape[1])]


def train_one(name, seed, Xfit, Yfit, Rfit, Xes, Yes, Gfit=None, Ges=None, n_groups=0):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_model(name, Xfit.shape[-1], n_groups, SET["dropout"]).to(DEVICE)
    loss_fn = make_loss().to(DEVICE)
    opt = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=SET["lr"], weight_decay=SET["weight_decay"])
    rng = np.random.default_rng(seed)
    best, best_state, best_ep, bad, hist = -1.0, None, 0, 0, []
    nh = Yes.shape[1]                       # E12: best epoch tracked per horizon too (does not change training)
    best_h, best_ep_h, state_h, bad_h = [-1.0] * nh, [0] * nh, [None] * nh, [0] * nh
    for ep in range(1, SET["max_epochs"] + 1):
        model.train(); t0 = time.time()
        order = rng.permutation(len(Xfit)); tot = 0.0
        for i in range(0, len(order), SET["batch_size"]):
            b = order[i:i + SET["batch_size"]]
            out = _call(model, _t(Xfit[b]), None if Gfit is None else _t(Gfit[b]))
            loss, _ = loss_fn(out, _t(Yfit[b]), _t(Rfit[b]))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(b)
        es_each = macro_f1_each(Yes, predict(model, Xes, Ges)[0])
        es = float(np.mean(es_each))
        for j in STOP.update_per_horizon(best_h, bad_h, es_each):
            best_ep_h[j] = ep
            state_h[j] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        hist.append({"epoch": ep, "train_loss": round(tot / len(Xfit), 5), "es_macro_f1": round(es, 4),
                     "es_per_horizon": [round(float(s), 4) for s in es_each], "sec": round(time.time() - t0, 1)})
        print(f"    {name} seed {seed} ep {ep}: loss {tot / len(Xfit):.4f}  inner-ES macro-F1 {es:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if es > best:
            best, best_ep, bad = es, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if STOP.should_stop(OPT["stop_rule"], bad, bad_h, SET["patience"]):
            break
    model.load_state_dict(best_state)
    meta = {"best_epoch": best_ep, "best_inner_es_macro_f1": round(best, 4), "epochs_run": len(hist), "history": hist,
            "best_epoch_per_horizon": best_ep_h, "best_inner_es_per_horizon": [round(b, 4) for b in best_h],
            "loss_weights": {"w_ce": round(float(torch.exp(-loss_fn.log_var[0].detach())), 4), "w_mse": round(float(torch.exp(-loss_fn.log_var[1].detach())), 4)}}
    return model, meta, state_h


def main(argv=None):
    global LOOKBACK                      # E14: --lookback overrides the module default
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["lstm", "patchtst"])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--max-epochs", type=int, default=SET["max_epochs"])
    ap.add_argument("--lr", type=float, default=SET["lr"], help="AdamW learning rate (v1: 1e-3; Phase 4.1: 1e-4)")
    ap.add_argument("--dropout", type=float, default=SET["dropout"], help="dropout in every model (v1: 0.1; Phase 4.1: 0.2)")
    ap.add_argument("--patience", type=int, default=SET["patience"], help="early-stopping patience in epochs on the inner hold-out")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="data/derived/baselines_deep.json")
    ap.add_argument("--news", default=None, help="add daily sentiment channels from this scores file (e.g. data/derived/sentiment_scores.csv)")
    ap.add_argument("--preds", default=None, help="validation predictions file (default: <out>_preds.csv.gz)")
    ap.add_argument("--extra-features", default="none", choices=sorted(BT.EXTRA), help="e03: add src/features_xs.py inputs")
    ap.add_argument("--eval-start", default=None, help="E05 walk-forward fold: first evaluation date (with --eval-end)")
    ap.add_argument("--eval-end", default=None, help="E05 walk-forward fold: last evaluation date")
    ap.add_argument("--no-revin", action="store_true", help="E10: switch off per-window normalisation (RevIN)")
    ap.add_argument("--cls-only", action="store_true", help="E11: train on the trend classes only (no return loss)")
    ap.add_argument("--label-smoothing", type=float, default=0.0, help="E16: cross-entropy label smoothing (e.g. 0.1)")
    ap.add_argument("--stop-rule", default="mean", choices=STOP.RULES,
                    help="E19: 'every' = stop only when every horizon has stalled for --patience epochs")
    ap.add_argument("--lookback", type=int, default=LOOKBACK, help="E14: look-back window in trading days (default 60)")
    ap.add_argument("--revin-stats", action="store_true",
                    help="E23: keep RevIN and feed each input's window mean and log-std to the heads (needs RevIN on)")
    args = ap.parse_args(argv)
    LOOKBACK = args.lookback
    if args.revin_stats and args.no_revin:
        ap.error("--revin-stats re-injects what RevIN removes; it cannot be combined with --no-revin")
    OPT.update(revin=not args.no_revin, cls_only=args.cls_only, label_smoothing=args.label_smoothing, stop_rule=args.stop_rule,
               revin_stats=args.revin_stats)
    SET.update(max_epochs=args.max_epochs, lr=args.lr, dropout=args.dropout, patience=args.patience)
    print("settings: " + ", ".join(f"{k}={SET[k]}" for k in ("lr", "dropout", "max_epochs", "patience")) + f", options={OPT}", flush=True)
    if args.threads:
        torch.set_num_threads(args.threads)

    data, info, full = BT.build_dataset(return_full=True, extra=args.extra_features, fold=BT.fold_arg(args))
    channels = list(F.FEATURES) + BT.EXTRA[args.extra_features]
    if args.news:                                   # daily sentiment channels, known at the close of each day
        import news_beyond_price as NB
        nf = NB.news_features(pd.read_csv(ROOT / args.news), full[["symbol", "date"]])
        key = pd.MultiIndex.from_frame(full[["symbol", "date"]])
        for c in ("sent_mean", "sent_n", "has_news"):
            full[c] = nf[c].reindex(key).to_numpy()
        channels += ["sent_mean", "sent_n", "has_news"]
    tr, va = data[data.split == "train"], data[data.split == "val"]
    # inner early-stopping block: last 10% of training dates, purged
    tdates = np.sort(tr.date.unique()); cut = int(len(tdates) * (1 - SET["inner_es_share"]))
    fit_end, es_start = tdates[cut - 1 - SET["purge"]], tdates[cut]
    fit, es = tr[tr.date <= fit_end], tr[tr.date >= es_start]

    z = SQ.standardise_full(full, channels, tr.index)            # scaler sees training rows only
    X = {k: SQ.windows(full, z, d.index, LOOKBACK) for k, d in (("fit", fit), ("es", es), ("val", va))}
    Y = {k: d[[f"y_{h}" for h in L.HORIZONS]].to_numpy(np.int64) for k, d in (("fit", fit), ("es", es), ("val", va))}
    r_sd = tr[[f"fwd_{h}" for h in L.HORIZONS]].std().to_numpy()   # return target scale from train only
    Rfit = (fit[[f"fwd_{h}" for h in L.HORIZONS]].to_numpy() / r_sd).astype(np.float32)
    Rva = va[[f"fwd_{h}" for h in L.HORIZONS]].to_numpy()
    sectors = pd.read_csv(ROOT / "data/sectors.csv")
    sec_id = {s: i + 1 for i, s in enumerate(sorted(sectors.sector.unique()))}       # 0 = unknown
    sym2g = dict(zip(sectors.symbol, sectors.sector.map(sec_id)))
    G = {k: d.symbol.map(sym2g).fillna(0).to_numpy(np.int64) for k, d in (("fit", fit), ("es", es), ("val", va))}
    print(f"rows: fit {len(fit)} / inner-ES {len(es)} / val {len(va)}; window {LOOKBACK} x {len(channels)}", flush=True)

    floors = {}
    for j, h in enumerate(L.HORIZONS):
        ytr, yva = tr[f"y_{h}"].to_numpy(), Y["val"][:, j]
        past = L.classify(va[f"r_{h}"].to_numpy(), va[f"sigma_{h}"].to_numpy(), info[f"h{h}"]["k"])
        q, pv = np.bincount(ytr, minlength=3) / len(ytr), np.bincount(yva, minlength=3) / len(yva)
        floors[f"h{h}"] = {"majority_class": M.trend_metrics(yva, np.full_like(yva, M.majority_class(ytr))),
                           "stratified_random_expected": {"macro_f1": round(float(np.mean(2 * pv * q / (pv + q))), 4)},
                           "trend_continues": M.trend_metrics(yva, past) | {"confusion": confusion(yva, past)},
                           "trend_reverses": M.trend_metrics(yva, 2 - past) | {"confusion": confusion(yva, 2 - past)},
                           "persistence_zero_return": M.return_metrics(Rva[:, j], np.zeros(len(Rva)))}

    res = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": BT.git_commit(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "torch": torch.__version__, "threads": torch.get_num_threads(), "device": str(DEVICE),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "evaluated_on": "validation only (test sealed)",
           "settings": SET | {"lookback": LOOKBACK, "channels": channels, "news_file": args.news, "options": dict(OPT)},
           "inner_es": {"fit_end": str(pd.Timestamp(fit_end).date()), "es_start": str(pd.Timestamp(es_start).date()),
                        "rows_fit": len(fit), "rows_es": len(es)},
           "data": info, "floors": floors, "models": {}}
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path = ROOT / args.preds if args.preds else out_path.with_name(out_path.stem + "_preds.csv.gz")
    preds = va[["symbol", "date"]].reset_index(drop=True).copy()      # one row per validation stock-day
    for j, h in enumerate(L.HORIZONS):
        preds[f"y_{h}"] = Y["val"][:, j]; preds[f"fwd_{h}"] = Rva[:, j]
    res["confusion_layout"] = "rows = actual (down, flat, up); columns = predicted (down, flat, up); validation block"
    res["preds_file"] = pred_path.name
    for name in args.models:
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            model, meta, state_h = train_one(name, seed, X["fit"], Y["fit"], Rfit, X["es"], Y["es"], G["fit"], G["es"], len(sec_id))
            P, R = predict(model, X["val"], G["val"])
            per_h = {}
            for j, h in enumerate(L.HORIZONS):
                rp = R[:, j] * r_sd[j]
                yhat = P[:, j].argmax(-1)
                per_h[f"h{h}"] = {**M.trend_metrics(Y["val"][:, j], yhat),
                                  **M.return_metrics(Rva[:, j], rp), "corr_with_outcome": round(float(np.corrcoef(rp, Rva[:, j])[0, 1]), 4),
                                  "confusion": confusion(Y["val"][:, j], yhat)}
                preds[f"{name}_s{seed}_class{h}"] = yhat                      # exact predicted class (0 down, 1 flat, 2 up)
                for c, cls in enumerate(L.CLASSES):
                    preds[f"{name}_s{seed}_p{h}_{cls}"] = P[:, j, c].round(6)
                preds[f"{name}_s{seed}_r{h}"] = rp.round(6)
            n_params = sum(p.numel() for p in model.parameters())
            per_h_best = {}                  # E12: score each horizon with the epoch that was best for that horizon
            for j, h in enumerate(L.HORIZONS):
                model.load_state_dict(state_h[j])
                Pj = predict(model, X["val"], G["val"])[0][:, j]
                yj = Pj.argmax(-1)
                per_h_best[f"h{h}"] = {**M.trend_metrics(Y["val"][:, j], yj), "confusion": confusion(Y["val"][:, j], yj),
                                       "best_epoch": meta["best_epoch_per_horizon"][j]}
                for c, cls in enumerate(L.CLASSES):
                    preds[f"{name}_ph_s{seed}_p{h}_{cls}"] = Pj[:, c].round(6)
            runs.append({"seed": seed, "seconds": round(time.time() - t0, 1),
                         "params": n_params, **meta, "val": per_h, "val_per_horizon_best": per_h_best})
            print(f"  {name} seed {seed}: " + "  ".join(f"h{h} F1 {per_h[f'h{h}']['macro_f1']:.3f}" for h in L.HORIZONS), flush=True)
            res["models"][name] = {"runs": runs}
            out_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")   # save after every run
            preds.to_csv(pred_path, index=False)
        summ = {}
        for h in L.HORIZONS:
            summ[f"h{h}"] = {k: {"mean": round(statistics.mean(r["val"][f"h{h}"][k] for r in runs), 5),
                                 "std": round(statistics.stdev(r["val"][f"h{h}"][k] for r in runs), 5) if len(runs) > 1 else None}
                             for k in ("macro_f1", "accuracy", "r2_os_vs_zero", "corr_with_outcome")}
        res["models"][name]["summary"] = M.report(summ, floors)
        out_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path} and {pred_path}")


if __name__ == "__main__":
    main()
