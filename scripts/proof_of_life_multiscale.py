"""Proof-of-life for the Multi-Scale PatchTST (milestone M3 risk; register Phase 1.11-1.14).

Question: can the multi-scale model be built and trained end to end (forward pass, both heads,
weighted multi-task loss, a few epochs) without errors, NaNs or a flat loss? It runs on SYNTHETIC
data with a planted signal, so the numbers say nothing about SET50; they only show the plumbing
works and the model can pick up a signal that is there.

Synthetic market: 6 "stocks", 1,000 days, 6 channels [log return, volume z, oscillator, trend gauge,
sentiment, pure noise]. Next-day return = 0.25 * sentiment(t) + 0.15 * sign(20-day momentum) + noise,
so the signal sits in one exogenous channel and in a 20-day pattern.
Labels: cumulative log return at h = 1, 5, 10; trend classes by terciles of TRAIN returns per h
(the k rule is still open, Phase 1.3); return targets standardised with TRAIN statistics.
Split: chronological 70/15/15 per stock. Configurations x 3 seeds: multi-scale patches (5,5)+(20,10)
and single-scale (16,8). Baselines: majority class (trend) and zero return (returns).

  python scripts/proof_of_life_multiscale.py            # CPU, a few minutes
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multiscale_patchtst import MultiScalePatchTST, UncertaintyWeightedLoss  # noqa: E402

LOOKBACK, HORIZONS = 60, (1, 5, 10)
CONFIGS = {"multi_scale_5_20": ((5, 5), (20, 10)), "single_scale_16": ((16, 8),)}
SEEDS = (1, 2, 3)
EPOCHS, BATCH, LR = 6, 128, 1e-3


def synthetic_market(n_stocks=6, n_days=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_stocks):
        sent = np.convolve(rng.normal(0, 1, n_days), np.ones(3) / 3, mode="same")
        r = np.zeros(n_days)
        for t in range(1, n_days):
            mom = r[max(0, t - 20):t].mean() if t > 1 else 0.0
            r[t] = 0.0025 * sent[t - 1] + 0.0015 * np.sign(mom) + rng.normal(0, 0.01)
        price = np.cumsum(r)
        vol = rng.normal(0, 1, n_days) + 2 * np.abs(r) / 0.01
        osc = np.tanh(np.convolve(r, np.ones(14) / 14, mode="same") / 0.003)
        gauge = price - np.convolve(price, np.ones(20) / 20, mode="same")
        noise = rng.normal(0, 1, n_days)
        out.append(np.stack([r, vol, osc, gauge, sent, noise], axis=1).astype(np.float32))
    return out


def windows(series):
    X, Y = [], []
    for s in series:
        r = s[:, 0]
        for t in range(LOOKBACK - 1, len(s) - max(HORIZONS)):
            X.append(s[t - LOOKBACK + 1:t + 1])                       # window ends at t (no future rows)
            Y.append([r[t + 1:t + 1 + h].sum() for h in HORIZONS])    # cumulative return after t
    return np.stack(X), np.array(Y, dtype=np.float32)


def split_chrono(series):
    parts = {"train": [], "val": [], "test": []}
    for s in series:
        n = len(s); a, b = int(0.70 * n), int(0.85 * n)
        parts["train"].append(s[:a]); parts["val"].append(s[a - LOOKBACK + 1:b]); parts["test"].append(s[b - LOOKBACK + 1:])
    return {k: windows(v) for k, v in parts.items()}


def macro_f1(y, p, k=3):
    f = []
    for c in range(k):
        tp = ((y == c) & (p == c)).sum(); fp = ((y != c) & (p == c)).sum(); fn = ((y == c) & (p != c)).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f))


def run(config, seed, data):
    torch.manual_seed(seed); np.random.seed(seed)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = data["train"], data["val"], data["test"]
    cuts = np.quantile(Ytr, [1 / 3, 2 / 3], axis=0)                    # terciles from TRAIN only
    lab = lambda Y: np.stack([np.digitize(Y[:, j], cuts[:, j]) for j in range(len(HORIZONS))], 1)  # 0 down,1 flat,2 up
    mu, sd = Ytr.mean(0), Ytr.std(0)                                   # TRAIN statistics only
    z = lambda Y: (Y - mu) / sd
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    Xtr_t, Ctr, Rtr = t(Xtr), t(lab(Ytr), torch.long), t(z(Ytr))
    model = MultiScalePatchTST(n_channels=Xtr.shape[2], lookback=LOOKBACK, patches=CONFIGS[config], horizons=HORIZONS)
    crit = UncertaintyWeightedLoss()
    opt = torch.optim.AdamW(list(model.parameters()) + list(crit.parameters()), lr=LR, weight_decay=0.01)
    hist, best, best_state = [], -1, None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); perm = torch.randperm(len(Xtr_t)); tot = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss, parts = crit(model(Xtr_t[b]), Ctr[b], Rtr[b])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(b)
        va = evaluate(model, Xva, lab(Yva), z(Yva))
        hist.append({"epoch": ep, "train_loss": round(tot / len(Xtr_t), 4), **{k: round(v, 4) for k, v in va.items()}, **{k: round(v, 3) for k, v in parts.items()}})
        if va["macro_f1_mean"] > best:
            best, best_state = va["macro_f1_mean"], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    te = evaluate(model, Xte, lab(Yte), z(Yte))
    maj = np.bincount(lab(Ytr).reshape(-1), minlength=3).argmax()
    base = {"majority_macro_f1": float(np.mean([macro_f1(lab(Yte)[:, j], np.full(len(Yte), maj)) for j in range(len(HORIZONS))]))}
    return {"config": config, "seed": seed, "params": sum(p.numel() for p in model.parameters()), "seconds": round(time.time() - t0, 1),
            "history": hist, "test": {k: round(v, 4) for k, v in te.items()}, "baseline_test": {k: round(v, 4) for k, v in base.items()},
            "loss_fell": hist[-1]["train_loss"] < hist[0]["train_loss"]}


@torch.no_grad()
def evaluate(model, X, C, R):
    model.eval()
    out = model(torch.tensor(X))
    pred = out["trend_logits"].argmax(-1).numpy()
    f1 = [macro_f1(C[:, j], pred[:, j]) for j in range(len(HORIZONS))]
    rhat = out["returns"].numpy()
    r2 = [1 - ((R[:, j] - rhat[:, j]) ** 2).sum() / (R[:, j] ** 2).sum() for j in range(len(HORIZONS))]   # vs zero forecast (standardised)
    return {"macro_f1_mean": float(np.mean(f1)), **{f"macro_f1_h{h}": float(v) for h, v in zip(HORIZONS, f1)},
            **{f"r2_vs_zero_h{h}": float(v) for h, v in zip(HORIZONS, r2)}}


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    data = split_chrono(synthetic_market())
    print({k: v[0].shape for k, v in data.items()})
    runs = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            r = run(cfg, seed, data)
            runs.append(r)
            print(f"{cfg:18s} seed {seed}: params {r['params']:,}  {r['seconds']:5.1f}s  loss {r['history'][0]['train_loss']:.3f}->{r['history'][-1]['train_loss']:.3f}  "
                  f"test macro-F1 {r['test']['macro_f1_mean']:.3f} (majority {r['baseline_test']['majority_macro_f1']:.3f})  "
                  f"R2 vs zero h1 {r['test']['r2_vs_zero_h1']:+.3f} h5 {r['test']['r2_vs_zero_h5']:+.3f}")
    summary = {}
    for cfg in CONFIGS:
        rs = [r for r in runs if r["config"] == cfg]
        f = [r["test"]["macro_f1_mean"] for r in rs]
        summary[cfg] = {"test_macro_f1_mean": round(statistics.mean(f), 4), "test_macro_f1_std": round(statistics.stdev(f), 4),
                        "majority_macro_f1": rs[0]["baseline_test"]["majority_macro_f1"],
                        "r2_vs_zero_h5_mean": round(statistics.mean(r["test"]["r2_vs_zero_h5"] for r in rs), 4),
                        "params": rs[0]["params"], "all_losses_fell": all(r["loss_fell"] for r in rs)}
    out = {"ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "data": "SYNTHETIC (planted signal); says nothing about SET50",
           "lookback": LOOKBACK, "horizons": HORIZONS, "configs": {k: v for k, v in CONFIGS.items()}, "epochs": EPOCHS,
           "torch": torch.__version__, "summary": summary, "runs": runs}
    path = ROOT / "data" / "derived" / "proof_of_life_multiscale.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
