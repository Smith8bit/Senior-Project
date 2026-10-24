"""Checks for the v2.02 training options (src/train_utils.py, src/stopping.py Selector, scripts/train_v2xx.train_one).

  python tests/test_v2xx_training.py        # needs torch; synthetic data only, about a minute on a CPU

1. Selection: the online Selector keeps exactly the epoch the offline rule picks, for both rules, on 2,000 random
   score sequences (with ties); f1_up_to_ce_min never keeps an epoch after the lowest-CE epoch.
2. EMA: after a step the averaged weights equal decay * old + (1 - decay) * new; buffers are not averaged.
3. BatchNorm recompute: running means equal the mean of the layer's inputs over the pass; a second pass gives the
   same numbers; momentum is restored; a model without BatchNorm is left alone.
4. Cosine schedule: starts at lr, reaches lr_min at the last step of the last epoch; constant gives no scheduler.
5. Diagnostics: cross-entropy of a uniform guess is ln 3; call shares sum to 1; agreement is None at the first epoch.
6. train_one on synthetic data with v2.02 settings: runs all epochs (no early stopping), the kept epoch equals the
   offline rule on its own logged history, the returned weights are the EMA weights of that epoch, and the
   best-so-far file holds the same epoch. With v2.01 settings it still keeps the best-mean-F1 epoch.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import stopping as STOP  # noqa: E402
import train_utils as TU  # noqa: E402

FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def test_selector():
    rng = np.random.default_rng(0)
    bad = {r: 0 for r in STOP.SELECTIONS}
    after_ce = 0
    for _ in range(2000):
        n = int(rng.integers(1, 31))
        f1 = np.round(rng.uniform(0.3, 0.4, n), 3).tolist()         # rounding makes ties common
        ce = np.round(rng.uniform(1.05, 1.10, n), 3).tolist()
        for rule in STOP.SELECTIONS:
            s = STOP.Selector(rule)
            for ep in range(1, n + 1):
                s.update(ep, f1[ep - 1], ce[ep - 1])
            if s.kept_epoch != STOP.select_offline(rule, f1, ce):
                bad[rule] += 1
            if rule == "f1_up_to_ce_min" and s.kept_epoch > s.ce_min_epoch:
                after_ce += 1
    for rule in STOP.SELECTIONS:
        check(bad[rule] == 0, f"selection {rule}: online kept epoch = offline rule on 2,000 random sequences ({bad[rule]} mismatches)")
    check(after_ce == 0, "f1_up_to_ce_min never keeps an epoch after the lowest-CE epoch")
    s = STOP.Selector("f1_up_to_ce_min")
    seq = [(0.33, 1.09), (0.36, 1.08), (0.35, 1.07), (0.38, 1.075), (0.37, 1.06)]
    kept = []
    for ep, (a, b) in enumerate(seq, 1):
        s.update(ep, a, b); kept.append(s.kept_epoch)
    check(kept == [1, 2, 2, 2, 4], f"worked example: kept epoch after each epoch {kept} (expected [1, 2, 2, 2, 4])")


def test_ema():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
    ema = TU.EmaWeights(m, 0.99)
    old = [p.detach().clone() for p in ema.model.parameters()]
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
        m[1].running_mean.add_(5.0)
    ema.update(m)
    ok = all(torch.allclose(pe, 0.99 * o + 0.01 * p, atol=1e-7) for pe, o, p in zip(ema.model.parameters(), old, m.parameters()))
    check(ok, "EMA update = 0.99 * old + 0.01 * new for every weight")
    check(torch.allclose(ema.model[1].running_mean, torch.zeros(3)), "EMA does not average BatchNorm buffers (they are recomputed instead)")
    check(all(not p.requires_grad for p in ema.model.parameters()), "EMA copy takes no gradients")


def test_recompute_bn():
    torch.manual_seed(1)
    m = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
    X = torch.randn(400, 4) * 3 + 2
    batches = [(X[i:i + 100],) for i in range(0, 400, 100)]
    n = TU.recompute_bn(m, batches)
    with torch.no_grad():
        h = m[0](X)
    check(n == 4 and torch.allclose(m[1].running_mean, h.mean(0), atol=1e-5),
          "running mean = mean of the BatchNorm inputs over the whole pass (equal batches)")
    first = m[1].running_mean.clone(), m[1].running_var.clone()
    TU.recompute_bn(m, batches)
    check(torch.allclose(first[0], m[1].running_mean) and torch.allclose(first[1], m[1].running_var), "a second pass gives the same statistics")
    check(m[1].momentum == 0.1, "BatchNorm momentum restored to its setting")
    plain = nn.Sequential(nn.Linear(4, 3))
    check(TU.recompute_bn(plain, batches) == 0, "a model without BatchNorm is left alone")


def test_scheduler():
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-4)
    tr = {"lr_schedule": "cosine", "lr_min": 0.0}
    spe, epochs = 7, 3
    sch = TU.make_scheduler(opt, tr, spe, epochs)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(spe * epochs):
        opt.step(); sch.step(); lrs.append(opt.param_groups[0]["lr"])
    check(abs(lrs[0] - 1e-4) < 1e-15 and abs(lrs[-1]) < 1e-15 and all(a >= b - 1e-18 for a, b in zip(lrs, lrs[1:])),
          f"cosine: starts at 1e-4, falls every step, ends at lr_min 0 after {spe * epochs} steps")
    mid = lrs[(spe * epochs) // 2]
    check(abs(mid - 0.5e-4 * (1 + math.cos(math.pi * ((spe * epochs) // 2) / (spe * epochs)))) < 1e-12, "cosine follows lr_min + (lr - lr_min)(1 + cos(pi t / T)) / 2")
    check(TU.make_scheduler(opt, {"lr_schedule": "constant"}, spe, epochs) is None, "constant: no scheduler")
    check(TU.steps_per_epoch(44060, 128) == 345, "44,060 fit rows at batch 128 = 345 steps per epoch")


def test_diagnostics():
    N = 300
    P = np.full((N, 3, 3), 1 / 3)
    Y = np.random.default_rng(2).integers(0, 3, (N, 3))
    d = TU.holdout_diagnostics(P, Y, None)
    check(all(abs(c - math.log(3)) < 1e-9 for c in d["ce"]) and d["agreement"] is None, "uniform guess: CE = ln 3 at every horizon; no agreement at epoch 1")
    P2 = np.random.default_rng(3).dirichlet([1, 1, 1], (N, 3))
    d2 = TU.holdout_diagnostics(P2, Y, d["calls"])
    check(all(abs(sum(s) - 1) < 1e-9 for s in d2["call_shares"]) and all(0 <= a <= 1 for a in d2["agreement"]),
          "call shares sum to 1; agreement between 0 and 1")


def synthetic_D(L=32, C=18, n_fit=640, n_es=320, n_stocks=10, n_sectors=5, seed=0):
    r = np.random.default_rng(seed)

    def block(n):
        X = r.normal(size=(n, L, C)).astype(np.float32)
        Y = r.integers(0, 3, (n, 3)).astype(np.int64)
        X[:, -1, 0] += (Y[:, 0] - 1) * 0.8                          # a little signal so selection has something to do
        return {"X": X, "Y": Y, "R": r.normal(size=(n, 3)).astype(np.float32),
                "S": r.integers(0, n_stocks + 1, n).astype(np.int64), "G": r.integers(0, n_sectors + 1, n).astype(np.int64)}
    return {"fit": block(n_fit), "es": block(n_es)}


def small_cfg(v202: bool):
    cfg = json.loads((ROOT / ("configs/v2_02.json" if v202 else "configs/v2_01.json")).read_text(encoding="utf-8"))
    cfg["model"].update(lookback=32, patch_lengths=[2, 5, 21], patch_strides=[2, 5, 21], d_model=16, n_heads=2, n_layers=1,
                        d_ff=32, fusion_hidden=8, d_stock=4, d_sector=4)
    cfg["training"].update(batch_size=64)
    return cfg


def test_train_one():
    import train_v2xx as TV
    D = synthetic_D()
    with tempfile.TemporaryDirectory() as tmp:
        save = Path(tmp) / "best.pt"
        cfg = small_cfg(True)
        model, meta, states, used = TV.train_one(cfg, 7, D, 18, 10, 5, 6, save_path=save)
        h = meta["history"]
        check(meta["epochs_run"] == 6, "v2.02: no early stopping, all 6 epochs run")
        f1 = [r["es_macro_f1"] for r in h]; ce = [r["es_ce_mean"] for r in h]
        off = STOP.select_offline("f1_up_to_ce_min", f1, ce)
        check(meta["best_epoch"] == off and used == [off] * 3, f"v2.02: kept epoch {meta['best_epoch']} = offline rule on the logged history ({off})")
        check(all({"es_ce_per_horizon", "es_call_shares", "es_call_agreement", "lr_end", "kept_epoch"} <= set(r) for r in h)
              and h[0]["es_call_agreement"] is None and h[1]["es_call_agreement"] is not None, "diagnostics logged every epoch")
        check(abs(h[-1]["lr_end"]) < 1e-12 and h[0]["lr_end"] < 1e-4, "cosine reaches 0 at the end of the last epoch")
        ck = torch.load(save, weights_only=False)
        check(ck["best_epoch"] == off and ck["weights"] == "ema", "best-so-far file holds the kept epoch, EMA weights")
        same = all(torch.equal(ck["states"][0][k], states[0][k].cpu()) for k in ck["states"][0])
        check(same, "saved weights = returned kept weights")
        bn_keys = [k for k in states[0] if k.endswith("running_mean")]
        check(len(bn_keys) > 0, f"kept state includes BatchNorm statistics ({len(bn_keys)} layers)")
        cfg1 = small_cfg(False)
        cfg1["training"].update(max_epochs=6, patience=50)
        model, meta1, _, _ = TV.train_one(cfg1, 7, D, 18, 10, 5, 6)
        f1b = [r["es_macro_f1"] for r in meta1["history"]]
        check(meta1["best_epoch"] == STOP.select_offline("f1_mean", f1b) and meta1["weights_scored"] == "raw",
              "v2.01 settings: kept epoch = best mean macro-F1, raw weights (unchanged behaviour)")


if __name__ == "__main__":
    test_selector()
    test_ema()
    test_recompute_bn()
    test_scheduler()
    test_diagnostics()
    test_train_one()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)
