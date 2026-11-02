"""Checks for the v2.05 gradient diagnostic (src/train_utils.shared_grad_diagnostics, training.grad_diagnostics).

  python tests/test_grad_diag.py        # synthetic data only, about a minute on a CPU

1. The diagnostic leaves no trace: no .grad written, BatchNorm statistics and weights unchanged, the model's train/eval
   mode restored, and torch's random state untouched.
2. It measures the right thing: the shared parameters are every parameter except the two heads; the norms equal a
   direct autograd computation; the weighted share follows w_p|g_p| / (w_t|g_t| + w_p|g_p|).
3. train_one with grad_diagnostics on gives exactly the same history and weights as with it off, and logs the
   diagnostic before training and after every epoch.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import train_utils as TU  # noqa: E402

FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def small_model_and_batches():
    import train_v2xx as TV
    from test_v2xx_training import small_cfg, synthetic_D
    cfg = small_cfg(False)
    torch.manual_seed(0)
    model = TV.build_model(cfg, 18, 10, 5)
    D = synthetic_D()
    f = D["fit"]
    batches = [tuple(torch.from_numpy(np.ascontiguousarray(f[k][i:i + 64])) for k in ("X", "S", "G", "Y", "R")) for i in (0, 64)]
    return cfg, model, batches, D


def test_no_trace():
    cfg, model, batches, _ = small_model_and_batches()
    model.train()
    before = copy.deepcopy(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    d = TU.shared_grad_diagnostics(model, batches, 1.0, 0.25, nn.MSELoss())
    check(all(p.grad is None for p in model.parameters()), "no .grad written")
    after = model.state_dict()
    check(all(torch.equal(before[k], after[k]) for k in before), "weights and BatchNorm statistics unchanged")
    check(model.training, "train mode restored")
    check(torch.equal(rng_before, torch.get_rng_state()), "torch random state untouched")
    check(all(np.isfinite(v) for v in d.values()) and 0 <= d["price_share_weighted"] <= 1 and -1 <= d["cosine"] <= 1,
          f"finite values in range ({ {k: round(v, 4) for k, v in d.items()} })")


def test_measures():
    cfg, model, batches, _ = small_model_and_batches()
    shared = TU.shared_parameters(model)
    heads = [p for n, p in model.named_parameters() if n.startswith(("trend_head.", "price_head."))]
    check(len(shared) + len(heads) == len(list(model.parameters())) and len(heads) > 0,
          f"shared = all parameters except the two heads ({len(shared)} shared tensors, {len(heads)} head tensors)")
    d1 = TU.shared_grad_diagnostics(model, batches, 1.0, 1.0, nn.MSELoss())
    model.eval()                                                     # direct computation for comparison
    gt = [torch.zeros_like(p) for p in shared]; gp = [torch.zeros_like(p) for p in shared]
    for X, S, G, Y, R in batches:
        out = model(X, S, G)
        lt = nn.functional.cross_entropy(out["trend_logits"].reshape(-1, 3), Y.reshape(-1))
        lp = nn.functional.mse_loss(out["returns"], R)
        for a, g in zip(gt, torch.autograd.grad(lt, shared, retain_graph=True, allow_unused=True)):
            a += 0 if g is None else g
        for a, g in zip(gp, torch.autograd.grad(lp, shared, allow_unused=True)):
            a += 0 if g is None else g
    nt = float(torch.cat([g.flatten() for g in gt]).norm() / 2); npr = float(torch.cat([g.flatten() for g in gp]).norm() / 2)
    check(abs(d1["trend_norm"] - nt) < 1e-6 * max(1, nt) and abs(d1["price_norm"] - npr) < 1e-6 * max(1, npr),
          "norms equal a direct autograd computation")
    d2 = TU.shared_grad_diagnostics(model, batches, 1.0, 0.25, nn.MSELoss())
    want = 0.25 * npr / (nt + 0.25 * npr)
    check(abs(d2["price_share_weighted"] - want) < 1e-9 and abs(d2["cosine"] - d1["cosine"]) < 1e-9,
          f"weighted share follows the weights (w_price 1 -> {d1['price_share_weighted']:.3f}, 0.25 -> {d2['price_share_weighted']:.3f}); cosine unaffected")


def test_train_one_unchanged():
    import train_v2xx as TV
    from test_v2xx_training import small_cfg, synthetic_D
    D = synthetic_D()
    base = small_cfg(False)
    base["training"].update(max_epochs=3, patience=50)
    base["loss"].update(w_price=0.25)
    on = copy.deepcopy(base); on["training"]["grad_diagnostics"] = True
    _, m0, s0, _ = TV.train_one(base, 7, D, 18, 10, 5, 3)
    _, m1, s1, _ = TV.train_one(on, 7, D, 18, 10, 5, 3)
    same_hist = [(h["train_loss"], h["es_macro_f1"]) for h in m0["history"]] == [(h["train_loss"], h["es_macro_f1"]) for h in m1["history"]]
    same_w = all(torch.equal(s0[0][k], s1[0][k]) for k in s0[0])
    check(same_hist and same_w, "grad_diagnostics on: same history and weights as off (training unchanged)")
    check("grad_shared_before_training" in m1 and all("grad_shared" in h for h in m1["history"])
          and not any("grad_shared" in h for h in m0["history"]), "diagnostic logged before training and after every epoch")
    sh = [h["grad_shared"]["price_share_weighted"] for h in m1["history"]]
    print("    price share of the weighted pull per epoch (synthetic data):", sh)


if __name__ == "__main__":
    test_no_trace()
    test_measures()
    test_train_one_unchanged()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)
