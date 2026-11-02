"""Checks for v2.06 (model.price_head false: no price head, trend only).

  python tests/test_no_price_head.py            # ~3 minutes on a CPU (the last check builds the real dataset)
  python tests/test_no_price_head.py --no-data  # skip the end-to-end run

1. Model: no price head module; "returns" is None; trend logits unchanged in shape; the weight count drops by exactly
   the price head's weights; the default (price_head true) is unchanged.
2. Loss: trend cross-entropy only (equals w_trend * CE), so the shared layers learn from the trend task alone.
3. train_one on synthetic data: runs, logs no price loss and no hold-out price error, keeps an epoch as usual.
4. End to end (scripts/train_v2xx.py main, --smoke, a tiny model, 2 seeds): the run JSON has no price metrics, the
   predictions file has no return columns, the seed ensemble is computed, and the learning-curve plot is drawn
   without the price panel.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def tiny_cfg(price_head: bool):
    from test_v2xx_training import small_cfg
    cfg = small_cfg(False)
    cfg["model"]["price_head"] = price_head
    if not price_head:
        cfg["loss"]["w_price"] = 0
    return cfg


def test_model_and_loss():
    import train_v2xx as TV
    from proposal_model import make_loss
    torch.manual_seed(0); with_h = TV.build_model(tiny_cfg(True), 18, 10, 5)
    torch.manual_seed(0); no_h = TV.build_model(tiny_cfg(False), 18, 10, 5)
    n_with, n_no = sum(p.numel() for p in with_h.parameters()), sum(p.numel() for p in no_h.parameters())
    n_head = sum(p.numel() for p in with_h.price_head.parameters())
    check(no_h.price_head is None and n_with - n_no == n_head, f"no price head; weights {n_with:,} -> {n_no:,} (price head {n_head:,})")
    x = torch.randn(4, 32, 18); s = torch.ones(4, dtype=torch.long); g = torch.ones(4, dtype=torch.long)
    o = no_h(x, s, g)
    check(o["returns"] is None and tuple(o["trend_logits"].shape) == (4, 3, 3), "forward: returns None, trend logits [4, 3, 3]")
    lf = make_loss(tiny_cfg(False)["loss"])
    y = torch.randint(0, 3, (4, 3)); r = torch.randn(4, 3)
    loss, parts = lf(o, y, r)
    ce = torch.nn.functional.cross_entropy(o["trend_logits"].reshape(-1, 3), y.reshape(-1))
    check(abs(loss.item() - ce.item()) < 1e-7 and set(parts) == {"ce"}, "loss = trend cross-entropy only")
    loss.backward()
    check(all(p.grad is not None for n, p in no_h.named_parameters() if n.startswith(("linear", "trend_head"))),
          "gradients reach the trend head and the shared Linear layer")
    o2 = with_h(x, s, g)
    check(o2["returns"] is not None and tuple(o2["returns"].shape) == (4, 3), "default model unchanged (price head present)")


def test_train_one():
    import train_v2xx as TV
    from test_v2xx_training import synthetic_D
    cfg = tiny_cfg(False); cfg["training"].update(max_epochs=3, patience=50)
    D = synthetic_D()
    model, meta, states, used = TV.train_one(cfg, 7, D, 18, 10, 5, 3)
    h = meta["history"]
    check(len(h) == 3 and all("train_price" not in e and e["es_return_rmse_std"] is None for e in h),
          "train_one: 3 epochs, no price loss and no hold-out price error logged")
    P, R, _ = TV.predict(model, D["es"]["X"], D["es"]["S"], D["es"]["G"], 64)
    check(R is None and P.shape == (len(D["es"]["X"]), 3, 3) and 1 <= meta["best_epoch"] <= 3, "predict: no returns; kept epoch as usual")


def test_end_to_end():
    cfg = json.loads((ROOT / "configs/v2_06.json").read_text(encoding="utf-8"))
    cfg["model"].update(d_model=16, n_heads=2, n_layers=1, d_ff=32, fusion_hidden=8, d_stock=4, d_sector=4)
    cfg["training"]["batch_size"] = 64
    with tempfile.TemporaryDirectory() as tmp:
        cp, out = Path(tmp) / "cfg.json", Path(tmp) / "run.json"
        cp.write_text(json.dumps(cfg), encoding="utf-8")
        r = subprocess.run([sys.executable, str(ROOT / "scripts/train_v2xx.py"), "--config", str(cp), "--out", str(out),
                            "--smoke", "--seeds", "1", "2"], capture_output=True, text=True, cwd=ROOT)
        ok = r.returncode == 0 and out.exists()
        check(ok, "end to end: train_v2xx.py --smoke with price_head false finishes" + ("" if ok else "\n" + r.stdout[-1500:] + r.stderr[-1500:]))
        if not ok:
            return
        res = json.loads(out.read_text(encoding="utf-8"))
        v = res["runs"][0]["val"]["h5"]
        check("macro_f1" in v and "r2_os_vs_zero" not in v and "rmse" not in res["summary"]["model"]["h5"],
              "run JSON: trend metrics present, no price metrics")
        check("seed_ensemble" in res and "macro_f1" in res["seed_ensemble"]["val"]["h5"], "seed ensemble computed")
        import pandas as pd
        pr = pd.read_csv(out.with_name("run_preds.csv.gz"))
        check(not any(c.startswith("v2xx_s1_r") or c.startswith("v2xx_ens_r") for c in pr.columns)
              and "v2xx_s1_p5_up" in pr.columns, "predictions: class probabilities, no return columns")
        import plot_v2xx_training as PT
        png = PT.plot(out)
        check(png.exists(), "learning-curve plot drawn without the price panel")


if __name__ == "__main__":
    test_model_and_loss()
    test_train_one()
    if "--no-data" not in sys.argv:
        test_end_to_end()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)
