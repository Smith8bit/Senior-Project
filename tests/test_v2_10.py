"""Checks for v2.10 (v2.06 + per-horizon class-weighted cross-entropy on the 3-class k x sigma label).

  python tests/test_v2_10.py            # ~5 minutes on a CPU (two --smoke runs on the real dataset)
  python tests/test_v2_10.py --no-data  # synthetic checks only

1. Config: v2_10.json differs from v2_06.json only in name, description, decided and loss.class_weights(_rule); the
   checker takes one weight list per horizon and refuses a wrong number of horizons or classes.
2. Loss: with per-horizon weights the trend loss is the mean over horizons of each horizon's weighted mean
   (sum w_hy * CE / sum w_hy); with all weights 1 it equals v2.06's plain cross-entropy; the one-list (v2.09) path is
   unchanged; the weights are a buffer of the loss.
3. train_one on synthetic 3-horizon data: runs with the weights, and training differs from the unweighted run.
4. Real data: the run stops when the weights are not the squared inverse shares (the plain inverse shares are refused);
   with the configured weights a --smoke run finishes and records them per horizon.
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
W10 = [[0.9299, 0.996, 1.0829], [0.9372, 0.9992, 1.0702], [0.9048, 1.0019, 1.1089]]


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def test_config():
    import run_config as RC
    from test_b2_label import flat
    a = json.loads((ROOT / "configs/v2_06.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / "configs/v2_10.json").read_text(encoding="utf-8"))
    fa, fb = flat(a), flat(b)
    changed = sorted(k for k in set(fa) | set(fb) if not k.startswith("_notes") and fa.get(k) != fb.get(k))
    check(changed == ["decided", "description", "loss.class_weights", "loss.class_weights_rule", "name"]
          and fb["loss.class_weights"] == W10 and fb["training.max_epochs"] == 30, f"v2_10 = v2_06 except {changed}")
    RC.load(str(ROOT / "configs/v2_10.json"))
    bad = []
    for cw in ([[1, 1, 1], [1, 1, 1]], [[1, 1], [1, 1], [1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, -1]]):
        c = copy.deepcopy(b); c["loss"]["class_weights"] = cw
        bad.append(bool([p for p in RC.problems(c) if "class_weights" in p]))
    c = copy.deepcopy(b); c["loss"]["class_weights"] = [1.0, 1.0, 1.0]
    check(all(bad) and not RC.problems(c), "checker: one list per horizon or one list for all; wrong shapes refused")


def test_loss():
    from proposal_model import make_loss
    cfg = json.loads((ROOT / "configs/v2_10.json").read_text(encoding="utf-8"))
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(256, 3, 3, generator=g); y = torch.randint(0, 3, (256, 3), generator=g)
    out = {"trend_logits": logits, "returns": None}
    lw, _ = make_loss(cfg["loss"])(out, y, None)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), y.reshape(-1), reduction="none").view(256, 3)
    W = torch.tensor(W10)
    want = np.mean([float((W[j][y[:, j]] * ce[:, j]).sum() / W[j][y[:, j]].sum()) for j in range(3)])
    check(abs(lw.item() - want) < 1e-6, f"per-horizon weighted loss = mean over horizons of sum(w CE) / sum(w) ({lw.item():.6f})")
    ones = dict(cfg["loss"], class_weights=[[1.0] * 3] * 3)
    plain = {k: v for k, v in cfg["loss"].items() if not k.startswith("class_weights")}
    l1, _ = make_loss(ones)(out, y, None); lp, _ = make_loss(plain)(out, y, None)
    check(abs(l1.item() - lp.item()) < 1e-6 and abs(lp.item() - ce.mean().item()) < 1e-7 and abs(lw.item() - lp.item()) > 1e-4,
          "all weights 1 = v2.06's plain cross-entropy; the v2.10 weights change the loss")
    one = dict(plain, class_weights=[0.9, 1.0, 1.1])
    l9, _ = make_loss(one)(out, y, None)
    w9 = torch.tensor([0.9, 1.0, 1.1])[y.reshape(-1)]
    check(abs(l9.item() - float((w9 * ce.reshape(-1)).sum() / w9.sum())) < 1e-6 and not make_loss(one).per_horizon,
          "one list for all horizons: torch's weighted mean as in v2.09 (unchanged path)")
    lf = make_loss(cfg["loss"]).to(torch.float64)
    check("hw" in dict(lf.named_buffers()) and lf.hw.dtype == torch.float64 and lf.hw.shape == (3, 3),
          "per-horizon weights are a [3, 3] buffer of the loss (moved with .to(); not trained)")


def test_train_one():
    import train_v2xx as TV
    from test_no_price_head import tiny_cfg
    from test_v2xx_training import synthetic_D
    D = synthetic_D()
    base = tiny_cfg(False); base["training"].update(max_epochs=3, patience=50)
    wcfg = copy.deepcopy(base); wcfg["loss"]["class_weights"] = [[2.0, 1.0, 0.5]] * 3
    _, m0, s0, _ = TV.train_one(base, 7, D, 18, 10, 5, 3)
    _, m1, s1, _ = TV.train_one(wcfg, 7, D, 18, 10, 5, 3)
    l0 = [h["train_loss"] for h in m0["history"]]; l1 = [h["train_loss"] for h in m1["history"]]
    check(l0 != l1 and not all(torch.equal(s0[0][k], s1[0][k]) for k in s0[0]) and len(m1["history"][0]["es_per_horizon"]) == 3,
          "train_one: per-horizon weights run on 3 horizons and change training")


def test_real_data():
    cfg = json.loads((ROOT / "configs/v2_10.json").read_text(encoding="utf-8"))
    tiny = dict(d_model=16, n_heads=2, n_layers=1, d_ff=32, fusion_hidden=8, d_stock=4, d_sector=4)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def run(c, name):
            c = copy.deepcopy(c); c["model"].update(tiny); c["training"]["batch_size"] = 64
            cp, out = tmp / f"{name}.cfg.json", tmp / f"{name}.json"
            cp.write_text(json.dumps(c), encoding="utf-8")
            r = subprocess.run([sys.executable, str(ROOT / "scripts/train_v2xx.py"), "--config", str(cp), "--out", str(out),
                                "--smoke", "--seeds", "1", "2"], capture_output=True, text=True, cwd=ROOT)
            return r, out

        wrong = copy.deepcopy(cfg); wrong["loss"]["class_weights"] = [[0.9643, 0.998, 1.0406], [0.9681, 0.9996, 1.0345], [0.9512, 1.0009, 1.053]]
        r, _ = run(wrong, "wrong")
        check(r.returncode != 0 and "differ from (1 / (K x training share))^2 = [0.9299, 0.996, 1.0829]" in (r.stdout + r.stderr),
              "the run stops when the weights are not the squared inverse shares (plain inverse shares refused)")
        r, out = run(cfg, "v2_10")
        ok = r.returncode == 0 and out.exists()
        check(ok, "end to end: train_v2xx.py --smoke with per-horizon class weights finishes" + ("" if ok else "\n" + r.stdout[-1500:] + r.stderr[-1500:]))
        if ok:
            res = json.loads(out.read_text(encoding="utf-8"))
            lines = [l for l in r.stdout.splitlines() if l.startswith("class weights")]
            check(res["loss_class_weights"] == W10 and len(lines) == 3 and set(res["runs"][0]["val"]) >= {"h1", "h5", "h10"},
                  "run JSON and printout record the weights for each of the 3 horizons")
            print("\n".join("    " + l for l in lines))


if __name__ == "__main__":
    test_config()
    test_loss()
    test_train_one()
    if "--no-data" not in sys.argv:
        test_real_data()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)
