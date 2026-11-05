"""Checks for v2.09 (loss.class_weights: weighted trend cross-entropy on the B2 label; v2.08 otherwise).

  python tests/test_class_weights.py            # ~8 minutes on a CPU (three --smoke runs on the real dataset)
  python tests/test_class_weights.py --no-data  # synthetic checks only

1. Config: v2_09.json differs from v2_08.json only in name, description, decided, loss.class_weights(_rule) and
   training.max_epochs (30 -> 40, Siwat 2026-11-05 15:04); the
   checker refuses a wrong count, non-positive weights and an unknown rule; earlier configs are unaffected.
2. Loss: with weights, the trend loss equals sum(w_y * CE) / sum(w_y) (torch's weighted mean); without weights it is
   exactly the plain cross-entropy of every earlier run; the weights move to the model's device with the loss.
3. train_one on synthetic 2-class data: runs with the weights, and training differs from the unweighted run.
4. Real data: the run stops when the weights differ from 1 / (2 x training share); with the configured weights a
   --smoke run finishes and records them; compare_b2.py --rule b2_bars_not_below_base runs against a v2.08 --smoke run
   on the same rows, with the tuned-cut-off XGBoost beside.
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


def test_config():
    import run_config as RC
    from test_b2_label import flat
    a = json.loads((ROOT / "configs/v2_08.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / "configs/v2_09.json").read_text(encoding="utf-8"))
    fa, fb = flat(a), flat(b)
    changed = sorted(k for k in set(fa) | set(fb) if not k.startswith("_notes") and fa.get(k) != fb.get(k))
    check(changed == ["decided", "description", "loss.class_weights", "loss.class_weights_rule", "name", "training.max_epochs"]
          and fb["loss.class_weights"] == [1.1504, 0.8844] and fb["training.max_epochs"] == 40, f"v2_09 = v2_08 except {changed}")
    RC.load(str(ROOT / "configs/v2_09.json"))
    bad = []
    for upd in ({"class_weights": [1.0]}, {"class_weights": [1.0, 0.0]}, {"class_weights": [1.0, True]},
                {"class_weights_rule": "balanced"}):
        c = copy.deepcopy(b); c["loss"].update(upd)
        bad.append(bool([p for p in RC.problems(c) if "class_weights" in p]))
    old = [not RC.problems(json.loads((ROOT / f"configs/{n}.json").read_text(encoding="utf-8"))) for n in ("v2_04", "v2_07", "v2_08")]
    check(all(bad) and all(old), "checker refuses a wrong count, zero / bool weights and an unknown rule; earlier configs pass")


def test_loss():
    from proposal_model import make_loss
    cfg = json.loads((ROOT / "configs/v2_09.json").read_text(encoding="utf-8"))
    r = torch.Generator().manual_seed(0)
    logits = torch.randn(64, 1, 2, generator=r); y = torch.randint(0, 2, (64, 1), generator=r)
    out = {"trend_logits": logits, "returns": None}
    lw, _ = make_loss(cfg["loss"])(out, y, None)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), y.reshape(-1), reduction="none")
    w = torch.tensor(cfg["loss"]["class_weights"])[y.reshape(-1)]
    check(abs(lw.item() - float((w * ce).sum() / w.sum())) < 1e-6, f"weighted loss = sum(w_y CE) / sum(w_y) ({lw.item():.6f})")
    plain = {k: v for k, v in cfg["loss"].items() if not k.startswith("class_weights")}
    lp, _ = make_loss(plain)(out, y, None)
    check(abs(lp.item() - ce.mean().item()) < 1e-7 and abs(lp.item() - lw.item()) > 1e-4,
          "without weights: exactly the plain cross-entropy (every earlier run); with weights: different")
    lf = make_loss(cfg["loss"]).to(torch.float64)
    check(lf.ce.weight is not None and lf.ce.weight.dtype == torch.float64 and "ce.weight" in dict(lf.named_buffers()),
          "the weights are a buffer of the loss (they move with .to(); not trained)")


def test_train_one():
    import train_v2xx as TV
    from test_b2_label import tiny_cfg
    from test_v2xx_training import synthetic_D
    D = synthetic_D()
    D = {k: dict(v, Y=(v["Y"][:, :1] > 0).astype(np.int64), R=v["R"][:, :1].copy()) for k, v in D.items()}
    base = tiny_cfg(True); base["training"].update(max_epochs=3, patience=50)
    wcfg = copy.deepcopy(base); wcfg["loss"]["class_weights"] = [2.0, 0.5]
    _, m0, s0, _ = TV.train_one(base, 7, D, 18, 10, 5, 3)
    _, m1, s1, _ = TV.train_one(wcfg, 7, D, 18, 10, 5, 3)
    l0 = [h["train_loss"] for h in m0["history"]]; l1 = [h["train_loss"] for h in m1["history"]]
    d0 = [h["es_call_shares"][0][0] for h in m0["history"]]; d1 = [h["es_call_shares"][0][0] for h in m1["history"]]
    check(l0 != l1 and not all(torch.equal(s0[0][k], s1[0][k]) for k in s0[0]),
          f"weighted training differs from plain (down-call share on the hold-out: plain {d0}, weighted {d1})")


def test_real_data():
    cfg = json.loads((ROOT / "configs/v2_09.json").read_text(encoding="utf-8"))
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

        wrong = copy.deepcopy(cfg); wrong["loss"]["class_weights"] = [1.1505, 0.8843]
        r, _ = run(wrong, "wrong")
        check(r.returncode != 0 and "differ from 1 / (K x training share) = [1.1504, 0.8844]" in (r.stdout + r.stderr),
              "the run stops when the weights differ from 1 / (2 x training share) (1.1505 / 0.8843 refused)")
        r, out = run(cfg, "v2_09")
        ok = r.returncode == 0 and out.exists()
        check(ok, "end to end: train_v2xx.py --smoke with class weights finishes" + ("" if ok else "\n" + r.stdout[-1500:] + r.stderr[-1500:]))
        if not ok:
            return
        res = json.loads(out.read_text(encoding="utf-8"))
        check(res["loss_class_weights"] == [1.1504, 0.8844] and "class weights [1.1504, 0.8844]" in r.stdout,
              "run JSON and printout record the class weights and the training shares")
        r, base = run(json.loads((ROOT / "configs/v2_08.json").read_text(encoding="utf-8")), "v2_08")
        check(r.returncode == 0 and base.exists(), "v2.08 --smoke run for the comparison")
        cmp_out = tmp / "cmp.json"
        r = subprocess.run([sys.executable, str(ROOT / "scripts/compare_b2.py"), "--run", str(out), "--base-run", str(base),
                            "--rule", "b2_bars_not_below_base", "--out", str(cmp_out), "--smoke"],
                           capture_output=True, text=True, cwd=ROOT)
        ok = r.returncode == 0 and cmp_out.exists()
        check(ok, "compare_b2.py --rule b2_bars_not_below_base --smoke finishes" + ("" if ok else "\n" + r.stdout[-1500:] + r.stderr[-1500:]))
        if ok:
            c = json.loads(cmp_out.read_text(encoding="utf-8"))
            check(set(c["adopt"]) == {"adopted", "a_above_xgboost", "b_above_random", "c_not_below_base"}
                  and "v2.09 - v2.08" in c["differences"] and "xgboost_b2_tuned_cutoff" in c["macro_f1"]
                  and len(c["xgboost"]["tuned_cutoffs_on_inner_holdout"]) == 2 and c["rule"].startswith("(v2.09"),
                  "comparison: v2.09 - v2.08, tuned-cut-off XGBoost, three adopt parts, rule text")
            print("\n".join("    " + l for l in r.stdout.splitlines()[-18:]))


if __name__ == "__main__":
    test_config()
    test_loss()
    test_train_one()
    if "--no-data" not in sys.argv:
        test_real_data()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)
