"""Checks for v2.07 (protocol.horizons [1]: the model predicts the 1-day trend only; v2.06 otherwise).

  python tests/test_one_horizon.py            # ~3 minutes on a CPU (the last check builds the real dataset)
  python tests/test_one_horizon.py --no-data  # skip the end-to-end run

1. Config: v2_07.json differs from v2_06.json only in name, description, decided and protocol.horizons; the checker
   accepts subsets of [1, 5, 10] in increasing order and refuses anything else.
2. Model: trend logits [B, 1, 3]; the trend head's last layer has 3 outputs instead of 9, so the weight count drops by
   exactly 6 x 64 + 6 = 390; everything before the head is the same size.
3. train_one on synthetic data: one horizon logged per epoch, one kept state, and the kept epoch is the epoch with the
   best 1-day hold-out macro-F1 (the "mean" over one horizon).
4. End to end (scripts/train_v2xx.py main, --smoke, a tiny model, 2 seeds): the run JSON, floors and seed ensemble hold
   the 1-day horizon only, the predictions file has 1-day columns only, and both plots are drawn with one horizon.
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


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        out |= flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v}
    return out


def test_config():
    import run_config as RC
    a = json.loads((ROOT / "configs/v2_06.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / "configs/v2_07.json").read_text(encoding="utf-8"))
    fa, fb = flat(a), flat(b)
    changed = sorted(k for k in set(fa) | set(fb) if not k.startswith("_notes") and fa.get(k) != fb.get(k))
    check(changed == ["decided", "description", "name", "protocol.horizons"] and fb["protocol.horizons"] == [1],
          f"v2_07 = v2_06 except {changed}")
    RC.load(str(ROOT / "configs/v2_07.json"))
    ok, bad = [], []
    for hz in ([1], [5], [1, 10], [1, 5, 10], [10, 1], [2], [], [1, 1]):
        c = copy.deepcopy(b); c["protocol"]["horizons"] = hz
        (ok if not [p for p in RC.problems(c) if "horizons" in p] else bad).append(hz)
    check(ok == [[1], [5], [1, 10], [1, 5, 10]] and bad == [[10, 1], [2], [], [1, 1]],
          f"horizon check: accepted {ok}, refused {bad}")


def tiny_cfg(hz):
    from test_no_price_head import tiny_cfg as base
    cfg = base(False)
    cfg["protocol"]["horizons"] = list(hz)
    return cfg


def one_h(D):
    return {k: dict(v, Y=v["Y"][:, :1].copy(), R=v["R"][:, :1].copy()) for k, v in D.items()}


def test_model():
    import train_v2xx as TV
    torch.manual_seed(0); m3 = TV.build_model(tiny_cfg((1, 5, 10)), 18, 10, 5)
    torch.manual_seed(0); m1 = TV.build_model(tiny_cfg((1,)), 18, 10, 5)
    n3, n1 = sum(p.numel() for p in m3.parameters()), sum(p.numel() for p in m1.parameters())
    last = m1.trend_head[-1] if isinstance(m1.trend_head, torch.nn.Sequential) else None
    body3 = sum(p.numel() for n, p in m3.named_parameters() if not n.startswith("trend_head"))
    body1 = sum(p.numel() for n, p in m1.named_parameters() if not n.startswith("trend_head"))
    check(n3 - n1 == 6 * 64 + 6 and body1 == body3 and (last is None or last.out_features == 3),
          f"weights {n3:,} -> {n1:,} (trend head output 9 -> 3); shared layers unchanged ({body1:,})")
    x = torch.randn(4, 32, 18); s = torch.ones(4, dtype=torch.long); g = torch.ones(4, dtype=torch.long)
    o = m1(x, s, g)
    check(tuple(o["trend_logits"].shape) == (4, 1, 3) and o["returns"] is None, "forward: trend logits [4, 1, 3], no returns")


def test_train_one():
    import train_v2xx as TV
    from test_v2xx_training import synthetic_D
    cfg = tiny_cfg((1,)); cfg["training"].update(max_epochs=4, patience=50)
    D = one_h(synthetic_D())
    model, meta, states, used = TV.train_one(cfg, 7, D, 18, 10, 5, 4)
    h = meta["history"]
    f1 = [e["es_per_horizon"][0] for e in h]
    check(all(len(e["es_per_horizon"]) == 1 and len(e["es_call_shares"]) == 1 for e in h) and len(used) == 1,
          "train_one: one horizon logged per epoch, one kept state")
    check(meta["best_epoch"] == h[int(np.argmax(f1))]["epoch"] and all(abs(e["es_macro_f1"] - e["es_per_horizon"][0]) < 1e-12 for e in h),
          f"kept epoch {meta['best_epoch']} = best 1-day hold-out F1 {[round(v, 3) for v in f1]}")
    P, R, _ = TV.predict(model, D["es"]["X"], D["es"]["S"], D["es"]["G"], 64)
    check(P.shape == (len(D["es"]["X"]), 1, 3) and R is None, "predict: [rows, 1, 3], no returns")


def test_end_to_end():
    cfg = json.loads((ROOT / "configs/v2_07.json").read_text(encoding="utf-8"))
    cfg["model"].update(d_model=16, n_heads=2, n_layers=1, d_ff=32, fusion_hidden=8, d_stock=4, d_sector=4)
    cfg["training"]["batch_size"] = 64
    with tempfile.TemporaryDirectory() as tmp:
        cp, out = Path(tmp) / "cfg.json", Path(tmp) / "run.json"
        cp.write_text(json.dumps(cfg), encoding="utf-8")
        r = subprocess.run([sys.executable, str(ROOT / "scripts/train_v2xx.py"), "--config", str(cp), "--out", str(out),
                            "--smoke", "--seeds", "1", "2"], capture_output=True, text=True, cwd=ROOT)
        ok = r.returncode == 0 and out.exists()
        check(ok, "end to end: train_v2xx.py --smoke with horizons [1] finishes" + ("" if ok else "\n" + r.stdout[-1500:] + r.stderr[-1500:]))
        if not ok:
            return
        res = json.loads(out.read_text(encoding="utf-8"))
        keys = (set(res["runs"][0]["val"]), set(res["summary"]["model"]), set(res["floors"]), set(res["seed_ensemble"]["val"]))
        check(all({k for k in ks if k.startswith("h")} == {"h1"} for ks in keys), f"run JSON: 1-day horizon only {keys}")
        import pandas as pd
        pr = pd.read_csv(out.with_name("run_preds.csv.gz"))
        pc = [c for c in pr.columns if c.startswith("v2xx_s1_p")]
        check(pc and all(c.startswith("v2xx_s1_p1_") for c in pc), f"predictions: 1-day columns only ({pc})")
        import plot_v2xx_training as PT
        png, calls = PT.plot(out), PT.plot_calls(out)
        check(png.exists() and calls is not None and calls.exists(), "learning curves and call shares drawn with one horizon")


if __name__ == "__main__":
    test_config()
    test_model()
    test_train_one()
    if "--no-data" not in sys.argv:
        test_end_to_end()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)
