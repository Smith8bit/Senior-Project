"""Checks for v2.08 (protocol.label_rule b2_raw_close: the B2 "true data" 2-class next-day label; v2.07 otherwise).

  python tests/test_b2_label.py            # ~5 minutes on a CPU (the real-data checks build the dataset)
  python tests/test_b2_label.py --no-data  # synthetic checks only

1. Config: v2_08.json differs from v2_07.json only in name, description, decided and protocol.label_rule / labels;
   the checker refuses B2 with horizons other than [1] and unknown label rules; earlier configs are unaffected.
2. Label rule on a hand-made frame: delta < 0 -> 0 (down), delta = 0 -> 1, delta > 0 -> 1; the next close never comes
   from another segment (a row at the end of a segment is refused); "previous move" is -1 on a segment's first day.
3. Model and metrics: trend logits [B, 1, 2]; 65 fewer weights than v2.07 (one output of 64 + 1); macro-F1 over 2
   classes equals sklearn's; the 3-class default and the bootstrap's resamples are unchanged.
4. train_one on synthetic 2-class data: hold-out call shares have 2 entries, the kept epoch is the best hold-out F1.
5. Real data: B2 labels on the model rows agree row for row with an independent rebuild from the raw price file (the
   decision's own rule); the class shares; an end-to-end --smoke run (2 seeds), both plots, and compare_b2.py --smoke.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
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
    a = json.loads((ROOT / "configs/v2_07.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / "configs/v2_08.json").read_text(encoding="utf-8"))
    fa, fb = flat(a), flat(b)
    changed = sorted(k for k in set(fa) | set(fb) if not k.startswith("_notes") and fa.get(k) != fb.get(k))
    check(changed == ["decided", "description", "name", "protocol.label_rule", "protocol.labels"]
          and fb["protocol.label_rule"] == "b2_raw_close", f"v2_08 = v2_07 except {changed}")
    RC.load(str(ROOT / "configs/v2_08.json"))
    bad = []
    for upd in ({"horizons": [1, 5, 10]}, {"horizons": [5]}, {"label_rule": "binary"}):
        c = copy.deepcopy(b); c["protocol"].update(upd)
        bad.append(bool([p for p in RC.problems(c) if "label_rule" in p]))
    old = [not RC.problems(json.loads((ROOT / f"configs/{n}.json").read_text(encoding="utf-8")))
           for n in ("v2_01", "v2_03", "v2_04", "v2_06", "v2_07")]
    check(all(bad) and all(old), "checker refuses B2 with horizons other than [1] and unknown rules; earlier configs still pass")


def test_rule():
    import labels as L
    #            seg A: 10 -> 9 (down) -> 9 (zero) -> 9.5 (up) -> end | seg B: 20 -> 21 -> 20.99 -> end
    full = pd.DataFrame({"seg_key": ["A"] * 4 + ["B"] * 3, "close": [10.0, 9.0, 9.0, 9.5, 20.0, 21.0, 20.99]})
    y, prev = L.b2_next_day(full, [0, 1, 2, 4, 5])
    check(y.tolist() == [0, 1, 1, 1, 0] and prev.tolist() == [-1, 0, 1, -1, 1],
          f"down / zero / up -> 0 / 1 / 1; previous move -1 on a segment's first day (y {y.tolist()}, prev {prev.tolist()})")
    try:
        L.b2_next_day(full, [3])
        check(False, "a row at the end of its segment is refused (its next close would be another segment's)")
    except ValueError:
        check(True, "a row at the end of its segment is refused (its next close would be another segment's)")
    f2 = pd.DataFrame({"seg_key": ["A", "A"], "close": [0.1 + 0.2, 0.3]})          # float noise -> still a zero move
    check(L.b2_next_day(f2, [0])[0].tolist() == [1], "a zero move with float noise (0.1 + 0.2 vs 0.3) counts as not down")


def tiny_cfg(b2: bool):
    from test_one_horizon import tiny_cfg as base
    cfg = base((1,))
    if b2:
        cfg["protocol"]["label_rule"] = "b2_raw_close"
    return cfg


def test_model_metrics():
    import train_v2xx as TV
    import metrics as M
    import posthoc as PH
    from sklearn.metrics import f1_score
    torch.manual_seed(0); m3 = TV.build_model(tiny_cfg(False), 18, 10, 5)
    torch.manual_seed(0); m2 = TV.build_model(tiny_cfg(True), 18, 10, 5)
    n3, n2 = sum(p.numel() for p in m3.parameters()), sum(p.numel() for p in m2.parameters())
    o = m2(torch.randn(4, 32, 18), torch.ones(4, dtype=torch.long), torch.ones(4, dtype=torch.long))
    check(n3 - n2 == 65 and tuple(o["trend_logits"].shape) == (4, 1, 2), f"weights {n3:,} -> {n2:,}; trend logits [4, 1, 2]")
    r = np.random.default_rng(1)
    y, yh = r.integers(0, 2, 500), r.integers(0, 2, 500)
    check(abs(M.trend_metrics(y, yh, 2)["macro_f1"] - round(f1_score(y, yh, average="macro"), 4)) < 1e-9
          and M.trend_metrics(y, yh, 3)["macro_f1"] < M.trend_metrics(y, yh, 2)["macro_f1"],
          "2-class macro-F1 = sklearn's; averaging over a 3rd empty class would lower it (why n_classes is passed)")
    dates = np.repeat(np.arange(60), 10)
    y3, c3 = r.integers(0, 3, 600), r.integers(0, 3, (3, 600))
    b_def, b_3 = PH.block_bootstrap(dates, y3, {"m": c3}), PH.block_bootstrap(dates, y3, {"m": c3}, n_classes=3)
    y2, c2 = r.integers(0, 2, 600), r.integers(0, 2, (3, 600))
    b2 = PH.block_bootstrap(dates, y2, {"m": c2}, n_classes=2)
    want = np.mean([f1_score(y2, c, average="macro") for c in c2])
    check(b_def == b_3 and abs(b2["values"]["m"]["point"] - round(want, 4)) < 1e-9,
          f"bootstrap: 3-class default unchanged; 2-class point = sklearn ({b2['values']['m']['point']} vs {want:.4f})")


def test_train_one():
    import train_v2xx as TV
    from test_v2xx_training import synthetic_D
    cfg = tiny_cfg(True); cfg["training"].update(max_epochs=4, patience=50)
    D = synthetic_D()
    D = {k: dict(v, Y=(v["Y"][:, :1] > 0).astype(np.int64), R=v["R"][:, :1].copy()) for k, v in D.items()}
    model, meta, states, used = TV.train_one(cfg, 7, D, 18, 10, 5, 4)
    h = meta["history"]
    f1 = [e["es_per_horizon"][0] for e in h]
    check(all(len(e["es_call_shares"][0]) == 2 for e in h) and len(meta["es_actual_shares"][0]) == 2,
          "train_one: hold-out call shares and actual shares over 2 classes")
    check(meta["best_epoch"] == h[int(np.argmax(f1))]["epoch"], f"kept epoch {meta['best_epoch']} = best hold-out F1 {[round(v, 3) for v in f1]}")
    P, R, _ = TV.predict(model, D["es"]["X"], D["es"]["S"], D["es"]["G"], 64)
    check(P.shape == (len(D["es"]["X"]), 1, 2) and np.allclose(P.sum(-1), 1, atol=1e-5), "predict: [rows, 1, 2] probabilities")


def test_real_data():
    import baselines_tabular as BT
    import labels as L
    data, _, full = BT.build_dataset(return_full=True)
    y, prev = L.b2_next_day(full, data.index)
    p = pd.read_csv(ROOT / "data/raw/prices/siamchart/set50_union_2019_2025.csv")
    p["date"] = pd.to_datetime(p.date.astype(str))
    breaks = {"TRUE": ["2023-03-03"], "SCB": ["2022-04-27"], "GULF": ["2020-04-16"], "BANPU": ["2021-08-16"], "TOP": ["2024-12-20"]}
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True); p["seg"] = 0
    for s, ds in breaks.items():                                      # the decision session's own rebuild
        for d in ds:
            p.loc[(p.symbol == s) & (p.date >= d), "seg"] += 1
    p["nc"] = p.groupby(["symbol", "seg"]).close.shift(-1)
    p = p[p.nc.notna()]
    p["yb"] = ((p.nc - p.close).round(6) >= 0).astype(int)
    m = data[["symbol", "date", "split"]].assign(y=y).merge(p[["symbol", "date", "yb"]], on=["symbol", "date"], how="left")
    check(m.yb.notna().all() and (m.y == m.yb).all(), f"B2 labels on all {len(m):,} model rows = the decision's independent rebuild")
    sh = m.groupby("split").y.mean().round(4).to_dict()
    check(abs(sh["train"] - 0.565) < 0.002 and abs(sh["val"] - 0.566) < 0.002 and (prev >= 0).all(),
          f"not-down share train {sh['train']} / val {sh['val']} (decision file: 0.565 / 0.566); every row has a previous move")

    cfg = json.loads((ROOT / "configs/v2_08.json").read_text(encoding="utf-8"))
    cfg["model"].update(d_model=16, n_heads=2, n_layers=1, d_ff=32, fusion_hidden=8, d_stock=4, d_sector=4)
    cfg["training"]["batch_size"] = 64
    with tempfile.TemporaryDirectory() as tmp:
        cp, out = Path(tmp) / "cfg.json", Path(tmp) / "run.json"
        cp.write_text(json.dumps(cfg), encoding="utf-8")
        r = subprocess.run([sys.executable, str(ROOT / "scripts/train_v2xx.py"), "--config", str(cp), "--out", str(out),
                            "--smoke", "--seeds", "1", "2"], capture_output=True, text=True, cwd=ROOT)
        ok = r.returncode == 0 and out.exists()
        check(ok, "end to end: train_v2xx.py --smoke with the B2 label finishes" + ("" if ok else "\n" + r.stdout[-1500:] + r.stderr[-1500:]))
        if not ok:
            return
        res = json.loads(out.read_text(encoding="utf-8"))
        fl = res["floors"]["h1"]
        check(res["label"]["classes"] == ["down", "not_down"] and len(res["runs"][0]["val"]["h1"]["confusion"]) == 2
              and len(fl["trend_continues"]["confusion"]) == 2 and abs(fl["stratified_random_expected"]["macro_f1"] - 0.5) < 0.02,
              f"run JSON: 2 classes, 2 x 2 confusion grids, floors on B2 (random expected {fl['stratified_random_expected']['macro_f1']})")
        pr = pd.read_csv(out.with_name("run_preds.csv.gz"))
        pc = sorted(c for c in pr.columns if c.startswith("v2xx_s1_p"))
        check(pc == ["v2xx_s1_p1_down", "v2xx_s1_p1_not_down"] and set(pr.y_1.unique()) <= {0, 1},
              f"predictions: {pc}; labels 0 / 1")
        import plot_v2xx_training as PT
        png, calls = PT.plot(out), PT.plot_calls(out)
        check(png.exists() and calls is not None and calls.exists(), "learning curves and call shares drawn for 2 classes")
        cmp_out = Path(tmp) / "cmp.json"
        r = subprocess.run([sys.executable, str(ROOT / "scripts/compare_b2.py"), "--run", str(out), "--out", str(cmp_out), "--smoke"],
                           capture_output=True, text=True, cwd=ROOT)
        ok = r.returncode == 0 and cmp_out.exists()
        check(ok, "compare_b2.py --smoke finishes" + ("" if ok else "\n" + r.stdout[-1500:] + r.stderr[-1500:]))
        if ok:
            c = json.loads(cmp_out.read_text(encoding="utf-8"))
            rv = c["macro_f1"]["random_b2_imbalance"]["point"]
            check(abs(rv - 0.5) < 0.01 and "v2.08 - xgboost_b2" in c["differences"] and set(c["adopt"]) >= {"adopted", "a_above_xgboost", "b_above_random"},
                  f"comparison: random (B2 imbalance) macro-F1 {rv} (expected 0.5); XGBoost and random differences; adopt fields")
            print("\n".join("    " + l for l in r.stdout.splitlines()[-14:]))


if __name__ == "__main__":
    test_config()
    test_rule()
    test_model_metrics()
    test_train_one()
    if "--no-data" not in sys.argv:
        test_real_data()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)
