"""Checks for v2.11 (v2.04 + inputs.extra_technical [plus_di, minus_di] + inputs.market set50_index_ohlcv; 18 -> 25 channels).

  python tests/test_v2_11.py            # ~4 minutes on a CPU (real dataset + one --smoke run)
  python tests/test_v2_11.py --no-data  # config, index file and forward-splice checks only

1. Config: v2_11.json differs from v2_04.json only in name, description, decided and the two new inputs keys; the
   checker refuses unknown names; channel order is price, technical, +DI / -DI, SET50 x 5, sentiment.
2. Index file: the frozen SET50 file has the expected sha256, 1,698 unique sorted days 2019-01-02..2025-12-30.
3. Market channels: equal to a hand computation from the file; causal (cutting the file after a date leaves every
   earlier value unchanged); the first day is missing (no previous close).
4. Forward splice (step0 prepare, --set50-forward-csv): rows after 2025-12-30 only; a file that matches the frozen index
   on Oct-Dec 2025 passes, with volume rescaled by the median overlap ratio; a file that does not match stops; a file
   that misses 2026 trading days stops.
5. Real data: the 18 v2.04 channels are unchanged when the new inputs are on; +DI / -DI equal src/features.py
   plus_di14 / minus_di14 on every sample row; the SET50 channels are the same for every stock on a date and never
   missing on sample rows; an end-to-end --smoke run builds 25-channel windows and finishes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

FAILS = []
INDEX_SHA = "2363de306a1e655b3ea3a2155883145c305f090303d7022e5b4cd3c202bc7139"


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def test_config():
    import run_config as RC
    import features_proposal as FP
    from test_b2_label import flat
    a = json.loads((ROOT / "configs/v2_04.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / "configs/v2_11.json").read_text(encoding="utf-8"))
    fa, fb = flat(a), flat(b)
    changed = sorted(k for k in set(fa) | set(fb) if not k.startswith("_notes") and fa.get(k) != fb.get(k))
    check(changed == ["decided", "description", "inputs.extra_technical", "inputs.market", "name"], f"v2_11 = v2_04 except {changed}")
    RC.load(str(ROOT / "configs/v2_11.json"))
    bad = []
    for upd in ({"extra_technical": ["adx"]}, {"extra_technical": ["plus_di", "plus_di"]}, {"market": "set100_index"}):
        c = copy.deepcopy(b); c["inputs"].update(upd)
        bad.append(bool(RC.problems(c)))
    ch = FP.channels(b["inputs"])
    check(all(bad) and len(ch) == 25 and ch[15:17] == ["plus_di", "minus_di"] and ch[17:22] == FP.MARKET_CH
          and ch[-3:] == FP.SENT_CH and FP.channels(a["inputs"]) == ch[:15] + ch[-3:],
          "checker refuses unknown names; 25 channels: price, technical, +DI / -DI, SET50 x 5, sentiment (v2.04's 18 in order)")


def test_index_and_market():
    import features_proposal as FP
    p = ROOT / FP.SET50_INDEX_FILE
    idx = FP.load_set50_index()
    check(hashlib.sha256(p.read_bytes()).hexdigest() == INDEX_SHA and len(idx) == 1698
          and str(idx.date.min().date()) == "2019-01-02" and str(idx.date.max().date()) == "2025-12-30",
          "frozen SET50 index file: sha256, 1,698 sorted unique days 2019-01-02..2025-12-30")
    mk = FP.market_inputs(idx)
    raw = pd.read_csv(p)
    pc = raw.close.shift(1)
    want = np.column_stack([np.log(raw.open / pc), np.log(raw.high / pc), np.log(raw.low / pc), np.log(raw.close / pc),
                            np.diff(np.log1p(raw.volume.astype(float)), prepend=np.nan)])
    check(np.allclose(mk[FP.MARKET_CH].to_numpy(), want, equal_nan=True) and mk.iloc[0][FP.MARKET_CH].isna().all(),
          "SET50 channels = ln(X_t / C_{t-1}) for O, H, L, C and the change of ln(1 + V); the first day is missing")
    cut = idx[idx.date <= "2022-06-30"]
    mk2 = FP.market_inputs(cut)
    check(np.allclose(mk2[FP.MARKET_CH].to_numpy(), mk[FP.MARKET_CH].to_numpy()[:len(cut)], equal_nan=True),
          "causal: cutting the index after 2022-06-30 leaves every earlier value unchanged")


def _fake_args(tmp: Path, csv: Path):
    return types.SimpleNamespace(set50_forward_csv=str(csv), redo=False, run_json=str(tmp / "none.json"))


def test_forward_splice():
    import features_proposal as FP
    import step0_v2_03beta as S0
    frozen = pd.read_csv(ROOT / FP.SET50_INDEX_FILE)
    ov = frozen[frozen.date >= 20251001].copy()
    days26 = [20260105, 20260106, 20260107, 20260108, 20260109]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def setup(fw: pd.DataFrame, px_days):
            P = {"out": tmp / "out", "fwd_root": tmp / "out/fwd_root"}
            (P["out"] / "inputs").mkdir(parents=True, exist_ok=True)
            pd.DataFrame([(f"S{i}", d) for d in px_days for i in range(40)], columns=["symbol", "date"]).to_csv(
                P["out"] / "inputs/forward_prices.csv", index=False)
            csv = tmp / "fw.csv"; fw.to_csv(csv, index=False)
            return P, _fake_args(tmp, csv)

        new = pd.DataFrame({"symbol": "SET50", "date": days26, "open": 840.0, "high": 845.0, "low": 835.0, "close": 842.0,
                            "volume": 2.0 * 1.3e9})
        good = pd.concat([ov.assign(volume=ov.volume * 2.0), new], ignore_index=True)
        P, args = setup(good, days26)
        S0.set50_forward(args, P)
        out = pd.read_csv(P["fwd_root"] / S0.SET50_FWD)
        meta = json.loads((P["out"] / "inputs/set50_index_forward_meta.json").read_text(encoding="utf-8"))
        check(list(out.date) == days26 and abs(meta["volume_rescale"] - 2.0) < 1e-9 and np.allclose(out.volume, 1.3e9),
              "forward splice: 2026 rows only; volume rescaled by the median overlap ratio (2.0)")
        idx = FP.load_set50_index(P["fwd_root"] / S0.SET50_FWD)
        check(len(idx) == 1698 + 5 and idx.date.is_monotonic_increasing, "load_set50_index adds only the forward rows after 2025-12-30")
        bad = pd.concat([ov.assign(close=ov.close * 1.02), new], ignore_index=True)
        P, args = setup(bad, days26)
        try:
            S0.set50_forward(args, P); check(False, "a forward file 2% off the frozen index stops")
        except SystemExit:
            check(True, "a forward file 2% off the frozen index stops")
        P, args = setup(good, days26 + [20260112, 20260113, 20260114])
        try:
            S0.set50_forward(args, P); check(False, "a forward file missing 3 of the 2026 trading days stops")
        except SystemExit:
            check(True, "a forward file missing 3 of the 2026 trading days stops")


def test_real_data():
    import baselines_tabular as BT
    import features_proposal as FP
    a = json.loads((ROOT / "configs/v2_04.json").read_text(encoding="utf-8"))["inputs"]
    b = json.loads((ROOT / "configs/v2_11.json").read_text(encoding="utf-8"))["inputs"]
    data, _, full = BT.build_dataset(return_full=True)
    fa, fb = FP.add_price_tech(full, a), FP.add_price_tech(full, b)
    old = FP.channels(a)[:-3]
    check(all(np.array_equal(fa[c].to_numpy(), fb[c].to_numpy(), equal_nan=True) for c in old),
          "the 15 v2.04 price / technical channels are unchanged with the new inputs on")
    rows = fb.loc[data.index]
    check(np.allclose(rows.plus_di, rows.plus_di14) and np.allclose(rows.minus_di, rows.minus_di14),
          f"+DI / -DI = src/features.py plus_di14 / minus_di14 on all {len(rows):,} sample rows")
    check(rows.groupby("date")[FP.MARKET_CH].nunique().max().max() == 1 and not rows[FP.MARKET_CH].isna().any().any(),
          "SET50 channels: one value per date across stocks, never missing on sample rows")
    cfg = json.loads((ROOT / "configs/v2_11.json").read_text(encoding="utf-8"))
    cfg["model"].update(d_model=16, n_heads=2, n_layers=1, d_ff=32, fusion_hidden=8, d_stock=4, d_sector=4)
    cfg["training"]["batch_size"] = 64
    with tempfile.TemporaryDirectory() as tmp:
        cp, out = Path(tmp) / "cfg.json", Path(tmp) / "run.json"
        cp.write_text(json.dumps(cfg), encoding="utf-8")
        r = subprocess.run([sys.executable, str(ROOT / "scripts/train_v2xx.py"), "--config", str(cp), "--out", str(out),
                            "--smoke", "--seeds", "1", "2"], capture_output=True, text=True, cwd=ROOT)
        ok = r.returncode == 0 and out.exists() and "x 25;" in r.stdout
        check(ok, "end to end: train_v2xx.py --smoke builds 126 x 25 windows and finishes"
              + ("" if ok else "\n" + r.stdout[-1500:] + r.stderr[-1500:]))
        if ok:
            res = json.loads(out.read_text(encoding="utf-8"))
            sw = [k for k in res["runs"][0] if k.startswith("scale_weights")]
            g = res["runs"][0][sw[0]] if sw else {}
            check(not sw or {"technical_extra", "market"} <= set(g), f"scale weights reported per group incl. the new ones ({sorted(g)})")


if __name__ == "__main__":
    test_config()
    test_index_and_market()
    test_forward_splice()
    if "--no-data" not in sys.argv:
        test_real_data()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)
