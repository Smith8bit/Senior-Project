"""Checks for v2.04 news-robust training (src/train_utils.blank_windows, scripts/train_v2xx news_blank_p).

  python tests/test_news_blank.py        # synthetic parts ~1 min; the real-data check builds the dataset (~1 min more)

1. blank_windows: chosen windows get the fill on every real day of the sentiment channels only; other windows,
   other channels and padding days (all channels 0) are untouched; the input array is not modified.
2. train_one with news_blank_p = 0 gives exactly the same history and weights as a config without the key
   (v2.03 behaviour is unchanged); with p = 0.2 about 20% of training windows are blanked each epoch.
3. Real data (configs/v2_04.json, --smoke subset): the blanking value of each sentiment channel equals the value that
   no-news days already have in the standardised windows, i.e. a blanked window looks exactly like a window with no
   news (the same inputs v2.03beta used for "news blanked").
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

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


def test_blank_windows():
    r = np.random.default_rng(0)
    X = r.normal(size=(6, 10, 5)).astype(np.float32)
    X[2, :3, :] = 0.0                                                # window 2 starts with 3 padding days
    X0 = X.copy()
    out = TU.blank_windows(X, np.array([1, 2]), [3, 4], [0.5, -0.25])
    check(np.array_equal(X, X0), "input array not modified")
    check(np.all(out[1, :, 3] == 0.5) and np.all(out[1, :, 4] == -0.25), "chosen window: fill on every day of the sentiment channels")
    check(np.all(out[2, :3] == 0) and np.all(out[2, 3:, 3] == 0.5), "padding days stay 0; real days get the fill")
    check(np.array_equal(out[:, :, :3], X0[:, :, :3]), "other channels untouched")
    check(np.array_equal(out[[0, 3, 4, 5]], X0[[0, 3, 4, 5]]), "other windows untouched")
    check(np.array_equal(TU.blank_windows(X, np.array([], int), [3], [1.0]), X0), "no rows -> unchanged copy")


def test_train_one():
    import train_v2xx as TV
    from test_v2xx_training import small_cfg, synthetic_D
    D = synthetic_D()
    D["news_blank"] = {"channels": [15, 16, 17], "values": [0.1, -0.2, 0.3]}
    base = small_cfg(False)
    base["training"].update(max_epochs=3, patience=50)
    zero = copy.deepcopy(base); zero["training"]["news_blank_p"] = 0.0; zero["training"]["news_blank_unit"] = "window"
    _, m0, s0, _ = TV.train_one(base, 7, D, 18, 10, 5, 3)
    _, m1, s1, _ = TV.train_one(zero, 7, D, 18, 10, 5, 3)
    same_hist = [(h["train_loss"], h["es_macro_f1"]) for h in m0["history"]] == [(h["train_loss"], h["es_macro_f1"]) for h in m1["history"]]
    same_w = all(torch.equal(s0[0][k], s1[0][k]) for k in s0[0])
    check(same_hist and same_w, "news_blank_p = 0 is exactly the run without the key (v2.03 behaviour)")
    p2 = copy.deepcopy(zero); p2["training"]["news_blank_p"] = 0.2
    _, m2, _, _ = TV.train_one(p2, 7, D, 18, 10, 5, 3)
    sh = [h["news_blanked_share"] for h in m2["history"]]
    check(all(0.15 <= s <= 0.25 for s in sh), f"p = 0.2 blanks about 20% of training windows per epoch ({sh})")
    check(m2["history"][0]["train_loss"] != m0["history"][0]["train_loss"], "blanking changes training")
    check(all(h["news_blanked_share"] == 0 for h in m0["history"]), "no blanking logged without the option")


def test_real_fill():
    import run_config as RC
    import train_v2xx as TV
    cfg = RC.load(ROOT / "configs/v2_04.json")
    B = TV.build_data(cfg, ROOT / TV.NEWS_FILE, None, smoke=True)
    nb, X = B["D"]["news_blank"], B["D"]["val"]["X"]
    ok = []
    for c, v in zip(nb["channels"], nb["values"]):
        vals, counts = np.unique(X[:, :, c], return_counts=True)
        ok.append(np.float32(v) == vals[counts.argmax()])
    check(all(ok) and nb["channels"] == [B["channels"].index(c) for c in ("p_pos", "p_neg", "p_neu")],
          f"real data: blanking value = the no-news value already in the windows ({np.round(nb['values'], 4)})")


if __name__ == "__main__":
    test_blank_windows()
    test_train_one()
    if "--no-data" not in sys.argv:
        test_real_fill()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)
