"""Checks for src/posthoc.py (E24 blend helpers, E30 calibration, D02 bootstrap). No data or torch needed.

  python tests/test_posthoc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import posthoc as PH  # noqa: E402

FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def main():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 5000)
    yhat = np.where(rng.random(5000) < 0.4, y, rng.integers(0, 3, 5000))
    yhat[:3] = 0
    check(abs(PH.macro_f1(y, yhat) - f1_score(y, yhat, average="macro", labels=[0, 1, 2], zero_division=0)) < 1e-12,
          "macro-F1 from counts equals sklearn")
    only0 = np.zeros_like(y)
    check(abs(PH.macro_f1(y, only0) - f1_score(y, only0, average="macro", labels=[0, 1, 2], zero_division=0)) < 1e-12,
          "macro-F1 with a class never predicted equals sklearn (zero_division=0)")

    # calibration: a model that under-calls "flat" gains from a positive flat bias
    logits = np.eye(3)[y] * 0.8 + rng.normal(0, 1, (5000, 3))
    logits[:, 1] -= 0.7
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    check((PH.predict(p) == p.argmax(1)).all(), "no bias = plain argmax")
    fit = PH.fit_bias([(p[:2500], y[:2500]), (p[2500:], y[2500:])])
    check(fit["bias_flat"] > 0.3 and fit["fitted_mean_macro_f1"] > fit["uncalibrated_mean_macro_f1"] + 0.01,
          f"calibration finds the flat bias ({fit['bias_flat']:+.2f}) and gains {fit['fitted_mean_macro_f1'] - fit['uncalibrated_mean_macro_f1']:+.3f}")
    cal = PH.predict(p, (fit["bias_flat"], fit["bias_up"]))
    check(abs(PH.macro_f1(y, cal) - fit["fitted_mean_macro_f1"]) < 0.01, "applying the fitted bias reproduces the fitted score")
    sym = np.exp(np.eye(3)[y] * 2 + rng.normal(0, 0.1, (5000, 3)))
    fit0 = PH.fit_bias([(sym / sym.sum(1, keepdims=True), y)])
    check(fit0["bias_flat"] == 0 and fit0["bias_up"] == 0, "an already perfect model gets no bias")

    # bootstrap
    dates = np.repeat(np.arange(200), 40)
    yy = rng.integers(0, 3, len(dates))
    a = np.where(rng.random((3, len(dates))) < 0.2, yy, rng.integers(0, 3, (3, len(dates))))
    bt = PH.block_bootstrap(dates, yy, {"a": a, "same": a.copy(), "rand": rng.integers(0, 3, (3, len(dates)))},
                            pairs=[("a", "same"), ("a", "rand")], b=500, bar=0.40)
    va = bt["values"]["a"]
    check(va["ci95"][0] < va["point"] < va["ci95"][1], f"point {va['point']} inside its interval {va['ci95']}")
    point_direct = np.mean([PH.macro_f1(yy, a[s]) for s in range(3)])
    check(abs(va["point"] - point_direct) < 1e-4, "point value = mean of the seeds' macro-F1")
    check(bt["differences"]["a - same"]["ci95"] == [0.0, 0.0], "a model against itself: difference interval is exactly 0")
    d = bt["differences"]["a - rand"]
    check(d["ci95"][0] > 0 and d["share_above_0"] == 1.0, f"a better model is detected: difference interval {d['ci95']}")
    check(0 < va["se"] < 0.03 and "share_of_resamples_at_or_above_bar" in va, f"standard error {va['se']} is plausible")
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
