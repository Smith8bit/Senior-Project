"""Post-processing of saved validation predictions (batches 4 and 7; claude/batches-4-7-design-2026-10-16.md).

No training and no torch, so everything here runs from the <out>_preds.csv.gz files the run scripts write.
  * probs / seeds_of: read class probabilities per seed and horizon from a predictions file.
  * macro_f1_groups: macro-F1 of several (seed, window) groups at once, from confusion counts.
  * fit_bias: E30 decision calibration. Two additive biases on the log-probabilities of "flat" and "up" ("down" fixed
    at 0) chosen on a grid to maximise the mean macro-F1 over the given groups (Lipton, Elkan & Naryanaswamy 2014,
    "Optimal Thresholding of Classifiers to Maximize F1 Measure", ECML PKDD). Grid -1.00 .. +1.00, step 0.05.
  * block_bootstrap: D02 date-block bootstrap of the 3-seed mean macro-F1 (and of paired differences). Validation
    dates are resampled in moving blocks of 10 trading days (the longest horizon, so overlapping labels stay
    together); every stock on a drawn date comes along, so cross-stock correlation on a day is kept.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

CLASSES = ("down", "flat", "up")
HORIZONS = (1, 5, 10)
GRID = np.round(np.arange(-1.0, 1.0001, 0.05), 2)


def seeds_of(df: pd.DataFrame, prefix: str) -> list[int]:
    return sorted({int(m.group(1)) for c in df.columns if (m := re.fullmatch(rf"{re.escape(prefix)}_s(\d+)_p1_down", c))})


def probs(df: pd.DataFrame, prefix: str, seed: int, h: int) -> np.ndarray:
    return df[[f"{prefix}_s{seed}_p{h}_{c}" for c in CLASSES]].to_numpy(float)


def merge(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """One-to-one merge of two prediction files on (symbol, date); labels must agree."""
    key = ["symbol", "date"]
    df = a.merge(b, on=key, suffixes=("", "_b"), validate="one_to_one")
    assert len(df) == len(a) == len(b), f"rows differ: {len(a)} vs {len(b)}, matched {len(df)}"
    for h in HORIZONS:
        assert (df[f"y_{h}"] == df[f"y_{h}_b"]).all(), f"labels differ at h={h}"
    return df.drop(columns=[c for c in df.columns if c.endswith("_b")])


def f1_from_counts(cnt: np.ndarray) -> np.ndarray:
    """cnt [..., 3, 3] (rows actual, cols predicted) -> macro-F1 [...] (classes with no support and no calls count 0,
    as sklearn's zero_division=0)."""
    tp = np.diagonal(cnt, axis1=-2, axis2=-1).astype(float)
    pred, act = cnt.sum(-2).astype(float), cnt.sum(-1).astype(float)
    denom = pred + act
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean(-1)


def macro_f1(y: np.ndarray, yhat: np.ndarray) -> float:
    cnt = np.bincount(y * 3 + yhat, minlength=9).reshape(3, 3)
    return float(f1_from_counts(cnt))


def predict(p: np.ndarray, bias=(0.0, 0.0)) -> np.ndarray:
    """argmax of log p + (0, b_flat, b_up)."""
    lp = np.log(np.clip(p, 1e-9, 1.0))
    lp[:, 1] += bias[0]; lp[:, 2] += bias[1]
    return lp.argmax(1)


def fit_bias(groups: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """groups: list of (p [n, 3], y [n]). Returns the grid bias maximising the mean macro-F1 over the groups.
    Ties are broken towards the smallest |b_flat| + |b_up| (no calibration if nothing beats it)."""
    lp = [np.log(np.clip(p, 1e-9, 1.0)) for p, _ in groups]
    ys = [y.astype(int) for _, y in groups]
    best_m, best_norm, best_b = -1.0, np.inf, (0.0, 0.0)
    for bf in GRID:
        for bu in GRID:
            f = []
            for l, y in zip(lp, ys):
                yhat = np.argmax(np.stack([l[:, 0], l[:, 1] + bf, l[:, 2] + bu], 1), 1)
                f.append(macro_f1(y, yhat))
            m, norm = float(np.mean(f)), abs(bf) + abs(bu)
            if m > best_m + 1e-9 or (abs(m - best_m) <= 1e-9 and norm < best_norm):
                best_m, best_norm, best_b = m, norm, (float(bf) + 0.0, float(bu) + 0.0)
    best = (best_m, best_norm, best_b)
    base = float(np.mean([macro_f1(y, l.argmax(1)) for l, y in zip(lp, ys)]))
    return {"bias_flat": best[2][0], "bias_up": best[2][1], "fitted_mean_macro_f1": round(best[0], 4),
            "uncalibrated_mean_macro_f1": round(base, 4), "groups": len(groups)}


def block_bootstrap(dates: np.ndarray, y: np.ndarray, preds: dict, pairs=(), b: int = 2000, block: int = 10,
                    seed: int = 20261016, bar: float | None = None) -> dict:
    """dates [n]; y [n]; preds {name: [S, n] predicted classes (S seeds)}. Statistic = mean over seeds of macro-F1.
    pairs: (name_a, name_b) differences a - b computed on the same resamples. Returns point values and 95% intervals."""
    ud, di = np.unique(dates, return_inverse=True)
    nd = len(ud)
    counts = {}
    for k, P in preds.items():
        P = np.atleast_2d(P)
        c = np.zeros((P.shape[0], nd, 9))
        for s in range(P.shape[0]):
            np.add.at(c[s], (di, y * 3 + P[s]), 1)
        counts[k] = c.reshape(P.shape[0], nd, 3, 3)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(nd / block))
    starts = rng.integers(0, nd - block + 1, size=(b, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(b, -1)[:, :nd]   # [b, nd] date indices
    w = np.zeros((b, nd))
    np.add.at(w, (np.repeat(np.arange(b), nd), idx.ravel()), 1)                           # times each date is drawn
    stat = {}
    for k, c in counts.items():
        tot = np.einsum("bd,sdij->bsij", w, c)                                              # resampled confusion
        stat[k] = f1_from_counts(tot).mean(1)                                               # [b]
    out = {"dates": int(nd), "resamples": b, "block_days": block, "values": {}}
    for k, c in counts.items():
        point = float(f1_from_counts(c.sum(1)).mean())
        lo, hi = np.percentile(stat[k], [2.5, 97.5])
        v = {"point": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "se": round(float(stat[k].std()), 4)}
        if bar is not None:
            v["share_of_resamples_at_or_above_bar"] = round(float((stat[k] >= bar).mean()), 3)
        out["values"][k] = v
    out["differences"] = {}
    for a, bb in pairs:
        d = stat[a] - stat[bb]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["differences"][f"{a} - {bb}"] = {"point": round(out["values"][a]["point"] - out["values"][bb]["point"], 4),
                                            "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                            "share_above_0": round(float((d > 0).mean()), 3)}
    return out
