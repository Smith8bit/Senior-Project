"""Checks for scripts/step0_v2_03beta.py (post-hoc methods; no data, no torch). Run: python tests/test_step0_v2_03beta.py"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import step0_v2_03beta as S  # noqa: E402

fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    fails += not cond


rng = np.random.default_rng(0)

# --- temperature / bias-corrected temperature scaling ------------------------------------------------------------
n = 20000
logits = rng.normal(0, 1.0, (n, 3))
true = S.softmax(logits)
y = np.array([rng.choice(3, p=p) for p in true])
over = S.softmax(logits * 2.5)                        # over-confident model: true temperature 2.5
th = S.fit_affine(over, y, bias=False)
check(abs(th[0] - 0.4) < 0.03 and th[1] == th[2] == 0.0, f"temperature recovered (a = {th[0]:.3f}, expected 0.4)")
check(S.nll(S.apply_affine(over, th), y) < S.nll(over, y), "temperature scaling lowers log-loss")
check((S.apply_affine(over, th).argmax(1) == over.argmax(1)).all(), "temperature scaling never changes a call")
shifted = S.softmax(logits * 2.5 + np.array([0.0, 0.8, -0.5]))
tb = S.fit_affine(shifted, y, bias=True)
check(abs(tb[0] - 0.4) < 0.03 and abs(tb[1] + 0.32) < 0.06 and abs(tb[2] - 0.2) < 0.06,
      f"bias-corrected scaling recovers a and biases ({np.round(tb, 3)}, expected [0.4, -0.32, 0.2])")
check(S.nll(S.apply_affine(shifted, tb), y) <= S.nll(S.apply_affine(shifted, S.fit_affine(shifted, y, False)), y) + 1e-9,
      "bias-corrected scaling fits at least as well as temperature only")

anti = S.softmax(-logits * 2.5)                       # anti-informative: the best free fit would flip every call
ta, tab = S.fit_affine(anti, y, bias=False), S.fit_affine(anti, y, bias=True)
check(ta[0] == S.A_MIN and tab[0] == S.A_MIN, f"temperature stays positive on anti-informative probabilities ({ta[0]}, {tab[0]})")
check((S.apply_affine(anti, ta).argmax(1) == anti.argmax(1)).all(), "positive temperature keeps every call on anti-informative input")

# --- per-date rank with fixed shares ------------------------------------------------------------------------------
p = S.softmax(rng.normal(0, 1, (50 * 4, 3)))
dates = np.repeat(np.array(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]), 50)
c = S.rank_calls(p, dates)
check(all(np.bincount(c[dates == d], minlength=3).tolist() == [17, 16, 17] for d in np.unique(dates)),
      "rank: 50 stocks -> 17 down / 16 flat / 17 up on every date")
q = S.softmax(rng.normal(0, 1, (7, 3)))
cc = S.rank_calls(q, np.zeros(7, int))
best = max((sum(np.log(q[i, a[i]]) for i in range(7)), a) for a in itertools.product(range(3), repeat=7)
           if list(np.bincount(a, minlength=3)) == [2, 3, 2])
check(abs(sum(np.log(q[i, cc[i]]) for i in range(7)) - best[0]) < 1e-12, "rank: assignment is optimal (brute force, 7 stocks)")
check((S.rank_calls(S.apply_affine(p, np.array([0.4, 0.3, -0.2])), dates) == c).all(),
      "rank: same calls after temperature + biases")

# --- Saerens EM per date --------------------------------------------------------------------------------------------
src = np.array([1 / 3, 1 / 3, 1 / 3]); tgt = np.array([0.6, 0.25, 0.15])
mu = np.eye(3) * 1.5
ys = rng.choice(3, size=4000, p=tgt)
x = mu[ys] + rng.normal(0, 1, (4000, 3))
post_src = S.softmax(x @ mu.T - 0.5 * (mu ** 2).sum(1))      # Bayes posterior under equal priors
adj = S.em_adjust(post_src, src, np.zeros(4000, int))
check(np.abs(adj.mean(0) - tgt).max() < 0.03, f"EM recovers a shifted class mix ({np.round(adj.mean(0), 3)} vs {tgt})")
check(S.f1(ys, adj.argmax(1)) >= S.f1(ys, post_src.argmax(1)) - 0.01 and np.allclose(adj.sum(1), 1), "EM output is a distribution")
same = S.em_adjust(post_src[:300], post_src[:300].mean(0), np.zeros(300, int))
check(np.allclose(same, post_src[:300], atol=1e-5), "EM leaves probabilities alone when their mean equals the start prior")

# --- sentiment variants -------------------------------------------------------------------------------------------
full = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 4 + ["2024-06-01"] * 6), "ch0": rng.normal(size=10)})
vals = rng.dirichlet([1, 1, 1], 10); has = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1], float)
vals[has == 0] = 1 / 3
for i, c_ in enumerate(S.FP.SENT_CH):
    full[c_] = vals[:, i]
chs = ["ch0"] + list(S.FP.SENT_CH)
V = S.SentimentVariants(full, chs, pd.Index(range(4)), (1 / 3, 1 / 3, 1 / 3), has)
per = (full.date >= "2024-03-01").to_numpy()
zp = V.get("placebo", per, 1)
check(np.array_equal(zp[~per], V.z[~per]), "placebo: rows outside the period unchanged")
m = per & (has == 1)
check(np.array_equal(zp[per & (has == 0)], V.z[per & (has == 0)]), "placebo: no-news days keep the fill")
check(sorted(map(tuple, zp[m][:, 1:].round(6))) == sorted(map(tuple, V.z[m][:, 1:].round(6))) and np.array_equal(zp[:, 0], V.z[:, 0]),
      "placebo: the same news vectors, only moved; price channel untouched")
check(not np.array_equal(zp, V.get("placebo", per, 2)), "placebo: different shuffles differ")
zb = V.get("blank", per)
check(np.allclose(zb[per][:, 1:], zb[per][0, 1:]) and np.array_equal(zb[~per], V.z[~per]), "blank: every day in the period gets the fill")

print(f"\n{fails} failure(s)")
sys.exit(1 if fails else 0)
