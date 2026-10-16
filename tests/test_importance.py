"""Checks for src/importance.py. Run: python tests/test_importance.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import importance as I  # noqa: E402

fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    fails += not cond


def macro_f1(y, p):
    f = []
    for c in range(3):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


rng = np.random.default_rng(0)
N, L = 3000, 10
X = rng.normal(size=(N, L, 4)).astype(np.float32)
G = rng.integers(0, 3, N)
y1 = np.digitize(X[:, -1, 0], [-0.43, 0.43])            # horizon 1 depends on input "a" only
y2 = (G + (X[:, :, 2].mean(1) > 0)) % 3                  # horizon 2 depends on input "c" and the group label
names = ["a", "b", "c", "d"]
X0 = X.copy()


def score(Xs, Gs):
    p1 = np.digitize(Xs[:, -1, 0], [-0.43, 0.43])
    p2 = (Gs + (Xs[:, :, 2].mean(1) > 0)) % 3
    return [macro_f1(y1, p1), macro_f1(y2, p2)]


res = I.permutation_importance(score, X, G, names, {"ab": ["a", "b"], "cd": ["c", "d"], "none": []}, repeats=3, seed=1)
m = {n: np.mean(res["single"][n], axis=0) for n in res["single"]}
check(res["base_macro_f1"] == [1.0, 1.0], "the toy model is perfect before shuffling")
check(m["a"][0] > 0.4 and abs(m["a"][1]) < 1e-9, f"shuffling 'a' hurts horizon 1 only (drops {m['a'].round(3)})")
check(m["c"][1] > 0.2 and abs(m["c"][0]) < 1e-9, f"shuffling 'c' hurts horizon 2 only (drops {m['c'].round(3)})")
check(np.allclose(m["b"], 0) and np.allclose(m["d"], 0), "unused inputs 'b' and 'd' show zero importance")
check(m[I.GROUP_LABEL][1] > 0.2 and abs(m[I.GROUP_LABEL][0]) < 1e-9, "shuffling the group label hurts horizon 2 only")
check(np.mean(res["groups"]["ab"], axis=0)[0] > 0.4 and "none" not in res["groups"], "groups work; empty groups are skipped")
check(np.array_equal(X, X0), "X is restored exactly after every shuffle")
r2 = I.permutation_importance(score, X, G, names, {}, repeats=3, seed=1)
check(r2["single"] == res["single"], "same seed gives the same result")
Xt = rng.normal(size=(500, 3)); yt = np.digitize(Xt[:, 1], [-0.43, 0.43])
rt = I.permutation_importance(lambda A, _: [macro_f1(yt, np.digitize(A[:, 1], [-0.43, 0.43]))], Xt, None, ["x", "y", "z"], {}, 2, 0)
check(np.mean(rt["single"]["y"]) > 0.4 and I.GROUP_LABEL not in rt["single"], "2-D (tabular) input works; no group label when G is None")
try:
    I.permutation_importance(score, X, G, ["a"], {}); check(False, "name count mismatch rejected")
except ValueError:
    check(True, "name count mismatch rejected")
print(f"\n{fails} failure(s)")
sys.exit(1 if fails else 0)
