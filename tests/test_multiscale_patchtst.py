"""Offline tests for src/multiscale_patchtst.py and the proof-of-life data handling.

Run from the repo root:  python tests/test_multiscale_patchtst.py      (needs torch; skips otherwise)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
try:
    import torch
except ImportError as e:
    print(f"skip: {e}")
    sys.exit(0)
import numpy as np  # noqa: E402

from multiscale_patchtst import MultiScalePatchTST, UncertaintyWeightedLoss  # noqa: E402
import proof_of_life_multiscale as P  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


def main():
    torch.manual_seed(0)
    m = MultiScalePatchTST(n_channels=6, lookback=60, patches=((5, 5), (20, 10)), horizons=(1, 5, 10))
    x = torch.randn(4, 60, 6)
    out = m(x)
    check(out["trend_logits"].shape == (4, 3, 3) and out["returns"].shape == (4, 3), "two heads: trend [B,3 horizons,3 classes], returns [B,3]")
    check([b.n_tokens for b in m.branches] == [12, 5], "patch (5,5) gives 12 tokens and (20,10) gives 5 over 60 days")

    m.eval()
    with torch.no_grad():
        a = m(x)["returns"][0]
        x2 = x.clone(); x2[1:] = x2[1:] * 50 + 7            # change every OTHER window in the batch
        b = m(x2)["returns"][0]
    check(torch.allclose(a, b, atol=1e-5), "RevIN: a window's output ignores the other windows (no cross-window statistics)")
    with torch.no_grad():
        x3 = x.clone(); x3[0] = x3[0] * 3 + 5               # affine change of the window itself
        c = m(x3)["returns"][0]
    check(torch.allclose(a, c, atol=1e-4), "RevIN: output is invariant to a window's own level and scale")

    m.train()
    crit = UncertaintyWeightedLoss()
    loss, parts = crit(m(x), torch.randint(0, 3, (4, 3)), torch.randn(4, 3))
    loss.backward()
    check(torch.isfinite(loss) and set(parts) == {"ce", "mse", "w_ce", "w_mse"}, "multi-task loss is finite and reports both parts")
    check(all(p.grad is not None and p.grad.abs().sum() > 0 for b in m.branches for p in b.embed.parameters()), "gradients reach every resolution branch")
    check(crit.log_var.grad is not None, "the task weights are learned")

    try:
        MultiScalePatchTST(n_channels=6, lookback=60, patches=((80, 10),)); ok = False
    except ValueError:
        ok = True
    check(ok, "a patch longer than the look-back is refused")
    try:
        m(torch.randn(2, 50, 6)); ok = False
    except ValueError:
        ok = True
    check(ok, "a window of the wrong length is refused")

    s = np.arange(200 * 6, dtype=np.float32).reshape(200, 6)
    X, Y = P.windows([s])
    t = P.LOOKBACK - 1
    check(X[0][-1][0] == s[t][0] and X[0][0][0] == s[0][0], "a window ends at day t and starts LOOKBACK-1 days earlier")
    check(Y[0][0] == s[t + 1, 0] and Y[0][1] == s[t + 1:t + 6, 0].sum(), "targets are the returns AFTER day t (1 and 5 days)")
    parts = P.split_chrono([s])
    tr_last_target_day = int(0.70 * 200) - 1
    check(parts["train"][0][-1][-1][0] == s[tr_last_target_day - max(P.HORIZONS)][0], "train windows stop early enough that their targets stay inside train")
    print("all multiscale_patchtst tests passed")


if __name__ == "__main__":
    main()
