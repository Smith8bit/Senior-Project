"""Checks for src/stopping.py (E19 stop rule). Run: python tests/test_stopping.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import stopping as S  # noqa: E402

fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    fails += not cond


def simulate(rule, curves, patience):
    """curves[epoch] = per-horizon scores; returns (stop epoch, best epoch per horizon)."""
    nh = len(curves[0])
    best_h, bad_h, best_ep = [-1.0] * nh, [0] * nh, [0] * nh
    best_mean, bad_mean = -1.0, 0
    for ep, sc in enumerate(curves, start=1):
        for j in S.update_per_horizon(best_h, bad_h, sc):
            best_ep[j] = ep
        m = sum(sc) / nh
        if m > best_mean:
            best_mean, bad_mean = m, 0
        else:
            bad_mean += 1
        if S.should_stop(rule, bad_mean, bad_h, patience):
            return ep, best_ep
    return len(curves), best_ep


# E10-like pattern: 5 d and 10 d peak at epoch 2, 1 d keeps rising slowly until epoch 15
curves = [[0.30 + 0.004 * min(e, 15), 0.39 if e == 2 else 0.36, 0.40 if e == 2 else 0.37] for e in range(1, 31)]
stop_mean, ep_mean = simulate("mean", curves, 10)
stop_every, ep_every = simulate("every", curves, 10)
check(stop_mean == 12, f"mean rule stops 10 epochs after the mean peak (got {stop_mean}; mean peaks at epoch 2)")
check(ep_mean[0] == 12, f"under the mean rule the 1-day best is cut off at the last epoch (got {ep_mean[0]})")
check(stop_every == 25, f"every rule waits until the 1-day head has stalled for 10 epochs (got {stop_every})")
check(ep_every == [15, 2, 2], f"every rule: 1 d best at 15, 5 and 10 d keep epoch 2 (got {ep_every})")

# max epochs cap
rising = [[0.3 + 0.001 * e] * 3 for e in range(1, 31)]
check(simulate("every", rising, 10)[0] == 30, "every rule never runs past max epochs")
# identical when all horizons move together
together = [[0.35 if e == 4 else 0.33] * 3 for e in range(1, 31)]
check(simulate("mean", together, 10)[0] == simulate("every", together, 10)[0] == 14, "both rules agree when horizons peak together")
# ties do not count as improvement
best_h, bad_h = [0.5], [0]
S.update_per_horizon(best_h, bad_h, [0.5])
check(bad_h == [1], "a tie is not an improvement")
try:
    S.should_stop("median", 0, [0], 10); check(False, "unknown rule rejected")
except ValueError:
    check(True, "unknown rule rejected")

print(f"\n{fails} failure(s)")
sys.exit(1 if fails else 0)
