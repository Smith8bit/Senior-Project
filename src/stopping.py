"""Early-stopping bookkeeping for the main model (experiment ledger E12 / E19). No torch, so it can be tested anywhere.

Two stop rules:
  mean  : stop when the mean inner hold-out macro-F1 over the 3 horizons has not improved for `patience` epochs
          (every run up to batch 1).
  every : E19. Stop only when EVERY horizon has gone `patience` epochs without improving its own inner hold-out
          macro-F1 (or at max epochs). Each horizon keeps its own best state either way (E12).
"""
from __future__ import annotations

RULES = ("mean", "every")


def update_per_horizon(best_h: list, bad_h: list, scores: list) -> list:
    """Update per-horizon bests and stall counters in place; return the horizons that improved this epoch."""
    improved = []
    for j, s in enumerate(scores):
        if s > best_h[j]:
            best_h[j], bad_h[j] = s, 0
            improved.append(j)
        else:
            bad_h[j] += 1
    return improved


def should_stop(rule: str, bad_mean: int, bad_h: list, patience: int) -> bool:
    if rule == "mean":
        return bad_mean >= patience
    if rule == "every":
        return all(b >= patience for b in bad_h)
    raise ValueError(f"unknown stop rule {rule!r}; choose from {RULES}")
