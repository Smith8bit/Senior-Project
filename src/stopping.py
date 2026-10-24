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


# ---------------------------------------------------------------- v2.02: which epoch is kept (shared checkpoint)
SELECTIONS = ("f1_mean", "f1_up_to_ce_min")


class Selector:
    """Online choice of the kept epoch from per-epoch inner hold-out scores (no torch; tested in tests/test_v2xx_training.py).

    f1_mean          : keep the epoch with the highest 3-horizon mean macro-F1 so far (v2.01).
    f1_up_to_ce_min  : keep the epoch with the highest mean macro-F1 among epochs 1..e*, where e* is the epoch with the
                       lowest mean hold-out cross-entropy so far (v2.02, Siwat 2026-10-25). Ties: the earlier epoch wins
                       (strict > for F1, strict < for CE).
    update() returns flags; the caller keeps a weight snapshot whenever `prefix_improved` is True and makes that
    snapshot the kept one whenever `kept_changed` is True (the kept epoch is always a past prefix-best epoch).
    """

    def __init__(self, rule: str):
        if rule not in SELECTIONS:
            raise ValueError(f"unknown selection {rule!r}; choose from {SELECTIONS}")
        self.rule = rule
        self.prefix_epoch, self.prefix_f1 = 0, float("-inf")
        self.ce_min_epoch, self.ce_min = 0, float("inf")
        self.kept_epoch, self.kept_f1 = 0, float("-inf")

    def update(self, epoch: int, f1: float, ce: float | None = None) -> dict:
        prefix_improved = f1 > self.prefix_f1
        if prefix_improved:
            self.prefix_epoch, self.prefix_f1 = epoch, f1
        ce_improved = False
        if self.rule == "f1_up_to_ce_min":
            if ce is None:
                raise ValueError("f1_up_to_ce_min needs the hold-out cross-entropy")
            ce_improved = ce < self.ce_min
            if ce_improved:
                self.ce_min_epoch, self.ce_min = epoch, ce
            new_kept = (self.prefix_epoch, self.prefix_f1) if ce_improved else (self.kept_epoch, self.kept_f1)
        else:
            new_kept = (self.prefix_epoch, self.prefix_f1)
        kept_changed = new_kept[0] != self.kept_epoch
        self.kept_epoch, self.kept_f1 = new_kept
        return {"prefix_improved": prefix_improved, "ce_improved": ce_improved, "kept_changed": kept_changed}


def select_offline(rule: str, f1: list, ce: list | None = None) -> int:
    """The kept epoch (1-based) from complete per-epoch lists; the reference the online Selector is tested against."""
    end = len(f1) if rule == "f1_mean" else min(range(len(ce)), key=lambda i: (ce[i], i)) + 1
    window = f1[:end]
    return max(range(len(window)), key=lambda i: (window[i], -i)) + 1
