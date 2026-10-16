"""Permutation importance (feature-importance diagnostic, 2026-10-16). No torch, so it can be tested anywhere.

For each input (or group of inputs) the validation rows of that input are shuffled across stock-days, keeping each
row's whole look-back window together, and the drop in macro-F1 is recorded. A large drop = the trained model relies
on that input; ~0 = it does not (it says nothing about whether the input COULD help a different model).
The group label (sector) is shuffled the same way. Every shuffle is undone before the next one (X is restored exactly).

score_fn(X, G) -> list of macro-F1, one per horizon, for the model(s) being explained.
X: [N, L, C] (sequence model) or [N, C] (tabular); the last axis is the input axis.
"""
from __future__ import annotations

import numpy as np

GROUP_LABEL = "sector_label"


def permutation_importance(score_fn, X: np.ndarray, G, names: list[str], groups: dict[str, list[str]],
                           repeats: int = 3, seed: int = 0) -> dict:
    if X.shape[-1] != len(names):
        raise ValueError(f"{len(names)} names for {X.shape[-1]} inputs")
    idx = {n: i for i, n in enumerate(names)}
    base = [float(b) for b in score_fn(X, G)]
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(len(X)) for _ in range(repeats)]

    def drops(cols, shuffle_label=False):
        out = []
        for perm in perms:
            saved = X[..., cols].copy()
            X[..., cols] = saved[perm]
            try:
                s = score_fn(X, G[perm] if shuffle_label else G)
            finally:
                X[..., cols] = saved
            out.append([round(b - float(v), 5) for b, v in zip(base, s)])
        return out                                            # [repeat][horizon]

    single = {n: drops([idx[n]]) for n in names}
    grp = {g: drops([idx[n] for n in members]) for g, members in groups.items() if members}
    if G is not None:
        single[GROUP_LABEL] = drops([], shuffle_label=True)
    return {"method": "permutation (rows shuffled across validation stock-days; whole window per input)",
            "repeats": repeats, "seed": seed, "base_macro_f1": [round(b, 5) for b in base],
            "single": single, "groups": grp}
