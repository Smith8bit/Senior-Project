"""Learning curves for a v2.xx run (or the PatchTST baseline), read from its run JSON.

  python scripts/plot_v2xx_training.py results/v2xx/v2_01.json              # writes v2_01_training.png beside it

One row per seed, three panels per row, all against epoch:
  1. Training loss by task: cross-entropy (trend) and MSE (price, on the train-standardised log return).
  2. Inner hold-out macro-F1 per horizon (the early-stopping signal), with chance (1/3) dashed.
  3. Inner hold-out price RMSE per horizon, in train-standardised units (1.0 = the size of a typical return).
The epoch whose weights were kept is marked in every panel. Validation scores are NOT plotted per epoch: the
validation block is scored once, after training, so it never steers training.

Colours: horizons 1 / 5 / 10 d always use the same three hues (validated palette: blue, orange, aqua) and are also
labelled directly at the line ends; the two task losses are both neutral ink, told apart by solid vs dashed line
and a direct label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HORIZON = {"1 d": "#2a78d6", "5 d": "#eb6834", "10 d": "#1baf7a"}


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("epoch", fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def _end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK2,
                annotation_clip=False)
    ax.plot([x], [y], marker="o", markersize=4, color=color)


def plot(run_path: Path, out_path: Path | None = None) -> Path:
    res = json.loads(run_path.read_text(encoding="utf-8"))
    runs = res["runs"]
    if not runs:
        raise SystemExit(f"{run_path} has no finished seeds yet")
    fig, axes = plt.subplots(len(runs), 3, figsize=(15, 3.6 * len(runs)), squeeze=False, facecolor=SURFACE)
    name = res["config"]["name"]
    for r, run in enumerate(runs):
        h = run["history"]
        ep = [e["epoch"] for e in h]
        best = run["best_epoch"]
        a0, a1, a2 = axes[r]
        # 1. training loss by task
        ce = [e.get("train_ce") for e in h]
        pr = [e.get("train_price") for e in h]
        a0.plot(ep, ce, color=INK, linewidth=2, label="trend: cross-entropy")
        if all(v is not None for v in pr):
            a0.plot(ep, pr, color=INK2, linewidth=2, linestyle="--", label="price: MSE")
            _end_label(a0, ep[-1], pr[-1], "price", INK2)
        _end_label(a0, ep[-1], ce[-1], "trend", INK)
        _style(a0, f"seed {run['seed']} · weights kept from epoch {best} · training loss by task", "loss")
        # 2. inner hold-out macro-F1 per horizon
        for j, (lab, col) in enumerate(HORIZON.items()):
            y = [e["es_per_horizon"][j] for e in h]
            a1.plot(ep, y, color=col, linewidth=2, label=lab)
            _end_label(a1, ep[-1], y[-1], lab, col)
        a1.axhline(1 / 3, color=INK2, linewidth=1, linestyle=":", label="chance (1/3)")
        _style(a1, "inner hold-out macro-F1 (early-stopping signal)", "macro-F1")
        # 3. inner hold-out price RMSE per horizon
        for j, (lab, col) in enumerate(HORIZON.items()):
            y = [e["es_return_rmse_std"][j] for e in h]
            a2.plot(ep, y, color=col, linewidth=2, label=lab)
            _end_label(a2, ep[-1], y[-1], lab, col)
        _style(a2, "inner hold-out price RMSE (train-standardised)", "RMSE")
        for ax in (a0, a1, a2):
            ax.axvline(best, color=INK2, linewidth=1, linestyle="-", alpha=0.6)
            ax.margins(x=0.08)
    fig.suptitle(f"{name} — learning curves (validation block not used during training)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, linewidth=2, label=l) for l, c in HORIZON.items()]
    handles += [Line2D([], [], color=INK2, linewidth=1, linestyle=":", label="chance (1/3)"),
                Line2D([], [], color=INK, linewidth=2, label="trend loss (cross-entropy)"),
                Line2D([], [], color=INK2, linewidth=2, linestyle="--", label="price loss (MSE)"),
                Line2D([], [], color=INK2, linewidth=1, alpha=0.6, label="epoch whose weights were kept")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out_path = out_path or run_path.with_name(run_path.stem + "_training.png")
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print("wrote", plot(Path(a.run_json), Path(a.out) if a.out else None))
