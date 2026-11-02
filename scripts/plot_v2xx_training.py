"""Learning curves for a v2.xx run (or the PatchTST baseline), read from its run JSON.

  python scripts/plot_v2xx_training.py results/v2xx/v2_01.json   # writes v2_01_training.png (+ v2_01_calls.png) beside it

One row per seed, all against epoch:
  1. Training loss by task: cross-entropy (trend) and MSE (price, on the train-standardised log return).
  2. Inner hold-out macro-F1 per horizon (the selection signal), with chance (1/3) dotted.
  3. Inner hold-out cross-entropy per horizon, with the uniform guess (ln 3) dotted   [runs from v2.02 on]
  4. Inner hold-out price RMSE per horizon, in train-standardised units (1.0 = the size of a typical return).
  5. Share of hold-out calls unchanged from the previous epoch, per horizon             [runs from v2.02 on]
  6. Shared-layer gradients: the price task's share of the weighted pull on the shared layers, and the cosine between
     the two tasks' gradients (epoch 0 = before training)                                [runs with grad_diagnostics]
The epoch whose weights were kept is marked in every panel (solid); with the v2.02 selection rule the
lowest-cross-entropy epoch is marked too (dashed).
A second figure (<run>_calls.png, v2.02 on) shows the predicted-class shares on the hold-out per seed and horizon
(down / flat / up), each beside the class's true share (dotted). Validation scores are NOT plotted per epoch: the
validation block is scored once, after training, so it never steers training.

Colours: horizons 1 / 5 / 10 d always use the same three hues (validated palette: blue, orange, aqua) and are also
labelled directly at the line ends; the two task losses are both neutral ink, told apart by solid vs dashed line
and a direct label.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HORIZON = {"1 d": "#2a78d6", "5 d": "#eb6834", "10 d": "#1baf7a"}     # each horizon keeps its colour across runs
CLASS = {"down": "#eda100", "flat": "#e87ba4", "up": "#008300"}   # validated (palette slots 4-6); labelled at line ends
CLASS_B2 = {"down": "#eda100", "not_down": "#008300"}              # v2.08 B2 label: the same colours for down / up side


def class_colours(res: dict) -> dict:
    """{class: colour} in the model's class order (3 classes, or 2 for v2.08's B2 label)."""
    return CLASS_B2 if res["config"].get("protocol", {}).get("label_rule") == "b2_raw_close" else CLASS


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("epoch", fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for s in ax.spines.values():
        s.set_visible(False)


def _end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK2,
                annotation_clip=False)
    ax.plot([x], [y], marker="o", markersize=4, color=color)


def horizon_colours(res: dict) -> dict:
    """{label: colour} for the run's horizons, in model output order (v2.07 predicts the 1-day horizon only)."""
    hz = res["config"].get("protocol", {}).get("horizons", [1, 5, 10])
    return {f"{h} d": HORIZON[f"{h} d"] for h in hz}


def _horizon_lines(ax, ep, h, key, hc):
    for j, (lab, col) in enumerate(hc.items()):
        pts = [(x, e[key][j]) for x, e in zip(ep, h) if e.get(key) is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=col, linewidth=2, label=lab)
            _end_label(ax, xs[-1], ys[-1], lab, col)


def plot(run_path: Path, out_path: Path | None = None) -> Path:
    res = json.loads(run_path.read_text(encoding="utf-8"))
    runs = res["runs"]
    if not runs:
        raise SystemExit(f"{run_path} has no finished seeds yet")
    has_ce = "es_ce_per_horizon" in runs[0]["history"][0]
    has_grad = "grad_shared" in runs[0]["history"][0]
    has_price = any(e.get("es_return_rmse_std") is not None for e in runs[0]["history"])      # false without a price head
    panels = (["loss", "f1"] + (["ce"] if has_ce else []) + (["rmse"] if has_price else []) + (["agree"] if has_ce else [])
              + (["grad"] if has_grad else []))
    fig, axes = plt.subplots(len(runs), len(panels), figsize=(5 * len(panels), 3.6 * len(runs)), squeeze=False,
                             facecolor=SURFACE)
    name = res["config"]["name"]
    hc = horizon_colours(res)
    K = len(class_colours(res))                                    # chance macro-F1 = 1 / K is not exact for K = 2,
    chance, uniform = 1 / K if K == 3 else 0.5, math.log(K)        # but a random guess at the class shares scores 0.5
    for r, run in enumerate(runs):
        h = run["history"]
        ep = [e["epoch"] for e in h]
        best = run["best_epoch"]
        ce_min = run.get("ce_min_epoch")
        ax = dict(zip(panels, axes[r]))
        a0 = ax["loss"]
        ce = [e.get("train_ce") for e in h]
        pr = [e.get("train_price") for e in h]
        a0.plot(ep, ce, color=INK, linewidth=2, label="trend: cross-entropy")
        if all(v is not None for v in pr):
            a0.plot(ep, pr, color=INK2, linewidth=2, linestyle="--", label="price: MSE")
            _end_label(a0, ep[-1], pr[-1], "price", INK2)
        _end_label(a0, ep[-1], ce[-1], "trend", INK)
        kept_note = f"kept epoch {best}" + (f" · lowest CE epoch {ce_min}" if ce_min else "")
        _style(a0, f"seed {run['seed']} · {kept_note}\ntraining loss by task", "loss")
        _horizon_lines(ax["f1"], ep, h, "es_per_horizon", hc)
        ax["f1"].axhline(chance, color=INK2, linewidth=1, linestyle=":")
        _style(ax["f1"], "inner hold-out macro-F1 (selection signal)", "macro-F1")
        if has_ce:
            _horizon_lines(ax["ce"], ep, h, "es_ce_per_horizon", hc)
            ax["ce"].axhline(uniform, color=INK2, linewidth=1, linestyle=":")
            _style(ax["ce"], f"inner hold-out cross-entropy (uniform guess = ln {K})", "cross-entropy")
            _horizon_lines(ax["agree"], ep, h, "es_call_agreement", hc)
            _style(ax["agree"], "hold-out calls unchanged from previous epoch", "share")
        if has_grad:
            g0 = run.get("grad_shared_before_training")
            gx = ([0] if g0 else []) + ep
            share = ([g0["price_share_weighted"]] if g0 else []) + [e["grad_shared"]["price_share_weighted"] for e in h]
            cos = ([g0["cosine"]] if g0 else []) + [e["grad_shared"]["cosine"] for e in h]
            ag = ax["grad"]
            ag.plot(gx, share, color=INK, linewidth=2)
            ag.plot(gx, cos, color=INK2, linewidth=2, linestyle="--")
            _end_label(ag, gx[-1], share[-1], "price share", INK)
            _end_label(ag, gx[-1], cos[-1], "cosine", INK2)
            ag.axhline(0, color=INK2, linewidth=1, linestyle=":")
            _style(ag, "shared layers: price share of the pull; cosine of the two tasks' gradients", "share / cosine")
        if has_price:
            _horizon_lines(ax["rmse"], ep, h, "es_return_rmse_std", hc)
            _style(ax["rmse"], "inner hold-out price RMSE (train-standardised)", "RMSE")
        for a in ax.values():
            a.axvline(best, color=INK2, linewidth=1, linestyle="-", alpha=0.6)
            if ce_min and ce_min != best:
                a.axvline(ce_min, color=INK2, linewidth=1, linestyle="--", alpha=0.6)
            a.margins(x=0.08)
    fig.suptitle(f"{name} — learning curves (validation block not used during training)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, linewidth=2, label=l) for l, c in hc.items()]
    handles += [Line2D([], [], color=INK2, linewidth=1, linestyle=":",
                       label=f"chance ({'1/3' if K == 3 else '0.5'}) / uniform guess (ln {K})"),
                Line2D([], [], color=INK, linewidth=2, label="trend loss (cross-entropy)"),
                Line2D([], [], color=INK2, linewidth=2, linestyle="--", label="price loss (MSE)"),
                Line2D([], [], color=INK2, linewidth=1, alpha=0.6, label="epoch whose weights were kept")]
    if any(run.get("ce_min_epoch") for run in runs):
        handles.append(Line2D([], [], color=INK2, linewidth=1, linestyle="--", alpha=0.6, label="lowest hold-out CE epoch"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out_path = out_path or run_path.with_name(run_path.stem + "_training.png")
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_calls(run_path: Path, out_path: Path | None = None) -> Path | None:
    """Predicted-class shares on the inner hold-out per epoch (rows = seeds, columns = horizons); None for runs
    before v2.02, which did not log them."""
    res = json.loads(run_path.read_text(encoding="utf-8"))
    runs = res["runs"]
    if not runs or "es_call_shares" not in runs[0]["history"][0]:
        return None
    hc = horizon_colours(res)
    fig, axes = plt.subplots(len(runs), len(hc), figsize=(5 * len(hc), 3.4 * len(runs)), squeeze=False, facecolor=SURFACE)
    for r, run in enumerate(runs):
        h = run["history"]
        ep = [e["epoch"] for e in h]
        actual = run.get("es_actual_shares")
        for j, hz in enumerate(hc):
            a = axes[r][j]
            for c, (cls, col) in enumerate(class_colours(res).items()):
                y = [e["es_call_shares"][j][c] for e in h]
                a.plot(ep, y, color=col, linewidth=2)
                _end_label(a, ep[-1], y[-1], cls, col)
                if actual:
                    a.axhline(actual[j][c], color=col, linewidth=1, linestyle=":")
            a.axvline(run["best_epoch"], color=INK2, linewidth=1, alpha=0.6)
            a.set_ylim(0, 1)
            a.margins(x=0.08)
            _style(a, f"seed {run['seed']} · {hz}: share of hold-out calls", "share of calls")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, linewidth=2, label=f"called {k.replace('_', ' ')}") for k, c in class_colours(res).items()]
    handles += [Line2D([], [], color=INK2, linewidth=1, linestyle=":", label="true share on the hold-out"),
                Line2D([], [], color=INK2, linewidth=1, alpha=0.6, label="epoch whose weights were kept")]
    fig.suptitle(f"{res['config']['name']} — what the model calls on the inner hold-out", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out_path = out_path or run_path.with_name(run_path.stem + "_calls.png")
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print("wrote", plot(Path(a.run_json), Path(a.out) if a.out else None))
    c = plot_calls(Path(a.run_json))
    if c:
        print("wrote", c)
