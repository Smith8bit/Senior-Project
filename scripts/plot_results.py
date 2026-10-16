"""Figures for the Phase 4 batches (Colab or PC). Reads the JSON files the batches write; writes PNGs.

  python scripts/plot_results.py walkforward  --vs results/batch3/e22_vs_e20_e05.json --out fig_walkforward.png
  python scripts/plot_results.py official     --summary results/batch3/batch3_stacking.json results/batch3/batch3_summary.json \
                                              --highlight E21 --out fig_official.png
  python scripts/plot_results.py importance   --deep results/importance/importance_e21.json \
                                              --xgb results/importance/importance_xgb.json --out-prefix fig_importance

Colours: fixed categorical order from the project's chart palette (blue, orange, aqua, yellow); reference lines in
neutral grey; text in ink colours, never series colours. Validation / walk-forward data only; the test block is sealed.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
import matplotlib.ticker  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4, fixed order
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
HS = ("h1", "h5", "h10")
HNAME = {"h1": "1 day", "h5": "5 days", "h10": "10 days"}
PASS_BAR = {"h1": 0.393, "h5": 0.400, "h10": 0.407}
GROUP_LABEL = "sector_label"


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(ncols, w=4.2, h=3.6):
    fig, axes = plt.subplots(1, ncols, figsize=(w * ncols, h), sharey=True, facecolor=SURFACE)
    return fig, np.atleast_1d(axes)


def walkforward(vs_path, out):
    rows = json.loads(Path(vs_path).read_text(encoding="utf-8"))["rows"]
    windows = list(dict.fromkeys(r["window"] for r in rows))
    series = [("E22: current parent (E21)", "e22"), ("E20: RevIN off (batch 2)", "e20"), ("E05: E03, RevIN on", "e05"),
              ("XGBoost", "xgb")]
    fig, axes = _fig(3)
    x = np.arange(len(windows))
    for ax, h in zip(axes, HS):
        rr = {r["window"]: r for r in rows if r["h"] == h}
        _style(ax, HNAME[h])
        if "official" in windows:
            ax.axvspan(windows.index("official") - 0.4, windows.index("official") + 0.4, color=GRID, alpha=0.6, linewidth=0)
        ax.plot(x, [rr[w]["floor"] for w in windows], color=MUTED, linewidth=1.5, linestyle="--", label="Best one-rule floor")
        for (lab, key), col in zip(series, SERIES):
            ys = [rr[w].get(key) for w in windows]
            ax.plot(x, ys, color=col, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=lab, zorder=3 if key == "e22" else 2)
        ax.set_xticks(x, [w.replace("official", "official\n(validation)") for w in windows])
    axes[0].set_ylabel("macro-F1", color=INK2)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=5, frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Walk-forward check: each window scored by a model trained only on earlier dates", color=INK, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def official(summary_paths, out, highlight=None):
    rows, seen = [], set()
    for p in summary_paths:
        for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]:
            if r["label"] not in seen:
                seen.add(r["label"]); rows.append(r)
    labels = [r["label"] for r in rows]
    fig, axes = _fig(3, w=4.0)
    x = np.arange(len(rows))
    for ax, h in zip(axes, HS):
        _style(ax, HNAME[h])
        m = [r["h"][h]["mean"] for r in rows]
        s = [r["h"][h]["std"] or 0 for r in rows]
        cols = [SERIES[0] if lab == highlight else "#8a8984" for lab in labels]
        # dot + error bar (the axis does not start at 0, so bars would exaggerate differences)
        for xi, mi, si, ci in zip(x, m, s, cols):
            ax.errorbar(xi, mi, yerr=si, fmt="o", color=ci, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        ecolor=ci, elinewidth=1.5, capsize=3, zorder=3)
            ax.text(xi + 0.12, mi, f"{mi:.3f}", ha="left", va="center", fontsize=8, color=INK2)
        ax.axhline(PASS_BAR[h], color=INK, linewidth=1.2, linestyle="--")
        ax.text(-0.4, PASS_BAR[h] + 0.002, f"pass bar {PASS_BAR[h]:.3f}", color=INK2, fontsize=8, ha="left", va="bottom")
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylim(0.33, 0.415); ax.set_xlim(-0.5, len(rows) - 0.3)
    axes[0].set_ylabel("validation macro-F1 (mean ± std, 3 seeds)", color=INK2)
    fig.suptitle("Official validation window, per-horizon best epoch" + (f" — {highlight} highlighted" if highlight else ""),
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def _collect(entries):
    """entries: list of importance dicts (one per seed). Returns {name: {h_index: [drops...]}} over seeds x repeats."""
    out = {"single": {}, "groups": {}}
    for e in entries:
        for kind in ("single", "groups"):
            for name, reps in e[kind].items():
                d = out[kind].setdefault(name, {})
                for rep in reps:
                    for j, v in enumerate(rep):
                        d.setdefault(j, []).append(v)
    return out


def load_importance(deep_path=None, xgb_path=None):
    res = {}
    if deep_path:
        d = json.loads(Path(deep_path).read_text(encoding="utf-8"))
        runs = next(iter(d["models"].values()))["runs"]
        res["deep"] = _collect([r["importance"] for r in runs])
    if xgb_path:
        x = json.loads(Path(xgb_path).read_text(encoding="utf-8"))["importance"]
        per_h = {h: _collect(x[h]) for h in HS if h in x}       # one classifier per horizon: horizon index 0 inside
        merged = {"single": {}, "groups": {}}
        for j, h in enumerate(HS):
            for kind in ("single", "groups"):
                for name, d in per_h[h][kind].items():
                    merged[kind].setdefault(name, {})[j] = d[0]
        res["xgb"] = merged
    return res


def importance(deep_path, xgb_path, prefix, top=12):
    imp = load_importance(deep_path, xgb_path)
    outs = []
    for key, col, title in (("deep", SERIES[0], "Main model (E21)"), ("xgb", SERIES[1], "XGBoost (29 price inputs)")):
        if key not in imp:
            continue
        single = imp[key]["single"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), facecolor=SURFACE)
        for j, (ax, h) in enumerate(zip(axes, HS)):
            stats = sorted(((n, statistics.mean(d[j]), statistics.pstdev(d[j])) for n, d in single.items()),
                           key=lambda t: t[1], reverse=True)[:top][::-1]
            _style(ax, HNAME[h])
            ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, linewidth=0.8)
            y = np.arange(len(stats))
            ax.barh(y, [s[1] for s in stats], xerr=[s[2] for s in stats], color=col, height=0.6, edgecolor=SURFACE,
                    linewidth=2, ecolor=INK2, capsize=2)
            ax.set_yticks(y, [s[0] for s in stats], fontsize=8)
            ax.axvline(0, color=INK2, linewidth=0.8)
            ax.set_xlabel("drop in macro-F1 when shuffled", color=INK2, fontsize=9)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.3f}"))
        fig.suptitle(f"{title}: top {top} inputs by permutation importance (mean ± sd over seeds × shuffles)",
                     color=INK, fontsize=12, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        p = f"{prefix}_{key}.png"
        fig.savefig(p, dpi=150, facecolor=SURFACE); plt.close(fig); outs.append(p)
    # groups, both models side by side
    groups = ["price_technical", "e03_relative", "news", GROUP_LABEL]
    fig, axes = _fig(3, w=4.2, h=3.4)
    for j, (ax, h) in enumerate(zip(axes, HS)):
        _style(ax, HNAME[h])
        x = np.arange(len(groups))
        for k, (key, col, lab) in enumerate((("deep", SERIES[0], "Main model (E21)"), ("xgb", SERIES[1], "XGBoost"))):
            if key not in imp:
                continue
            src = {**imp[key]["groups"], **{GROUP_LABEL: imp[key]["single"].get(GROUP_LABEL, {})}}
            vals = [statistics.mean(src[g][j]) if g in src and j in src[g] else np.nan for g in groups]
            ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=col, edgecolor=SURFACE, linewidth=2, label=lab)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(x, ["price &\ntechnical (18)", "relative\n(E03, 11)", "news (3)", "sector\nlabel"], fontsize=8)
    axes[0].set_ylabel("drop in macro-F1 when shuffled", color=INK2)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Input groups shuffled together (blank = input not used by that model)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    p = f"{prefix}_groups.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE); plt.close(fig); outs.append(p)
    return outs


def importance_table(deep_path=None, xgb_path=None):
    """Rows: input, model, horizon, mean drop, sd, n. For printing or saving as CSV."""
    imp = load_importance(deep_path, xgb_path)
    rows = []
    for key, d in imp.items():
        for kind in ("single", "groups"):
            for name, per_h in d[kind].items():
                for j, vals in per_h.items():
                    rows.append({"model": key, "kind": kind, "input": name, "horizon": HS[j],
                                 "mean_drop": round(statistics.mean(vals), 5), "sd": round(statistics.pstdev(vals), 5), "n": len(vals)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("walkforward"); a.add_argument("--vs", required=True); a.add_argument("--out", required=True)
    b = sub.add_parser("official"); b.add_argument("--summary", nargs="+", required=True); b.add_argument("--highlight")
    b.add_argument("--out", required=True)
    c = sub.add_parser("importance"); c.add_argument("--deep"); c.add_argument("--xgb"); c.add_argument("--out-prefix", required=True)
    c.add_argument("--csv")
    args = ap.parse_args(argv)
    if args.cmd == "walkforward":
        print(walkforward(args.vs, args.out))
    elif args.cmd == "official":
        print(official(args.summary, args.out, args.highlight))
    else:
        print(importance(args.deep, args.xgb, args.out_prefix))
        if args.csv:
            import csv
            rows = importance_table(args.deep, args.xgb)
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            print(args.csv)


if __name__ == "__main__":
    main()
