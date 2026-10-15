"""Figure: Siwat's blind labels vs the off-the-shelf WangchanBERTa sentiment model (sentiment accuracy check).

Reads the frozen labels and the saved model predictions; draws no new model output.
  python scripts/plot_sentiment_check.py   ->  reports/figures/sentiment_check_you_vs_wangchanberta.png
"""
import csv
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/raw/sentiment_check/labels.csv"
PREDS = ROOT / "data/derived/sentiment_check_model_predictions.csv"
OUT = ROOT / "reports/figures/sentiment_check_you_vs_wangchanberta.png"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
HUMAN, MODEL = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated: CVD dE 24.7, contrast >= 3:1)
CLASSES = ["pos", "neu", "neg"]
NAMES = {"pos": "Good news", "neu": "Neutral", "neg": "Bad news"}


def read(p, enc):
    with open(p, encoding=enc, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def main():
    lab, pr = read(LABELS, "utf-8-sig"), read(PREDS, "utf-8")
    ids = sorted(lab)
    human = [lab[i]["sentiment"] for i in ids]
    model = [pr[i]["pred3"] for i in ids]
    lean = {i: math.log10(float(pr[i]["p_pos"]) / float(pr[i]["p_neg"])) for i in ids}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor(SURFACE)
    for a in (a1, a2):
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    # Panel 1: label counts, you vs the model
    x = range(len(CLASSES))
    w = 0.38
    hc = [human.count(c) for c in CLASSES]
    mc = [model.count(c) for c in CLASSES]
    b1 = a1.bar([i - w / 2 - 0.01 for i in x], hc, w, color=HUMAN, label="You (blind labels)", zorder=3)
    b2 = a1.bar([i + w / 2 + 0.01 for i in x], mc, w, color=MODEL, label="WangchanBERTa (off-the-shelf)", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            a1.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, f"{int(b.get_height())}",
                    ha="center", va="bottom", color=INK, fontsize=10)
    a1.set_xticks(list(x), [NAMES[c] for c in CLASSES])
    a1.set_ylabel("Headlines (of 250)")
    a1.set_ylim(0, 290)
    a1.set_yticks([0, 50, 100, 150, 200, 250])
    a1.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    a1.tick_params(length=0)
    a1.legend(frameon=False, loc="upper left", ncol=2, fontsize=9.5, labelcolor=INK2)
    a1.set_title("1. The model calls almost everything neutral\nAgreement with you: kappa = -0.005 (chance level)",
                 loc="left", color=INK, fontsize=11.5, pad=10)

    # Panel 2: the model's hidden lean, grouped by your label
    rng = random.Random(20260930)
    for row, c in enumerate(CLASSES):
        v = [lean[i] for i in ids if lab[i]["sentiment"] == c]
        y = [row + rng.uniform(-0.27, 0.27) for _ in v]
        a2.scatter(v, y, s=26, color=HUMAN, alpha=0.55, edgecolors=SURFACE, linewidths=0.6, zorder=3)
        share = sum(t > 0 for t in v) / len(v)
        a2.text(1.02, row, f"{share:.0%} lean good\n(n = {len(v)})", transform=a2.get_yaxis_transform(),
                va="center", ha="left", color=INK2, fontsize=9.5)
    a2.axvline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
    a2.set_yticks(range(3), [f"You said:\n{NAMES[c]}" for c in CLASSES])
    a2.set_ylim(2.6, -0.6)
    ticks = [-2, -1, 0, 1, 2]
    a2.set_xticks(ticks, ["100x\nbad", "10x\nbad", "even", "10x\ngood", "100x\ngood"])
    a2.set_xlim(-2.6, 2.6)
    a2.set_xlabel("Model's hidden lean: how many times more 'good' than 'bad' it scores the headline")
    a2.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    a2.tick_params(length=0)
    a2.set_title("2. ...but underneath, it leans the right way\nYour good news sits right of 'even', your bad news left",
                 loc="left", color=INK, fontsize=11.5, pad=10)

    fig.text(0.01, 0.01, "Sentiment accuracy check, 250 Kaohoon headlines (2020-2025), labels frozen before any model output. "
             "Dots jittered vertically.\n"
             "Model: airesearch/wangchanberta-base-att-spm-uncased, revision finetuned@wisesight_sentiment, headline only.\n"
             "Lean = log10(p_pos / p_neg). Good-vs-bad separation (AUC): 0.95 on this lean, 0.93 on p_pos - p_neg. "
             "Exploratory, not pre-registered.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 0.93, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor=SURFACE)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
