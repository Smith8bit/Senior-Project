"""Score every relevant Kaohoon headline ONCE with the selected fine-tuned WangchanBERTa and cache it (hard rule 9).

Run after `sentiment_finetune.py select` (needs the models saved by `train --save-models`):
  python scripts/sentiment_score_corpus.py            # -> data/derived/sentiment_scores.csv (+ .json)

Units: (article, ticker) pairs from the article-page crawl (passB_v1 contents in data/raw/kaohoon_articles):
  * Thai only, parsed ok, with a publication time;
  * news trading day by hard rule 4 (16:30 ICT cutoff), 2019-2025 calendar;
  * ticker in the 70-ticker union (all days, not only member days: sequence windows look back 60 days);
  * sections `column` and `summary` excluded (Phase 1.21 decision, 2026-10-15).
Score per pair: s = p_pos - p_neg, averaged over the selected configuration's 3 seeds
(pre-registered in claude/pre-registration-amendment-2026-10-15-news-beyond-price.md).
`restatement` flags the price-restatement control set: section breakingnews, or a percentage with a move verb.
No headline text is written to the output (scraped text stays out of git); join on url if needed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sentiment_check_score as S  # noqa: E402
import sentiment_finetune as FT  # noqa: E402
from news_coverage_check import news_trading_day, trading_calendar  # noqa: E402

EXCLUDED_SECTIONS = {"column", "summary"}
MOVE_VERBS = ["วิ่ง", "บวก", "พุ่ง", "ดีด", "เด้ง", "ทะยาน", "ร่วง", "รูด", "ดิ่ง", "ทรุด", "ลบ", "ดิ่งหนัก"]
PCT = re.compile(r"\d+(\.\d+)?\s*%")


def is_restatement(section: str, headline: str) -> bool:
    return section == "breakingnews" or (bool(PCT.search(headline)) and any(v in headline for v in MOVE_VERBS))


def population(articles: Path, prices: Path, membership: Path) -> list[dict]:
    cal = [d for d in trading_calendar(prices) if date(2019, 1, 1) <= d <= date(2025, 12, 31)]
    union = {r["symbol"] for r in csv.DictReader(open(membership, encoding="utf-8"))}
    pop = []
    with open(articles, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            if a.get("parse_status") != "ok" or a.get("language") != "th" or not a.get("date_published") or not a.get("headline"):
                continue
            if a.get("url_section") in EXCLUDED_SECTIONS:
                continue
            td = news_trading_day(datetime.fromisoformat(a["date_published"]), cal)
            if td is None:
                continue
            for sym in a.get("symbols_tagged") or []:
                if sym in union:
                    pop.append({"url": a["url"], "ticker": sym, "section": a["url_section"], "headline": a["headline"],
                                "published": a["date_published"], "trading_day": td.isoformat(),
                                "restatement": is_restatement(a["url_section"], a["headline"])})
    pop.sort(key=lambda p: (p["url"], p["ticker"]))
    return pop


def load_saved(model_dir: Path):
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    names = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    if not {"pos", "neu", "neg"} <= set(names):
        sys.exit(f"{model_dir}: unexpected labels {model.config.id2label}")
    return model, [names[c] for c in ("pos", "neu", "neg")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--articles", default="data/raw/kaohoon_articles/articles.jsonl")
    ap.add_argument("--prices", default="data/raw/prices/siamchart/set50_union_2019_2025.csv")
    ap.add_argument("--membership", default="data/set50_membership.csv")
    ap.add_argument("--out", default="data/derived/sentiment_scores.csv")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)
    root = args.root
    T = FT.TrainPaths(root)
    sel_path = T.out / "selection_sentiment.json"
    if not sel_path.exists():
        sys.exit("no selection yet: run `sentiment_finetune.py select` first.")
    out = root / args.out
    if out.exists():
        sys.exit(f"{out} already exists: scores are computed once and cached (hard rule 9).")
    sel = S.read_json(sel_path)
    cfg = sel["chosen"]
    inp = cfg.split("+")[1]
    runs = [r for r in FT.read_runs(T) if r["target"] == "sentiment" and r["config"] == cfg]
    dirs = [Path(r["model_dir"]) if r.get("model_dir") else None for r in sorted(runs, key=lambda r: r["seed"])]
    if len(dirs) != 3 or any(d is None for d in dirs):
        sys.exit("the selected configuration's 3 runs were not saved with --save-models; retrain them with saving on.")
    dirs = [d if d.is_absolute() else root / d for d in dirs]

    import torch
    pop = population(root / args.articles, root / args.prices, root / args.membership)
    texts = [FT.text_of(p, inp) for p in pop]
    uniq = sorted(set(texts), key=len)                               # score each distinct input text once
    print(f"{len(pop)} (article, ticker) pairs, {len(uniq)} distinct inputs, config {cfg}", flush=True)
    tok = FT.load_tokenizer()
    per_seed = []
    for d, r in zip(dirs, sorted(runs, key=lambda r: r["seed"])):
        t0 = time.time()
        model, idx = load_saved(d)
        probs = FT.predict(model, idx, tok, uniq, torch.device("cpu"), batch=args.batch)
        per_seed.append(dict(zip(uniq, probs)))
        print(f"  seed {r['seed']}: scored {len(uniq)} inputs in {time.time() - t0:.0f}s", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["url", "ticker", "section", "published", "trading_day", "restatement", "p_pos", "p_neu", "p_neg", "s"] + \
             [f"s_seed{r['seed']}" for r in sorted(runs, key=lambda r: r["seed"])]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(fields)
        for p, t in zip(pop, texts):
            ps = [m[t] for m in per_seed]
            mean = [sum(x[i] for x in ps) / len(ps) for i in range(3)]
            seeds_s = [x[0] - x[2] for x in ps]
            w.writerow([p["url"], p["ticker"], p["section"], p["published"], p["trading_day"], int(p["restatement"]),
                        *[round(v, 6) for v in mean], round(sum(seeds_s) / len(seeds_s), 6), *[round(v, 6) for v in seeds_s]])
    meta = {"computed_at": S.now_utc(), "config": cfg, "model_dirs": [str(d) for d in dirs],
            "selection_sha256": S.sha256(sel_path), "labels_sha256": runs[0]["labels_sha256"],
            "pairs": len(pop), "distinct_inputs": len(uniq), "excluded_sections": sorted(EXCLUDED_SECTIONS),
            "restatement_pairs": sum(p["restatement"] for p in pop), "output_sha256": S.sha256(out),
            "git_commit": FT.git_commit(root), "versions": {"torch": torch.__version__,
                                                            "transformers": __import__("transformers").__version__}}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
