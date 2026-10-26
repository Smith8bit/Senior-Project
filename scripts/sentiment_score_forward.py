"""Score the 2026 (forward) Kaohoon headlines ONCE with the same fine-tuned WangchanBERTa as data/derived/sentiment_scores.csv.

Same rules as scripts/sentiment_score_corpus.py (whose functions are reused unchanged):
  * Thai only, parsed ok, with a publication time; sections `column` and `summary` excluded;
  * the selected configuration (wisesight+ticker_section_headline), its 3 saved seeds, the same input text, the same
    batch prediction on the CPU; s = p_pos - p_neg averaged over the seeds; `restatement` flag as before;
  * hard rule 9: written once, never overwritten.
What is different, and why:
  * Articles: the forward crawl (data/raw/kaohoon_articles_fwd2026/articles.jsonl), not the frozen passB_v1 crawl.
  * Population: only articles that the frozen file could NOT hold, i.e. whose trading day (hard rule 4, 16:30 ICT) falls
    after the frozen calendar's last day (2025-12-30). So the two files never overlap and nothing is missing between them:
    an article published on 2025-12-30 after 16:30, or later, is here; anything earlier is in the frozen file.
  * Tickers: the 70-ticker union plus the 2026 SET50 members (MRDIYT, TFG, THAI are new).
  * trading_day: the true 2026 trading calendar is not on the PC (2026 prices are Yahoo data on Drive), so trading_day
    here uses Monday-Friday. The forward test recomputes it from `published` with the real calendar before use
    (scripts/forward_test_v2xx.py --news-fwd); a holiday date here always rolls forward to the same row either way.

  python scripts/sentiment_score_forward.py                  # -> data/derived/sentiment_scores_fwd2026.csv (+ .json)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sentiment_check_score as S  # noqa: E402
import sentiment_score_corpus as SC  # noqa: E402
from news_coverage_check import news_trading_day, trading_calendar  # noqa: E402

SET50_2026 = sorted(set(
    "ADVANC AOT AWC BANPU BBL BDMS BEM BH BJC BTS CBG CCET CENTEL COM7 CPALL CPF CPN CRC DELTA EGCO GPSC GULF HMPRO IVL "
    "KBANK KKP KTB KTC LH MINT MTC OR OSP PTT PTTEP PTTGC RATCH SAWAD SCB SCC SCGP TCAP TIDLOR TISCO TLI TOP TRUE TTB TU "
    "WHA BCP MRDIYT TFG THAI".split()))                    # H1 + H2 2026 lists (SET PDFs, see forward_test_v2.py)
LAST_FROZEN_DAY = date(2025, 12, 30)                         # last trading day of data/derived/sentiment_scores.csv


def weekday_calendar(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def population(articles: Path, prices: Path, membership: Path, end: date) -> tuple[list[dict], dict]:
    frozen_cal = [d for d in trading_calendar(prices) if date(2019, 1, 1) <= d <= date(2025, 12, 31)]
    assert frozen_cal[-1] == LAST_FROZEN_DAY, f"frozen calendar ends {frozen_cal[-1]}, expected {LAST_FROZEN_DAY}"
    fwd_cal = weekday_calendar(LAST_FROZEN_DAY + timedelta(days=1), end)
    tickers = {r["symbol"] for r in csv.DictReader(open(membership, encoding="utf-8"))} | set(SET50_2026)
    pop, why = [], Counter()
    with open(articles, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            if a.get("parse_status") != "ok" or a.get("language") != "th" or not a.get("date_published") or not a.get("headline"):
                why["not parsed / not Thai / no time or headline"] += 1; continue
            if a.get("url_section") in SC.EXCLUDED_SECTIONS:
                why["section column or summary"] += 1; continue
            pub = datetime.fromisoformat(a["date_published"])
            if news_trading_day(pub, frozen_cal) is not None:      # belongs to the frozen file's period
                why["trading day on or before 2025-12-30 (frozen file's period)"] += 1; continue
            td = news_trading_day(pub, fwd_cal)
            if td is None:
                why["after the end date"] += 1; continue
            n0 = len(pop)
            for sym in a.get("symbols_tagged") or []:
                if sym in tickers:
                    pop.append({"url": a["url"], "ticker": sym, "section": a["url_section"], "headline": a["headline"],
                                "published": a["date_published"], "trading_day": td.isoformat(),
                                "restatement": SC.is_restatement(a["url_section"], a["headline"])})
            why["kept" if len(pop) > n0 else "no tagged ticker in the universe"] += 1
    pop.sort(key=lambda p: (p["url"], p["ticker"]))
    return pop, dict(why)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--articles", default="data/raw/kaohoon_articles_fwd2026/articles.jsonl")
    ap.add_argument("--prices", default="data/raw/prices/siamchart/set50_union_2019_2025.csv")
    ap.add_argument("--membership", default="data/set50_membership.csv")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--out", default="data/derived/sentiment_scores_fwd2026.csv")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true", help="count the population only; no model, nothing written")
    args = ap.parse_args(argv)
    root = args.root
    out = root / args.out
    if out.exists() and not args.dry_run:
        sys.exit(f"{out} already exists: scores are computed once and cached (hard rule 9).")
    pop, why = population(root / args.articles, root / args.prices, root / args.membership, date.fromisoformat(args.end))
    by_month = Counter(p["trading_day"][:7] for p in pop)
    by_ticker = Counter(p["ticker"] for p in pop)
    print(json.dumps({"pairs": len(pop), "articles": why, "pairs_by_month": dict(sorted(by_month.items())),
                      "tickers": len(by_ticker), "2026_members_without_news": sorted(set(SET50_2026) - set(by_ticker))},
                     ensure_ascii=False, indent=1), flush=True)
    if args.dry_run:
        return 0
    import sentiment_finetune as FT
    import torch
    T = FT.TrainPaths(root)
    sel_path = T.out / "selection_sentiment.json"
    sel = S.read_json(sel_path)
    cfg = sel["chosen"]
    inp = cfg.split("+")[1]
    runs = sorted([r for r in FT.read_runs(T) if r["target"] == "sentiment" and r["config"] == cfg], key=lambda r: r["seed"])
    dirs = [Path(r["model_dir"]) for r in runs]
    dirs = [d if d.is_absolute() else root / d for d in dirs]
    frozen_meta = json.loads((root / "data/derived/sentiment_scores.json").read_text(encoding="utf-8"))
    if cfg != frozen_meta["config"] or [str(d) for d in dirs] != frozen_meta["model_dirs"]:
        sys.exit(f"stop: the selected models differ from the ones that scored sentiment_scores.csv\n  now: {cfg} {dirs}\n"
                 f"  then: {frozen_meta['config']} {frozen_meta['model_dirs']}")
    texts = [FT.text_of(p, inp) for p in pop]
    uniq = sorted(set(texts), key=len)
    print(f"{len(pop)} (article, ticker) pairs, {len(uniq)} distinct inputs, config {cfg}", flush=True)
    tok = FT.load_tokenizer()
    per_seed = []
    for d, r in zip(dirs, runs):
        t0 = time.time()
        model, idx = SC.load_saved(d)
        probs = FT.predict(model, idx, tok, uniq, torch.device("cpu"), batch=args.batch)
        per_seed.append(dict(zip(uniq, probs)))
        print(f"  seed {r['seed']}: scored {len(uniq)} inputs in {time.time() - t0:.0f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["url", "ticker", "section", "published", "trading_day", "restatement", "p_pos", "p_neu", "p_neg", "s"] + \
             [f"s_seed{r['seed']}" for r in runs]                        # same columns as sentiment_scores.csv
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
            "articles_file": args.articles, "articles_sha256": S.sha256(root / args.articles),
            "population_rule": "trading day (16:30 ICT rule) after 2025-12-30; trading_day column uses Mon-Fri, recomputed "
                               "with the real 2026 calendar by the forward test",
            "pairs": len(pop), "distinct_inputs": len(uniq), "articles": why, "pairs_by_month": dict(sorted(by_month.items())),
            "excluded_sections": sorted(SC.EXCLUDED_SECTIONS), "restatement_pairs": sum(p["restatement"] for p in pop),
            "output_sha256": S.sha256(out), "git_commit": FT.git_commit(root),
            "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__}}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
