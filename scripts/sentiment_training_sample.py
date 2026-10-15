"""Draw the hand-labelling sample used to TRAIN (fine-tune) WangchanBERTa on finance headlines.

The 250 items of the sentiment accuracy check (data/raw/sentiment_check/) are the exam and must
never be trained on. This sample comes from the same population (Thai (article, ticker) pairs,
16:30 news-day rule, inside SET50 membership, 2020-2025) but excludes every article that appears
in the exam sample, and every exact headline that appears there, so no exam headline can leak in.

Writes, under --out (gitignored: scraped headlines never go into git):
  sample.csv    the pairs to label
  items.json    the same pairs for the labelling site (id, ticker, section, trading_day, headline)
  SAMPLE.json   seed, population, exclusions, sample hash, label protocol (same as the exam)

  python scripts/sentiment_training_sample.py            # n=1000, seed 20261006
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_check_sample import PROTOCOL, build_population  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--articles", default="data/raw/kaohoon_articles/articles.jsonl")
    ap.add_argument("--membership", default="data/set50_membership.csv")
    ap.add_argument("--prices", default="data/raw/prices/siamchart/set50_union_2019_2025.csv")
    ap.add_argument("--exam", default="data/raw/sentiment_check/sample.csv")
    ap.add_argument("--out", default="data/raw/sentiment_training")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()
    out = Path(args.out)
    if (out / "sample.csv").exists():
        print(f"{out / 'sample.csv'} already exists; the sample is drawn once.")
        return 1

    with open(args.exam, encoding="utf-8-sig", newline="") as f:
        exam = list(csv.DictReader(f))
    exam_urls = {r["url"] for r in exam}
    exam_heads = {r["headline"].strip() for r in exam}

    pop = build_population(args)
    eligible = [p for p in pop if p["url"] not in exam_urls and p["headline"].strip() not in exam_heads]
    rng = random.Random(args.seed)
    sample = rng.sample(eligible, args.n)
    for k, s in enumerate(sample, 1):
        s["id"] = f"train-{k:04d}"
    assert not ({s["url"] for s in sample} & exam_urls) and not ({s["headline"].strip() for s in sample} & exam_heads)

    out.mkdir(parents=True, exist_ok=True)
    fields = ["id", "ticker", "url", "section", "published", "trading_day", "headline"]
    with (out / "sample.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: s[k] for k in fields} for s in sample)
    digest = hashlib.sha256((out / "sample.csv").read_bytes()).hexdigest()
    items = [{k: s[k] for k in ("id", "ticker", "section", "trading_day", "headline")} for s in sample]
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    meta = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "purpose": "training labels for fine-tuning",
            "seed": args.seed, "n": args.n, "population_pairs": len(pop),
            "excluded_pairs_overlapping_exam": len(pop) - len(eligible), "eligible_pairs": len(eligible),
            "exam_sample_sha256": hashlib.sha256(Path(args.exam).read_bytes()).hexdigest(),
            "sample_sha256": digest, "protocol": PROTOCOL,
            "sample_sections": dict(Counter(s["section"] for s in sample).most_common()),
            "sample_years": dict(sorted(Counter(s["trading_day"][:4] for s in sample).items())),
            "sample_distinct_tickers": len({s["ticker"] for s in sample}),
            "sample_distinct_articles": len({s["url"] for s in sample})}
    (out / "SAMPLE.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "protocol"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
