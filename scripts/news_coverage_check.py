"""News coverage check: is there enough Thai news per stock-day for a sentiment feature? Go/no-go.

(Pre-registration item A3a; open-items register Phase 1.7.)

Coverage of a stock = (trading days in its SET50 membership window with >= 1
Thai article) / (trading days in that window). Verdict from the MEDIAN stock:
  >= 30% viable as specified | 10-30% subset only | < 10% reframe

Definitions fixed before running (claude/log-entry-2026-09-30-a3a-coverage.md):
  trading days   dates on which >= 30 union tickers have a Siamchart row
  window         union of the stock's rows in set50_membership.csv
  day of news    hard rule 4: published <= 16:30 ICT on a trading day -> that
                 day, otherwise the next trading day (weekends/holidays roll forward)
  primary        any Thai article tagged to the stock (Kaohoon tag page, from the article-list crawl)
  S1             primary minus sections summary, column
  S2             S1 and the ticker appears in JSON-LD keywords
  S3             primary and the ticker appears in the headline

  python scripts/news_coverage_check.py
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime, time
from pathlib import Path

CLOSE = time(16, 30)
BANDS = ((0.30, "viable as specified"), (0.10, "viable for a subset only"), (0.0, "reframe"))


def trading_calendar(prices: Path, min_symbols: int = 30) -> list[date]:
    n = defaultdict(int)
    with prices.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n[r["date"]] += 1
    return sorted(datetime.strptime(d, "%Y%m%d").date() for d, c in n.items() if c >= min_symbols)


def news_trading_day(published: datetime, cal: list[date]) -> date | None:
    """Hard rule 4. `published` must be timezone-aware in ICT (+07:00)."""
    d = published.date()
    i = bisect.bisect_left(cal, d)
    if i < len(cal) and cal[i] == d and published.time() <= CLOSE:
        return d
    j = i + 1 if (i < len(cal) and cal[i] == d) else i
    return cal[j] if j < len(cal) else None


def band(x: float) -> str:
    return next(label for cut, label in BANDS if x >= cut)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--articles", default="data/raw/kaohoon_articles/articles.jsonl")
    ap.add_argument("--membership", default="data/set50_membership.csv")
    ap.add_argument("--prices", default="data/raw/prices/siamchart/set50_union_2019_2025.csv")
    ap.add_argument("--out", default="data/derived/news_coverage.csv")
    args = ap.parse_args()

    cal = [d for d in trading_calendar(Path(args.prices)) if date(2020, 1, 1) <= d <= date(2025, 12, 31)]
    windows: dict[str, set] = defaultdict(set)
    with open(args.membership, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a, b = date.fromisoformat(r["effective_from"]), date.fromisoformat(r["effective_to"])
            windows[r["symbol"]].update(d for d in cal if a <= d <= b)

    days = {k: defaultdict(set) for k in ("primary", "S1", "S2", "S3")}
    n_articles = 0
    with open(args.articles, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            if a["language"] != "th" or not a["date_published"]:
                continue
            td = news_trading_day(datetime.fromisoformat(a["date_published"]), cal)
            if td is None:
                continue
            n_articles += 1
            kw = {k.strip().upper() for k in (a["keywords"] or "").split(",")}
            head = (a["headline"] or "").upper()
            s1 = a["url_section"] not in ("summary", "column")
            for sym in a["symbols_tagged"]:
                days["primary"][sym].add(td)
                if s1:
                    days["S1"][sym].add(td)
                    if sym in kw:
                        days["S2"][sym].add(td)
                if sym in head:
                    days["S3"][sym].add(td)

    rows = []
    for sym in sorted(windows):
        w = windows[sym]
        row = {"symbol": sym, "window_days": len(w)}
        for k in days:
            row[k] = round(len(days[k][sym] & w) / len(w), 4) if w else None
        rows.append(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    print(f"trading days 2020-2025: {len(cal)}   Thai articles mapped: {n_articles}   stocks: {len(rows)}")
    summary = {}
    for k in days:
        v = sorted(r[k] for r in rows if r[k] is not None)
        q = statistics.quantiles(v, n=4)
        summary[k] = {"median": statistics.median(v), "q1": q[0], "q3": q[2], "min": v[0], "max": v[-1],
                      ">=30%": sum(x >= .30 for x in v), "10-30%": sum(.10 <= x < .30 for x in v),
                      "<10%": sum(x < .10 for x in v), "band": band(statistics.median(v))}
        s = summary[k]
        print(f"{k:8} median {s['median']:.1%}  IQR {s['q1']:.1%}-{s['q3']:.1%}  range {s['min']:.1%}-{s['max']:.1%}  "
              f"stocks >=30%/10-30%/<10%: {s['>=30%']}/{s['10-30%']}/{s['<10%']}  -> {s['band'].upper()}")
    print(f"per-stock table: {args.out}")
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
