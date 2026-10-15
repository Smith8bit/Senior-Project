"""Article-list crawl: crawl Kaohoon tag listing pages, newest to oldest, per symbol.

Stop rule (agreed 2026-09-15): walk /tag/{symbol}/page/{N} for N = 1, 2, ...
and stop after the first page whose LAST item is dated before the floor
(default 2020-01-01). Every item on every page is recorded, including items
newer than 2025; filtering to the study window happens at modelling time.

Date of an item, in order of preference:
  1. listing <time datetime="...">
  2. Thai date text in the listing card, e.g. "14 ก.ย. 2569" (Buddhist year -543)
  3. fallback: open the article and read JSON-LD datePublished
     (only done for the last item on a page, and the page is saved so the article-page crawl
     never re-fetches it)

Safeguards:
  - listing order is checked; out-of-order dates are logged as order_violations
  - a symbol also stops on: empty page, HTTP error, block/CAPTCHA, max pages
  - never re-fetches a stored URL (hard rule 6), except one stored as a 5xx or
    a block: that is an outage, not content, and is retried on the next run
  - scraped_at is UTC (Phase 1.15)
  - one tab, fixed delay >= 3 s, stops the whole run on any block (Phase 1.19)

Usage:
  python scripts/crawl_article_lists.py --symbols BANPU
  python scripts/crawl_article_lists.py --symbols-file symbols.txt      # one ticker per line

  The browser is resolved the same way as the spike: Playwright's bundled
  Chromium when this environment has it downloaded, otherwise an installed
  Edge / Chrome / Brave. Force one with --browser msedge (or --browser-exe
  <path>). The browser used is recorded on every symbols.jsonl row.
Outputs (in --out, default data/raw/kaohoon_article_lists):
  listings.jsonl   one row per (symbol, article)
  symbols.jsonl    one summary row per finished symbol
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

# Shared helpers live in the spike for now; move to a common module before the article-page crawl.
from spike_kaohoon import (ARTICLE_RE, BASE, LISTING_JS, BROWSER_ORDER, Store,
                           ensure_terminated, fetch, is_article, parse_jsonld,
                           pick_browser, replay, utc_now)

THAI_MONTHS = {
    "มค": 1, "กพ": 2, "มีค": 3, "เมย": 4, "พค": 5, "มิย": 6,
    "กค": 7, "สค": 8, "กย": 9, "ตค": 10, "พย": 11, "ธค": 12,
}
THAI_TEXT_DATE = re.compile(r"(?<!\d)(\d{1,2})\s*([\u0E01-\u0E4E]{1,4}\.?[\u0E01-\u0E4E]{0,3}\.?)\s*(\d{2,4})")
NUM_DATE = re.compile(r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})")


def to_ce(year: int) -> int:
    if year < 100:            # two-digit year, assume Buddhist era 25xx
        year += 2500
    return year - 543 if year > 2400 else year


def listing_date(item: dict) -> tuple[date | None, str | None, str | None]:
    """Return (date, raw_string, source) from a listing item, or (None, None, None)."""
    dt = item.get("time_datetime")
    if dt:
        try:
            return datetime.fromisoformat(dt.replace("Z", "+00:00")).date(), dt, "time_datetime"
        except ValueError:
            pass
    text = " ".join(filter(None, [item.get("time_text"), item.get("card_text")]))
    m = THAI_TEXT_DATE.search(text)
    if m:
        month = THAI_MONTHS.get(m.group(2).replace(".", ""))
        if month:
            try:
                return date(to_ce(int(m.group(3))), month, int(m.group(1))), m.group(0), "thai_text"
            except ValueError:
                pass
    m = NUM_DATE.search(text)
    if m:
        try:
            return date(to_ce(int(m.group(3))), int(m.group(2)), int(m.group(1))), m.group(0), "numeric_text"
        except ValueError:
            pass
    return None, None, None


def article_date(page, store: Store, url: str, delay: float):
    """Fallback: JSON-LD datePublished from the article page."""
    try:
        r = fetch(page, store, url, delay)
    except Exception as e:
        return None, None, None  # network error: item stays undated, symbol stops safely
    if r["blocked"]:
        return None, None, r["blocked"]
    replay(page, url, r["html"])
    for node in parse_jsonld(page):
        if is_article(node) and node.get("datePublished"):
            raw = node["datePublished"]
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).date(), raw, None
            except ValueError:
                return None, raw, None
    return None, None, None


def crawl_symbol(page, store: Store, symbol: str, floor: date, delay: float,
                 max_pages: int, listings_fh, base: str = BASE) -> dict:
    slug = symbol.lower()
    seen: set[str] = set()
    summary = {"symbol": symbol, "pages": 0, "items": 0, "earliest_date": None,
               "latest_date": None, "stop_reason": None, "order_violations": 0,
               "article_fallbacks": 0, "short_pages": 0, "started_at": utc_now()}
    prev_date: date | None = None

    for n in range(1, max_pages + 1):
        url = f"{base}/tag/{slug}" if n == 1 else f"{base}/tag/{slug}/page/{n}"
        try:
            r = fetch(page, store, url, delay)
        except Exception as e:  # timeout, DNS, connection reset: leave symbol unfinished, move on
            summary["stop_reason"] = f"ERROR page {n}: {type(e).__name__}: {str(e).splitlines()[0][:150]}"
            break
        if r["blocked"]:
            summary["stop_reason"] = f"BLOCKED page {n}: {r['blocked']}"
            break
        if r["status"] and r["status"] >= 400:
            summary["stop_reason"] = f"HTTP {r['status']} at page {n} (end of listing)"
            break
        replay(page, url, r["html"])
        listing = page.evaluate(LISTING_JS)
        items = [i for i in listing["items"] if i["url"] not in seen]
        if listing["n_cards"] == 0 and n > 0:
            summary["stop_reason"] = f"LAYOUT: no .post-item cards at page {n}; listing markup changed?"
            break
        if not items:
            summary["stop_reason"] = f"no new items at page {n} (end of listing)"
            break
        summary["pages"] = n
        if listing["n_cards"] != 12:
            summary["short_pages"] += 1   # expected 12; only the final page may be short

        last_d = None
        for idx, item in enumerate(items):
            seen.add(item["url"])
            d, raw, src = listing_date(item)
            is_last = idx == len(items) - 1
            if d is None and is_last:
                d, raw, err = article_date(page, store, item["url"], delay)
                src = "jsonld_fallback"
                summary["article_fallbacks"] += 1
                if err:
                    summary["stop_reason"] = f"BLOCKED on article fallback: {err}"
            if d is not None:
                if prev_date is not None and d > prev_date:
                    summary["order_violations"] += 1
                prev_date = d
                summary["earliest_date"] = min(filter(None, [summary["earliest_date"], d.isoformat()]))
                summary["latest_date"] = max(filter(None, [summary["latest_date"], d.isoformat()]))
            if is_last:
                last_d = d
            m = ARTICLE_RE.search(urlparse(item["url"]).path)
            row = {
                "symbol_queried": symbol, "page": n, "position": idx,
                "url": item["url"], "url_section": item.get("section"),
                "article_id": int(m.group(2)) if m else None,
                "cards_on_page": listing["n_cards"],
                "headline": max(item["link_texts"], key=len) if item["link_texts"] else None,
                "listing_date": d.isoformat() if d else None,
                "listing_date_raw": raw, "listing_date_source": src,
                "scraped_at": utc_now(),
            }
            listings_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary["items"] += 1
        listings_fh.flush()

        if summary["stop_reason"]:
            break
        if last_d is None:
            summary["stop_reason"] = f"UNDATED last item at page {n}; stopped to be safe"
            break
        if last_d < floor:
            summary["stop_reason"] = f"reached floor at page {n} (last item {last_d})"
            break
    else:
        summary["stop_reason"] = f"max_pages ({max_pages}) reached"

    summary["floor_reached"] = bool(summary["earliest_date"] and summary["earliest_date"] < floor.isoformat())
    summary["finished_at"] = utc_now()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--symbols", nargs="+")
    g.add_argument("--symbols-file")
    ap.add_argument("--floor", default="2020-01-01")
    ap.add_argument("--delay", type=float, default=4.0)
    ap.add_argument("--max-pages", type=int, default=2000)
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--browser", default="auto", choices=("auto", *BROWSER_ORDER),
                    help="which browser to drive (default: first one installed)")
    ap.add_argument("--browser-exe", help="full path to a Chromium-based browser, overrides --browser")
    ap.add_argument("--out", default="data/raw/kaohoon_article_lists")
    ap.add_argument("--base", default=BASE, help=argparse.SUPPRESS)  # for offline tests
    args = ap.parse_args()
    if args.delay < 3:
        ap.error("--delay must be at least 3 seconds")

    symbols = args.symbols or [s.strip() for s in Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
                               if s.strip() and not s.startswith("#")]
    floor = date.fromisoformat(args.floor)
    out = Path(args.out)
    store = Store(out)
    done_path = out / "symbols.jsonl"
    done = set()
    if done_path.exists():
        done = {json.loads(l)["symbol"] for l in done_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    exit_code = 0
    ensure_terminated(out / "listings.jsonl")
    with sync_playwright() as p, (out / "listings.jsonl").open("a", encoding="utf-8") as lfh:
        launch_kwargs, browser_label = pick_browser(p, args.browser, args.browser_exe)
        print(f"browser: {browser_label}", file=sys.stderr)
        browser = p.chromium.launch(headless=not args.headful, **launch_kwargs)
        ctx = browser.new_context(locale="th-TH", timezone_id="Asia/Bangkok")
        page = ctx.new_page()
        for sym in symbols:
            if sym in done:
                print(f"{sym}: already finished, skipping")
                continue
            t0 = time.perf_counter()
            s = crawl_symbol(page, store, sym, floor, args.delay, args.max_pages, lfh, args.base)
            s["wall_seconds"] = round(time.perf_counter() - t0, 1)
            s["browser"] = browser_label
            print(json.dumps(s, ensure_ascii=False))
            if s["stop_reason"].startswith("BLOCKED"):
                print("Blocked: stopping the whole run. Do not retry; see Phase 1.5.")
                exit_code = 2
                break
            # 404 is the real end of a tag's pagination. Other 4xx are blocks
            # (they never reach here, the blocked branch catches them first) and
            # must not be mistaken for a completed symbol.
            natural_end = s["stop_reason"].startswith(("reached floor", "HTTP 404", "no new items"))
            if natural_end:
                ensure_terminated(done_path)
                with done_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            else:
                print(f"{sym}: NOT marked finished ({s['stop_reason']}); fix and rerun, stored pages are reused")
        browser.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
