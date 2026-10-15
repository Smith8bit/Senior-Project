"""Article-page crawl: fetch Kaohoon article pages (and the sitemap fallback) for every article
found by the article-list crawl.

Three subcommands, all resumable and all obeying hard rule 6 (a stored page is
never fetched again; parsing always runs from the stored copy):

  sitemaps   Fetch /sitemap.xml and every /post-sitemap*.xml. Writes
             sitemap_lastmod.jsonl: one row per post with its `lastmod` (UTC).
             ~257 page loads. This is fallback L2 of the Phase 1.5 plan:
             lastmod >= publish time always, so using it for the 16:30 cutoff
             can push an article later but never earlier (leak-safe).
  articles   Fetch each unique article URL from the article-list crawl in the study
             window, priority tickers first. Writes articles.jsonl: one parsed
             row per URL (datePublished/dateModified verbatim, headline,
             keywords, articleSection, body length, language guess).
  reparse    Rebuild articles.jsonl and sitemap_lastmod.jsonl from the stored
             pages only. No browser, no network. Use after fixing the parser.

Why article pages are cheap now: the server HTML already carries the full
Article JSON-LD (datePublished +07:00, keywords, articleSection, articleBody),
so the page is taken at `domcontentloaded` and the raw server response is
stored. The spike's 15 s "network idle" wait is gone.

Stop rules are pre-committed in claude/phase-1.5-scraping-contingency.md:
  exit 2  hard block (401/403/429/503 or challenge text): stop, wait >= 24 h,
          resume once at >= 2x delay. Never work around a block.
  exit 3  3 pauses for transient errors in one run (site or network down)
  exit 4  10 layout failures in a row (fix parser, then `reparse`)
  exit 5  gone rate > 5% after 200 attempts (URL scheme change?)
  exit 0  finished, or stopped at --limit / --max-hours

Usage (Windows, repo root):
  python scripts/crawl_articles.py sitemaps
  python scripts/crawl_articles.py articles --limit 50            # timing trial
  python scripts/crawl_articles.py articles --only-priority       # M1 subset only
  python scripts/crawl_articles.py articles --max-hours 10        # overnight chunk
  python scripts/crawl_articles.py reparse
"""
from __future__ import annotations

import argparse
import html as htmllib
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from spike_kaohoon import (ARTICLE_RE, BASE, BROWSER_ORDER, Store, ensure_terminated,
                           looks_blocked, pick_browser, utc_now)

DEFAULT_PRIORITY = "KBANK,PTT,BANPU,TCAP,DTAC"   # M1 subset (Phase 1 milestone)
LDJSON_RE = re.compile(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.S | re.I)
SITEMAP_ENTRY_RE = re.compile(r"<url>\s*<loc>([^<]+)</loc>\s*(?:<lastmod>([^<]+)</lastmod>)?", re.S)
SITEMAP_INDEX_RE = re.compile(r"<sitemap>\s*<loc>([^<]+)</loc>", re.S)
THAI_CHAR = re.compile(r"[฀-๿]")
LATIN_CHAR = re.compile(r"[A-Za-z]")
BLOCK_STATUSES = {401, 403, 429, 503}

# Phase 1.5 thresholds
PAUSE_AFTER_ERRORS = 5
PAUSE_SECONDS = 600
MAX_PAUSES = 3
MAX_LAYOUT_STREAK = 10
GONE_RATE_LIMIT = 0.05
GONE_RATE_MIN_ATTEMPTS = 200
MIN_FREE_BYTES = 2 * 1024**3


# --------------------------------------------------------------------------
# Parsing: pure functions over stored HTML/XML. No browser, no network.
# --------------------------------------------------------------------------

def _walk(node, out: list) -> None:
    if isinstance(node, list):
        for x in node:
            _walk(x, out)
    elif isinstance(node, dict):
        out.append(node)
        for key in ("@graph", "mainEntity", "itemListElement"):
            if key in node:
                _walk(node[key], out)


def _is_article(node: dict) -> bool:
    t = node.get("@type")
    types = t if isinstance(t, list) else [t]
    return any(isinstance(x, str) and x.endswith("Article") for x in types)


def _text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, list):
        v = ",".join(str(x) for x in v)
    return htmllib.unescape(str(v)).strip()


def language_guess(text: str) -> str:
    """'th' / 'en' / 'unknown' from the Thai share of letters (JSON-LD has no inLanguage)."""
    th, la = len(THAI_CHAR.findall(text)), len(LATIN_CHAR.findall(text))
    if th + la < 20:
        return "unknown"
    return "th" if th / (th + la) >= 0.3 else "en"


def parse_article(html: str, url: str) -> dict:
    """Extract the Article JSON-LD. `parse_status` is ok | layout."""
    nodes: list[dict] = []
    for block in LDJSON_RE.findall(html):
        try:
            _walk(json.loads(block, strict=False), nodes)
        except json.JSONDecodeError:
            continue
    arts = [n for n in nodes if _is_article(n)]
    art = arts[0] if arts else {}
    m = ARTICLE_RE.search(urlparse(url).path)
    body = _text(art.get("articleBody")) or ""
    headline = _text(art.get("headline"))
    row = {
        "url": url,
        "article_id": int(m.group(2)) if m else None,
        "url_section": m.group(1) if m else None,
        "date_published": art.get("datePublished"),     # verbatim, +07:00
        "date_modified": art.get("dateModified"),
        "headline": headline,
        "keywords": _text(art.get("keywords")),
        "article_section": _text(art.get("articleSection")),
        "body_chars": len(body),
        "language": language_guess(f"{headline or ''} {body[:2000]}"),
        "n_ldjson_nodes": len(nodes),
    }
    row["parse_status"] = "ok" if (arts and row["date_published"]) else "layout"
    return row


def body_text(html: str) -> str:
    """Full articleBody for offline feature work (not written to articles.jsonl)."""
    for block in LDJSON_RE.findall(html):
        try:
            nodes: list[dict] = []
            _walk(json.loads(block, strict=False), nodes)
        except json.JSONDecodeError:
            continue
        for n in nodes:
            if _is_article(n) and n.get("articleBody"):
                return _text(n["articleBody"]) or ""
    return ""


def parse_sitemap(xml: str) -> tuple[list[str], list[tuple[str, str | None]]]:
    """Return (child sitemap URLs, [(loc, lastmod)])."""
    children = [htmllib.unescape(u.strip()) for u in SITEMAP_INDEX_RE.findall(xml)]
    entries = [(htmllib.unescape(loc.strip()), lm.strip() if lm else None)
               for loc, lm in SITEMAP_ENTRY_RE.findall(xml)]
    return children, entries


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def fetch_raw(page, store: Store, url: str, delay: float) -> dict:
    """Return {'text','status','final_url','blocked','cached','seconds','error'}.

    Stores the RAW server response (not the rendered DOM): smaller, no ad
    markup, and exactly what the server sent. Exceptions are not stored, so a
    timeout is retried on the next run (Phase 1.5 F2).
    """
    if store.have(url):
        rec = store.index[url]
        return {"text": store.load(url), "status": rec.get("status"), "final_url": rec.get("final_url"),
                "blocked": rec.get("blocked"), "cached": True, "seconds": rec.get("seconds"), "error": None}
    time.sleep(delay)
    t0 = time.perf_counter()
    try:
        resp = page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        status = resp.status if resp else None
        text = resp.text() if resp else ""
        title = page.title()
        try:
            visible = page.inner_text("body", timeout=5_000)
        except Exception:
            visible = text[:3000]
    except Exception as e:  # timeout, DNS, connection reset...
        return {"text": "", "status": None, "final_url": None, "blocked": None, "cached": False,
                "seconds": round(time.perf_counter() - t0, 2), "error": f"{type(e).__name__}: {str(e)[:200]}"}
    seconds = round(time.perf_counter() - t0, 2)
    blocked = looks_blocked(status, title, visible)
    if blocked and status == 200 and parse_article(text, url)["parse_status"] == "ok":
        # A block-marker word inside a real article ("captcha" in a tech story)
        # is not a block: the page parsed as a complete Article.
        blocked = None
    store.save(url, text, {"status": status, "final_url": page.url, "title": title, "seconds": seconds,
                           "scraped_at": utc_now(), "blocked": blocked, "kind": "pass_b"})  # value kept: matches the archived records
    return {"text": text, "status": status, "final_url": page.url, "blocked": blocked, "cached": False,
            "seconds": seconds, "error": None}


def open_browser(p, args):
    launch_kwargs, label = pick_browser(p, args.browser, args.browser_exe)
    browser = p.chromium.launch(headless=not args.headful, **launch_kwargs)
    ctx = browser.new_context(locale="th-TH", timezone_id="Asia/Bangkok")
    if not args.load_assets:
        # Images, fonts and media are never needed: the data is in the HTML.
        ctx.route("**/*", lambda route: route.abort()
                  if route.request.resource_type in ("image", "media", "font") else route.continue_())
    return browser, ctx.new_page(), label


def jsonl_append(path: Path, row: dict) -> None:
    ensure_terminated(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def id_of(url: str | None) -> str | None:
    m = ARTICLE_RE.search(urlparse(url or "").path)
    return m.group(2) if m else None


# --------------------------------------------------------------------------
# Work list
# --------------------------------------------------------------------------

def build_worklist(listings: Path, date_from: str, date_to: str, priority: list[str],
                   only: list[str] | None, host: str | None = None) -> list[dict]:
    """Unique article URLs in the window. Priority tickers first, then the rest;
    newest first inside each group. An article tagged to several tickers is
    fetched once. Links to other sites (e.g. a facebook.com video card found in
    the EA listing, 2026-09-25) are skipped when `host` is given."""
    tags: dict[str, set] = defaultdict(set)
    meta: dict[str, dict] = {}
    for r in read_jsonl(listings):
        if not (date_from <= r["listing_date"] <= date_to):
            continue
        if host and urlparse(r["url"]).netloc != host:
            continue
        tags[r["url"]].add(r["symbol_queried"])
        meta.setdefault(r["url"], {"listing_date": r["listing_date"], "article_id": r["article_id"]})
    rank = {s: i for i, s in enumerate(priority)}
    work = []
    for url, syms in tags.items():
        if only and not (syms & set(only)):
            continue
        best = min((rank[s] for s in syms if s in rank), default=len(rank))
        work.append({"url": url, "symbols": sorted(syms), "group": best, **meta[url]})
    work.sort(key=lambda w: (w["group"], -int(w["article_id"])))
    return work


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------

def cmd_sitemaps(args, out: Path) -> int:
    from playwright.sync_api import sync_playwright
    store = Store(out / "sitemaps")
    rows_path = out / "sitemap_lastmod.jsonl"
    index_url = f"{args.base}/sitemap.xml"
    exit_code, n_rows, n_files = 0, 0, 0
    with sync_playwright() as p:
        browser, page, label = open_browser(p, args)
        print(f"browser: {label}", file=sys.stderr)
        r = fetch_raw(page, store, index_url, args.delay)
        if r["blocked"] or r["error"] or r["status"] != 200:
            print(f"sitemap index failed: {r['blocked'] or r['error'] or r['status']}")
            browser.close()
            return 2 if r["blocked"] else 3
        children, _ = parse_sitemap(r["text"])
        posts = [c for c in children if "/post-sitemap" in c]
        print(f"{len(posts)} post sitemaps listed")
        for i, url in enumerate(posts, 1):
            r = fetch_raw(page, store, url, args.delay)
            if r["blocked"]:
                print(f"BLOCKED on {url}: {r['blocked']}. Stop; see Phase 1.5 F1.")
                exit_code = 2
                break
            if r["error"] or (r["status"] or 0) >= 400:
                print(f"skip {url}: {r['error'] or r['status']} (rerun to retry)")
                continue
            n_files += 1
            if i % 25 == 0:
                print(f"{i}/{len(posts)} sitemaps")
        browser.close()
    n_rows = rebuild_sitemap_rows(out)
    print(json.dumps({"post_sitemaps": len(posts), "fetched_ok": n_files, "rows": n_rows,
                      "output": str(rows_path)}, ensure_ascii=False))
    return exit_code


def rebuild_sitemap_rows(out: Path) -> int:
    """sitemap_lastmod.jsonl from stored sitemaps; one row per post URL, latest snapshot wins."""
    store = Store(out / "sitemaps")
    rows: dict[str, dict] = {}
    for url, rec in sorted(store.index.items(), key=lambda kv: kv[1].get("scraped_at") or ""):
        if "/post-sitemap" not in url or not store.have(url):
            continue
        _, entries = parse_sitemap(store.load(url))
        for loc, lastmod in entries:
            m = ARTICLE_RE.search(urlparse(loc).path)
            rows[loc] = {"url": loc, "article_id": int(m.group(2)) if m else None,
                         "url_section": m.group(1) if m else None, "lastmod_utc": lastmod,
                         "sitemap": url, "sitemap_scraped_at": rec.get("scraped_at")}
    path = out / "sitemap_lastmod.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def cmd_articles(args, out: Path) -> int:
    from playwright.sync_api import sync_playwright
    free = shutil.disk_usage(out).free
    if free < MIN_FREE_BYTES:
        print(f"only {free / 1024**3:.1f} GB free; need >= 2 GB (Phase 1.5 F6)")
        return 6
    priority = [s.strip().upper() for s in args.priority.split(",") if s.strip()]
    only = priority if args.only_priority else ([s.upper() for s in args.symbols] if args.symbols else None)
    work = build_worklist(Path(args.listings), args.date_from, args.date_to, priority, only,
                          urlparse(args.base).netloc)
    store = Store(out)
    arts_path, att_path = out / "articles.jsonl", out / "attempts.jsonl"
    done = {r["url"] for r in read_jsonl(arts_path)}
    gone = {r["url"] for r in read_jsonl(att_path) if r.get("outcome") == "gone"}
    todo = [w for w in work if w["url"] not in done and w["url"] not in gone]
    print(json.dumps({"worklist": len(work), "already_parsed": len(work) - len(todo), "todo": len(todo),
                      "priority": priority, "only": only}, ensure_ascii=False))
    if args.limit:
        todo = todo[: args.limit]

    stats = defaultdict(int)
    err_streak = layout_streak = pauses = 0
    fresh_secs: list[float] = []
    t_start = time.time()
    exit_code, stop_reason = 0, "finished"
    with sync_playwright() as p:
        browser, page, label = open_browser(p, args)
        print(f"browser: {label}", file=sys.stderr)
        for i, w in enumerate(todo, 1):
            if args.max_hours and time.time() - t_start > args.max_hours * 3600:
                stop_reason = f"max_hours ({args.max_hours}) reached"
                break
            r = fetch_raw(page, store, w["url"], args.delay)
            att = {"url": w["url"], "at": utc_now(), "status": r["status"], "cached": r["cached"],
                   "seconds": r["seconds"], "final_url": r["final_url"]}
            if not r["cached"] and r["seconds"] is not None:
                fresh_secs.append(r["seconds"] + args.delay)

            if r["blocked"]:
                att["outcome"] = "blocked"
                att["detail"] = r["blocked"]
                jsonl_append(att_path, att)
                stop_reason, exit_code = f"BLOCKED: {r['blocked']} on {w['url']}", 2
                break
            if r["error"] or (r["status"] or 0) >= 500:
                att["outcome"] = "error"
                att["detail"] = r["error"] or f"HTTP {r['status']}"
                jsonl_append(att_path, att)
                stats["error"] += 1
                err_streak += 1
                if err_streak >= PAUSE_AFTER_ERRORS:
                    pauses += 1
                    if pauses >= MAX_PAUSES:
                        stop_reason, exit_code = f"{pauses} error pauses: site or network down", 3
                        break
                    print(f"{err_streak} errors in a row; pausing {args.pause_seconds / 60:.0f} min ({pauses}/{MAX_PAUSES - 1})")
                    time.sleep(args.pause_seconds)
                    err_streak = 0
                continue
            err_streak = 0
            if r["status"] in (404, 410):
                att["outcome"] = "gone"
                jsonl_append(att_path, att)
                stats["gone"] += 1
                attempted = stats["ok"] + stats["layout"] + stats["gone"]
                if attempted >= GONE_RATE_MIN_ATTEMPTS and stats["gone"] / attempted > GONE_RATE_LIMIT:
                    stop_reason, exit_code = f"gone rate {stats['gone']}/{attempted} > 5%: check URL scheme", 5
                    break
                continue

            row = parse_article(r["text"], w["url"])
            if id_of(r["final_url"]) not in (None, id_of(w["url"])):
                row["parse_status"] = "layout"
                row["redirected_to"] = r["final_url"]
            row.update({"http_status": r["status"], "symbols_tagged": w["symbols"],
                        "listing_date": w["listing_date"], "scraped_at": store.index[w["url"]].get("scraped_at")})
            jsonl_append(arts_path, row)
            att["outcome"] = row["parse_status"]
            jsonl_append(att_path, att)
            stats[row["parse_status"]] += 1
            if row["parse_status"] == "layout":
                layout_streak += 1
                if layout_streak >= MAX_LAYOUT_STREAK:
                    stop_reason, exit_code = f"{layout_streak} layout failures in a row: fix parser, then reparse", 4
                    break
            else:
                layout_streak = 0

            if i % 50 == 0 or i == len(todo):
                recent = fresh_secs[-200:]
                rate = sum(recent) / len(recent) if recent else None
                left = len(todo) - i
                eta = f"{left * rate / 3600:.1f} h" if rate else "n/a"
                print(f"{i}/{len(todo)}  ok={stats['ok']} layout={stats['layout']} gone={stats['gone']} "
                      f"err={stats['error']}  {rate or 0:.1f} s/article  ETA {eta}", flush=True)
        browser.close()

    summary = {"run_finished_at": utc_now(), "stop_reason": stop_reason, "exit_code": exit_code,
               "attempted": sum(stats.values()), **stats,
               "mean_s_per_fresh_article": round(sum(fresh_secs) / len(fresh_secs), 2) if fresh_secs else None,
               "wall_hours": round((time.time() - t_start) / 3600, 2), "browser": label,
               "delay": args.delay, "limit": args.limit, "only": only}
    jsonl_append(out / "runs.jsonl", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if exit_code == 2:
        print("Blocked. Do not retry today. Wait >= 24 h, then rerun ONCE with --delay 8 (Phase 1.5 F1).")
    return exit_code


def cmd_reparse(args, out: Path) -> int:
    """Rebuild articles.jsonl from stored pages. Keeps tagging info from the work list."""
    store = Store(out)
    work = {w["url"]: w for w in build_worklist(Path(args.listings), args.date_from, args.date_to, [], None,
                                                urlparse(args.base).netloc)}
    rows, stats = [], defaultdict(int)
    for url, rec in store.index.items():
        if not store.have(url) or rec.get("status") in (404, 410) or url not in work:
            continue
        row = parse_article(store.load(url), url)
        if id_of(rec.get("final_url")) not in (None, id_of(url)):
            row["parse_status"] = "layout"
            row["redirected_to"] = rec.get("final_url")
        row.update({"http_status": rec.get("status"), "symbols_tagged": work[url]["symbols"],
                    "listing_date": work[url]["listing_date"], "scraped_at": rec.get("scraped_at")})
        rows.append(row)
        stats[row["parse_status"]] += 1
    tmp = out / "articles.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out / "articles.jsonl")
    n_sm = rebuild_sitemap_rows(out) if (out / "sitemaps" / "manifest.jsonl").exists() else 0
    print(json.dumps({"articles": len(rows), **stats, "sitemap_rows": n_sm}, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("sitemaps", "articles", "reparse"))
    ap.add_argument("--listings", default="data/raw/kaohoon_article_lists/listings.jsonl")
    ap.add_argument("--out", default="data/raw/kaohoon_articles")
    ap.add_argument("--from", dest="date_from", default="2019-12-25",
                    help="listing-date window start (buffer before 2020-01-01)")
    ap.add_argument("--to", dest="date_to", default="2026-01-05",
                    help="listing-date window end (buffer after 2025-12-31)")
    ap.add_argument("--priority", default=DEFAULT_PRIORITY, help="comma list fetched first (M1 subset)")
    ap.add_argument("--only-priority", action="store_true", help="fetch only articles tagged to --priority")
    ap.add_argument("--symbols", nargs="+", help="fetch only articles tagged to these tickers")
    ap.add_argument("--limit", type=int, help="stop after N articles (timing trial)")
    ap.add_argument("--max-hours", type=float, help="stop cleanly after this many hours")
    ap.add_argument("--delay", type=float, default=4.0, help="seconds between requests (min 3)")
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--load-assets", action="store_true", help="do not block images/fonts/media")
    ap.add_argument("--browser", default="auto", choices=("auto", *BROWSER_ORDER))
    ap.add_argument("--browser-exe")
    ap.add_argument("--base", default=BASE, help=argparse.SUPPRESS)  # offline tests
    ap.add_argument("--pause-seconds", type=float, default=PAUSE_SECONDS, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.delay < 3:
        ap.error("--delay must be at least 3 seconds")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    return {"sitemaps": cmd_sitemaps, "articles": cmd_articles, "reparse": cmd_reparse}[args.command](args, out)


if __name__ == "__main__":
    sys.exit(main())
