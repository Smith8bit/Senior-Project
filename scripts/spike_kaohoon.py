"""Phase 1.4 spike: Playwright test run against Kaohoon for ONE symbol.

Answers:
  Q1  Does Playwright get past bot detection?
  Q3  What does a /tag/{symbol} listing page expose (links, headlines, dates)?
  Q4  Do article pages carry the JSON-LD fields the scraper spec needs?

Also records load time per page and page size (inputs for Q6/Q7).

Rules enforced here (see senior-project-status.md):
  - Hard rule 6: every fetched page is saved (gzipped) with a manifest line, so
    nothing from the spike is wasted or re-fetched. A URL already in the
    manifest is never fetched again; its stored copy is used instead.
  - Phase 1.15: published dates are kept verbatim; scraped_at is tz-aware UTC.
  - Phase 1.19: one tab, fixed polite delay, stop immediately on any block or
    CAPTCHA. The script never tries to get around a block.

Usage (Windows, from the repo root):
  pip install playwright
  playwright install chromium
  python scripts/spike_kaohoon.py --symbol BANPU
  python scripts/spike_kaohoon.py --symbol BANPU --headful --articles 5

  The browser is auto-detected: Playwright's bundled Chromium if this
  environment actually has it downloaded, otherwise an installed Edge /
  Chrome / Brave. Force one with --browser msedge (or --browser-exe <path>).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright

BASE = "https://www.kaohoon.com"
# Kaohoon item URLs are /{section}/{id}, where section may be nested:
# news, breakingnews, summary, column, infographic, news/local, ...
# (found 2026-09-15: everything except news/breakingnews was being dropped).
ARTICLE_RE = re.compile(r"^/([a-z][a-z-]*(?:/[a-z][a-z-]*)*)/(\d+)/?$")
BLOCK_MARKERS = (
    "just a moment", "attention required", "captcha", "cf-chl",
    "access denied", "verify you are human", "are you a robot",
    "request blocked", "unusual traffic",
)
BLOCK_STATUSES = {401, 403, 429, 503}
REQUIRED_LD = ("headline", "datePublished", "dateModified", "keywords", "articleSection")
OFFSET_RE = re.compile(r"[+-]\d{2}:?\d{2}$|Z$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


BROWSER_PATHS = {
    "msedge": (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
               r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    "chrome": (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
               r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
               str(Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe")),
    "brave": (r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
              r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe"),
}
BROWSER_ORDER = ("chromium", "msedge", "chrome", "brave")


def pick_browser(p, choice: str, exe: str | None) -> tuple[dict, str]:
    """Resolve --browser/--browser-exe into launch kwargs plus a label.

    Playwright's bundled Chromium only works when the exact revision this
    playwright build expects has been downloaded into the browser cache, so a
    missing one falls back to a browser already installed on the machine
    rather than crashing on launch. Which browser was used is recorded in the
    report: it is part of the answer to Q1.
    """
    if exe:
        if not Path(exe).exists():
            raise SystemExit(f"--browser-exe not found: {exe}")
        return {"executable_path": exe}, f"custom ({exe})"

    names = BROWSER_ORDER if choice == "auto" else (choice,)
    for name in names:
        if name == "chromium":
            # channel="chromium" runs the full browser, so a missing
            # chromium_headless_shell build is not a problem.
            path = Path(p.chromium.executable_path)
            if path.exists():
                return {"channel": "chromium"}, f"chromium bundled ({path})"
            continue
        for cand in BROWSER_PATHS[name]:
            if Path(cand).exists():
                kwargs = {"executable_path": cand} if name == "brave" else {"channel": name}
                return kwargs, f"{name} ({cand})"
    raise SystemExit(
        f"no usable browser found (tried: {', '.join(names)}).\n"
        "Download Playwright's Chromium for this environment:\n"
        "  python -m playwright install chromium\n"
        "or point the script at one you already have:\n"
        "  --browser msedge | chrome | brave   --browser-exe <full path to .exe>"
    )


class Store:
    """Minimal snapshot store: out/raw/<sha256>.html.gz + out/manifest.jsonl."""

    def __init__(self, root: Path):
        self.raw = root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.manifest = root / "manifest.jsonl"
        self.index: dict[str, dict] = {}
        if self.manifest.exists():
            for line in self.manifest.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.index[rec["url"]] = rec

    def have(self, url: str) -> bool:
        return url in self.index

    def load(self, url: str) -> str:
        rec = self.index[url]
        with gzip.open(self.raw / f"{rec['sha256']}.html.gz", "rt", encoding="utf-8") as f:
            return f.read()

    def save(self, url: str, html: str, meta: dict) -> dict:
        data = html.encode("utf-8")
        sha = hashlib.sha256(data).hexdigest()
        path = self.raw / f"{sha}.html.gz"
        if not path.exists():
            with gzip.open(path, "wb") as f:
                f.write(data)
        rec = {"url": url, "sha256": sha, "bytes_raw": len(data),
               "bytes_gz": path.stat().st_size, **meta}
        with self.manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.index[url] = rec
        return rec


def replay(page, url: str, html: str) -> None:
    """Load a stored page offline, keeping its original URL as the base so
    relative links still resolve. No network request is made."""
    base = f'<base href="{url}">'
    if re.search(r"<head[^>]*>", html, flags=re.I):
        html = re.sub(r"(<head[^>]*>)", lambda m: m.group(1) + base, html, count=1, flags=re.I)
    else:
        html = base + html
    page.set_content(html, wait_until="domcontentloaded")


def looks_blocked(status: int | None, title: str, body_text: str) -> str | None:
    if status in BLOCK_STATUSES:
        return f"HTTP {status}"
    hay = f"{title}\n{body_text[:3000]}".lower()
    for m in BLOCK_MARKERS:
        if m in hay:
            return f"marker '{m}'"
    return None


def fetch(page, store: Store, url: str, delay: float) -> dict:
    """Return {'html', 'status', 'seconds', 'blocked', 'cached'}."""
    if store.have(url):
        rec = store.index[url]
        return {"html": store.load(url), "status": rec.get("status"),
                "seconds": rec.get("seconds"), "blocked": None, "cached": True}
    time.sleep(delay)
    t0 = time.perf_counter()
    resp = page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    try:
        page.wait_for_load_state("networkidle", timeout=15_000)
    except Exception:
        pass  # some pages never go idle; domcontentloaded is enough
    seconds = round(time.perf_counter() - t0, 2)
    status = resp.status if resp else None
    html = page.content()
    title = page.title()
    body_text = page.inner_text("body") if page.query_selector("body") else ""
    blocked = looks_blocked(status, title, body_text)
    store.save(url, html, {
        "status": status, "final_url": page.url, "title": title,
        "seconds": seconds, "scraped_at": utc_now(), "blocked": blocked,
    })
    return {"html": html, "status": status, "seconds": seconds,
            "blocked": blocked, "cached": False}


LISTING_JS = r"""
() => {
  // Primary: WordPress listing cards <div class="post-item post-{id} ...">.
  // 12 per page on /tag/{slug}/page/{N} (verified 2026-09-15 on BANPU pages 1 and 40).
  // Scoped to the .posts-items list: the site-wide news ticker and sidebars
  // also contain article links that do NOT belong to this tag.
  const re = /^\/([a-z][a-z-]*(?:\/[a-z][a-z-]*)*)\/(\d+)\/?$/;
  const pathOf = h => { try { return new URL(h, document.baseURI).pathname; } catch (e) { return ''; } };
  const cards = [...document.querySelectorAll('.posts-items .post-item')];
  const items = [];
  const seen = new Set();
  for (const card of cards) {
    const idm = (card.className || '').match(/\bpost-(\d+)\b/);
    const links = [...card.querySelectorAll('a[href]')].filter(a => re.test(pathOf(a.href)));
    const main = (idm && links.find(a => pathOf(a.href).match(re)[2] === idm[1])) || links[0];
    if (!main || seen.has(main.href)) continue;
    seen.add(main.href);
    const t = card.querySelector('time');
    const texts = [...new Set(links.filter(a => a.href === main.href)
                  .map(a => (a.innerText || a.getAttribute('title') || '').trim()).filter(Boolean))];
    items.push({url: main.href, section: pathOf(main.href).match(re)[1],
                post_id: idm ? idm[1] : null, link_texts: texts,
                time_datetime: t ? t.getAttribute('datetime') : null,
                time_text: t ? (t.innerText || '').trim() : null,
                card_text: (card.innerText || '').trim().slice(0, 300)});
  }
  const pager = [...document.querySelectorAll('a[href]')]
    .map(a => a.href).filter(h => /\/page\/\d+/.test(h));
  return {items, n_cards: cards.length, pager_links: [...new Set(pager)].slice(0, 20)};
}
"""

THAI_DATE_RE = re.compile(
    r"\d{1,2}\s*(ม\.?ค|ก\.?พ|มี\.?ค|เม\.?ย|พ\.?ค|มิ\.?ย|ก\.?ค|ส\.?ค|ก\.?ย|ต\.?ค|พ\.?ย|ธ\.?ค)"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
)


def parse_jsonld(page) -> list[dict]:
    raw = page.eval_on_selector_all(
        'script[type="application/ld+json"]', "els => els.map(e => e.textContent)")
    objs: list[dict] = []
    for block in raw:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                objs.append(node)
                for key in ("@graph", "mainEntity", "itemListElement"):
                    v = node.get(key)
                    if isinstance(v, list):
                        stack.extend(v)
                    elif isinstance(v, dict):
                        stack.append(v)
            elif isinstance(node, list):
                stack.extend(node)
    return objs


def is_article(node: dict) -> bool:
    t = node.get("@type")
    types = t if isinstance(t, list) else [t]
    return any(isinstance(x, str) and x.endswith("Article") for x in types)


def check_article(page, url: str) -> dict:
    nodes = parse_jsonld(page)
    arts = [n for n in nodes if is_article(n)]
    art = arts[0] if arts else {}
    meta_pub = page.eval_on_selector_all(
        'meta[property="article:published_time"]', "els => els.map(e => e.content)")
    fields = {k: (k in art and art[k] not in (None, "", [])) for k in REQUIRED_LD}
    pub = art.get("datePublished")
    mod = art.get("dateModified")
    return {
        "url": url,
        "article_id": int(ARTICLE_RE.search(urlparse(url).path).group(2)),
        "jsonld_types": sorted({str(n.get("@type")) for n in nodes}),
        "article_node_found": bool(arts),
        "fields_present": fields,
        "all_required_present": all(fields.values()),
        "datePublished": pub,                     # verbatim, never converted
        "dateModified": mod,
        "has_tz_offset": bool(pub and OFFSET_RE.search(pub)),
        "is_plus_0700": bool(pub and pub.endswith("+07:00")),
        "headline": art.get("headline"),
        "keywords": art.get("keywords"),
        "articleSection": art.get("articleSection"),
        "inLanguage": art.get("inLanguage"),
        "meta_article_published_time": meta_pub or None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BANPU")
    ap.add_argument("--tag-url", help="override listing URL (default {BASE}/tag/{symbol lowercase})")
    ap.add_argument("--articles", type=int, default=3, help="articles to open from page 1")
    ap.add_argument("--delay", type=float, default=4.0, help="seconds between requests (min 3)")
    ap.add_argument("--headful", action="store_true", help="show the browser window")
    ap.add_argument("--browser", default="auto",
                    choices=("auto", "chromium", "msedge", "chrome", "brave"),
                    help="which browser to drive (default: first one installed)")
    ap.add_argument("--browser-exe", help="full path to a Chromium-based browser, overrides --browser")
    ap.add_argument("--out", default="spike_out", help="output folder (gitignored)")
    args = ap.parse_args()
    if args.delay < 3:
        ap.error("--delay must be at least 3 seconds")

    out = Path(args.out)
    store = Store(out)
    # Tag slugs are lowercase; pagination is /tag/{slug}/page/{N} (verified by Siwat 2026-09-15)
    tag_url = args.tag_url or f"{BASE}/tag/{args.symbol.lower()}"
    report: dict = {"symbol": args.symbol, "tag_url": tag_url, "started_at": utc_now(),
                    "delay_s": args.delay, "pages": [], "stopped_because": None}

    with sync_playwright() as p:
        launch_kwargs, browser_label = pick_browser(p, args.browser, args.browser_exe)
        report["browser"] = browser_label
        browser = p.chromium.launch(headless=not args.headful, **launch_kwargs)
        ctx = browser.new_context(locale="th-TH", timezone_id="Asia/Bangkok")
        page = ctx.new_page()

        # ---- Q1 + Q3: listing page -------------------------------------
        r = fetch(page, store, tag_url, args.delay)
        report["pages"].append({"url": tag_url, "kind": "listing", "status": r["status"],
                                "seconds": r["seconds"], "cached": r["cached"],
                                "blocked": r["blocked"]})
        if r["blocked"]:
            report["stopped_because"] = f"blocked on listing: {r['blocked']}"
        else:
            if r["cached"]:
                replay(page, tag_url, r["html"])
            listing = page.evaluate(LISTING_JS)
            items = listing["items"]
            n = len(items)
            with_time = sum(1 for i in items if i["time_datetime"] or i["time_text"])
            with_text_date = sum(1 for i in items if THAI_DATE_RE.search(i["card_text"]))
            with_headline = sum(1 for i in items if any(len(t) > 15 for t in i["link_texts"]))
            report["q3_listing"] = {
                "article_links": n,
                "cards_on_page": listing["n_cards"],
                "with_headline_text": with_headline,
                "with_time_element": with_time,
                "with_date_like_text_in_card": with_text_date,
                "pager_links_sample": listing["pager_links"],
                "listing_jsonld_types": sorted({str(x.get("@type")) for x in parse_jsonld(page)}),
                "sample": items[:5],
            }

            # ---- Q4: article pages ------------------------------------
            checks = []
            for item in items[: args.articles]:
                url = item["url"]
                a = fetch(page, store, url, args.delay)
                report["pages"].append({"url": url, "kind": "article", "status": a["status"],
                                        "seconds": a["seconds"], "cached": a["cached"],
                                        "blocked": a["blocked"]})
                if a["blocked"]:
                    report["stopped_because"] = f"blocked on article: {a['blocked']}"
                    break
                if a["cached"]:
                    replay(page, url, a["html"])
                checks.append(check_article(page, url))
            report["q4_articles"] = checks

        browser.close()

    # ---- summary -------------------------------------------------------
    fresh = [pg for pg in report["pages"] if not pg["cached"] and pg["seconds"] is not None]
    recs = [store.index[pg["url"]] for pg in report["pages"] if pg["url"] in store.index]
    report["timing"] = {
        "fresh_fetches": len(fresh),
        "mean_load_s": round(sum(pg["seconds"] for pg in fresh) / len(fresh), 2) if fresh else None,
        "effective_s_per_page_incl_delay": round(
            sum(pg["seconds"] for pg in fresh) / len(fresh) + args.delay, 2) if fresh else None,
        "mean_bytes_gz": int(sum(x["bytes_gz"] for x in recs) / len(recs)) if recs else None,
    }
    blocked = report["stopped_because"]
    q4 = report.get("q4_articles", [])
    report["answers"] = {
        "Q1_passes_bot_detection": None if not report["pages"] else not blocked,
        "Q3_listing_has_dates": (report.get("q3_listing", {}).get("with_time_element", 0)
                                 + report.get("q3_listing", {}).get("with_date_like_text_in_card", 0)) > 0
                                if "q3_listing" in report else None,
        "Q4_all_articles_have_required_jsonld": all(c["all_required_present"] for c in q4) if q4 else None,
        "Q4_all_dates_plus_0700": all(c["is_plus_0700"] for c in q4) if q4 else None,
    }
    report["finished_at"] = utc_now()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"spike_report_{args.symbol}_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"answers": report["answers"], "timing": report["timing"],
                      "stopped_because": blocked, "report": str(path)},
                     ensure_ascii=False, indent=2))
    return 2 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
