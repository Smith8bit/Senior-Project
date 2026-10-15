"""Offline tests for scripts/crawl_articles.py against tests/fake_kaohoon.py.

Run from the repo root:  python tests/test_crawl_articles.py
Needs Playwright + a Chromium build, and spike_out/ (stored real Kaohoon pages).
The polite delay is patched out here only; the real script refuses --delay < 3.
"""
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import crawl_articles  # noqa: E402

RUNNER = f"""
import sys, time
sys.path.insert(0, {str(ROOT / 'scripts')!r})
time.sleep = lambda s: None          # tests only: no polite delay, no pauses
import crawl_articles
sys.argv = ['crawl_articles.py'] + sys.argv[1:]
sys.exit(crawl_articles.main())
"""


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def run(tmp, base, *args):
    p = subprocess.run([sys.executable, "-c", RUNNER, *args, "--base", base, "--out", str(tmp / "out"),
                        "--listings", str(tmp / "listings.jsonl"), "--browser", "chromium"],
                       capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout + p.stderr


def write_listings(tmp, base, spec):
    with (tmp / "listings.jsonl").open("w", encoding="utf-8") as f:
        for sym, section, aid, d in spec:
            f.write(json.dumps({"symbol_queried": sym, "url": f"{base}/{section}/{aid}", "article_id": aid,
                                "listing_date": d, "url_section": section}) + "\n")


def jl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()] if Path(path).exists() else []


def hits(tmp):
    p = tmp / "hits.log"
    return p.read_text().splitlines() if p.exists() else []


def main():
    failures = []

    def check(name, cond, info=""):
        print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"   <- {info}"))
        if not cond:
            failures.append(name)

    # ---- 0. parser on real stored pages --------------------------------
    import gzip
    man = jl(ROOT / "spike_out" / "manifest.jsonl")
    arts = [m for m in man if "/tag/" not in m["url"]]
    for m in arts:
        html = gzip.open(ROOT / "spike_out" / "raw" / f"{m['sha256']}.html.gz", "rt", encoding="utf-8").read()
        r = crawl_articles.parse_article(html, m["url"])
        check(f"parse {m['url'].rsplit('/', 1)[-1]}: ok, +07:00, th, body",
              r["parse_status"] == "ok" and r["date_published"].endswith("+07:00") and r["language"] == "th"
              and r["body_chars"] > 500 and r["keywords"] and "&#" not in (r["headline"] or ""), r)
    check("parse: page without JSON-LD is layout",
          crawl_articles.parse_article("<html><body>x</body></html>", "https://www.kaohoon.com/news/1")["parse_status"] == "layout")
    check("parse: entity unescaped",
          crawl_articles._text("BANPU &#8220;x&#8221;") == "BANPU “x”")
    check("language: english", crawl_articles.language_guess("Kaohoon International reports that the SET index rose today") == "en")

    tmp = Path(tempfile.mkdtemp())
    spike = tmp / "spike"
    shutil.copytree(ROOT / "spike_out", spike)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    srv = subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_kaohoon.py"), str(port), str(spike)])
    time.sleep(1.5)
    try:
        # ---- 1. sitemaps ------------------------------------------------
        write_listings(tmp, base, [])
        code, out = run(tmp, base, "sitemaps")
        rows = {r["url"].rsplit("/", 1)[-1]: r for r in jl(tmp / "out" / "sitemap_lastmod.jsonl")}
        check("sitemaps: exit 0, 3 rows from 2 post sitemaps (misc skipped)",
              code == 0 and set(rows) == {"863237", "866524", "332650"}, out[-600:])
        check("sitemaps: lastmod kept verbatim (UTC)", rows.get("332650", {}).get("lastmod_utc") == "2019-12-23T07:23:23+00:00")
        n = len(hits(tmp))
        code, out = run(tmp, base, "sitemaps")
        check("sitemaps: rerun makes no requests (hard rule 6)", code == 0 and len(hits(tmp)) == n, hits(tmp)[n:])

        # ---- 2. normal articles run with priority + gone + redirect ---------
        spec = [("PTT", "news", 100001, "2021-03-01"), ("KBANK", "news", 100002, "2020-05-01"),
                ("KBANK", "breakingnews", 100003, "2024-01-01"), ("PTT", "news", 100003, "2024-01-01"),
                ("AOT", "column", 100004, "2023-01-01"), ("AOT", "news", 404001, "2022-01-01"),
                ("AOT", "news", 301301, "2022-01-01"), ("AOT", "news", 100005, "2019-06-01"),   # outside window
                ("BANPU", "news", 100006, "2025-12-31")]
        spec[2] = ("KBANK", "news", 100003, "2024-01-01")
        write_listings(tmp, base, spec)
        with (tmp / "listings.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"symbol_queried": "EA", "url": "https://www.facebook.com/kaohoononline/videos/712609566582203/",
                                "article_id": 712609566582203, "listing_date": "2022-04-12",
                                "url_section": "kaohoononline/videos"}) + "\n")
        n = len(hits(tmp))
        code, out = run(tmp, base, "articles", "--priority", "KBANK,BANPU")
        A = {r["article_id"]: r for r in jl(tmp / "out" / "articles.jsonl")}
        att = jl(tmp / "out" / "attempts.jsonl")
        order = [h.rsplit("/", 1)[-1] for h in hits(tmp)[n:]]
        check("articles: exit 0", code == 0, out[-800:])
        check("articles: window filter drops 2019-06 item", 100005 not in A and "100005" not in order, order)
        check("articles: shared article fetched once", order.count("100003") == 1, order)
        check("articles: off-site link (facebook video) skipped",
              not any("712609566582203" in a["url"] for a in jl(tmp / "out" / "attempts.jsonl")), order)
        check("articles: priority tickers first, newest first", order[:3] == ["100003", "100002", "100006"], order)
        check("articles: ok rows parsed", all(A[i]["parse_status"] == "ok" for i in (100001, 100002, 100003, 100004, 100006)), A.keys())
        check("articles: tagged symbols merged", A[100003]["symbols_tagged"] == ["KBANK", "PTT"], A[100003])
        check("articles: 404 recorded as gone", any(a["outcome"] == "gone" and a["url"].endswith("404001") for a in att))
        check("articles: redirect counted as layout", A.get(301301, {}).get("parse_status") == "layout", A.get(301301))
        summ = jl(tmp / "out" / "runs.jsonl")[-1]
        check("articles: run summary written", summ["ok"] == 5 and summ["gone"] == 1 and summ["layout"] == 1, summ)

        # ---- 3. resume: nothing refetched, nothing duplicated ---------------
        n = len(hits(tmp))
        code, out = run(tmp, base, "articles")
        check("resume: no requests, no duplicate rows",
              code == 0 and len(hits(tmp)) == n and len(jl(tmp / "out" / "articles.jsonl")) == len(A), hits(tmp)[n:])

        # ---- 4. crash recovery: page stored but row missing -----------------
        rows = jl(tmp / "out" / "articles.jsonl")
        (tmp / "out" / "articles.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[:-1]),
                                                   encoding="utf-8")
        n = len(hits(tmp))
        code, out = run(tmp, base, "articles")
        check("crash recovery: row rebuilt from store without a request",
              len(jl(tmp / "out" / "articles.jsonl")) == len(rows) and len(hits(tmp)) == n, out[-400:])

        # ---- 5. reparse offline --------------------------------------------
        before = sorted(json.dumps({k: v for k, v in r.items() if k != "scraped_at"}, sort_keys=True)
                        for r in jl(tmp / "out" / "articles.jsonl"))
        n = len(hits(tmp))
        code, out = run(tmp, base, "reparse")
        after = sorted(json.dumps({k: v for k, v in r.items() if k != "scraped_at"}, sort_keys=True)
                       for r in jl(tmp / "out" / "articles.jsonl"))
        check("reparse: offline, same rows", code == 0 and len(hits(tmp)) == n and before == after, out[-400:])

        # ---- 6. hard block stops the run, is not cached ---------------------
        write_listings(tmp, base, spec + [("KBANK", "news", 403403, "2025-06-01"), ("KBANK", "news", 100007, "2025-05-01")])
        n = len(hits(tmp))
        code, out = run(tmp, base, "articles")
        new = [h.rsplit("/", 1)[-1] for h in hits(tmp)[n:]]
        check("block: exit 2 and stops before next article", code == 2 and "100007" not in new, (code, new))
        code2, _ = run(tmp, base, "articles")
        check("block: not cached as content (retried next run)", code2 == 2 and hits(tmp)[-1].endswith("403403"))

        # ---- 7. 503 is treated as a block (Cloudflare uses it for challenges)
        write_listings(tmp, base, [("KBANK", "news", 503001, "2025-01-01")])
        code, _ = run(tmp, base, "articles")
        check("503: treated as block, exit 2", code == 2)

        # ---- 8. transient 500s: pause, then stop after 3 pauses --------------
        write_listings(tmp, base, [("KBANK", "news", 500000 + i, "2025-01-01") for i in range(1, 16)])
        code, out = run(tmp, base, "articles", "--pause-seconds", "0")
        att = [a for a in jl(tmp / "out" / "attempts.jsonl") if a["url"].rsplit("/", 1)[-1].startswith("5000")]
        check("errors: exit 3 after 3 pauses (15 errors)", code == 3 and len(att) == 15, out[-500:])

        # ---- 9. layout drift streak ------------------------------------------
        write_listings(tmp, base, [("KBANK", "news", 700000 + i, "2025-01-01") for i in range(1, 13)])
        code, out = run(tmp, base, "articles")
        lay = [a for a in jl(tmp / "out" / "attempts.jsonl") if a.get("outcome") == "layout" and "/7000" in a["url"]]
        check("layout: exit 4 after 10 in a row", code == 4 and len(lay) == 10, out[-500:])

        # ---- 10. --limit and --only-priority --------------------------------
        write_listings(tmp, base, [("KBANK", "news", 110000 + i, "2025-01-01") for i in range(1, 6)]
                       + [("AOT", "news", 120000 + i, "2025-01-01") for i in range(1, 6)])
        code, out = run(tmp, base, "articles", "--only-priority", "--priority", "KBANK", "--limit", "3")
        A = jl(tmp / "out" / "articles.jsonl")
        got = {r["article_id"] for r in A if 110000 < r["article_id"] < 130000}
        check("limit+only-priority: 3 KBANK articles, no AOT", code == 0 and len(got) == 3 and all(g < 120000 for g in got), got)
    finally:
        srv.terminate()
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
