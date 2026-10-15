"""Fake Kaohoon for offline article-page crawl tests. Serves real stored article HTML plus failure cases.

  /news/{id}         id in OK_IDS   -> a real stored Kaohoon article page (200)
                     id 404xxx      -> 404
                     id 503xxx      -> 503 with a plain error page (transient)
                     id 5000xx      -> 500 (transient)
                     id 700xxx      -> 200 with no JSON-LD (layout drift)
                     id 403403      -> 403 Cloudflare challenge (block)
                     id 301301      -> 302 redirect to the home page
  /sitemap.xml, /post-sitemap.xml, /post-sitemap2.xml
"""
import gzip
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SPIKE = Path(sys.argv[2])
PAGES = []
for line in (SPIKE / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
    rec = json.loads(line)
    if "/tag/" not in rec["url"]:
        PAGES.append(gzip.open(SPIKE / "raw" / f"{rec['sha256']}.html.gz", "rt", encoding="utf-8").read())
HITS = []

INDEX = """<?xml version='1.0' encoding='UTF-8'?><sitemapindex xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
<sitemap><loc>{b}/sitemap-misc.xml</loc></sitemap>
<sitemap><loc>{b}/post-sitemap.xml</loc><lastmod>2026-09-25T08:56:10+00:00</lastmod></sitemap>
<sitemap><loc>{b}/post-sitemap2.xml</loc></sitemap></sitemapindex>"""
SM1 = """<?xml version='1.0' encoding='UTF-8'?><urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
\t<url>\n\t\t<loc>{b}/news/863237</loc>\n\t\t<lastmod>2026-09-15T07:20:21+00:00</lastmod>\n\t\t<changefreq>always</changefreq>\n\t</url>
\t<url>\n\t\t<loc>{b}/pr/866524</loc>\n\t\t<lastmod>2026-09-25T08:56:10+00:00</lastmod>\n\t</url></urlset>"""
SM2 = """<?xml version='1.0' encoding='UTF-8'?><urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
\t<url>\n\t\t<loc>{b}/breakingnews/332650</loc>\n\t\t<lastmod>2019-12-23T07:23:23+00:00</lastmod>\n\t</url></urlset>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        HITS.append(self.path)
        with open(SPIKE.parent / "hits.log", "a") as f:
            f.write(self.path + "\n")
        b = f"http://{self.headers['Host']}"
        if self.path == "/sitemap.xml":
            return self.send(200, INDEX.format(b=b), "application/xml")
        if self.path == "/post-sitemap.xml":
            return self.send(200, SM1.format(b=b), "application/xml")
        if self.path == "/post-sitemap2.xml":
            return self.send(200, SM2.format(b=b), "application/xml")
        if self.path in ("/", ""):
            return self.send(200, "<html><head><title>Kaohoon home</title></head><body>home</body></html>")
        parts = self.path.strip("/").split("/")
        aid = parts[-1]
        if aid == "403403":
            return self.send(403, "<html><head><title>Just a moment...</title></head><body>Checking your browser cf-chl</body></html>")
        if aid == "301301":
            return self.send(302, "", extra={"Location": "/"})
        if aid.startswith("404"):
            return self.send(404, "<html><head><title>Not found</title></head><body>404</body></html>")
        if aid.startswith("503"):
            return self.send(503, "<html><head><title>Service Unavailable</title></head><body>down</body></html>")
        if aid.startswith("5000"):
            return self.send(500, "<html><head><title>Error</title></head><body>error</body></html>")
        if aid.startswith("700"):
            return self.send(200, "<html><head><title>Redesigned</title></head><body><p>new layout, no json-ld</p></body></html>")
        page = PAGES[int(aid) % len(PAGES)]
        return self.send(200, page)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
