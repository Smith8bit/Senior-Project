"""Sentiment accuracy check, step 1: draw the blind labelling sample and build the labelling page.

(Pre-registration item A3b; open-items register Phase 1.9.)

Population = the exact units the sentiment feature will use (primary definition of the news coverage check):
  (article, ticker) pairs where the article is Thai, parsed from the article-page crawl (archive passB_v1), and its
  news trading day (hard rule 4, 16:30 ICT cutoff) falls inside the ticker's SET50
  membership window, 2020-01-01 .. 2025-12-31.

Draws a simple random sample (fixed seed) and writes, under --out (gitignored:
scraped headlines never go into git):
  sample.csv        the pairs to label (no model output anywhere)
  labeling.html     self-contained page, runs locally in a browser, exports labels.csv
  SAMPLE.json       seed, population size, sample hash, label protocol

Blindness: this script never loads or runs a sentiment model. Model scores are
produced only after labels.csv is frozen (hashed) by a separate step.

  python scripts/sentiment_check_sample.py                 # n=250, seed 20260930
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from news_coverage_check import news_trading_day, trading_calendar  # noqa: E402

PROTOCOL = {
    "unit": "one (headline, ticker) pair; judge the HEADLINE ONLY",
    "sentiment_question": "For a shareholder of TICKER, is this headline good news, bad news, or neither?",
    "sentiment_labels": {
        "pos": "good news for TICKER (earnings beat, upgrade, new contract, price rise reported, positive outlook)",
        "neu": "neither: factual/neutral, mixed, unclear, or not really about TICKER",
        "neg": "bad news for TICKER (loss, downgrade, lawsuit, price fall reported, negative outlook)",
    },
    "relevance_question": "Is the headline mainly about TICKER?",
    "relevance_labels": {"yes": "TICKER is the main subject",
                         "partly": "TICKER is one of several stocks, or mentioned in passing",
                         "no": "about the market, sector, economy or another company"},
    "rules": ["Label blind: never look at any model output before labels.csv is frozen",
              "Label in one sitting if possible, so the criterion stays stable",
              "Do not open the article; the model sees only the headline too",
              "When unsure between pos/neg and neu, choose neu"],
}


def build_population(args) -> list[dict]:
    cal = [d for d in trading_calendar(Path(args.prices)) if date(2020, 1, 1) <= d <= date(2025, 12, 31)]
    calset = set(cal)
    windows: dict[str, list[tuple[date, date]]] = defaultdict(list)
    with open(args.membership, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            windows[r["symbol"]].append((date.fromisoformat(r["effective_from"]), date.fromisoformat(r["effective_to"])))

    def in_window(sym: str, d: date) -> bool:
        return d in calset and any(a <= d <= b for a, b in windows.get(sym, ()))

    pop = []
    with open(args.articles, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            if a.get("parse_status") != "ok" or a["language"] != "th" or not a["date_published"] or not a["headline"]:
                continue
            td = news_trading_day(datetime.fromisoformat(a["date_published"]), cal)
            if td is None:
                continue
            for sym in a["symbols_tagged"]:
                if in_window(sym, td):
                    pop.append({"url": a["url"], "ticker": sym, "headline": a["headline"],
                                "section": a["url_section"], "published": a["date_published"],
                                "trading_day": td.isoformat()})
    pop.sort(key=lambda p: (p["url"], p["ticker"]))          # deterministic before sampling
    return pop


PAGE = """<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Sentiment check labelling</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--card:#fff;--line:#e7e5e4;--acc:#2563eb;--pos:#15803d;--neg:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--card:#292524;--line:#44403c;--acc:#60a5fa;--pos:#4ade80;--neg:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,"Noto Sans Thai",sans-serif}
main{max-width:760px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin:12px 0}
.meta{color:var(--mut);font-size:14px}.tick{font-size:28px;font-weight:700;letter-spacing:.02em}
.head{font-size:22px;line-height:1.6;margin:12px 0}
.row{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
button{font:inherit;padding:10px 14px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer;min-width:120px}
button.on{outline:3px solid var(--acc)}button.pos.on{background:var(--pos);color:#fff}button.neg.on{background:var(--neg);color:#fff}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--acc)}
details{color:var(--mut)}kbd{border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:13px}
</style></head><body><main>
<h1 style="font-size:20px;margin:4px 0">Sentiment check: blind labelling <span class="meta" id="count"></span></h1>
<div class="bar"><i id="bar" style="width:0"></i></div>
<details><summary>Protocol (read once before starting)</summary><div id="proto"></div></details>
<div class="card">
 <div class="meta" id="info"></div><div class="tick" id="tick"></div><div class="head" id="head"></div>
 <div class="meta">1. For a shareholder of <b id="t2"></b>: good news, bad news, or neither?</div>
 <div class="row"><button class="pos" data-s="pos">Positive <kbd>1</kbd></button><button data-s="neu">Neutral <kbd>2</kbd></button><button class="neg" data-s="neg">Negative <kbd>3</kbd></button></div>
 <div class="meta">2. Is the headline mainly about <b id="t3"></b>?</div>
 <div class="row"><button data-r="yes">Yes <kbd>Q</kbd></button><button data-r="partly">Partly <kbd>W</kbd></button><button data-r="no">No <kbd>E</kbd></button></div>
 <div class="row"><button id="prev">&larr; Back <kbd>&larr;</kbd></button><button id="next">Next &rarr; <kbd>Enter</kbd></button><button id="jump">First unlabelled</button></div>
</div>
<div class="row"><button id="export">Download labels.csv</button><span class="meta" id="saved"></span></div>
<p class="meta">Labels are kept in this browser tab and in this browser's local storage for this file. Download labels.csv when done and save it next to this page.</p>
</main><script>
const ITEMS=__ITEMS__, PROTO=__PROTO__, KEY="sentiment_check_labels___SAMPLEHASH__";
let L={}; try{L=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0, shown=Date.now();
const $=id=>document.getElementById(id);
$("proto").innerHTML="<p><b>Unit:</b> "+PROTO.unit+"</p><p><b>Sentiment:</b> "+PROTO.sentiment_question+"</p><ul>"+
 Object.entries(PROTO.sentiment_labels).map(([k,v])=>"<li><b>"+k+"</b>: "+v+"</li>").join("")+"</ul><p><b>Relevance:</b> "+PROTO.relevance_question+"</p><ul>"+
 Object.entries(PROTO.relevance_labels).map(([k,v])=>"<li><b>"+k+"</b>: "+v+"</li>").join("")+"</ul><ul>"+PROTO.rules.map(r=>"<li>"+r+"</li>").join("")+"</ul>";
function save(){try{localStorage.setItem(KEY,JSON.stringify(L))}catch(e){} $("saved").textContent="saved "+new Date().toLocaleTimeString()}
function done(){return ITEMS.filter(x=>L[x.id]&&L[x.id].s&&L[x.id].r).length}
function show(){const x=ITEMS[i],l=L[x.id]||{};shown=Date.now();
 $("info").textContent="#"+(i+1)+" of "+ITEMS.length+" \u00b7 "+x.trading_day+" \u00b7 "+x.section;
 $("tick").textContent=x.ticker;$("t2").textContent=x.ticker;$("t3").textContent=x.ticker;$("head").textContent=x.headline;
 document.querySelectorAll("[data-s]").forEach(b=>b.classList.toggle("on",b.dataset.s===l.s));
 document.querySelectorAll("[data-r]").forEach(b=>b.classList.toggle("on",b.dataset.r===l.r));
 const d=done();$("count").textContent=d+" / "+ITEMS.length+" labelled";$("bar").style.width=(100*d/ITEMS.length)+"%"}
function set(k,v){const x=ITEMS[i];L[x.id]=Object.assign(L[x.id]||{},{[k]:v,at:new Date().toISOString()});
 L[x.id].sec=(L[x.id].sec||0)+Math.round((Date.now()-shown)/1000);shown=Date.now();save();show();
 if(L[x.id].s&&L[x.id].r&&i<ITEMS.length-1){setTimeout(()=>{i++;show()},150)}}
document.querySelectorAll("[data-s]").forEach(b=>b.onclick=()=>set("s",b.dataset.s));
document.querySelectorAll("[data-r]").forEach(b=>b.onclick=()=>set("r",b.dataset.r));
$("prev").onclick=()=>{if(i>0){i--;show()}};$("next").onclick=()=>{if(i<ITEMS.length-1){i++;show()}};
$("jump").onclick=()=>{const j=ITEMS.findIndex(x=>!(L[x.id]&&L[x.id].s&&L[x.id].r));if(j>=0){i=j;show()}};
document.addEventListener("keydown",e=>{const k=e.key.toLowerCase();
 if(k==="1")set("s","pos");else if(k==="2")set("s","neu");else if(k==="3")set("s","neg");
 else if(k==="q")set("r","yes");else if(k==="w")set("r","partly");else if(k==="e")set("r","no");
 else if(e.key==="ArrowLeft")$("prev").onclick();else if(e.key==="Enter"||e.key==="ArrowRight")$("next").onclick()});
$("export").onclick=()=>{const rows=[["id","ticker","url","sentiment","relevance","labeled_at","seconds"]];
 ITEMS.forEach(x=>{const l=L[x.id]||{};rows.push([x.id,x.ticker,x.url,l.s||"",l.r||"",l.at||"",l.sec||""])});
 const csv=rows.map(r=>r.map(v=>'"'+String(v).replace(/"/g,'""')+'"').join(",")).join("\\r\\n");
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["\\ufeff"+csv],{type:"text/csv"}));a.download="labels.csv";a.click()};
show();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--articles", default="data/raw/kaohoon_articles/articles.jsonl")
    ap.add_argument("--membership", default="data/set50_membership.csv")
    ap.add_argument("--prices", default="data/raw/prices/siamchart/set50_union_2019_2025.csv")
    ap.add_argument("--out", default="data/raw/sentiment_check")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()
    out = Path(args.out)
    if (out / "sample.csv").exists():
        print(f"{out / 'sample.csv'} already exists; the sample is drawn once. Delete it only if no labelling has started.")
        return 1

    pop = build_population(args)
    rng = random.Random(args.seed)
    sample = rng.sample(pop, args.n)
    for k, s in enumerate(sample, 1):
        s["id"] = f"a3b-{k:03d}"  # old prefix kept on purpose: it is inside the frozen sample.csv
    out.mkdir(parents=True, exist_ok=True)
    fields = ["id", "ticker", "url", "section", "published", "trading_day", "headline"]
    with (out / "sample.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: s[k] for k in fields} for s in sample)
    sample_hash = hashlib.sha256((out / "sample.csv").read_bytes()).hexdigest()

    items = [{k: s[k] for k in ("id", "ticker", "url", "section", "trading_day", "headline")} for s in sample]
    page = (PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
                .replace("__PROTO__", json.dumps(PROTOCOL, ensure_ascii=False))
                .replace("__SAMPLEHASH__", sample_hash[:12]))
    (out / "labeling.html").write_text(page, encoding="utf-8")

    meta = {"created_at": datetime.now().astimezone().isoformat(), "seed": args.seed, "n": args.n,
            "population_pairs": len(pop), "population_articles": len({p["url"] for p in pop}),
            "sample_sha256": sample_hash, "protocol": PROTOCOL,
            "population_sections": dict(Counter(p["section"] for p in pop).most_common()),
            "sample_sections": dict(Counter(s["section"] for s in sample).most_common()),
            "sample_years": dict(sorted(Counter(s["trading_day"][:4] for s in sample).items())),
            "sample_distinct_tickers": len({s["ticker"] for s in sample})}
    (out / "SAMPLE.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("population_pairs", "population_articles", "n", "seed",
                                           "sample_sections", "sample_years", "sample_distinct_tickers")},
                     ensure_ascii=False, indent=1))
    print(f"open {out / 'labeling.html'} in a browser; sample sha256 {sample_hash[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
