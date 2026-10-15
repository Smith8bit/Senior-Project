"""Sentiment accuracy check, step 3: a Thai LLM labels the same 250 headlines (zero-shot).

This is the comparator declared in advance on 2026-09-30 and fixed in writing on 2026-10-06
(project doc claude/pre-registration-amendment-2026-10-06-llm-comparator.md). The model,
prompt and settings below are frozen; the script records the prompt's SHA-256 so any later
change is visible.

Runs on Siwat's PC against a local Ollama server (http://localhost:11434). Standard library only.

  python scripts/sentiment_check_llm.py selftest     # 4 made-up headlines + time estimate (no sample items)
  python scripts/sentiment_check_llm.py label        # the 250 sample items (resumable; refuses unless labels are frozen)
  python scripts/sentiment_check_llm.py score        # agreement with your labels (half B is the headline number)
  python scripts/sentiment_check_llm.py --model scb10x/typhoon2.5-qwen3-4b selftest   # only under the declared fallback rule
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sentiment_check_score as S  # noqa: E402

PRIMARY_MODEL = "scb10x/typhoon2.1-gemma3-12b"
FALLBACK_MODEL = "scb10x/typhoon2.5-qwen3-4b"
OLLAMA = "http://localhost:11434"
OPTIONS = {"temperature": 0, "seed": 20260930, "num_predict": 64}
FALLBACK_MAX_HOURS = 3.0  # fallback rule: switch only if the selftest projects more than this for 250 items

SYSTEM = ("You label Thai financial news headlines for a research study. "
          "Judge the headline only. Reply with JSON only.")
TEMPLATE = """Stock: {ticker}
Section: {section}
Headline: {headline}

1. For a shareholder of {ticker}, is this headline good news, bad news, or neither?
   pos = good news for {ticker} (earnings beat, upgrade, new contract, price rise reported, positive outlook)
   neu = neither: factual/neutral, mixed, unclear, or not really about {ticker}
   neg = bad news for {ticker} (loss, downgrade, lawsuit, price fall reported, negative outlook)
   When unsure between pos/neg and neu, choose neu.

2. Is the headline mainly about {ticker}?
   yes = {ticker} is the main subject
   partly = {ticker} is one of several stocks, or mentioned in passing
   no = about the market, sector, economy or another company

Answer with JSON: {{"sentiment": "pos" or "neu" or "neg", "relevance": "yes" or "partly" or "no"}}"""
SCHEMA = {"type": "object",
          "properties": {"sentiment": {"type": "string", "enum": S.CLASSES},
                         "relevance": {"type": "string", "enum": S.RELEVANCE}},
          "required": ["sentiment", "relevance"]}
PROMPT_SHA256 = hashlib.sha256((SYSTEM + "\n\n" + TEMPLATE).encode("utf-8")).hexdigest()

SELFTEST = [  # made up for this test, not from the sample
    ("pos", "yes", "PTT", "news", "PTT กำไรไตรมาส 2 พุ่ง 50% โบรกฯ ปรับเป้าราคาขึ้น"),
    ("neg", "yes", "PTT", "news", "PTT ขาดทุนหนักต่อเนื่อง หุ้นร่วงหลุดแนวรับ"),
    ("neu", "yes", "PTT", "news", "PTT แจ้งกำหนดประชุมผู้ถือหุ้นวันที่ 20 เมษายน"),
    ("neu", "no", "PTT", "column", "ตลาดหุ้นเอเชียเปิดบวกเล็กน้อย จับตาตัวเลขเงินเฟ้อสหรัฐคืนนี้"),
]


class Out:
    def __init__(self, P: S.Paths, model: str):
        tag = "" if model == PRIMARY_MODEL else "_" + model.split("/")[-1].replace(":", "_").replace(".", "-")
        d = P.preds.parent
        self.partial = d / f"sentiment_check_llm_predictions{tag}.partial.jsonl"
        self.csv = d / f"sentiment_check_llm_predictions{tag}.csv"
        self.meta = d / f"sentiment_check_llm_predictions{tag}.json"
        self.results = d / f"sentiment_check_llm_results{tag}.json"


def http(path: str, payload: dict | None = None, timeout: float = 600) -> dict:
    req = urllib.request.Request(OLLAMA + path, data=None if payload is None else json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def server_info(model: str) -> dict:
    try:
        version = http("/api/version", timeout=10).get("version")
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"Ollama is not reachable at {OLLAMA} ({e}). Start the Ollama app, then retry.")
    tags = http("/api/tags", timeout=30).get("models", [])
    names = {m.get("name"): m for m in tags} | {m.get("model"): m for m in tags}
    m = names.get(model) or names.get(model + ":latest")
    if m is None:
        sys.exit(f"model {model} is not pulled. Run:  ollama pull {model}")
    return {"ollama_version": version, "model": model, "model_digest": m.get("digest"), "model_size_bytes": m.get("size"),
            "model_details": m.get("details")}


def ask(model: str, ticker: str, section: str, headline: str) -> tuple[dict | None, str, float]:
    payload = {"model": model, "stream": False, "format": SCHEMA, "options": OPTIONS, "keep_alive": "30m",
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": TEMPLATE.format(ticker=ticker, section=section, headline=headline)}]}
    if "qwen3" in model:
        payload["think"] = False
    t0 = time.time()
    raw = http("/api/chat", payload).get("message", {}).get("content", "")
    secs = time.time() - t0
    try:
        obj = json.loads(raw)
        if obj.get("sentiment") in S.CLASSES and obj.get("relevance") in S.RELEVANCE:
            return {"sentiment": obj["sentiment"], "relevance": obj["relevance"]}, raw, secs
    except (json.JSONDecodeError, AttributeError):
        pass
    return None, raw, secs


def ask_with_retry(model, ticker, section, headline):
    res, raw, secs = ask(model, ticker, section, headline)
    retried = False
    if res is None:  # declared: one retry with identical settings; then counted as invalid -> neu / no
        retried = True
        res, raw, s2 = ask(model, ticker, section, headline)
        secs += s2
    valid = res is not None
    res = res or {"sentiment": "neu", "relevance": "no"}
    return {**res, "valid": valid, "retried": retried, "raw": raw, "seconds": round(secs, 2)}


def cmd_selftest(P, args):
    info = server_info(args.model)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("prompt sha256:", PROMPT_SHA256)
    ask(args.model, "PTT", "news", "ทดสอบ")  # warm-up: loads the model into memory, not timed
    times = []
    for want_s, want_r, t, sec, h in SELFTEST:
        r = ask_with_retry(args.model, t, sec, h)
        times.append(r["seconds"])
        flag = "ok" if (r["sentiment"], r["relevance"]) == (want_s, want_r) else "CHECK"
        print(f"[{flag}] expected {want_s}/{want_r:6s} got {r['sentiment']}/{r['relevance']:6s} {r['seconds']:5.1f}s valid={r['valid']}  {h}")
    hours = sum(times) / len(times) * 250 / 3600
    print(f"\nprojected time for 250 items: {hours:.2f} h")
    if args.model == PRIMARY_MODEL and hours > FALLBACK_MAX_HOURS:
        print(f"Over {FALLBACK_MAX_HOURS} h: the declared fallback applies. Use --model {FALLBACK_MODEL}")


def cmd_label(P, args):
    sample = S.load_sample(P)
    _, lmeta = S.load_frozen_labels(P)  # refuses unless frozen and unchanged
    O = Out(P, args.model)
    if O.csv.exists():
        sys.exit(f"{O.csv} already exists; refusing to relabel.")
    info = server_info(args.model)
    done = {}
    if O.partial.exists():
        for line in O.partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("model_digest") != info["model_digest"] or r.get("prompt_sha256") != PROMPT_SHA256:
                    sys.exit("partial file was made with a different model or prompt; refusing to mix.")
                done[r["id"]] = r
    started = time.time()
    with open(O.partial, "a", encoding="utf-8") as f:
        for k, s in enumerate(sample, 1):
            if s["id"] in done:
                continue
            r = {"id": s["id"], **ask_with_retry(args.model, s["ticker"], s["section"], s["headline"]),
                 "model_digest": info["model_digest"], "prompt_sha256": PROMPT_SHA256}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            done[s["id"]] = r
            el = time.time() - started
            print(f"{k:3d}/250 {s['id']} {r['sentiment']}/{r['relevance']} {r['seconds']:.1f}s  (elapsed {el / 60:.1f} min)")
    with open(O.csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "sentiment", "relevance", "valid", "retried", "seconds"])
        for s in sample:
            r = done[s["id"]]
            w.writerow([s["id"], r["sentiment"], r["relevance"], r["valid"], r["retried"], r["seconds"]])
    meta = {"labelled_at": S.now_utc(), "labels_sha256_at_label": lmeta["labels_sha256"],
            "sample_sha256": S.read_json(P.sample_meta)["sample_sha256"], "preds_sha256": S.sha256(O.csv),
            **info, "options": OPTIONS, "prompt_sha256": PROMPT_SHA256, "system": SYSTEM, "template": TEMPLATE,
            "input": "ticker + section + headline (what the human labeller saw, minus the date)",
            "n_invalid_after_retry": sum(not done[s["id"]]["valid"] for s in sample),
            "n_retried": sum(done[s["id"]]["retried"] for s in sample)}
    S.write_json(O.meta, meta)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("template", "system")}, ensure_ascii=False, indent=2))
    print(f"\nwrote {O.csv}")


def cmd_score(P, args):
    sample = S.load_sample(P)
    labels, lmeta = S.load_frozen_labels(P)
    O = Out(P, args.model)
    if not O.csv.exists():
        sys.exit("no LLM labels yet: run `label` first.")
    m = S.read_json(O.meta)
    if m["preds_sha256"] != S.sha256(O.csv) or m["labels_sha256_at_label"] != lmeta["labels_sha256"]:
        sys.exit("hash mismatch between labels, LLM output and its record; refusing.")
    if lmeta["frozen_at"] > m["labelled_at"]:
        sys.exit("human labels were frozen after the LLM output; not blind.")
    llm = {r["id"]: r for r in S.read_csv(O.csv)}
    wb = {r["id"]: r for r in S.read_csv(P.preds)} if P.preds.exists() else {}
    ids = [l["id"] for l in labels]
    hum = {l["id"]: l for l in labels}
    half_a = [i for i in ids if int(i.split("-")[1]) in S.HALF_A]
    half_b = [i for i in ids if int(i.split("-")[1]) not in S.HALF_A]

    def block(sub, ci=True, rel=False):
        key = "relevance" if rel else "sentiment"
        h = [hum[i][key] for i in sub]
        p = [llm[i][key] for i in sub]
        if rel:
            k = S.cohen_kappa(h, p, S.RELEVANCE)
            return {"n": len(sub), "kappa": round(k, 4),
                    "accuracy": round(sum(a == b for a, b in zip(h, p)) / len(sub), 4),
                    "confusion_rows_human_cols_llm": S.confusion(h, p, S.RELEVANCE)}
        return S.agreement_block(h, p, ci=ci)

    kb = block(half_b)
    sel_llm_a = block(half_a, ci=False)["kappa"]
    sel_wb_a = round(S.cohen_kappa([hum[i]["sentiment"] for i in half_a], [wb[i]["pred3"] for i in half_a]), 4) if wb else None
    res = {
        "computed_at": S.now_utc(), "declared_in": "claude/pre-registration-amendment-2026-10-06-llm-comparator.md",
        "model": m["model"], "model_digest": m["model_digest"], "prompt_sha256": m["prompt_sha256"],
        "n_invalid_after_retry": m["n_invalid_after_retry"],
        "headline_result_half_B": {**kb, "band": S.band(kb["kappa"])},
        "selection_on_half_A": {"llm_kappa": sel_llm_a, "wangchanberta_kappa": sel_wb_a,
                                "chosen": "llm" if sel_wb_a is None or sel_llm_a > sel_wb_a else "wangchanberta"},
        "half_A": block(half_a, ci=False),
        "all_250": block(ids),
        "relevance_yes_half_B": block([i for i in half_b if hum[i]["relevance"] == "yes"]),
        "relevance_yes_all": block([i for i in ids if hum[i]["relevance"] == "yes"]),
        "relevance_agreement_all_exploratory": block(ids, rel=True),
        "llm_counts": {c: sum(llm[i]["sentiment"] == c for i in ids) for c in S.CLASSES},
        "llm_relevance_counts": {c: sum(llm[i]["relevance"] == c for i in ids) for c in S.RELEVANCE},
    }
    S.write_json(O.results, res)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    lo, hi = kb["kappa_ci95_bootstrap"]
    print(f"\nHALF B (headline result): kappa = {kb['kappa']:.3f} (95% CI {lo:.3f}..{hi:.3f}), "
          f"macro-F1 = {kb['macro_f1']:.3f} -> {S.band(kb['kappa'])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--model", default=PRIMARY_MODEL, choices=[PRIMARY_MODEL, FALLBACK_MODEL])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("selftest", "label", "score"):
        sp.add_parser(c)
    args = ap.parse_args(argv)
    {"selftest": cmd_selftest, "label": cmd_label, "score": cmd_score}[args.cmd](S.Paths(args.root), args)


if __name__ == "__main__":
    main()
