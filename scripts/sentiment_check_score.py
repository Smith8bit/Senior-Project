"""Sentiment accuracy check, step 2: freeze the blind labels, run the pre-registered
model, and score how often it agrees with the human labels.

(Pre-registration item A3b; open-items register Phase 1.9.)

Specification: project doc claude/pre-registration-amendment-2026-09-30-a3b.md
(written before the sample was drawn). Order is enforced by this script:

  freeze    validate data/raw/sentiment_check/labels.csv against sample.csv and record its
            SHA-256 in LABELS.json. Refuses if any model output already exists.
  describe  label distribution and relevance share vs the pre-registered 40-60%.
            Standard library only; needs no model output.
  predict   run airesearch/wangchanberta-base-att-spm-uncased @ finetuned@wisesight_sentiment
            on the HEADLINE ONLY (no ticker). Refuses unless LABELS.json matches labels.csv.
            Needs torch, transformers, sentencepiece, pythainlp, emoji and access to huggingface.co.
  predict --selftest
            run the model on a few made-up sentences (no sample, no labels) to check the
            label mapping and the tokenizer before the real run.
  score     Cohen's kappa on 3 classes (q -> neu), macro-F1, confusion matrix, kappa on
            relevance = yes, halves A/B, bootstrap 95% CI, verdict against the bands.
            Standard library only.

  python scripts/sentiment_check_score.py freeze
  python scripts/sentiment_check_score.py describe
  python scripts/sentiment_check_score.py predict --selftest
  python scripts/sentiment_check_score.py predict
  python scripts/sentiment_check_score.py score
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = "airesearch/wangchanberta-base-att-spm-uncased"
MODEL_REVISION = "finetuned@wisesight_sentiment"
TOKENIZER_REVISION = "main"  # tokenizer files live on main, as in the model card example
WISESIGHT_ORDER = ["pos", "neu", "neg", "q"]  # ClassLabel order of the wisesight_sentiment dataset
TO3 = {"pos": "pos", "neu": "neu", "neg": "neg", "q": "neu"}  # q -> neu (pre-registered)
CLASSES = ["pos", "neu", "neg"]
RELEVANCE = ["yes", "partly", "no"]
# Landis & Koch bands as pre-registered. Banding uses the unrounded kappa against the lower
# edges, so [0.41, 0.61) is "usable" (decided before any model output existed).
BANDS = [(0.61, "use as-is"), (0.41, "usable with the limitation stated"),
         (float("-inf"), "fine-tune or demote to exploratory")]
KAPPA_PREDICTION = (0.30, 0.50)
RELEVANCE_PREDICTION = (0.40, 0.60)
HALF_A = range(1, 126)  # items a3b-001 .. a3b-125 (item ids keep the old prefix: they are in the frozen files)
SEED = 20260930
BOOTSTRAP_B = 2000

SELFTEST = [  # made up for this test, not from the sample
    ("pos", "กำไรไตรมาส 1 พุ่ง 50% โบรกฯ ปรับเป้าราคาขึ้น"),
    ("neg", "ขาดทุนหนักต่อเนื่อง หุ้นร่วงหลุดแนวรับ"),
    ("neu", "บริษัทแจ้งกำหนดประชุมผู้ถือหุ้นวันที่ 20 เมษายน"),
    ("q", "หุ้นตัวนี้น่าซื้อไหมครับ"),
]


class Paths:
    def __init__(self, root: Path):
        self.lab = root / "data" / "raw" / "sentiment_check"
        self.sample, self.sample_meta = self.lab / "sample.csv", self.lab / "SAMPLE.json"
        self.labels, self.labels_meta = self.lab / "labels.csv", self.lab / "LABELS.json"
        derived = root / "data" / "derived"
        self.preds = derived / "sentiment_check_model_predictions.csv"
        self.preds_meta = derived / "sentiment_check_model_predictions.json"
        self.describe = derived / "sentiment_check_labels_summary.json"
        self.results = derived / "sentiment_check_results.json"


# ---------------------------------------------------------------- helpers
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_sample(P: Paths) -> list[dict]:
    want = read_json(P.sample_meta)["sample_sha256"]
    got = sha256(P.sample)
    if got != want:
        sys.exit(f"sample.csv hash {got} does not match SAMPLE.json {want}; refusing.")
    return read_csv(P.sample)


def load_frozen_labels(P: Paths) -> tuple[list[dict], dict]:
    if not P.labels_meta.exists():
        sys.exit("LABELS.json missing: run `freeze` first. No model output may exist before the labels are frozen.")
    meta = read_json(P.labels_meta)
    got = sha256(P.labels)
    if got != meta["labels_sha256"]:
        sys.exit(f"labels.csv changed after freezing ({got} != {meta['labels_sha256']}); refusing.")
    return read_csv(P.labels), meta


def band(k: float) -> str:
    return next(name for lo, name in BANDS if k >= lo)


def within(x: float, lo_hi: tuple[float, float]) -> bool:
    return lo_hi[0] <= x <= lo_hi[1]


# ---------------------------------------------------------------- metrics (stdlib only)
def cohen_kappa(a: list[str], b: list[str], classes=CLASSES) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in classes)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def macro_f1(truth: list[str], pred: list[str], classes=CLASSES) -> tuple[float, dict]:
    per, f1s = {}, []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(truth, pred))
        fp = sum(t != c and p == c for t, p in zip(truth, pred))
        fn = sum(t == c and p != c for t, p in zip(truth, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": tp + fn}
    return sum(f1s) / len(classes), per


def confusion(truth: list[str], pred: list[str], classes=CLASSES) -> dict:
    """rows = human label, columns = model label"""
    return {t: {p: sum(x == t and y == p for x, y in zip(truth, pred)) for p in classes} for t in classes}


def bootstrap_ci(truth, pred, fn, B=BOOTSTRAP_B, seed=SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(truth)
    vals = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        v = fn([truth[i] for i in idx], [pred[i] for i in idx])
        if v == v:  # drop nan
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def agreement_block(truth: list[str], pred: list[str], ci: bool = True) -> dict:
    n = len(truth)
    if n == 0:
        return {"n": 0}
    k = cohen_kappa(truth, pred)
    f1, per = macro_f1(truth, pred)
    out = {"n": n, "kappa": round(k, 4), "macro_f1": round(f1, 4),
           "accuracy": round(sum(t == p for t, p in zip(truth, pred)) / n, 4)}
    if ci:
        lo, hi = bootstrap_ci(truth, pred, cohen_kappa)
        out["kappa_ci95_bootstrap"] = [round(lo, 4), round(hi, 4)]
    out["per_class"] = per
    out["confusion_rows_human_cols_model"] = confusion(truth, pred)
    return out


# ---------------------------------------------------------------- freeze
def cmd_freeze(P: Paths, args) -> None:
    sample = load_sample(P)
    if not P.labels.exists():
        sys.exit(f"{P.labels} not found. Click 'Download labels.csv' in labeling.html and save the file there.")
    labels = read_csv(P.labels)
    problems = []
    if len(labels) != len(sample):
        problems.append(f"{len(labels)} rows, expected {len(sample)}")
    for s, l in zip(sample, labels):
        if (l.get("id"), l.get("ticker"), l.get("url")) != (s["id"], s["ticker"], s["url"]):
            problems.append(f"row mismatch at {s['id']}: labels.csv has {l.get('id')} {l.get('ticker')}")
            break
    bad = [l.get("id") for l in labels if l.get("sentiment") not in CLASSES or l.get("relevance") not in RELEVANCE]
    if bad:
        more = " ..." if len(bad) > 20 else ""
        problems.append(f"{len(bad)} items lack a valid sentiment and relevance label: {', '.join(bad[:20])}{more}")
    if problems:
        sys.exit("labels.csv NOT frozen:\n  " + "\n  ".join(problems))

    digest = sha256(P.labels)
    if P.labels_meta.exists():
        old = read_json(P.labels_meta)
        if old["labels_sha256"] == digest:
            print(f"Already frozen at {old['frozen_at']}: sha256 {digest}")
            return
        sys.exit(f"LABELS.json already records a different hash ({old['labels_sha256']}). "
                 "Labels must not change after freezing; refusing.")
    if P.preds.exists() or P.preds_meta.exists():
        sys.exit("Model output already exists in data/derived; the labels can no longer be frozen blind. Stop and record this.")

    secs = [int(l["seconds"]) for l in labels if str(l.get("seconds", "")).strip().isdigit()]
    ats = sorted(l["labeled_at"] for l in labels if l.get("labeled_at"))
    meta = {
        "frozen_at": now_utc(),
        "labels_sha256": digest,
        "sample_sha256": read_json(P.sample_meta)["sample_sha256"],
        "n": len(labels),
        "sentiment_counts": {c: sum(l["sentiment"] == c for l in labels) for c in CLASSES},
        "relevance_counts": {c: sum(l["relevance"] == c for l in labels) for c in RELEVANCE},
        "labeled_at_earliest_last_edit": ats[0] if ats else None,
        "labeled_at_latest_last_edit": ats[-1] if ats else None,
        "seconds_on_page_total": sum(secs),
        "seconds_on_page_median": statistics.median(secs) if secs else None,
        "model_output_existed_at_freeze": False,
    }
    write_json(P.labels_meta, meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nFROZEN. labels.csv sha256 = {digest}")


# ---------------------------------------------------------------- describe
def describe(sample: list[dict], labels: list[dict]) -> dict:
    n = len(labels)
    yes = sum(l["relevance"] == "yes" for l in labels)
    by_section: dict[str, dict] = {}
    for s, l in zip(sample, labels):
        d = by_section.setdefault(s["section"], {"n": 0, "relevance_yes": 0, "pos": 0, "neu": 0, "neg": 0})
        d["n"] += 1
        d["relevance_yes"] += l["relevance"] == "yes"
        d[l["sentiment"]] += 1
    cross = {r: {c: sum(l["relevance"] == r and l["sentiment"] == c for l in labels) for c in CLASSES} for r in RELEVANCE}
    share = yes / n
    return {
        "n": n,
        "sentiment_share": {c: round(sum(l["sentiment"] == c for l in labels) / n, 4) for c in CLASSES},
        "relevance_share": {r: round(sum(l["relevance"] == r for l in labels) / n, 4) for r in RELEVANCE},
        "relevance_yes_share": round(share, 4),
        "relevance_yes_prediction": list(RELEVANCE_PREDICTION),
        "relevance_yes_within_prediction": within(share, RELEVANCE_PREDICTION),
        "crosstab_relevance_by_sentiment": cross,
        "by_section": dict(sorted(by_section.items(), key=lambda kv: -kv[1]["n"])),
    }


def cmd_describe(P: Paths, args) -> None:
    sample = load_sample(P)
    labels, lmeta = load_frozen_labels(P)
    out = {"computed_at": now_utc(), "labels_sha256": lmeta["labels_sha256"], **describe(sample, labels)}
    write_json(P.describe, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- predict
# Preprocessing vendored from thai2transformers 0.1.2, thai2transformers/preprocess.py
# (Apache License 2.0, VISTEC-depa AI Research Institute of Thailand): process_transformers,
# the preprocessing used for WangchanBERTa's fine-tuning.
_TK_URL, _TK_END = "<url>", "</s>"
SPACE_TOKEN = "<_>"


def _fix_html(text: str) -> str:
    text = (text.replace("#39;", "'").replace("amp;", "&").replace("#146;", "'")
            .replace("nbsp;", " ").replace("#36;", "$").replace("\\n", "\n").replace("quot;", "'")
            .replace("<br />", "\n").replace('\\"', '"').replace(" @.@ ", ".").replace(" @-@ ", "-")
            .replace(" @,@ ", ",").replace("\\", " \\ "))
    return re.sub(r"  +", " ", html.unescape(text))


def _rm_brackets(t: str) -> str:
    t = re.sub(r"\(\)", "", t); t = re.sub(r"\{\}", "", t); t = re.sub(r"\[\]", "", t)
    t = re.sub(r"\([^a-zA-Z0-9ก-๙]+\)", "", t)
    t = re.sub(r"\{[^a-zA-Z0-9ก-๙]+\}", "", t)
    t = re.sub(r"\[[^a-zA-Z0-9ก-๙]+\]", "", t)
    t = re.sub(r"(?<=\()[^a-zA-Z0-9ก-๙]+(?=[a-zA-Z0-9ก-๙])", "", t)
    t = re.sub(r"(?<=\{)[^a-zA-Z0-9ก-๙]+(?=[a-zA-Z0-9ก-๙])", "", t)
    t = re.sub(r"(?<=\[)[^a-zA-Z0-9ก-๙]+(?=[a-zA-Z0-9ก-๙])", "", t)
    t = re.sub(r"(?<=[a-zA-Z0-9ก-๙])[^a-zA-Z0-9ก-๙]+(?=\))", "", t)
    t = re.sub(r"(?<=[a-zA-Z0-9ก-๙])[^a-zA-Z0-9ก-๙]+(?=\})", "", t)
    t = re.sub(r"(?<=[a-zA-Z0-9ก-๙])[^a-zA-Z0-9ก-๙]+(?=\])", "", t)
    return t


def _replace_rep_after(t: str) -> str:
    return re.compile(r"(\S)(\1{3,})").sub(lambda m: m.group(1), t)


def _ungroup_emoji(toks):
    import emoji
    res = []
    for tok in toks:
        res.extend(list(tok)) if emoji.emoji_count(tok) == len(tok) else res.append(tok)
    return res


def _replace_wrep_post(toks):
    prev, rep, res = None, 0, []
    for cur in list(toks) + [_TK_END]:
        if cur == prev:
            rep += 1
        elif cur != prev and rep > 0:
            res += [prev]
            rep = 0
        else:
            res.append(prev)
        prev = cur
    return res[1:]


def process_transformers(text: str) -> str:
    from pythainlp.tokenize import word_tokenize
    text = text.lower()
    for rule in (_fix_html, _rm_brackets, lambda t: re.sub(r"[\n]", " ", t.strip()),
                 lambda t: re.sub(" {2,}", " ", t), lambda t: re.sub(" ", SPACE_TOKEN, t), _replace_rep_after):
        text = rule(text)
    toks = word_tokenize(text)
    toks = _replace_wrep_post(_ungroup_emoji(toks))
    return "".join(toks)


def load_model():
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=TOKENIZER_REVISION)
    vocab = tok.get_vocab()
    if SPACE_TOKEN not in vocab:
        sys.exit(f"tokenizer vocab has no {SPACE_TOKEN}; wrong tokenizer.")
    if SPACE_TOKEN not in tok.tokenize(f"ก{SPACE_TOKEN}ข"):
        extra = [t for t in ("<s>NOTUSED", "</s>NOTUSED", SPACE_TOKEN) if t in vocab]
        tok.add_special_tokens({"additional_special_tokens": extra})  # all already in vocab: no resize
        if SPACE_TOKEN not in tok.tokenize(f"ก{SPACE_TOKEN}ข"):
            sys.exit(f"tokenizer splits {SPACE_TOKEN}; refusing to score with broken spacing.")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.eval()

    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    names = [id2label[i] for i in range(len(id2label))]
    if set(names) == set(WISESIGHT_ORDER):
        mapping_source = "config.id2label"
    elif len(names) == 4 and names == [f"label_{i}" for i in range(4)]:
        names = list(WISESIGHT_ORDER)
        mapping_source = "ASSUMED wisesight_sentiment ClassLabel order (config labels are generic); check --selftest"
    else:
        sys.exit(f"unexpected labels in config: {names}")

    commit = getattr(model.config, "_commit_hash", None)
    if not commit:  # transformers 5 may not set it; read the branch ref from the local Hugging Face cache
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            ref = Path(HF_HUB_CACHE) / f"models--{MODEL_ID.replace('/', '--')}" / "refs" / MODEL_REVISION
            commit = ref.read_text(encoding="utf-8").strip() if ref.exists() else None
        except Exception:
            commit = None
    info = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "model_commit": commit,
        "tokenizer_revision": TOKENIZER_REVISION, "tokenizer_class": type(tok).__name__,
        "id2label_raw": {str(k): v for k, v in model.config.id2label.items()},
        "label_names_used": names, "label_mapping_source": mapping_source, "to3": TO3,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "preprocessing": "thai2transformers 0.1.2 process_transformers (vendored), then headline only",
        "input": "headline only; no ticker or article text",
    }
    try:
        import pythainlp
        import sentencepiece
        info["versions"].update(pythainlp=pythainlp.__version__, sentencepiece=sentencepiece.__version__)
    except Exception:
        pass
    return tok, model, names, info


def run_model(tok, model, names, texts, batch=32):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=416, return_tensors="pt")
            probs = model(**enc).logits.softmax(-1).tolist()
            for p in probs:
                raw = names[max(range(len(p)), key=p.__getitem__)]
                out.append({"raw": raw, "pred3": TO3[raw], **{f"p_{n}": round(v, 6) for n, v in zip(names, p)}})
    return out


def cmd_predict(P: Paths, args) -> None:
    if args.selftest:
        tok, model, names, info = load_model()
        texts = [process_transformers(t) for _, t in SELFTEST]
        print("label mapping:", info["label_names_used"], "-", info["label_mapping_source"])
        print("model commit:", info["model_commit"])
        print("tokens of first sentence:", tok.tokenize(texts[0]))
        for (want, raw), proc, r in zip(SELFTEST, texts, run_model(tok, model, names, texts)):
            flag = "ok" if r["raw"] == want else "CHECK"
            print(f"[{flag}] expected {want:3s} got {r['raw']:3s}  {raw}  ->  {proc}")
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return

    sample = load_sample(P)
    _, lmeta = load_frozen_labels(P)  # refuses unless frozen and unchanged
    if P.preds.exists() and not args.force:
        sys.exit(f"{P.preds} exists; use --force to overwrite.")
    tok, model, names, info = load_model()
    texts = [process_transformers(s["headline"]) for s in sample]
    rows = run_model(tok, model, names, texts)
    P.preds.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "raw", "pred3"] + [f"p_{n}" for n in names]
    with open(P.preds, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s, r in zip(sample, rows):
            w.writerow({"id": s["id"], **r})
    meta = {"predicted_at": now_utc(), "labels_sha256_at_predict": lmeta["labels_sha256"],
            "sample_sha256": read_json(P.sample_meta)["sample_sha256"],
            "preds_sha256": sha256(P.preds), "n": len(rows), **info}
    write_json(P.preds_meta, meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nwrote {P.preds}")


# ---------------------------------------------------------------- score
def cmd_score(P: Paths, args) -> None:
    sample = load_sample(P)
    labels, lmeta = load_frozen_labels(P)
    if not P.preds.exists():
        sys.exit("no model predictions yet: run `predict` first.")
    pmeta = read_json(P.preds_meta)
    if pmeta["preds_sha256"] != sha256(P.preds):
        sys.exit("predictions file changed after it was written; refusing.")
    if pmeta["labels_sha256_at_predict"] != lmeta["labels_sha256"]:
        sys.exit("predictions were made against a different labels.csv; refusing.")
    if lmeta["frozen_at"] > pmeta["predicted_at"]:
        sys.exit("labels were frozen AFTER predictions were made; the test is not blind.")
    preds = {r["id"]: r for r in read_csv(P.preds)}
    ids = [l["id"] for l in labels]
    if set(ids) != set(preds):
        sys.exit("prediction ids do not match label ids.")

    human = [l["sentiment"] for l in labels]
    model = [preds[i]["pred3"] for i in ids]
    rel = [l["relevance"] for l in labels]
    num = [int(i.split("-")[1]) for i in ids]

    overall = agreement_block(human, model)
    k = overall["kappa"]
    sub = lambda mask: ([h for h, m_ in zip(human, mask) if m_], [m for m, m_ in zip(model, mask) if m_])
    results = {
        "computed_at": now_utc(),
        "inputs": {"sample_sha256": pmeta["sample_sha256"], "labels_sha256": lmeta["labels_sha256"],
                   "preds_sha256": pmeta["preds_sha256"], "model": f"{MODEL_ID}@{MODEL_REVISION}",
                   "model_commit": pmeta.get("model_commit"), "label_mapping_source": pmeta.get("label_mapping_source")},
        "primary": {
            **overall,
            "band": band(k),
            "bands": "kappa >= 0.61 use as-is; 0.41 <= kappa < 0.61 usable with limitation; < 0.41 fine-tune or demote",
            "kappa_prediction": list(KAPPA_PREDICTION),
            "kappa_within_prediction": within(k, KAPPA_PREDICTION),
        },
        "relevance_yes": agreement_block(*sub([r == "yes" for r in rel])),
        "by_relevance": {r: agreement_block(*sub([x == r for x in rel]), ci=False) for r in RELEVANCE},
        "half_A": agreement_block(*sub([n in HALF_A for n in num]), ci=False),
        "half_B": agreement_block(*sub([n not in HALF_A for n in num]), ci=False),
        "model_raw_counts": {c: sum(preds[i]["raw"] == c for i in ids) for c in WISESIGHT_ORDER},
        "human_counts": {c: human.count(c) for c in CLASSES},
        "model_counts": {c: model.count(c) for c in CLASSES},
        "describe": describe(sample, labels),
    }
    write_json(P.results, results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    lo, hi = overall["kappa_ci95_bootstrap"]
    print(f"\nPRIMARY: kappa = {k:.3f} (95% bootstrap CI {lo:.3f}..{hi:.3f}), macro-F1 = {overall['macro_f1']:.3f}"
          f" -> {band(k)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="repo root (for tests)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("freeze")
    sp.add_parser("describe")
    p = sp.add_parser("predict")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--force", action="store_true")
    sp.add_parser("score")
    args = ap.parse_args(argv)
    P = Paths(args.root)
    {"freeze": cmd_freeze, "describe": cmd_describe, "predict": cmd_predict, "score": cmd_score}[args.cmd](P, args)


if __name__ == "__main__":
    main()
