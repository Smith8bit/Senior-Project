"""Fine-tune WangchanBERTa on Siwat's 1,000 hand-labelled finance headlines, then sit the exam once.

Rules: project doc claude/pre-registration-amendment-2026-10-06-finetune.md (fixed before any
training label existed). In short:
  * training labels: data/raw/sentiment_training/ (frozen by `freeze-train` before any run);
  * 80/20 train/validation split of those labels, stratified, seed 20261006;
  * 4 configurations (start from the Wisesight sentiment checkpoint or the plain model) x
    (headline only, or ticker + section + headline), 3 seeds each, fixed settings;
  * the configuration with the best mean validation macro-F1 is selected (`select`);
  * the exam (the 250 frozen labels in data/raw/sentiment_check/) is scored once (`exam`).
Exam LABELS are read only by `exam`. During training each run predicts the exam HEADLINES
(inputs only) and seals the predictions with a SHA-256 in runs.jsonl; `exam` scores only the
selected configuration's sealed files.

  python scripts/sentiment_finetune.py import-store --json-dir <export dir>   # labels from the labelling page
  python scripts/sentiment_finetune.py freeze-train
  python scripts/sentiment_finetune.py train            # all 12 runs; resumable; add --configs to split sessions
  python scripts/sentiment_finetune.py select
  python scripts/sentiment_finetune.py exam
  add --target relevance  for the exploratory relevance classifier (same steps)

Needs torch, transformers, sentencepiece, pythainlp, emoji, and access to huggingface.co (Colab or the PC).
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sentiment_check_score as S  # noqa: E402

MODEL_ID = S.MODEL_ID
INITS = {"wisesight": S.MODEL_REVISION, "base": "main"}
INPUTS = ["headline", "ticker_section_headline"]
CONFIGS = [f"{i}+{x}" for i in INITS for x in INPUTS]          # 4 configurations
SEEDS = [20261006, 20261007, 20261008]
SPLIT_SEED = 20261006
VAL_SHARE = 0.20
SETTINGS = {"lr": 2e-5, "weight_decay": 0.01, "batch_size": 16, "max_epochs": 5, "patience": 2,
            "warmup_share": 0.10, "max_length": 128, "loss": "class-weighted cross-entropy (inverse train frequency)",
            "select_on": "validation macro-F1, mean over 3 seeds"}
TARGETS = {"sentiment": S.CLASSES, "relevance": S.RELEVANCE}
ABOUT_STOCK = ("yes", "partly")   # primary exam subset, amendment 2026-10-15 (labelling-standard shift on off-topic items)
TEST_MODE = os.environ.get("SENTIMENT_FT_TEST") == "1"           # tests only: tiny local models, 2 epochs
LOCAL_MODELS = os.environ.get("SENTIMENT_FT_LOCAL_MODELS")       # tests only: <dir>/{tokenizer,wisesight,base}


class TrainPaths:
    """The training-label folder, shaped like sentiment_check_score.Paths so its freeze can be reused."""
    def __init__(self, root: Path):
        self.root = root
        self.lab = root / "data" / "raw" / "sentiment_training"
        self.exam_sample = root / "data" / "raw" / "sentiment_check" / "sample.csv"
        self.sample, self.sample_meta = self.lab / "sample.csv", self.lab / "SAMPLE.json"
        self.labels, self.labels_meta = self.lab / "labels.csv", self.lab / "LABELS.json"
        self.out = root / "data" / "derived" / "sentiment_finetune"
        self.runs = self.out / "runs.jsonl"
        self.sealed = self.out / "sealed_exam_predictions"
        self.models = root / "models" / "sentiment_finetune"
        # freeze() refuses if any model output already exists:
        self.preds, self.preds_meta = self.runs, self.out / "selection_sentiment.json"


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ labels in
def cmd_import_store(T: TrainPaths, args):
    """Turn the labelling page's store export (one JSON per item) into labels.csv in the exam's format."""
    if T.labels.exists():
        sys.exit(f"{T.labels} already exists; refusing to overwrite.")
    sample = S.load_sample(T)
    docs = {}
    for p in Path(args.json_dir).rglob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        body = d.get("data", d) if isinstance(d, dict) else {}
        docs[d.get("id", p.stem)] = body
    rows = [["id", "ticker", "url", "sentiment", "relevance", "labeled_at", "seconds"]]
    for s in sample:
        b = docs.get(s["id"], {})
        if b and b.get("t") not in (None, s["ticker"]):
            sys.exit(f"ticker mismatch for {s['id']}: store has {b.get('t')}, sample has {s['ticker']}")
        rows.append([s["id"], s["ticker"], s["url"], b.get("s") or "", b.get("r") or "", b.get("at") or "", b.get("sec") or ""])
    text = "\r\n".join(",".join('"' + str(v).replace('"', '""') + '"' for v in r) for r in rows)
    T.labels.write_bytes(("﻿" + text).encode("utf-8"))
    n = sum(1 for r in rows[1:] if r[3] and r[4])
    print(f"wrote {T.labels}: {n} of {len(sample)} items fully labelled; {len(docs)} store documents read")


def cmd_freeze_train(T: TrainPaths, args):
    S.cmd_freeze(T, args)


def load_training(T: TrainPaths):
    sample = S.load_sample(T)
    labels, meta = S.load_frozen_labels(T)
    by = {s["id"]: s for s in sample}
    return [{**by[l["id"]], "sentiment": l["sentiment"], "relevance": l["relevance"]} for l in labels], meta


def split(items: list[dict], target: str):
    """80/20 stratified by the target label, fixed seed; identical for every run."""
    rng = random.Random(SPLIT_SEED)
    train, val = [], []
    for c in TARGETS[target]:
        group = sorted((x for x in items if x[target] == c), key=lambda x: x["id"])
        rng.shuffle(group)
        k = round(len(group) * VAL_SHARE)
        val += group[:k]
        train += group[k:]
    return sorted(train, key=lambda x: x["id"]), sorted(val, key=lambda x: x["id"])


def text_of(x: dict, inp: str) -> str:
    t = x["headline"] if inp == "headline" else f"{x['ticker']} {x['section']} {x['headline']}"
    return S.process_transformers(t)


# ------------------------------------------------------------------ model
def load_tokenizer():
    from transformers import AutoTokenizer
    src = (Path(LOCAL_MODELS) / "tokenizer",) if LOCAL_MODELS else (MODEL_ID,)
    tok = AutoTokenizer.from_pretrained(*src, **({} if LOCAL_MODELS else {"revision": "main"}))
    if S.SPACE_TOKEN not in tok.get_vocab():
        sys.exit("tokenizer has no <_> token; wrong tokenizer.")
    if S.SPACE_TOKEN not in tok.tokenize(f"ก{S.SPACE_TOKEN}ข"):
        extra = [t for t in ("<s>NOTUSED", "</s>NOTUSED", S.SPACE_TOKEN) if t in tok.get_vocab()]
        tok.add_special_tokens({"additional_special_tokens": extra})
        if S.SPACE_TOKEN not in tok.tokenize(f"ก{S.SPACE_TOKEN}ข"):
            sys.exit("tokenizer splits <_>; refusing to train with broken spacing.")
    return tok


def load_model(init: str, target: str):
    """Returns (model, index of each target class in the model's logits)."""
    from transformers import AutoModelForSequenceClassification
    classes = TARGETS[target]
    if LOCAL_MODELS:
        src, kw = (str(Path(LOCAL_MODELS) / init),), {}
    else:
        src, kw = (MODEL_ID,), {"revision": INITS[init]}
    if init == "wisesight" and target == "sentiment":
        # keep the pretrained 4-way head (pos, neu, neg, q); train and predict on pos/neu/neg only
        model = AutoModelForSequenceClassification.from_pretrained(*src, **kw)
        names = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
        if not set(classes) <= set(names):
            sys.exit(f"unexpected labels in the Wisesight head: {model.config.id2label}")
        return model, [names[c] for c in classes]
    model = AutoModelForSequenceClassification.from_pretrained(
        *src, **kw, num_labels=len(classes), ignore_mismatched_sizes=True,
        id2label=dict(enumerate(classes)), label2id={c: i for i, c in enumerate(classes)})
    return model, list(range(len(classes)))


def encode(tok, texts):
    return tok(texts, padding=True, truncation=True, max_length=SETTINGS["max_length"], return_tensors="pt")


def predict(model, idx, tok, texts, device, batch=64):
    import torch
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = {k: v.to(device) for k, v in encode(tok, texts[i:i + batch]).items()}
            logits = model(**enc).logits[:, idx]
            probs += logits.softmax(-1).cpu().tolist()
    return probs


def metrics(truth, pred, classes):
    f1, _ = S.macro_f1(truth, pred, classes)
    return {"macro_f1": round(f1, 4), "kappa": round(S.cohen_kappa(truth, pred, classes), 4),
            "accuracy": round(sum(a == b for a, b in zip(truth, pred)) / len(truth), 4)}


def train_one(cfg: str, seed: int, target: str, train, val, exam_inputs, tok, device, T: TrainPaths, lmeta, save_model):
    import numpy as np
    import torch
    init, inp = cfg.split("+")
    classes = TARGETS[target]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, idx = load_model(init, target)
    model.to(device)

    xs = [text_of(x, inp) for x in train]
    ys = torch.tensor([classes.index(x[target]) for x in train])
    counts = [max(1, int((ys == k).sum())) for k in range(len(classes))]
    weights = torch.tensor([len(ys) / (len(classes) * c) for c in counts], dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    vx = [text_of(x, inp) for x in val]
    vy = [x[target] for x in val]

    max_epochs = 2 if TEST_MODE else SETTINGS["max_epochs"]
    bs = SETTINGS["batch_size"]
    steps_per_epoch = math.ceil(len(xs) / bs)
    total = steps_per_epoch * max_epochs
    warm = max(1, int(SETTINGS["warmup_share"] * total))
    opt = torch.optim.AdamW(model.parameters(), lr=SETTINGS["lr"] * (50 if TEST_MODE else 1),
                            weight_decay=SETTINGS["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    gen = torch.Generator().manual_seed(seed)

    best, best_state, best_epoch, history, since = -1.0, None, 0, [], 0
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(xs), generator=gen).tolist()
        total_loss = 0.0
        for b in range(0, len(order), bs):
            ids = order[b:b + bs]
            enc = {k: v.to(device) for k, v in encode(tok, [xs[i] for i in ids]).items()}
            logits = model(**enc).logits[:, idx]
            loss = loss_fn(logits, ys[ids].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total_loss += float(loss) * len(ids)
        vp = [classes[max(range(len(p)), key=p.__getitem__)] for p in predict(model, idx, tok, vx, device)]
        m = {"epoch": epoch, "train_loss": round(total_loss / len(xs), 4), **metrics(vy, vp, classes)}
        history.append(m)
        print(f"  {target} {cfg} seed {seed} epoch {epoch}: loss {m['train_loss']:.3f}  val macro-F1 {m['macro_f1']:.3f}  kappa {m['kappa']:.3f}")
        if m["macro_f1"] > best:
            best, best_epoch, since = m["macro_f1"], epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            since += 1
            if since >= SETTINGS["patience"]:
                break
    model.load_state_dict(best_state)

    # seal predictions on the exam HEADLINES (inputs only; exam labels are not read here)
    run_id = f"{target}__{cfg}__seed{seed}"
    T.sealed.mkdir(parents=True, exist_ok=True)
    ep = predict(model, idx, tok, [text_of(x, inp) for x in exam_inputs], device)
    sealed = T.sealed / f"{run_id}.csv"
    with open(sealed, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pred"] + [f"p_{c}" for c in classes])
        for x, p in zip(exam_inputs, ep):
            w.writerow([x["id"], classes[max(range(len(p)), key=p.__getitem__)]] + [round(v, 6) for v in p])
    model_dir = None
    if save_model:
        model_dir = T.models / target / cfg / f"seed{seed}"
        model.save_pretrained(model_dir); tok.save_pretrained(model_dir)

    import transformers
    return {"run_id": run_id, "target": target, "config": cfg, "init": init, "input": inp, "seed": seed,
            "best_epoch": best_epoch, "best_val": next(h for h in history if h["epoch"] == best_epoch),
            "history": history, "n_train": len(train), "n_val": len(val), "class_counts_train": dict(zip(classes, counts)),
            "class_weights": [round(float(w), 4) for w in weights], "settings": SETTINGS, "test_mode": TEST_MODE,
            "labels_sha256": lmeta["labels_sha256"], "sealed_exam_file": sealed.name, "sealed_exam_sha256": S.sha256(sealed),
            "model_dir": str(model_dir) if model_dir else None, "seconds": round(time.time() - t0, 1), "device": str(device),
            "git_commit": git_commit(T.root), "finished_at": S.now_utc(),
            "versions": {"torch": __import__("torch").__version__, "transformers": transformers.__version__}}


def read_runs(T: TrainPaths) -> list[dict]:
    if not T.runs.exists():
        return []
    return [json.loads(l) for l in T.runs.read_text(encoding="utf-8").splitlines() if l.strip()]


def cmd_train(T: TrainPaths, args):
    import torch
    items, lmeta = load_training(T)                      # refuses unless frozen
    if (T.out / f"selection_{args.target}.json").exists():
        sys.exit("selection already made for this target; training more runs now would bias the choice.")
    exam_inputs = S.read_csv(T.exam_sample)              # exam headlines only; exam labels are never read here
    train, val = split(items, args.target)
    done = {(r["config"], r["seed"]) for r in read_runs(T) if r["target"] == args.target}
    old = {r["labels_sha256"] for r in read_runs(T)}
    if old and old != {lmeta["labels_sha256"]}:
        sys.exit("runs.jsonl holds runs made with different training labels; refusing to mix.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer()
    configs = args.configs or CONFIGS
    print(f"{args.target}: train {len(train)} / val {len(val)} on {device}; configs {configs}")
    T.out.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        if cfg not in CONFIGS:
            sys.exit(f"unknown config {cfg}; choose from {CONFIGS}")
        for seed in SEEDS:
            if (cfg, seed) in done:
                print(f"  skip {cfg} seed {seed} (done)")
                continue
            rec = train_one(cfg, seed, args.target, train, val, exam_inputs, tok, device, T, lmeta, args.save_models)
            with open(T.runs, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def cmd_select(T: TrainPaths, args):
    runs = [r for r in read_runs(T) if r["target"] == args.target]
    have = {(r["config"], r["seed"]) for r in runs}
    missing = [(c, s) for c in CONFIGS for s in SEEDS if (c, s) not in have]
    if missing:
        sys.exit(f"{len(missing)} runs missing, e.g. {missing[:3]}; finish `train` first.")
    out = T.out / f"selection_{args.target}.json"
    if out.exists():
        sys.exit(f"{out} already exists; the selection is made once.")
    table = []
    for c in CONFIGS:
        rs = [r for r in runs if r["config"] == c]
        f1 = [r["best_val"]["macro_f1"] for r in rs]
        k = [r["best_val"]["kappa"] for r in rs]
        table.append({"config": c, "val_macro_f1_mean": round(statistics.mean(f1), 4), "val_macro_f1_std": round(statistics.stdev(f1), 4),
                      "val_kappa_mean": round(statistics.mean(k), 4), "seeds": [r["seed"] for r in rs]})
    chosen = max(table, key=lambda t: (t["val_macro_f1_mean"], t["val_kappa_mean"], -CONFIGS.index(t["config"])))
    sel = {"selected_at": S.now_utc(), "target": args.target, "rule": SETTINGS["select_on"], "chosen": chosen["config"],
           "table": table, "labels_sha256": runs[0]["labels_sha256"]}
    S.write_json(out, sel)
    print(json.dumps(sel, ensure_ascii=False, indent=2))


def cmd_exam(T: TrainPaths, args):
    sel_path = T.out / f"selection_{args.target}.json"
    if not sel_path.exists():
        sys.exit("no selection yet: run `select` first. The exam is scored only for the selected configuration.")
    res_path = T.out / f"exam_results_{args.target}.json"
    if res_path.exists():
        sys.exit(f"{res_path} already exists; the exam is sat once.")
    sel = S.read_json(sel_path)
    E = S.Paths(T.root)
    S.load_sample(E)
    labels, emeta = S.load_frozen_labels(E)
    hum = {l["id"]: l for l in labels}
    classes = TARGETS[args.target]
    runs = [r for r in read_runs(T) if r["target"] == args.target and r["config"] == sel["chosen"]]
    per_seed, preds_by_seed = [], {}
    for r in sorted(runs, key=lambda r: r["seed"]):
        f = T.sealed / r["sealed_exam_file"]
        if S.sha256(f) != r["sealed_exam_sha256"]:
            sys.exit(f"sealed exam predictions {f.name} changed after training; refusing.")
        p = {row["id"]: row["pred"] for row in S.read_csv(f)}
        ids = [l["id"] for l in labels]
        truth, pred = [hum[i][args.target] for i in ids], [p[i] for i in ids]
        blk = S.agreement_block(truth, pred) if args.target == "sentiment" else \
            {"n": len(ids), "kappa": round(S.cohen_kappa(truth, pred, classes), 4), **metrics(truth, pred, classes)}
        if args.target == "sentiment":
            # Primary since amendment 2026-10-15: items the exam labelled about the stock (yes/partly).
            about = [i for i in ids if hum[i]["relevance"] in ABOUT_STOCK]
            blk["about_stock"] = S.agreement_block([hum[i]["sentiment"] for i in about], [p[i] for i in about])
            yes = [i for i in ids if hum[i]["relevance"] == "yes"]
            blk["relevance_yes"] = metrics([hum[i]["sentiment"] for i in yes], [p[i] for i in yes], classes) | {"n": len(yes)}
            for name, sub in (("half_A", [i for i in ids if int(i.split("-")[1]) in S.HALF_A]),
                              ("half_B", [i for i in ids if int(i.split("-")[1]) not in S.HALF_A])):
                blk[name] = metrics([hum[i]["sentiment"] for i in sub], [p[i] for i in sub], classes) | {"n": len(sub)}
        per_seed.append({"seed": r["seed"], **blk})
        preds_by_seed[r["seed"]] = pred
    ks_all = [s["kappa"] for s in per_seed]
    if args.target == "sentiment":   # amendment 2026-10-15: bands apply to the about-the-stock subset
        primary = "about_stock (exam relevance yes or partly)"
        ks = [s["about_stock"]["kappa"] for s in per_seed]
        f1s = [s["about_stock"]["macro_f1"] for s in per_seed]
    else:
        primary, ks, f1s = "all items", ks_all, [s["macro_f1"] for s in per_seed]
    mean_k = statistics.mean(ks)
    res = {"computed_at": S.now_utc(), "target": args.target, "config": sel["chosen"],
           "exam_labels_sha256": emeta["labels_sha256"], "primary_measure": primary,
           "primary_n": per_seed[0]["about_stock"]["n"] if args.target == "sentiment" else per_seed[0]["n"],
           "kappa_mean": round(mean_k, 4), "kappa_std": round(statistics.stdev(ks), 4),
           "macro_f1_mean": round(statistics.mean(f1s), 4),
           "kappa_mean_all_items": round(statistics.mean(ks_all), 4), "kappa_std_all_items": round(statistics.stdev(ks_all), 4),
           "band_on_mean_kappa": S.band(mean_k) if args.target == "sentiment" else "exploratory (no bands)",
           "prediction_0.45_0.65": (0.45 <= mean_k <= 0.65) if args.target == "sentiment" else None,
           "per_seed": per_seed}
    S.write_json(res_path, res)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\nEXAM ({args.target}, {sel['chosen']}), primary = {primary}: kappa {mean_k:.3f} ± {statistics.stdev(ks):.3f} over {len(ks)} seeds"
          + (f" -> {S.band(mean_k)}" if args.target == "sentiment" else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--target", choices=list(TARGETS), default="sentiment")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("import-store"); p.add_argument("--json-dir", required=True)
    sp.add_parser("freeze-train")
    p = sp.add_parser("train")
    p.add_argument("--configs", nargs="*", help=f"subset of {CONFIGS} (to split work across sessions)")
    p.add_argument("--save-models", action="store_true", help="also save each run's best weights under models/ (about 420 MB each)")
    sp.add_parser("select")
    sp.add_parser("exam")
    args = ap.parse_args(argv)
    T = TrainPaths(args.root)
    {"import-store": cmd_import_store, "freeze-train": cmd_freeze_train, "train": cmd_train,
     "select": cmd_select, "exam": cmd_exam}[args.cmd](T, args)


if __name__ == "__main__":
    main()
