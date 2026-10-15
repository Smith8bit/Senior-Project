"""Offline tests for scripts/sentiment_finetune.py with tiny stand-in models (no download, no real data).

Run from the repo root:  python tests/test_sentiment_finetune.py
Needs torch, transformers, pythainlp, emoji (skips otherwise). Uses a tiny tokenizer and tiny Camembert
models built on the fly, 2 epochs per run (test mode only), so the whole file runs in about a minute on CPU.
"""
import csv
import hashlib
import io
import json
import statistics
import os
import random
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import pythainlp  # noqa: F401
except ImportError as e:
    print(f"skip: {e}")
    sys.exit(0)

WORDS = {"pos": ["กำไรพุ่ง", "หุ้นขึ้น", "โบรกฯปรับเป้าขึ้น"], "neu": ["ประชุมผู้ถือหุ้น", "แจ้งวันปิดสมุด", "ตลาดทรงตัว"],
         "neg": ["ขาดทุนหนัก", "หุ้นร่วง", "ถูกฟ้องร้อง"]}


def build_models(d: Path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import (CamembertConfig, CamembertForMaskedLM, CamembertForSequenceClassification,
                              PreTrainedTokenizerFast)
    corpus = [" ".join(w) for ws in WORDS.values() for w in ws] + ["<_>".join(sum(WORDS.values(), []))] * 20
    tk = Tokenizer(models.WordPiece(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split("<_>", behavior="isolated")
    tk.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=300, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<_>"]))
    tk.post_processor = processors.TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                  pad_token="<pad>", mask_token="<mask>", cls_token="<s>", sep_token="</s>")
    tok.save_pretrained(d / "tokenizer")
    cfg = dict(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
               max_position_embeddings=140, pad_token_id=1, bos_token_id=0, eos_token_id=2)
    torch.manual_seed(0)
    CamembertForSequenceClassification(CamembertConfig(**cfg, num_labels=4, id2label={0: "pos", 1: "neu", 2: "neg", 3: "q"},
                                                       label2id={"pos": 0, "neu": 1, "neg": 2, "q": 3})).save_pretrained(d / "wisesight")
    CamembertForMaskedLM(CamembertConfig(**cfg)).save_pretrained(d / "base")


def write_sample(folder: Path, prefix: str, n: int, seed: int):
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    rows, truth = [], {}
    for k in range(1, n + 1):
        c = rng.choice(["pos", "neu", "neg"])
        rel = rng.choice(["yes", "partly", "no"])
        t = rng.choice(["PTT", "KBANK", "SCB"])
        rid = f"{prefix}-{k:04d}" if prefix == "train" else f"a3b-{k:03d}"
        rows.append({"id": rid, "ticker": t, "url": f"https://example.invalid/{prefix}/{k}", "section": "news",
                     "published": "2024-01-02T10:00:00+07:00", "trading_day": "2024-01-02",
                     "headline": f"{t} {rng.choice(WORDS[c])} {k}"})
        truth[rid] = (c, rel)
    with open(folder / "sample.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (folder / "SAMPLE.json").write_text(json.dumps({"sample_sha256": hashlib.sha256((folder / "sample.csv").read_bytes()).hexdigest()}), encoding="utf-8")
    return rows, truth


def write_labels_csv(folder: Path, rows, truth):
    lines = [["id", "ticker", "url", "sentiment", "relevance", "labeled_at", "seconds"]]
    lines += [[r["id"], r["ticker"], r["url"], *truth[r["id"]], "2026-10-06T08:00:00Z", "5"] for r in rows]
    text = "\r\n".join(",".join('"' + v + '"' for v in row) for row in lines)
    (folder / "labels.csv").write_bytes(("﻿" + text).encode("utf-8"))


def main():
    import sentiment_check_score as S
    from test_sentiment_check_score import check
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        build_models(t / "models")
        os.environ["SENTIMENT_FT_TEST"] = "1"
        os.environ["SENTIMENT_FT_LOCAL_MODELS"] = str(t / "models")
        import sentiment_finetune as F

        def run(*argv):
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    F.main(["--root", str(t), *argv])
            except SystemExit as e:
                return str(e.code), buf.getvalue()
            return None, buf.getvalue()

        tr_dir, ex_dir = t / "data/raw/sentiment_training", t / "data/raw/sentiment_check"
        tr_rows, tr_truth = write_sample(tr_dir, "train", 60, 1)
        ex_rows, ex_truth = write_sample(ex_dir, "exam", 250, 2)
        write_labels_csv(ex_dir, ex_rows, ex_truth)
        with redirect_stdout(io.StringIO()):
            S.main(["--root", str(t), "freeze"])                      # exam labels frozen

        err, _ = run("train")
        check(err and "freeze" in err, "train refuses before the training labels are frozen")

        # labels arrive from the page's store export (one JSON per item)
        exp = t / "export/labels"; exp.mkdir(parents=True)
        for r in tr_rows:
            c, rel = tr_truth[r["id"]]
            (exp / f"{r['id']}.json").write_text(json.dumps({"id": r["id"], "data": {"s": c, "r": rel, "t": r["ticker"],
                                                 "at": "2026-10-06T09:00:00Z", "sec": 5, "n": 1}}), encoding="utf-8")
        bad = json.loads((exp / "train-0001.json").read_text()); bad["data"]["t"] = "WRONG"
        (exp / "train-0001.json").write_text(json.dumps(bad))
        err, _ = run("import-store", "--json-dir", str(t / "export"))
        check(err and "ticker mismatch" in err, "import refuses a label whose ticker does not match the sample")
        bad["data"]["t"] = tr_rows[0]["ticker"]; (exp / "train-0001.json").write_text(json.dumps(bad))
        err, out = run("import-store", "--json-dir", str(t / "export"))
        check(err is None and "60 of 60" in out, "import writes labels.csv in the exam's format")
        err, out = run("freeze-train")
        check(err is None and "FROZEN" in out, "freeze-train freezes the training labels")

        # hide the exam labels while training: training must not need them
        (ex_dir / "labels.csv").rename(ex_dir / "labels.hidden")
        err, out = run("train", "--configs", "wisesight+headline")
        check(err is None and len(F.read_runs(F.TrainPaths(t))) == 3, "train runs 3 seeds for one config, without exam labels present")
        err, out = run("exam")
        check(err and "select" in err, "exam refuses before selection")
        err, out = run("select")
        check(err and "missing" in err, "select refuses until all 12 runs exist")
        err, out = run("train")
        runs = F.read_runs(F.TrainPaths(t))
        check(err is None and "skip wisesight+headline" in out and len(runs) == 12, "train resumes: skips done runs, completes 12")
        check(all(r["test_mode"] and r["sealed_exam_sha256"] and r["git_commit"] for r in runs), "each run logs test mode, sealed hash and commit")
        check(len({r["labels_sha256"] for r in runs}) == 1, "all runs used the same frozen labels")
        sp = F.split([{**r, "sentiment": tr_truth[r["id"]][0], "relevance": tr_truth[r["id"]][1]} for r in tr_rows], "sentiment")
        check(len(sp[1]) == 12 and not ({x["id"] for x in sp[0]} & {x["id"] for x in sp[1]}), "split is 80/20 with no overlap")

        err, out = run("select")
        sel = S.read_json(t / "data/derived/sentiment_finetune/selection_sentiment.json")
        check(err is None and sel["chosen"] in F.CONFIGS and len(sel["table"]) == 4, f"select picks one of 4 configs ({sel['chosen']})")
        err, _ = run("select"); check(bool(err), "select runs once")
        err, _ = run("train"); check(err and "selection already made" in err, "no training after selection")

        (ex_dir / "labels.hidden").rename(ex_dir / "labels.csv")
        chosen = [r for r in runs if r["config"] == sel["chosen"]][0]
        sealed = t / "data/derived/sentiment_finetune/sealed_exam_predictions" / chosen["sealed_exam_file"]
        keep = sealed.read_bytes(); sealed.write_bytes(keep.replace(b"pos", b"neg", 1))
        err, _ = run("exam"); check(err and "changed after training" in err, "exam refuses tampered sealed predictions")
        sealed.write_bytes(keep)
        err, out = run("exam")
        res = S.read_json(t / "data/derived/sentiment_finetune/exam_results_sentiment.json")
        check(err is None and len(res["per_seed"]) == 3 and "kappa_mean" in res and res["per_seed"][0]["n"] == 250, "exam scores the chosen config's 3 seeds on 250 items")
        check("half_B" in res["per_seed"][0] and "relevance_yes" in res["per_seed"][0], "exam reports halves and the relevance=yes subset")
        check(res["primary_measure"].startswith("about_stock") and "about_stock" in res["per_seed"][0]
              and res["primary_n"] == sum(1 for l in S.read_csv(t / "data/raw/sentiment_check/labels.csv") if l["relevance"] in ("yes", "partly"))
              and res["kappa_mean"] == round(statistics.mean(s["about_stock"]["kappa"] for s in res["per_seed"]), 4)
              and "kappa_mean_all_items" in res, "primary exam measure is the about-the-stock subset (amendment 2026-10-15)")
        err, _ = run("exam"); check(err and "once" in err, "exam is sat once")

        err, out = run("--target", "relevance", "train", "--configs", "base+headline")
        rel_runs = [r for r in F.read_runs(F.TrainPaths(t)) if r["target"] == "relevance"]
        check(err is None and len(rel_runs) == 3, "relevance classifier trains (new 3-way head)")

        lab = tr_dir / "labels.csv"; lab.write_bytes(lab.read_bytes().replace(b'"pos"', b'"neg"', 1))
        err, _ = run("train"); check(bool(err), "train refuses after the training labels were edited")
        print(f"\n(chosen in this toy run: {sel['chosen']}, exam kappa {res['kappa_mean']})")
    print("all sentiment_finetune tests passed")


if __name__ == "__main__":
    main()
