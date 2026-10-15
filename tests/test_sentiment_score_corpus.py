"""Offline test of scripts/sentiment_score_corpus.py with tiny stand-in models (needs torch + transformers).

  python tests/test_sentiment_score_corpus.py
"""
from __future__ import annotations

import csv
import io
import json
import os
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
except ImportError as e:
    print(f"skip: {e}"); sys.exit(0)

FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def main():
    from test_sentiment_finetune import build_models
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        build_models(t / "models")
        os.environ["SENTIMENT_FT_LOCAL_MODELS"] = str(t / "models")
        import importlib
        import sentiment_finetune as FT
        importlib.reload(FT)
        import sentiment_score_corpus as SC
        importlib.reload(SC)
        # prices: 35 symbols on 6 trading days (calendar needs >= 30 symbols per day)
        (t / "data/raw/prices/siamchart").mkdir(parents=True)
        days = ["20240102", "20240103", "20240104", "20240105", "20240108", "20240109"]
        with open(t / "data/raw/prices/siamchart/set50_union_2019_2025.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["symbol", "date", "open", "high", "low", "close", "volume"])
            for d in days:
                for k in range(35):
                    w.writerow([f"S{k}", d, 1, 1, 1, 1, 100])
        with open(t / "data/set50_membership.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["symbol", "period", "effective_from", "effective_to", "published", "provenance"])
            for s in ("PTT", "KBANK", "SCB"):
                w.writerow([s, "2024H1", "2024-01-01", "2024-06-30", "", ""])
        (t / "data/raw/kaohoon_articles").mkdir(parents=True)
        arts = [
            ("u1", "news", "PTT กำไรพุ่ง", "2024-01-03T10:00:00+07:00", ["PTT"]),
            ("u2", "breakingnews", "KBANK หุ้นขึ้น 3%", "2024-01-03T17:00:00+07:00", ["KBANK", "SCB"]),
            ("u3", "column", "ตลาดทรงตัว", "2024-01-04T09:00:00+07:00", ["PTT"]),          # excluded section
            ("u4", "news", "SCB ขาดทุนหนัก ร่วง 5%", "2024-01-06T11:00:00+07:00", ["SCB", "XYZ"]),  # Saturday; XYZ not in union
            ("u5", "news", "English headline", "2024-01-04T09:00:00+07:00", ["PTT"]),
        ]
        with open(t / "data/raw/kaohoon_articles/articles.jsonl", "w", encoding="utf-8") as f:
            for u, sec, h, p, syms in arts:
                f.write(json.dumps({"url": u, "url_section": sec, "headline": h, "date_published": p, "symbols_tagged": syms,
                                    "parse_status": "ok", "language": "en" if u == "u5" else "th"}, ensure_ascii=False) + "\n")

        def run(*argv):
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    SC.main(["--root", str(t), *argv])
            except SystemExit as e:
                return str(e.code), buf.getvalue()
            return None, buf.getvalue()

        err, _ = run()
        check(err and "select" in err, "refuses before selection")
        T = FT.TrainPaths(t); T.out.mkdir(parents=True)
        (T.out / "selection_sentiment.json").write_text(json.dumps({"chosen": "wisesight+headline"}))
        runs = [{"target": "sentiment", "config": "wisesight+headline", "seed": s, "labels_sha256": "x",
                 "model_dir": None} for s in FT.SEEDS]
        T.runs.write_text("\n".join(json.dumps(r) for r in runs) + "\n")
        err, _ = run()
        check(err and "save-models" in err, "refuses when the selected runs' models were not saved")
        for r in runs:
            r["model_dir"] = str(t / "models" / "wisesight")
        T.runs.write_text("\n".join(json.dumps(r) for r in runs) + "\n")
        err, out = run()
        check(err is None, f"scores the corpus ({err})")
        rows = list(csv.DictReader(open(t / "data/derived/sentiment_scores.csv", encoding="utf-8")))
        keys = {(r["url"], r["ticker"]) for r in rows}
        check(keys == {("u1", "PTT"), ("u2", "KBANK"), ("u2", "SCB"), ("u4", "SCB")},
              "pairs: Thai only, column excluded, union tickers only")
        by = {(r["url"], r["ticker"]): r for r in rows}
        check(by[("u2", "KBANK")]["trading_day"] == "2024-01-04" and by[("u4", "SCB")]["trading_day"] == "2024-01-08",
              "16:30 rule and weekend roll-forward")
        check(by[("u2", "KBANK")]["restatement"] == "1" and by[("u4", "SCB")]["restatement"] == "1" and by[("u1", "PTT")]["restatement"] == "0",
              "restatement flag: breakingnews, or % with a move verb")
        check(all(-1 <= float(r["s"]) <= 1 and abs(float(r["p_pos"]) + float(r["p_neu"]) + float(r["p_neg"]) - 1) < 1e-4 for r in rows),
              "s = p_pos - p_neg in [-1, 1]; probabilities sum to 1")
        check("headline" not in rows[0], "no headline text in the output")
        err, _ = run()
        check(err and "already exists" in err, "scores are computed once (hard rule 9)")
    print("all sentiment_score_corpus tests passed" if not FAILS else f"{len(FAILS)} FAILED")
    if FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()
