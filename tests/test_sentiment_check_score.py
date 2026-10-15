"""Offline tests for scripts/sentiment_check_score.py on synthetic data (no real headlines, no model).

Run from the repo root:  python tests/test_sentiment_check_score.py
Standard library only; the preprocessing test runs only if pythainlp and emoji are installed.
"""
import csv
import hashlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sentiment_check_score as S  # noqa: E402

N = 250
# rows = human, cols = model, order pos/neu/neg. Hand-computed: po = 0.76, pe = 0.388,
# kappa = 0.372 / 0.612 = 0.607843..., macro-F1 = (0.8 + 0.769231 + 0.714286) / 3 = 0.761172
MATRIX = {"pos": {"pos": 40, "neu": 10, "neg": 0},
          "neu": {"pos": 10, "neu": 100, "neg": 15},
          "neg": {"pos": 0, "neu": 25, "neg": 50}}


def run(root, *argv):
    """Run a subcommand; return (exit message or None, stdout)."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            S.main(["--root", str(root), *argv])
    except SystemExit as e:
        return str(e.code), buf.getvalue()
    return None, buf.getvalue()


def make_root(tmp: Path) -> Path:
    lab = tmp / "data" / "raw" / "sentiment_check"
    lab.mkdir(parents=True)
    rows = [{"id": f"a3b-{i:03d}", "ticker": f"T{i % 7}", "url": f"https://example.invalid/news/{i}",
             "section": ["news", "column", "summary"][i % 3], "published": "2024-01-02T10:00:00+07:00",
             "trading_day": "2024-01-02", "headline": f"synthetic headline {i}"} for i in range(1, N + 1)]
    with open(lab / "sample.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    digest = hashlib.sha256((lab / "sample.csv").read_bytes()).hexdigest()
    (lab / "SAMPLE.json").write_text(json.dumps({"sample_sha256": digest, "n": N}), encoding="utf-8")
    return tmp


def pairs():
    out = []
    for h in S.CLASSES:
        for m in S.CLASSES:
            out += [(h, m)] * MATRIX[h][m]
    assert len(out) == N
    return out


def write_labels(root: Path, blank=(), swap=False):
    """Same format as labeling.html's export: BOM, every field quoted, CRLF."""
    sample = S.read_csv(root / "data/raw/sentiment_check/sample.csv")
    rel = ["yes", "partly", "no", "yes"]
    lines = [["id", "ticker", "url", "sentiment", "relevance", "labeled_at", "seconds"]]
    for k, (s, (h, _)) in enumerate(zip(sample, pairs())):
        sent, r = ("", "") if s["id"] in blank else (h, rel[k % 4])
        lines.append([s["id"], s["ticker"], s["url"], sent, r, f"2026-10-05T10:{k // 60:02d}:{k % 60:02d}.000Z", "12"])
    if swap:
        lines[1], lines[2] = lines[2], lines[1]
    text = "\r\n".join(",".join('"' + v.replace('"', '""') + '"' for v in row) for row in lines)
    (root / "data/raw/sentiment_check/labels.csv").write_bytes(("﻿" + text).encode("utf-8"))


def write_fake_preds(root: Path, predicted_at="2099-01-01T00:00:00+00:00"):
    P = S.Paths(root)
    labels, lmeta = S.load_frozen_labels(P)
    P.preds.parent.mkdir(parents=True, exist_ok=True)
    with open(P.preds, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "raw", "pred3", "p_pos", "p_neu", "p_neg", "p_q"])
        for k, (l, (_, m)) in enumerate(zip(labels, pairs())):
            raw = "q" if (m == "neu" and k % 5 == 0) else m  # some q, which must map to neu
            w.writerow([l["id"], raw, m, 0.25, 0.25, 0.25, 0.25])
    S.write_json(P.preds_meta, {"predicted_at": predicted_at, "labels_sha256_at_predict": lmeta["labels_sha256"],
                                "sample_sha256": S.read_json(P.sample_meta)["sample_sha256"],
                                "preds_sha256": S.sha256(P.preds), "label_mapping_source": "test"})


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


def test_metrics():
    h, m = zip(*pairs())
    h, m = list(h), list(m)
    k = S.cohen_kappa(h, m)
    check(abs(k - 0.372 / 0.612) < 1e-12, f"kappa matches hand computation ({k:.6f})")
    f1, _ = S.macro_f1(h, m)
    check(abs(f1 - (0.8 + 2 * (100 / 135) * 0.8 / (100 / 135 + 0.8) + 2 * (50 / 65) * (50 / 75) / (50 / 65 + 50 / 75)) / 3) < 1e-9,
          f"macro-F1 matches hand computation ({f1:.6f})")
    check(S.confusion(h, m) == MATRIX, "confusion matrix rows = human, cols = model")
    check(S.band(0.6078) == "usable with the limitation stated", "0.6078 bands as usable, not as-is")
    check(S.band(0.61) == "use as-is" and S.band(0.41) == "usable with the limitation stated"
          and S.band(0.4099) == "fine-tune or demote to exploratory", "band edges")
    check(abs(S.cohen_kappa(h, h) - 1) < 1e-12, "self-agreement kappa = 1")


def test_flow():
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t))
        err, _ = run(root, "freeze")
        check(err and "not found" in err, "freeze refuses when labels.csv is missing")
        err, _ = run(root, "predict")
        check(err and "freeze" in err, "predict refuses before labels are frozen")

        write_labels(root, blank={"a3b-007", "a3b-200"})
        err, _ = run(root, "freeze")
        check(err and "2 items" in err and "a3b-007" in err, "freeze refuses incomplete labels and names them")
        write_labels(root, swap=True)
        err, _ = run(root, "freeze")
        check(err and "row mismatch" in err, "freeze refuses reordered rows")

        write_labels(root)
        err, out = run(root, "freeze")
        check(err is None and "FROZEN" in out, "freeze succeeds on complete labels")
        err, out = run(root, "freeze")
        check(err is None and "Already frozen" in out, "re-freeze of identical file is a no-op")

        err, out = run(root, "describe")
        d = json.loads(out)
        check(err is None and d["relevance_yes_share"] == 0.5 and d["relevance_yes_within_prediction"],
              "describe: relevance share 0.5, inside 40-60%")

        err, _ = run(root, "score")
        check(err and "predict" in err, "score refuses without predictions")
        write_fake_preds(root)
        err, out = run(root, "score")
        check(err is None, "score runs")
        res = S.read_json(S.Paths(root).results)
        check(abs(res["primary"]["kappa"] - 0.6078) < 1e-4, "score: kappa 0.6078")
        check(res["primary"]["band"] == "usable with the limitation stated", "score: band")
        check(res["primary"]["kappa_within_prediction"] is False, "score: 0.61 is outside predicted 0.3-0.5")
        lo, hi = res["primary"]["kappa_ci95_bootstrap"]
        check(lo < 0.6078 < hi and hi - lo < 0.25, f"bootstrap CI brackets kappa ({lo}, {hi})")
        check(res["half_A"]["n"] == 125 and res["half_B"]["n"] == 125, "halves are 125 + 125")
        check(res["relevance_yes"]["n"] == 125, "relevance=yes subset size")
        check(res["model_raw_counts"]["q"] > 0 and res["model_counts"]["neu"] == 135, "q counted raw, mapped to neu")

        err, _ = run(root, "freeze")
        check(err is None, "freeze is still a no-op after predictions when the file is unchanged")
        lab = root / "data/raw/sentiment_check/labels.csv"
        lab.write_bytes(lab.read_bytes().replace(b'"pos","yes"', b'"neg","yes"', 1))
        for cmd in ("freeze", "describe", "score"):
            err, _ = run(root, cmd)
            check(bool(err), f"{cmd} refuses after labels.csv was edited post-freeze")

    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t))
        write_labels(root)
        run(root, "freeze")
        write_fake_preds(root, predicted_at="2000-01-01T00:00:00+00:00")
        err, _ = run(root, "score")
        check(err and "AFTER" in err, "score refuses when predictions predate the freeze")

    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t))
        write_labels(root)
        P = S.Paths(root)
        P.preds.parent.mkdir(parents=True)
        P.preds.write_text("id\n", encoding="utf-8")
        err, _ = run(root, "freeze")
        check(err and "blind" in err, "freeze refuses if model output already exists")


def test_preprocess():
    try:
        import emoji  # noqa: F401
        import pythainlp  # noqa: F401
    except ImportError:
        print("skip preprocessing test (pythainlp/emoji not installed)")
        return
    # Expected strings produced by the original thai2transformers 0.1.2 process_transformers
    # (quirks included, e.g. the stray "("); the vendored copy matched it on 254 strings incl.
    # all 250 sample headlines, checked 2026-10-06.
    cases = {"KTB  บวก 3%\nกำไร (()) ดีมากกกกก": "ktb<_>บวก<_>3%<_>กำไร<_>(ดีมาก",
             "ปตท. ไตรมาส 3/67 ‘กำไร’ โต": "ปตท.<_>ไตรมาส<_>3/67<_>‘กำไร’<_>โต"}
    for raw, want in cases.items():
        got = S.process_transformers(raw)
        check(got == want, f"process_transformers: {got!r}")


if __name__ == "__main__":
    test_metrics()
    test_flow()
    test_preprocess()
    print("\nall sentiment_check_score tests passed")
