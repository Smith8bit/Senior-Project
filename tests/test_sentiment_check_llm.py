"""Offline tests for scripts/sentiment_check_llm.py against a fake Ollama server (no real model).

Run from the repo root:  python tests/test_sentiment_check_llm.py
"""
import hashlib
import io
import json
import sys
import tempfile
import threading
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import sentiment_check_llm as L  # noqa: E402
import test_sentiment_check_score as T  # noqa: E402

CALLS = []
GARBAGE = "synthetic headline 77"  # the fake model never answers this one properly


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/api/version":
            self._send({"version": "0.0-fake"})
        elif self.path == "/api/tags":
            self._send({"models": [{"name": L.PRIMARY_MODEL + ":latest", "model": L.PRIMARY_MODEL + ":latest",
                                    "digest": "sha256:fake", "size": 1}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        CALLS.append(req)
        user = req["messages"][1]["content"]
        head = user.split("Headline: ")[1].split("\n")[0]
        if head == GARBAGE:
            content = "not json"
        else:
            h = int(hashlib.sha256(head.encode()).hexdigest(), 16)
            content = json.dumps({"sentiment": ["pos", "neu", "neg"][h % 3], "relevance": ["yes", "partly", "no"][h % 7 % 3]})
        self._send({"message": {"role": "assistant", "content": content}})


def run(root, *argv):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            L.main(["--root", str(root), *argv])
    except SystemExit as e:
        return str(e.code), buf.getvalue()
    return None, buf.getvalue()


def main():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    L.OLLAMA = f"http://127.0.0.1:{srv.server_address[1]}"
    check = T.check

    check(L.TEMPLATE.count("{ticker}") == 8 and "{headline}" in L.TEMPLATE, "template has the expected placeholders")
    req_fields = set(L.SCHEMA["properties"])
    check(req_fields == {"sentiment", "relevance"}, "schema asks for sentiment + relevance only")

    with tempfile.TemporaryDirectory() as t:
        root = T.make_root(Path(t))
        T.write_labels(root)
        err, out = run(root, "selftest")
        check(err is None and "projected time" in out, "selftest runs against the server")
        err, _ = run(root, "label")
        check(err and "freeze" in err, "label refuses before the human labels are frozen")
        T.run(root, "freeze")

        # simulate an interrupted run: 10 items already in the partial file
        P = T.S.Paths(root)
        O = L.Out(P, L.PRIMARY_MODEL)
        O.partial.parent.mkdir(parents=True, exist_ok=True)
        sample = T.S.read_csv(P.sample)
        with open(O.partial, "w", encoding="utf-8") as f:
            for s in sample[:10]:
                f.write(json.dumps({"id": s["id"], "sentiment": "pos", "relevance": "yes", "valid": True, "retried": False,
                                    "raw": "{}", "seconds": 1.0, "model_digest": "sha256:fake",
                                    "prompt_sha256": L.PROMPT_SHA256}) + "\n")
        CALLS.clear()
        err, out = run(root, "label")
        check(err is None, "label runs after freeze")
        check(len(CALLS) == 240 + 1, f"resume: only the 240 missing items were asked (+1 retry) [{len(CALLS)} calls]")
        rows = {r["id"]: r for r in T.S.read_csv(O.csv)}
        check(len(rows) == 250, "250 rows written")
        bad = rows["a3b-077"]
        check(bad["valid"] == "False" and bad["retried"] == "True" and (bad["sentiment"], bad["relevance"]) == ("neu", "no"),
              "invalid output: retried once, then counted as neu / no")
        check(all(c["options"] == L.OPTIONS and c["format"] == L.SCHEMA for c in CALLS), "frozen options + JSON schema sent every time")
        check(all("think" not in c for c in CALLS), "no 'think' flag for the Gemma model")
        meta = T.S.read_json(O.meta)
        check(meta["prompt_sha256"] == L.PROMPT_SHA256 and meta["n_invalid_after_retry"] == 1, "record has prompt hash and invalid count")
        err, _ = run(root, "label")
        check(err and "refusing" in err, "label refuses to overwrite a finished run")

        err, out = run(root, "score")
        check(err is None, "score runs")
        res = T.S.read_json(O.results)
        check(res["headline_result_half_B"]["n"] == 125 and res["half_A"]["n"] == 125, "half A / half B are 125 each")
        check(res["selection_on_half_A"]["wangchanberta_kappa"] is None, "selection handles missing WangchanBERTa file")
        check("relevance_agreement_all_exploratory" in res, "relevance agreement reported")

        # a mixed partial (different prompt) must be refused
        O.csv.unlink(); O.meta.unlink()
        lines = O.partial.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0]); first["prompt_sha256"] = "other"
        O.partial.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n", encoding="utf-8")
        err, _ = run(root, "label")
        check(err and "different model or prompt" in err, "refuses to mix runs made with a different prompt")
    srv.shutdown()
    print("\nall sentiment_check_llm tests passed")


if __name__ == "__main__":
    main()
