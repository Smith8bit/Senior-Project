"""Checks for scripts/select_final.py (batch 7 assembly rules) on made-up batch summaries. No data or torch needed.

  python tests/test_select_final.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import select_final as SF  # noqa: E402

FAILS = []
STD = {"h1": 0.004, "h5": 0.001, "h10": 0.009}


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def summary(d, gains, base=(0.353, 0.384, 0.384)):
    rows = [{"label": "E21", "file": "e21.json", "h": {h: {"mean": b, "minus_control": 0.0, "control_std": STD[h]} for h, b in zip(SF.HS, base)}}]
    for lb, g in gains.items():
        rows.append({"label": lb, "file": f"{lb}.json",
                     "h": {h: {"mean": b + x, "minus_control": x, "control_std": STD[h]} for h, b, x in zip(SF.HS, base, g)}})
    p = Path(d) / f"s{len(list(Path(d).iterdir()))}.json"
    p.write_text(json.dumps({"rows": rows}))
    return str(p)


def run(step, **kw):
    a = type("A", (), {"summary": [], "out": None, "config": None, **kw})
    getattr(SF, step)(a)


def main():
    with tempfile.TemporaryDirectory() as d:
        # nothing kept
        s = summary(d, {"E26": (0.001, 0.0, 0.0), "E27": (-0.005, 0.0, 0.0)})
        out = str(Path(d) / "f0.json"); run("select", summary=[s], out=out)
        c = json.loads(Path(out).read_text())
        check(c["final_label"] == "E21" and c["final_args"] == SF.E21, "no counted gain -> final is E21")
        # one kept; another has a gain but also a counted loss
        s = summary(d, {"E26": (0.006, 0.0, 0.0), "E28": (0.006, -0.002, 0.0)})
        out = str(Path(d) / "f1.json"); run("select", summary=[s], out=out)
        c = json.loads(Path(out).read_text())
        check(c["used"] == ["E26"] and c["final_label"] == "E26" and "e03_e26" in c["final_args"] and not c["needs_stacked_run"],
              "one kept change becomes the final run; a run with a counted loss is not kept")
        # patch conflicts: E07 + E08 + E09 kept, E09 larger -> 2,1;10,5 ; E15 kept but smaller
        s5 = summary(d, {"E26": (0.005, 0.0, 0.0), "E27": (0.0, 0.004, 0.0), "E06": (0.0, 0.003, 0.0)})
        s6 = summary(d, {"E07": (0.005, 0.0, 0.0), "E08": (0.0, 0.002, 0.0), "E09": (0.0, 0.0, 0.010), "E15": (0.0, 0.002, 0.0)})
        out = str(Path(d) / "f2.json"); run("select", summary=[s5, s6], out=out)
        c = json.loads(Path(out).read_text())
        a = c["stacked_args"]
        check(a[a.index("--patches") + 1] == "2,1;10,5", f"patch conflict resolved to {a[a.index('--patches') + 1]}")
        check("E06" in c["dropped_by_conflict"] and "--context-min" not in a and "--lookback" in a, "look-back conflict keeps the larger gain (E27)")
        check(set(c["dropped_by_conflict"]) == {"E08", "E15", "E06"} and c["needs_stacked_run"], f"dropped {c['dropped_by_conflict']}")
        check(a[a.index("--extra-features") + 1] == "e03_e26" and a.count("--extra-features") == 1, "E26 replaces the input set, once")
        # stacked step: stack has a counted loss -> best single kept run
        st = summary(d, {"STACK": (0.02, -0.01, 0.0), "E26": (0.005, 0.0, 0.0), "E07": (0.005, 0.0, 0.0), "E09": (0.0, 0.0, 0.010),
                         "E27": (0.0, 0.002, 0.0)})
        run("stacked", summary=[st], config=out)
        c = json.loads(Path(out).read_text())
        check(c["final_label"] == "E09" and c["final_args"][c["final_args"].index("--patches") + 1] == "5,5;10,5",
              f"stack with a counted loss is skipped; best single run chosen ({c['final_label']})")
        # E13 kept
        e = summary(d, {"E13": (0.0, 0.0, 0.012)})
        run("e13", summary=[e], config=out)
        c = json.loads(Path(out).read_text())
        check(c["final_label"] == "E09+E13" and "--recent-weight" in c["final_args"] and c["final_run"] == "E13.json", "E13 kept is added to the final")
        # E15 wins the patch group outright
        s6 = summary(d, {"E07": (0.002, 0.0, 0.0), "E15": (0.0, 0.0, 0.012)})
        out = str(Path(d) / "f3.json"); run("select", summary=[s6], out=out)
        c = json.loads(Path(out).read_text())
        check(c["used"] == ["E15"] and c["final_args"][c["final_args"].index("--patches") + 1] == "16,8", "E15 alone when its gain is largest")
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
