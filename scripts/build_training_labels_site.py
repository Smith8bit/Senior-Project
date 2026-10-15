"""Build the private labelling page for the WangchanBERTa training set (published as a claude.ai artifact).

Fills reports/labelling_site/training_labels_site.template.html with data/raw/sentiment_training/items.json
and the label protocol from its SAMPLE.json. The built page contains scraped headlines, so it is written
under data/raw/ (gitignored), never into git. Labels are saved in the artifact's private store
(collection "labels", one document per item id: s, r, t, at, sec, n); Claude exports them to
data/raw/sentiment_training/labels.csv and records the SHA-256 before any training.

  python scripts/build_training_labels_site.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "reports/labelling_site/training_labels_site.template.html"
SRC = ROOT / "data/raw/sentiment_training"
OUT = SRC / "kaohoon-training-labels.html"


def esc(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\!--")


def main() -> None:
    items = json.loads((SRC / "items.json").read_text(encoding="utf-8"))
    proto = json.loads((SRC / "SAMPLE.json").read_text(encoding="utf-8"))["protocol"]
    page = TEMPLATE.read_text(encoding="utf-8").replace("__ITEMS__", esc(items)).replace("__PROTO__", esc(proto))
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({len(items)} items)")


if __name__ == "__main__":
    main()
