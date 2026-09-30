"""Verify a crawl store and freeze it into a versioned bundle (Phase 1.16/1.17, F8).

  python scripts/freeze_pass.py --src data/raw/pass_b --name passB_v1

Steps:
  1. verify: every manifest row has raw/<sha256>.html.gz, it decompresses, and
     the SHA-256 of the decompressed page equals the file name. Any failure
     stops the freeze (nothing is written) unless --force.
  2. bundle into bundles/<name>/:
       <name>_meta.tar.gz  every top-level *.jsonl in --src, plus any
                           sub-store folders (e.g. sitemaps/) other than raw/
       <name>_raw.tar      raw/ (already gzipped per file, so not recompressed)
       SHA256SUMS          checksums of the two archives
       FREEZE.json         counts, commit, timestamps (machine-readable)
Never modifies --src. Refuses to overwrite an existing bundle folder.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(src: Path) -> dict:
    rows = [json.loads(l) for l in (src / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    shas = {r["sha256"] for r in rows}
    missing, corrupt = [], []
    for i, sha in enumerate(sorted(shas), 1):
        p = src / "raw" / f"{sha}.html.gz"
        if not p.exists():
            missing.append(sha)
            continue
        try:
            with gzip.open(p, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != sha:
                    corrupt.append(sha)
        except Exception:
            corrupt.append(sha)
        if i % 5000 == 0:
            print(f"  verified {i}/{len(shas)}", flush=True)
    on_disk = {p.name[:-8] for p in (src / "raw").glob("*.html.gz")}
    return {"manifest_rows": len(rows), "unique_urls": len({r["url"] for r in rows}),
            "unique_files": len(shas), "missing": len(missing), "corrupt": len(corrupt),
            "orphan_files": len(on_disk - shas), "missing_sample": missing[:5], "corrupt_sample": corrupt[:5]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True, help="e.g. passB_v1")
    ap.add_argument("--out", default="bundles")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="bundle even if verification finds problems")
    args = ap.parse_args()
    src, dest = Path(args.src), Path(args.out) / args.name

    print(f"verifying {src} ...")
    v = verify(src)
    print(json.dumps(v, indent=2))
    ok = v["missing"] == 0 and v["corrupt"] == 0
    print("VERIFY: OK" if ok else "VERIFY: PROBLEMS FOUND")
    if args.verify_only:
        return 0 if ok else 1
    if not ok and not args.force:
        print("not freezing; fix or rerun the crawl (stored copies are reused), or pass --force")
        return 1
    if dest.exists():
        print(f"{dest} already exists; a frozen bundle is never overwritten. Use a new --name.")
        return 1
    dest.mkdir(parents=True)

    meta = dest / f"{args.name}_meta.tar.gz"
    with tarfile.open(meta, "w:gz") as t:
        for p in sorted(src.glob("*.jsonl")):
            t.add(p, arcname=p.name)
        for d in sorted(x for x in src.iterdir() if x.is_dir() and x.name != "raw"):
            t.add(d, arcname=d.name)
    print(f"wrote {meta}")
    raw = dest / f"{args.name}_raw.tar"
    with tarfile.open(raw, "w") as t:
        t.add(src / "raw", arcname="raw")
    print(f"wrote {raw}")

    sums = {p.name: sha256_file(p) for p in (meta, raw)}
    (dest / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()), encoding="utf-8")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    info = {"name": args.name, "src": str(src), "frozen_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit, "verify": v, "archives": {n: {"sha256": h, "bytes": (dest / n).stat().st_size}
                                                            for n, h in sums.items()},
            "jsonl_rows": {p.name: sum(1 for _ in p.open(encoding="utf-8")) for p in sorted(src.glob("*.jsonl"))}}
    (dest / "FREEZE.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info["archives"], indent=2))
    print(f"frozen: {dest}  (upload the whole folder to Drive)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
