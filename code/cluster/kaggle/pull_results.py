"""Download finished Kaggle shard outputs and merge them into results/raw/<exp>/ (duplicate task keys are harmless:
the analysis keeps the first record per key).

    python code/cluster/kaggle/pull_results.py E3 --shards 6 --user <kaggle-username>

Exit code 0 only if every shard was fetched: its download succeeded and its kernel log came back (a shard that died
early has a log but no JSONL, which still counts). The wave driver never pushes the next wave until this succeeds,
because a push makes the unpulled output an older kernel version. Files land atomically (copy to .part, then rename).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import quiet  # noqa: E402,F401  (no console windows for child processes)

import argparse
import os
import sys
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KAGGLE = __import__("shutil").which("kaggle", path=str(Path(sys.executable).parent)) or "kaggle"


def _place(src: Path, dst: Path):
    tmp = dst.with_name(dst.name + ".part")
    shutil.copy(src, tmp)
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp"); ap.add_argument("--shards", type=int, default=4); ap.add_argument("--user", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    slug = f"surrpaper-{a.exp.lower().replace('_', '-')}" + (f"-{a.tag}" if a.tag else "")
    dest = ROOT / "results" / "raw" / a.exp; dest.mkdir(parents=True, exist_ok=True)
    logs = ROOT / "results" / "kaggle_logs"; logs.mkdir(exist_ok=True)
    failed = []
    for i in range(1, a.shards + 1):
        name = f"{a.user}/{slug}-shard-{i}of{a.shards}"
        status = subprocess.run([KAGGLE, "kernels", "status", name], capture_output=True, text=True).stdout.strip()
        print(name, "->", status, flush=True)
        if "has status" not in status:
            from waves import shard_status
            if shard_status(name) == "missing":                              # never pushed: nothing to fetch
                print("  not on Kaggle (never pushed); skipped", flush=True); continue
        with tempfile.TemporaryDirectory() as td:
            # results and the kernel log only: the full output also holds the unpacked project tree
            r = subprocess.run([KAGGLE, "kernels", "output", name, "-p", td, "-q", "--file-pattern", r"(\.jsonl|\.log)$"],
                               capture_output=True, text=True)
            got = list(Path(td).rglob("*.jsonl")); log = list(Path(td).glob("*.log"))
            if r.returncode or not log:
                failed.append(name)
                print(f"  FAILED to fetch (rc={r.returncode}, log={bool(log)}): {(r.stderr or r.stdout)[-300:]}", flush=True)
                continue
            stamp = int(time.time())
            for f in got:                     # shard files are named part_<i>of<k>_<pid>.jsonl, unique per kernel run
                if any(o.stat().st_size == f.stat().st_size for o in dest.glob(f"kaggle{a.tag}_{i}of{a.shards}_*_{f.name}")):
                    print(f"  {f.name} already pulled", flush=True); continue
                _place(f, dest / f"kaggle{a.tag}_{i}of{a.shards}_{stamp}_{f.name}")
            for f in log:                                                    # kept for post-mortems
                _place(f, logs / f"{stamp}_{f.name}")
            print(f"  {len(got)} result file(s), {sum(1 for f in got for _ in open(f, encoding='utf8'))} records",
                  flush=True)
    print("merged into", dest, flush=True)
    sys.exit(f"not fetched: {failed}" if failed else 0)


if __name__ == "__main__":
    main()
