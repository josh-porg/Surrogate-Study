"""Download finished Kaggle shard outputs and merge them into results/raw/<exp>/ (duplicate task keys are harmless:
the analysis keeps the first record per key).

    python code/cluster/kaggle/pull_results.py E3 --shards 6 --user <kaggle-username>
"""
from __future__ import annotations

import argparse
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KAGGLE = __import__("shutil").which("kaggle", path=str(Path(sys.executable).parent)) or "kaggle"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp"); ap.add_argument("--shards", type=int, default=4); ap.add_argument("--user", required=True)
    a = ap.parse_args()
    slug = f"surrpaper-{a.exp.lower().replace('_', '-')}"
    dest = ROOT / "results" / "raw" / a.exp; dest.mkdir(parents=True, exist_ok=True)
    for i in range(1, a.shards + 1):
        name = f"{a.user}/{slug}-shard-{i}of{a.shards}"
        status = subprocess.run([KAGGLE, "kernels", "status", name], capture_output=True, text=True).stdout.strip()
        print(name, "->", status)
        with tempfile.TemporaryDirectory() as td:
            subprocess.run([KAGGLE, "kernels", "output", name, "-p", td], check=False)
            for f in Path(td).rglob("*.jsonl"):
                shutil.copy(f, dest / f"kaggle_{i}of{a.shards}_{f.name}")
    print("merged into", dest)


if __name__ == "__main__":
    main()
