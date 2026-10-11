"""Mirror results/ into the author's KU OneDrive so a lost or broken laptop loses nothing (results/raw is gitignored).

    python code/cluster/backup_results.py            copy new or changed files; never deletes anything in the backup

Called by the Kaggle wave driver after every successful pull; safe to run any time, including during a local run
(a JSONL being appended is simply copied again next time). Files land atomically (.part, then rename).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results"
DST = Path(os.environ.get("SURR_BACKUP", Path.home() / "OneDrive - University of Kansas" / "SurrogatePaper_backup" / "results"))


def main():
    if not DST.parent.parent.exists():
        sys.exit(f"backup target missing: {DST.parent.parent}")
    n = 0
    for f in SRC.rglob("*"):
        if not f.is_file() or f.suffix == ".part":
            continue
        d = DST / f.relative_to(SRC)
        s = f.stat()
        if d.exists() and d.stat().st_size == s.st_size and d.stat().st_mtime >= s.st_mtime:
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_name(d.name + ".part")
        shutil.copy2(f, tmp)
        os.replace(tmp, d)
        n += 1
    print(f"backup: {n} file(s) copied to {DST}", flush=True)


if __name__ == "__main__":
    main()
