"""Set records aside so the runner treats their tasks as not done and runs them again.

    python code/cluster/set_aside.py E6_mf --where method=feature_aug --label feature_aug_was_nargp
    python code/cluster/set_aside.py E6_mf --where method=feature_aug --label feature_aug_was_nargp --apply

Every results/raw/<exp>/*.jsonl file holding a matching record is moved, unchanged, into
results/raw/<exp>/set_aside_<label>/, and a copy without the matching lines takes its place (same name). Nothing is
deleted. The runner reads only top-level JSONL files; pull_results.py compares against subfolders too, so a re-pull of
an old Kaggle output does not bring the records back. Without --apply it only reports what it would move.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def matches(cfg, where):
    return all(str(cfg.get(k)) == v for k, v in where.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp"); ap.add_argument("--where", nargs="+", required=True, help="cfg key=value (all must match)")
    ap.add_argument("--label", required=True); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    where = dict(w.split("=", 1) for w in a.where)
    folder = ROOT / "results" / "raw" / a.exp
    dest = folder / f"set_aside_{a.label}"
    total = 0
    for f in sorted(folder.glob("*.jsonl")):
        keep, hit = [], 0
        for line in f.read_text(encoding="utf8").splitlines(keepends=True):
            try:
                cfg = json.loads(line)["cfg"]
            except (json.JSONDecodeError, KeyError):
                keep.append(line); continue
            if matches(cfg, where):
                hit += 1
            else:
                keep.append(line)
        if not hit:
            continue
        total += hit
        print(f"{f.name}: {hit} matching record(s) of {hit + len(keep)}")
        if a.apply:
            dest.mkdir(exist_ok=True)
            if (dest / f.name).exists():
                raise SystemExit(f"{dest / f.name} exists; pick another label")
            shutil.move(str(f), dest / f.name)
            tmp = f.with_name(f.name + ".part")
            tmp.write_text("".join(keep), encoding="utf8"); tmp.replace(f)
    print(f"{total} record(s) {'set aside in ' + str(dest) if a.apply else 'would be set aside (dry run; add --apply)'}")


if __name__ == "__main__":
    main()
