"""Run an experiment on Kaggle in repeated waves until no tasks remain.

    python code/cluster/kaggle/waves.py E3_learners --shards 5 --user joshuadavidpoznanski --skip-learners smt_krg,smt_kpls

Each wave: push the shards (the archive carries the keys of every task already finished, from any machine), wait for
all shards to finish (each stops itself after 10.5 h), pull their JSONL into results/raw/<exp>/, count what is left.
Stops when nothing is left or after --max-waves. Safe to interrupt and restart: state lives in results/raw/<exp>/, a
restarted driver first waits for and pulls its own running shards, a wave is never pushed before the previous one is
pulled (retried until it succeeds), and every pull is mirrored to OneDrive (code/cluster/backup_results.py).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STOP = ROOT / "results" / "STOP_DRIVERS"          # kill switch: see stop_tracks.py


def nap(seconds):
    """Sleep, but leave cleanly (exit 0) as soon as the kill switch exists. Only called between Kaggle operations, so a
    stop never interrupts a push or a pull; the next start resumes (running shards are waited for and pulled)."""
    end = time.time() + seconds
    while True:
        if STOP.exists():
            print(f"[stop] {STOP.name} present - driver exiting cleanly", flush=True)
            sys.exit(0)
        if time.time() >= end:
            return
        time.sleep(min(30, max(0.0, end - time.time())))
PY = sys.executable
KAGGLE = __import__("shutil").which("kaggle", path=str(Path(sys.executable).parent)) or "kaggle"


def remaining(exp, skip, only=""):
    args = [PY, "-m", "surr.runner", exp, "--dry"] + (["--skip-learners", skip] if skip else []) +            (["--only-learners", only] if only else [])
    out = subprocess.run(args, cwd=ROOT / "code", capture_output=True, text=True).stdout
    # "<exp>: N tasks in shard 1/1, D done, R to run ..."
    return int(out.split(" done, ")[1].split(" to run")[0]), out.strip()


def shard_status(name):
    """'running' | 'done' | 'missing' (never pushed) | 'unknown' (network/API failure; never read as finished)."""
    try:
        r = subprocess.run([KAGGLE, "kernels", "status", name], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "unknown"
    out = r.stdout + r.stderr
    if "has status" in out:
        return "running" if ("RUNNING" in out or "QUEUED" in out) else "done"
    if "Cannot access kernel" in out or "404" in out:
        return "missing"
    return "unknown"


def statuses(slug, user, k):
    return [shard_status(f"{user}/{slug}-shard-{i}of{k}") for i in range(1, k + 1)]


def wait_all(slug, user, k, poll=300):
    """Block until no shard is running or unknown. A failed status call counts as still running: on 2026-10-07 a DNS
    outage made the SMT track read "" as finished and quit mid-wave."""
    while True:
        st = statuses(slug, user, k)
        if all(x in ("done", "missing") for x in st):
            return st
        nap(poll)


def until_ok(cmd, what, wait=600):
    """Run cmd until it exits 0, retrying every `wait` s (network drops, laptop asleep or travelling). Never gives up:
    pushing the next wave before the previous one is pulled would make its output an older kernel version."""
    k = 0
    while subprocess.run(cmd).returncode != 0:
        k += 1
        print(f"  {what} failed (attempt {k}); retrying in {wait // 60} min", flush=True)
        nap(wait)


def collect(pull):
    until_ok(pull, "pull")
    subprocess.run([PY, str(HERE.parent / "backup_results.py")], check=False)   # off-laptop copy (OneDrive)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp"); ap.add_argument("--shards", type=int, default=5); ap.add_argument("--user", required=True)
    ap.add_argument("--skip-learners", default=""); ap.add_argument("--max-waves", type=int, default=20)
    ap.add_argument("--only-learners", default=""); ap.add_argument("--tag", default="")
    ap.add_argument("--gpu", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--grow-to", type=int, default=0, help="use this many shards once the track named by --after is done")
    ap.add_argument("--after", default="", help="--skip-learners list of the other track whose slots this one inherits")
    ap.add_argument("--start-after", default="", help="EXP:ONLY_LEARNERS — wait until that track has no tasks left "
                                                      "(it holds the Kaggle session slots until then)")
    a = ap.parse_args()
    if a.start_after:
        exp2, only2 = a.start_after.split(":")
        while (left := remaining(exp2, "", only2)[0]) > 0:
            print(f"[queued] waiting for {exp2} ({only2}): {left} tasks left", flush=True)
            nap(1800)
    slug = f"surrpaper-{a.exp.lower().replace('_', '-')}" + (f"-{a.tag}" if a.tag else "")
    def pull(k):
        return [PY, str(HERE / "pull_results.py"), a.exp, "--shards", str(k), "--user", a.user] + \
               (["--tag", a.tag] if a.tag else [])
    # startup: whatever this track left on Kaggle (running, or finished but never pulled because the driver died) is
    # waited for and pulled before anything is pushed, at every shard count it may have used; re-pulls add nothing
    for k in sorted({a.shards, a.grow_to} - {0}):
        st = wait_all(slug, a.user, k)
        print(f"[resume] {k}-shard statuses: {st}", flush=True)
        if any(x == "done" for x in st):
            collect(pull(k))
    prev = None
    for w in range(1, a.max_waves + 1):
        left, msg = remaining(a.exp, a.skip_learners, a.only_learners)
        print(f"[wave {w}] {msg}", flush=True)
        if left == 0:
            print("nothing left — done", flush=True); return
        if prev is not None and left >= prev:                            # a wave that finished nothing: stop, don't loop
            sys.exit(f"wave {w - 1} made no progress ({left} left) — check results/kaggle_logs/")
        prev = left
        nap(0)                                                           # last stop check before pushing
        k = a.shards
        if a.grow_to and remaining(a.exp, a.after)[0] == 0:              # the other track is finished: take its slots
            k = a.grow_to
        push = [PY, str(HERE / "push_shards.py"), a.exp, "--shards", str(k), "--user", a.user]
        for flag, v in (("--skip-learners", a.skip_learners), ("--only-learners", a.only_learners), ("--tag", a.tag)):
            push += [flag, v] if v else []
        push += (["--gpu"] if a.gpu else []) + (["--limit", str(a.limit)] if a.limit else [])
        until_ok(push, "push")       # a re-push of a just-started shard only restarts it; resume skips done tasks
        nap(120)
        st = wait_all(slug, a.user, k)
        print(f"[wave {w}] {k}-shard statuses: {st}", flush=True)
        collect(pull(k))
    print("max waves reached", flush=True)


if __name__ == "__main__":
    main()
