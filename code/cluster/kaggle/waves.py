"""Run an experiment on Kaggle in repeated waves until no tasks remain.

    python code/cluster/kaggle/waves.py E3_learners --shards 5 --user joshuadavidpoznanski --skip-learners smt_krg,smt_kpls

Each wave: push the shards (the archive carries the keys of every task already finished, from any machine), wait for
all shards to finish (each stops itself after 10.5 h), pull their JSONL into results/raw/<exp>/, count what is left.
Stops when nothing is left or after --max-waves. Safe to interrupt and restart: state lives in results/raw/<exp>/, a
restarted driver first waits for and pulls its own running shards, a wave is never pushed before the previous one is
pulled (retried until it succeeds), and every pull is mirrored to OneDrive (code/cluster/backup_results.py).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import quiet  # noqa: E402,F401  (no console windows for child processes)

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


def broken(exp):
    """The marker a driver leaves when most of a wave's tasks errored (see check_errors)."""
    f = ROOT / "results" / "raw" / exp / "BROKEN.txt"
    return f.read_text(encoding="utf8").strip() if f.exists() else ""


def check_errors(exp, limit=0.5, min_n=50):
    """Error records count as done, so a broken setup finishes a whole experiment in seconds and lets the next track
    start (E8, 2026-11-08: fstar_table.json not shipped, 7 840 of 7 840 errors). Stop the track instead: leave
    BROKEN.txt, which also holds back every track that starts after this one."""
    import json
    best = {}
    for f in (ROOT / "results" / "raw" / exp).glob("*.jsonl"):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            best[r["key"]] = best.get(r["key"]) == "ok" and "ok" or r.get("status")
    n = len(best); bad = [k for k, v in best.items() if v == "error"]
    if n >= min_n and len(bad) > limit * n:
        msg = f"{len(bad)} of {n} {exp} records are errors — fix the cause, move the error records aside, delete this file"
        (ROOT / "results" / "raw" / exp / "BROKEN.txt").write_text(msg, encoding="utf8")
        sys.exit(msg)


def remaining(exp, skip, only=""):
    if broken(exp):
        return 10 ** 9, f"{exp}: broken — {broken(exp)}"
    args = [PY, "-m", "surr.runner", exp, "--dry"] + (["--skip-learners", skip] if skip else []) +            (["--only-learners", only] if only else [])
    # "<exp>: N tasks in shard 1/1, D done, R to run ..."; retried, because a dry run can fail transiently (e.g. while
    # the code is being edited) and a crash here used to stop the driver (IndexError, 2026-10-13)
    for attempt in range(12):
        r = subprocess.run(args, cwd=ROOT / "code", capture_output=True, text=True)
        try:
            return int(r.stdout.split(" done, ")[1].split(" to run")[0]), r.stdout.strip()
        except (IndexError, ValueError):
            print(f"  dry run failed (attempt {attempt + 1}): {(r.stderr or r.stdout)[-200:]}", flush=True)
            nap(300)
    raise SystemExit("dry run keeps failing; check the runner")


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


def existing_kernels(slug):
    """Names of this account's kernels matching slug, or None if the listing failed (network/API). Since 2026-11 the
    CLI prints nothing for a never-pushed kernel instead of a 404, so 'missing' is decided here: a kernel absent from a
    successful listing was never pushed (E6 shard 5 read as 'unknown' and would have blocked the track forever)."""
    try:
        r = subprocess.run([KAGGLE, "kernels", "list", "--mine", "--search", slug, "--page-size", "100"],
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or "ref" not in r.stdout:
        return None
    return {ln.split()[0] for ln in r.stdout.splitlines() if ln.strip() and "/" in ln.split()[0]}


def statuses(slug, user, k):
    names = [f"{user}/{slug}-shard-{i}of{k}" for i in range(1, k + 1)]
    have = existing_kernels(slug)
    out = []
    for n in names:
        s = shard_status(n)
        if s == "unknown" and have is not None and n not in have:
            s = "missing"
        out.append(s)
    return out


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
    while True:
        # output captured and echoed: inheriting the log handle left 70+ silent failures (E5/E14, 2026-11-09)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf8", errors="replace",
                               stdin=subprocess.DEVNULL)
            rc, out = r.returncode, (r.stdout or "") + (r.stderr or "")
        except OSError as e:
            rc, out = -1, repr(e)
        lines = [s for s in out.splitlines() if "outdated" not in s]
        if rc == 0:
            print("\n".join(lines[-3:]), flush=True)
            return
        k += 1
        print(f"  {what} failed (attempt {k}, rc={rc}); retrying in {wait // 60} min\n    "
              + "\n    ".join(lines[-6:]), flush=True)
        nap(wait)


def collect(pull):
    until_ok(pull, "pull")
    if "Efstar_refine" in pull:                       # fold refined optima into results/fstar_table.json (atomic)
        subprocess.run([PY, str(ROOT / "code" / "analysis" / "fstar_table.py"), "--merge-kaggle"], check=False)
    subprocess.run([PY, str(HERE.parent / "backup_results.py")], check=False)   # off-laptop copy (OneDrive)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp"); ap.add_argument("--shards", type=int, default=5); ap.add_argument("--user", required=True)
    ap.add_argument("--skip-learners", default=""); ap.add_argument("--max-waves", type=int, default=20)
    ap.add_argument("--only-learners", default=""); ap.add_argument("--tag", default="")
    ap.add_argument("--gpu", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stop-local", default="", help="when this track starts, end laptop processes whose command line "
                                                     "contains this text (the same work moving to Kaggle)")
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
    if a.stop_local:
        import psutil
        me = {psutil.Process().pid, psutil.Process().ppid()}
        for pr in psutil.process_iter(["cmdline"]):
            cl = " ".join(pr.info["cmdline"] or [])
            if a.stop_local in cl and "waves.py" not in cl and "--dry" not in cl and pr.pid not in me:
                print(f"[handover] ending laptop process {pr.pid}: {cl[:100]}", flush=True)
                pr.terminate()
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
    if broken(a.exp):
        sys.exit(f"{a.exp}: {broken(a.exp)}")
    check_errors(a.exp)
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
        check_errors(a.exp)
    print("max waves reached", flush=True)


if __name__ == "__main__":
    main()
