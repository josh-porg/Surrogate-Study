"""Laptop job queue: run the local-only studies one after another, resumably, at below-normal priority.

    python code/cluster/local_queue.py            (started by kaggle/ensure_tracks.py; safe to start twice: exits if running)

Steps run in order; each is resumable (the runner skips finished task keys; the E10 feature step caches per cell), so
a reboot, sleep or kill only loses the tasks in flight. Stop: the same kill switch as the Kaggle drivers
(results/STOP_DRIVERS, or `python code/cluster/kaggle/stop_tracks.py`), checked every 30 s; the running step is
terminated with its worker processes. Log: results/local_queue.log.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import quiet  # noqa: E402,F401  (no console windows for child processes)

import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
STOP = ROOT / "results" / "STOP_DRIVERS"
PY = sys.executable
# (name, command, cwd). R-AIR is real data: it never leaves this machine (push_shards refuses it).
STEPS = [
    # runner steps stop after 1 h and are restarted at once (loop below): the runner sizes its worker pool from the
    # RAM free at start, so hourly restarts let it grow when the author frees memory (2.3 GB free on 2026-10-25 -> 1)
    ("E4_rair", [PY, "-u", "-m", "surr.runner", "E4_rair", "--workers", "8", "--timeout", "900", "--max-hours", "1"],
     ROOT / "code"),
    # E11 on the thesis FEA cache (read in place, read-only; real data, never leaves this machine)
    ("E11_fea", [PY, "-u", "-m", "surr.runner", "E11_fea", "--workers", "8", "--timeout", "900", "--max-hours", "1"],
     ROOT / "code"),
    ("E11_al", [PY, "-u", "-m", "surr.runner", "E11_al", "--workers", "8", "--timeout", "3600", "--max-hours", "1"],
     ROOT / "code"),
    ("E10_features", [PY, "-u", str(ROOT / "code" / "analysis" / "e10_selector.py"), "--features", "--workers", "8"], ROOT),
    ("E4_mechanisms", [PY, "-u", "-m", "surr.runner", "E4_mechanisms", "--workers", "8", "--timeout", "900",
                       "--max-hours", "1"], ROOT / "code"),
]
DONE = ROOT / "results" / "local_queue_done.txt"     # names of finished steps (a step that exits 0 with nothing left)


def _others_running():
    # exclude this process and its ancestors: a venv python.exe on Windows re-launches the base interpreter as a
    # child with the same command line, so the parent would otherwise count as "another" queue
    me = {psutil.Process().pid, *(a.pid for a in psutil.Process().parents())}
    return any("local_queue.py" in " ".join(p.info["cmdline"] or []) and p.pid not in me and
               "python" in (p.info["name"] or "").lower() for p in psutil.process_iter(["cmdline", "name"]))


def _kill(proc):
    try:
        kids = psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        kids = []
    for k in kids + [psutil.Process(proc.pid)]:
        try:
            k.terminate()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(kids, timeout=15)


def main():
    if _others_running():
        print("local queue already running"); return
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    for name, cmd, cwd in STEPS:
        if name in done:
            continue
        if STOP.exists():
            print("[stop] STOP_DRIVERS present - local queue not starting", flush=True); return
        prev = None
        while True:
            if STOP.exists():
                print("[stop] STOP_DRIVERS present - local queue stopping", flush=True); return
            print(f"[{time.strftime('%Y-%m-%d %H:%M')}] {name}: start", flush=True)
            proc = subprocess.Popen(cmd, cwd=cwd)
            try:
                psutil.Process(proc.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 10)
            except Exception:
                pass
            t_start, watchdog = time.time(), False
            while proc.poll() is None:
                if STOP.exists():
                    print(f"[stop] STOP_DRIVERS present - ending {name} (resumes on the next start)", flush=True)
                    _kill(proc); return
                # watchdog: the runner checks --max-hours only when a task finishes, so a stalled start (2026-10-25:
                # 21 h with 2.5 s of CPU under memory pressure) never ended; the queue enforces 70 min itself
                if name.startswith(("E4", "E11")) and time.time() - t_start > 70 * 60:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M')}] {name}: watchdog - ending the runner after 70 min "
                          "(resumable)", flush=True)
                    _kill(proc); watchdog = True
                    break
                time.sleep(30)
            if watchdog:
                time.sleep(5); continue
            if proc.returncode != 0:
                print(f"[{time.strftime('%Y-%m-%d %H:%M')}] {name}: exited {proc.returncode}; will retry on next start",
                      flush=True)
                return
            if not name.startswith(("E4", "E11")):
                break
            # runner steps: a clean exit is also the hourly stop or the disk guard; done only when nothing is left
            r = subprocess.run([PY, "-m", "surr.runner", name, "--dry"], cwd=ROOT / "code", capture_output=True, text=True)
            if " 0 to run" in r.stdout:
                break
            left = r.stdout.split(" done, ")[-1].split(" to run")[0]
            if left == prev:                                             # no progress (e.g. disk guard): wait for the
                print(f"  {name}: no progress; waiting for the next scheduled start", flush=True); return   # next start
            prev = left
            print(f"  {name}: {r.stdout.strip()} - restarting (re-sizes the worker pool to free RAM)", flush=True)
            time.sleep(5)
        with DONE.open("a") as fh:
            fh.write(name + "\n")
        print(f"[{time.strftime('%Y-%m-%d %H:%M')}] {name}: done", flush=True)
    print("local queue: all steps done", flush=True)


if __name__ == "__main__":
    main()
