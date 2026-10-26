"""Start every Kaggle wave driver in TRACKS that is not already running; leave running ones alone.

    python code/cluster/kaggle/ensure_tracks.py

Safe to run any time (at logon, hourly, after a reboot or a trip): a restarted driver waits for and pulls its own
running shards before pushing anything (waves.py). Drivers run detached and append to results/E3_waves_<track>.log.
Remove a track from TRACKS when it prints "nothing left — done".
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import quiet  # noqa: E402,F401  (no console windows for child processes)

import subprocess
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[3]
USER = "joshuadavidpoznanski"
TRACKS = {
    # "main" (E3 minus SMT, 4 shards) finished 2026-10-10. Queue (5 Kaggle sessions each, one at a time):
    # smt -> reinstated (E3 D-2) -> gprestarts (N-14 sensitivity) -> e3x (D-9) -> fstar -> e12 -> tuning (symmetric budget arm)
    # "smt" finished 2026-10-12 (42 460 fits); "reinstated" finished 2026-10-13 (146 780 fits)
    # "gprestarts" finished 2026-10-15 (45 160 fits)
    # E3x (D-9): the 89 problems E3's core set left out, full design; ahead of the tuning check (selection first)
    # "e3x" finished 2026-10-24 (1 057 520 fits); "fstar" (refinement) finished 2026-10-25
    "e12": ["E12_rational", "--shards", "5", "--max-waves", "20", "--start-after", "Efstar_refine:",
            "--stop-local", "surr.runner E12_rational"],
    # E9 (optimization on the surrogate, 2026-10-25) ahead of the tuning check: it matters more for the paper.
    # One track at a time on Kaggle (the author's call, 2026-10-25): splitting the 5 sessions adds no capacity.
    "e9": ["E9_optim", "--shards", "5", "--max-waves", "10", "--start-after", "E12_rational:"],
    "tuning": ["E3s_tuning", "--shards", "5", "--max-waves", "10", "--start-after", "E9_optim:"],
    # coded 2026-10-25, still one at a time: gradient-enhanced (E13) -> multi-fidelity (E6) -> upscaling (E8) -> sampling (E5)
    "e13": ["E13_gradient", "--shards", "5", "--max-waves", "10", "--start-after", "E3s_tuning:"],
    "e6": ["E6_mf", "--shards", "5", "--max-waves", "10", "--start-after", "E13_gradient:"],
    "e8": ["E8_upscale", "--shards", "5", "--max-waves", "10", "--start-after", "E6_mf:"],
    "e5": ["E5_sampling", "--shards", "5", "--max-waves", "10", "--start-after", "E8_upscale:"],
    # gputest (2026-10-12): done — API GPU pushes run on CPU, no extra capacity
}


def _running(args):
    want = " ".join(args)
    for p in psutil.process_iter(["cmdline"]):
        cl = " ".join(p.info["cmdline"] or [])
        if "waves.py" in cl and cl.endswith(want):
            return True
    return False


def main():
    if (ROOT / "results" / "STOP_DRIVERS").exists():
        print("STOP_DRIVERS present - not starting anything (python code/cluster/kaggle/stop_tracks.py --resume)")
        return
    for name, args in TRACKS.items():
        args = args + ["--user", USER]
        if _running(args):
            print(f"{name}: running"); continue
        log = ROOT / "results" / f"E3_waves_{name}.log"
        cmd = [sys.executable, "-u", str(Path(__file__).parent / "waves.py"), *args]
        if sys.platform == "win32":
            # create through WMI so the driver is a child of the WMI service, not of whatever launched this script:
            # drivers started from an app's shell died with that app's job object (2026-10-07)
            line = f'cmd /c "{subprocess.list2cmdline(cmd)} >> "{log}" 2>&1"'
            # ShowWindow = 0 (SW_HIDE): the driver's console exists but is hidden, and its children inherit it, so
            # nothing pops up over the user's work (2026-10-13)
            ps = ("$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; "
                  "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
                  f"@{{CommandLine='{line.replace(chr(39), chr(39) * 2)}'; CurrentDirectory='{ROOT}'; "
                  "ProcessStartupInformation=$si}; exit $r.ReturnValue")
            rc = subprocess.run(["powershell", "-NoProfile", "-Command", ps]).returncode
            print(f"{name}: started" if rc == 0 else f"{name}: start FAILED (WMI code {rc})")
        else:
            subprocess.Popen(cmd, cwd=ROOT, stdout=open(log, "a"), stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
            print(f"{name}: started")


def ensure_local():
    """The laptop job queue (code/cluster/local_queue.py: R-AIR, E10 features, E4 mechanisms), started hidden the
    same way; it exits at once if already running or finished."""
    if (ROOT / "results" / "STOP_DRIVERS").exists():
        return
    if any("local_queue.py" in " ".join(p.info["cmdline"] or []) and "python" in (p.info["name"] or "").lower()
           for p in psutil.process_iter(["cmdline", "name"])):
        print("local queue: running"); return
    log = ROOT / "results" / "local_queue.log"
    cmd = [sys.executable, "-u", str(Path(__file__).resolve().parents[1] / "local_queue.py")]
    if sys.platform == "win32":
        line = f'cmd /c "{subprocess.list2cmdline(cmd)} >> "{log}" 2>&1"'
        ps = ("$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; "
              "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
              f"@{{CommandLine='{line.replace(chr(39), chr(39) * 2)}'; CurrentDirectory='{ROOT}'; "
              "ProcessStartupInformation=$si}; exit $r.ReturnValue")
        rc = subprocess.run(["powershell", "-NoProfile", "-Command", ps]).returncode
        print("local queue: started" if rc == 0 else f"local queue: start FAILED (WMI code {rc})")
    else:
        subprocess.Popen(cmd, cwd=ROOT, stdout=open(log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)


if __name__ == "__main__":
    main()
    ensure_local()
    try:                                                   # status e-mail when due (no-op without email_config.json)
        subprocess.run([sys.executable, str(Path(__file__).parent / "email_report.py")], timeout=600)
    except Exception as e:
        print("email report failed:", e)
