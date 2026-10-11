"""Start every Kaggle wave driver in TRACKS that is not already running; leave running ones alone.

    python code/cluster/kaggle/ensure_tracks.py

Safe to run any time (at logon, hourly, after a reboot or a trip): a restarted driver waits for and pulls its own
running shards before pushing anything (waves.py). Drivers run detached and append to results/E3_waves_<track>.log.
Remove a track from TRACKS when it prints "nothing left — done".
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[3]
USER = "joshuadavidpoznanski"
TRACKS = {
    # "main" (E3 minus SMT, 4 shards) finished 2026-10-10. Queue (5 Kaggle sessions each, one at a time):
    # smt -> reinstated (E3 D-2) -> gprestarts (N-14 sensitivity) -> tuning (symmetric budget arm)
    "smt": ["E3_learners", "--shards", "5", "--only-learners", "smt_krg,smt_kpls", "--tag", "smt", "--max-waves", "80"],
    "reinstated": ["E3_learners", "--shards", "5", "--only-learners", "idw,knn,qrf,ngboost,mc_dropout,poly_ridge",
                   "--tag", "re", "--max-waves", "10", "--start-after", "E3_learners:smt_krg,smt_kpls"],
    "gprestarts": ["E3s_gp_restarts", "--shards", "5", "--max-waves", "10",
                   "--start-after", "E3_learners:idw,knn,qrf,ngboost,mc_dropout,poly_ridge"],
    "tuning": ["E3s_tuning", "--shards", "5", "--max-waves", "10", "--start-after", "E3s_gp_restarts:"],
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
            ps = ("$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
                  f"@{{CommandLine='{line.replace(chr(39), chr(39) * 2)}'; CurrentDirectory='{ROOT}'}}; "
                  "exit $r.ReturnValue")
            rc = subprocess.run(["powershell", "-NoProfile", "-Command", ps]).returncode
            print(f"{name}: started" if rc == 0 else f"{name}: start FAILED (WMI code {rc})")
        else:
            subprocess.Popen(cmd, cwd=ROOT, stdout=open(log, "a"), stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
            print(f"{name}: started")


if __name__ == "__main__":
    main()
    try:                                                   # status e-mail when due (no-op without email_config.json)
        subprocess.run([sys.executable, str(Path(__file__).parent / "email_report.py")], timeout=600)
    except Exception as e:
        print("email report failed:", e)
