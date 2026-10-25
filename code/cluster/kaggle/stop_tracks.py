"""Kill switch and auto-start control for the Kaggle wave drivers.

    python code/cluster/kaggle/stop_tracks.py              STOP: create results/STOP_DRIVERS and end every driver now
    python code/cluster/kaggle/stop_tracks.py --resume     remove STOP_DRIVERS and start the drivers again
    python code/cluster/kaggle/stop_tracks.py --install    register the Windows logon / every-30-min auto-start task
    python code/cluster/kaggle/stop_tracks.py --uninstall  remove that task (and STOP, as above)
    python code/cluster/kaggle/stop_tracks.py --status     show the task, the switch and the running drivers

Stopping is always safe: results are written line by line and Kaggle keeps finished shard output until the next push,
which a restarted driver only makes after pulling. Kernels already running on Kaggle finish on their own (≤ 12 h).
Without Python, the same stop by hand: create an empty file results/STOP_DRIVERS, and in Task Scheduler disable or
delete the task "SurrogatePaper Kaggle drivers".
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
STOP = ROOT / "results" / "STOP_DRIVERS"
TASK = "SurrogatePaper Kaggle drivers"
HERE = Path(__file__).resolve().parent


def drivers():
    """Kaggle wave drivers and the laptop job queue, with the queue's runner and worker processes."""
    out = []
    for p in psutil.process_iter(["cmdline", "name"]):
        cl = " ".join(p.info["cmdline"] or [])
        # python processes only (and the hidden cmd wrapper), so shells whose command text mentions a script are spared
        if not any(t in (p.info["name"] or "").lower() for t in ("python", "cmd.exe")):
            continue
        if "waves.py" in cl or "local_queue.py" in cl:
            out.append(p)
            if "local_queue.py" in cl:
                try:
                    out += p.children(recursive=True)
                except psutil.NoSuchProcess:
                    pass
    return out


def ps(cmd):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)


def stop():
    STOP.write_text("Delete this file (or run stop_tracks.py --resume) to let the Kaggle drivers run again.\n")
    procs = drivers()
    for p in procs:
        p.terminate()
    psutil.wait_procs(procs, timeout=10)
    print(f"stopped {len(procs)} driver process(es); {STOP} created - nothing restarts until it is removed")


def install():
    pyw = Path(sys.executable).with_name("pythonw.exe")
    exe = pyw if pyw.exists() else Path(sys.executable)
    cmd = (f"$a = New-ScheduledTaskAction -Execute '{exe}' -Argument '\"{HERE / 'ensure_tracks.py'}\"' "
           f"-WorkingDirectory '{ROOT}'; "
           "$t1 = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME; "
           "$t2 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 30); "
           "$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable "
           "-MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5); "
           f"Register-ScheduledTask -TaskName '{TASK}' -Action $a -Trigger $t1,$t2 -Settings $s "
           "-Description 'Restarts the SurrogatePaper Kaggle wave drivers if they are not running. Stop: python "
           f"code/cluster/kaggle/stop_tracks.py (or create results/STOP_DRIVERS). Remove: stop_tracks.py --uninstall' -Force")
    r = ps(cmd)
    print("installed" if r.returncode == 0 else "install FAILED:\n" + r.stderr[-800:])


def uninstall():
    stop()
    r = ps(f"Unregister-ScheduledTask -TaskName '{TASK}' -Confirm:$false")
    print("task removed" if r.returncode == 0 else "no task removed: " + r.stderr.strip()[-300:])


def status():
    r = ps(f"Get-ScheduledTask -TaskName '{TASK}' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty State")
    print("auto-start task:", r.stdout.strip() or "not installed")
    print("kill switch    :", "ON (STOP_DRIVERS present)" if STOP.exists() else "off")
    for p in drivers():
        try:
            cl = " ".join(p.cmdline())
        except psutil.Error:
            continue
        if "waves.py" in cl:
            print("driver         :", p.pid, cl.split("waves.py")[-1].strip())
        elif "local_queue.py" in cl or "surr.runner" in cl or "e10_selector" in cl:
            print("local          :", p.pid, cl.split("python.exe")[-1].strip()[:110])


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--resume":
        STOP.unlink(missing_ok=True)
        subprocess.run([sys.executable, str(HERE / "ensure_tracks.py")])
    elif arg == "--install":
        install()
    elif arg == "--uninstall":
        uninstall()
    elif arg == "--status":
        status()
    else:
        stop()
