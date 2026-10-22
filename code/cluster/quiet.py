"""Import first in any script that the auto-start task or a background driver runs: on Windows, every child process
(kaggle.exe, python, powershell) then starts without a console window. Without this, a parent that has no console
(pythonw from Task Scheduler) makes each child pop a new window on top of the user's work (reported 2026-10-13)."""
import subprocess
import sys

CREATE_NO_WINDOW = 0x08000000

if sys.platform == "win32" and not getattr(subprocess.Popen, "_surr_quiet", False):
    _orig_init = subprocess.Popen.__init__

    def _quiet_init(self, *args, **kwargs):
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | CREATE_NO_WINDOW
        _orig_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _quiet_init
    subprocess.Popen._surr_quiet = True
