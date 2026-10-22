"""Status e-mails for the Kaggle runs, sent through the author's own Outlook (no passwords in any script).

    python code/cluster/kaggle/email_report.py --preview     print the report, send nothing
    python code/cluster/kaggle/email_report.py --send-now    send one report now (if enabled in the config)
    python code/cluster/kaggle/email_report.py               send only if due (called every 30 min by ensure_tracks.py)

Settings live in results/email_config.json: {"enabled": true, "from": "...", "to": "...", "every_hours": 24}.
A report is due every `every_hours`, and at once when something happens: a track finishes, a driver reports an error
or no progress, or the kill switch is turned on. Turn off: set "enabled": false or delete the config file.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import quiet  # noqa: E402,F401  (no console windows for child processes)

import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CFG = ROOT / "results" / "email_config.json"
STATE = ROOT / "results" / "email_state.json"
sys.path.insert(0, str(HERE))
from ensure_tracks import TRACKS  # noqa: E402

EVENT = re.compile(r"nothing left|made no progress|Traceback|FAILED|failed \(attempt|\[stop\]")


def _remaining(args):
    exp = args[0]
    flags = [args[i:i + 2] for i in range(1, len(args) - 1) if args[i] in ("--only-learners", "--skip-learners")]
    cmd = [sys.executable, "-m", "surr.runner", exp, "--dry"] + [x for f in flags for x in f]
    out = subprocess.run(cmd, cwd=ROOT / "code", capture_output=True, text=True).stdout.strip()
    m = re.search(r"(\d+) tasks in shard 1/1, (\d+) done, (\d+) to run", out)
    return tuple(map(int, m.groups())) if m else None


def build(offsets: dict):
    """offsets: log name -> bytes already reported; only lines appended since then count as new events. On the very
    first run ("_first") existing history is skipped."""
    lines, events, new_off = [], [], {}
    for name, args in TRACKS.items():
        r = _remaining(args)
        log = ROOT / "results" / f"E3_waves_{name}.log"
        tail = log.read_text(encoding="utf8", errors="replace").splitlines()[-400:] if log.exists() else []
        last = next((l for l in reversed(tail) if l.startswith(("[wave", "[resume", "[queued", "[stop", "nothing"))), "")
        state = ("finished" if r and r[2] == 0 else "queued" if last.startswith("[queued") else "running")
        if r:
            lines.append(f"- {name}: {state} | {r[1]:,} of {r[0]:,} done ({100 * r[1] / max(r[0], 1):.0f} %), "
                         f"{r[2]:,} left")
        else:
            lines.append(f"- {name}: status unavailable")
        lines.append(f"    last driver line: {last[:160]}")
        if log.exists():
            raw = log.read_bytes(); new_off[name] = len(raw)
            start = offsets.get(name, len(raw) if offsets.get("_first") else 0)
            fresh = raw[start:].decode("utf8", "replace")
            events += [f"{name}: {l.strip()[:200]}" for l in fresh.splitlines() if EVENT.search(l)][-5:]
    stop = (ROOT / "results" / "STOP_DRIVERS").exists()
    free = shutil.disk_usage(ROOT).free / 1e9
    body = [f"SurrogatePaper Kaggle runs - {datetime.now():%Y-%m-%d %H:%M}", "", "Tracks:", *lines, "",
            f"Kill switch: {'ON - drivers stopped' if stop else 'off'}   |   free disk: {free:.0f} GB", ""]
    if events:
        body += ["Since the last report:", *[f"  {e}" for e in events], ""]
    body += ["Stop everything: python code/cluster/kaggle/stop_tracks.py   (resume: --resume)",
             "Turn these e-mails off: set \"enabled\": false in results/email_config.json"]
    urgent = bool(stop or any(re.search(r"no progress|Traceback|FAILED|nothing left", e) for e in events))
    return "\n".join(body), urgent, new_off


def send(cfg, subject, body):
    import win32com.client as w
    o = w.Dispatch("Outlook.Application")
    m = o.CreateItem(0)
    m.To, m.Subject, m.Body = cfg["to"], subject, body
    acct = next((a for a in o.Session.Accounts if a.SmtpAddress.lower() == cfg["from"].lower()), None)
    if acct is not None:
        m._oleobj_.Invoke(*(64209, 0, 8, 0, acct))                      # SendUsingAccount
    m.Send()


def main():
    st = json.loads(STATE.read_text()) if STATE.exists() else {"last": 0.0, "offsets": {"_first": True}}
    body, urgent, off = build(st.get("offsets", {}))
    if "--preview" in sys.argv:
        print(body); return
    if not CFG.exists():
        return
    cfg = json.loads(CFG.read_text())
    due = time.time() - st["last"] > 3600 * cfg.get("every_hours", 24)
    if not cfg.get("enabled") or not ("--send-now" in sys.argv or due or urgent):
        return
    send(cfg, ("[ACTION?] " if urgent else "") + "SurrogatePaper runs: status", body)
    STATE.write_text(json.dumps({"last": time.time(), "offsets": off}))
    print("sent to", cfg["to"])


if __name__ == "__main__":
    main()
