# Running experiment shards on Kaggle (free CPU notebooks)

One-time setup (author):
1. Create a Kaggle account, then *Settings → API → Create New Token*; save the downloaded `kaggle.json` to
   `C:\Users\<you>\.kaggle\kaggle.json`. (Claude never reads or types the token.)
2. `pip install kaggle` into the project venv (done by `push_shards.py` if missing).

Per experiment:
```bash
python code/cluster/kaggle/push_shards.py E3 --shards 6        # packages code, uploads it as a private dataset, pushes 6 notebooks
python code/cluster/kaggle/pull_results.py E3                   # downloads finished shard outputs into results/raw/E3/
```
Each notebook installs the pinned requirements, runs `python -m surr.runner <exp> --shard i/k --workers 4`, and writes
JSONL to `/kaggle/working`. Sessions stop at 12 h; because the runner skips finished task keys, re-pushing the same
shard continues where it stopped (copy its earlier output back in first — `pull_results.py` handles merging).

Only synthetic-problem experiments go to Kaggle. Datasets and notebooks are created **private**.

## Keeping data safe (laptop sleeps, travels, reboots)

- **Within a run:** every finished task is one flushed JSONL line, and a resumed run skips finished task keys. A shard
  killed at any point loses at most the tasks in flight.
- **Kaggle → laptop:** `waves.py` never pushes a wave until `pull_results.py` has fetched every shard of the previous
  one (exit 0 = download OK and kernel log present). Failed pulls and pushes are retried every 10 min indefinitely, so
  a closed lid, a flight or a network drop only delays the run. Kaggle keeps finished output until the next push.
- **Laptop → elsewhere:** after every pull, `code/cluster/backup_results.py` mirrors `results/` (gitignored raw data
  included) into `OneDrive - University of Kansas/SurrogatePaper_backup/results`. It never deletes anything there.
- **Drivers:** `python code/cluster/kaggle/ensure_tracks.py` starts any E3 track driver that isn't running and leaves
  running ones alone. Run it after a reboot. A restarted driver waits for and pulls its own running shards before
  pushing anything.

## Auto-start and the kill switch

A Windows scheduled task, **"SurrogatePaper Kaggle drivers"**, runs `ensure_tracks.py` at logon and every 30 min.
That restarts any driver that isn't running and never starts duplicates. Installed 2026-10-11 with the author's
approval.

- **Stop everything now:** `python code/cluster/kaggle/stop_tracks.py`. This creates `results/STOP_DRIVERS` and ends
  every driver. Nothing restarts while that file exists, and drivers exit at their next safe point, never mid-push.
- **Resume:** `python code/cluster/kaggle/stop_tracks.py --resume`
- **Remove the auto-start:** `python code/cluster/kaggle/stop_tracks.py --uninstall`
- **See what's running:** `python code/cluster/kaggle/stop_tracks.py --status`
- **Without Python:** create an empty file `results/STOP_DRIVERS`, or disable/delete the task in Task Scheduler.

Stopping loses nothing. Kernels already on Kaggle finish by themselves, and their output is pulled on the next start.

**Laptop job queue (2026-10-25).** The same task also starts `code/cluster/local_queue.py`, which runs the
local-only studies one after another at below-normal priority: R-AIR bias arm → E10 features → E4 mechanisms
(log: `results/local_queue.log`; finished steps: `results/local_queue_done.txt`). The same kill switch stops it
within 30 s, together with its worker processes. Every step resumes where it stopped.

**Kaggle order:** E12 → E9 → tuning check → E13 → E6 → E8 → E5, one track at a time (`TRACKS` in
`ensure_tracks.py`). Each waiting driver only checks every 30 min whether its predecessor has finished.

## Status e-mails

Every 30 minutes, `ensure_tracks.py` runs `email_report.py`. It sends through the author's Outlook, so no
passwords are kept in any file. Routine reports go out every 24 h. A report goes out immediately when a track
finishes, a driver logs an error or no progress, or the kill switch turns on. In that case the subject starts with
`[ACTION?]`.

The config is `results/email_config.json`: from/to `saturn5j@hotmail.com`, `every_hours: 24`. To turn the e-mails
off, set `"enabled": false`. `python code/cluster/kaggle/email_report.py --preview` shows the next report without
sending it.
