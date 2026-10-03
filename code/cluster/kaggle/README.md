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
