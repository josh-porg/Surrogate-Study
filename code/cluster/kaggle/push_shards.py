"""Package the study code as one archive in a private Kaggle dataset and push one private CPU notebook per shard.

    python code/cluster/kaggle/push_shards.py E1_canon --shards 6 --user joshuadavidpoznanski

The archive holds code/ (minus cluster tooling), previously finished results for the experiment (so shards resume),
and — only for experiments that need it — the read-only COMPASS surrogate package (compass/__init__.py +
compass/surrogate/, no data). Uses the author's Kaggle credentials from ~/.kaggle (never read here). Real-data
experiments are refused (STUDY_PLAN §4b data rule).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import quiet  # noqa: E402,F401  (no console windows for child processes)

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMPASS = ROOT.parent / "COMPASS"
KAGGLE = __import__("shutil").which("kaggle", path=str(Path(sys.executable).parent)) or "kaggle"
NEEDS_COMPASS = ("E1",)
# pinned to the laptop's versions so results are comparable; torch is left to Kaggle's (CPU use only)
# packages missing from Kaggle's image (python 3.13), installed offline from the surrpaper-wheels-cp313 dataset;
# numpy/scipy/xgboost/lightgbm/catboost/torch come from the image (versions recorded per shard in env_shard*.json)
PIP = ("pebble scikit-learn==1.9.1 smt==2.15.0 cma==4.5.0 gpytorch==1.15.2 mf2==2022.6.0 pygam==0.12.0 ngboost==0.5.11 "
       "casadi==3.8.1")      # casadi (IPOPT, E9): not in the wheels dataset; installed online by the fallback below
WHEELS_DS = "surrpaper-wheels-cp313"
TIMEOUTS = {"Efstar_refine": 3600, "E9_optim": 3600, "E9b_managed": 5400, "E5_sampling": 3600, "E8_upscale": 3600}       # per-task wall cap [s]; a 10-D IPOP-CMA refinement can exceed 15 min

KERNEL = r'''
import glob, json, os, platform, subprocess, sys, tarfile
W = "/kaggle/working"
import shutil
arch = glob.glob("/kaggle/input/**/proj.tar.gz", recursive=True)
if arch:
    tarfile.open(arch[0]).extractall(W)
else:                                          # Kaggle auto-extracted the archive: copy its root (holds proj/, COMPASS/)
    src = [p for p in glob.glob("/kaggle/input/**/proj/code", recursive=True) if os.path.isdir(p)][0]
    shutil.copytree(os.path.dirname(os.path.dirname(src)), W, dirs_exist_ok=True)
assert os.path.isdir(f"{{W}}/proj/code/surr"), os.listdir(W)
# offline install from the private wheels dataset (API-pushed notebooks may have no internet)
wh = sorted({{os.path.dirname(p) for p in glob.glob("/kaggle/input/**/*.whl", recursive=True)}})
for pkg in "{pip}".split():
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", *[f"--find-links={{d}}" for d in wh],
                        pkg], capture_output=True, text=True)
    if r.returncode and not pkg.startswith("smt=="):          # not in the wheels dataset: try PyPI (internet is on)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True, text=True)
    print(pkg, "ok" if r.returncode == 0 else "FAILED " + r.stderr[-400:], flush=True)
    if r.returncode and pkg.startswith("smt=="):
        # every pydoe 1.x declares numpy>=2.2.6 (image: 2.1.3); pydoe is pure Python and smt only uses it for LHS, so
        # install it and smt without dependency resolution, then prove a kriging fit works (BUGS N-10)
        for p2, nd in (("pydoe<2", True), ("jenn<3", False), (pkg, True)):
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", *(["--no-deps"] if nd else []),
                            *[f"--find-links={{d}}" for d in wh], p2], check=True)
        chk = subprocess.run([sys.executable, "-c", "import numpy as np; from smt.surrogate_models import KRG; "
                              "x = np.random.rand(20, 2); m = KRG(print_global=False); "
                              "m.set_training_values(x, np.sin(6 * x).sum(1)); m.train(); print(m.predict_values(x[:2]))"],
                             capture_output=True, text=True)
        print(pkg, "no-deps fallback", "ok" if chk.returncode == 0 else "FAILED " + chk.stderr[-400:], flush=True)
        if chk.returncode and {needs_smt}:     # error records count as done: never let a broken install "finish" SMT tasks
            raise SystemExit("SMT unusable on this image; not running SMT tasks")
env =subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
os.makedirs(f"{{W}}/proj/results/raw/{exp}", exist_ok=True)
json.dump(dict(python=platform.python_version(), cpus=os.cpu_count(), shard="{i}/{k}", freeze=env.splitlines()),
          open(f"{{W}}/proj/results/raw/{exp}/env_shard{i}of{k}.json", "w"))
os.chdir(f"{{W}}/proj/code")
try:
    import torch
    if torch.cuda.is_available():
        os.environ["SURR_DEVICE"] = "cuda"; print("GPU:", torch.cuda.get_device_name(0), flush=True)
except Exception as e:
    print("no torch/cuda:", e, flush=True)
rc = subprocess.run([sys.executable, "-m", "surr.runner", "{exp}", "--shard", "{i}/{k}", "--workers", str(os.cpu_count()),
                     "--timeout", "{timeout}", "--max-hours", "10.5", *({extra!r})]).returncode
if rc:                                         # make Kaggle report ERROR instead of COMPLETE so the wave driver stops
    raise SystemExit(f"runner exited with code {{rc}}")
'''


def sh(*args):
    print("+", " ".join(args[1:] if args[0] == KAGGLE else args), flush=True)
    subprocess.run(args, check=True)


def build_archive(exp: str, path: Path):
    with tarfile.open(path, "w:gz") as tf:
        skip = lambda ti: None if ("__pycache__" in ti.name or ti.name.endswith(".pyc") or "/cluster" in ti.name) else ti
        tf.add(ROOT / "code", arcname="proj/code", filter=skip)
        for f in [*(ROOT / "results").glob("*_shortlist.json"),         # inputs later experiments read (E3 ← E2a,
                  *((ROOT / "results").glob("fstar_table.json") if exp.startswith(("Efstar", "E9")) else [])]:  # ← f*
            tf.add(f, arcname=f"proj/results/{f.name}")
        prev = ROOT / "results" / "raw" / exp
        if prev.exists():                                                # only the finished task keys, not records
            sys.path.insert(0, str(ROOT / "code"))
            from surr.runner import _done_keys
            kf = path.parent / "done_keys.txt"
            kf.write_text("\n".join(sorted(_done_keys(prev))), encoding="utf8")
            tf.add(kf, arcname=f"proj/results/raw/{exp}/done_keys_prev.txt")
        if exp.startswith(NEEDS_COMPASS):
            tf.add(COMPASS / "compass" / "__init__.py", arcname="COMPASS/compass/__init__.py")
            tf.add(COMPASS / "compass" / "surrogate", arcname="COMPASS/compass/surrogate", filter=skip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp"); ap.add_argument("--shards", type=int, default=4); ap.add_argument("--user", required=True)
    ap.add_argument("--skip-learners", default=""); ap.add_argument("--only-learners", default="")
    ap.add_argument("--limit", type=int, default=0, help="tasks per shard (smoke tests)")
    ap.add_argument("--tag", default="", help="slug suffix, so a second track (e.g. smt) never touches the main shards")
    ap.add_argument("--gpu", action="store_true", help="GPU session (Kaggle weekly GPU quota); nets train on cuda")
    a = ap.parse_args()
    if a.exp.startswith("E11") or "fea" in a.exp.lower() or "rair" in a.exp.lower():
        sys.exit("refusing: real-data experiments stay on this machine (STUDY_PLAN §4b data rule)")
    slug = f"surrpaper-{a.exp.lower().replace('_', '-')}" + (f"-{a.tag}" if a.tag else "")
    extra = (["--skip-learners", a.skip_learners] if a.skip_learners else []) +             (["--only-learners", a.only_learners] if a.only_learners else []) + (["--limit", str(a.limit)] if a.limit else [])
    smt = {"smt_krg", "smt_kpls"}
    needs_smt = bool(smt & set(a.only_learners.split(","))) if a.only_learners else not smt <= set(a.skip_learners.split(","))
    with tempfile.TemporaryDirectory() as td:
        ds = (Path(td) / "ds").resolve(); ds.mkdir()
        build_archive(a.exp, ds / "proj.tar.gz")
        print(f"archive {(ds / 'proj.tar.gz').stat().st_size / 1e6:.2f} MB")
        (ds / "dataset-metadata.json").write_text(json.dumps(
            {"title": slug, "id": f"{a.user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}))
        exists = subprocess.run([KAGGLE, "datasets", "status", f"{a.user}/{slug}"], capture_output=True, text=True)
        if exists.returncode == 0 and "ready" in exists.stdout.lower():
            sh(KAGGLE, "datasets", "version", "-p", str(ds), "-m", "update")
        else:
            sh(KAGGLE, "datasets", "create", "-p", str(ds))                 # private unless --public
        import time
        for _ in range(60):                                                  # dataset must be processed first
            st = subprocess.run([KAGGLE, "datasets", "status", f"{a.user}/{slug}"], capture_output=True, text=True)
            if "ready" in st.stdout.lower():
                break
            time.sleep(10)
        print("dataset status:", st.stdout.strip())
        for i in range(1, a.shards + 1):
            kd = Path(td) / f"k{i}"; kd.mkdir()
            name = f"{slug}-shard-{i}of{a.shards}"
            (kd / "run.py").write_text(KERNEL.format(pip=PIP, exp=a.exp, i=i, k=a.shards,
                                                     extra=extra, needs_smt=needs_smt,
                                                     timeout=TIMEOUTS.get(a.exp, 900)))
            (kd / "kernel-metadata.json").write_text(json.dumps({
                "id": f"{a.user}/{name}", "title": name, "code_file": "run.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": bool(a.gpu), "enable_internet": True,
                "dataset_sources": [f"{a.user}/{slug}", f"{a.user}/{WHEELS_DS}"]}))
            sh(KAGGLE, "kernels", "push", "-p", str(kd), *(["--accelerator", "NvidiaTeslaT4"] if a.gpu else []))


if __name__ == "__main__":
    main()
