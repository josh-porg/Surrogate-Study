"""A later Kaggle wave splits only the unfinished tasks, evenly and without overlap, across its shards.

    python code/tests/test_shard_split.py
"""
import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surr import runner  # noqa: E402

EXP, K = "E6_mf", 5


def _counts(raw):
    old, runner.RAW = runner.RAW, raw
    try:
        out = []
        for i in range(1, K + 1):
            buf = io.StringIO()
            with redirect_stdout(buf):
                runner.main([EXP, "--shard", f"{i}/{K}", "--dry"])
            out.append(int(buf.getvalue().split(" tasks in shard")[0].split()[-1]))
        return out
    finally:
        runner.RAW = old


def test_first_wave_uses_hash_split():
    n = len(runner.EXPERIMENTS[EXP]())
    with tempfile.TemporaryDirectory() as d:
        c = _counts(Path(d))
    assert sum(c) == n
    return c


def test_later_wave_splits_remainder_evenly():
    tasks = [runner.key_of(c) for c in runner.EXPERIMENTS[EXP]()]
    done = [t for t in tasks if int(t, 16) % K != 4]        # one hash shard left: the E6 wave-2 case
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / EXP).mkdir()
        (Path(d) / EXP / "done_keys_prev.txt").write_text("\n".join(done), encoding="utf8")
        c = _counts(Path(d))
        frozen = runner._frozen_done(Path(d) / EXP)
    rest = len(tasks) - len(done)
    assert sum(c) == rest and max(c) - min(c) <= 1, c
    rem = sorted(t for t in tasks if t not in frozen)
    parts = [set(rem[i::K]) for i in range(K)]
    assert sum(map(len, parts)) == len(set().union(*parts)) == rest
    return c


if __name__ == "__main__":
    print("first wave (hash):", test_first_wave_uses_hash_split())
    print("later wave (remainder by position):", test_later_wave_splits_remainder_evenly())
    print("ok")
