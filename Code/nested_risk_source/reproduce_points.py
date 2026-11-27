"""Regenerate the Monte Carlo points in Supplementary Figure C.1."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from tempfile import TemporaryDirectory

import run_main_figures_n2000 as original
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
MODELS = ("a3", "a5")
GROUP = ["n", "p", "gamma", "model", "snr", "k", "method"]


def read_points(path, task):
    _, p, start = task
    rows = pd.read_csv(path)
    points = rows.loc[rows["model"].isin(MODELS)]
    expected = pd.MultiIndex.from_product(
        [MODELS, (0, *original.COUNTS), range(start, start + 10)],
        names=["model", "k", "rep"],
    )
    keys = pd.MultiIndex.from_frame(points[["model", "k", "rep"]])
    methods = points["k"].map(lambda k: f"PLS{k}" if k else "OLS")
    if (len(keys) != len(expected) or not keys.is_unique or not keys.isin(expected).all()
            or not points["n"].eq(original.N).all() or not points["p"].eq(p).all()
            or not points["gamma"].eq(p / original.N).all() or not points["snr"].eq(5).all()
            or not points["method"].eq(methods).all()
            or not np.isfinite(points["risk"]).all() or not points["risk"].ge(0).all()):
        raise ValueError(f"Unexpected simulation records in {path}")
    return points


def run_batch(task, work):
    original.OUT = work
    path = original.checkpoint(task)
    if path.exists():
        return read_points(path, task)

    # Publish a complete checkpoint even if another run shares this work directory.
    with TemporaryDirectory(dir=work) as tmp:
        original.OUT = Path(tmp)
        original.checkpoint(task).parent.mkdir()
        original.batch(task)
        generated = original.checkpoint(task)
        points = read_points(generated, task)
        generated.replace(path)
    return points


def simulate(tasks, work, workers):
    if workers == 1:
        yield from map(run_batch, tasks, repeat(work))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(run_batch, tasks, repeat(work))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1, help="parallel simulation processes")
    parser.add_argument("--work-dir", type=Path, default=CODE / "nested-risk-work",
                        help="directory for reusable ten-replication checkpoints")
    parser.add_argument("--output", type=Path, default=CODE / "results/nested-risk-simulation.csv")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    work = args.work_dir.resolve()
    (work / "checkpoints").mkdir(parents=True, exist_ok=True)
    # The original isotropic jobs also contain the two omitted-signal designs.
    tasks = [task for task in original.jobs() if task[0] == "iso"]
    batches = []
    for i, frame in enumerate(simulate(tasks, work, args.workers), 1):
        batches.append(frame)
        if i % 100 == 0 or i == len(tasks):
            print(f"{i}/{len(tasks)} batches ready", flush=True)

    samples = pd.concat(batches, ignore_index=True)
    summary = samples.groupby(GROUP, as_index=False).agg(
        mean=("risk", "mean"), std=("risk", "std"), repeats=("risk", "size")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        output = Path(tmp) / args.output.name
        summary.to_csv(output, index=False)
        output.replace(args.output)
    print(f"Wrote {len(summary)} rows to {args.output}")


if __name__ == "__main__":
    main()
