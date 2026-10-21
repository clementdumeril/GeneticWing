"""Small hyperparameter sweep of the real-coded GA on training instances (S1, S3).

Run: ``python scenarios/ga_tuning.py [--workers 8]``
"""

from __future__ import annotations

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from geneticwing.benchmarks import SCENARIOS  # noqa: E402
from geneticwing.evolution import RealGA  # noqa: E402
from geneticwing.soaring import SoaringProblem  # noqa: E402

BUDGET = 20_000
INSTANCES = (100, 101)
GRID = {
    "population": (50, 100, 200),
    "eta_crossover": (2.0, 15.0),
    "mutation_scale": (1.0, 4.0),  # mutation rate = scale / dim
}


def run(job):
    scenario, instance, population, eta_c, scale = job
    problem = SoaringProblem(SCENARIOS[scenario](instance))
    ga = RealGA(population=population, eta_crossover=eta_c, mutation_rate=scale / problem.dim)
    result = ga.minimize(problem.fitness, problem.dim, BUDGET, np.random.default_rng(instance), problem.straight_genome()[None])
    return job, result.f


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    configs = list(itertools.product(*GRID.values()))
    jobs = [(sc, inst, *cfg) for cfg in configs for sc in ("S1", "S3") for inst in INSTANCES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, jobs))
    print(f"{'pop':>5}{'eta_c':>7}{'mut':>5}{'mean f':>10}  (S1 and S3, 2 instances each; >= 1000 means infeasible)")
    ranked = []
    for cfg in configs:
        fs = [f for job, f in results if job[2:] == cfg]
        ranked.append((np.mean(fs), cfg))
    for mean_f, (pop, eta, scale) in sorted(ranked):
        print(f"{pop:>5}{eta:>7.0f}{scale:>5.0f}{mean_f:>10.2f}")


if __name__ == "__main__":
    main()
