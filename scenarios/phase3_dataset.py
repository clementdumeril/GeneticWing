"""Phase 3, step 1: the teacher's dataset on the randomized S1 family.

The phase 2 study made the teacher a portfolio: the reactive law tuned
offline, whose flights are fitted to a genome and refined by the NLP. Every
instance is solved twice, once per orbit direction (counter-clockwise and
clockwise). Both feasible solutions are kept, so the dataset holds the two
modes the student must learn to tell apart.

The teacher solves with tightened constraints (``with_margins``) and its
solutions are scored under the nominal ones: optimal solutions on their
active constraints break under the slightest imitation error.

Run: ``python scenarios/phase3_dataset.py [--train 700 --test 100] [--workers 8]``
Writes ``results/phase3/dataset.npz``.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from geneticwing.benchmarks import heading_frame_context, s1_family, with_margins  # noqa: E402
from geneticwing.planning.local import solve_local  # noqa: E402
from geneticwing.policies import simulate_law, tune_law  # noqa: E402
from geneticwing.seeding import fit_genome  # noqa: E402
from geneticwing.soaring import SoaringProblem  # noqa: E402

LAW_TRAINING = tuple(range(90_000, 90_004))
LAW_BUDGET = 1_200
NLP_BUDGET = 6_000
DIRECTION_GENE = 6  # > 0.5 is counter-clockwise


def solve_instance(job):
    instance, theta, nlp_budget = job
    scenario, context = s1_family(instance)
    problem = SoaringProblem(scenario)
    robust = SoaringProblem(with_margins(scenario))
    solutions = []
    for ccw in (True, False):
        th = np.array(theta)
        th[DIRECTION_GENE] = 0.75 if ccw else 0.25
        law = simulate_law(robust, th[None])
        genome = fit_genome(robust, law.state.t[0], law.state.position[0])
        refined, _ = solve_local(robust, genome, nlp_budget)
        if not robust.evaluate(refined.x[None]).feasible[0]:
            continue
        ev = problem.evaluate(refined.x[None])
        solutions.append(
            {
                "genome": refined.x,
                "cost": float(ev.cost[0]),
                "feasible": bool(ev.feasible[0]),
                "law_cost": float(law.cost[0]),
                "law_feasible": bool(law.feasible[0]),
                "ccw": ccw,
            }
        )
    return instance, (context, heading_frame_context(scenario)), solutions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=int, default=700)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path("results/phase3"))
    parser.add_argument("--law-budget", type=int, default=LAW_BUDGET)
    parser.add_argument("--nlp-budget", type=int, default=NLP_BUDGET)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    train_problems = [SoaringProblem(s1_family(i)[0]) for i in LAW_TRAINING]
    theta, f = tune_law(train_problems, args.law_budget, np.random.default_rng(3))
    print(f"[{time.perf_counter() - start:5.0f}s] law tuned on the family, mean fitness {f:.2f}", flush=True)

    instances = list(range(args.train + args.test))
    records = {"instance": [], "context": [], "heading_context": [], "genome": [], "cost": [], "ccw": [], "split": []}
    teacher = {"instance": [], "best_cost": [], "law_cost": [], "feasible_modes": []}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, (instance, (context, heading_context), solutions) in enumerate(pool.map(solve_instance, [(i, theta, args.nlp_budget) for i in instances], chunksize=4)):
            split = "train" if instance < args.train else "test"
            feasible = [s for s in solutions if s["feasible"]]
            for s in feasible:
                records["instance"].append(instance)
                records["context"].append(context)
                records["heading_context"].append(heading_context)
                records["genome"].append(s["genome"])
                records["cost"].append(s["cost"])
                records["ccw"].append(s["ccw"])
                records["split"].append(split)
            teacher["instance"].append(instance)
            teacher["best_cost"].append(min((s["cost"] for s in feasible), default=np.nan))
            teacher["law_cost"].append(min((s["law_cost"] for s in solutions if s["law_feasible"]), default=np.nan))
            teacher["feasible_modes"].append(len(feasible))
            if k % 50 == 0:
                print(f"[{time.perf_counter() - start:5.0f}s] {k + 1}/{len(instances)} instances", flush=True)

    np.savez(
        args.out / "dataset.npz",
        theta=theta,
        **{f"rec_{key}": np.array(value) for key, value in records.items()},
        **{f"teacher_{key}": np.array(value) for key, value in teacher.items()},
    )
    modes = np.array(teacher["feasible_modes"])
    print(
        f"done in {time.perf_counter() - start:.0f}s: {len(records['genome'])} solutions, "
        f"instances with 2 / 1 / 0 feasible modes: {(modes == 2).sum()} / {(modes == 1).sum()} / {(modes == 0).sum()}, "
        f"teacher mean cost {np.nanmean(teacher['best_cost']):.2f}, law alone {np.nanmean(teacher['law_cost']):.2f}"
    )


if __name__ == "__main__":
    main()
