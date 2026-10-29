"""Phase 3, step 1: the teacher's dataset on the randomized S1 family.

The phase 2 study made the teacher a portfolio: the reactive law tuned
offline, whose flights are fitted to a genome and refined by the NLP. Every
instance is solved twice, once per orbit direction (counter-clockwise and
clockwise). Both feasible solutions are kept, so the dataset holds the two
modes the student must learn to tell apart.

The teacher solves with tightened constraints (``with_margins``) and its
solutions are scored under the nominal ones: optimal solutions on their
active constraints break under the slightest imitation error. With
``--strip``, every problem carries the strip-theory wing (inner-wing stall,
ailerons, aileron drag). The law is still tuned without it: it only seeds
the NLP, and tuned under the strip wing it converges to orbits the NLP cannot
repair (3 instances of 16 feasible against 13 of 16, test costs +0.45 against
-5.41).

Run, whole dataset at once: ``python scenarios/phase3_dataset.py --train 600 --test 80``
(writes ``results/phase3/dataset.npz``). On a cluster, in shards sharing one law:

    python scenarios/phase3_dataset.py --tune-law-only --out D
    python scenarios/phase3_dataset.py --theta D/theta.npy --first 1000 --count 500 --name shard_000 --out D/shards
    python scenarios/phase3_merge.py D/shards --out D/dataset.npz
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from geneticwing.strip import sbxc_wing  # noqa: E402

LAW_TRAINING = tuple(range(90_000, 90_004))
LAW_BUDGET = 1_200
NLP_BUDGET = 6_000
DIRECTION_GENE = 6  # > 0.5 is counter-clockwise


def _family(instance: int, strip: bool):
    scenario, context = s1_family(instance)
    return (replace(scenario, strip_wing=sbxc_wing()) if strip else scenario), context


def solve_instance(job):
    instance, theta, nlp_budget, strip = job
    scenario, context = _family(instance, strip)
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


def tune_family_law(budget: int) -> tuple[np.ndarray, float]:
    """Tune the seed law on the point-mass wing, also for ``--strip`` datasets."""
    train_problems = [SoaringProblem(_family(i, strip=False)[0]) for i in LAW_TRAINING]
    return tune_law(train_problems, budget, np.random.default_rng(3))


def generate(instances, splits, theta, nlp_budget, workers, start, path: Path, strip: bool = False) -> None:
    """Solve ``instances`` and save one dataset file (a full dataset or a shard)."""
    records = {"instance": [], "context": [], "heading_context": [], "genome": [], "cost": [], "ccw": [], "split": []}
    teacher = {"instance": [], "best_cost": [], "law_cost": [], "feasible_modes": []}
    split_of = dict(zip(instances, splits))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(i, theta, nlp_budget, strip) for i in instances]
        for k, (instance, (context, heading_context), solutions) in enumerate(pool.map(solve_instance, jobs, chunksize=4)):
            feasible = [s for s in solutions if s["feasible"]]
            for s in feasible:
                records["instance"].append(instance)
                records["context"].append(context)
                records["heading_context"].append(heading_context)
                records["genome"].append(s["genome"])
                records["cost"].append(s["cost"])
                records["ccw"].append(s["ccw"])
                records["split"].append(split_of[instance])
            teacher["instance"].append(instance)
            teacher["best_cost"].append(min((s["cost"] for s in feasible), default=np.nan))
            teacher["law_cost"].append(min((s["law_cost"] for s in solutions if s["law_feasible"]), default=np.nan))
            teacher["feasible_modes"].append(len(feasible))
            if k % 50 == 0:
                print(f"[{time.perf_counter() - start:5.0f}s] {k + 1}/{len(instances)} instances", flush=True)

    np.savez(
        path,
        theta=theta,
        **{f"rec_{key}": np.array(value) for key, value in records.items()},
        **{f"teacher_{key}": np.array(value) for key, value in teacher.items()},
    )
    modes = np.array(teacher["feasible_modes"])
    print(
        f"done in {time.perf_counter() - start:.0f}s: {len(records['genome'])} solutions, "
        f"instances with 2 / 1 / 0 feasible modes: {(modes == 2).sum()} / {(modes == 1).sum()} / {(modes == 0).sum()}, "
        f"teacher mean cost {np.nanmean(teacher['best_cost']):.2f}, law alone {np.nanmean(teacher['law_cost']):.2f}"
        f" -> {path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=int, default=700, help="whole-dataset mode: instances 0 .. train-1")
    parser.add_argument("--test", type=int, default=100, help="whole-dataset mode: the next instances")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path("results/phase3"))
    parser.add_argument("--law-budget", type=int, default=LAW_BUDGET)
    parser.add_argument("--nlp-budget", type=int, default=NLP_BUDGET)
    parser.add_argument("--tune-law-only", action="store_true", help="tune the law, save <out>/theta.npy, stop")
    parser.add_argument("--theta", type=Path, help="reuse a tuned law instead of tuning one")
    parser.add_argument("--first", type=int, help="shard mode: first instance id")
    parser.add_argument("--count", type=int, help="shard mode: number of instances")
    parser.add_argument("--split", choices=("train", "test"), default="train", help="shard mode: split label")
    parser.add_argument("--name", default="dataset", help="output file name, without .npz")
    parser.add_argument("--strip", action="store_true", help="teacher and scoring with the strip-theory wing")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    if args.theta is not None:
        theta = np.load(args.theta)
        print(f"law loaded from {args.theta}", flush=True)
    else:
        theta, f = tune_family_law(args.law_budget)
        print(f"[{time.perf_counter() - start:5.0f}s] law tuned on the family, mean fitness {f:.2f}", flush=True)
    if args.tune_law_only:
        np.save(args.out / "theta.npy", theta)
        print(f"wrote {args.out / 'theta.npy'}")
        return

    if args.first is not None:
        instances = list(range(args.first, args.first + args.count))
        splits = [args.split] * len(instances)
    else:
        instances = list(range(args.train + args.test))
        splits = ["train" if i < args.train else "test" for i in instances]
    generate(instances, splits, theta, args.nlp_budget, args.workers, start, args.out / f"{args.name}.npz", args.strip)


if __name__ == "__main__":
    main()
