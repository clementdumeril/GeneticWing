"""Phase 2: offline teacher study on scenarios S1 to S3 (hypothesis H1).

For every scenario and test instance, each planner gets the same genome, the
same straight-flight seed and the same evaluation budget:

* NLP          - SLSQP from the straight-flight genome (ARPL-style local planner)
* NLP-multi    - SLSQP restarted from the seed then random genomes
* GA           - real-coded GA (SBX, polynomial mutation)
* ZohdiGA      - the scheme of T. Zohdi's GA course
* CMA-ES       - covariance matrix adaptation
* CMA-ES+NLP   - half the budget for CMA-ES, then SLSQP from its best
* GA@100k, CMA-ES@100k - the evolutionary methods with 5x the budget
* Law          - the reactive orbit law tuned offline by ZohdiGA on separate
                 training instances (the "a la Zohdi" baseline)

Run: ``python scenarios/phase2_teacher.py [--out results/phase2] [--workers 8]``
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from geneticwing.benchmarks import SCENARIOS  # noqa: E402
from geneticwing.evolution import CMAES, RealGA, ZohdiGA  # noqa: E402
from geneticwing.planning.local import multistart, single_start, solve_local  # noqa: E402
from geneticwing.policies import simulate_law, tune_law  # noqa: E402
from geneticwing.soaring import SoaringProblem  # noqa: E402

BUDGET = 20_000
LARGE_BUDGET = 100_000
TEST_INSTANCES = (0, 1, 2)
TRAIN_INSTANCES = (100, 101, 102)
LAW_BUDGET = 1_200
CMA_POPULATION = 32


def cma_then_nlp(problem, budget, rng, seeds):
    half = budget // 2
    cma = CMAES(population=CMA_POPULATION).minimize(problem.fitness, problem.dim, half, rng, seeds)
    local, used = solve_local(problem, cma.x, budget - cma.evaluations)
    best = local if local.f < cma.f else cma
    best.evaluations = cma.evaluations + used
    best.history = cma.history + [(best.evaluations, min(cma.f, local.f))]
    return best


PLANNERS = {
    "NLP": lambda pb, rng, seeds: single_start(pb, BUDGET, seeds=seeds),
    "NLP-multi": lambda pb, rng, seeds: multistart(pb, BUDGET, rng, seeds),
    "GA": lambda pb, rng, seeds: RealGA().minimize(pb.fitness, pb.dim, BUDGET, rng, seeds),
    "ZohdiGA": lambda pb, rng, seeds: ZohdiGA().minimize(pb.fitness, pb.dim, BUDGET, rng, seeds),
    "CMA-ES": lambda pb, rng, seeds: CMAES(population=CMA_POPULATION).minimize(pb.fitness, pb.dim, BUDGET, rng, seeds),
    "CMA-ES+NLP": lambda pb, rng, seeds: cma_then_nlp(pb, BUDGET, rng, seeds),
    "GA@100k": lambda pb, rng, seeds: RealGA().minimize(pb.fitness, pb.dim, LARGE_BUDGET, rng, seeds),
    "CMA-ES@100k": lambda pb, rng, seeds: CMAES(population=CMA_POPULATION).minimize(
        pb.fitness, pb.dim, LARGE_BUDGET, rng, seeds
    ),
}


def summarize(problem: SoaringProblem, ev, index: int = 0) -> dict:
    return {
        "cost": float(ev.cost[index]),
        "feasible": bool(ev.feasible[index]),
        "violation": float(ev.violation[index]),
        "altitude_gain_m": float(ev.altitude_gain[index]),
        "propulsive_energy_Wh": float(ev.propulsive_energy[index] / 3600.0),
        "net_energy_Wh": float(ev.net_energy[index] / 3600.0),
    }


def run_planner(job):
    scenario_key, instance, method = job
    problem = SoaringProblem(SCENARIOS[scenario_key](instance))
    rng = np.random.default_rng(10_000 * instance + list(PLANNERS).index(method))
    seeds = problem.straight_genome()[None]
    start = time.perf_counter()
    result = PLANNERS[method](problem, rng, seeds)
    elapsed = time.perf_counter() - start
    ev = problem.evaluate(result.x[None])
    return {
        "scenario": scenario_key,
        "instance": instance,
        "method": method,
        "evaluations": int(result.evaluations),
        "seconds": elapsed,
        "genome": result.x.tolist(),
        "history": [(int(e), float(f)) for e, f in result.history],
        **summarize(problem, ev),
    }


def run_law(scenario_key):
    train = [SoaringProblem(SCENARIOS[scenario_key](i)) for i in TRAIN_INSTANCES]
    start = time.perf_counter()
    theta, _ = tune_law(train, LAW_BUDGET, np.random.default_rng(7))
    elapsed = time.perf_counter() - start
    rows = []
    for instance in TEST_INSTANCES:
        problem = SoaringProblem(SCENARIOS[scenario_key](instance))
        ev = simulate_law(problem, theta[None])
        rows.append(
            {
                "scenario": scenario_key,
                "instance": instance,
                "method": "Law",
                "evaluations": LAW_BUDGET * len(TRAIN_INSTANCES),
                "seconds": elapsed / len(TEST_INSTANCES),
                "theta": theta.tolist(),
                "history": [],
                **summarize(problem, ev),
            }
        )
    return rows


def table(rows: list[dict]) -> str:
    methods = list(PLANNERS) + ["Law"]
    lines = []
    for sc in SCENARIOS:
        lines.append(f"\n{sc}  (cost: 1 = level cruise on the motor, < 0 = net energy gain)")
        lines.append(f"{'method':<13}{'mean cost':>10}{'best':>8}{'feasible':>10}{'alt gain (m)':>14}{'E_prop (Wh)':>13}{'time (s)':>10}")
        for m in methods:
            rs = [r for r in rows if r["scenario"] == sc and r["method"] == m]
            if not rs:
                continue
            cost = [r["cost"] if r["feasible"] else np.nan for r in rs]
            lines.append(
                f"{m:<13}{np.nanmean(cost) if np.any(np.isfinite(cost)) else float('nan'):>10.2f}"
                f"{np.nanmin(cost) if np.any(np.isfinite(cost)) else float('nan'):>8.2f}"
                f"{sum(r['feasible'] for r in rs):>7}/{len(rs)}"
                f"{np.mean([r['altitude_gain_m'] for r in rs]):>14.1f}"
                f"{np.mean([r['propulsive_energy_Wh'] for r in rs]):>13.3f}"
                f"{np.mean([r['seconds'] for r in rs]):>10.1f}"
            )
        winners = {}
        for inst in TEST_INSTANCES:
            rs = [r for r in rows if r["scenario"] == sc and r["instance"] == inst and r["feasible"]]
            if rs:
                w = min(rs, key=lambda r: r["cost"])
                winners[inst] = f"{w['method']} ({w['cost']:.2f})"
        lines.append(f"teacher (best per instance): {winners}")
    return "\n".join(lines)


def plot(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from geneticwing.policies import lift_anchors

    methods = ["NLP", "GA", "CMA-ES", "CMA-ES+NLP", "CMA-ES@100k", "Law"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for col, sc in enumerate(SCENARIOS):
        problem = SoaringProblem(SCENARIOS[sc](0))
        top, side = axes[0, col], axes[1, col]
        for m in methods:
            r = next((r for r in rows if r["scenario"] == sc and r["instance"] == 0 and r["method"] == m), None)
            if r is None:
                continue
            if m == "Law":
                s = simulate_law(problem, np.asarray(r["theta"])[None]).state
            else:
                s = problem.evaluate(np.asarray(r["genome"])[None], keep_state=True).state
            label = f"{m} ({r['cost']:.2f}{'' if r['feasible'] else ', infeasible'})"
            top.plot(s.position[0, :, 0], s.position[0, :, 1], label=label, lw=1.2)
            side.plot(s.t[0], s.position[0, :, 2], lw=1.2)
        for c0, drift in lift_anchors(problem):
            top.plot(*c0, "k^", ms=8)
        top.plot(*problem.scenario.start[:2], "ko", ms=5)
        top.set(title=f"{problem.scenario.name}, instance 0", xlabel="x east (m)", ylabel="y north (m)", aspect="equal")
        top.legend(fontsize=7)
        side.set(xlabel="t (s)", ylabel="altitude (m)")
    fig.savefig(out / "phase2_trajectories.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for ax, sc in zip(axes, SCENARIOS):
        for m in ["NLP-multi", "GA", "ZohdiGA", "CMA-ES", "GA@100k", "CMA-ES@100k"]:
            curves = [r["history"] for r in rows if r["scenario"] == sc and r["method"] == m and r["history"]]
            if not curves:
                continue
            grid = np.linspace(1, max(c[-1][0] for c in curves), 200)
            ys = []
            for c in curves:
                e, f = np.array(c).T
                f = np.where(f < 1e3, f, np.nan)
                ys.append(np.interp(grid, e, f, left=np.nan))
            ax.plot(grid, np.nanmedian(np.array(ys), axis=0), label=m)
        law = [r["cost"] for r in rows if r["scenario"] == sc and r["method"] == "Law" and r["feasible"]]
        if law:
            ax.axhline(np.mean(law), color="k", ls="--", lw=1, label="Law (mean)")
        ax.set(title=sc, xlabel="evaluations", ylabel="best feasible cost (median)", xscale="log")
        ax.legend(fontsize=7)
    fig.savefig(out / "phase2_convergence.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/phase2"))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(sc, inst, m) for sc in SCENARIOS for inst in TEST_INSTANCES for m in PLANNERS]
    # Longest jobs first so the pool stays busy.
    jobs.sort(key=lambda j: ("100k" not in j[2], j[0] != "S2"))
    start = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        law_futures = [pool.submit(run_law, sc) for sc in SCENARIOS]
        for row in pool.map(run_planner, jobs):
            rows.append(row)
            print(f"[{time.perf_counter() - start:6.0f}s] {row['scenario']} #{row['instance']} {row['method']:<12} "
                  f"cost={row['cost']:7.2f} feasible={row['feasible']}", flush=True)
        for f in law_futures:
            rows.extend(f.result())

    (args.out / "phase2_results.json").write_text(json.dumps(rows, indent=1))
    summary = table(rows)
    (args.out / "phase2_summary.txt").write_text(summary)
    print(summary)
    plot(rows, args.out)
    print(f"done in {time.perf_counter() - start:.0f}s, wrote {args.out}")


if __name__ == "__main__":
    main()
