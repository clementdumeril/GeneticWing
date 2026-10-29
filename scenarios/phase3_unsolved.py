"""Phase 3: why the strip-wing teacher leaves situations unsolved.

The S1 family has a motor, so every situation has feasible trajectories (a
wide powered circle, for one); a situation without any teacher solution is an
optimizer failure. For each unsolved situation of a ``--strip`` dataset this
script records the constraints still violated after the teacher's NLP, then
retries the NLP from other seeds: the straight glide, the law with wider or
faster orbits, and the law seed solved without the teacher's margins.

Run: ``python scenarios/phase3_unsolved.py --data D/dataset.npz --theta D/theta.npy --count 128 --out D/unsolved.json``
(``--part k/n`` takes every n-th situation from the k-th, to split the work across machines).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from geneticwing.benchmarks import s1_family, with_margins  # noqa: E402
from geneticwing.planning.local import solve_local  # noqa: E402
from geneticwing.policies import simulate_law  # noqa: E402
from geneticwing.seeding import fit_genome  # noqa: E402
from geneticwing.soaring import SoaringProblem  # noqa: E402
from geneticwing.strip import sbxc_wing  # noqa: E402

DIRECTION_GENE, RADIUS_GENE, ORBIT_SPEED_GENE = 6, 0, 1
# Law variants: gene -> value (radius 20..150 m, orbit speed 0.85..1.5 of min-drag speed).
VARIANTS = {
    "law": {},
    "law_wide": {RADIUS_GENE: 0.45},  # about 80 m
    "law_wider": {RADIUS_GENE: 0.75},  # about 120 m
    "law_fast": {ORBIT_SPEED_GENE: 0.8},
    "law_wide_fast": {RADIUS_GENE: 0.45, ORBIT_SPEED_GENE: 0.8},
}


def _law_seed(problem, theta, ccw, overrides):
    th = np.array(theta)
    th[DIRECTION_GENE] = 0.75 if ccw else 0.25
    for gene, value in overrides.items():
        th[gene] = value
    law = simulate_law(problem, th[None])
    return fit_genome(problem, law.state.t[0], law.state.position[0])


def diagnose(job):
    instance, theta, nlp_budget = job
    scenario = replace(s1_family(instance)[0], strip_wing=sbxc_wing())
    nominal, robust = SoaringProblem(scenario), SoaringProblem(with_margins(scenario))
    out = {"instance": int(instance), "violated": [], "strategies": {}}

    def attempt(problem, x0):
        refined, _ = solve_local(problem, x0, nlp_budget)
        ev_r = robust.evaluate(refined.x[None])
        ev_n = nominal.evaluate(refined.x[None])
        return refined.x, ev_r, ev_n

    for ccw in (True, False):
        mode = "ccw" if ccw else "cw"
        for name, overrides in VARIANTS.items():
            x, ev_r, ev_n = attempt(robust, _law_seed(robust, theta, ccw, overrides))
            if name == "law":  # the teacher's own attempt: what is still violated
                cons = ev_r.constraints[0]
                out["violated"] += [f"{mode}:{c}" for c, v in zip(SoaringProblem.CONSTRAINTS, cons) if v > 0.0]
            out["strategies"][f"{name}_{mode}"] = float(ev_n.cost[0]) if ev_r.feasible[0] else None
        # Same seed, solved and judged without the margins.
        x, _, ev_n = attempt(nominal, _law_seed(robust, theta, ccw, {}))
        out["strategies"][f"no_margins_{mode}"] = float(ev_n.cost[0]) if ev_n.feasible[0] else None
    x, ev_r, ev_n = attempt(robust, robust.straight_genome())
    out["strategies"]["straight"] = float(ev_n.cost[0]) if ev_r.feasible[0] else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="merged --strip dataset")
    parser.add_argument("--theta", type=Path, required=True, help="the law that seeded the teacher")
    parser.add_argument("--count", type=int, default=128, help="unsolved situations: all test ones, then train ones")
    parser.add_argument("--part", default="0/1", help="k/n: every n-th situation from the k-th")
    parser.add_argument("--nlp-budget", type=int, default=6_000)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    d = np.load(args.data)
    inst, modes = d["teacher_instance"], d["teacher_feasible_modes"]
    unsolved = inst[modes == 0]
    order = np.concatenate([np.sort(unsolved[unsolved < 1000]), np.sort(unsolved[unsolved >= 1000])])
    k, n = (int(v) for v in args.part.split("/"))
    chosen = order[: args.count][k::n]
    theta = np.load(args.theta)
    start = time.perf_counter()
    print(f"{len(unsolved)} unsolved situations, diagnosing {len(chosen)} (part {args.part})", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, r in enumerate(pool.map(diagnose, [(int(s), theta, args.nlp_budget) for s in chosen])):
            results.append(r)
            if i % 8 == 0:
                print(f"[{time.perf_counter() - start:5.0f}s] {i + 1}/{len(chosen)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"done in {time.perf_counter() - start:.0f}s -> {args.out}", flush=True)
    summarize(results)


def summarize(results) -> None:
    n = len(results)
    names = list(results[0]["strategies"])
    print(f"\n{n} unsolved situations")
    for name in names:
        costs = [r["strategies"][name] for r in results if r["strategies"][name] is not None]
        mean = f"{np.mean(costs):6.2f}" if costs else "     -"
        print(f"  {name:20s} rescues {len(costs):4d} ({100 * len(costs) / n:3.0f} %), mean cost {mean}")
    margin = [r for r in results if any(r["strategies"][s] is not None for s in names if not s.startswith("no_margins"))]
    print(f"  any seed, with margins: {len(margin)} ({100 * len(margin) / n:.0f} %)")
    counts = {}
    for r in results:
        for v in {v.split(":")[1] for v in r["violated"]}:
            counts[v] = counts.get(v, 0) + 1
    print("  constraints still violated by the teacher's NLP (situations):",
          ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
