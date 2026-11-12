"""NMPC baseline on the single-climb benchmark: IPOPT against the latent GA, same problem, same climbs.

The 49 held-out strip-wing climbs of ``piecewise_baseline.py`` (the d = 16
row of ``phase3_physics.py --strip --margins``): same command knots and bounds,
same simulator, cost and constraints. The NMPC (:mod:`geneticwing.nmpc`) is a
multi-start from the reactive law's four flights; every start is recorded, so
the table also gives the NMPC with one start (the best single seed, chosen
afterwards) and its time.

Run: ``python scenarios/nmpc_baseline.py --workers 4`` (then compare with
``results/piecewise_baseline/runs.json``, which holds the latent GA on the same climbs).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from geneticwing.benchmarks import canonical_scenario
from geneticwing.nmpc import NMPCPlanner
from geneticwing.soaring import INFEASIBLE_OFFSET, SoaringProblem
from geneticwing.strip import sbxc_wing

_W: dict = {}


def _init(theta, max_seconds):
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from piecewise_baseline import sim_config

    _W.update(cfg=sim_config(), planner=NMPCPlanner(sim_config(), theta, max_seconds=max_seconds))


def solve(job):
    instance, context = job
    problem = SoaringProblem(canonical_scenario(context, strip_wing=sbxc_wing()))
    tic = time.perf_counter()
    ev, _, names, statuses = _W["planner"].plan(context, _W["cfg"], problem)
    log = _W["planner"].log[-1]
    starts = [{"name": n, "status": s, "seconds": t, "feasible": bool(ev.fitness[k] < INFEASIBLE_OFFSET),
               "cost": float(ev.cost[k]) if ev.fitness[k] < INFEASIBLE_OFFSET else None}
              for k, (n, s, t) in enumerate(zip(names, statuses, log["start_seconds"]))]
    return instance, {"seconds": time.perf_counter() - tic, "starts": starts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("results/phase3_strip/dataset.npz"))
    parser.add_argument("--theta", type=Path, default=Path("results/phase3_strip/theta.npy"))
    parser.add_argument("--latent", type=Path, default=Path("results/piecewise_baseline/runs.json"))
    parser.add_argument("--out", type=Path, default=Path("results/nmpc_baseline"))
    parser.add_argument("--max-seconds", type=float, default=30.0, help="IPOPT CPU limit per start")
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count()))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = np.load(args.data)
    teacher = dict(zip(d["teacher_instance"].tolist(), d["teacher_best_cost"].tolist()))
    test_mask = d["rec_split"] == "test"
    test = [i for i in sorted(set(d["rec_instance"][test_mask].tolist())) if np.isfinite(teacher[i])][: args.max_test]
    jobs = [(i, d["rec_heading_context"][test_mask & (d["rec_instance"] == i)][0]) for i in test]
    rows, start = [], time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(np.load(args.theta), args.max_seconds)) as pool:
        for n, (i, out) in enumerate(pool.map(solve, jobs), 1):
            rows.append({"instance": i, "teacher": teacher[i], **out})
            (args.out / "runs.json").write_text(json.dumps(rows))
            best = min((s["cost"] for s in out["starts"] if s["feasible"]), default=None)
            print(f"[{time.perf_counter() - start:6.0f}s] {n}/{len(jobs)} climb {i}: NMPC "
                  f"{'inf' if best is None else f'{best:.2f}'} in {out['seconds']:.0f}s (teacher {teacher[i]:.2f})",
                  flush=True)
    summary = summarize(rows, args.latent)
    print("\n" + summary)
    (args.out / "summary.txt").write_text(summary)


def summarize(rows, latent_path: Path) -> str:
    t = np.array([r["teacher"] for r in rows])
    best = np.array([min((s["cost"] for s in r["starts"] if s["feasible"]), default=np.nan) for r in rows])
    names = [s["name"] for s in rows[0]["starts"]]
    lines = [f"{len(rows)} climbs, teacher mean cost {t.mean():.2f}",
             f"{'method':<34}{'solved':>8}{'mean cost':>11}{'vs teacher':>12}{'s / plan':>10}"]

    def line(name, cost, seconds):
        ok = np.isfinite(cost)
        lines.append(f"{name:<34}{100 * ok.mean():>7.0f}%{np.nanmean(cost):>11.2f}{np.nanmean(cost - t):>12.2f}"
                     f"{np.median(seconds):>10.1f}")

    line(f"NMPC, {len(names)} law starts", best, [r["seconds"] for r in rows])
    for j, name in enumerate(names):
        cost = np.array([r["starts"][j]["cost"] if r["starts"][j]["feasible"] else np.nan for r in rows])
        line(f"NMPC, one start ({name})", cost, [r["starts"][j]["seconds"] for r in rows])
    if latent_path.exists():
        lat = {(r["instance"], r["seed"]): r["methods"] for r in json.loads(latent_path.read_text())}
        for m in ("latent d=16 | real-coded GA", "latent d=16 | Zohdi GA"):
            cost = np.array([np.nanmean([lat[(r["instance"], s)][m]["cost"] if lat[(r["instance"], s)][m]["feasible"]
                                         else np.nan for s in range(3) if (r["instance"], s) in lat]) for r in rows])
            secs = [lat[(r["instance"], 0)][m]["s"] for r in rows]
            line(m + " (3 seeds)", cost, secs)
            wins = np.mean(best < cost)
            lines.append(f"  NMPC better than {m} on {100 * wins:.0f}% of climbs, by {np.nanmean(cost - best):.2f} on average")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
