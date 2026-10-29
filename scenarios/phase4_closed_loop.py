"""Phase 4: the planners flown in closed loop, in turbulence and with a wrong belief.

Every test situation of a strip dataset is flown for 60 s by each method, in
several conditions of the air:

* calm        - the air the planners believe in, no turbulence (tracking losses only)
* turbulence  - Dryden turbulence (``--w20``) on top of the situation's wind
* estimate    - the planner's thermal centre, w* and wind are off (``--center-sigma``...)
* both        - turbulence and the wrong belief

Methods (``_open``: planned once and flown; ``_mpc``: replanned every
``--replan-every`` seconds from the state reached):

* teacher - the dataset's teacher (law seeds, NLP with margins)
* ga      - the GA over the command VAE (``--search-budget`` flights per plan)
* prior   - the best of 64 prior samples of the command VAE, no search

In one condition every method flies the same turbulence and the same wrong
belief. Run: ``python scenarios/phase4_closed_loop.py --data D/dataset.npz --theta D/theta.npy --vae D/physics/command_vae_d16.pt``
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
import torch  # noqa: E402

from geneticwing.aircraft import SBXC_APPROX  # noqa: E402
from geneticwing.benchmarks import MOTOR_THRUST, s1_family  # noqa: E402
from geneticwing.closedloop import (  # noqa: E402
    LatentPlanner,
    LoopConfig,
    PriorPlanner,
    TeacherPlanner,
    fly_closed_loop,
    sample_belief,
    turbulent_air,
)
from geneticwing.latent import ConditionalVAE  # noqa: E402
from geneticwing.strip import sbxc_wing  # noqa: E402
from geneticwing.torchsim import SimConfig  # noqa: E402

CONDITIONS = ("calm", "turbulence", "estimate", "both")
METHODS = ("teacher_open", "teacher_mpc", "ga_open", "ga_mpc", "prior_open", "prior_mpc")
_STATE: dict = {}


def _init_worker(vae_state, cfg):
    torch.set_num_threads(1)
    c = vae_state["config"]
    vae = ConditionalVAE(c["dim"], 6, c["latent_dim"], cfg["width"], cfg["depth"], c["beta"], device="cpu")
    vae.load(vae_state)
    sim = SimConfig(SBXC_APPROX, dt=cfg["dt"], knots=cfg["knots"], cl_limit=0.75, bank_max_deg=38.0,
                    load_max=2.1, thrust_max=0.8 * MOTOR_THRUST)  # the teacher's margins, as phase3_physics --margins
    _STATE.update(cfg=cfg, planners={
        "teacher": TeacherPlanner(cfg["theta"], cfg["nlp_budget"]),
        "ga": LatentPlanner(vae, sim, cfg["search_budget"]),
        "prior": PriorPlanner(vae, sim, 64),
    })


def run(job):
    instance, condition, method = job
    cfg = _STATE["cfg"]
    situation = replace(s1_family(instance)[0], strip_wing=sbxc_wing())
    k = CONDITIONS.index(condition)
    turbulent = condition in ("turbulence", "both")
    wrong = condition in ("estimate", "both")
    air = turbulent_air(situation, cfg["w20"] if turbulent else 0.0, cfg["duration"], seed=instance)
    # The same wrong belief in "estimate" and "both", so that "both" only adds the turbulence.
    belief = sample_belief(situation, np.random.default_rng([instance, 1]), *(cfg["errors"] if wrong else (0.0, 0.0, 0.0)))
    name, mode = method.rsplit("_", 1)
    loop = LoopConfig(duration=cfg["duration"], replan_every=None if mode == "open" else cfg["replan_every"])
    rng = np.random.default_rng([instance, k, METHODS.index(method)])
    tic = time.perf_counter()
    flight = fly_closed_loop(situation, air, belief, _STATE["planners"][name], loop, rng)
    return {
        "instance": instance, "condition": condition, "method": method,
        "cost": float(flight.evaluation.cost[0]), "feasible": flight.feasible,
        "stalled": flight.stalled, "violated": flight.violated,
        "duration": float(flight.state.t[0, -1]),
        "predicted": [p.predicted_cost for p in flight.plans],
        "predicted_feasible": [p.predicted_feasible for p in flight.plans],
        "plan_seconds": flight.plan_seconds, "wall": time.perf_counter() - tic,
    }


def summarize(results, conditions, methods) -> str:
    lines = [f"{'condition':<11} | {'method':<13}{'flown cost':>11}{'no violation':>14}{'stalled':>9}"
             f"{'planned':>9}{'plan s':>8}"]
    for c in conditions:
        for m in methods:
            rows = [r for r in results if r["condition"] == c and r["method"] == m]
            if not rows:
                continue
            cost = np.array([r["cost"] for r in rows])
            lines.append(
                f"{c:<11} | {m:<13}{cost.mean():>11.2f}{100 * np.mean([r['feasible'] for r in rows]):>13.0f}%"
                f"{100 * np.mean([r['stalled'] for r in rows]):>8.0f}%{np.mean([r['predicted'][0] for r in rows]):>9.2f}"
                f"{np.median([s for r in rows for s in r['plan_seconds']]):>8.1f}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="strip dataset (its test situations are flown)")
    parser.add_argument("--theta", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True, help="command VAE trained with --margins")
    parser.add_argument("--out", type=Path, default=Path("results/phase4"))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--replan-every", type=float, default=10.0)
    parser.add_argument("--w20", type=float, default=7.5, help="Dryden intensity, wind 20 ft above ground (m/s)")
    parser.add_argument("--center-sigma", type=float, default=15.0, help="thermal centre error, m per axis")
    parser.add_argument("--w-star-sigma", type=float, default=0.2, help="log-normal error on w*")
    parser.add_argument("--wind-sigma", type=float, default=0.5, help="wind error, m/s per axis")
    parser.add_argument("--search-budget", type=int, default=3_000)
    parser.add_argument("--nlp-budget", type=int, default=6_000)
    parser.add_argument("--knots", type=int, default=61)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    d = np.load(args.data)
    test = sorted(set(d["rec_instance"][d["rec_split"] == "test"].tolist()))[: args.max_test]
    cfg = {"theta": np.load(args.theta), "nlp_budget": args.nlp_budget, "search_budget": args.search_budget,
           "duration": args.duration, "replan_every": args.replan_every, "w20": args.w20,
           "errors": (args.center_sigma, args.w_star_sigma, args.wind_sigma),
           "knots": args.knots, "dt": args.dt, "width": args.width, "depth": args.depth}
    state = torch.load(args.vae, weights_only=False, map_location="cpu")
    # The slowest jobs first, so the pool does not wait on a long tail.
    weight = {"teacher_mpc": 0, "ga_mpc": 1, "teacher_open": 2, "ga_open": 3, "prior_mpc": 4, "prior_open": 5}
    jobs = sorted(((i, c, m) for c in args.conditions for m in args.methods for i in test), key=lambda j: weight[j[2]])
    print(f"{len(test)} situations x {len(args.conditions)} conditions x {len(args.methods)} methods = "
          f"{len(jobs)} flights, {args.workers} workers", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(state, cfg)) as pool:
        for k, r in enumerate(pool.map(run, jobs, chunksize=1)):
            results.append(r)
            if k % 50 == 0:
                print(f"[{time.perf_counter() - start:6.0f}s] {k + 1}/{len(jobs)}", flush=True)
    (args.out / "closed_loop.json").write_text(json.dumps(results, indent=1))
    text = summarize(results, args.conditions, args.methods)
    (args.out / "closed_loop_summary.txt").write_text(text)
    print("\n" + text)
    print(f"done in {time.perf_counter() - start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
