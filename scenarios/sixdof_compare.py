"""Plan with the point mass, fly with the rigid body: how much does the 3-DOF model flatter the climbs?

Every planner plans its 60 s climb with the point mass, as in the papers'
results. Each plan is then flown three ways, in the same air:

* ``3dof``      the point mass and its tracker (attitude lag 0.3 s, bank rate 60 deg/s), as in the results;
* ``3dof_soft`` the point mass with the 6-DOF's softer tracker gains (separates the guidance's share);
* ``6dof``      the rigid-body SB-XC of :mod:`geneticwing.sixdof` (estimated derivatives), its autopilot taking
                the same C_L, bank and thrust commands through elevator, ailerons and rudder.

``open``: one plan, flown 60 s; ``mpc``: replanned every 10 s from the state
reached. The open-loop plan is the same for the three airframes (cached), so
the cost differences are the airframe's alone.

Planners: ``law`` (the tuned reactive law's flight, instant), ``teacher``
(law seeds + NLP with margins, needs ``--theta``), ``ga`` (latent GA over the
command VAE, needs ``--vae``), ``fast`` (the distilled generator, needs
``--vae`` and ``--flow``). Situations: the weather family's test ids 600-679,
strip wing on, motor available (15 N), as the weather chain's dataset.

Local smoke test (4 workers at most on the big PC)::

    python scenarios/sixdof_compare.py --theta results/phase3_strip_v2/theta.npy --planners law \
        --count 16 --workers 4 --out results/sixdof_smoke

On Savio, with the weather chain's artifacts: ``slurm/sixdof.sbatch``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from geneticwing.benchmarks import weather_family  # noqa: E402
from geneticwing.closedloop import LoopConfig, Plan, _select, exact_belief, fly_closed_loop, turbulent_air  # noqa: E402
from geneticwing.policies import seed_theta, simulate_law  # noqa: E402
from geneticwing.soaring import SoaringProblem  # noqa: E402
from geneticwing.strip import sbxc_wing  # noqa: E402

AIRFRAMES = ("3dof", "3dof_soft", "6dof")
MODES = ("open", "mpc")
PLANNERS = ("law", "teacher", "ga", "fast")
_STATE: dict = {}


class LawPlanner:
    """The tuned reactive law's flight from the state reached, used as a plan (no search: the cheap planner)."""

    def __init__(self, theta):
        self.theta = np.asarray(theta)

    def __call__(self, scenario, rng, start=None):
        problem = SoaringProblem(scenario)
        ccw = True if start is None or abs(start.bank) < np.radians(10.0) else bool(start.bank < 0.0)
        ev = simulate_law(problem, seed_theta(self.theta, ccw, {})[None])
        return Plan(_select(ev.state, 0), float(ev.cost[0]), bool(ev.feasible[0]))


class FirstPlanCache:
    """The first plan of a flight is computed once per situation and shared by the airframes."""

    def __init__(self, planner, key):
        self.planner, self.key = planner, key

    def __call__(self, scenario, rng, start=None):
        if start is None:
            cache = _STATE.setdefault("cache", {})
            if self.key not in cache:
                cache[self.key] = self.planner(scenario, rng, start)
            return cache[self.key]
        return self.planner(scenario, rng, start)


def _init_worker(cfg):
    _STATE.update(cfg=cfg, planners={})
    if cfg.get("theta"):
        from geneticwing.closedloop import TeacherPlanner

        theta = np.load(cfg["theta"])
        _STATE["planners"]["law"] = LawPlanner(theta)
        _STATE["planners"]["teacher"] = TeacherPlanner(theta, cfg["nlp_budget"])
    if cfg.get("vae"):
        import torch

        from geneticwing.aircraft import SBXC_APPROX
        from geneticwing.benchmarks import MOTOR_THRUST
        from geneticwing.closedloop import DistilledPlanner, LatentPlanner, load_distilled
        from geneticwing.latent import ConditionalVAE, context_dim_of
        from geneticwing.torchsim import SimConfig

        torch.set_num_threads(1)
        state = torch.load(cfg["vae"], map_location="cpu", weights_only=False)
        c = state["config"]
        vae = ConditionalVAE(c["dim"], context_dim_of(state), c["latent_dim"], 512, 3, c["beta"], device="cpu")
        vae.load(state)
        sim = SimConfig(SBXC_APPROX, dt=0.4, knots=61, cl_limit=0.75, bank_max_deg=38.0, load_max=2.1,
                        thrust_max=0.8 * MOTOR_THRUST)  # the teacher's margins
        _STATE["planners"]["ga"] = LatentPlanner(vae, sim, cfg["search_budget"])
        if cfg.get("flow"):
            _STATE["planners"]["fast"] = DistilledPlanner(vae, sim, load_distilled(cfg["flow"], c["latent_dim"]), 64)


def _loop(cfg, airframe, mode):
    from geneticwing.sixdof import sbxc_estimated, sixdof_tracker

    loop = LoopConfig(duration=cfg["duration"], replan_every=None if mode == "open" else cfg["replan_every"],
                      inner_wing_protection=0.9)
    if airframe == "3dof_soft":
        loop = replace(loop, tracker=sixdof_tracker(loop.tracker))
    elif airframe == "6dof":
        loop = replace(loop, sixdof=sbxc_estimated(inertia_scale=cfg["inertia_scale"]))
    return loop


def run(job):
    instance, planner_name, mode = job
    cfg = _STATE["cfg"]
    situation = replace(weather_family(instance)[0], strip_wing=sbxc_wing())
    air = turbulent_air(situation, cfg["w20"], cfg["duration"], seed=instance)
    belief = exact_belief(situation)
    planner = FirstPlanCache(_STATE["planners"][planner_name], (instance, planner_name))
    rows = []
    for airframe in cfg["airframes"]:
        rng = np.random.default_rng([instance, PLANNERS.index(planner_name), MODES.index(mode)])
        tic = time.perf_counter()
        try:
            flight = fly_closed_loop(situation, air, belief, planner, _loop(cfg, airframe, mode), rng)
        except Exception as exc:  # a flight that breaks the integrator counts as lost, and is reported
            rows.append(dict(instance=instance, planner=planner_name, mode=mode, airframe=airframe, error=repr(exc)))
            continue
        s = flight.state
        ref = flight.plans[0].reference
        first = s.t[0] <= (ref.t[-1] if mode == "open" else cfg["replan_every"])
        planned = np.stack([np.interp(s.t[0][first], ref.t, ref.position[:, i]) for i in range(3)], axis=-1)
        rows.append(dict(
            instance=instance, planner=planner_name, mode=mode, airframe=airframe,
            cost=float(flight.evaluation.cost[0]), feasible=flight.feasible, stalled=flight.stalled,
            violated=flight.violated, altitude_gain=float(s.position[0, -1, 2] - s.position[0, 0, 2]),
            planned_cost=float(flight.plans[0].predicted_cost),
            first_plan_error=float(np.mean(np.linalg.norm(s.position[0][first] - planned, axis=-1))),
            airspeed_min=float(s.airspeed.min()), bank_max_deg=float(np.degrees(np.abs(s.bank).max())),
            sixdof=flight.sixdof, wall=time.perf_counter() - tic,
        ))
    return rows


def summarize(rows) -> str:
    ok = [r for r in rows if "error" not in r]
    lines = [f"{'planner':<8}{'mode':<6}{'airframe':<11}{'n':>4}{'cost':>8}{'dz m':>8}{'no viol.':>9}{'stalled':>8}"
             f"{'err m':>7}{'Vmin':>6}{'|beta|':>7}{'sat %':>7}"]
    keys = sorted({(r["planner"], r["mode"]) for r in ok}, key=lambda k: (PLANNERS.index(k[0]), MODES.index(k[1])))
    for planner, mode in keys:
        for af in AIRFRAMES:
            sel = [r for r in ok if (r["planner"], r["mode"], r["airframe"]) == (planner, mode, af)]
            if not sel:
                continue
            sx = [r["sixdof"] for r in sel if r.get("sixdof")]
            beta = np.mean([x["beta_max_deg"] for x in sx]) if sx else float("nan")
            sat = 100 * np.mean([x["saturated_fraction"] for x in sx]) if sx else float("nan")
            lines.append(
                f"{planner:<8}{mode:<6}{af:<11}{len(sel):>4}{np.mean([r['cost'] for r in sel]):>8.2f}"
                f"{np.mean([r['altitude_gain'] for r in sel]):>8.1f}{100 * np.mean([r['feasible'] for r in sel]):>8.0f}%"
                f"{100 * np.mean([r['stalled'] for r in sel]):>7.0f}%{np.median([r['first_plan_error'] for r in sel]):>7.1f}"
                f"{np.mean([r['airspeed_min'] for r in sel]):>6.1f}{beta:>7.1f}{sat:>7.1f}")
        # paired: the same situations, 6-DOF minus 3-DOF
        pairs = {}
        for r in ok:
            if (r["planner"], r["mode"]) == (planner, mode):
                pairs.setdefault(r["instance"], {})[r["airframe"]] = r
        d = [p["6dof"]["altitude_gain"] - p["3dof"]["altitude_gain"] for p in pairs.values() if {"3dof", "6dof"} <= p.keys()]
        if d:
            lost = sum(1 for p in pairs.values() if "6dof" in p and p["6dof"]["altitude_gain"] < -50.0)
            lines.append(f"{'':<14}6dof - 3dof altitude gain: median {np.median(d):+.1f} m, mean {np.mean(d):+.1f} m "
                         f"over {len(d)} situations; 6-DOF flights that lost over 50 m: {lost}")
    errors = [r for r in rows if "error" in r]
    if errors:
        lines.append(f"{len(errors)} flights raised: {errors[0]['error']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--theta", type=Path, help="the tuned law (law and teacher planners)")
    parser.add_argument("--vae", type=Path, help="command VAE (ga and fast planners)")
    parser.add_argument("--flow", type=Path, help="distilled generator (fast planner)")
    parser.add_argument("--planners", nargs="+", default=["law"], choices=PLANNERS)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--airframes", nargs="+", default=list(AIRFRAMES), choices=AIRFRAMES)
    parser.add_argument("--first", type=int, default=600)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--replan-every", type=float, default=10.0)
    parser.add_argument("--w20", type=float, default=0.0, help="Dryden turbulence, wind 20 ft above ground (m/s)")
    parser.add_argument("--inertia-scale", type=float, default=1.0, help="scales the estimated inertia (sensitivity)")
    parser.add_argument("--nlp-budget", type=int, default=6_000)
    parser.add_argument("--search-budget", type=int, default=3_000)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--out", type=Path, default=Path("results/sixdof_compare"))
    args = parser.parse_args()

    cfg = dict(theta=str(args.theta) if args.theta else None, vae=str(args.vae) if args.vae else None,
               flow=str(args.flow) if args.flow else None, duration=args.duration, replan_every=args.replan_every,
               w20=args.w20, inertia_scale=args.inertia_scale, nlp_budget=args.nlp_budget,
               search_budget=args.search_budget, airframes=args.airframes)
    jobs = [(i, p, m) for p in args.planners for m in args.modes for i in range(args.first, args.first + args.count)]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs x {len(args.airframes)} airframes on {args.workers} workers")
    tic = time.perf_counter()
    rows = []
    if args.workers <= 1:
        _init_worker(cfg)
        for j in jobs:
            rows.extend(run(j))
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for k, out in enumerate(pool.map(run, jobs, chunksize=1)):
                rows.extend(out)
                if (k + 1) % 10 == 0:
                    print(f"  {k + 1}/{len(jobs)} jobs, {time.perf_counter() - tic:.0f} s", flush=True)
    (args.out / "flights.json").write_text(json.dumps(rows, indent=1, default=float))
    text = summarize(rows)
    header = (f"sixdof_compare: {args.count} weather-family situations from {args.first}, planners {args.planners}, "
              f"w20 {args.w20} m/s, inertia x{args.inertia_scale}, {time.perf_counter() - tic:.0f} s\n")
    (args.out / "summary.txt").write_text(header + text + "\n")
    print(header + text)


if __name__ == "__main__":
    main()
