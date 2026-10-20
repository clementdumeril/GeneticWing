"""Scenario S0: reproduce the ARPL straight glide (Morando et al., arXiv 2605.15619).

Setup taken from the paper's real-world glide: a 400 m leg started at 120 m,
airspeed reference 12.5 m/s, about 5.2 m/s of wind, treated here as a pure
tailwind along the leg (the paper gives the wind direction, not the leg's).

Three columns are compared: the paper's flight data, the plan in uniform wind,
and closed-loop point-mass flights of that plan through Dryden turbulence
(tracker + attitude lag, no engine). The simulated flights are the fair
comparison with the paper, but remain a point-mass model. Airframe mass and wing area are placeholders
(see ``geneticwing.aircraft``).

Run: ``python scenarios/s0_glide.py [--out results]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geneticwing import STRATOSURFER_APPROX
from geneticwing.energy import netto
from geneticwing.planning import GlideProblem, plan_glide
from geneticwing.simulation import TrackerConfig, simulate_tracking
from geneticwing.wind import DrydenTurbulence, UniformWind

# Values reported by arXiv 2605.15619 for the real-world glide mission.
ARPL_REFERENCE = {
    "airspeed_mean": 12.5,
    "sink_rate_mean": 1.48,
    "airspeed_rmse": 1.09,
    "sink_rate_rmse": 0.39,
    "netto_rmse": 0.69,
    "glide_ratio": 9.83,
}

WIND_SPEED = 5.2
AIRSPEED_REF = 12.5
START = (0.0, 0.0, 120.0)
GOAL_XY = (400.0, 0.0)
# The ARPL netto uses a Savitzky-Golay derivative; its window is not given, 1 s is assumed.
NETTO_WINDOW_S = 1.0


def make_problem() -> GlideProblem:
    return GlideProblem(
        aircraft=STRATOSURFER_APPROX,
        wind=UniformWind((WIND_SPEED, 0.0, 0.0)),
        start=START,
        goal_xy=GOAL_XY,
        airspeed_ref=AIRSPEED_REF,
    )


def glide_metrics(state, aircraft) -> dict:
    distance = float(np.sum(np.linalg.norm(np.diff(state.position[:, :2], axis=0), axis=-1)))
    altitude_lost = float(state.position[0, 2] - state.position[-1, 2])
    return {
        "airspeed_mean": float(state.airspeed.mean()),
        "sink_rate_mean": float(state.sink_rate_air.mean()),
        "airspeed_rmse": float(np.sqrt(np.mean((state.airspeed - AIRSPEED_REF) ** 2))),
        "sink_rate_rmse": float(
            np.sqrt(np.mean((state.sink_rate_air - aircraft.nominal_sink_rate(AIRSPEED_REF)) ** 2))
        ),
        "netto_rmse": float(np.sqrt(np.mean(netto(aircraft, state, NETTO_WINDOW_S) ** 2))),
        "glide_ratio": distance / altitude_lost,
        "ground_distance": distance,
        "altitude_lost": altitude_lost,
        "duration": float(state.t[-1]),
    }


def convergence_study(problem: GlideProblem, reference_plan, n_trials: int, seed: int) -> dict:
    """Re-solve from perturbed initial guesses: does the NLP land on the same glide?"""
    rng = np.random.default_rng(seed)
    final_altitudes, iterations, successes = [], [], 0
    for _ in range(n_trials):
        cp = reference_plan.control_points.copy()
        cp[3:-1] += rng.normal(0.0, 10.0, cp[3:-1].shape)
        duration = reference_plan.duration * rng.uniform(0.85, 1.15)
        plan = plan_glide(problem, initial=(cp, duration))
        successes += plan.success
        iterations.append(plan.iterations)
        final_altitudes.append(plan.control_points[-1, 2])
    return {
        "trials": n_trials,
        "success_rate": successes / n_trials,
        "iterations_median": float(np.median(iterations)),
        "final_altitude_spread_m": float(np.ptp(final_altitudes)),
    }


def gusts(plan, seed: int) -> DrydenTurbulence:
    # Longitudinal gust axis along the leg (east); W20 taken equal to the mean wind.
    return DrydenTurbulence(
        duration=2.0 * plan.duration + 1.0,
        airspeed=AIRSPEED_REF,
        altitude=100.0,
        w20=WIND_SPEED,
        heading_deg=90.0,
        seed=seed,
    )


def fly(plan, wind):
    """Closed-loop, unpowered flight of the plan in ``wind``."""
    reference = plan.sample(2000)
    return simulate_tracking(
        plan.problem.aircraft, reference, wind, reference.position[0], reference.velocity[0], TrackerConfig()
    )


def turbulence_study(plan, aircraft, n_seeds: int) -> dict:
    """Mean metrics of closed-loop flights through Dryden turbulence realizations."""
    runs = [glide_metrics(fly(plan, plan.problem.wind + gusts(plan, s)), aircraft) for s in range(n_seeds)]
    return {key: float(np.mean([r[key] for r in runs])) for key in runs[0]}


def plot(plan, turbulent_state, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nominal = plan.sample()
    ac = plan.problem.aircraft
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(nominal.position[:, 0], nominal.position[:, 2], label="plan, uniform wind")
    ax.plot(turbulent_state.position[:, 0], turbulent_state.position[:, 2], alpha=0.7, label="closed loop, Dryden")
    ax.set(xlabel="ground distance (m)", ylabel="altitude (m)", title="Glide profile")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(nominal.t, nominal.airspeed, label="uniform wind")
    ax.plot(turbulent_state.t, turbulent_state.airspeed, alpha=0.7, label="closed loop, Dryden")
    ax.axhline(AIRSPEED_REF, color="k", lw=0.8, ls="--")
    ax.set(xlabel="t (s)", ylabel="airspeed (m/s)", title="Airspeed")
    ax.legend()
    ax = axes[1, 0]
    ax.plot(nominal.t, netto(ac, nominal, NETTO_WINDOW_S), label="uniform wind")
    ax.plot(turbulent_state.t, netto(ac, turbulent_state, NETTO_WINDOW_S), alpha=0.7, label="closed loop, Dryden")
    ax.set(xlabel="t (s)", ylabel="netto (m/s)", title="Netto variometer")
    ax.legend()
    ax = axes[1, 1]
    ax.plot(turbulent_state.t, np.degrees(turbulent_state.bank), color="C1")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set(xlabel="t (s)", ylabel="bank (deg)", title="Bank, closed loop in Dryden")
    fig.suptitle("S0: straight glide, 400 m from 120 m, 5.2 m/s tailwind")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=50)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    problem = make_problem()
    ac = problem.aircraft
    plan = plan_glide(problem)
    if not plan.success:
        raise SystemExit(f"S0 planning failed: {plan.message}")

    nominal = glide_metrics(plan.sample(), ac)
    turbulent = turbulence_study(plan, ac, args.seeds)
    convergence = convergence_study(problem, plan, args.trials, seed=0)

    rows = ["airspeed_mean", "sink_rate_mean", "glide_ratio", "airspeed_rmse", "sink_rate_rmse", "netto_rmse"]
    print(f"{'metric':<18}{'ARPL flight':>12}{'plan':>10}{'sim+Dryden':>12}")
    for key in rows:
        print(f"{key:<18}{ARPL_REFERENCE[key]:>12.2f}{nominal[key]:>10.2f}{turbulent[key]:>12.2f}")
    print(f"\nconvergence from perturbed guesses: {convergence}")

    plot(plan, fly(plan, problem.wind + gusts(plan, 0)), args.out / "s0_glide.png")
    report = {
        "aircraft": {"name": ac.name, "mass": ac.mass, "wing_area": ac.wing_area, "cd0": ac.cd0, "k": ac.k},
        "arpl_reference": ARPL_REFERENCE,
        "plan": nominal,
        "closed_loop_dryden_mean": turbulent,
        "convergence": convergence,
    }
    (args.out / "s0_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out / 's0_glide.png'} and {args.out / 's0_metrics.json'}")


if __name__ == "__main__":
    main()
