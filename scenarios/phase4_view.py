"""Phase 4: export closed-loop flights to the 3D flight viewer (``docs/viewer/flight3d.html``).

Flies one S1 situation (strip wing) in several conditions, open loop and
replanned, and writes a self-contained HTML page: the flown path coloured by
climb rate, every plan the aircraft was given, the drifting thermal core, the
believed centre, and the aircraft's altitude, climb rate, airspeed, bank and
motor over time.

Run: ``python scenarios/phase4_view.py --vae D/physics/command_vae_d16.pt --out results/phase4_view/flights.html``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from geneticwing.aircraft import SBXC_APPROX
from geneticwing.benchmarks import MOTOR_THRUST, s1_family
from geneticwing.closedloop import (
    LatentPlanner,
    LoopConfig,
    PriorPlanner,
    _air_parts,
    exact_belief,
    fly_closed_loop,
    sample_belief,
    turbulent_air,
)
from geneticwing.latent import ConditionalVAE
from geneticwing.strip import sbxc_wing
from geneticwing.torchsim import SimConfig

TEMPLATE = Path(__file__).resolve().parents[1] / "docs" / "viewer" / "flight3d.html"

# (key, label, turbulence, wrong belief, replan every)
CASES = (
    ("calm_open", "Air calme, plan fait une fois", False, False, None),
    ("gusts_open", "Rafales, plan fait une fois", True, False, None),
    ("gusts_mpc", "Rafales, replan toutes les 10 s", True, False, 10.0),
    ("wrong_open", "Thermique mal estimé, plan fait une fois", False, True, None),
    ("wrong_mpc", "Thermique mal estimé, replan toutes les 10 s", False, True, 10.0),
)


def _r(a, digits=1):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def export_flight(situation, air, belief, planner, loop, rng, every_s=0.2) -> dict:
    flight = fly_closed_loop(situation, air, belief, planner, loop, rng)
    s = flight.state
    t = s.t[0]
    keep = np.unique(np.searchsorted(t, np.arange(0.0, t[-1] + 1e-9, every_s)).clip(0, t.size - 1))
    thrust = np.clip(s.thrust_required[0], 0.0, None)
    plans, t0 = [], 0.0
    for plan in flight.plans:
        ref = plan.reference
        sel = np.unique(np.searchsorted(ref.t, np.arange(0.0, ref.t[-1] + 1e-9, 1.0)).clip(0, ref.t.size - 1))
        plans.append({"t0": round(t0, 2), "p": _r(ref.position[sel]), "cost": round(plan.predicted_cost, 2)})
        t0 += loop.duration if loop.replan_every is None else loop.replan_every
    ev = flight.evaluation
    return {
        "t": _r(t[keep], 2),
        "p": _r(s.position[0][keep]),
        "v": _r(s.velocity[0][keep], 2),
        "bank": _r(np.degrees(s.bank[0][keep])),
        "airspeed": _r(s.airspeed[0][keep], 2),
        "thrust": _r(thrust[keep], 2),
        "updraft": _r(s.wind[0][keep, 2], 2),
        "cl": _r(s.cl[0][keep], 3),
        "plans": plans,
        "cost": round(float(ev.cost[0]), 2),
        "stored": round(float(ev.stored_energy[0])),
        "battery": round(float(ev.propulsive_energy[0])),
        "violated": flight.violated,
        "stalled": flight.stalled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--instance", type=int, default=600)
    parser.add_argument("--planner", choices=("ga", "prior"), default="ga")
    parser.add_argument("--search-budget", type=int, default=600)
    parser.add_argument("--w20", type=float, default=7.5)
    parser.add_argument("--out", type=Path, default=Path("results/phase4_view/flights.html"))
    args = parser.parse_args()

    state = torch.load(args.vae, weights_only=False, map_location="cpu")
    c = state["config"]
    vae = ConditionalVAE(c["dim"], 6, c["latent_dim"], 512, 3, c["beta"], device="cpu")
    vae.load(state)
    sim = SimConfig(SBXC_APPROX, dt=0.4, knots=61, cl_limit=0.75, bank_max_deg=38.0, load_max=2.1,
                    thrust_max=0.8 * MOTOR_THRUST)
    planner = LatentPlanner(vae, sim, args.search_budget) if args.planner == "ga" else PriorPlanner(vae, sim, 64)

    situation = replace(s1_family(args.instance)[0], strip_wing=sbxc_wing())
    thermal, wind = _air_parts(situation)
    exact = exact_belief(situation)
    wrong = sample_belief(situation, np.random.default_rng([args.instance, 1]), 15.0, 0.2, 0.5)
    flights = []
    for key, label, gusts, mistaken, every in CASES:
        air = turbulent_air(situation, args.w20 if gusts else 0.0, 60.0, seed=args.instance)
        belief = wrong if mistaken else exact
        data = export_flight(situation, air, belief, planner, LoopConfig(replan_every=every), np.random.default_rng(0))
        data.update(key=key, label=label, gusts=gusts, replan=every,
                    belief={"center": _r(belief.center), "wind": _r(belief.wind, 2), "w_star": round(belief.w_star, 2)})
        flights.append(data)
        print(f"{label}: score {data['cost']}, battery {data['battery']} J, violated {data['violated']}", flush=True)

    z = np.linspace(100.0, 450.0, 15)
    payload = {
        "instance": args.instance,
        "planner": args.planner,
        "thermal": {"center": _r(thermal.center), "drift": _r(thermal.drift, 2), "w_star": round(thermal.w_star, 2),
                    "z": _r(z, 0), "core_radius": _r(thermal.outer_radius(z) / 2.0), "peak": _r(thermal.peak_updraft(z), 2)},
        "wind": _r(wind, 2),
        "motor_max": float(MOTOR_THRUST),
        "flights": flights,
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*FLIGHTS_JSON*/null", json.dumps(payload, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
