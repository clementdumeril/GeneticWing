"""Phase 5: export a cross-country task to the 3D viewer (``docs/viewer/xc3d.html``).

Flies one 20 km task twice -- the MacCready tactician with the classic
circling rule, and the same tactician with every climb flown by the latent GA
(motor off in the climbs) -- and writes a self-contained page: both paths, the
drifting thermals, the climbs, a barogram and the battery over time.

Run: ``python scenarios/phase5_view.py --vae D/physics/command_vae_d16.pt --out results/phase5_view/xc.html``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from geneticwing.aircraft import SBXC_APPROX
from geneticwing.closedloop import LatentPlanner
from geneticwing.crosscountry import EFFICIENCY, MacCreadyPilot, PlannedClimbPilot, fly_xc, xc_family
from geneticwing.latent import ConditionalVAE
from geneticwing.torchsim import SimConfig

TEMPLATE = Path(__file__).resolve().parents[1] / "docs" / "viewer" / "xc3d.html"


def _r(a, digits=1):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def export(task, pilot, label, key, every_s=1.0) -> dict:
    f = fly_xc(task, pilot)
    s = f.state
    t = s.t[0]
    keep = np.unique(np.searchsorted(t, np.arange(0.0, t[-1] + 1e-9, every_s)).clip(0, t.size - 1))
    power = np.clip(s.thrust_required[0], 0.0, None) * s.airspeed[0] / EFFICIENCY
    battery = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(t))])
    goal = np.asarray(task.goal)
    return {
        "key": key, "label": label,
        "t": _r(t[keep], 1), "p": _r(s.position[0][keep], 1), "vz": _r(s.velocity[0][keep, 2], 2),
        "airspeed": _r(s.airspeed[0][keep], 1), "bank": _r(np.degrees(s.bank[0][keep]), 0),
        "mode": f.modes[keep].tolist(), "target": f.targets[keep].tolist(),
        "battery": _r(battery[keep] / 1e3, 2), "to_goal": _r(np.linalg.norm(s.position[0][keep, :2] - goal, axis=-1), 0),
        "climbs": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in c.items()} for c in f.climbs],
        "arrived": f.arrived, "duration": round(f.duration, 1), "speed_kmh": round(f.speed_kmh, 2),
        "cost": round(f.cost, 3), "battery_kj": round(f.battery / 1e3, 2), "violated": f.violated, "stalled": f.stalled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--macready", type=float, default=1.0)
    parser.add_argument("--search-budget", type=int, default=600)
    parser.add_argument("--out", type=Path, default=Path("results/phase5_view/xc.html"))
    args = parser.parse_args()

    state = torch.load(args.vae, weights_only=False, map_location="cpu")
    c = state["config"]
    vae = ConditionalVAE(c["dim"], 6, c["latent_dim"], 512, 3, c["beta"], device="cpu")
    vae.load(state)
    glider = SimConfig(SBXC_APPROX, dt=0.4, knots=61, cl_limit=0.75, bank_max_deg=38.0, load_max=2.1, thrust_max=1e-6)

    task = xc_family(args.instance)
    flights = []
    for key, label, pilot in (
        ("rule", f"Spirale classique (MacCready {args.macready:g})", MacCreadyPilot(macready=args.macready)),
        ("ga", f"Montées par notre GA (MacCready {args.macready:g})",
         PlannedClimbPilot(macready=args.macready, planner=LatentPlanner(vae, glider, args.search_budget),
                           motor_in_climbs=False)),
    ):
        data = export(task, pilot, label, key)
        flights.append(data)
        print(f"{label}: {data['duration']:.0f} s, {data['speed_kmh']} km/h, battery {data['battery_kj']} kJ, "
              f"{len(data['climbs'])} climbs, violated {data['violated']}", flush=True)

    z_ref = 400.0
    payload = {
        "task": {"name": task.name, "start": _r(task.start), "goal": _r(task.goal), "goal_radius": task.goal_radius,
                 "wind": _r(task.wind, 2), "z_floor": task.z_floor, "z_ceiling": task.z_ceiling,
                 "distance": task.distance, "time_limit": task.time_limit},
        "thermals": [{"c": _r(th.center), "d": _r(th.drift, 2), "w": round(th.w_star, 2),
                      "r": round(float(th.outer_radius(z_ref)) / 2.0, 1)} for th in task.thermals],
        "flights": flights,
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*XC_JSON*/null", json.dumps(payload, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
