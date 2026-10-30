"""Phase 5: the cross-country benchmark -- 20 km tasks, classic circling against planned climbs.

Every task of :func:`geneticwing.crosscountry.xc_family` is flown by:

* motor       - straight to the goal on the motor (the reference, score about 1)
* mc0.5, mc1  - the MacCready tactician with the classic circling rule
* ga0.5, ga1  - the same tactician, every climb flown by the latent GA over the
                command VAE (``--search-budget`` flights per plan), motor off
                in the climbs: the circling alone is compared
* prior0.5    - the same, with the best of 64 prior samples instead of the GA

The tactician and the thermal field are identical for one task, so the
pilots differ only by how they climb (and by where that takes them next).
Run: ``python scenarios/phase5_crosscountry.py --vae D/physics/command_vae_d16.pt --tasks 64``
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from geneticwing.aircraft import SBXC_APPROX  # noqa: E402
from geneticwing.closedloop import LatentPlanner, PriorPlanner  # noqa: E402
from geneticwing.crosscountry import MacCreadyPilot, MotorPilot, PlannedClimbPilot, fly_xc, xc_family  # noqa: E402
from geneticwing.latent import ConditionalVAE  # noqa: E402
from geneticwing.torchsim import SimConfig  # noqa: E402

PILOTS = ("motor", "mc0.5", "mc1", "ga0.5", "ga1", "prior0.5")
_STATE: dict = {}


def _init_worker(vae_state, cfg):
    torch.set_num_threads(1)
    c = vae_state["config"]
    vae = ConditionalVAE(c["dim"], 6, c["latent_dim"], cfg["width"], cfg["depth"], c["beta"], device="cpu")
    vae.load(vae_state)
    # The teacher's margins, and no motor: the planned climbs are pure soaring.
    glider = SimConfig(SBXC_APPROX, dt=0.4, knots=61, cl_limit=0.75, bank_max_deg=38.0, load_max=2.1, thrust_max=1e-6)
    _STATE.update(cfg=cfg, ga=LatentPlanner(vae, glider, cfg["search_budget"]), prior=PriorPlanner(vae, glider, 64))


def make_pilot(name: str):
    if name == "motor":
        return MotorPilot()
    kind = name.rstrip("0123456789.")
    setting = float(name[len(kind):])
    if kind == "mc":
        return MacCreadyPilot(macready=setting)
    return PlannedClimbPilot(macready=setting, planner=_STATE[kind], motor_in_climbs=False)


def run(job):
    instance, name = job
    task = xc_family(instance, distance=_STATE["cfg"]["distance"])
    tic = time.perf_counter()
    f = fly_xc(task, make_pilot(name))
    rates = [(c["z1"] - c["z0"]) / max(c["t1"] - c["t0"], 1.0) for c in f.climbs]
    return {
        "instance": instance, "pilot": name, "thermals": len(task.thermals), "arrived": f.arrived,
        "duration": f.duration, "speed_kmh": f.speed_kmh, "cost": f.cost, "battery": f.battery, "stored": f.stored,
        "climbs": f.climbs, "climb_rate": float(np.mean(rates)) if rates else None,
        "climb_time": float(sum(c["t1"] - c["t0"] for c in f.climbs)),
        "violated": f.violated, "stalled": f.stalled, "wall": time.perf_counter() - tic,
    }


def summarize(results, pilots) -> str:
    lines = [f"{'pilot':<9}{'arrived':>8}{'cost':>7}{'battery kJ':>11}{'km/h':>7}{'climbs':>7}{'climb m/s':>10}"
             f"{'no violation':>13}{'stalled':>8}"]
    for name in pilots:
        rows = [r for r in results if r["pilot"] == name]
        if not rows:
            continue
        rates = [r["climb_rate"] for r in rows if r["climb_rate"] is not None]
        lines.append(
            f"{name:<9}{100 * np.mean([r['arrived'] for r in rows]):>7.0f}%{np.mean([r['cost'] for r in rows]):>7.2f}"
            f"{np.mean([r['battery'] for r in rows]) / 1e3:>11.1f}{np.mean([r['speed_kmh'] for r in rows]):>7.1f}"
            f"{np.mean([len(r['climbs']) for r in rows]):>7.1f}{(np.mean(rates) if rates else float('nan')):>10.2f}"
            f"{100 * np.mean([not r['violated'] for r in rows]):>12.0f}%{100 * np.mean([r['stalled'] for r in rows]):>7.0f}%"
        )
    # Paired comparison on the same tasks: the planned climbs against the circling rule at the same setting.
    for planned, rule in (("ga0.5", "mc0.5"), ("ga1", "mc1"), ("prior0.5", "mc0.5")):
        a = {r["instance"]: r for r in results if r["pilot"] == planned}
        b = {r["instance"]: r for r in results if r["pilot"] == rule}
        common = sorted(set(a) & set(b))
        if not common:
            continue
        d_cost = np.array([a[i]["cost"] - b[i]["cost"] for i in common])
        d_speed = np.array([a[i]["speed_kmh"] - b[i]["speed_kmh"] for i in common])
        lines.append(f"{planned} - {rule} on {len(common)} tasks: cost {d_cost.mean():+.3f} (better on "
                     f"{100 * np.mean(d_cost < 0):.0f} %), speed {d_speed.mean():+.2f} km/h (faster on "
                     f"{100 * np.mean(d_speed > 0):.0f} %)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vae", type=Path, required=True, help="command VAE trained with --margins")
    parser.add_argument("--out", type=Path, default=Path("results/phase5"))
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--distance", type=float, default=20_000.0)
    parser.add_argument("--pilots", nargs="+", default=list(PILOTS), choices=PILOTS)
    parser.add_argument("--search-budget", type=int, default=1_000)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    cfg = {"distance": args.distance, "search_budget": args.search_budget, "width": args.width, "depth": args.depth}
    state = torch.load(args.vae, weights_only=False, map_location="cpu")
    slow_first = {"ga1": 0, "ga0.5": 1, "prior0.5": 2, "mc0.5": 3, "mc1": 4, "motor": 5}
    jobs = sorted(((i, p) for i in range(args.first, args.first + args.tasks) for p in args.pilots),
                  key=lambda j: slow_first[j[1]])
    print(f"{args.tasks} tasks of {args.distance / 1000:g} km x {len(args.pilots)} pilots = {len(jobs)} flights, "
          f"{args.workers} workers", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(state, cfg)) as pool:
        for k, r in enumerate(pool.map(run, jobs, chunksize=1)):
            results.append(r)
            if k % 25 == 0:
                print(f"[{time.perf_counter() - start:6.0f}s] {k + 1}/{len(jobs)}", flush=True)
    (args.out / "crosscountry.json").write_text(json.dumps(results, indent=1))
    text = summarize(results, args.pilots)
    (args.out / "crosscountry_summary.txt").write_text(text)
    print("\n" + text)
    print(f"done in {time.perf_counter() - start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
