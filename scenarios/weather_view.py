"""Export one real-weather task to the 3D viewer (``docs/viewer/weather3d.html``).

Flies one 20 km task of :func:`geneticwing.weather.weather_task` (thermals placed from the archived ICON-D2 forecast
over Issoudun) with the three rule pilots -- the motor all the way, and the wind-aware MacCready tactician at 0.5 and
1 m/s -- and writes a self-contained page: the three paths, the thermals that come near them (born, growing, fading
and drifting with the wind), the forecast w* on the ground hour by hour, a barogram and the battery over time.

Run: ``python scenarios/weather_view.py --day 2025-07-12 --task 1 --out docs/viewer/weather_issoudun.html``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geneticwing.crosscountry import EFFICIENCY, Autopilot, MacCreadyPilot, MotorPilot, fly_xc
from geneticwing.weather import ConvectiveDay, _r2, weather_task

TEMPLATE = Path(__file__).resolve().parents[1] / "docs" / "viewer" / "weather3d.html"
DAYS = {"2025-07-12": "strong blue thermals", "2025-08-02": "cumulus day", "2025-06-22": "windy, low layer"}


def _r(a, digits=1):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def export(task, pilot, key, label, every_s=1.0) -> tuple[dict, object]:
    f = fly_xc(task, pilot, autopilot=Autopilot(inner_wing_protection=0.9))
    s = f.state
    t = s.t[0]
    keep = np.unique(np.searchsorted(t, np.arange(0.0, t[-1] + 1e-9, every_s)).clip(0, t.size - 1))
    power = np.clip(s.thrust_required[0], 0.0, None) * s.airspeed[0] / EFFICIENCY
    battery = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(t))])
    goal = np.asarray(task.goal)
    data = {
        "key": key, "label": label,
        "t": _r(t[keep], 1), "p": _r(s.position[0][keep], 1), "vz": _r(s.velocity[0][keep, 2], 2),
        "airspeed": _r(s.airspeed[0][keep], 1), "bank": _r(np.degrees(s.bank[0][keep]), 0),
        "mode": np.asarray(f.modes)[keep].tolist(), "target": np.asarray(f.targets)[keep].tolist(),
        "battery": _r(battery[keep] / 1e3, 2),
        "to_goal": _r(np.linalg.norm(s.position[0][keep, :2] - goal, axis=-1), 0),
        "climbs": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in c.items()} for c in f.climbs],
        "arrived": f.arrived, "duration": round(f.duration, 1), "speed_kmh": round(f.speed_kmh, 2),
        "battery_kj": round(f.battery / 1e3, 2),
    }
    return data, f


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", default="2025-07-12", choices=list(DAYS))
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--near", type=float, default=1500.0, help="keep the thermals that come this close, m")
    parser.add_argument("--out", type=Path, default=Path("docs/viewer/weather_issoudun.html"))
    args = parser.parse_args()

    day = ConvectiveDay.load(Path(f"data/weather/issoudun_{args.day}.npz"))
    task = weather_task(day, args.task)
    flights, paths = [], []
    for key, label, pilot in (("mc1", "Circling rule, MacCready 1", MacCreadyPilot(macready=1.0, wind_aware=True)),
                              ("mc05", "Circling rule, MacCready 0.5", MacCreadyPilot(macready=0.5, wind_aware=True)),
                              ("motor", "Motor all the way", MotorPilot(wind_aware=True))):
        data, f = export(task, pilot, key, label)
        flights.append(data)
        paths.append((f.state.t[0], f.state.position[0]))
        print(f"{label}: {data['duration']:.0f} s, {data['speed_kmh']:.1f} km/h, {data['battery_kj']:.1f} kJ, "
              f"{len(data['climbs'])} climbs, arrived {data['arrived']}", flush=True)

    # The thermals that come within ``near`` of a path while they are alive (the rest of the day's are not drawn).
    t_end = max(p[0][-1] for p in paths)
    c = np.array([th.center for th in task.thermals])
    d = np.array([th.drift for th in task.thermals])
    b, life = np.asarray(task.births), np.asarray(task.lives)
    keep = np.zeros(len(c), bool)
    for t, p in paths:
        step = max(1, int(round(10.0 / np.median(np.diff(t)))))
        for tk, pk in zip(t[::step], p[::step]):
            alive = (b < tk) & (b + life > tk)
            keep |= alive & (np.hypot(c[:, 0] + d[:, 0] * tk - pk[0], c[:, 1] + d[:, 1] * tk - pk[1]) < args.near)
    used = {cl["thermal"] for f in flights for cl in f["climbs"]}
    idx = np.nonzero(keep)[0]
    thermals = [{"i": int(i), "c": _r(c[i]), "b": round(float(b[i]), 1), "l": round(float(life[i]), 1),
                 "w": round(float(task.thermals[i].w_star), 2), "zi": round(float(task.thermals[i].z_i), 0),
                 "r": round(float(_r2(0.4 * task.z_ceiling, task.thermals[i].z_i)), 1), "used": int(i) in used}
                for i in idx]

    # Forecast w* on the ground, in the task frame, at every hour the flight spans (lerped in the page).
    x = np.linspace(-3_000.0, task.distance + 3_000.0, 105)
    y = np.linspace(-6_000.0, 6_000.0, 49)
    qx, qy = np.meshgrid(x, y)
    m = task.to_map(np.stack([qx, qy], -1))
    h0 = int(np.floor(task.start_hour))
    hours = list(range(h0, int(np.ceil(task.start_hour + t_end / 3600.0)) + 1))
    ground = [_r(day.at(day.w_star, h, m[..., 0], m[..., 1]), 2) for h in hours]
    zi_route = float(day.at(day.z_i, task.start_hour + t_end / 7200.0, *task.to_map(np.array([task.distance / 2, 0.0]))))
    north = float(np.degrees(np.pi / 2 - task.bearing))  # the goal's compass bearing
    cb, sb = np.cos(task.bearing), np.sin(task.bearing)
    wind_map = (cb * task.wind[0] - sb * task.wind[1], sb * task.wind[0] + cb * task.wind[1])
    wind_from = float(np.degrees(np.arctan2(-wind_map[0], -wind_map[1])) % 360.0)

    # The batch over 48 tasks per day (results/weather_v2/<day>.json, merged from phase5_crosscountry.py runs).
    summary = []
    for date, name in DAYS.items():
        path = Path(f"results/weather_v2/{date}.json")
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        for key in ("mc1", "mc0.5", "motor"):
            rs = [r for r in rows if r["pilot"] == key]
            rates = [r["climb_rate"] for r in rs if r["climb_rate"] is not None]
            summary.append({"day": date, "day_label": name, "pilot": key, "tasks": len(rs),
                            "arrived": round(100 * float(np.mean([r["arrived"] for r in rs])), 0),
                            "kj": round(float(np.mean([r["battery"] for r in rs])) / 1e3, 1),
                            "kmh": round(float(np.mean([r["speed_kmh"] for r in rs])), 1),
                            "climb": round(float(np.mean(rates)), 2) if rates else None})

    payload = {
        "task": {"name": task.name, "day": args.day, "day_label": DAYS[args.day], "instance": args.task,
                 "start": _r(task.start), "goal": _r(task.goal), "goal_radius": task.goal_radius,
                 "wind": _r(task.wind, 2), "z_floor": task.z_floor, "z_ceiling": task.z_ceiling,
                 "distance": task.distance, "time_limit": round(task.time_limit, 0),
                 "start_hour": round(task.start_hour, 4), "bearing_deg": round(north % 360.0, 1),
                 "zi_route": round(zi_route, 0), "sink": round(float(task.sink(np.array([0.0, 0.0, 300.0]))[2]), 2),
                 "thermals_total": len(task.thermals), "wind_from_deg": round(wind_from, 0),
                 "w_route": round(float(day.at(day.w_star, task.start_hour + t_end / 7200.0,
                                               *task.to_map(np.array([task.distance / 2, 0.0])))), 2)},
        "summary": summary,
        "thermals": thermals,
        "ground": {"x": [float(x[0]), float(x[-1])], "y": [float(y[0]), float(y[-1])], "nx": len(x), "ny": len(y),
                   "hours": hours, "w": ground},
        "flights": flights,
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*WX_JSON*/null", json.dumps(payload, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB), {len(thermals)} thermals drawn of "
          f"{len(task.thermals)}")


if __name__ == "__main__":
    main()
