"""3D animation of one real-weather task (Issoudun, ICON-D2 forecast) for the README and the site.

Reads the page written by ``weather_view.py`` (its ``DATA`` block) and renders it off screen with PyVista: the
forecast w* on the ground, the thermals near the route born, growing, fading and drifting with the forecast wind,
the wind-aware circling rule (MacCready 1, blue, its trail coloured by climb rate) and the motor-only baseline
(grey), and a camera that follows the blue sailplane. Time runs slower in its climbs than in its glides. The
aircraft are drawn larger than life.
Run: ``python scenarios/weather_render3d.py docs/viewer/weather_issoudun.html docs/figures/weather_issoudun_3d.gif
--fps 10 --climb-step 3 --glide-step 20 --size 560 315 --frame-ms 100``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xc_render3d import attitude, glider, load, pose, reencode, smooth  # noqa: E402

GROUND = LinearSegmentedColormap.from_list("ground", ["#d3dcbd", "#dcdbb4", "#e2d29c", "#e0bd7c", "#d6a35c"])
BLUE, GREY, THERMAL, CORE = "#1f5fa8", "#6b747c", "#ffb066", "#ff7a1a"


def envelope(s):
    return float(np.clip(min(s, 1.0 - s) / 0.2, 0.0, 1.0)) if 0.0 <= s <= 1.0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("page", type=Path, help="HTML page written by weather_view.py")
    parser.add_argument("out", type=Path, help=".gif")
    parser.add_argument("--size", type=int, nargs=2, default=(720, 405))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--climb-step", type=float, default=3.0, help="seconds of flight per frame while climbing")
    parser.add_argument("--glide-step", type=float, default=20.0, help="seconds of flight per frame while gliding")
    parser.add_argument("--scale", type=float, default=22.0, help="aircraft drawn this many times larger")
    parser.add_argument("--until", type=float, help="stop at this flight time, s (default: the blue one arrived)")
    parser.add_argument("--frame-ms", type=int, help="re-encode the GIF at this frame duration, ms (shared palette)")
    parser.add_argument("--hold-ms", type=int, default=3000, help="last frame held this long when re-encoding")
    parser.add_argument("--preview", type=int, nargs="*", help="only save these frames as PNG next to OUT")
    args = parser.parse_args()

    data = load(args.page)
    task, thermals, G = data["task"], data["thermals"], data["ground"]
    flights = {f["key"]: f for f in data["flights"]}
    hero, motor = flights["mc1"], flights["motor"]
    for f in (hero, motor):
        for k in ("t", "p", "vz", "mode", "battery"):
            f[k] = np.array(f[k])
        f["heading"], f["gamma"], f["bank_s"], f["v"] = attitude(f["p"], f["t"], np.array(f["bank"]))
    wind = np.array(task["wind"])
    ceiling = float(task["z_ceiling"])

    # Frame clock: slow in the blue sailplane's climbs, fast in its glides.
    t_end = hero["t"][-1] if args.until is None else min(args.until, hero["t"][-1])
    clock, now = [], 0.0
    while now < t_end:
        clock.append(now)
        i = min(np.searchsorted(hero["t"], now), len(hero["t"]) - 1)
        now += args.climb_step if hero["mode"][i] == 1 else args.glide_step
    clock.append(t_end)

    v_dir = np.column_stack([hero["v"][:, :2], np.zeros(len(hero["t"]))])
    v_dir = smooth(v_dir, hero["t"], 120.0)
    v_dir /= np.maximum(np.linalg.norm(v_dir, axis=1, keepdims=True), 1e-6)
    focus = smooth(hero["p"], hero["t"], 12.0)

    pl = pv.Plotter(off_screen=True, window_size=list(args.size))
    pl.set_background("#eef4fa", top="#8fb8de")
    pl.enable_depth_peeling(number_of_peels=6)
    pl.enable_anti_aliasing("fxaa")
    # Ground: the forecast w* at mid-flight, on the task grid (x toward the goal).
    x = np.linspace(G["x"][0], G["x"][1], G["nx"])
    y = np.linspace(G["y"][0], G["y"][1], G["ny"])
    hour = task["start_hour"] + 0.5 * t_end / 3600.0
    k = min(max(hour - G["hours"][0], 0.0), len(G["hours"]) - 1.0001)
    w = np.array(G["w"][int(k)]) * (1 - k % 1) + np.array(G["w"][min(int(k) + 1, len(G["w"]) - 1)]) * (k % 1)
    qx, qy = np.meshgrid(x, y)
    grid = pv.StructuredGrid(qx, qy, np.zeros_like(qx))
    grid["w*"] = w.ravel(order="F")  # the order PyVista reads meshgrid points in
    lo, hi = np.floor(w.min() * 4) / 4, np.ceil(w.max() * 4) / 4
    pl.add_mesh(pv.Plane(center=(10000, 0, -2), i_size=80000, j_size=60000), color="#d3dcbd", ambient=0.35)
    pl.add_mesh(grid, scalars="w*", cmap=GROUND, clim=(lo, hi), ambient=0.35, show_scalar_bar=False,
                interpolate_before_map=True)
    goal = np.array(task["goal"])
    pl.add_mesh(pv.Cylinder(center=(goal[0], goal[1], 150), direction=(0, 0, 1), radius=6, height=300), color="#14212b")
    pl.add_mesh(pv.Sphere(center=(goal[0], goal[1], 310), radius=25), color="#14212b")
    thermal_actors = []
    for th in thermals:
        r = float(th["r"])
        column = pv.Cylinder(center=(th["c"][0], th["c"][1], ceiling / 2), direction=(0, 0, 1), radius=r,
                             height=ceiling, resolution=28, capping=False)
        core = pv.Cylinder(center=(th["c"][0], th["c"][1], ceiling / 2), direction=(0, 0, 1), radius=0.4 * r,
                           height=ceiling, resolution=20, capping=False)
        a1 = pl.add_mesh(column, color=THERMAL, opacity=0.0, ambient=0.5, specular=0.0)
        a2 = pl.add_mesh(core, color=CORE, opacity=0.0, ambient=0.5, specular=0.0)
        strength = float(np.clip((th["w"] - 1.0) / 2.5, 0.0, 1.0))
        thermal_actors.append((th, strength, a1, a2))
    planes = {"mc1": pl.add_mesh(glider(args.scale), color=BLUE, smooth_shading=True, specular=0.3),
              "motor": pl.add_mesh(glider(args.scale), color=GREY, smooth_shading=True, specular=0.3)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.preview is None:
        pl.open_gif(str(args.out), fps=args.fps, palettesize=128)
    clock_of = lambda s: f"{int(task['start_hour'] + s / 3600):02d}:{int(60 * ((task['start_hour'] + s / 3600) % 1)):02d}"  # noqa: E731
    for n, now in enumerate(clock):
        if args.preview is not None and n not in args.preview:
            continue
        for th, strength, a1, a2 in thermal_actors:
            g = envelope((now - th["b"]) / th["l"])
            a1.prop.opacity = g * (0.02 + 0.03 * strength)
            a2.prop.opacity = g * (0.08 + 0.10 * strength)
            a1.visibility = a2.visibility = g > 0.02
            a1.position = a2.position = (wind[0] * now, wind[1] * now, 0.0)
        for key, f in (("mc1", hero), ("motor", motor)):
            i = min(np.searchsorted(f["t"], now), len(f["t"]) - 1)
            planes[key].user_matrix = pose(f["p"][i], f["heading"][i], f["gamma"][i], f["bank_s"][i])
            if i >= 1:
                trail = pv.lines_from_points(f["p"][: i + 1])
                shadow = pv.lines_from_points(f["p"][: i + 1] * [1, 1, 0] + [0, 0, 0.5])
                if key == "mc1":
                    trail["climb (m/s)"] = f["vz"][: i + 1]
                    pl.add_mesh(trail, name="trail_mc1", scalars="climb (m/s)", cmap="RdYlGn", clim=(-2.5, 4.0),
                                line_width=4, show_scalar_bar=False)
                else:
                    pl.add_mesh(trail, name="trail_motor", color="#8c8c8c", line_width=3)
                pl.add_mesh(shadow, name=f"shadow_{key}", color="#7a6a4a", line_width=2, opacity=0.5)
        i = min(np.searchsorted(hero["t"], now), len(hero["t"]) - 1)
        pl.camera.focal_point = tuple(focus[i])
        eye = focus[i] - 560 * v_dir[i] + np.array([-200 * v_dir[i][1], 200 * v_dir[i][0], 170])
        pl.camera.position = tuple(eye)
        pl.camera.up = (0, 0, 1)
        pl.camera.view_angle = 40
        pl.camera.clipping_range = (5, 40000)
        j = min(np.searchsorted(motor["t"], now), len(motor["t"]) - 1)
        state = "arrived" if now >= hero["t"][-1] else ("climbing" if hero["mode"][i] == 1 else "gliding")
        motor_state = "arrived" if now >= motor["t"][-1] else f"{max(motor['p'][j, 0], 0.0) / 1000:4.1f} km"
        pl.add_text(f"{clock_of(now)}    {max(hero['p'][i, 0], 0.0) / 1000:4.1f} km of 20    "
                    f"altitude {hero['p'][i, 2]:3.0f} m    vario {hero['vz'][i]:+.1f} m/s    {state}",
                    name="hud", position="upper_left", font_size=10, color="#1b2733")
        pl.add_text(f"blue: circling rule   {hero['battery'][i]:5.1f} kJ\n"
                    f"grey: motor ({motor_state})   {motor['battery'][j]:5.1f} kJ",
                    name="legend", position="lower_right", font_size=9, color="#1b2733")
        pl.add_text(f"Issoudun, {task['day']}\nICON-D2 forecast thermals", name="caption",
                    position="lower_left", font_size=8, color="#3d4b57")
        if args.preview is not None:
            pl.screenshot(str(args.out.with_name(f"{args.out.stem}_frame{n}.png")))
            continue
        pl.write_frame()
        if n % 50 == 0:
            print(f"frame {n}/{len(clock)}", flush=True)
    pl.close()
    if args.preview is not None:
        print(f"{len(clock)} frames in all; previews saved next to {args.out}")
        return
    count = len(clock)
    if args.frame_ms:
        count = reencode(args.out, args.frame_ms, args.hold_ms, repeats=0,
                         swatches=(BLUE, "#174a84", "#5d8fcc", GREY, "#8c8c8c", "#14212b", CORE, THERMAL))
    print(f"{count} frames -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
