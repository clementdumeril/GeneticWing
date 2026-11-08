"""3D animation of one 20 km task for the README: the sailplane circling up the drifting thermal columns.

Reads the flights saved by ``phase5_view.py`` (the ``DATA`` block of its HTML page) and renders them off screen with
PyVista: thermal columns drifting with the wind, the planned-climb sailplane (red) and the circling-rule one (grey),
each with its trail (the red one coloured by climb rate), and a camera that follows the red sailplane. Time runs
slower in the climbs than in the glides. The aircraft and the thermal cores are drawn larger than life.
Run: ``python scenarios/xc_render3d.py results/phase5_view/xc.html docs/figures/xc_20km_3d.gif``
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyvista as pv

CEILING = 600.0


def load(page: Path) -> dict:
    text = page.read_text(encoding="utf-8")
    return json.loads(re.search(r"const DATA\s*=\s*(\{.*?\});\s*\n", text, re.S).group(1))


def glider(scale: float) -> pv.PolyData:
    """A simple sailplane in body axes (x forward, y left, z up), 4.6 m span before scaling."""
    parts = [
        pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(1, 0, 0), radius=0.09, height=1.7, resolution=12),
        pv.Cone(center=(0.95, 0.0, 0.0), direction=(1, 0, 0), height=0.2, radius=0.09, resolution=12),
        pv.Box(bounds=(-0.06, 0.18, -2.3, 2.3, -0.015, 0.015)),
        pv.Box(bounds=(-0.95, -0.8, -0.45, 0.45, -0.01, 0.01)),
        pv.Box(bounds=(-0.95, -0.78, -0.01, 0.01, 0.0, 0.38)),
    ]
    body = parts[0].triangulate()
    for part in parts[1:]:
        body = body.merge(part.triangulate())
    return body.scale(scale, inplace=False)


def attitude(p: np.ndarray, t: np.ndarray, bank_deg: np.ndarray):
    """Heading, path angle and signed bank (positive = left wing down) along the flight."""
    v = np.gradient(p, t, axis=0)
    heading = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    gamma = np.arctan2(v[:, 2], np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-3))
    turn = np.gradient(heading, t)
    bank = np.radians(np.abs(bank_deg)) * np.sign(turn)
    return heading, gamma, bank, v


def pose(position, heading, gamma, bank) -> np.ndarray:
    ch, sh = np.cos(heading), np.sin(heading)
    rz = np.array([[ch, -sh, 0], [sh, ch, 0], [0, 0, 1]])
    cg, sg = np.cos(-gamma), np.sin(-gamma)
    ry = np.array([[cg, 0, sg], [0, 1, 0], [-sg, 0, cg]])
    cb, sb = np.cos(-bank), np.sin(-bank)
    rx = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
    m = np.eye(4)
    m[:3, :3] = rz @ ry @ rx
    m[:3, 3] = position
    return m


def smooth(x: np.ndarray, t: np.ndarray, tau: float) -> np.ndarray:
    """First-order low-pass of samples ``x`` (N, d) along ``t``."""
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        a = 1.0 - np.exp(-(t[i] - t[i - 1]) / tau)
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("page", type=Path, help="HTML page written by phase5_view.py")
    parser.add_argument("out", type=Path, help=".gif")
    parser.add_argument("--size", type=int, nargs=2, default=(720, 405))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--climb-step", type=float, default=3.0, help="seconds of flight per frame while climbing")
    parser.add_argument("--glide-step", type=float, default=24.0, help="seconds of flight per frame while gliding")
    parser.add_argument("--scale", type=float, default=22.0, help="aircraft drawn this many times larger")
    parser.add_argument("--thermal-radius", type=float, default=45.0, help="drawn radius of a thermal column, m")
    parser.add_argument("--preview", type=int, nargs="*", help="only save these frames as PNG next to OUT")
    args = parser.parse_args()

    data = load(args.page)
    task, thermals = data["task"], data["thermals"]
    flights = {f["key"]: f for f in data["flights"]}
    ga, rule = flights["ga"], flights["rule"]
    for f in (ga, rule):
        f["t"], f["p"] = np.array(f["t"]), np.array(f["p"])
        f["vz"], f["mode"] = np.array(f["vz"]), np.array(f["mode"])
        f["heading"], f["gamma"], f["bank_s"], f["v"] = attitude(f["p"], f["t"], np.array(f["bank"]))

    # Frame clock: slow in the red sailplane's climbs, fast in its glides, on until both have arrived.
    t_end = max(ga["t"][-1], rule["t"][-1])
    clock, now = [], 0.0
    while now < t_end:
        clock.append(now)
        i = min(np.searchsorted(ga["t"], now), len(ga["t"]) - 1)
        climbing = now < ga["t"][-1] and ga["mode"][i] == 1
        now += args.climb_step if climbing else args.glide_step
    clock.append(t_end)
    clock += [t_end] * args.fps * 2

    # Camera: behind the red sailplane along its slow direction of travel, focal point smoothed so the spirals
    # turn around the centre of the picture.
    v_dir = np.column_stack([ga["v"][:, :2], np.zeros(len(ga["t"]))])
    v_dir = smooth(v_dir, ga["t"], 120.0)
    v_dir /= np.maximum(np.linalg.norm(v_dir, axis=1, keepdims=True), 1e-6)
    focus = smooth(ga["p"], ga["t"], 12.0)

    pl = pv.Plotter(off_screen=True, window_size=list(args.size))
    pl.set_background("#eef4fa", top="#8fb8de")
    pl.enable_depth_peeling(number_of_peels=6)
    pl.enable_anti_aliasing("fxaa")
    pl.add_mesh(pv.Plane(center=(10000, 900, 0), i_size=40000, j_size=30000), color="#d3dcbd", ambient=0.35)
    goal = np.array(task["goal"])
    pl.add_mesh(pv.Cylinder(center=(goal[0], goal[1], 150), direction=(0, 0, 1), radius=6, height=300), color="#1F3FD6")
    pl.add_mesh(pv.Sphere(center=(goal[0], goal[1], 310), radius=25), color="#1F3FD6")
    thermal_actors = []
    for th in thermals:
        strength = np.clip((th["w"] - 1.0) / 2.0, 0.0, 1.0)
        column = pv.Cylinder(center=(th["c"][0], th["c"][1], CEILING / 2), direction=(0, 0, 1),
                             radius=args.thermal_radius, height=CEILING, resolution=28, capping=False)
        core = pv.Cylinder(center=(th["c"][0], th["c"][1], CEILING / 2), direction=(0, 0, 1),
                           radius=0.4 * args.thermal_radius, height=CEILING, resolution=20, capping=False)
        cloud = pv.ParametricEllipsoid(2.6 * args.thermal_radius, 2.6 * args.thermal_radius, 45.0)
        cloud.translate((th["c"][0], th["c"][1], CEILING + 45.0), inplace=True)
        a1 = pl.add_mesh(column, color="#ffb066", opacity=0.05 + 0.08 * strength, ambient=0.5, specular=0.0)
        a2 = pl.add_mesh(core, color="#ff7a1a", opacity=0.14 + 0.16 * strength, ambient=0.5, specular=0.0)
        a3 = pl.add_mesh(cloud, color="#ffffff", opacity=0.6, ambient=0.6, smooth_shading=True)
        thermal_actors.append((np.array(th["d"]), a1, a2, a3))
    planes = {}
    for key, color in (("ga", "#d7301f"), ("rule", "#7d7d7d")):
        planes[key] = pl.add_mesh(glider(args.scale), color=color, smooth_shading=True, specular=0.3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.preview is None:
        pl.open_gif(str(args.out), fps=args.fps, palettesize=128)
    for k, now in enumerate(clock):
        if args.preview is not None and k not in args.preview:
            continue
        for drift, *actors in thermal_actors:
            for actor in actors:
                actor.position = (drift[0] * now, drift[1] * now, 0.0)
        for key, f in (("ga", ga), ("rule", rule)):
            i = min(np.searchsorted(f["t"], now), len(f["t"]) - 1)
            planes[key].user_matrix = pose(f["p"][i], f["heading"][i], f["gamma"][i], f["bank_s"][i])
            if i >= 1:
                trail = pv.lines_from_points(f["p"][: i + 1])
                shadow = pv.lines_from_points(f["p"][: i + 1] * [1, 1, 0] + [0, 0, 0.5])
                if key == "ga":
                    trail["climb (m/s)"] = f["vz"][: i + 1]
                    pl.add_mesh(trail, name="trail_ga", scalars="climb (m/s)", cmap="RdYlGn", clim=(-2.5, 4.0),
                                line_width=4, show_scalar_bar=False)
                else:
                    pl.add_mesh(trail, name="trail_rule", color="#8c8c8c", line_width=3)
                pl.add_mesh(shadow, name=f"shadow_{key}", color="#6f7a5c", line_width=2, opacity=0.5)
        i = min(np.searchsorted(ga["t"], now), len(ga["t"]) - 1)
        cam_target = focus[i]
        pl.camera.focal_point = tuple(cam_target)
        # Behind and beside the sailplane, a little above it but always under the cumulus.
        eye = cam_target - 400 * v_dir[i] + np.array([-140 * v_dir[i][1], 140 * v_dir[i][0], 70])
        eye[2] = min(eye[2], CEILING - 40.0)
        pl.camera.position = tuple(eye)
        pl.camera.up = (0, 0, 1)
        pl.camera.view_angle = 38
        pl.camera.clipping_range = (5, 30000)
        z, vz = ga["p"][i, 2], ga["vz"][i]
        arrived = now >= ga["t"][-1]
        state = "arrived" if arrived else ("climbing" if ga["mode"][i] == 1 else "gliding")
        j = min(np.searchsorted(rule["t"], now), len(rule["t"]) - 1)
        rule_state = "arrived" if now >= rule["t"][-1] else f"{max(rule['p'][j, 0], 0.0) / 1000:4.1f} km"
        pl.add_text(f"t = {int(now // 60):02d}:{int(now % 60):02d}    {max(ga['p'][i, 0], 0.0) / 1000:4.1f} km of 20    "
                    f"altitude {z:3.0f} m    vario {vz:+.1f} m/s    {state}", name="hud", position="upper_left",
                    font_size=10, color="#1b2733")
        pl.add_text(f"red: planned climbs (latent GA)\ngrey: circling rule ({rule_state})", name="legend",
                    position="lower_right", font_size=9, color="#1b2733")
        pl.add_text("SB-XC, 20 km through drifting thermals (drawn larger than life)",
                    name="caption", position="lower_left", font_size=8, color="#3d4b57")
        if args.preview is not None:
            pl.screenshot(str(args.out.with_name(f"{args.out.stem}_frame{k}.png")))
            continue
        pl.write_frame()
        if k % 50 == 0:
            print(f"frame {k}/{len(clock)}", flush=True)
    pl.close()
    if args.preview is not None:
        print(f"{len(clock)} frames in all; previews saved next to {args.out}")
        return
    print(f"{len(clock)} frames -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
