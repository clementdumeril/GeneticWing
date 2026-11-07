"""Animated 3D flights for the README: ``docs/figures/flight_thermal.gif`` and ``flight_xc20km.gif``.

Reads the flights in ``docs/data/`` (written by ``--extract`` from the pages of
``scenarios/phase4_view.py`` and ``scenarios/phase5_view.py``) and draws them
with matplotlib 3D:

* ``flight_thermal.gif``: the 60 s climb in one drifting thermal, latent GA plan, calm air;
* ``flight_xc20km.gif``: the 20 km task, a camera following the planned climbs, with
  the circling rule's path faint and both barograms underneath.

Run:
    python scripts/make_flight_gifs.py                       # both GIFs from docs/data
    python scripts/make_flight_gifs.py --frames tmp/frames   # also save every 10th frame as PNG (--preview: only those)
    python scripts/make_flight_gifs.py --extract results/phase4_view/flights.html results/phase5_view/xc.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402
from PIL import Image  # noqa: E402

PREVIEW = False  # --preview: draw every 10th frame only, no GIF
ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "docs" / "data", ROOT / "docs" / "figures"

INK, MUTED, GRID, FAINT = "#1a1a1a", "#5f5f5f", "#dcdcdc", "#a8a8a8"
THERMAL = "#d98c2b"
CLIMB = LinearSegmentedColormap.from_list("sink_climb", ["#b2182b", "#9e9e9e", "#2166ac"])
NORM = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=5.0)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


# ---------------------------------------------------------------- data

def _page_json(path: Path) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("const DATA ="):
            return json.loads(line[len("const DATA ="):].strip().rstrip(";"))
    raise ValueError(f"no 'const DATA =' line in {path}")


def extract(thermal_page: Path, xc_page: Path) -> None:
    """Keep only what the GIFs draw, from the viewer pages of phase4_view.py and phase5_view.py."""
    d = _page_json(thermal_page)
    calm = next(f for f in d["flights"] if f["key"] == "calm_open")
    keep = ("key", "label", "t", "p", "v", "bank", "airspeed", "thrust", "updraft", "plans", "cost", "battery",
            "violated", "stalled")
    out = {"source": "scenarios/phase4_view.py", "instance": d["instance"], "planner": d["planner"],
           "thermal": d["thermal"], "wind": d["wind"], "flight": {k: calm[k] for k in keep}}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "flight_thermal.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")

    x = _page_json(xc_page)
    keep = ("key", "label", "t", "p", "vz", "bank", "mode", "climbs", "arrived", "duration", "speed_kmh",
            "battery_kj", "violated", "stalled")
    out = {"source": "scenarios/phase5_view.py", "task": x["task"], "thermals": x["thermals"],
           "flights": [{k: f[k] for k in keep} for f in x["flights"]]}
    (DATA / "flight_xc20km.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    for name in ("flight_thermal.json", "flight_xc20km.json"):
        print(f"wrote docs/data/{name} ({(DATA / name).stat().st_size / 1e3:.0f} kB)")


# ---------------------------------------------------------------- drawing helpers

def _style3d(ax) -> None:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"].update(color=GRID, linewidth=0.5)
        axis._axinfo["tick"].update(inward_factor=0.0, outward_factor=0.2)
        axis.line.set_color(FAINT)
    ax.tick_params(labelsize=7, pad=0)
    ax.zaxis.pane.set_facecolor((0.965, 0.965, 0.96, 1.0))  # the floor


def _frame_axes(v: np.ndarray, bank_deg: float):
    """Forward, right and up unit vectors of an aircraft flying along v, banked right-wing-down positive."""
    f = v / max(np.linalg.norm(v), 1e-6)
    r0 = np.cross(f, [0.0, 0.0, 1.0])
    r0 /= max(np.linalg.norm(r0), 1e-6)
    u0 = np.cross(r0, f)
    phi = np.radians(bank_deg)
    return f, np.cos(phi) * r0 - np.sin(phi) * u0, np.cos(phi) * u0 + np.sin(phi) * r0


def _glider(pos, v, bank_deg, span):
    """Wing, tailplane and fin polygons, plus the fuselage line, of a sailplane glyph of the given span."""
    f, r, u = _frame_axes(np.asarray(v, float), bank_deg)
    s = span / 2.0
    c = 0.12 * span  # root chord, wider than the real wing so that the planform reads
    at = lambda a, b, d=0.0: pos + a * f + b * r + d * u  # noqa: E731
    wing = [at(0.35 * c, 0.0), at(0.45 * c, s), at(-0.25 * c, s), at(-0.65 * c, 0.0),
            at(-0.25 * c, -s), at(0.45 * c, -s)]
    tail_x = -0.42 * span
    tail = [at(tail_x + 0.5 * c, 0.0), at(tail_x + 0.1 * c, 0.2 * span), at(tail_x - 0.35 * c, 0.2 * span),
            at(tail_x - 0.35 * c, -0.2 * span), at(tail_x + 0.1 * c, -0.2 * span)]
    fin = [at(tail_x + 0.4 * c, 0.0), at(tail_x - 0.4 * c, 0.0, 0.16 * span), at(tail_x - 0.7 * c, 0.0, 0.16 * span),
           at(tail_x - 0.6 * c, 0.0)]
    body = np.array([at(0.22 * span, 0.0), at(tail_x - 0.5 * c, 0.0)])
    return [wing, tail, fin], body, u


def _draw_glider(ax, pos, v, bank_deg, span, elev, azim, color=INK):
    """Dark when the camera sees the top of the wing, light grey when it sees the underside: shows the bank."""
    polys, body, up = _glider(np.asarray(pos, float), v, bank_deg, span)
    e, a = np.radians(elev), np.radians(azim)
    top_seen = up @ np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]) > 0
    face = color if top_seen else "#9a9a9a"
    ax.add_collection3d(Poly3DCollection(polys, facecolors=face, edgecolors=color, linewidths=0.6))
    ax.plot(*body.T, color=color, lw=1.6, solid_capstyle="round")


def _cylinder(ax, cx, cy, z, radius, alpha, azim, color=THERMAL):
    """Translucent column drawn as its outline seen from azimuth ``azim`` (one polygon, so no facet seams):
    left silhouette up, far half of the top ring, right silhouette down, near half of the bottom ring."""
    z, r = np.asarray(z, float), np.asarray(radius, float)
    a = np.radians(azim)
    left, right = a + np.pi / 2, a - np.pi / 2
    top = np.linspace(left, right + 2 * np.pi, 24)
    bottom = np.linspace(right, left, 24)
    pts = [(cx + r[j] * np.cos(left), cy + r[j] * np.sin(left), z[j]) for j in range(len(z))]
    pts += [(cx + r[-1] * np.cos(q), cy + r[-1] * np.sin(q), z[-1]) for q in top]
    pts += [(cx + r[j] * np.cos(right), cy + r[j] * np.sin(right), z[j]) for j in range(len(z) - 1, -1, -1)]
    pts += [(cx + r[0] * np.cos(q), cy + r[0] * np.sin(q), z[0]) for q in bottom]
    ax.add_collection3d(Poly3DCollection([pts], facecolors=color, edgecolors="none", alpha=alpha))
    ring = np.linspace(0.0, 2 * np.pi, 48)
    ax.plot(cx + r[-1] * np.cos(ring), cy + r[-1] * np.sin(ring), z[-1], color=color, lw=0.6, alpha=0.5)


def _trail(ax, p, vz, lw=2.0, mask=None):
    seg = np.stack([p[:-1], p[1:]], axis=1)
    c = 0.5 * (vz[:-1] + vz[1:])
    if mask is not None:
        keep = mask[:-1] & mask[1:]
        seg, c = seg[keep], c[keep]
    if len(seg):
        ax.add_collection3d(Line3DCollection(seg, colors=CLIMB(NORM(c)), linewidths=lw, capstyle="round"))


def _colorbar(fig, rect):
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CLIMB)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-2, 0, 2, 4])
    cb.outline.set_edgecolor(FAINT)
    cb.ax.tick_params(labelsize=7, length=2, color=FAINT)
    cb.ax.set_title("climb rate (m/s)", fontsize=7.5, color=MUTED, pad=3)


def _image(fig) -> Image.Image:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[..., :3].copy())


def _save_gif(frames: list[Image.Image], path: Path, fps: float, hold: int) -> None:
    if PREVIEW:
        return
    """One shared 112-colour palette, pure white kept; nearest-colour mapping done here (Pillow's maps white to grey)."""
    n = 96

    def snapped(im):  # near-white anti-aliasing to pure white: large flat areas compress well
        a = np.asarray(im, dtype=np.int64).copy()
        a[a.min(axis=2) >= 246] = 255
        return a

    mid = snapped(frames[len(frames) // 2])
    pal = Image.fromarray(mid.astype(np.uint8)).quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    rgb = np.array(pal.getpalette()[: 3 * n], dtype=np.int64).reshape(-1, 3)
    rgb[np.argmax(rgb.sum(axis=1))] = 255
    flat = rgb.astype(np.uint8).flatten().tolist()
    q = []
    for im in frames:
        a = snapped(im)
        u, inv = np.unique((a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2], return_inverse=True)
        cols = np.stack([u >> 16, (u >> 8) & 255, u & 255], axis=1)
        best = ((cols[:, None, :] - rgb[None]) ** 2).sum(axis=-1).argmin(axis=1).astype(np.uint8)
        p = Image.frombytes("P", (a.shape[1], a.shape[0]), best[inv].tobytes())
        p.putpalette(flat)
        q.append(p)
    q += [q[-1]] * hold
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"wrote {path.relative_to(ROOT)} ({len(q)} frames, {path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------- (a) one thermal

def thermal_gif(out: Path, frames_dir: Path | None, fps: float = 12.0) -> None:
    d = json.loads((DATA / "flight_thermal.json").read_text(encoding="utf-8"))
    f, th = d["flight"], d["thermal"]
    t, p, v = np.array(f["t"]), np.array(f["p"]), np.array(f["v"])
    bank, vz = np.array(f["bank"]), np.array(f["v"])[:, 2]
    plan = np.array(f["plans"][0]["p"])
    zt, rt = np.array(th["z"]), np.array(th["core_radius"])
    c0, drift = np.array(th["center"]), np.array(th["drift"])
    zlo, zhi = 180.0, 380.0
    zc = np.linspace(zlo, zhi, 9)
    rc = np.interp(zc, zt, rt)
    lim = {"x": (-95.0, 100.0), "y": (-65.0, 65.0)}

    def scene(fig, i, clean=False):
        ax = fig.add_axes([0.0, 0.01, 1.0, 1.0], projection="3d", computed_zorder=False)
        _style3d(ax)
        ax.set_xlim(*lim["x"]), ax.set_ylim(*lim["y"]), ax.set_zlim(zlo, zhi)
        ax.set_box_aspect((np.ptp(lim["x"]), np.ptp(lim["y"]), zhi - zlo), zoom=0.95)
        ax.set_xlabel("east (m)", fontsize=8, labelpad=-2)
        ax.set_ylabel("north (m)", fontsize=8, labelpad=-2)
        ax.set_zlabel("altitude (m)", fontsize=8, labelpad=-2)
        if clean:
            ax.set_axis_off()
        azim = -64.0 + 48.0 * i / (len(t) - 1)
        ax.view_init(elev=22.0, azim=azim)

        cx, cy = c0 + drift * t[i]
        ring = np.linspace(0, 2 * np.pi, 60)
        r_floor = np.interp(zlo, zt, rt)
        ax.plot(cx + r_floor * np.cos(ring), cy + r_floor * np.sin(ring), zlo, color=THERMAL, lw=0.8, alpha=0.7)
        _cylinder(ax, cx, cy, zc, rc, alpha=0.13, azim=azim)
        ax.plot([cx, cx], [cy, cy], [zlo, zhi], color=THERMAL, lw=0.8, ls=(0, (4, 3)), alpha=0.9)

        ax.plot(*plan[:, :2].T, plan[:, 2], color=MUTED, lw=0.9, ls=(0, (2, 2)), alpha=0.7)
        ax.plot(p[: i + 1, 0], p[: i + 1, 1], zlo, color="#c4c4c4", lw=1.0)  # shadow on the floor
        _trail(ax, p[: i + 1], vz[: i + 1], lw=2.2)
        _draw_glider(ax, p[i], v[i], bank[i], 4.6 * 5, 22.0, azim)

    fig = plt.figure(figsize=(10.0, 7.2), dpi=100, facecolor="white")
    idx = list(range(0, len(t), 2)) + [len(t) - 1]
    images = []
    for k, i in enumerate(idx):
        if PREVIEW and k % 10 and i != idx[-1]:
            continue
        fig.clf()
        scene(fig, i)
        fig.text(0.035, 0.93, f"t = {t[i]:4.1f} s", fontsize=11, family="DejaVu Sans Mono")
        fig.text(0.035, 0.895, f"altitude {p[i, 2]:5.0f} m   climb {vz[i]:+4.1f} m/s   bank {bank[i]:+3.0f}°",
                 fontsize=9, family="DejaVu Sans Mono", color=MUTED)
        fig.text(0.035, 0.86, "SB-XC sailplane, drawn 5x\none latent-GA plan, flown open loop in calm air\n"
                 "orange: thermal core, drifting with the wind\ndashed: the plan",
                 fontsize=8, color=MUTED, linespacing=1.6, va="top")
        _colorbar(fig, [0.76, 0.9, 0.18, 0.018])
        images.append(_image(fig))
        if frames_dir is not None and (k % 10 == 0 or i == idx[-1]):
            images[-1].save(frames_dir / f"thermal_{k:03d}.png")
    plt.close(fig)
    _save_gif(images, out, fps, hold=int(1.5 * fps))

    # still for overview.tex: the last frame without axes or text, cropped to what is drawn
    fig = plt.figure(figsize=(10.0, 7.2), dpi=200, facecolor="white")
    scene(fig, idx[-1], clean=True)
    still = np.asarray(_image(fig))
    plt.close(fig)
    ys, xs = np.nonzero(still.min(axis=2) < 250)
    still = still[ys.min() - 8: ys.max() + 8, xs.min() - 8: xs.max() + 8]
    alpha = np.where(still.min(axis=2) == 255, 0, 255).astype(np.uint8)  # white background made transparent
    Image.fromarray(np.dstack([still, alpha])).save(FIGS / "overview" / "flight_thermal.png")


# ---------------------------------------------------------------- (b) 20 km task

def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    k = np.exp(-0.5 * (np.arange(-3 * width, 3 * width + 1) / width) ** 2)
    pad = np.pad(x, (3 * width, 3 * width), mode="edge")
    return np.convolve(pad, k / k.sum(), mode="valid")


def xc_gif(out: Path, frames_dir: Path | None, fps: float = 12.0) -> None:
    d = json.loads((DATA / "flight_xc20km.json").read_text(encoding="utf-8"))
    task, thermals = d["task"], d["thermals"]
    rule, ga = (next(f for f in d["flights"] if f["key"] == k) for k in ("rule", "ga"))
    goal = np.array(task["goal"])
    G = {k: np.array(ga[k]) for k in ("t", "p", "vz", "bank", "mode")}
    R = {k: np.array(rule[k]) for k in ("t", "p", "vz", "mode")}
    for F in (G, R):
        F["dist"] = (task["distance"] - np.linalg.norm(F["p"][:, :2] - goal, axis=1)) / 1e3
    vel = np.gradient(G["p"], G["t"], axis=0)
    cam_x, cam_y = _smooth(G["p"][:, 0], 40), _smooth(G["p"][:, 1], 40)
    hx, hy, zlo, zhi, vscale = 250.0, 150.0, 200.0, 625.0, 0.6  # window half-sizes (m); heights drawn at 0.6x

    # frame times: slow in the climbs, fast in the glides
    times, tt = [], 0.0
    while tt < G["t"][-1]:
        times.append(tt)
        j = min(np.searchsorted(G["t"], tt), len(G["t"]) - 1)
        tt += 3.5 if G["mode"][j] == 1 else 32.0
    times.append(G["t"][-1])

    fig = plt.figure(figsize=(12.0, 7.6), dpi=100, facecolor="white")
    images = []
    for k, tnow in enumerate(times):
        if PREVIEW and k % 10 and k != len(times) - 1:
            continue
        fig.clf()
        i = min(np.searchsorted(G["t"], tnow), len(G["t"]) - 1)
        ir = min(np.searchsorted(R["t"], tnow), len(R["t"]) - 1)
        cx = np.clip(cam_x[i], G["p"][i, 0] - 0.75 * hx, G["p"][i, 0] + 0.75 * hx)
        cy = np.clip(cam_y[i], G["p"][i, 1] - 0.75 * hy, G["p"][i, 1] + 0.75 * hy)
        xl, yl = (cx - hx, cx + hx), (cy - hy, cy + hy)

        ax = fig.add_axes([0.04, 0.29, 0.96, 0.73], projection="3d", computed_zorder=False)
        _style3d(ax)
        ax.set_xlim(*xl), ax.set_ylim(*yl), ax.set_zlim(zlo, zhi)
        ax.set_box_aspect((2 * hx, 2 * hy, (zhi - zlo) * vscale), zoom=1.35)
        ax.set_xlabel("east (m)", fontsize=8, labelpad=-2)
        ax.set_ylabel("north (m)", fontsize=8, labelpad=-2)
        ax.set_zlabel("altitude (m)", fontsize=8, labelpad=-2)
        azim = -72.0 + 10.0 * np.sin(2 * np.pi * tnow / 1500.0)
        ax.view_init(elev=16.0, azim=azim)

        zc = np.array([zlo, task["z_ceiling"]])
        for th in thermals:
            tx, ty = np.array(th["c"]) + np.array(th["d"]) * tnow
            if xl[0] + th["r"] < tx < xl[1] - th["r"] and yl[0] + th["r"] < ty < yl[1] - th["r"]:
                a = 0.07 + 0.10 * np.clip((th["w"] - 1.2) / 1.6, 0, 1)
                _cylinder(ax, tx, ty, zc, [th["r"]] * 2, alpha=a, azim=azim)
                ring = np.linspace(0, 2 * np.pi, 48)
                ax.plot(tx + th["r"] * np.cos(ring), ty + th["r"] * np.sin(ring), zlo, color=THERMAL, lw=0.7,
                        alpha=0.6)

        def inside(P):
            return ((P[:, 0] > xl[0]) & (P[:, 0] < xl[1]) & (P[:, 1] > yl[0]) & (P[:, 1] < yl[1])
                    & (P[:, 2] > zlo))

        Pr = R["p"][: ir + 1]
        mr = inside(Pr)
        ax.plot(*np.where(mr[:, None], Pr, np.nan).T, color=FAINT, lw=1.0, alpha=0.8)
        Pg = G["p"][: i + 1]
        mg = inside(Pg)
        shadow = np.where(mg[:, None], Pg, np.nan)
        ax.plot(shadow[:, 0], shadow[:, 1], zlo, color="#c8c8c8", lw=0.9)
        _trail(ax, Pg, G["vz"][: i + 1], lw=2.0, mask=mg)
        if mr[-1]:
            ax.scatter(*Pr[-1], s=14, color=FAINT, depthshade=False)
        _draw_glider(ax, G["p"][i], vel[i], G["bank"][i], 4.6 * 6, 16.0, azim)

        fig.text(0.03, 0.95, f"t = {int(tnow // 60):2d} min {int(tnow % 60):02d} s", fontsize=11,
                 family="DejaVu Sans Mono")
        fig.text(0.03, 0.918, f"planned climbs: {max(G['dist'][i], 0):5.1f} km, {G['p'][i, 2]:4.0f} m\n"
                 f"circling rule:  {max(R['dist'][ir], 0):5.1f} km, {R['p'][ir, 2]:4.0f} m",
                 fontsize=9, family="DejaVu Sans Mono", color=MUTED, va="top", linespacing=1.5)
        fig.text(0.03, 0.80, "camera follows the planned climbs; heights drawn at 0.6x\nglider drawn 6x; grey: circling rule\n"
                 "orange: thermals, drifting with the wind",
                 fontsize=8, color=MUTED, va="top", linespacing=1.5)
        _colorbar(fig, [0.03, 0.66, 0.15, 0.016])

        bx = fig.add_axes([0.075, 0.07, 0.89, 0.17])
        for side in ("top", "right"):
            bx.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            bx.spines[side].set_color(FAINT)
        bx.tick_params(labelsize=7.5, length=2, color=FAINT)
        for z in (task["z_floor"], task["z_ceiling"]):
            bx.axhline(z, color=GRID, lw=0.8, ls=(0, (3, 3)))
        bx.text(20.0, task["z_ceiling"] + 12, "ceiling", fontsize=7, color=FAINT, ha="right", va="bottom")
        bx.text(20.0, task["z_floor"] + 12, "motor floor", fontsize=7, color=FAINT, ha="right", va="bottom")
        bx.plot(R["dist"], R["p"][:, 2], color=FAINT, lw=0.7, alpha=0.35)
        bx.plot(G["dist"], G["p"][:, 2], color=INK, lw=0.7, alpha=0.15)
        bx.plot(R["dist"][: ir + 1], R["p"][: ir + 1, 2], color=FAINT, lw=1.3, label="circling rule")
        bx.plot(G["dist"][: i + 1], G["p"][: i + 1, 2], color=INK, lw=1.3, label="planned climbs (latent GA)")
        bx.plot(R["dist"][ir], R["p"][ir, 2], "o", ms=4, color=FAINT)
        bx.plot(G["dist"][i], G["p"][i, 2], "o", ms=4, color=INK)
        bx.set_xlim(0, 20.0), bx.set_ylim(100, 660)
        bx.set_yticks([200, 400, 600])
        bx.set_xlabel("distance flown toward the goal (km)", fontsize=8, labelpad=1)
        bx.set_ylabel("altitude (m)", fontsize=8)
        bx.legend(loc="upper left", bbox_to_anchor=(0.0, 1.32), ncol=2, frameon=False, fontsize=8,
                  handlelength=1.6)
        images.append(_image(fig))
        if frames_dir is not None and (k % 10 == 0 or k == len(times) - 1):
            images[-1].save(frames_dir / f"xc_{k:03d}.png")
    plt.close(fig)
    _save_gif(images, out, fps, hold=int(2 * fps))

    # still for overview.tex: the two barograms, large enough to read at thumbnail size
    fig, bx = plt.subplots(figsize=(3.6, 1.75), dpi=300)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.25, top=0.97)
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
    bx.plot(R["dist"], R["p"][:, 2], color=FAINT, lw=1.1, label="circling rule")
    bx.plot(G["dist"], G["p"][:, 2], color=INK, lw=1.1, label="planned climbs")
    bx.set_xlim(0, 20), bx.set_ylim(150, 680), bx.set_yticks([200, 400, 600]), bx.set_xticks([0, 5, 10, 15, 20])
    bx.tick_params(labelsize=7, length=2)
    bx.set_xlabel("distance toward the goal (km)", fontsize=7, labelpad=1)
    bx.set_ylabel("altitude (m)", fontsize=7, labelpad=1)
    bx.legend(loc="upper right", ncol=2, frameon=False, fontsize=6.5, handlelength=1.4, borderaxespad=0.0)
    fig.savefig(FIGS / "overview" / "flight_xc20km.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extract", nargs=2, type=Path, metavar=("THERMAL_HTML", "XC_HTML"),
                        help="write docs/data/*.json from the phase4_view.py and phase5_view.py pages, then stop")
    parser.add_argument("--only", choices=("thermal", "xc"))
    parser.add_argument("--frames", type=Path, help="also save every 10th frame as PNG in this folder")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--preview", action="store_true", help="with --frames: every 10th frame only, no GIF")
    args = parser.parse_args()
    global PREVIEW
    PREVIEW = args.preview and args.frames is not None
    if args.extract:
        extract(*args.extract)
        return
    if args.frames:
        args.frames.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "thermal"):
        thermal_gif(FIGS / "flight_thermal.gif", args.frames, args.fps)
    if args.only in (None, "xc"):
        xc_gif(FIGS / "flight_xc20km.gif", args.frames, args.fps)


if __name__ == "__main__":
    main()
