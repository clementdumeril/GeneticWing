"""Build the animated GeneticWing presentation: the template plus the real flights it replays."""

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).parent / "geneticwing_live.html"
OUT = ROOT / "results" / "presentation" / "geneticwing_live.html"


def data(path):
    return json.loads(re.search(r"const DATA = (\{.*?\});\n", (ROOT / path).read_text(encoding="utf-8"), re.S).group(1))


def r(a, d=1):
    return np.round(np.asarray(a, float), d).tolist()


loop = data("results/phase4_view/flights.html")
flights = {}
for f in loop["flights"]:
    if f["key"] not in ("calm_open", "wrong_mpc"):
        continue
    p = np.array(f["p"])
    flights[f["key"]] = {"t": r(f["t"], 1), "x": r(p[:, 0]), "y": r(p[:, 1]), "z": r(p[:, 2], 0),
                         "vz": r(np.array(f["v"])[:, 2], 2),
                         "plans": [{"t0": pl["t0"], "x": r(np.array(pl["p"])[:, 0]), "y": r(np.array(pl["p"])[:, 1])} for pl in f["plans"]],
                         "belief": f["belief"], "cost": f["cost"]}
th = loop["thermal"]
core = float(np.interp(300, th["z"], th["core_radius"]))

xc = data("results/phase5_view/xc.html")
xflights = []
for f in xc["flights"]:
    t = np.array(f["t"]); keep = np.unique(np.searchsorted(t, np.arange(0, t[-1], 5.0)))
    p = np.array(f["p"])[keep]
    xflights.append({"label": "spirale classique" if f["key"] == "rule" else "montées par le GA",
                     "t": r(t[keep], 0), "x": r(p[:, 0], 0), "y": r(p[:, 1], 0), "z": r(p[:, 2], 0),
                     "mode": [f["mode"][k] for k in keep], "duration": f["duration"], "speed": f["speed_kmh"]})
thermals = [{"x": t["c"][0], "y": t["c"][1], "dx": t["d"][0], "dy": t["d"][1], "w": t["w"]}
            for t in xc["thermals"] if abs(t["c"][1]) < 1200]

payload = {"loop": {"flights": flights, "thermal": {"c": th["center"], "d": th["drift"], "core": round(core, 1)}},
           "xc": {"flights": xflights, "thermals": thermals, "distance": xc["task"]["distance"]}}
html = TEMPLATE.read_text(encoding="utf-8").replace("/*LIVE_JSON*/null", json.dumps(payload, separators=(",", ":")))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")
