"""Publish the two interactive 3D viewers in English under docs/viewer/, for GitHub Pages.

The pages written by ``phase4_view.py`` (a 60 s climb in closed loop) and ``phase5_view.py`` (a 20 km task) are in
French and carry their flights in a ``DATA`` block. This script translates their interface and flight labels and
copies them, data included, to ``docs/viewer/climb_60s.html`` and ``docs/viewer/xc_20km.html``.
Run: ``python scenarios/publish_viewers.py``
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMON = [
    ('const fmt = (x, d = 1) => x.toFixed(d).replace(".", ",").replace("-", "−");',
     'const fmt = (x, d = 1) => x.toFixed(d).replace("-", "−");'),
    ("Glisser : tourner · Molette : zoom · Clic droit : déplacer", "Drag: rotate · Wheel: zoom · Right-click: pan"),
    ("Suivre l'avion", "Follow the aircraft"),
    ('"Lecture"', '"Play"'),
    (">Lecture<", ">Play<"),
    ('aria-label="Temps du vol"', 'aria-label="Flight time"'),
    (">Vitesse<", ">Speed<"),
    ("<span class=\"k\">Vitesse air</span>", "<span class=\"k\">Airspeed</span>"),
    ("<span class=\"k\">Altitude</span>", "<span class=\"k\">Altitude</span>"),
]

XC = [
    ("Le SB-XC doit rejoindre un but à 20 km en enchaînant les thermiques, avec le moins de batterie possible.",
     "The SB-XC must reach a goal 20 km away by chaining thermals, using as little battery as possible."),
    ("Le même tacticien (MacCready) choisit les thermiques ; seule change la façon de monter dedans : la spirale "
     "classique, ou notre GA.",
     "The same MacCready tactician picks the thermals; only the way of climbing in them changes: the classic "
     "circling rule, or our planner."),
    ('aria-label="Vol affiché"', 'aria-label="Flight shown"'),
    ("transition entre thermiques", "glide between thermals"),
    ("montée dans un thermique", "climb in a thermal"),
    ("plané final vers le but", "final glide to the goal"),
    ("</i>moteur</span>", "</i>motor</span>"),
    ("thermique (plus opaque = plus fort)", "thermal (more opaque = stronger)"),
    ("Exagération verticale", "Vertical exaggeration"),
    ("Largeur des thermiques", "Thermal width"),
    ("<span class=\"k\">Temps de vol</span>", "<span class=\"k\">Flight time</span>"),
    ("<span class=\"k\">Distance au but</span>", "<span class=\"k\">Distance to goal</span>"),
    ("Bilan des deux vols", "Both flights"),
    ("Aucune donnée : génère la page avec scenarios/phase5_view.py.", "No data: build the page with scenarios/phase5_view.py."),
    ('["Transition", "--glide"], ["Montée", "--climb"], ["Moteur", "--motor"], ["Plané final", "--final"]',
     '["Glide", "--glide"], ["Climb", "--climb"], ["Motor", "--motor"], ["Final glide", "--final"]'),
    ("` (thermique ${f.target[i]})`", "` (thermal ${f.target[i]})`"),
    ('row("Durée"', 'row("Duration"'),
    ('row("Vitesse"', 'row("Speed"'),
    ('row("Batterie"', 'row("Battery"'),
    ('row("Mont\u00e9es"', 'row("Climbs"'),
    ('row("Montée moyenne"', 'row("Mean climb"'),
    ('row("Limites", f => f.violated.length ? (f.stalled ? "décrochage" : "marge entamée") : "ok")',
     'row("Limits", f => f.violated.length ? (f.stalled ? "stall" : "margin used") : "ok")'),
    ('"Mont\u00e9es de ce vol : "', '"Climbs of this flight: "'),
    ("m en ${Math.round(c.t1 - c.t0)} s", "m in ${Math.round(c.t1 - c.t0)} s"),
    ('|| "aucune")', '|| "none")'),
    ("Barogramme : altitude selon la distance parcourue vers le but · plancher moteur et plafond en pointillés",
     "Barogram: altitude against the distance flown toward the goal · motor floor and cloud base dotted"),
    ('aria-label="Barogramme : altitude selon la distance parcourue vers le but"',
     'aria-label="Barogram: altitude against the distance flown toward the goal"'),
    ('"Les deux superposés"', '"Both together"'),
    ("` Tâche ${task.name.replace(\"XC \", \"\")}, ${DATA.thermals.length} thermiques, vent ",
     "` Task ${task.name.replace(\"XC \", \"\")}, ${DATA.thermals.length} thermals, wind "),
    ("Les thermiques sont dessinés plus larges que nature et l'altitude est exagérée (réglables).",
     "Thermals are drawn wider than life and altitude is exaggerated (both adjustable)."),
    ("<span class=\"k\">Batterie dépensée</span>", "<span class=\"k\">Battery used</span>"),
    ("Spirale classique (MacCready", "Circling rule (MacCready"),
    (r"Mont\u00e9es par notre GA (MacCready", "Planned climbs, latent GA (MacCready"),  # escaped in the JSON
]

CLIMB = [
    ("<h1>GeneticWing en vol</h1>", "<h1>GeneticWing: a 60 s climb</h1>"),
    ("<title>GeneticWing en vol</title>", "<title>GeneticWing: a 60 s climb</title>"),
    ("Un SB-XC (4,6 m d'envergure) qui part à côté d'un thermique et a 60 s pour y monter en dépensant le moins de "
     "batterie possible.",
     "An SB-XC (4.6 m span) starts next to a thermal and has 60 s to climb in it on as little battery as possible."),
    ("Le planificateur (le GA sur le décodeur physique) calcule une trajectoire, l'autopilote la suit dans l'air réel.",
     "The planner (the GA over the physics decoder) computes a trajectory; the autopilot flies it in the real air."),
    ("Choisis un vol, tourne la scène à la souris, lance la lecture.",
     "Pick a flight, turn the scene with the mouse, press play."),
    ('aria-label="Choix du vol"', 'aria-label="Flight"'),
    ("vol réel, coloré par le vario (descend → monte)", "flight, coloured by the vario (sinking → climbing)"),
    ("plan en cours", "current plan"),
    ("cœur du thermique (réel)", "thermal core (real)"),
    ("centre que croit le planificateur", "centre the planner believes"),
    ("<span class=\"k\">Gain depuis le départ</span>", "<span class=\"k\">Gain since start</span>"),
    ("<span class=\"k\">Vario (vitesse verticale)</span>", "<span class=\"k\">Vario (vertical speed)</span>"),
    ("<span class=\"k\">Ascendance sous l'avion</span>", "<span class=\"k\">Updraft under the aircraft</span>"),
    ("<span class=\"k\">Inclinaison</span>", "<span class=\"k\">Bank</span>"),
    ("<span class=\"k\">Moteur</span>", "<span class=\"k\">Motor</span>"),
    ("<span class=\"k\">Distance au cœur</span>", "<span class=\"k\">Distance to core</span>"),
    ("Bilan du vol (60 s)", "This flight (60 s)"),
    ("60,0 s", "60.0 s"),
    ("<span>Altitude (trait)</span><span>Moteur (aire)</span><span>Vario (couleur du trait)</span>"
     "<span>Repli : temps du plan en cours</span>",
     "<span>Altitude (line)</span><span>Motor (area)</span><span>Vario (line colour)</span>"
     "<span>Replanning: time of the current plan</span>"),
    ('aria-label="Altitude et moteur au cours du vol"', 'aria-label="Altitude and motor during the flight"'),
    ("Aucune donnée de vol : génère la page avec scenarios/phase4_view.py.",
     "No flight data: build the page with scenarios/phase4_view.py."),
    ('wing_stall: "décrochage aile intérieure", aileron: "ailerons en butée", cl: "C_L trop haut", '
     'thrust_max: "moteur au maximum", load: "facteur de charge", bank: "inclinaison", mission: "sortie de zone", '
     'airspeed_max: "vitesse max"',
     'wing_stall: "inner-wing stall", aileron: "aileron at its limit", cl: "C_L too high", '
     'thrust_max: "motor at maximum", load: "load factor", bank: "bank", mission: "left the zone", '
     'airspeed_max: "maximum airspeed"'),
    ('.join(", ") : "aucune"', '.join(", ") : "none"'),
    ("<span>Score (plus négatif = mieux)</span>", "<span>Score (more negative = better)</span>"),
    ("<span>Énergie gagnée (altitude + vitesse)</span>", "<span>Energy gained (altitude + speed)</span>"),
    ("<span>Batterie dépensée</span>", "<span>Battery used</span>"),
    ("<span>Plans calculés</span>", "<span>Plans computed</span>"),
    ("<span>Limites dépassées</span>", "<span>Limits broken</span>"),
    ("L'air est exactement celui que le planificateur croit. Un seul plan, suivi 60 s : la ligne réelle se pose sur "
     "le plan.",
     "The air is exactly what the planner believes. One plan, flown for 60 s: the real line sits on the plan."),
    ("Mêmes conditions plus des rafales. L'autopilote garde la trajectoire, mais il lui faut plus de moteur, et les "
     "à-coups font décrocher l'aile intérieure par moments.",
     "The same, plus gusts. The autopilot holds the path but needs more motor, and the jolts stall the inner wing "
     "at times."),
    ("Rafales, et un nouveau plan toutes les 10 s depuis l'endroit où l'avion se trouve vraiment.",
     "Gusts, and a new plan every 10 s from where the aircraft really is."),
    ("Le planificateur croit le thermique à la croix pointillée. L'avion suit bien son plan, mais tourne au mauvais "
     "endroit.",
     "The planner believes the thermal is at the dotted cross. The aircraft follows its plan well, but circles in "
     "the wrong place."),
    ("Même erreur, avec un nouveau plan toutes les 10 s. Replanifier ne corrige pas la croyance : l'avion recalcule "
     "de bons cercles… autour du mauvais centre.",
     "The same error, with a new plan every 10 s. Replanning does not correct the belief: the aircraft recomputes "
     "good circles… around the wrong centre."),
    ('"Air calme, plan fait une fois"', '"Calm air, planned once"'),
    ('"Rafales, plan fait une fois"', '"Gusts, planned once"'),
    ('"Rafales, replan toutes les 10 s"', '"Gusts, replanned every 10 s"'),
    (r'"Thermique mal estim\u00e9, plan fait une fois"', '"Thermal misplaced, planned once"'),  # escaped in the JSON
    (r'"Thermique mal estim\u00e9, replan toutes les 10 s"', '"Thermal misplaced, replanned every 10 s"'),
]


def translate(src: Path, dst: Path, table: list[tuple[str, str]]) -> None:
    text = src.read_text(encoding="utf-8")
    missing = [a[:60] for a, _ in table if a not in text]
    for a, b in table:
        text = text.replace(a, b)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    print(f"{src.name} -> {dst} ({len(text) / 1e3:.0f} kB); not found: {missing or 'none'}")


def main() -> None:
    translate(ROOT / "results/phase5_view/xc.html", ROOT / "docs/viewer/xc_20km.html", COMMON + XC)
    translate(ROOT / "results/phase4_view/flights.html", ROOT / "docs/viewer/climb_60s.html", COMMON + CLIMB)


if __name__ == "__main__":
    main()
