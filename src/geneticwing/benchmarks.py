"""Benchmark soaring scenarios S1 to S3 of the GeneticWing protocol.

Each factory takes an ``instance`` seed that randomizes the start (position
and heading) around a fixed wind field, so methods are compared on the same
set of instances.

The airframe is the SB-XC sailplane fitted with a small motor (15 N): the
Albatross framing, where soaring is measured by the onboard energy it saves.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from geneticwing.aircraft import SBXC_APPROX
from geneticwing.soaring import GoalMission, LoiterMission, SoaringScenario
from geneticwing.wind import RidgeLift, Thermal, UniformWind
from geneticwing.wind.base import WindField
from geneticwing.wind.thermal import EnvironmentSink, thermal_density

MOTOR_THRUST = 15.0  # N, placeholder motor-glider conversion
CRUISE_AIRSPEED = 9.0  # m/s, slightly above best L/D


def s1_single_thermal(instance: int = 0) -> SoaringScenario:
    """S1: stay within 300 m of a known thermal for 60 s, drifting in a 2 m/s breeze."""
    rng = np.random.default_rng(instance)
    wind = UniformWind((2.0, 0.0, 0.0))
    thermal = Thermal(center=(0.0, 0.0), w_star=2.0, z_i=1000.0, drift=(2.0, 0.0))
    bearing = rng.uniform(0.0, 2.0 * np.pi)
    distance = rng.uniform(80.0, 180.0)
    start = (distance * np.cos(bearing), distance * np.sin(bearing), rng.uniform(220.0, 280.0))
    heading = rng.uniform(0.0, 360.0)
    return SoaringScenario(
        name="S1 single thermal",
        aircraft=SBXC_APPROX,
        wind=wind + thermal,
        start=start,
        heading_deg=heading,
        airspeed=CRUISE_AIRSPEED,
        horizon=60.0,
        mission=LoiterMission(center=(0.0, 0.0), radius=300.0),
        thrust_max=MOTOR_THRUST,
        n_control=32,
    )


def s2_thermal_field(instance: int = 0) -> SoaringScenario:
    """S2: reach a goal 1.3 km away in 150 s, with two thermals near the route."""
    rng = np.random.default_rng(1000 + instance)
    wind = UniformWind((1.5, 0.0, 0.0))
    thermals = [
        Thermal(center=(400.0 + rng.normal(0, 40), 80.0 + rng.normal(0, 40)), w_star=2.2, drift=(1.5, 0.0)),
        Thermal(center=(900.0 + rng.normal(0, 40), -60.0 + rng.normal(0, 40)), w_star=1.8, drift=(1.5, 0.0)),
    ]
    field = wind
    for th in thermals:
        field = field + th
    return SoaringScenario(
        name="S2 thermal field to goal",
        aircraft=SBXC_APPROX,
        wind=field,
        start=(0.0, rng.normal(0, 30), 200.0 + rng.normal(0, 15)),
        heading_deg=90.0 + rng.normal(0, 15),
        airspeed=CRUISE_AIRSPEED,
        horizon=150.0,
        mission=GoalMission(goal=(1300.0, 0.0), radius=60.0),
        thrust_max=MOTOR_THRUST,
        n_control=56,
    )


def s3_ridge(instance: int = 0) -> SoaringScenario:
    """S3: loiter for 60 s on the windward side of a ridge, 8 m/s wind across it."""
    rng = np.random.default_rng(2000 + instance)
    # Crest along north; wind blows east (from the west), windward side is x < 0.
    ridge = RidgeLift(crest=(0.0, 0.0), axis_heading_deg=0.0, radius=60.0, wind_speed=8.0)

    def clearance(p: np.ndarray) -> np.ndarray:
        # Distance to the cylindrical ridge surface, or height above flat ground.
        to_axis = np.hypot(p[..., 0], p[..., 2]) - ridge.radius
        return np.minimum(to_axis, p[..., 2])

    return SoaringScenario(
        name="S3 ridge lift",
        aircraft=SBXC_APPROX,
        wind=ridge,
        start=(-110.0 + rng.normal(0, 15), rng.normal(0, 40), 110.0 + rng.normal(0, 10)),
        heading_deg=rng.choice([0.0, 180.0]) + rng.normal(0, 10),
        airspeed=CRUISE_AIRSPEED,
        horizon=60.0,
        mission=LoiterMission(center=(-90.0, 0.0), radius=250.0),
        thrust_max=MOTOR_THRUST,
        z_min=20.0,
        clearance=clearance,
        n_control=32,
    )


SCENARIOS = {"S1": s1_single_thermal, "S2": s2_thermal_field, "S3": s3_ridge}

# Context of the S1 family used to train the student, and its normalization.
CONTEXT_NAMES = ("dx", "dy", "z", "heading_east", "heading_north", "w_star", "wind_x", "wind_y")
_CONTEXT_CENTER = np.array([0.0, 0.0, 250.0, 0.0, 0.0, 2.0, 0.0, 0.0])
_CONTEXT_SCALE = np.array([200.0, 200.0, 70.0, 1.0, 1.0, 0.8, 4.0, 4.0])


def s1_family(instance: int) -> tuple[SoaringScenario, np.ndarray]:
    """Randomized single-thermal scenario and its normalized context vector.

    Thermal strength, wind (speed and direction, the thermal drifting with it)
    and the start (bearing, distance, altitude, heading) all vary; the thermal
    is at the origin at t = 0 and the loiter area is centered on it.
    """
    rng = np.random.default_rng(50_000 + instance)
    w_star = rng.uniform(1.2, 2.8)
    wind_speed, wind_dir = rng.uniform(0.0, 4.0), rng.uniform(0.0, 2.0 * np.pi)
    wind = np.array([wind_speed * np.cos(wind_dir), wind_speed * np.sin(wind_dir)])
    bearing, distance = rng.uniform(0.0, 2.0 * np.pi), rng.uniform(60.0, 200.0)
    start = (distance * np.cos(bearing), distance * np.sin(bearing), rng.uniform(180.0, 320.0))
    heading = rng.uniform(0.0, 360.0)
    scenario = SoaringScenario(
        name="S1 family",
        aircraft=SBXC_APPROX,
        wind=UniformWind((wind[0], wind[1], 0.0)) + Thermal(center=(0.0, 0.0), w_star=w_star, drift=tuple(wind)),
        start=start,
        heading_deg=heading,
        airspeed=CRUISE_AIRSPEED,
        horizon=60.0,
        mission=LoiterMission(center=(0.0, 0.0), radius=300.0),
        thrust_max=MOTOR_THRUST,
        n_control=32,
    )
    h = np.radians(heading)
    raw = np.array([start[0], start[1], start[2], np.sin(h), np.cos(h), w_star, wind[0], wind[1]])
    return scenario, (raw - _CONTEXT_CENTER) / _CONTEXT_SCALE


# The same context, expressed in the aircraft's initial heading frame. The genome
# is heading-relative (turn rates), so this context is invariant to rotating the
# whole scene, and mirroring it left/right is exact (see ``mirror``).
HEADING_CONTEXT_NAMES = ("thermal_ahead", "thermal_left", "z", "w_star", "wind_ahead", "wind_left")
_HEADING_CENTER = np.array([0.0, 0.0, 250.0, 2.0, 0.0, 0.0])
_HEADING_SCALE = np.array([200.0, 200.0, 70.0, 0.8, 4.0, 4.0])


# The weather family's context adds the layer depth z_i: its thermals have Allen's own profile and the air
# between them sinks (:class:`~geneticwing.wind.thermal.EnvironmentSink`), as on the real-weather maps. Its
# ranges cover the three ICON-D2 days over Issoudun (w* 1.1-4.0 m/s, z_i 760-2350 m, wind 3.7-7.5 m/s, 1st to
# 99th percentile of the thermals), so the scale is wider. A context's length tells the family: 6 or 7.
WEATHER_CONTEXT_NAMES = HEADING_CONTEXT_NAMES + ("z_i",)
_WEATHER_CENTER = np.array([0.0, 0.0, 350.0, 2.7, 0.0, 0.0, 1600.0])
_WEATHER_SCALE = np.array([200.0, 200.0, 150.0, 1.0, 5.0, 5.0, 600.0])
WEATHER_CONTEXT_DIM = len(WEATHER_CONTEXT_NAMES)


def _norm(dim: int):
    return (_WEATHER_CENTER, _WEATHER_SCALE) if dim == WEATHER_CONTEXT_DIM else (_HEADING_CENTER, _HEADING_SCALE)


def weather_family(instance: int) -> tuple[SoaringScenario, np.ndarray]:
    """Randomized single-thermal scenario of the weather family and its normalized heading-frame context.

    As :func:`s1_family`, with real-weather ranges: w* 0.9-4.5 m/s, wind 0-8 m/s, z_i 600-2600 m, Allen's
    profile and the environment's sink, and the climb starting anywhere from inside the thermal (a blind pilot
    starts circling where it felt the lift) to 250 m away, between 120 and 600 m (the small UAV's band of the
    cross-country tasks), below 0.8 z_i.
    """
    rng = np.random.default_rng(70_000 + instance)
    w_star, z_i = rng.uniform(0.9, 4.5), rng.uniform(600.0, 2600.0)
    wind_speed, wind_dir = rng.uniform(0.0, 8.0), rng.uniform(0.0, 2.0 * np.pi)
    wind = (float(wind_speed * np.cos(wind_dir)), float(wind_speed * np.sin(wind_dir)))
    bearing, distance = rng.uniform(0.0, 2.0 * np.pi), rng.uniform(0.0, 250.0)
    z = rng.uniform(120.0, min(600.0, 0.8 * z_i))
    start = (float(distance * np.cos(bearing)), float(distance * np.sin(bearing)), float(z))
    scenario = SoaringScenario(
        name="weather family",
        aircraft=SBXC_APPROX,
        wind=weather_air((0.0, 0.0), w_star, z_i, wind),
        start=start,
        heading_deg=float(rng.uniform(0.0, 360.0)),
        airspeed=CRUISE_AIRSPEED,
        horizon=60.0,
        mission=LoiterMission(center=(0.0, 0.0), radius=300.0),
        thrust_max=MOTOR_THRUST,
        n_control=32,
    )
    return scenario, heading_frame_context(scenario, weather=True)


def weather_air(center, w_star: float, z_i: float, wind) -> WindField:
    """Uniform wind, one drifting thermal with Allen's profile and the sink of the air between thermals."""
    return (UniformWind((wind[0], wind[1], 0.0))
            + Thermal(center=tuple(center), w_star=w_star, z_i=z_i, drift=tuple(wind), profile="allen")
            + EnvironmentSink(float(thermal_density(z_i)), w_star, z_i))


def heading_frame_context(scenario: SoaringScenario, weather: bool = False) -> np.ndarray:
    """Normalized heading-frame context of a single-thermal scenario (``weather``: the 7-term one, with z_i)."""
    parts = scenario.wind.components
    thermal = next(p for p in parts if isinstance(p, Thermal))
    wind = next(p for p in parts if isinstance(p, UniformWind)).velocity
    h = np.radians(scenario.heading_deg)
    ahead, left = np.array([np.sin(h), np.cos(h)]), np.array([-np.cos(h), np.sin(h)])
    to_thermal = np.asarray(thermal.center) - np.asarray(scenario.start[:2])
    w = np.asarray(wind[:2])
    raw = [to_thermal @ ahead, to_thermal @ left, scenario.start[2], thermal.w_star, w @ ahead, w @ left]
    if weather:
        raw.append(thermal.z_i)
    center, scale = _norm(len(raw))
    return (np.array(raw) - center) / scale


def heading_context_raw(context: np.ndarray) -> np.ndarray:
    """Normalized heading-frame context -> physical units (m, m/s)."""
    context = np.asarray(context)
    center, scale = _norm(context.shape[-1])
    return context * scale + center


def canonical_scenario(context: np.ndarray, **overrides) -> SoaringScenario:
    """The single-thermal situation seen from the aircraft: start at the origin, nose along +x.

    Built from the normalized heading-frame context alone (S1 family, or the weather family for a 7-term
    context). Energy and constraints are invariant to this rotation, so scores match the original
    scenario's.
    """
    raw = heading_context_raw(context)
    ahead, left, z, w_star, wind_ahead, wind_left = raw[:6]
    if len(raw) == WEATHER_CONTEXT_DIM:
        wind = weather_air((ahead, left), w_star, raw[6], (wind_ahead, wind_left))
    else:
        wind = UniformWind((wind_ahead, wind_left, 0.0)) + Thermal(center=(ahead, left), w_star=w_star,
                                                                    drift=(wind_ahead, wind_left))
    scenario = SoaringScenario(
        name="single thermal (canonical frame)",
        aircraft=SBXC_APPROX,
        wind=wind,
        start=(0.0, 0.0, z),
        heading_deg=90.0,  # compass east = +x
        airspeed=CRUISE_AIRSPEED,
        horizon=60.0,
        mission=LoiterMission(center=(ahead, left), radius=300.0),
        thrust_max=MOTOR_THRUST,
        n_control=32,
    )
    return replace(scenario, **overrides) if overrides else scenario


def to_canonical(points: np.ndarray, scenario: SoaringScenario) -> np.ndarray:
    """World positions ``(..., 3)`` of a scenario -> its canonical frame."""
    h = np.radians(scenario.heading_deg)
    ahead, left = np.array([np.sin(h), np.cos(h)]), np.array([-np.cos(h), np.sin(h)])
    rel = np.asarray(points)[..., :2] - np.asarray(scenario.start[:2])
    return np.stack([rel @ ahead, rel @ left, np.asarray(points)[..., 2]], axis=-1)


def mirror(contexts: np.ndarray, genomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left/right mirror image: lateral context terms and turn-rate genes change sign.

    Turn-rate genes are normalized symmetrically around zero, so their sign
    flip is ``x -> 1 - x``; speed and path-angle genes are unchanged.
    """
    c, g = contexts.copy(), genomes.copy()
    c[:, [1, 5]] *= -1.0
    g[:, 0::3] = 1.0 - g[:, 0::3]
    return c, g


def with_horizon(scenario: SoaringScenario, horizon: float, n_control: int) -> SoaringScenario:
    return replace(scenario, horizon=horizon, n_control=n_control)


def with_margins(scenario: SoaringScenario) -> SoaringScenario:
    """Tightened constraints for the teacher (constraint tightening, as in robust MPC).

    Optimal solutions sit on their active constraints, so any imitation error
    breaks them. Solving with margins and judging under the nominal
    constraints leaves room for the student's errors.
    """
    return replace(
        scenario,
        cl_margin=0.75,
        bank_max_deg=38.0,
        load_max=2.1,
        thrust_max=0.8 * scenario.thrust_max,
        brake_capacity=0.03,
        airspeed_max=0.9 * scenario.airspeed_max,
    )
