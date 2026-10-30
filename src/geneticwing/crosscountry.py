"""Phase 5: cross-country flight -- reach a distant goal by chaining thermals.

A task (:func:`xc_family`) puts the SB-XC at a start point with a goal some
kilometres away and a field of thermals scattered along the corridor, all
drifting with the wind. The flight ends at the goal or at the time limit.

The pilot has two levels, as a cross-country glider pilot does:

* the **tactician** chooses where to go next -- the thermal that brings the
  most progress toward the goal among those reachable in a glide, or the goal
  itself once a final glide is safe -- and when to leave a thermal (the
  MacCready rule: once the climb falls below the setting ``M``, or at the
  ceiling);
* the **flying** level glides at the MacCready speed-to-fly, circles in the
  thermal, and uses the motor only below a floor.

:class:`MacCreadyPilot` flies both levels with simple rules. The circling can be
handed to a planner instead (:class:`PlannedClimb`): every climb is flown in
blocks of 60 s, each planned once from the state reached and flown by the path
tracker, so phase 3-4 planners are used exactly as they were trained.

Score: the battery energy spent minus the energy stored on arrival (altitude
and speed, through the motor's efficiency), divided by the energy of flying the
same distance on the motor at the best lift-to-drag speed. Flying the whole way
on the motor scores about 1; below 1 the thermals saved battery; below 0 the
aircraft arrives with more energy than it left with.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from geneticwing.aircraft import Aircraft, SBXC_APPROX
from geneticwing.benchmarks import MOTOR_THRUST
from geneticwing.flatness import FlatState, flat_state
from geneticwing.simulation import aerodynamic_acceleration, rk4_step
from geneticwing.soaring import GoalMission, LoiterMission, SoaringProblem, SoaringScenario
from geneticwing.strip import StripWing, sbxc_wing, turn_loads
from geneticwing.wind.base import CompositeWind, UniformWind, WindField
from geneticwing.wind.thermal import _GAUSSIAN_DISK_MEAN, Thermal

EFFICIENCY = 0.5  # propulsive efficiency, as the soaring problems


# -- task -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class ThermalField(WindField):
    """Many :class:`Thermal` at once, vectorized (the same field as their sum, much faster)."""

    thermals: tuple[Thermal, ...]

    def __post_init__(self):
        th = self.thermals
        f = lambda xs: np.array(xs, dtype=float)  # noqa: E731
        object.__setattr__(self, "_c", f([x.center for x in th]).reshape(-1, 2))
        object.__setattr__(self, "_d", f([x.drift for x in th]).reshape(-1, 2))
        object.__setattr__(self, "_w", f([x.w_star for x in th]))
        object.__setattr__(self, "_zi", f([x.z_i for x in th]))
        object.__setattr__(self, "_sink", f([x.environment_sink for x in th]))

    def __call__(self, position, t=0.0):
        position = np.asarray(position, dtype=float)
        t = np.broadcast_to(np.asarray(t, dtype=float), position.shape[:-1])[..., None]
        z = position[..., 2][..., None]
        ratio = np.clip(z / self._zi, 0.0, None)
        cbrt = np.cbrt(ratio)
        mean = np.clip(self._w * cbrt * (1.0 - 1.1 * ratio), 0.0, None)
        core = np.maximum(10.0, 0.102 * cbrt * (1.0 - 0.25 * ratio) * self._zi) / 2.0
        dx = position[..., 0][..., None] - (self._c[:, 0] + self._d[:, 0] * t)
        dy = position[..., 1][..., None] - (self._c[:, 1] + self._d[:, 1] * t)
        w = mean / _GAUSSIAN_DISK_MEAN * np.exp(-(dx**2 + dy**2) / core**2)
        w = np.where((z > 0.0) & (z < self._zi), w - self._sink, 0.0)
        out = np.zeros(position.shape)
        out[..., 2] = w.sum(axis=-1)
        return out


@dataclass(frozen=True)
class XCTask:
    name: str
    start: tuple[float, float, float]
    heading_deg: float  # compass heading at the start (toward the goal)
    goal: tuple[float, float]
    wind: tuple[float, float]  # m/s, horizontal; every thermal drifts with it
    thermals: tuple[Thermal, ...]
    aircraft: Aircraft = SBXC_APPROX
    strip_wing: StripWing | None = field(default_factory=sbxc_wing)
    thrust_max: float = MOTOR_THRUST
    goal_radius: float = 100.0  # m
    z_floor: float = 120.0  # m, the motor starts below
    z_ceiling: float = 600.0  # m, cloud base: climbs stop there
    airspeed: float = 9.0  # m/s at the start
    time_limit: float = 1500.0  # s

    @property
    def air(self) -> WindField:
        return UniformWind((self.wind[0], self.wind[1], 0.0)) + ThermalField(self.thermals)

    @property
    def distance(self) -> float:
        return float(np.hypot(self.goal[0] - self.start[0], self.goal[1] - self.start[1]))


def xc_family(
    instance: int, distance: float = 20_000.0, density: float = 1.0, corridor: float = 2_000.0
) -> XCTask:
    """Randomized task: goal ``distance`` m east of the start, thermals in a corridor around the route.

    Thermals: a Poisson field of ``density`` per km^2 over the corridor (``corridor`` m wide,
    from 500 m before the start to 500 m past the goal), strength w* in [1.2, 2.8] m/s
    (the S1 family's range), all drifting with a wind of 0-4 m/s from any direction.
    """
    rng = np.random.default_rng(70_000 + instance)
    wind_speed, wind_dir = rng.uniform(0.0, 4.0), rng.uniform(0.0, 2.0 * np.pi)
    wind = (wind_speed * np.cos(wind_dir), wind_speed * np.sin(wind_dir))
    x0, x1 = -500.0, distance + 500.0
    n = rng.poisson(density * (x1 - x0) * corridor / 1e6)
    thermals = tuple(
        Thermal(center=(float(rng.uniform(x0, x1)), float(rng.uniform(-corridor / 2, corridor / 2))),
                w_star=float(rng.uniform(1.2, 2.8)), drift=wind)
        for _ in range(n)
    )
    return XCTask(
        name=f"XC {distance / 1000:g} km #{instance}",
        start=(0.0, 0.0, float(rng.uniform(250.0, 350.0))),
        heading_deg=90.0,
        goal=(distance, 0.0),
        wind=wind,
        thermals=thermals,
        time_limit=distance / 3.5,
    )


# -- polar helpers --------------------------------------------------------------------------
def stall_speed(ac: Aircraft, cl_fraction: float = 0.9, bank: float = 0.0) -> float:
    return float(np.sqrt(2.0 * ac.weight / (ac.rho * ac.wing_area * cl_fraction * ac.cl_max * np.cos(bank))))


def speed_to_fly(ac: Aircraft, macready: float, air_vertical: float = 0.0) -> float:
    """MacCready speed-to-fly: the airspeed minimizing ``(sink(V) - w + M) / V`` (time per metre, climbs included)."""
    speeds = np.linspace(1.15 * stall_speed(ac), 22.0, 300)
    return float(speeds[np.argmin((ac.nominal_sink_rate(speeds) - air_vertical + macready) / speeds)])


def expected_climb(ac: Aircraft, thermal: Thermal, z: float, orbit_speed: float, bank_deg: float = 30.0) -> float:
    """Rough climb rate circling in ``thermal`` at altitude ``z``: mean core updraft minus the circling sink."""
    return float(0.6 * thermal.peak_updraft(z) - ac.nominal_sink_rate(orbit_speed, np.radians(bank_deg)))


# -- low-level flight --------------------------------------------------------------------------
@dataclass(frozen=True)
class Autopilot:
    """The loops between a guidance command and the airframe (as the reactive law and the tracker)."""

    attitude_time_constant: float = 0.3  # s
    bank_rate_max_deg: float = 60.0
    bank_max_deg: float = 40.0
    cl_max_fraction: float = 0.9  # angle-of-attack protection
    k_vertical: float = 1.0  # 1/s, vertical speed loop
    k_energy: float = 0.5  # vertical speed traded per m/s of airspeed error
    k_speed: float = 0.3  # 1/s, along-track acceleration per m/s of airspeed error, on the motor
    motor_climb_max: float = 1.5  # m/s
    glide_turn_rate_max: float = 0.25  # rad/s between thermals


@dataclass
class Guidance:
    """What the pilot asks for at one instant."""

    course_rate: float  # rad/s, ground-track turn rate
    v_ref: float  # m/s, airspeed to hold
    motor: bool = False  # climb on the motor toward ``z_target``
    z_target: float = 0.0


def _flat_commands(ac, t, p, v, w, guidance: Guidance, ap: Autopilot, task: XCTask):
    """Guidance -> lift coefficient, bank and thrust through the flat map (as :func:`simulate_law`)."""
    v_air = v - w
    airspeed = float(np.linalg.norm(v_air))
    ground_speed = max(float(np.hypot(v[0], v[1])), 1.0)
    sink = float(ac.nominal_sink_rate(guidance.v_ref))
    vz_cmd = w[2] - sink + ap.k_energy * (airspeed - guidance.v_ref)
    if guidance.motor:
        vz_cmd = max(vz_cmd, float(np.clip(0.2 * (guidance.z_target - p[2]), 0.0, ap.motor_climb_max)))
    track = v[:2] / ground_speed
    a_cmd = np.array([-ground_speed * guidance.course_rate * track[1], ground_speed * guidance.course_rate * track[0],
                      ap.k_vertical * (vz_cmd - v[2])])
    if guidance.motor:  # the motor holds the airspeed; gliding, the airspeed is traded with height
        a_cmd += ap.k_speed * (guidance.v_ref - airspeed) * v_air / max(airspeed, 1.0)
    c = flat_state(ac, t, p, v, a_cmd, w)
    thrust = float(np.clip(c.thrust_required, 0.0, task.thrust_max)) if guidance.motor else 0.0
    return float(c.cl), float(c.bank), thrust, airspeed


@dataclass
class XCFlight:
    state: FlatState  # (1, N) samples, the whole flight
    arrived: bool
    duration: float  # s
    battery: float  # J
    stored: float  # J, weight x change of specific energy
    cost: float  # see the module docstring
    speed_kmh: float  # cross-country speed (distance made good toward the goal / time)
    violated: list[str]
    stalled: bool
    modes: np.ndarray  # per sample: 0 glide, 1 climb, 2 motor, 3 final glide
    targets: np.ndarray  # per sample: index of the thermal aimed at (-1 = the goal)
    climbs: list[dict]  # one entry per climb: thermal, start/end time and altitude


def reference_energy(task: XCTask) -> float:
    """Battery energy of flying the task's distance on the motor at the best lift-to-drag speed."""
    ac = task.aircraft
    return ac.weight / ac.max_glide_ratio * task.distance / EFFICIENCY


def score_flight(task: XCTask, t, p, v, a, modes, targets, climbs, arrived) -> XCFlight:
    ac = task.aircraft
    air = task.air
    w = air(p, t)
    s = flat_state(ac, t[None], p[None], v[None], a[None], w[None])
    scenario = SoaringScenario(
        name=task.name, aircraft=ac, wind=air, start=task.start, heading_deg=task.heading_deg,
        airspeed=task.airspeed, horizon=float(t[-1]), mission=GoalMission(task.goal, task.goal_radius),
        thrust_max=task.thrust_max, strip_wing=task.strip_wing, z_min=30.0,
    )
    problem = SoaringProblem(scenario)
    ev = problem.score(s)
    hinges = problem.hinges(s)
    violated = [n for n in SoaringProblem.CONSTRAINTS if n != "mission" and np.max(hinges[n]) > 1e-3]
    stalled = bool(np.max(s.cl) > ac.cl_max)
    if task.strip_wing is not None:
        loads = turn_loads(task.strip_wing, ac, s.airspeed, s.bank, s.cl)
        stalled |= bool(np.max(loads.cl_max_local) > task.strip_wing.local_cl_max)
    battery, stored = float(ev.propulsive_energy[0]), float(ev.stored_energy[0])
    goal = np.asarray(task.goal)
    made_good = np.linalg.norm(goal - np.asarray(task.start[:2])) - np.linalg.norm(goal - p[-1, :2])
    return XCFlight(
        state=s, arrived=arrived, duration=float(t[-1]), battery=battery, stored=stored,
        cost=(battery - stored / EFFICIENCY) / reference_energy(task),
        speed_kmh=3.6 * float(made_good) / float(t[-1]),
        violated=violated, stalled=stalled, modes=np.asarray(modes), targets=np.asarray(targets), climbs=climbs,
    )


# -- the MacCready pilot ------------------------------------------------------------------------
@dataclass
class MacCreadyPilot:
    """Rule-based cross-country pilot: thermal choice, MacCready speed-to-fly, circling, motor below the floor."""

    macready: float = 1.0  # m/s, the climb rate worth stopping for
    orbit_radius: float = 30.0  # m
    orbit_speed_factor: float = 1.25  # x min-drag speed: the faster orbit that keeps the inner wing flying
    k_orbit: float = 2.0
    k_course: float = 0.8
    final_glide_margin: float = 40.0  # m above the floor on arrival
    arrival_margin: float = 60.0  # m above the floor on reaching a thermal
    climb_window: float = 20.0  # s, climb rate averaged over it to decide to leave
    capture_factor: float = 1.5  # enter the climb within this many core radii of the centre

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.mode, self.target, self.visited, self.climb_start = "glide", None, set(), None
        self._checked = -1e9
        self.history = []  # (t, z) while climbing
        self.motor_on = False

    # Where the pilot believes a thermal is (the forecast); a subclass can make it wrong.
    def thermal_center(self, task: XCTask, i: int, t: float) -> np.ndarray:
        th = task.thermals[i]
        return np.asarray(th.center) + np.asarray(th.drift) * t

    def _glide_ratio(self, task, v_air):
        ac = task.aircraft
        return v_air / ac.nominal_sink_rate(v_air)

    def choose_target(self, task: XCTask, t: float, p: np.ndarray):
        """Next target: the goal if a final glide is safe, else the most useful reachable thermal."""
        ac = task.aircraft
        goal = np.asarray(task.goal)
        to_goal = float(np.linalg.norm(goal - p[:2]))
        v_fg = speed_to_fly(ac, 0.0)
        height = p[2] - task.z_floor - self.final_glide_margin
        wind_along = np.dot(task.wind, (goal - p[:2]) / max(to_goal, 1.0))
        glide = self._glide_ratio(task, v_fg) * (v_fg + wind_along) / v_fg
        if to_goal < glide * height:
            return -1
        # Thermals reachable with a margin, ranked by the rate of progress toward the goal, counting
        # the time to climb back the height the glide there costs (MacCready's reasoning).
        v_stf = speed_to_fly(ac, self.macready)
        ld = self._glide_ratio(task, v_stf)
        orbit = self.orbit_speed_factor * ac.min_drag_airspeed
        best, best_rate, rescue, rescue_d = None, 0.0, None, np.inf
        for i, th in enumerate(task.thermals):
            if i in self.visited:
                continue
            c = self.thermal_center(task, i, t)
            d = float(np.linalg.norm(c - p[:2]))
            z_arrive = p[2] - d / ld
            if z_arrive < task.z_floor + self.arrival_margin:
                continue
            climb = expected_climb(ac, th, z_arrive, orbit)
            if climb < 0.5 * self.macready:
                continue
            if d < rescue_d:
                rescue, rescue_d = i, d
            progress = to_goal - float(np.linalg.norm(goal - c))
            rate = progress / (d / v_stf + (d / ld) / climb)
            if rate > best_rate:
                best, best_rate = i, rate
        if best is None:
            best = rescue  # nothing ahead: the nearest usable thermal, to stay up
        return -1 if best is None else best

    def guidance(self, task: XCTask, t: float, p: np.ndarray, v: np.ndarray, w: np.ndarray) -> Guidance:
        ac = task.aircraft
        if self.target is None:
            self.target = self.choose_target(task, t, p)
        # Motor below the floor (hysteresis), whatever else is going on.
        if p[2] < task.z_floor:
            self.motor_on = True
        elif p[2] > task.z_floor + 60.0:
            self.motor_on = False

        goal = np.asarray(task.goal)
        aim = goal if self.target == -1 else self.thermal_center(task, self.target, t)
        rel = p[:2] - aim
        d = float(np.linalg.norm(rel))
        ground_speed = max(float(np.hypot(v[0], v[1])), 1.0)
        course = np.arctan2(v[1], v[0])

        if self.mode != "climb" and self.target != -1:
            core = task.thermals[self.target].outer_radius(p[2]) / 2.0
            if d < self.capture_factor * max(core, self.orbit_radius):
                self.mode, self.climb_start, self.history = "climb", (t, float(p[2])), []

        if self.mode == "climb":
            self.history.append((t, float(p[2])))
            while self.history and self.history[0][0] < t - self.climb_window:
                self.history.pop(0)
            rate = (self.history[-1][1] - self.history[0][1]) / max(self.history[-1][0] - self.history[0][0], 1e-6)
            settled = t - self.climb_start[0] > self.climb_window
            if p[2] >= task.z_ceiling or (settled and rate < self.macready) or t > task.time_limit - 60.0:
                self.climbs.append({"thermal": self.target, "t0": self.climb_start[0], "z0": self.climb_start[1],
                                    "t1": t, "z1": float(p[2])})
                self.visited.add(self.target)
                self.mode, self.target = "glide", self.choose_target(task, t, p)
                return self.guidance(task, t, p, v, w)
            lam = 1.0  # counter-clockwise
            theta = np.arctan2(rel[1], rel[0])
            course_cmd = theta + lam * (np.pi / 2 + np.arctan(self.k_orbit * (d - self.orbit_radius) / self.orbit_radius))
            rate_cmd = self.k_course * _wrap(course_cmd - course) + lam * ground_speed / self.orbit_radius
            return Guidance(rate_cmd, self.orbit_speed_factor * ac.min_drag_airspeed, self.motor_on, task.z_floor + 80.0)

        if self.target != -1 and t - getattr(self, "_checked", -1e9) > 10.0:  # final glide now safe?
            self._checked = t
            if self.choose_target(task, t, p) == -1:
                self.target, aim = -1, goal
        bearing = np.arctan2(aim[1] - p[1], aim[0] - p[0])
        rate_cmd = float(np.clip(self.k_course * _wrap(bearing - course), -0.25, 0.25))  # gentle turns between thermals
        v_ref = speed_to_fly(ac, 0.0 if self.target == -1 else self.macready, float(w[2]))
        if self.motor_on:
            v_ref = ac.min_drag_airspeed
        return Guidance(rate_cmd, v_ref, self.motor_on, task.z_floor + 80.0)

    @property
    def mode_code(self) -> int:
        if self.motor_on:
            return 2
        return {"glide": 3 if self.target == -1 else 0, "climb": 1}[self.mode]


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def fly_xc(task: XCTask, pilot: MacCreadyPilot, dt: float = 0.1, autopilot: Autopilot = Autopilot()) -> XCFlight:
    """Fly ``task`` with ``pilot`` until the goal or the time limit (point mass, RK4, attitude loops)."""
    ac, air = task.aircraft, task.air
    pilot.reset()
    pilot.climbs = []
    h = np.radians(task.heading_deg)
    p = np.asarray(task.start, dtype=float)
    v = task.airspeed * np.array([np.sin(h), np.cos(h), 0.0]) + air(p, 0.0)
    cl = 2.0 * ac.weight / (ac.rho * ac.wing_area * task.airspeed**2)
    bank = 0.0
    alpha = dt / (autopilot.attitude_time_constant + dt)
    bank_max, bank_rate = np.radians(autopilot.bank_max_deg), np.radians(autopilot.bank_rate_max_deg) * dt
    ts, ps, vs, accs, modes, targets = [], [], [], [], [], []
    goal = np.asarray(task.goal)
    arrived = False
    t, a, thrust = 0.0, np.zeros(3), 0.0
    while t <= task.time_limit + 1e-9:
        if isinstance(pilot, PlannedClimbPilot) and pilot.mode == "climb":
            # The climb is handed to the planner, one block at a time.
            z0 = float(p[2])
            block = pilot.fly_block(task, t, p, v, a, cl, bank, thrust)
            s = block.state
            # The tactician leaves at the cloud base: the block stops where it reaches it.
            above = np.nonzero(s.position[0, :, 2] >= task.z_ceiling)[0]
            n = int(above[0]) + 1 if above.size else s.t.shape[1]
            n = max(n, 2)
            ts.extend(s.t[0, 1:n] + t), ps.extend(s.position[0, 1:n]), vs.extend(s.velocity[0, 1:n])
            accs.extend(s.acceleration[0, 1:n])
            modes.extend([1] * (n - 1)), targets.extend([pilot.target] * (n - 1))
            p, v, a = s.position[0, n - 1], s.velocity[0, n - 1], s.acceleration[0, n - 1]
            cl, bank = float(s.cl[0, n - 1]), float(s.bank[0, n - 1])
            thrust = float(np.clip(s.thrust_required[0, n - 1], 0.0, task.thrust_max))
            t += float(s.t[0, n - 1])
            pilot.after_block(task, t, p, z0, float(s.t[0, n - 1]))
            if np.linalg.norm(p[:2] - goal) < task.goal_radius:
                arrived = True
                break
            if p[2] < 5.0:
                break
            continue
        w = air(p, t)
        g = pilot.guidance(task, t, p, v, w)
        c_cl, c_bank, thrust, airspeed = _flat_commands(ac, t, p, v, w, g, autopilot, task)
        cl += alpha * (np.clip(c_cl, 0.0, autopilot.cl_max_fraction * ac.cl_max) - cl)
        bank += np.clip(alpha * (np.clip(c_bank, -bank_max, bank_max) - bank), -bank_rate, bank_rate)
        if task.strip_wing is not None:
            thrust -= float(turn_loads(task.strip_wing, ac, airspeed, bank, cl).extra_drag)

        def accel(pos, vel, time, cl=cl, bank=bank, thrust=thrust):
            return aerodynamic_acceleration(ac, vel - air(pos, time), cl, bank, thrust)

        p_next, v_next, a = rk4_step(accel, p, v, t, dt)
        ts.append(t), ps.append(p), vs.append(v), accs.append(a)
        modes.append(pilot.mode_code), targets.append(-1 if pilot.target is None else pilot.target)
        if np.linalg.norm(p[:2] - goal) < task.goal_radius:
            arrived = True
            break
        if p[2] < 5.0:
            break
        p, v = p_next, v_next
        t += dt
    if pilot.mode == "climb":
        pilot.climbs.append({"thermal": pilot.target, "t0": pilot.climb_start[0], "z0": pilot.climb_start[1],
                             "t1": ts[-1], "z1": float(ps[-1][2])})
    return score_flight(task, np.array(ts), np.array(ps), np.array(vs), np.array(accs), modes, targets,
                        pilot.climbs, arrived)


@dataclass
class MotorPilot(MacCreadyPilot):
    """Baseline: straight to the goal on the motor, holding the start altitude at the best lift-to-drag speed."""

    def guidance(self, task, t, p, v, w):
        self.target, self.motor_on, self.mode = -1, True, "glide"
        goal = np.asarray(task.goal)
        bearing = np.arctan2(goal[1] - p[1], goal[0] - p[0])
        rate = self.k_course * _wrap(bearing - np.arctan2(v[1], v[0]))
        return Guidance(rate, task.aircraft.min_drag_airspeed, True, task.start[2])

    @property
    def mode_code(self) -> int:
        return 2


@dataclass
class PlannedClimbPilot(MacCreadyPilot):
    """The MacCready tactician, with every climb flown by a phase 3-4 planner.

    A climb is flown in blocks of ``block`` seconds. Each block is one S1-like
    problem -- the thermal being climbed and the wind, from the state reached --
    flown by :func:`geneticwing.closedloop.fly_closed_loop` (planned once, or
    replanned every ``replan_every`` seconds); after each block the tactician
    decides to stay (the block climbed at least ``macready``) or to leave.
    """

    planner: object = None
    block: float = 60.0
    replan_every: float | None = None
    seed: int = 0
    motor_in_climbs: bool = True  # False: the climbs are pure soaring (give the planner a motorless simulator too)

    def reset(self):
        super().reset()
        self.rng = np.random.default_rng(self.seed)

    def fly_block(self, task, t, p, v, a, cl, bank, thrust):
        from geneticwing.closedloop import LoopConfig, Shifted, exact_belief, fly_closed_loop

        ac = task.aircraft
        th = task.thermals[self.target]
        center = tuple(float(x) for x in self.thermal_center(task, self.target, t))
        wind = UniformWind((task.wind[0], task.wind[1], 0.0))
        v_air = np.asarray(v) - task.air(np.asarray(p), t)
        situation = SoaringScenario(
            name="XC climb", aircraft=ac, wind=wind + Thermal(center=center, w_star=th.w_star, z_i=th.z_i, drift=th.drift),
            start=tuple(float(x) for x in p), heading_deg=float(np.degrees(np.arctan2(v_air[0], v_air[1])) % 360.0),
            airspeed=float(np.linalg.norm(v_air)), horizon=self.block, mission=LoiterMission(center, 300.0),
            thrust_max=task.thrust_max if self.motor_in_climbs else 0.0, n_control=32, strip_wing=task.strip_wing,
        )
        span = min(self.block, task.time_limit - t)
        loop = LoopConfig(duration=max(span, 1.0), replan_every=self.replan_every)
        if not self.motor_in_climbs:
            loop = replace(loop, tracker=replace(loop.tracker, powered=False))
        return fly_closed_loop(situation, Shifted(task.air, t), exact_belief(situation), self.planner, loop, self.rng,
                               initial=(v, a, cl, bank, thrust))

    def after_block(self, task, t, p, z0, duration):
        rate = (float(p[2]) - z0) / max(duration, 1e-6)
        if p[2] >= task.z_ceiling or rate < self.macready or t > task.time_limit - 60.0:
            self.climbs.append({"thermal": self.target, "t0": self.climb_start[0], "z0": self.climb_start[1],
                                "t1": t, "z1": float(p[2])})
            self.visited.add(self.target)
            self.mode, self.target = "glide", self.choose_target(task, t, p)
