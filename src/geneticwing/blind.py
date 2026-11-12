"""Blind cross-country pilots: no thermal map, only a variometer and an EKF on the thermal.

The pilots of :mod:`geneticwing.crosscountry` know every thermal's centre and
strength. Here the pilot knows the goal, the wind and its own state, and reads
a variometer: the vertical air velocity where it is (netto), plus white noise.
It cruises toward the goal; when the filtered netto passes a threshold it
starts circling, and an extended Kalman filter estimates the thermal as a
Gaussian updraft ``W exp(-r^2 / R^2)`` drifting with the wind (as ArduSoar,
Tabor et al., IROS 2018). The circling rule orbits the estimated centre; a
planned climb plans in the estimate and replans every ``replan_every`` seconds
as the estimate improves. It leaves by the MacCready rule, or when the
estimated thermal is too weak; a thermal it left is not re-entered for a while.

The background air between thermals (sink) is estimated by a slow average of
netto while cruising, and the EKF fits the lift above it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geneticwing.closedloop import Belief, LoopConfig, Shifted, fly_closed_loop
from geneticwing.crosscountry import Guidance, MacCreadyPilot, PlannedClimbPilot, XCTask, _wrap, speed_to_fly
from geneticwing.soaring import LoiterMission, SoaringScenario
from geneticwing.wind.base import UniformWind
from geneticwing.wind.thermal import Thermal

PLANNER_Z_I = 1000.0  # the command planners' canonical thermal depth (their context has no z_i)


@dataclass
class ThermalEKF:
    """EKF on ``[W, R, x, y]``: peak lift above the background, radius, centre in the air-mass frame."""

    meas_sigma: float = 0.4  # m/s: vario noise and the Gaussian model's error on a real thermal
    q: tuple[float, float, float, float] = (0.02, 0.5, 0.8, 0.8)  # process noise, std per sqrt(s)
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))
    P: np.ndarray = field(default_factory=lambda: np.eye(4))

    def start(self, xy: np.ndarray, lift: float) -> None:
        self.x = np.array([max(1.5 * lift, 0.8), 40.0, xy[0], xy[1]])
        self.P = np.diag([1.5**2, 20.0**2, 40.0**2, 40.0**2])

    def update(self, xy: np.ndarray, lift: float, dt: float) -> None:
        self.P = self.P + np.diag(np.square(self.q)) * max(dt, 0.0)
        W, R, cx, cy = self.x
        dx, dy = xy[0] - cx, xy[1] - cy
        d2 = dx * dx + dy * dy
        e = np.exp(-d2 / R**2)
        h = W * e
        H = np.array([e, 2.0 * W * e * d2 / R**3, 2.0 * W * e * dx / R**2, 2.0 * W * e * dy / R**2])
        S = H @ self.P @ H + self.meas_sigma**2
        K = self.P @ H / S
        self.x = self.x + K * (lift - h)
        self.P = (np.eye(4) - np.outer(K, H)) @ self.P
        self.x[0] = np.clip(self.x[0], 0.0, 8.0)
        self.x[1] = np.clip(self.x[1], 15.0, 150.0)

    @property
    def center(self) -> np.ndarray:
        return self.x[2:4]

    @property
    def strength(self) -> float:
        return float(self.x[0])


@dataclass
class BlindMixin:
    """The blind tactician and sensing, shared by the circling rule and the planned climbs."""

    vario_sigma: float = 0.2  # m/s, white noise on netto
    vario_tau: float = 2.0  # s, filter of the detection
    background_tau: float = 60.0  # s, average of netto while cruising: the sink between thermals
    detect_base: float = 0.2  # m/s; threshold = detect_base + detect_factor x MacCready
    detect_factor: float = 0.5
    ekf_period: float = 0.5  # s between EKF updates
    weak: float = 0.3  # m/s: an estimated peak lift below this after `give_up` s ends the climb
    give_up: float = 30.0
    cooldown: float = 300.0  # s during which a thermal left is not re-entered...
    cooldown_radius: float = 250.0  # ...within this distance (air-mass frame)
    vario_seed: int = 0

    # -- sensing ---------------------------------------------------------------------------------------
    def reset(self):
        super().reset()
        self.vrng = np.random.default_rng(self.vario_seed)
        self.ekf = ThermalEKF()
        self.vario_f, self.background, self._last_t, self._last_ekf = 0.0, 0.0, None, -1e9
        self.left: list[tuple[np.ndarray, float]] = []
        self.final = False

    def air_xy(self, task: XCTask, p: np.ndarray, t: float) -> np.ndarray:
        return np.asarray(p[:2], float) - np.asarray(task.wind, float) * t

    def read_vario(self, w_true: float) -> float:
        return float(w_true + self.vario_sigma * self.vrng.standard_normal())

    def ekf_step(self, task, t, p, netto):
        if t - self._last_ekf >= self.ekf_period - 1e-9:
            self.ekf.update(self.air_xy(task, p, t), netto - self.background, t - self._last_ekf)
            self._last_ekf = t

    def planner_z_i(self, task: XCTask) -> float:
        """The layer depth the planner's thermal model uses: the forecast's (the day's median), if it takes one."""
        if (getattr(self.planner, "use_z_i", False) or getattr(self.planner, "weather", False)) and task.thermals:
            return float(np.median([th.z_i for th in task.thermals]))
        return PLANNER_Z_I

    def belief_thermal(self, task: XCTask, t: float, z: float) -> tuple[np.ndarray, float]:
        """Estimated centre now (world frame) and the w* of a planner thermal with the estimated peak at ``z``."""
        center = self.ekf.center + np.asarray(task.wind, float) * t
        z_i = self.planner_z_i(task)
        ratio = np.clip(z / z_i, 0.05, 0.85)
        if getattr(self.planner, "weather", False):  # Allen's profile: peak = w* x allen_peak(z, 1, z_i)
            from geneticwing.wind.thermal import allen_peak

            shape = float(allen_peak(ratio * z_i, 1.0, z_i))
        else:
            shape = np.cbrt(ratio) * (1.0 - 1.1 * ratio) / ((1.0 - np.exp(-4.0)) / 4.0)
        return center, max(self.ekf.strength + self.background, 0.2) / shape

    # -- the tactician -------------------------------------------------------------------------------
    def start_climb(self, task, t, p, v, netto):
        v_air_dir = np.asarray(v[:2], float) / max(float(np.hypot(v[0], v[1])), 1.0)
        self.ekf.start(self.air_xy(task, p, t) + 10.0 * v_air_dir, netto - self.background)
        self._last_ekf = t
        self.mode, self.target, self.climb_start, self.history = "climb", -2, (t, float(p[2])), []

    def end_climb(self, task, t, p):
        self.climbs.append({"thermal": -2, "t0": self.climb_start[0], "z0": self.climb_start[1], "t1": t,
                            "z1": float(p[2])})
        self.left.append((self.ekf.center.copy(), t))
        self.mode, self.target = "glide", -1

    def near_left(self, task, t, p) -> bool:
        xy = self.air_xy(task, p, t)
        return any(t - tl < self.cooldown and np.linalg.norm(xy - c) < self.cooldown_radius for c, tl in self.left)

    def guidance(self, task: XCTask, t: float, p: np.ndarray, v: np.ndarray, w: np.ndarray) -> Guidance:
        ac = task.aircraft
        # (after a planned block the last call is a minute old: the filters only take one step)
        dt = 0.0 if self._last_t is None else min(t - self._last_t, 0.5)
        self._last_t = t
        netto = self.read_vario(float(w[2]))
        self.vario_f += min(dt / self.vario_tau, 1.0) * (netto - self.vario_f)
        if self.target is None:
            self.target = -1
        if p[2] < task.z_floor:
            self.motor_on = True
        elif p[2] > task.z_floor + 60.0:
            self.motor_on = False
        ground_speed = max(float(np.hypot(v[0], v[1])), 1.0)
        course = np.arctan2(v[1], v[0])

        if self.mode == "climb":
            self.ekf_step(task, t, p, netto)
            self.history.append((t, float(p[2])))
            while self.history and self.history[0][0] < t - self.climb_window:
                self.history.pop(0)
            rate = (self.history[-1][1] - self.history[0][1]) / max(self.history[-1][0] - self.history[0][0], 1e-6)
            age = t - self.climb_start[0]
            if (p[2] >= task.z_ceiling or (age > self.climb_window and rate < self.macready)
                    or (age > self.give_up and self.ekf.strength < self.weak) or t > task.time_limit - 60.0
                    or self.motor_on):
                self.end_climb(task, t, p)
            else:
                aim = self.ekf.center + np.asarray(task.wind, float) * t
                rel = p[:2] - aim
                d = float(np.linalg.norm(rel))
                theta = np.arctan2(rel[1], rel[0])
                course_cmd = theta + (np.pi / 2 + np.arctan(self.k_orbit * (d - self.orbit_radius) / self.orbit_radius))
                rate_cmd = self.k_course * _wrap(course_cmd - course) + ground_speed / self.orbit_radius
                return Guidance(rate_cmd, self.orbit_speed_factor * ac.min_drag_airspeed, False, task.z_floor + 80.0)

        # Cruise toward the goal; the background sink is learnt on the way.
        self.background += min(dt / self.background_tau, 1.0) * (float(np.clip(netto, -1.5, 0.5)) - self.background)
        if t - self._checked > 10.0:
            self._checked = t
            self.final = self.final_glide_ok(task, p)
        threshold = self.detect_base + self.detect_factor * self.macready
        if (not self.motor_on and not self.final and p[2] < task.z_ceiling - 30.0
                and self.vario_f - self.background > threshold and not self.near_left(task, t, p)):
            self.start_climb(task, t, p, v, self.vario_f)
            return self.guidance(task, t, p, v, w)
        goal = np.asarray(task.goal)
        bearing = np.arctan2(goal[1] - p[1], goal[0] - p[0])
        rate_cmd = float(np.clip(self.k_course * _wrap(bearing - course), -0.25, 0.25))
        v_ref = speed_to_fly(ac, 0.0 if self.final else self.macready, netto)
        if self.motor_on:
            v_ref = self.motor_airspeed(task, p, goal)
        return Guidance(rate_cmd, v_ref, self.motor_on, task.z_floor + 80.0)

    @property
    def mode_code(self) -> int:
        if self.motor_on:
            return 2
        if self.mode == "climb":
            return 1
        return 3 if self.final else 0


@dataclass
class BlindRulePilot(BlindMixin, MacCreadyPilot):
    """The circling rule around the EKF's centre."""


@dataclass
class BlindPlannedPilot(BlindMixin, PlannedClimbPilot):
    """Planned climbs in the EKF's belief, replanned every ``replan_every`` s as the estimate improves."""

    replan_every: float | None = 10.0

    def fly_block(self, task, t, p, v, a, cl, bank, thrust):
        ac = task.aircraft
        wind = np.asarray(task.wind, float)
        center, w_star = self.belief_thermal(task, t, float(p[2]))
        v_air = np.asarray(v) - task.air(np.asarray(p), t)
        center_t = tuple(float(x) for x in center)
        situation = SoaringScenario(
            name="XC climb (blind)", aircraft=ac,
            wind=UniformWind((wind[0], wind[1], 0.0)) + Thermal(center=center_t, w_star=w_star, z_i=self.planner_z_i(task),
                                                                drift=tuple(wind)),
            start=tuple(float(x) for x in p), heading_deg=float(np.degrees(np.arctan2(v_air[0], v_air[1])) % 360.0),
            airspeed=float(np.linalg.norm(v_air)), horizon=self.block, mission=LoiterMission(center_t, 300.0),
            thrust_max=0.0, n_control=32, strip_wing=task.strip_wing,
        )
        span = min(self.block, task.time_limit - t)
        loop = LoopConfig(duration=max(span, 1.0), replan_every=self.replan_every,
                          inner_wing_protection=self.inner_wing_protection)
        from dataclasses import replace

        loop = replace(loop, tracker=replace(loop.tracker, powered=False))
        air = task.air

        def on_segment(t0, seg):
            last = None
            for j in range(len(seg.t)):
                ts = t + t0 + float(seg.t[j])
                if last is not None and ts - last < self.ekf_period - 1e-9:
                    continue
                last = ts
                pos = seg.position[j]
                netto = self.read_vario(float(air(pos, ts)[2]))
                self.ekf.update(self.air_xy(task, pos, ts), netto - self.background, self.ekf_period)
            t_end = t + t0 + float(seg.t[-1])
            c_now, ws = self.belief_thermal(task, t_end, float(seg.position[-1][2]))
            # Belief centre at the block's t = 0 (the loop drifts it with the wind from there).
            c0 = c_now - wind * (t0 + float(seg.t[-1]))
            return Belief((float(c0[0]), float(c0[1])), float(ws), (float(wind[0]), float(wind[1])))

        belief = Belief(center_t, float(w_star), (float(wind[0]), float(wind[1])))
        return fly_closed_loop(situation, Shifted(air, t), belief, self.planner, loop, self.rng,
                               initial=(v, a, cl, bank, thrust), on_segment=on_segment)

    def after_block(self, task, t, p, z0, duration):
        rate = (float(p[2]) - z0) / max(duration, 1e-6)
        age = t - self.climb_start[0]
        if (p[2] >= task.z_ceiling or rate < self.macready or t > task.time_limit - 60.0
                or (age > self.give_up and self.ekf.strength < self.weak) or p[2] < task.z_floor):
            self.end_climb(task, t, p)
