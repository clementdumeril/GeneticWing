"""Nonlinear MPC baseline: the climb as an optimal-control problem, solved by IPOPT.

The standard model-based alternative to a learned planner (e.g. Liu, Longo
and Kerrigan's NMPC for soaring UAVs), written for exactly the problem the
command planners of :mod:`geneticwing.closedloop` solve, so the comparison
isolates the solver:

* same decision variables: C_L, bank and thrust at the ``cfg.knots`` knots,
  linearly interpolated, in the same bounds (:class:`~geneticwing.torchsim.SimConfig`);
* same dynamics: the point mass of :mod:`geneticwing.torchsim`, RK4 at
  ``cfg.dt``, transcribed by multiple shooting on every integration step, so
  the NMPC's prediction is the flight :func:`~geneticwing.physvae.fly` scores;
* same objective and constraints: the net onboard energy of
  :meth:`~geneticwing.soaring.SoaringProblem.score` (aileron drag included)
  and every hinge of the problem as a hard path constraint, the strip wing's
  section stall and aileron authority written in closed form.

IPOPT (through CasADi, exact derivatives) needs a starting guess, and a
soaring problem has many local optima (orbit direction, where to circle), so
every plan is a multi-start from the tuned reactive orbit law's flights (both
directions, both orbit speeds: the offline teacher's seeds), plus the rest of
the plan being flown when replanning in flight (warm start). Each
solution is re-flown by the NumPy simulator and the best by Deb's rule wins.
"""

from __future__ import annotations

import time
from dataclasses import replace

import casadi as ca
import numpy as np

from geneticwing.benchmarks import with_margins
from geneticwing.closedloop import Plan, Start, _CommandPlanner, _directions
from geneticwing.policies import law_seeds, seed_theta, simulate_law
from geneticwing.physvae import fly, from_unit_commands, to_unit_commands
from geneticwing.soaring import LoiterMission, SoaringProblem
from geneticwing.torchsim import SimConfig, simulate_flights
from geneticwing.wind.base import UniformWind
from geneticwing.wind.thermal import _GAUSSIAN_DISK_MEAN, Thermal

_TIGHT = 1e-3  # every constraint is solved this far inside, so the re-flown plan scores as feasible


def _knot_weights(t: float, cfg: SimConfig) -> tuple[int, float]:
    spacing = cfg.horizon / (cfg.knots - 1)
    j = min(int(t / spacing), cfg.knots - 2)
    return j, (t - j * spacing) / spacing


class _Transcription:
    """The NLP for one number of integration steps; the situation enters as parameters."""

    # parameters: thermal centre (2), w*, z_i, sink, wind (2), initial p (3), v (3), first knot (3), fix first, mission centre (2)
    N_PARAM = 19

    def __init__(self, cfg: SimConfig, problem: SoaringProblem, n_steps: int, max_iter: int, max_seconds: float,
                 stride: int = 5):
        sc, ac = problem.scenario, problem.scenario.aircraft
        self.cfg, self.n = cfg, n_steps
        cl_lim, bank_lim, thrust_lim = cfg.safety * cfg.cl_limit * ac.cl_max, cfg.safety * np.radians(cfg.bank_max_deg), cfg.thrust_max
        K, dt = cfg.knots, cfg.dt
        U = ca.SX.sym("u", K, 3)  # unit box, as the command planners
        # Multiple shooting: a node every `stride` RK4 steps, single shooting in between.
        self.nodes = list(range(0, n_steps, stride)) + [n_steps]
        X = ca.SX.sym("x", len(self.nodes), 6)
        P = ca.SX.sym("p", self.N_PARAM)
        center, w_star, z_i, sink, wind = P[0:2], P[2], P[3], P[4], P[5:7]
        x0, first, fix_first, mission_c = P[7:13], P[13:16], P[16], P[17:19]

        def cmd(t):
            j, w = _knot_weights(t, cfg)
            u = (1.0 - w) * U[j, :] + w * U[j + 1, :]
            return u[0] * cl_lim, (2.0 * u[1] - 1.0) * bank_lim, u[2] * thrust_lim

        def air(p, t):
            ratio = ca.fmax(p[2] / z_i, 1e-6)
            cbrt = ratio ** (1.0 / 3.0)
            mean = w_star * cbrt * (1.0 - 1.1 * ratio)
            core = 0.102 * cbrt * (1.0 - 0.25 * ratio) * z_i / 2.0
            c = center + wind * t
            d2 = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
            return ca.vertcat(wind[0], wind[1], mean / _GAUSSIAN_DISK_MEAN * ca.exp(-d2 / core**2) - sink)

        def frame(v_air):
            speed = ca.sqrt(ca.sumsqr(v_air) + 1e-9)
            v_hat = v_air / speed
            up = ca.DM([0.0, 0.0, 1.0])
            up_perp = up - ca.dot(up, v_hat) * v_hat
            up_perp = up_perp / ca.sqrt(ca.sumsqr(up_perp) + 1e-12)
            return speed, v_hat, up_perp, ca.cross(v_hat, up_perp)

        # The simulator's load limiter caps C_L at every RK4 stage; the NLP keeps every stage below it instead
        # (a smooth constraint), so the limiter never acts and the prediction is the flight.
        n_max = cfg.safety * cfg.load_max
        stage_loads = []

        def accel(p, v, t):
            cl, bank, thrust = cmd(t)
            speed, v_hat, up_perp, right = frame(v - air(p, t))
            q_s = 0.5 * ac.rho * speed**2 * ac.wing_area
            stage_loads.append(q_s * cl / ac.weight)
            lift = q_s * cl * (ca.cos(bank) * up_perp + ca.sin(bank) * right)
            drag = q_s * (ac.cd0 + ac.k * cl**2)
            return (lift + (thrust - drag) * v_hat) / ac.mass - ca.DM([0.0, 0.0, ac.g])

        wing = sc.strip_wing
        if wing is not None:
            y, chord, width, side = (np.asarray(a, dtype=float) for a in wing.stations)

        def strip(speed, bank, cl):
            """Mean-to-peak section C_L, aileron delta and aileron drag of :func:`geneticwing.strip.turn_loads`."""
            tan_b = ca.sqrt(ca.tan(bank) ** 2 + 1e-10)
            curvature = ac.g * tan_b / speed**2
            ratio = 1.0 + ca.DM(y) * (ca.cos(bank) * curvature)
            weight = ca.DM(chord * width) * ratio**2
            a, b = ca.sum1(weight), ca.sum1(weight * ca.DM(side))
            c, d = ca.sum1(weight * ca.DM(y)), ca.sum1(weight * ca.DM(side * y))
            cl_body = cl * ac.wing_area / (a - b * c / d)
            delta = -c * cl_body / d
            extra = 0.5 * ac.rho * speed**2 * wing.aileron_drag * delta**2 * ca.sum1(weight * ca.DM(np.abs(side)))
            return cl_body, delta, extra

        g, lbg, ubg = [], [], []

        def le(expr, bound):  # expr <= bound, backed off a little (the plan is re-flown and must stay feasible)
            g.append(expr), lbg.append(-ca.inf), ubg.append(bound - _TIGHT * max(1.0, abs(bound)))

        g.append(X[0, :].T - x0), lbg.extend([0.0] * 6), ubg.extend([0.0] * 6)
        for i in range(3):  # in flight, the first knot is the current command
            g.append(fix_first * (U[0, i] - first[i])), lbg.append(0.0), ubg.append(0.0)

        def path(k, p, v):
            """Constraints at sample k; returns the propulsive power there."""
            t = k * dt
            cl, bank, thrust = cmd(t)
            speed = ca.sqrt(ca.sumsqr(v - air(p, t)) + 1e-9)
            q_s = 0.5 * ac.rho * speed**2 * ac.wing_area
            le(q_s * cl / ac.weight, min(sc.load_max, n_max))
            le(cl, sc.cl_margin * ac.cl_max)
            le(speed, sc.airspeed_max)
            le(-p[2], -sc.z_min)
            if isinstance(sc.mission, LoiterMission):
                le((p[0] - mission_c[0]) ** 2 + (p[1] - mission_c[1]) ** 2, sc.mission.radius**2)
            extra = 0.0
            if wing is not None:
                cl_body, delta, extra = strip(speed, bank, cl)
                le(cl_body + delta, sc.cl_margin * ac.cl_max)
                le(cl_body - delta, sc.cl_margin * ac.cl_max)
                le(delta, wing.aileron_cl_max)
                le(-delta, wing.aileron_cl_max)
            le(thrust + extra, sc.thrust_max)
            return (thrust + extra) * speed / sc.efficiency

        powers = []
        for i, (k0, k1) in enumerate(zip(self.nodes[:-1], self.nodes[1:])):
            p, v = X[i, 0:3].T, X[i, 3:6].T
            for k in range(k0, k1):  # RK4, as rk4_step
                powers.append(path(k, p, v))
                t = k * dt
                a1 = accel(p, v, t)
                v2 = v + 0.5 * dt * a1
                a2 = accel(p + 0.5 * dt * v, v2, t + 0.5 * dt)
                v3 = v + 0.5 * dt * a2
                a3 = accel(p + 0.5 * dt * v2, v3, t + 0.5 * dt)
                v4 = v + dt * a3
                a4 = accel(p + dt * v3, v4, t + dt)
                p, v = p + dt * v + dt**2 / 6.0 * (a1 + a2 + a3), v + dt / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
            g.append(X[i + 1, :].T - ca.vertcat(p, v)), lbg.extend([0.0] * 6), ubg.extend([0.0] * 6)
        powers.append(path(n_steps, X[-1, 0:3].T, X[-1, 3:6].T))
        for load in stage_loads:
            le(load, n_max)
        propulsive = sum(0.5 * dt * (a + b) for a, b in zip(powers[:-1], powers[1:]))
        e0 = X[0, 2] + (ca.sumsqr(X[0, 3:6].T - air(X[0, 0:3].T, 0.0))) / (2.0 * ac.g)
        e1 = X[-1, 2] + (ca.sumsqr(X[-1, 3:6].T - air(X[-1, 0:3].T, n_steps * dt))) / (2.0 * ac.g)
        stored = ac.weight * (e1 - e0)
        cost = (propulsive - stored / sc.efficiency) / problem.reference_energy
        smooth = 1e-3 * ca.sumsqr(U[1:, :] - U[:-1, :])  # a whisper of regularization against chattering knots
        w = ca.vertcat(ca.vec(U), ca.vec(X))
        nlp = {"x": w, "p": P, "f": cost + smooth, "g": ca.vertcat(*g)}
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": max_iter, "ipopt.max_cpu_time": max_seconds,
                "ipopt.tol": 1e-7, "ipopt.constr_viol_tol": 1e-8, "ipopt.acceptable_tol": 1e-5,
                "ipopt.acceptable_constr_viol_tol": 1e-7, "ipopt.mu_strategy": "adaptive", "ipopt.sb": "yes"}
        self.solver = ca.nlpsol("nmpc", "ipopt", nlp, opts)
        self.lbg, self.ubg = np.array(lbg, dtype=float), np.array(ubg, dtype=float)
        m = len(self.nodes)
        self.lbx = np.concatenate([np.zeros(3 * K), np.full(6 * m, -np.inf)])
        self.ubx = np.concatenate([np.ones(3 * K), np.full(6 * m, np.inf)])
        self.K = K

    def solve(self, params: np.ndarray, u_guess: np.ndarray, x_guess: np.ndarray) -> tuple[np.ndarray, str]:
        """``x_guess``: the guess flown at every step ``(n + 1, 6)``; only its nodes are used."""
        w0 = np.concatenate([u_guess.reshape(self.K, 3).ravel(order="F"), x_guess[self.nodes].ravel(order="F")])
        sol = self.solver(x0=w0, p=params, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
        self.last_w = np.asarray(sol["x"]).ravel()
        u = np.asarray(sol["x"][: 3 * self.K]).reshape(self.K, 3, order="F")
        return np.clip(u, 0.0, 1.0).ravel(), self.solver.stats()["return_status"]


class NMPCPlanner(_CommandPlanner):
    """Multi-start IPOPT on the command knots, seeded by the reactive law; plugs into the closed loop."""

    def __init__(self, cfg: SimConfig, theta: np.ndarray, max_iter: int = 500, max_seconds: float = 30.0,
                 cl_margin: float | None = None):
        super().__init__(None, cfg)
        self.theta = np.asarray(theta, dtype=float)
        self.max_iter, self.max_seconds, self.cl_margin = max_iter, max_seconds, cl_margin
        self._cache: dict = {}
        self.log: list[dict] = []  # per plan: solve time, statuses, which start won

    def _transcription(self, cfg: SimConfig, problem: SoaringProblem, n_steps: int) -> _Transcription:
        sc = problem.scenario
        key = (n_steps, sc.cl_margin, sc.load_max, sc.thrust_max, sc.airspeed_max, sc.z_min, sc.strip_wing is not None,
               type(sc.mission).__name__, getattr(sc.mission, "radius", None), cfg.thrust_max, cfg.cl_limit,
               cfg.bank_max_deg, cfg.load_max)
        if key not in self._cache:
            self._cache[key] = _Transcription(cfg, problem, n_steps, self.max_iter, self.max_seconds)
        return self._cache[key]

    @staticmethod
    def _rest(start: Start | None, cfg: SimConfig) -> np.ndarray | None:
        """The rest of the plan being flown, shifted to now, its first knot the current command (as the others)."""
        if start is None or start.previous is None or start.previous.commands is None:
            return None
        first = to_unit_commands(np.array([[[start.cl, start.bank, start.thrust]]]), cfg)[0]
        knots = start.previous.commands.reshape(cfg.knots, 3)
        shift = min(int(round(start.elapsed / (cfg.horizon / (cfg.knots - 1)))), cfg.knots - 1)
        rest = np.vstack([knots[shift:], np.repeat(knots[-1:], shift, axis=0)]).ravel()
        rest[:3] = first
        return rest[None]

    def guesses(self, cfg: SimConfig, problem: SoaringProblem, start: Start | None = None,
                rest: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Starting points: the law's flights read off as commands at the knots, and the warm start."""
        sc = problem.scenario
        robust = SoaringProblem(with_margins(sc))
        knot_t = np.linspace(0.0, cfg.horizon, cfg.knots)
        out = {}
        for ccw in _directions(start):
            for j, overrides in enumerate(law_seeds(sc.strip_wing is not None)):
                s = simulate_law(robust, seed_theta(self.theta, ccw, overrides)[None]).state
                t = s.t[0]
                cmd = np.stack([np.interp(knot_t, t, x[0]) for x in
                                (s.cl, s.bank, np.clip(s.thrust_required, 0.0, cfg.thrust_max))], axis=-1)
                out[f"law {'ccw' if ccw else 'cw'}{' fast' if j else ''}"] = to_unit_commands(cmd[None], cfg)[0]
        if rest is not None:
            out["warm start"] = rest[0]
        return out

    def plan(self, context: np.ndarray, cfg: SimConfig, problem: SoaringProblem, start: Start | None = None):
        """Multi-start solve in a canonical situation: evaluation of every start, their commands, names, statuses."""
        tic = time.perf_counter()
        rest = self._rest(start, cfg)
        sc = problem.scenario
        n_steps = int(np.searchsorted(cfg.times, sc.horizon + 1e-9)) - 1
        nlp = self._transcription(cfg, problem, n_steps)
        thermal = next(c for c in sc.wind.components if isinstance(c, Thermal))
        wind = np.asarray(next(c for c in sc.wind.components if isinstance(c, UniformWind)).velocity[:2], float)
        p0 = np.array(sc.start, dtype=float)
        v0 = np.array([cfg.initial_airspeed, 0.0, 0.0]) + sc.wind(p0, 0.0)
        first = rest[0, :3] if rest is not None else np.zeros(3)
        params = np.concatenate([thermal.center, [thermal.w_star, thermal.z_i, thermal.environment_sink], wind, p0, v0,
                                 first, [1.0 if rest is not None else 0.0], np.asarray(sc.mission.center, float)])
        guesses = self.guesses(cfg, problem, start, rest)
        candidates, statuses, seconds = [], [], []
        for u in guesses.values():
            tic_start = time.perf_counter()
            u = u.copy()
            if rest is not None:
                u[:3] = first
            _, ps, vs, _ = simulate_flights(from_unit_commands(u[None], cfg), context, cfg)
            x_guess = np.concatenate([ps[0, : n_steps + 1], vs[0, : n_steps + 1]], axis=1)
            sol, status = nlp.solve(params, u, x_guess)
            candidates.append(sol)
            statuses.append(status)
            seconds.append(time.perf_counter() - tic_start)
        candidates = np.array(candidates)
        ev = fly(candidates, context, cfg, problem)
        k = int(np.argmin(ev.fitness))
        self.log.append({"seconds": time.perf_counter() - tic, "start_seconds": seconds, "statuses": statuses,
                         "winner": list(guesses)[k],
                         "feasible": bool(ev.feasible[k]), "cost": float(ev.cost[k])})
        return ev, candidates, list(guesses), statuses

    def __call__(self, scenario, rng: np.random.Generator, start: Start | None = None) -> Plan:
        context, cfg, problem = self._problem(scenario)
        ev, candidates, _, _ = self.plan(context, cfg, problem, start)
        k = int(np.argmin(ev.fitness))
        return self._to_world(ev, k, scenario, candidates[k])
