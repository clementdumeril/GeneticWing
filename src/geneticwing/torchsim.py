"""Differentiable, batched point-mass simulator in PyTorch, for the physics decoder.

The physics decoder of :mod:`geneticwing.physvae` does not output positions:
it outputs commands (lift coefficient, bank, thrust) at a few knots, and this
simulator turns them into a trajectory. The commands are squashed into their
limits, and a load limiter caps the lift coefficient at every step, so C_L,
bank, thrust and load factor cannot break their constraints, whatever the
network outputs: dynamic feasibility holds by construction.

The model is the 3-DOF point mass of :mod:`geneticwing.simulation` (lift,
drag, thrust, gravity, wind; commands applied without attitude lag), and the
wind is the S1 family's (uniform wind plus one drifting thermal with Allen's
scaling), all in the canonical frame of :func:`geneticwing.benchmarks.canonical_scenario`:
a batch of different situations is simulated at once, each described by its
heading-frame context. Integration is RK4 (see
:func:`geneticwing.simulation.rk4_step`: explicit Euler injects energy in
turns, and a search exploits it). :func:`simulate_flights` is the NumPy twin
used by the searches: one situation, many command sequences, no gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from geneticwing.aircraft import Aircraft
from geneticwing.benchmarks import CRUISE_AIRSPEED, MOTOR_THRUST, canonical_scenario, heading_context_raw
from geneticwing.simulation import aerodynamic_acceleration, rk4_step
from geneticwing.wind.thermal import _GAUSSIAN_DISK_MEAN

N_COMMANDS = 3  # lift coefficient, bank, thrust


@dataclass(frozen=True)
class SimConfig:
    aircraft: Aircraft
    horizon: float = 60.0
    dt: float = 0.2
    knots: int = 31  # command knots over the horizon (every 2 s)
    initial_airspeed: float = CRUISE_AIRSPEED
    thrust_max: float = MOTOR_THRUST
    cl_limit: float = 0.9  # fraction of cl_max, as the scenario's constraint
    bank_max_deg: float = 45.0
    load_max: float = 2.5
    safety: float = 0.99  # commands stay just inside the constraints
    z_i: float = 1000.0  # convective layer depth of the family's thermals

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon / self.dt))

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_steps + 1) * self.dt


def commands_from_raw(raw: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    """Unbounded network outputs ``(B, knots, 3)`` -> C_L, bank (rad), thrust (N) within limits."""
    ac = cfg.aircraft
    cl = cfg.safety * cfg.cl_limit * ac.cl_max * torch.sigmoid(raw[..., 0])
    bank = cfg.safety * np.radians(cfg.bank_max_deg) * torch.tanh(raw[..., 1])
    thrust = cfg.thrust_max * torch.sigmoid(raw[..., 2])
    return torch.stack([cl, bank, thrust], dim=-1)


def _thermal_updraft(p, t, center, drift, w_star, z_i):
    z = p[..., 2]
    ratio = torch.clamp(z / z_i, min=1e-6)
    cbrt = ratio ** (1.0 / 3.0)
    mean = torch.clamp(w_star * cbrt * (1.0 - 1.1 * ratio), min=0.0)
    outer = torch.clamp(0.102 * cbrt * (1.0 - 0.25 * ratio) * z_i, min=10.0)
    c = center + drift * t
    d2 = torch.sum((p[..., :2] - c) ** 2, dim=-1)
    w = mean / _GAUSSIAN_DISK_MEAN * torch.exp(-d2 / (outer / 2.0) ** 2)
    return torch.where((z > 0.0) & (z < z_i), w, torch.zeros_like(w))


class CanonicalWorld:
    """Wind of a batch of S1-family situations in their canonical frames."""

    def __init__(self, contexts: np.ndarray, cfg: SimConfig, device="cpu"):
        raw = heading_context_raw(np.atleast_2d(contexts))
        f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        self.center = f(raw[:, 0:2])
        self.z0 = f(raw[:, 2])
        self.w_star = f(raw[:, 3])
        self.wind = f(raw[:, 4:6])
        self.cfg = cfg

    def __call__(self, p: torch.Tensor, t: float) -> torch.Tensor:
        w = torch.zeros_like(p)
        w[..., :2] = self.wind
        w[..., 2] = _thermal_updraft(p, t, self.center, self.wind, self.w_star, self.cfg.z_i)
        return w


def aero_acceleration(v_air, cl, bank, thrust, ac: Aircraft) -> torch.Tensor:
    up = torch.zeros_like(v_air)
    up[..., 2] = 1.0
    speed = torch.clamp(torch.linalg.norm(v_air, dim=-1, keepdim=True), min=1e-3)
    v_hat = v_air / speed
    up_perp = up - torch.sum(up * v_hat, dim=-1, keepdim=True) * v_hat
    up_perp = up_perp / torch.clamp(torch.linalg.norm(up_perp, dim=-1, keepdim=True), min=1e-6)
    right = torch.linalg.cross(v_hat, up_perp, dim=-1)
    q_s = 0.5 * ac.rho * speed**2 * ac.wing_area
    cl, bank, thrust = cl[..., None], bank[..., None], thrust[..., None]
    lift = q_s * cl * (torch.cos(bank) * up_perp + torch.sin(bank) * right)
    drag = q_s * (ac.cd0 + ac.k * cl**2)
    return (lift + (thrust - drag) * v_hat) / ac.mass - ac.g * up


def _knot_weights(t: float, cfg: SimConfig) -> tuple[int, float]:
    """Index of the knot before ``t`` and the linear interpolation weight of the next one."""
    spacing = cfg.horizon / (cfg.knots - 1)
    j = min(int(t / spacing), cfg.knots - 2)
    return j, (t - j * spacing) / spacing


def simulate(commands: torch.Tensor, contexts: np.ndarray, cfg: SimConfig):
    """Fly bounded commands ``(B, knots, 3)`` in each situation's canonical frame (differentiable).

    Returns ``(t, p, v, a)``: times ``(N,)`` and tensors ``(B, N, 3)`` sampled at
    every step, with ``a`` the acceleration at each sample (the flat map of
    ``(p, v, a)`` then reproduces the commands exactly).
    """
    ac, device = cfg.aircraft, commands.device
    world = CanonicalWorld(contexts, cfg, device)
    batch = commands.shape[0]
    p = torch.zeros(batch, 3, device=device)
    p[:, 2] = world.z0
    v = torch.zeros(batch, 3, device=device)
    v[:, 0] = cfg.initial_airspeed
    v = v + world(p, 0.0)
    n_max_cl = cfg.safety * cfg.load_max * ac.weight

    def accel(pos, vel, time):
        j, w = _knot_weights(time, cfg)
        cmd = (1.0 - w) * commands[:, j] + w * commands[:, j + 1]
        v_air = vel - world(pos, time)
        q_s = 0.5 * ac.rho * torch.sum(v_air**2, dim=-1) * ac.wing_area
        cl = torch.minimum(cmd[:, 0], n_max_cl / torch.clamp(q_s, min=1e-3))  # load limiter
        return aero_acceleration(v_air, cl, cmd[:, 1], cmd[:, 2], ac)

    ps, vs, accs = [], [], []
    for k in range(cfg.n_steps + 1):
        p_next, v_next, a = rk4_step(accel, p, v, k * cfg.dt, cfg.dt)
        ps.append(p)
        vs.append(v)
        accs.append(a)
        p, v = p_next, v_next
    t = torch.as_tensor(cfg.times, dtype=torch.float32, device=device)
    return t, torch.stack(ps, dim=1), torch.stack(vs, dim=1), torch.stack(accs, dim=1)


def simulate_flights(commands: np.ndarray, context: np.ndarray, cfg: SimConfig):
    """NumPy twin of :func:`simulate` for one situation and many command sequences ``(B, knots, 3)``."""
    ac = cfg.aircraft
    wind = canonical_scenario(context).wind
    commands = np.asarray(commands, dtype=float)
    batch = commands.shape[0]
    p = np.zeros((batch, 3))
    p[:, 2] = heading_context_raw(context)[2]
    v = np.zeros((batch, 3))
    v[:, 0] = cfg.initial_airspeed
    v = v + wind(p, 0.0)
    n_max_cl = cfg.safety * cfg.load_max * ac.weight

    def accel(pos, vel, time):
        j, w = _knot_weights(time, cfg)
        cmd = (1.0 - w) * commands[:, j] + w * commands[:, j + 1]
        v_air = vel - wind(pos, time)
        q_s = 0.5 * ac.rho * np.sum(v_air**2, axis=-1) * ac.wing_area
        cl = np.minimum(cmd[:, 0], n_max_cl / np.maximum(q_s, 1e-3))  # load limiter
        return aerodynamic_acceleration(ac, v_air, cl, cmd[:, 1], cmd[:, 2])

    n = cfg.n_steps + 1
    ps, vs, accs = np.empty((batch, n, 3)), np.empty((batch, n, 3)), np.empty((batch, n, 3))
    for k in range(n):
        ps[:, k], vs[:, k] = p, v
        p, v, accs[:, k] = rk4_step(accel, p, v, k * cfg.dt, cfg.dt)
    return cfg.times, ps, vs, accs
