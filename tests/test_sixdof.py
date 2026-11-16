import numpy as np
import pytest
from dataclasses import replace

from geneticwing.aircraft import SBXC_APPROX
from geneticwing.sixdof import (DA, PHI, AutopilotGains, SixDofPlant, air_data, derivatives, linearize, rotation,
                                sbxc_estimated, trim_glide)
from geneticwing.wind.base import UniformWind

AF = sbxc_estimated()
STILL = UniformWind((0.0, 0.0, 0.0))


@pytest.mark.parametrize("airspeed", [9.0, 12.0])
def test_trimmed_glide_sinks_like_the_point_mass(airspeed):
    x0, u0 = trim_glide(AF, airspeed)
    xdot, _ = derivatives(AF, x0, u0[:3], 0.0, np.zeros(3))
    assert np.allclose(xdot[3:], 0.0, atol=1e-6)  # a true equilibrium (position aside)
    sink = (rotation(*x0[6:9]) @ x0[3:6])[2]
    assert sink == pytest.approx(float(SBXC_APPROX.steady_glide(airspeed)[1]), rel=0.05)


def test_blended_lift_curve_peaks_at_the_point_mass_cl_max():
    a = np.linspace(0.0, 0.4, 4001)
    assert AF.static_cl(a).max() == pytest.approx(SBXC_APPROX.cl_max, abs=5e-3)
    # nearly linear where the point mass flies: within 2 % up to 0.9 C_L max (the blend bends it slightly)
    a_lin = (0.9 * SBXC_APPROX.cl_max - AF.CL0) / AF.CLa
    assert float(AF.static_cl(a_lin)) == pytest.approx(AF.CL0 + AF.CLa * a_lin, rel=0.02)


def test_linearization_has_the_rigid_body_modes():
    x0, u0 = trim_glide(AF, 9.0)
    a, b = linearize(AF, x0, u0)
    assert a.shape == (12, 12) and b.shape == (12, 4)
    ev = np.linalg.eigvals(a)
    oscillatory = ev[ev.imag > 1e-3]
    assert len(oscillatory) == 3  # short period, Dutch roll, phugoid
    damping = -oscillatory.real / np.abs(oscillatory)
    assert np.all(damping > 0.0)
    assert np.max(np.abs(ev.real[ev.real < 0])) > 5.0  # a fast roll subsidence


def test_rigid_body_conserves_energy_without_air():
    af = replace(AF, area=1e-12)
    x = np.zeros(12)
    x[2], x[3], x[6], x[9], x[11] = -400.0, 10.0, 0.3, 0.5, -0.2
    h = 0.01

    def energy(x):
        v = rotation(*x[6:9]) @ x[3:6]
        omega = x[9:12]
        rot = 0.5 * (af.jx * omega[0] ** 2 + af.jy * omega[1] ** 2 + af.jz * omega[2] ** 2)
        return af.mass * af.g * (-x[2]) + 0.5 * af.mass * v @ v + rot

    e0 = energy(x)
    s = np.zeros(3)
    for _ in range(500):
        f = lambda y: derivatives(af, y, s, 0.0, np.zeros(3))[0]  # noqa: E731
        k1 = f(x); k2 = f(x + h / 2 * k1); k3 = f(x + h / 2 * k2); k4 = f(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert energy(x) == pytest.approx(e0, rel=1e-6)


def _fly(bank_cmd, seconds):
    plant = SixDofPlant(AF)
    plant.begin(np.array([0.0, 0.0, 400.0]), np.array([9.5, 0.0, -0.35]), STILL, 0.0, 0.7, 0.0)
    for k in range(int(seconds / 0.02)):
        ad = air_data(AF, plant.x[:12], np.zeros(3))
        cl = AF.weight / np.cos(bank_cmd) / (0.5 * AF.rho * ad.airspeed**2 * AF.area) + 0.05 * (ad.airspeed - 9.5)
        plant.step(STILL, k * 0.02, 0.02, float(np.clip(cl, 0.1, 1.08)), bank_cmd, 0.0)
    return plant


def test_autopilot_holds_a_coordinated_turn():
    plant = _fly(np.radians(30.0), 8.0)
    tail = slice(-100, None)
    assert np.degrees(np.abs(np.asarray(plant.log["phi"][tail]) - np.radians(30.0))).max() < 6.0
    assert np.degrees(np.abs(plant.log["beta"][tail])).max() < 3.0


def test_autopilot_holds_wings_level():
    plant = _fly(0.0, 6.0)
    assert np.degrees(abs(plant.x[PHI])) < 0.5 and np.degrees(abs(plant.x[DA])) < 0.5


def test_tracker_flies_a_circling_plan_with_the_rigid_body():
    from geneticwing.benchmarks import weather_family
    from geneticwing.closedloop import Plan, exact_belief, fly_closed_loop, LoopConfig
    from geneticwing.policies import seed_theta, simulate_law
    from geneticwing.soaring import SoaringProblem
    from geneticwing.closedloop import _select

    theta = np.full(10, 0.5)
    theta[[0, 1]] = 0.1, 0.4  # a tight orbit at a moderate speed

    def planner(scenario, rng, start=None):
        ev = simulate_law(SoaringProblem(scenario), seed_theta(theta, True, {})[None])
        return Plan(_select(ev.state, 0), float(ev.cost[0]), bool(ev.feasible[0]))

    situation = replace(weather_family(600)[0], horizon=20.0)
    flight = fly_closed_loop(situation, situation.wind, exact_belief(situation), planner,
                             LoopConfig(duration=20.0, replan_every=None, sixdof=AF), np.random.default_rng(0))
    ref = flight.plans[0].reference
    s = flight.state
    planned = np.stack([np.interp(s.t[0], ref.t, ref.position[:, i]) for i in range(3)], axis=-1)
    assert np.mean(np.linalg.norm(s.position[0] - planned, axis=-1)) < 25.0
    assert flight.sixdof["beta_max_deg"] < 30.0
    assert np.all(s.thrust_required >= 0.0)  # the motor's actual thrust (no strip wing here)
