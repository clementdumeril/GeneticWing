import numpy as np
import torch

from geneticwing.aircraft import SBXC_APPROX as AC
from geneticwing.benchmarks import canonical_scenario, heading_frame_context, s1_family
from geneticwing.flatness import flat_state
from geneticwing.physvae import from_unit_commands, mirror_commands, to_unit_commands
from geneticwing.soaring import SoaringProblem
from geneticwing.torchsim import SimConfig, commands_from_raw, simulate, simulate_flights

# A situation whose thermal is 3 km away: the air where we fly is still.
STILL = (np.array([3000.0, 0.0, 300.0, 1.2, 0.0, 0.0]) - [0, 0, 250, 2, 0, 0]) / [200, 200, 70, 0.8, 4, 4]


def test_steady_glide_turn_loses_exactly_the_polar_sink_even_with_large_steps():
    airspeed, bank = 9.0, np.radians(30.0)
    _, sink, cl = AC.steady_glide(airspeed, bank)
    cfg = SimConfig(AC, dt=0.4, knots=61, initial_airspeed=airspeed)
    t, p, v, _ = simulate_flights(np.tile([cl, bank, 0.0], (1, 61, 1)), STILL, cfg)
    energy = p[0, :, 2] + np.sum(v[0] ** 2, axis=-1) / (2 * AC.g)
    q = len(t) // 4  # skip the transient from the level-flight start
    rate = (energy[-1] - energy[q]) / (t[-1] - t[q])
    assert abs(rate + sink) < 2e-3


def test_numpy_and_torch_simulators_agree():
    cfg = SimConfig(AC, dt=0.4, knots=31)
    ctx = heading_frame_context(s1_family(2)[0])
    # Smooth flight: the float32 (torch) and float64 (NumPy) runs agree over the whole horizon.
    _, sink, cl = AC.steady_glide(9.0, np.radians(20.0))
    smooth = np.tile([cl, np.radians(20.0), 2.0], (2, cfg.knots, 1))
    _, pt, _, _ = simulate(torch.tensor(smooth, dtype=torch.float32), np.tile(ctx, (2, 1)), cfg)
    _, pn, _, _ = simulate_flights(smooth, ctx, cfg)
    np.testing.assert_allclose(pt.numpy(), pn, atol=0.05)
    # Random, aggressive commands are chaotic: rounding differences grow, so compare the first 15 s.
    cmds = commands_from_raw(torch.randn(3, cfg.knots, 3, generator=torch.Generator().manual_seed(0)), cfg)
    _, pt, _, _ = simulate(cmds, np.tile(ctx, (3, 1)), cfg)
    _, pn, _, _ = simulate_flights(cmds.numpy(), ctx, cfg)
    early = int(15.0 / cfg.dt)
    np.testing.assert_allclose(pt.numpy()[:, :early], pn[:, :early], atol=0.05)


def test_any_bounded_commands_respect_the_dynamic_constraints():
    cfg = SimConfig(AC, dt=0.4, knots=31)
    ctx = heading_frame_context(s1_family(4)[0])
    problem = SoaringProblem(canonical_scenario(ctx))
    raw = 3.0 * torch.randn(32, cfg.knots, 3, generator=torch.Generator().manual_seed(1))
    t, p, v, a = simulate_flights(commands_from_raw(raw, cfg).numpy(), ctx, cfg)
    tt = np.broadcast_to(t, p.shape[:-1])
    hinges = problem.hinges(flat_state(AC, tt, p, v, a, problem.scenario.wind(p, tt)))
    for name in ("cl", "bank", "load", "thrust_max", "brake"):
        assert hinges[name].max() <= 1e-6, name


def test_unit_commands_round_trip_and_mirror():
    cfg = SimConfig(AC, knots=5)
    rng = np.random.default_rng(0)
    u = rng.uniform(size=(4, 15))
    np.testing.assert_allclose(to_unit_commands(from_unit_commands(u, cfg), cfg), u)
    ctx = rng.normal(size=(4, 6))
    mc, mu = mirror_commands(ctx, u)
    np.testing.assert_allclose(from_unit_commands(mu, cfg)[..., 1], -from_unit_commands(u, cfg)[..., 1])
    np.testing.assert_allclose(mc[:, [1, 5]], -ctx[:, [1, 5]])
    np.testing.assert_allclose(mc[:, [0, 2, 3, 4]], ctx[:, [0, 2, 3, 4]])
