import numpy as np
import pytest

from geneticwing.benchmarks import CONTEXT_NAMES, s1_family
from geneticwing.seeding import direct_genome, fit_genome
from geneticwing.soaring import SoaringProblem
from geneticwing.student import FlowMatchingStudent, RegressionStudent, Standardizer


@pytest.fixture(scope="module")
def family_problem():
    scenario, context = s1_family(0)
    return SoaringProblem(scenario), context


def test_family_context_is_normalized_and_reproducible(family_problem):
    _, context = family_problem
    assert context.shape == (len(CONTEXT_NAMES),)
    assert np.all(np.abs(context) < 3.0)
    np.testing.assert_array_equal(context, s1_family(0)[1])
    assert not np.array_equal(context, s1_family(1)[1])


def test_fitting_a_genome_recovers_a_known_trajectory(family_problem):
    problem, _ = family_problem
    genes = problem.to_physical(problem.straight_genome())
    genes[0::3] = np.radians(15.0)  # a steady turn the genome can represent exactly
    target = problem.flat(problem.to_normalized(genes))
    t, p = target.t[0], target.position[0]
    rough = problem.flat(direct_genome(problem, t, p)).position[0]
    fitted = problem.flat(fit_genome(problem, t, p)).position[0]
    assert np.max(np.abs(fitted - p)) < 2.0
    assert np.max(np.abs(fitted - p)) <= np.max(np.abs(rough - p)) + 1e-9


def test_standardizer_round_trip():
    x = np.random.default_rng(0).uniform(size=(50, 4))
    s = Standardizer.fit(x)
    np.testing.assert_allclose(s.inverse(s.forward(x)), x)


def test_flow_matching_learns_two_modes_that_regression_averages():
    # One context, two equally likely answers: +0.8 and +0.2 on every gene.
    rng = np.random.default_rng(0)
    modes = np.where(rng.uniform(size=(400, 1)) < 0.5, 0.8, 0.2) * np.ones((1, 3))
    contexts = np.zeros((400, 1))
    fm = FlowMatchingStudent(3, 1, width=64, depth=3)
    fm.fit(modes, contexts, steps=1500, batch=128)
    samples = fm.sample(np.zeros(1), 200, n_steps=16, rng=np.random.default_rng(1))
    near_mode = np.minimum(np.abs(samples[:, 0] - 0.8), np.abs(samples[:, 0] - 0.2)) < 0.12
    assert near_mode.mean() > 0.8
    assert 0.2 < (samples[:, 0] > 0.5).mean() < 0.8  # both modes present

    reg = RegressionStudent(3, 1, width=64, depth=3)
    reg.fit(modes, contexts, steps=500, batch=128)
    assert abs(reg.predict(np.zeros(1))[0, 0] - 0.5) < 0.1  # the average of the two modes


def test_learned_mutation_stays_in_the_unit_box():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(100, 5))
    fm = FlowMatchingStudent(5, 2, width=32, depth=2)
    fm.fit(x, rng.normal(size=(100, 2)), steps=50)
    children = fm.mutate(x[:4], np.zeros(2), rng=rng)
    assert children.shape == (4, 5) and np.all((children >= 0) & (children <= 1))


def test_heading_context_is_invariant_to_rotating_the_scene():
    from dataclasses import replace

    from geneticwing.benchmarks import heading_frame_context
    from geneticwing.wind import Thermal, UniformWind

    scenario, _ = s1_family(3)
    thermal = next(p for p in scenario.wind.components if isinstance(p, Thermal))
    wind = next(p for p in scenario.wind.components if isinstance(p, UniformWind)).velocity
    a = np.radians(70.0)  # rotate everything counter-clockwise about the thermal (at the origin)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    start = rot @ np.asarray(scenario.start[:2])
    w = rot @ np.asarray(wind[:2])
    rotated = replace(
        scenario,
        start=(start[0], start[1], scenario.start[2]),
        heading_deg=scenario.heading_deg - 70.0,  # compass headings turn clockwise
        wind=UniformWind((w[0], w[1], 0.0)) + replace(thermal, drift=tuple(w)),
    )
    np.testing.assert_allclose(heading_frame_context(rotated), heading_frame_context(scenario), atol=1e-9)


def test_margins_tighten_every_limit():
    from geneticwing.benchmarks import with_margins

    scenario, _ = s1_family(0)
    tight = with_margins(scenario)
    assert tight.cl_margin < scenario.cl_margin and tight.bank_max_deg < scenario.bank_max_deg
    assert tight.load_max < scenario.load_max and tight.thrust_max < scenario.thrust_max
    assert tight.brake_capacity < scenario.brake_capacity
