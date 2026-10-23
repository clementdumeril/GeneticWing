import numpy as np

from geneticwing.benchmarks import heading_frame_context, s1_family
from geneticwing.latent import LATENT_BOX, ConditionalVAE, latent_search, to_box, to_latent
from geneticwing.soaring import SoaringProblem


def test_box_and_latent_maps_are_inverse():
    z = np.random.default_rng(0).uniform(-LATENT_BOX, LATENT_BOX, size=(10, 4))
    np.testing.assert_allclose(to_latent(to_box(z)), z)


def test_vae_reconstructs_low_dimensional_data():
    # Genomes generated from 2 hidden factors and the context: a 2-d latent suffices.
    rng = np.random.default_rng(0)
    factors, contexts = rng.uniform(size=(600, 2)), rng.uniform(size=(600, 1))
    mixing = rng.normal(size=(3, 12))
    genomes = 1.0 / (1.0 + np.exp(-np.hstack([factors, contexts]) @ mixing))
    vae = ConditionalVAE(12, 1, latent_dim=2, width=64, depth=2, beta=0.01, device="cpu")
    vae.fit(genomes, contexts, steps=2500, batch=128)
    decoded = np.vstack([vae.decode(vae.encode(g, c), c) for g, c in zip(genomes[:50], contexts[:50])])
    assert np.mean(np.abs(decoded - genomes[:50])) < 0.03


def test_weighted_fit_and_fine_tuning_run():
    rng = np.random.default_rng(1)
    g, c = rng.uniform(size=(100, 6)), rng.normal(size=(100, 2))
    vae = ConditionalVAE(6, 2, latent_dim=3, width=32, depth=2, device="cpu")
    first = vae.fit(g, c, steps=50)
    second = vae.fit(g, c, weights=np.linspace(1, 2, 100), steps=50, lr=3e-4)
    assert len(first.losses) == len(second.losses) == 50


def test_latent_search_returns_a_valid_genome():
    scenario, _ = s1_family(0)
    problem = SoaringProblem(scenario)
    context = heading_frame_context(scenario)
    rng = np.random.default_rng(0)
    genomes = np.clip(problem.straight_genome() + rng.normal(0, 0.02, size=(64, problem.dim)), 0, 1)
    vae = ConditionalVAE(problem.dim, len(context), latent_dim=4, width=32, depth=2, device="cpu")
    vae.fit(genomes, np.tile(context, (64, 1)), steps=100)
    result = latent_search(vae, problem, context, budget=200, rng=rng, seeds_z=np.zeros((1, 4)))
    assert result.genome.shape == (problem.dim,)
    assert np.all((result.genome >= 0) & (result.genome <= 1))
    assert result.search.evaluations <= 200
    assert np.isfinite(result.fitness)
