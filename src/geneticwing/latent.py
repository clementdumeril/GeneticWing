"""Genetic search in a learned latent space of trajectories.

The phase 2 study showed the GA fails on the 87 coupled genes of a trajectory
but works on a handful of parameters. This module gives it a handful of
parameters: a conditional VAE (Kingma and Welling 2014; Sohn et al. 2015)
compresses the teacher's trajectories into a latent vector ``z`` of a few
dimensions, given the situation ``c``. The GA then searches ``z``; every
individual is decoded, smoothed and checked by the flat evaluation.

This is latent variable evolution (Volz et al. 2018; Bontrager et al. 2018).
The retraining loop of ``scenarios/phase3_latent.py`` follows data-driven encodings
(Gaier, Asteroth and Mouret, GECCO 2020) and weighted retraining (Tripp et al.,
NeurIPS 2020): better solutions found in the latent space replace the
teacher's, and the VAE is retrained on them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from geneticwing.evolution import CMAES, OptimizationResult, RealGA
from geneticwing.planning.local import solve_local
from geneticwing.soaring import SoaringProblem
from geneticwing.student import Standardizer, TrainingLog, _mlp, resolve_device, smooth_genomes

LATENT_BOX = 3.0  # the search covers z in [-3, 3]^d, i.e. +-3 prior standard deviations


class ConditionalVAE:
    def __init__(
        self,
        dim: int,
        context_dim: int,
        latent_dim: int,
        width: int = 512,
        depth: int = 3,
        beta: float = 0.1,
        seed: int = 0,
        device: str = "auto",
    ):
        torch.manual_seed(seed)
        self.device = resolve_device(device)
        self.encoder = _mlp(dim + context_dim, 2 * latent_dim, width, depth).to(self.device)
        self.decoder = _mlp(latent_dim + context_dim, dim, width, depth).to(self.device)
        self.latent_dim, self.dim, self.beta = latent_dim, dim, beta
        self.scaler: Standardizer | None = None

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def fit(self, genomes, contexts, weights=None, steps=20_000, batch=256, lr=1e-3, seed=0) -> TrainingLog:
        """Weighted beta-VAE training; reuses the current weights when called again (fine-tuning)."""
        if self.scaler is None:
            self.scaler = Standardizer.fit(genomes)
        x_all = torch.as_tensor(self.scaler.forward(genomes), dtype=torch.float32, device=self.device)
        c_all = torch.as_tensor(contexts, dtype=torch.float32, device=self.device)
        p = None if weights is None else np.asarray(weights, float) / np.sum(weights)
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        rng, losses, start = np.random.default_rng(seed), [], time.perf_counter()
        for _ in range(steps):
            idx = rng.choice(len(x_all), size=batch, p=p) if p is not None else rng.integers(0, len(x_all), batch)
            idx = torch.as_tensor(idx, device=self.device)
            x, c = x_all[idx], c_all[idx]
            mu, logvar = self.encoder(torch.cat([x, c], dim=-1)).chunk(2, dim=-1)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            x_hat = self.decoder(torch.cat([z, c], dim=-1))
            reconstruction = torch.sum((x_hat - x) ** 2, dim=-1).mean()
            kl = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=-1).mean()
            loss = reconstruction + self.beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
        return TrainingLog(losses, time.perf_counter() - start)

    @torch.no_grad()
    def encode(self, genomes: np.ndarray, contexts: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(self.scaler.forward(np.atleast_2d(genomes)), dtype=torch.float32, device=self.device)
        c = torch.as_tensor(np.atleast_2d(contexts), dtype=torch.float32, device=self.device)
        mu, _ = self.encoder(torch.cat([x, c], dim=-1)).chunk(2, dim=-1)
        return mu.cpu().numpy()

    @torch.no_grad()
    def decode(self, z: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Latent vectors ``(P, d)`` for one context -> genomes ``(P, dim)`` in the unit box."""
        z = torch.as_tensor(np.atleast_2d(z), dtype=torch.float32, device=self.device)
        c = torch.as_tensor(np.tile(context, (len(z), 1)), dtype=torch.float32, device=self.device)
        return self.scaler.inverse(self.decoder(torch.cat([z, c], dim=-1)).cpu().numpy())

    def state(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "scaler": (self.scaler.mean, self.scaler.std),
            "config": {"latent_dim": self.latent_dim, "dim": self.dim, "beta": self.beta},
        }

    def load(self, state: dict) -> "ConditionalVAE":
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        self.scaler = Standardizer(*state["scaler"])
        return self


def to_latent(x: np.ndarray) -> np.ndarray:
    """Optimizer box [0, 1]^d -> latent [-LATENT_BOX, LATENT_BOX]^d."""
    return LATENT_BOX * (2.0 * np.asarray(x) - 1.0)


def to_box(z: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(z) / LATENT_BOX + 1.0) / 2.0, 0.0, 1.0)


@dataclass
class LatentResult:
    genome: np.ndarray  # best decoded (then polished) genome
    fitness: float
    search: OptimizationResult


def latent_search(
    vae: ConditionalVAE,
    problem: SoaringProblem,
    context: np.ndarray,
    budget: int,
    rng: np.random.Generator,
    seeds_z: np.ndarray | None = None,
    optimizer: str = "GA",
    nlp_budget: int = 0,
) -> LatentResult:
    """GA (or CMA-ES) over the latent space for one situation, then optional NLP polishing."""

    def decode(x):
        return smooth_genomes(vae.decode(to_latent(x), context))

    def objective(x):
        return problem.evaluate(decode(x)).fitness

    seeds = None if seeds_z is None else to_box(seeds_z)
    if optimizer == "GA":
        search = RealGA(population=50).minimize(objective, vae.latent_dim, budget, rng, seeds)
    else:
        search = CMAES(population=16).minimize(objective, vae.latent_dim, budget, rng, seeds)
    genome = decode(search.x[None])[0]
    fitness = search.f
    if nlp_budget:
        polished, _ = solve_local(problem, genome, nlp_budget)
        if polished.f < fitness:
            genome, fitness = polished.x, polished.f
    return LatentResult(genome, float(fitness), search)
