"""Students that imitate the teacher: conditional flow matching and a regression baseline.

Flow matching (Lipman et al. 2023; rectified flow, Liu et al. 2023) learns a
velocity field ``v_theta(x, t, c)`` transporting Gaussian noise ``x_0`` to the
teacher's genomes ``x_1`` along straight lines:

    L = E || v_theta((1 - t) x_0 + t x_1, t, c) - (x_1 - x_0) ||^2

and generates by integrating ``dx/dt = v_theta`` from ``t = 0`` to ``1`` with a
few Euler steps. Because it models a distribution, it can put mass on both
orbit directions instead of averaging them, which is what the deterministic
regression baseline (``c -> x``, mean squared error) does (hypothesis H3).

Genomes are standardized with the training mean and spread; generated
genomes are mapped back and clipped to the unit box of the gene bounds.
Both students train and sample on ``device`` ("cpu", "cuda", or "auto").
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _mlp(n_in: int, n_out: int, width: int, depth: int) -> nn.Sequential:
    layers, n = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(n, width), nn.SiLU()]
        n = width
    layers.append(nn.Linear(n, n_out))
    return nn.Sequential(*layers)


class _TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.linspace(0.0, math.log(1000.0), dim // 2)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t[:, None] * self.freqs[None]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class VelocityField(nn.Module):
    def __init__(self, dim: int, context_dim: int, width: int = 512, depth: int = 4, time_dim: int = 16):
        super().__init__()
        self.time = _TimeEmbedding(time_dim)
        self.net = _mlp(dim + context_dim + time_dim, dim, width, depth)

    def forward(self, x, t, c):
        return self.net(torch.cat([x, c, self.time(t)], dim=-1))


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        return cls(x.mean(axis=0), x.std(axis=0) + 1e-3)

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, z):
        return np.clip(z * self.std + self.mean, 0.0, 1.0)


@dataclass
class TrainingLog:
    losses: list[float]
    seconds: float


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _batches(n: int, batch: int, steps: int, rng: np.random.Generator):
    for _ in range(steps):
        yield rng.integers(0, n, size=batch)


class FlowMatchingStudent:
    def __init__(self, dim: int, context_dim: int, width: int = 512, depth: int = 4, seed: int = 0, device: str = "auto"):
        torch.manual_seed(seed)
        self.device = resolve_device(device)
        self.model = VelocityField(dim, context_dim, width, depth).to(self.device)
        self.scaler: Standardizer | None = None
        self.dim = dim

    def fit(self, genomes, contexts, steps=20_000, batch=256, lr=1e-3, seed=0) -> TrainingLog:
        self.scaler = Standardizer.fit(genomes)
        x1_all = torch.as_tensor(self.scaler.forward(genomes), dtype=torch.float32, device=self.device)
        c_all = torch.as_tensor(contexts, dtype=torch.float32, device=self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        rng, losses, start = np.random.default_rng(seed), [], time.perf_counter()
        self.model.train()
        for idx in _batches(len(x1_all), batch, steps, rng):
            idx = torch.as_tensor(idx, device=self.device)
            x1, c = x1_all[idx], c_all[idx]
            x0 = torch.randn_like(x1)
            t = torch.rand(len(x1), device=self.device)
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1
            loss = torch.mean((self.model(xt, t, c) - (x1 - x0)) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
        self.model.eval()
        return TrainingLog(losses, time.perf_counter() - start)

    @torch.no_grad()
    def integrate(self, z: np.ndarray, contexts: np.ndarray, t0: float = 0.0, n_steps: int = 8) -> np.ndarray:
        """Euler integration of the flow from ``t0`` to 1, in standardized space."""
        x = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        c = torch.as_tensor(contexts, dtype=torch.float32, device=self.device)
        ts = np.linspace(t0, 1.0, n_steps + 1)
        for a, b in zip(ts[:-1], ts[1:]):
            t = torch.full((len(x),), float(a), device=self.device)
            x = x + (b - a) * self.model(x, t, c)
        return x.cpu().numpy()

    def sample(self, context: np.ndarray, n: int, n_steps: int = 8, rng: np.random.Generator | None = None) -> np.ndarray:
        """``n`` genomes (unit box) for one context."""
        rng = rng or np.random.default_rng()
        z = rng.standard_normal((n, self.dim))
        c = np.tile(context, (n, 1))
        return self.scaler.inverse(self.integrate(z, c, 0.0, n_steps))

    def mutate(self, genomes: np.ndarray, context: np.ndarray, t0: float = 0.6, n_steps: int = 4, rng=None) -> np.ndarray:
        """Learned mutation (Diffusion-ES style): partly re-noise, then flow back to the data."""
        rng = rng or np.random.default_rng()
        x1 = self.scaler.forward(genomes)
        xt = t0 * x1 + (1 - t0) * rng.standard_normal(x1.shape)
        c = np.tile(context, (len(genomes), 1))
        return self.scaler.inverse(self.integrate(xt, c, t0, n_steps))


def smooth_genomes(genomes: np.ndarray, width: int = 3, genes_per_point: int = 3) -> np.ndarray:
    """Moving average of each gene channel along the control polygon.

    The teacher's trajectories are smooth; the student's errors are mostly
    high-frequency jitter between neighbouring control points, which the
    second derivatives (C_L, thrust) amplify. Filtering removes it.
    """
    from scipy.ndimage import uniform_filter1d

    g = genomes.reshape(len(genomes), -1, genes_per_point)
    return uniform_filter1d(g, width, axis=1, mode="nearest").reshape(len(genomes), -1)


class RegressionStudent:
    """Deterministic baseline: one genome per context, trained with mean squared error."""

    def __init__(self, dim: int, context_dim: int, width: int = 512, depth: int = 4, seed: int = 0, device: str = "auto"):
        torch.manual_seed(seed)
        self.device = resolve_device(device)
        self.model = _mlp(context_dim, dim, width, depth).to(self.device)
        self.scaler: Standardizer | None = None

    def fit(self, genomes, contexts, steps=20_000, batch=256, lr=1e-3, seed=0) -> TrainingLog:
        self.scaler = Standardizer.fit(genomes)
        y_all = torch.as_tensor(self.scaler.forward(genomes), dtype=torch.float32, device=self.device)
        c_all = torch.as_tensor(contexts, dtype=torch.float32, device=self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        rng, losses, start = np.random.default_rng(seed), [], time.perf_counter()
        self.model.train()
        for idx in _batches(len(y_all), batch, steps, rng):
            idx = torch.as_tensor(idx, device=self.device)
            loss = torch.mean((self.model(c_all[idx]) - y_all[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
        self.model.eval()
        return TrainingLog(losses, time.perf_counter() - start)

    @torch.no_grad()
    def predict(self, context: np.ndarray) -> np.ndarray:
        y = self.model(torch.as_tensor(context[None], dtype=torch.float32, device=self.device)).cpu().numpy()
        return self.scaler.inverse(y)
