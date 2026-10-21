"""Turn any flown or planned trajectory into a genome, to seed the planners.

The reactive law flies the orbit structure the planners struggle to discover
(phase 2 study). Fitting a genome to one of its flights hands that structure
to CMA-ES or the NLP as a starting point.

The fit has two stages: a direct read of the genes from the trajectory sampled
at the Greville abscissae (a control polygon close to the curve), then a
least-squares refinement of the sampled positions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from geneticwing.soaring import SoaringProblem


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def direct_genome(problem: SoaringProblem, t: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Genes read off the trajectory at the control points' Greville times."""
    times = problem.spline.greville[3:] * problem.scenario.horizon
    targets = np.stack([np.interp(times, t, positions[:, i]) for i in range(3)], axis=-1)
    polygon = np.vstack([problem.fixed[2], targets])
    d = np.diff(polygon, axis=0)
    horizontal = np.hypot(d[:, 0], d[:, 1])
    heading = np.unwrap(np.concatenate([[problem.heading0], np.arctan2(d[:, 1], d[:, 0])]))
    turn_rate = np.diff(heading) / problem.step_durations
    speed = np.linalg.norm(d, axis=-1) / problem.step_durations
    gamma = np.arctan2(d[:, 2], horizontal)
    genes = np.stack([turn_rate, speed, gamma], axis=-1).ravel()
    return np.clip(problem.to_normalized(genes), 0.0, 1.0)


def fit_genome(problem: SoaringProblem, t: np.ndarray, positions: np.ndarray, max_nfev: int = 60) -> np.ndarray:
    """Genome whose spline best matches ``positions`` sampled at times ``t`` (least squares)."""
    x0 = direct_genome(problem, t, positions)
    target = np.stack([np.interp(problem.t, t, positions[:, i]) for i in range(3)], axis=-1)
    basis = problem._bases[0]

    def residual(x):
        cp = problem.decode(x)[0]
        return (basis @ cp - target).ravel()

    fit = least_squares(residual, x0, bounds=(0.0, 1.0), max_nfev=max_nfev, diff_step=1e-4)
    return fit.x
