"""GeneticWing: evolutionary, flatness-based planning of energy-harvesting
trajectories for fixed-wing UAVs.

Conventions used throughout the package:

* Inertial frame is ENU (x east, y north, z up), SI units.
* Wind ``W(p, t)`` is the velocity of the air mass in that frame, so a
  positive ``W[..., 2]`` is rising air.
* Arrays carry time along the second-to-last axis and space along the last
  one, ``(..., N, 3)``, so a whole population can be evaluated at once.
"""

from geneticwing.aircraft import Aircraft, STRATOSURFER_APPROX
from geneticwing.flatness import FlatState, flat_state

__all__ = ["Aircraft", "STRATOSURFER_APPROX", "FlatState", "flat_state"]
