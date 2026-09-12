# -*- coding: utf-8 -*-
"""
Solvers related to (balanced) sliced transport.

"""

# Author: Laetitia Chapel <laetitia.chapel@irisa.fr>
#
# License: MIT License

# All submodules and packages

from ._utils import (
    get_random_projections,
    get_projections_sphere,
    get_random_rotations,
    projection_sphere_to_circle,
    get_projections_spiral,
    projection_sphere_to_ball,
)
from ._sliced_distances import (
    sliced_wasserstein_distance,
    max_sliced_wasserstein_distance,
)
from ._spherical_sliced import (
    sliced_wasserstein_sphere,
    sliced_wasserstein_sphere_unif,
    linear_sliced_wasserstein_sphere,
    stereographic_sliced_wasserstein_sphere,
)
from ._sliced_plans import min_sliced_transport_plan, expected_sliced_plan, sliced_plans

__all__ = [
    "get_random_projections",
    "get_projections_sphere",
    "get_random_rotations",
    "projection_sphere_to_circle",
    "projection_sphere_to_ball",
    "min_sliced_transport_plan",
    "expected_sliced_plan",
    "sliced_plans",
    "sliced_wasserstein_distance",
    "max_sliced_wasserstein_distance",
    "sliced_wasserstein_sphere",
    "sliced_wasserstein_sphere_unif",
    "linear_sliced_wasserstein_sphere",
    "get_projections_spiral",
    "stereographic_sliced_wasserstein_sphere",
]
