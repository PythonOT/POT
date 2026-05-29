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
    projection_sphere_to_circle,
)
from ._sliced_distances import (
    sliced_wasserstein_distance,
    max_sliced_wasserstein_distance,
)
from ._spherical_sliced import (
    sliced_wasserstein_sphere,
    sliced_wasserstein_sphere_unif,
    linear_sliced_wasserstein_sphere,
)
from ._sliced_plans import min_pivot_sliced, expected_sliced, sliced_plans

__all__ = [
    "get_random_projections",
    "get_projections_sphere",
    "projection_sphere_to_circle",
    "min_pivot_sliced",
    "expected_sliced",
    "sliced_plans",
    "sliced_wasserstein_distance",
    "max_sliced_wasserstein_distance",
    "sliced_wasserstein_sphere",
    "sliced_wasserstein_sphere_unif",
    "linear_sliced_wasserstein_sphere",
]
