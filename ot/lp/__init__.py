# -*- coding: utf-8 -*-
"""
Solvers for the original linear program OT problem.

"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

from . import cvx
from .cvx import barycenter
from .dmmot import dmmot_monge_1dgrid_loss, dmmot_monge_1dgrid_optimize
from .network_simplex import emd, emd2
from .barycenter import (
    free_support_barycenter, 
    generalized_free_support_barycenter
)

# import compiled emd
from .emd_wrap import emd_1d_sorted
from .solver_1d import (
    emd_1d,
    emd2_1d,
    wasserstein_1d,
    binary_search_circle,
    wasserstein_circle,
    semidiscrete_wasserstein2_unif_circle,
)

__all__ = [
    "emd",
    "emd2",
    "barycenter",
    "free_support_barycenter",
    "cvx",
    "emd_1d_sorted",
    "emd_1d",
    "emd2_1d",
    "wasserstein_1d",
    "generalized_free_support_barycenter",
    "binary_search_circle",
    "wasserstein_circle",
    "semidiscrete_wasserstein2_unif_circle",
    "dmmot_monge_1dgrid_loss",
    "dmmot_monge_1dgrid_optimize",
]
