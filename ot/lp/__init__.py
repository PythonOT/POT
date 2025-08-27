# -*- coding: utf-8 -*-
"""
Solvers for the original linear program OT problem.

"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

from .dmmot import dmmot_monge_1dgrid_loss, dmmot_monge_1dgrid_optimize
from ._network_simplex import emd, emd2
from ._barycenter_solvers import (
    barycenter,
    free_support_barycenter,
    generalized_free_support_barycenter,
    free_support_barycenter_generic_costs,
    ot_barycenter_energy,
    NorthWestMMGluing,
)
from ..utils import check_number_threads

# import compiled emd
from .emd_wrap import emd_1d_sorted
from .solver_1d import (
    emd_1d,
    emd2_1d,
    wasserstein_1d,
    binary_search_circle,
    wasserstein_circle,
    semidiscrete_wasserstein2_unif_circle,
    linear_circular_ot,
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
    "linear_circular_ot",
    "dmmot_monge_1dgrid_loss",
    "dmmot_monge_1dgrid_optimize",
    "check_number_threads",
    "free_support_barycenter_generic_costs",
    "NorthWestMMGluing",
    "ot_barycenter_energy",
]
