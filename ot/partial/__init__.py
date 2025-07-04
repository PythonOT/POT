# -*- coding: utf-8 -*-
"""
Efficient 1D solver for the partial optimal transport problem.
"""

# Author: Romain Tavenard <romain.tavenard@univ-rennes2.fr>
#
# License: MIT License

# import compiled emd
from .partial_solvers import (
    partial_wasserstein_lagrange,
    partial_wasserstein,
    partial_wasserstein2,
    entropic_partial_wasserstein,
    gwgrad_partial,
    gwloss_partial,
    partial_gromov_wasserstein,
    partial_gromov_wasserstein2,
    entropic_partial_gromov_wasserstein,
    entropic_partial_gromov_wasserstein2,
    partial_wasserstein_1d,
)

__all__ = [
    "partial_wasserstein_1d",
    "partial_wasserstein_lagrange",
    "partial_wasserstein",
    "partial_wasserstein2",
    "entropic_partial_wasserstein",
    "gwgrad_partial",
    "gwloss_partial",
    "partial_gromov_wasserstein",
    "partial_gromov_wasserstein2",
    "entropic_partial_gromov_wasserstein",
    "entropic_partial_gromov_wasserstein2",
]
