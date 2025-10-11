# -*- coding: utf-8 -*-
"""
Solvers related to Bregman projections for entropic regularized OT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

from ._utils import geometricBar, geometricMean, projR, projC

from ._sinkhorn import (
    sinkhorn,
    sinkhorn2,
    sinkhorn_knopp,
    sinkhorn_log,
    greenkhorn,
    sinkhorn_stabilized,
    sinkhorn_epsilon_scaling,
)

from ._barycenter import (
    barycenter,
    barycenter_sinkhorn,
    free_support_sinkhorn_barycenter,
    barycenter_stabilized,
    barycenter_debiased,
    jcpot_barycenter,
)

from ._convolutional import (
    convolutional_barycenter2d,
    convolutional_barycenter2d_debiased,
)

from ._empirical import (
    empirical_sinkhorn,
    empirical_sinkhorn2,
    empirical_sinkhorn_divergence,
    empirical_sinkhorn_nystroem,
    empirical_sinkhorn_nystroem2,
)

from ._screenkhorn import screenkhorn

from ._dictionary import unmix

from ._geomloss import empirical_sinkhorn2_geomloss, geomloss


__all__ = [
    "geometricBar",
    "geometricMean",
    "projR",
    "projC",
    "sinkhorn",
    "sinkhorn2",
    "sinkhorn_knopp",
    "sinkhorn_log",
    "greenkhorn",
    "sinkhorn_stabilized",
    "sinkhorn_epsilon_scaling",
    "barycenter",
    "barycenter_sinkhorn",
    "free_support_sinkhorn_barycenter",
    "barycenter_stabilized",
    "barycenter_debiased",
    "jcpot_barycenter",
    "convolutional_barycenter2d",
    "convolutional_barycenter2d_debiased",
    "empirical_sinkhorn",
    "empirical_sinkhorn2",
    "empirical_sinkhorn2_geomloss",
    "empirical_sinkhorn_divergence",
    "empirical_sinkhorn_nystroem",
    "empirical_sinkhorn_nystroem2",
    "geomloss",
    "screenkhorn",
    "unmix",
]
