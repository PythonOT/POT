# -*- coding: utf-8 -*-
"""
Solvers related to Unbalanced Optimal Transport problems.

"""

# Author: Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

# All submodules and packages
from ._sinkhorn import (
    sinkhorn_knopp_unbalanced,
    sinkhorn_unbalanced,
    sinkhorn_stabilized_unbalanced,
    sinkhorn_unbalanced_translation_invariant,
    sinkhorn_unbalanced2,
    barycenter_unbalanced_sinkhorn,
    barycenter_unbalanced_stabilized,
    barycenter_unbalanced,
)

from ._mm import mm_unbalanced, mm_unbalanced2

from ._lbfgs import lbfgsb_unbalanced, lbfgsb_unbalanced2

__all__ = [
    "sinkhorn_knopp_unbalanced",
    "sinkhorn_unbalanced",
    "sinkhorn_stabilized_unbalanced",
    "sinkhorn_unbalanced_translation_invariant",
    "sinkhorn_unbalanced2",
    "barycenter_unbalanced_sinkhorn",
    "barycenter_unbalanced_stabilized",
    "barycenter_unbalanced",
    "mm_unbalanced",
    "mm_unbalanced2",
    "_get_loss_unbalanced",
    "lbfgsb_unbalanced",
    "lbfgsb_unbalanced2",
]
