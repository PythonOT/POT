# -*- coding: utf-8 -*-
"""
(DEPRECATED) LP solvers for optimal transport using cvxopt
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import warnings
from ._barycenter_solvers import barycenter


__all__ = ["barycenter"]


warnings.warn(
    "The module ot.lp.cvx is deprecated and will be removed in future versions."
    "The function `barycenter` was moved to ot.lp._barycenter_solvers and can"
    "be importer via ot.lp."
)
