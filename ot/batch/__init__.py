# -*- coding: utf-8 -*-
"""
Batch operations for optimal transport.
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Paul Krzakala <paul.krzakala@gmail.com>
#
# License: MIT License

from ._linear import linear_solver_batch
from ._quadratic import quadratic_solver_batch

__all__ = ["linear_solver_batch", "quadratic_solver_batch"]
