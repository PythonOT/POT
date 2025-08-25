# -*- coding: utf-8 -*-
"""
Batch operations for optimal transport.
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Paul Krzakala <paul.krzakala@gmail.com>
#
# License: MIT License

from ._linear import solve_batch
from ._quadratic import solve_gromov_batch

__all__ = ["solve_batch", "solve_gromov_batch"]
