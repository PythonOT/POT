# -*- coding: utf-8 -*-
"""
General OT solvers with unified API
"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

# All submodules and packages
from ._linear import solve, solve_sample

from ._gromov import (
    solve_gromov,
)

from ._bary import (
    solve_bary_sample,
)


__all__ = [
    "solve",
    "solve_sample",
    "solve_gromov",
    "solve_bary_sample",
]
