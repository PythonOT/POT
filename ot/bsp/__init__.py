# -*- coding: utf-8 -*-
"""
Solvers for the Binary Space Partitioning (BSP) tree based OT problem.

"""

# Author : Baptiste Genest <baptistegenest@gmail.com>
#
# License: MIT License

from .bsp_ot import compute_bspot_bijection, merge_bijections

__all__ = ["compute_bspot_bijection", "merge_bijections"]
