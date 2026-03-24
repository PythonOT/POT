# -*- coding: utf-8 -*-
"""
Efficient combinatorial optimization for low transport cost bijections based on Binary Space Partitioning trees (BSP-OT).

"""

# Author : Baptiste Genest <baptistegenest@gmail.com>
#
# License: MIT License

from .bsp_ot import compute_bspot_bijection, merge_bijections

__all__ = ["compute_bspot_bijection", "merge_bijections"]
