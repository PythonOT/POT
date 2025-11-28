# -*- coding: utf-8 -*-
"""
Solvers for the Binary Space Partitioning (BSP) tree based OT problem.

"""

from .bsp_wrap import bsp_solve, merge_plans

__all__ = ['bsp_solve', 'merge_plans']