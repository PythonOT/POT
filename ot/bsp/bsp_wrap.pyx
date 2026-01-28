# -*- coding: utf-8 -*-
"""
Cython linker with C solver
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
cimport numpy as np

cimport cython
cimport libc.math as math
from libc.stdint cimport uint64_t

cdef extern from "bsp_wrapper.h":
    double BSPOT_wrap(int n, int d, double *X, double *Y, uint64_t nb_plans, int *plans, int *plan)


@cython.boundscheck(False)
@cython.wraparound(False)
def bsp_solve(np.ndarray[double, ndim=2, mode="c"] X, np.ndarray[double, ndim=2, mode="c"] Y,  int n_plans=64):
    """
        Solves the Binary Space Partitioning (BSP) tree based OT problem and returns the optimal transport cost

        cost = bsp_solve(X,Y,plans)

    where :

    - X and Y are the input point clouds
    - plans is the set of BSP partitioning hyperplanes

    Returns the optimal transport cost.


    """
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef np.ndarray[int, ndim=2, mode="c"] plans = np.zeros((n, n_plans), dtype=np.int32) 
    cdef np.ndarray[int, ndim=1, mode="c"] plan = np.zeros(n, dtype=np.int32) 

    cdef double cost

    cost = BSPOT_wrap(n, d, <double*>X.data, <double*>Y.data, n_plans, <int*> plans.data, <int*> plan.data)

    # add 

    return cost, plan
    

@cython.boundscheck(False)
@cython.wraparound(False)
def merge_plans(np.ndarray[int, ndim=2, mode="c"] plans):
    """
        Merges OT plans

    where :

    - plans1 and plans2 are the input sets of BSP partitioning hyperplanes

    Returns the merged set of BSP partitioning hyperplanes.
    """
    
    plan = np.zeros((plans.shape[0],), dtype=np.int64)

    # add merging code here

    return plan


    