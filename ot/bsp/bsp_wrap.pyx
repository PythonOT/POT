# -*- coding: utf-8 -*-
"""
Cython linker for C++ BSP-OT
"""

# Author: Baptiste Genest <baptistegenest@gmail.com>
#
# License: MIT License

import numpy as np
cimport numpy as np

cimport cython
cimport libc.math as math
from libc.stdint cimport uint64_t
from libcpp cimport bool


cdef extern from "bsp_wrapper.h":
    double BSPOT_wrap(int n, int d, double *X, double *Y, uint64_t nb_plans, int *plans, int *plan,int lp_power,int* initial_plan,bool gaussian,int seed)
    double MergeBijections(int n, int d, double *X, double *Y, uint64_t nb_plans, int *plans, int *plan,int lp_power)


@cython.boundscheck(False)
@cython.wraparound(False)
def bsp_solve_c(np.ndarray[double, ndim=2, mode="c"] X, np.ndarray[double, ndim=2, mode="c"] Y,  int n_plans=64,int lp_power = 2,np.ndarray[int,ndim=1,mode="c"] initial_plan = None, bint gaussian = False,int seed = 0):
    """
    
    Builds nb_plans BSP Matchings and merges them in a single bijection.
        
        cost,plan,plans = bsp_solve(X,Y,n_plans)

    where :

    - X and Y are the input point clouds
    - n_plans is the number of BSP Matchings used to compute the final bijection
    - cost_name is ground metric name, by default "sqnorm"
    - initial_plan bijection to use for initializing merging (optional)

    Returns the transport cost of the final bijection, the final bijection, and the intermediary ones

    """

    if not X.flags['C_CONTIGUOUS']:
        X = np.ascontiguousarray(X, dtype=np.float64)
        print("not contiugous")
    if not Y.flags['C_CONTIGUOUS']:
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        print("not contiugous")
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef np.ndarray[int, ndim=2, mode="c"] plans = np.zeros((n_plans,n), dtype=np.int32) 
    cdef np.ndarray[int, ndim=1, mode="c"] plan = np.zeros(n, dtype=np.int32) 

    cdef bool gauss = gaussian

    cdef double cost
    
    if initial_plan is None:
        cost = BSPOT_wrap(n, d, <double*>X.data, <double*>Y.data, n_plans, <int*> plans.data, <int*> plan.data,lp_power, NULL,gauss, seed)
    else:
        cost = BSPOT_wrap(n, d, <double*>X.data, <double*>Y.data, n_plans, <int*> plans.data, <int*> plan.data,lp_power, <int*>initial_plan.data,gauss,seed)

    return cost,plan, plans
    

@cython.boundscheck(False)
@cython.wraparound(False)
def merge_bijections_c(np.ndarray[double, ndim=2, mode="c"] X, np.ndarray[double, ndim=2, mode="c"] Y,  np.ndarray[int, ndim=2, mode="c"] plans,int lp_power=2):
    """
        Merges transport bijections

    where :

    - X and Y are the input point clouds
    - plans input bijections
    - metric name, by default "sqnorm"

    Returns the merged bijection and its transport cost.
    """
    
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef int k = plans.shape[0]

    if plans.shape[1] != n:
        raise ValueError(
            "plans must have shape (n_plans, n_samples) with one entry per point")

    if k == 0:
        raise ValueError("plans must contain at least one plan")

    # C++ indexes the point clouds with these entries, so a stack whose rows are not a
    # permutation of 0..n-1 must not reach BijectiveMatching. Counting is linear in the
    # size of the stack and does not copy it.
    cdef np.ndarray[int, ndim=1, mode="c"] seen = np.zeros(n, dtype=np.int32)
    cdef int i
    cdef int j
    cdef int v

    for i in range(k):
        for j in range(n):
            seen[j] = 0
        for j in range(n):
            v = plans[i, j]
            if v < 0 or v >= n or seen[v] != 0:
                raise ValueError(
                    "each row of plans must be a permutation of 0..n_samples-1")
            seen[v] = 1

    cdef np.ndarray[int, ndim=1, mode="c"] plan = np.zeros(n, dtype=np.int32) 

    cdef double cost_val

    # add merging code here
    cost_val = MergeBijections(n, d, <double*>X.data, <double*>Y.data, k, <int*> plans.data, <int*> plan.data, lp_power)


    return cost_val, plan


    
