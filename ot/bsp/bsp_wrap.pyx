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
    double BSPOT_wrap(int n, int d, double *X, double *Y, uint64_t nb_plans, int *plans, int *plan,const char* cost_name,int* initial_plan)
    double MergeBijections(int n, int d, double *X, double *Y, uint64_t nb_plans, int *plans, int *plan,const char* cost_name)


@cython.boundscheck(False)
@cython.wraparound(False)
def bsp_solve(np.ndarray[double, ndim=2, mode="c"] X, np.ndarray[double, ndim=2, mode="c"] Y,  int n_plans=64,str cost_name="sqnorm",np.ndarray[int,ndim=1,mode="c"] initial_plan = None):
    """
    
    Builds nb_plans BSP Matchings and merges them in a single bijection.
        
        cost,plan,plans = bsp_solve(X,Y,n_plans)

    where :

    - X and Y are the input point clouds
    - n_plans is the number of BSP Matchings used to compute the final bijection

    Returns the transport cost of the final bijection, the final bijection, and the intermediary ones

    """
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef np.ndarray[int, ndim=2, mode="c"] plans = np.zeros((n, n_plans), dtype=np.int32) 
    cdef np.ndarray[int, ndim=1, mode="c"] plan = np.zeros(n, dtype=np.int32) 

    cdef double cost
    
    cdef bytes cost_bytes = cost_name.encode("utf-8")
    cdef const char* cost_c = cost_bytes

    if initial_plan is None:
        cost = BSPOT_wrap(n, d, <double*>X.data, <double*>Y.data, n_plans, <int*> plans.data, <int*> plan.data,cost_c, NULL)
    else:
        cost = BSPOT_wrap(n, d, <double*>X.data, <double*>Y.data, n_plans, <int*> plans.data, <int*> plan.data,cost_c, <int*>initial_plan.data)

    # add 

    return cost, plan, plans
    

@cython.boundscheck(False)
@cython.wraparound(False)
def merge_bijections(np.ndarray[double, ndim=2, mode="c"] X, np.ndarray[double, ndim=2, mode="c"] Y,  np.ndarray[int, ndim=2, mode="c"] plans,str cost = "sqnorm"):
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
    cdef int k = plans.shape[1]
    cdef np.ndarray[int, ndim=1, mode="c"] plan = np.zeros(n, dtype=np.int32) 

    cdef double cost_val
    
    cdef bytes cost_bytes = cost.encode("utf-8")
    cdef const char* cost_c = cost_bytes

    # add merging code here
    
    cost_val = MergeBijections(n, d, <double*>X.data, <double*>Y.data, k, <int*> plans.data, <int*> plan.data,cost_c)


    return cost_val,plan


    