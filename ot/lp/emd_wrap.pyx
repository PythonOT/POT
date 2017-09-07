# -*- coding: utf-8 -*-
"""
Cython linker with C solver
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import warnings
import numpy as np
cimport numpy as np

cimport cython



cdef extern from "EMD.h":
    int EMD_wrap(int n1,int n2, double *X, double *Y,double *D, double *G, double* alpha, double* beta, double *cost, int numItermax)
    cdef enum ProblemType: INFEASIBLE, OPTIMAL, UNBOUNDED, MAX_ITER_REACHED



@cython.boundscheck(False)
@cython.wraparound(False)
def emd_c( np.ndarray[double, ndim=1, mode="c"] a,np.ndarray[double, ndim=1, mode="c"]  b,np.ndarray[double, ndim=2, mode="c"]  M, int numItermax):
    """
        Solves the Earth Movers distance problem and returns the optimal transport matrix

        gamm=emd(a,b,M)

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the metric cost matrix
    - a and b are the sample weights

    Parameters
    ----------
    a : (ns,) ndarray, float64
        source histogram
    b : (nt,) ndarray, float64
        target histogram
    M : (ns,nt) ndarray, float64
        loss matrix
    numItermax : int
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.


    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters

    """
    cdef int n1= M.shape[0]
    cdef int n2= M.shape[1]

    cdef double cost=0
    cdef np.ndarray[double, ndim=2, mode="c"] G=np.zeros([n1, n2])
    cdef np.ndarray[double, ndim=1, mode="c"] alpha=np.zeros(n1)
    cdef np.ndarray[double, ndim=1, mode="c"] beta=np.zeros(n2)


    if not len(a):
        a=np.ones((n1,))/n1

    if not len(b):
        b=np.ones((n2,))/n2

    # calling the function
    cdef int resultSolver = EMD_wrap(n1,n2,<double*> a.data,<double*> b.data,<double*> M.data,<double*> G.data, <double*> alpha.data, <double*> beta.data, <double*> &cost, numItermax)
    if resultSolver != OPTIMAL:
        if resultSolver == INFEASIBLE:
            warnings.warn("Problem infeasible. Check that a and b are in the simplex")
        elif resultSolver == UNBOUNDED:
            warnings.warn("Problem unbounded")
        elif resultSolver == MAX_ITER_REACHED:
            warnings.warn("numItermax reached before optimality. Try to increase numItermax.")

    return G, cost, alpha, beta
