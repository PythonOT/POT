# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 08:42:08 2014

@author: rflamary
"""
import numpy as np
cimport numpy as np

cimport cython



cdef extern from "EMD.h":
    void EMD_wrap(int n1,int n2, double *X, double *Y,double *D, double *G, double *cost)



@cython.boundscheck(False)
@cython.wraparound(False)
def emd( np.ndarray[double, ndim=1, mode="c"] a,np.ndarray[double, ndim=1, mode="c"]  b,np.ndarray[double, ndim=2, mode="c"]  M):
    """
        Solves the Earth Movers distance problem and returns the optimal transport matrix
        
        
 
        :param a: m weights of the source distribution (must sum to one)
        :param b: n weights of the target distribution (must sum to one)
        :param M: m x n cost matrix
        :type a: np.ndarray
        :type b: np.ndarray
        :type M: np.ndarray
        :return: Optimal transport matrix
        :rtype: np.ndarray
 
 
    """
    cdef int n1= M.shape[0]
    cdef int n2= M.shape[1]

    cdef float cost=0
    cdef np.ndarray[double, ndim=2, mode="c"] G=np.zeros([n1, n2])

    # calling the function
    EMD_wrap(n1,n2,<double*> a.data,<double*> b.data,<double*> M.data,<double*> G.data,<double*> &cost)

    return G
