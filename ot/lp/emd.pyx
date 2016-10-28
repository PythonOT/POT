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
def emd_c( np.ndarray[double, ndim=1, mode="c"] a,np.ndarray[double, ndim=1, mode="c"]  b,np.ndarray[double, ndim=2, mode="c"]  M):
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
  
    
    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
 
    """
    cdef int n1= M.shape[0]
    cdef int n2= M.shape[1]

    cdef float cost=0
    cdef np.ndarray[double, ndim=2, mode="c"] G=np.zeros([n1, n2])
    
    if not len(a):
        a=np.ones((n1,))/n1

    if not len(b):
        b=np.ones((n2,))/n2

    # calling the function
    EMD_wrap(n1,n2,<double*> a.data,<double*> b.data,<double*> M.data,<double*> G.data,<double*> &cost)

    return G
