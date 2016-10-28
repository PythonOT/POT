# -*- coding: utf-8 -*-
"""
Various function that can be usefull
"""
import numpy as np
from scipy.spatial.distance import cdist


def unif(n):
    """ return a uniform histogram of length n (simplex) 
    
    Parameters
    ----------

    n : int
        number of bins in the histogram
  
    Returns
    -------
    h : np.array (n,)
        histogram of length n such that h_i=1/n for all i    
    
    
    """
    return np.ones((n,))/n


def dist(x1,x2=None,metric='sqeuclidean'):
    """Compute distance between samples in x1 and x2 using function scipy.spatial.distance.cdist
    
    Parameters
    ----------

    x1 : np.array (n1,d)
        matrix with n1 samples of size d
    x2 : np.array (n2,d), optional
        matrix with n2 samples of size d (if None then x2=x1)
    metric : str, fun, optional
        name of the metric to be computed (full list in the doc of scipy),  If a string, 
        the distance function can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’,
        ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’,
        ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
        ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’.

  
    Returns
    -------
    
    M : np.array (n1,n2)
        distance matrix computed with given metric
    
    """
    if x2 is None:
        return cdist(x1,x1,metric=metric)
    else:
        return cdist(x1,x2,metric=metric)  
        
def dist0(n,method='lin_square'):
    """Compute standard cost matrices of size (n,n) for OT problems
    
    Parameters
    ----------

    n : int
        size of the cost matrix
    method : str, optional
        Type of loss matrix chosen from:

        * 'lin_square' : linear sampling between 0 and n-1, quadratic loss

  
    Returns
    -------
    
    M : np.array (n1,n2)
        distance matrix computed with given metric    
    
    
    """
    res=0
    if method=='lin_square':
        x=np.arange(n,dtype=np.float64).reshape((n,1))
        res=dist(x,x)
    return res
    

def dots(*args):
    """ dots function for multiple matrix multiply """
    return reduce(np.dot,args)