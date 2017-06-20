# -*- coding: utf-8 -*-
"""
Various function that can be usefull
"""
import numpy as np
from scipy.spatial.distance import cdist
import multiprocessing

import time
__time_tic_toc=time.time()

def tic():
    """ Python implementation of Matlab tic() function """
    global __time_tic_toc
    __time_tic_toc=time.time()

def toc(message='Elapsed time : {} s'):
    """ Python implementation of Matlab toc() function """
    t=time.time()
    print(message.format(t-__time_tic_toc))
    return t-__time_tic_toc

def toq():
    """ Python implementation of Julia toc() function """
    t=time.time()
    return t-__time_tic_toc


def kernel(x1,x2,method='gaussian',sigma=1,**kwargs):
    """Compute kernel matrix"""
    if method.lower() in ['gaussian','gauss','rbf']:
        K=np.exp(-dist(x1,x2)/(2*sigma**2))
    return K

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

def clean_zeros(a,b,M):
    """ Remove all components with zeros weights in a and b 
    """
    M2=M[a>0,:][:,b>0].copy() # copy force c style matrix (froemd)
    a2=a[a>0]
    b2=b[b>0]
    return a2,b2,M2

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
        the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.


    Returns
    -------

    M : np.array (n1,n2)
        distance matrix computed with given metric

    """
    if x2 is None:
        x2=x1

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

def fun(f, q_in, q_out):
    """ Utility function for parmap with no serializing problems """
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    """ paralell map for multiprocessing """
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]    