# -*- coding: utf-8 -*-
"""
Solvers for the original linear program OT problem
"""

import numpy as np
# import compiled emd
from .emd import emd_c
import multiprocessing

def emd(a, b, M):
    """Solves the Earth Movers distance problem and returns the OT matrix


    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F

        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :

    - M is the metric cost matrix
    - a and b are the sample weights

    Uses the algorithm proposed in [1]_

    Parameters
    ----------
    a : (ns,) ndarray, float64
        Source histogram (uniform weigth if empty list)
    b : (nt,) ndarray, float64
        Target histogram (uniform weigth if empty list)
    M : (ns,nt) ndarray, float64
        loss matrix

    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters


    Examples
    --------

    Simple example with obvious solution. The function emd accepts lists and
    perform automatic conversion to numpy arrays
    >>> import ot
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ot.emd(a,b,M)
    array([[ 0.5,  0. ],
           [ 0. ,  0.5]])

    References
    ----------

    .. [1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W.
        (2011, December).  Displacement interpolation using Lagrangian mass
        transport. In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p.
        158). ACM.

    See Also
    --------
    ot.bregman.sinkhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT"""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # if empty array given then use unifor distributions
    if len(a) == 0:
        a = np.ones((M.shape[0], ), dtype=np.float64)/M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1], ), dtype=np.float64)/M.shape[1]

    return emd_c(a, b, M)

def emd2(a, b, M,processes=multiprocessing.cpu_count()):
    """Solves the Earth Movers distance problem and returns the loss 

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F

        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :

    - M is the metric cost matrix
    - a and b are the sample weights

    Uses the algorithm proposed in [1]_

    Parameters
    ----------
    a : (ns,) ndarray, float64
        Source histogram (uniform weigth if empty list)
    b : (nt,) ndarray, float64
        Target histogram (uniform weigth if empty list)
    M : (ns,nt) ndarray, float64
        loss matrix

    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters


    Examples
    --------

    Simple example with obvious solution. The function emd accepts lists and
    perform automatic conversion to numpy arrays
    >>> import ot
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ot.emd2(a,b,M)
    0.0
    
    References
    ----------

    .. [1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W.
        (2011, December).  Displacement interpolation using Lagrangian mass
        transport. In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p.
        158). ACM.

    See Also
    --------
    ot.bregman.sinkhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT"""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # if empty array given then use unifor distributions
    if len(a) == 0:
        a = np.ones((M.shape[0], ), dtype=np.float64)/M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1], ), dtype=np.float64)/M.shape[1]
        
    if len(b.shape)==1:
        return np.sum(emd_c(a, b, M)*M)
    else:
        nb=b.shape[1]
        ls=[(a,b[:,k],M) for k in range(nb)]
        def f(l):
            return emd2(l[0],l[1],l[2])
        # run emd in multiprocessing
        res=parmap(f, ls,processes)
        return np.array(res)
#        with Pool(processes) as p:
#            res=p.map(f, ls)
#            return np.array(res)
        

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=multiprocessing.cpu_count()):
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