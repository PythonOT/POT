# -*- coding: utf-8 -*-
"""
Solvers for the original linear program OT problem
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import multiprocessing

import numpy as np

# import compiled emd
from .emd_wrap import emd_c, check_result
from ..utils import parmap


def emd(a, b, M, numItermax=100000, log=False):
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
    numItermax : int, optional (default=100000)
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.
    log: boolean, optional (default=False)
        If True, returns a dictionary containing the cost and dual
        variables. Otherwise returns only the optimal transportation matrix.

    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log: dict
        If input log is true, a dictionary containing the cost and dual
        variables and exit status


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
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    G, cost, u, v, result_code = emd_c(a, b, M, numItermax)
    result_code_string = check_result(result_code)
    if log:
        log = {}
        log['cost'] = cost
        log['u'] = u
        log['v'] = v
        log['warning'] = result_code_string
        log['result_code'] = result_code
        return G, log
    return G


def emd2(a, b, M, processes=multiprocessing.cpu_count(), numItermax=100000, log=False, return_matrix=False):
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
    numItermax : int, optional (default=100000)
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.
    log: boolean, optional (default=False)
        If True, returns a dictionary containing the cost and dual
        variables. Otherwise returns only the optimal transportation cost.
    return_matrix: boolean, optional (default=False)
        If True, returns the optimal transportation matrix in the log.

    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log: dict
        If input log is true, a dictionary containing the cost and dual
        variables and exit status


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
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    if log or return_matrix:
        def f(b):
            G, cost, u, v, resultCode = emd_c(a, b, M, numItermax)
            result_code_string = check_result(resultCode)
            log = {}
            if return_matrix:
                log['G'] = G
            log['u'] = u
            log['v'] = v
            log['warning'] = result_code_string
            log['result_code'] = resultCode
            return [cost, log]
    else:
        def f(b):
            G, cost, u, v, result_code = emd_c(a, b, M, numItermax)
            check_result(result_code)
            return cost

    if len(b.shape) == 1:
        return f(b)
    nb = b.shape[1]

    res = parmap(f, [b[:, i] for i in range(nb)], processes)
    return res
