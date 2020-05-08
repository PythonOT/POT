# -*- coding: utf-8 -*-
"""
Solvers for the original linear program OT problem

"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import multiprocessing
import sys

import numpy as np
from scipy.sparse import coo_matrix

from . import cvx
from .cvx import barycenter
# import compiled emd
from .emd_wrap import emd_c, check_result, emd_1d_sorted
from ..utils import dist
from ..utils import parmap

__all__ = ['emd', 'emd2', 'barycenter', 'free_support_barycenter', 'cvx',
           'emd_1d', 'emd2_1d', 'wasserstein_1d']


def center_ot_dual(alpha0, beta0, a=None, b=None):
    r"""Center dual OT potentials w.r.t. theirs weights

    The main idea of this function is to find unique dual potentials
    that ensure some kind of centering/fairness. The main idea is to find dual potentials that lead to the same final objective value for both source and targets (see below for more details). It will help having
    stability when multiple calling of the OT solver with small changes.

    Basically we add another constraint to the potential that will not
    change the objective value but will ensure unicity. The constraint
    is the following:

    .. math::
        \alpha^T a= \beta^T b

    in addition to the OT problem constraints.

    since :math:`\sum_i a_i=\sum_j b_j` this can be solved by adding/removing
    a constant from both  :math:`\alpha_0` and :math:`\beta_0`.

    .. math::
        c=\frac{\beta0^T b-\alpha_0^T a}{1^Tb+1^Ta}

        \alpha=\alpha_0+c

        \beta=\beta0+c

    Parameters
    ----------
    alpha0 : (ns,) numpy.ndarray, float64
        Source dual potential
    beta0 : (nt,) numpy.ndarray, float64
        Target dual potential
    a : (ns,) numpy.ndarray, float64
        Source histogram (uniform weight if empty list)
    b : (nt,) numpy.ndarray, float64
        Target histogram (uniform weight if empty list)

    Returns
    -------
    alpha : (ns,) numpy.ndarray, float64
        Source centered dual potential
    beta : (nt,) numpy.ndarray, float64
        Target centered dual potential

    """
    # if no weights are provided, use uniform
    if a is None:
        a = np.ones(alpha0.shape[0]) / alpha0.shape[0]
    if b is None:
        b = np.ones(beta0.shape[0]) / beta0.shape[0]

    # compute constant that balances the weighted sums of the duals
    c = (b.dot(beta0) - a.dot(alpha0)) / (a.sum() + b.sum())

    # update duals
    alpha = alpha0 + c
    beta = beta0 - c

    return alpha, beta


def estimate_dual_null_weights(alpha0, beta0, a, b, M):
    r"""Estimate feasible values for 0-weighted dual potentials

    The feasible values are computed efficiently but rather coarsely.

    .. warning::
        This function is necessary because the C++ solver in emd_c
        discards all samples in the distributions with 
        zeros weights. This means that while the primal variable (transport 
        matrix) is exact, the solver only returns feasible dual potentials
        on the samples with weights different from zero. 

    First we compute the constraints violations:

    .. math::
        V=\alpha+\beta^T-M

    Next we compute the max amount of violation per row (alpha) and
    columns (beta)

    .. math::
        v^a_i=\max_j V_{i,j}

        v^b_j=\max_i V_{i,j}

    Finally we update the dual potential with 0 weights if a
    constraint is violated

    .. math::
        \alpha_i = \alpha_i -v^a_i \quad \text{ if } a_i=0 \text{ and } v^a_i>0

        \beta_j = \beta_j -v^b_j \quad \text{ if } b_j=0 \text{ and } v^b_j>0

    In the end the dual potentials are centered using function
    :ref:`center_ot_dual`.

    Note that all those updates do not change the objective value of the
    solution but provide dual potentials that do not violate the constraints.

    Parameters
    ----------
    alpha0 : (ns,) numpy.ndarray, float64
        Source dual potential
    beta0 : (nt,) numpy.ndarray, float64
        Target dual potential
    alpha0 : (ns,) numpy.ndarray, float64
        Source dual potential
    beta0 : (nt,) numpy.ndarray, float64
        Target dual potential
    a : (ns,) numpy.ndarray, float64
        Source distribution (uniform weights if empty list)
    b : (nt,) numpy.ndarray, float64
        Target distribution (uniform weights if empty list)
    M : (ns,nt) numpy.ndarray, float64
        Loss matrix (c-order array with type float64)

    Returns
    -------
    alpha : (ns,) numpy.ndarray, float64
        Source corrected dual potential
    beta : (nt,) numpy.ndarray, float64
        Target corrected dual potential

    """

    # binary indexing of non-zeros weights
    asel = a != 0
    bsel = b != 0

    # compute dual constraints violation
    constraint_violation = alpha0[:, None] + beta0[None, :] - M

    # Compute largest violation per line and columns
    aviol = np.max(constraint_violation, 1)
    bviol = np.max(constraint_violation, 0)

    # update corrects violation of
    alpha_up = -1 * ~asel * np.maximum(aviol, 0)
    beta_up = -1 * ~bsel * np.maximum(bviol, 0)

    alpha = alpha0 + alpha_up
    beta = beta0 + beta_up

    return center_ot_dual(alpha, beta, a, b)


def emd(a, b, M, numItermax=100000, log=False, center_dual=True):
    r"""Solves the Earth Movers distance problem and returns the OT matrix


    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the metric cost matrix
    - a and b are the sample weights

    .. warning::
        Note that the M matrix needs to be a C-order numpy.array in float64
        format.

    Uses the algorithm proposed in [1]_

    Parameters
    ----------
    a : (ns,) numpy.ndarray, float64
        Source histogram (uniform weight if empty list)
    b : (nt,) numpy.ndarray, float64
        Target histogram (uniform weight if empty list)
    M : (ns,nt) numpy.ndarray, float64
        Loss matrix (c-order array with type float64)
    numItermax : int, optional (default=100000)
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.
    log: bool, optional (default=False)
        If True, returns a dictionary containing the cost and dual
        variables. Otherwise returns only the optimal transportation matrix.
    center_dual: boolean, optional (default=True)
        If True, centers the dual potential using function
        :ref:`center_ot_dual`.

    Returns
    -------
    gamma: (ns x nt) numpy.ndarray
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
    array([[0.5, 0. ],
           [0. , 0.5]])

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

    # if empty array given then use uniform distributions
    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    assert (a.shape[0] == M.shape[0] and b.shape[0] == M.shape[1]), \
        "Dimension mismatch, check dimensions of M with a and b"

    asel = a != 0
    bsel = b != 0

    G, cost, u, v, result_code = emd_c(a, b, M, numItermax)

    if center_dual:
        u, v = center_ot_dual(u, v, a, b)

    if np.any(~asel) or np.any(~bsel):
        u, v = estimate_dual_null_weights(u, v, a, b, M)
    
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


def emd2(a, b, M, processes=multiprocessing.cpu_count(),
         numItermax=100000, log=False, return_matrix=False,
         center_dual=True):
    r"""Solves the Earth Movers distance problem and returns the loss

    .. math::
        \min_\gamma <\gamma,M>_F

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the metric cost matrix
    - a and b are the sample weights

    .. warning::
        Note that the M matrix needs to be a C-order numpy.array in float64
        format.

    Uses the algorithm proposed in [1]_

    Parameters
    ----------
    a : (ns,) numpy.ndarray, float64
        Source histogram (uniform weight if empty list)
    b : (nt,) numpy.ndarray, float64
        Target histogram (uniform weight if empty list)
    M : (ns,nt) numpy.ndarray, float64
        Loss matrix (c-order array with type float64)
    processes : int, optional (default=nb cpu)
        Nb of processes used for multiple emd computation (not used on windows)
    numItermax : int, optional (default=100000)
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.
    log: boolean, optional (default=False)
        If True, returns a dictionary containing the cost and dual
        variables. Otherwise returns only the optimal transportation cost.
    return_matrix: boolean, optional (default=False)
        If True, returns the optimal transportation matrix in the log.
    center_dual: boolean, optional (default=True)
        If True, centers the dual potential using function
        :ref:`center_ot_dual`.

    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log: dictnp
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

    # problem with pikling Forks
    if sys.platform.endswith('win32'):
        processes = 1

    # if empty array given then use uniform distributions
    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    assert (a.shape[0] == M.shape[0] and b.shape[0] == M.shape[1]), \
        "Dimension mismatch, check dimensions of M with a and b"

    asel = a != 0

    if log or return_matrix:
        def f(b):
            bsel = b != 0
            
            G, cost, u, v, result_code = emd_c(a, b, M, numItermax)

            if center_dual:
                u, v = center_ot_dual(u, v, a, b)

            if np.any(~asel) or np.any(~bsel):
                u, v = estimate_dual_null_weights(u, v, a, b, M)

            result_code_string = check_result(result_code)
            log = {}
            if return_matrix:
                log['G'] = G
            log['u'] = u
            log['v'] = v
            log['warning'] = result_code_string
            log['result_code'] = result_code
            return [cost, log]
    else:
        def f(b):
            bsel = b != 0
            G, cost, u, v, result_code = emd_c(a, b, M, numItermax)

            if center_dual:
                u, v = center_ot_dual(u, v, a, b)

            if np.any(~asel) or np.any(~bsel):
                u, v = estimate_dual_null_weights(u, v, a, b, M)

            check_result(result_code)
            return cost

    if len(b.shape) == 1:
        return f(b)
    nb = b.shape[1]

    if processes > 1:
        res = parmap(f, [b[:, i] for i in range(nb)], processes)
    else:
        res = list(map(f, [b[:, i].copy() for i in range(nb)]))

    return res


def free_support_barycenter(measures_locations, measures_weights, X_init, b=None, weights=None, numItermax=100,
                            stopThr=1e-7, verbose=False, log=None):
    """
    Solves the free support (locations of the barycenters are optimized, not the weights) Wasserstein barycenter problem (i.e. the weighted Frechet mean for the 2-Wasserstein distance)

    The function solves the Wasserstein barycenter problem when the barycenter measure is constrained to be supported on k atoms.
    This problem is considered in [1] (Algorithm 2). There are two differences with the following codes:
    - we do not optimize over the weights
    - we do not do line search for the locations updates, we use i.e. theta = 1 in [1] (Algorithm 2). This can be seen as a discrete implementation of the fixed-point algorithm of [2] proposed in the continuous setting.

    Parameters
    ----------
    measures_locations : list of (k_i,d) numpy.ndarray
        The discrete support of a measure supported on k_i locations of a d-dimensional space (k_i can be different for each element of the list)
    measures_weights : list of (k_i,) numpy.ndarray
        Numpy arrays where each numpy array has k_i non-negatives values summing to one representing the weights of each discrete input measure

    X_init : (k,d) np.ndarray
        Initialization of the support locations (on k atoms) of the barycenter
    b : (k,) np.ndarray
        Initialization of the weights of the barycenter (non-negatives, sum to 1)
    weights : (k,) np.ndarray
        Initialization of the coefficients of the barycenter (non-negatives, sum to 1)

    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    X : (k,d) np.ndarray
        Support locations (on k atoms) of the barycenter

    References
    ----------

    .. [1] Cuturi, Marco, and Arnaud Doucet. "Fast computation of Wasserstein barycenters." International Conference on Machine Learning. 2014.

    .. [2]  Álvarez-Esteban, Pedro C., et al. "A fixed-point approach to barycenters in Wasserstein space." Journal of Mathematical Analysis and Applications 441.2 (2016): 744-762.

    """

    iter_count = 0

    N = len(measures_locations)
    k = X_init.shape[0]
    d = X_init.shape[1]
    if b is None:
        b = np.ones((k,)) / k
    if weights is None:
        weights = np.ones((N,)) / N

    X = X_init

    log_dict = {}
    displacement_square_norms = []

    displacement_square_norm = stopThr + 1.

    while (displacement_square_norm > stopThr and iter_count < numItermax):

        T_sum = np.zeros((k, d))

        for (measure_locations_i, measure_weights_i, weight_i) in zip(measures_locations, measures_weights,
                                                                      weights.tolist()):
            M_i = dist(X, measure_locations_i)
            T_i = emd(b, measure_weights_i, M_i)
            T_sum = T_sum + weight_i * np.reshape(1. / b, (-1, 1)) * np.matmul(T_i, measure_locations_i)

        displacement_square_norm = np.sum(np.square(T_sum - X))
        if log:
            displacement_square_norms.append(displacement_square_norm)

        X = T_sum

        if verbose:
            print('iteration %d, displacement_square_norm=%f\n', iter_count, displacement_square_norm)

        iter_count += 1

    if log:
        log_dict['displacement_square_norms'] = displacement_square_norms
        return X, log_dict
    else:
        return X


def emd_1d(x_a, x_b, a=None, b=None, metric='sqeuclidean', p=1., dense=True,
           log=False):
    r"""Solves the Earth Movers distance problem between 1d measures and returns
    the OT matrix


    .. math::
        \gamma = arg\min_\gamma \sum_i \sum_j \gamma_{ij} d(x_a[i], x_b[j])

        s.t. \gamma 1 = a,
             \gamma^T 1= b,
             \gamma\geq 0
    where :

    - d is the metric
    - x_a and x_b are the samples
    - a and b are the sample weights

    When 'minkowski' is used as a metric, :math:`d(x, y) = |x - y|^p`.

    Uses the algorithm detailed in [1]_

    Parameters
    ----------
    x_a : (ns,) or (ns, 1) ndarray, float64
        Source dirac locations (on the real line)
    x_b : (nt,) or (ns, 1) ndarray, float64
        Target dirac locations (on the real line)
    a : (ns,) ndarray, float64, optional
        Source histogram (default is uniform weight)
    b : (nt,) ndarray, float64, optional
        Target histogram (default is uniform weight)
    metric: str, optional (default='sqeuclidean')
        Metric to be used. Only strings listed in :func:`ot.dist` are accepted.
        Due to implementation details, this function runs faster when
        `'sqeuclidean'`, `'cityblock'`,  or `'euclidean'` metrics are used.
    p: float, optional (default=1.0)
         The p-norm to apply for if metric='minkowski'
    dense: boolean, optional (default=True)
        If True, returns math:`\gamma` as a dense ndarray of shape (ns, nt).
        Otherwise returns a sparse representation using scipy's `coo_matrix`
        format. Due to implementation details, this function runs faster when
        `'sqeuclidean'`, `'minkowski'`, `'cityblock'`,  or `'euclidean'` metrics
        are used.
    log: boolean, optional (default=False)
        If True, returns a dictionary containing the cost.
        Otherwise returns only the optimal transportation matrix.

    Returns
    -------
    gamma: (ns, nt) ndarray
        Optimal transportation matrix for the given parameters
    log: dict
        If input log is True, a dictionary containing the cost


    Examples
    --------

    Simple example with obvious solution. The function emd_1d accepts lists and
    performs automatic conversion to numpy arrays

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> x_a = [2., 0.]
    >>> x_b = [0., 3.]
    >>> ot.emd_1d(x_a, x_b, a, b)
    array([[0. , 0.5],
           [0.5, 0. ]])
    >>> ot.emd_1d(x_a, x_b)
    array([[0. , 0.5],
           [0.5, 0. ]])

    References
    ----------

    .. [1]  Peyré, G., & Cuturi, M. (2017). "Computational Optimal
        Transport", 2018.

    See Also
    --------
    ot.lp.emd : EMD for multidimensional distributions
    ot.lp.emd2_1d : EMD for 1d distributions (returns cost instead of the
        transportation matrix)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x_a = np.asarray(x_a, dtype=np.float64)
    x_b = np.asarray(x_b, dtype=np.float64)

    assert (x_a.ndim == 1 or x_a.ndim == 2 and x_a.shape[1] == 1), \
        "emd_1d should only be used with monodimensional data"
    assert (x_b.ndim == 1 or x_b.ndim == 2 and x_b.shape[1] == 1), \
        "emd_1d should only be used with monodimensional data"

    # if empty array given then use uniform distributions
    if a.ndim == 0 or len(a) == 0:
        a = np.ones((x_a.shape[0],), dtype=np.float64) / x_a.shape[0]
    if b.ndim == 0 or len(b) == 0:
        b = np.ones((x_b.shape[0],), dtype=np.float64) / x_b.shape[0]

    x_a_1d = x_a.reshape((-1,))
    x_b_1d = x_b.reshape((-1,))
    perm_a = np.argsort(x_a_1d)
    perm_b = np.argsort(x_b_1d)

    G_sorted, indices, cost = emd_1d_sorted(a[perm_a], b[perm_b],
                                            x_a_1d[perm_a], x_b_1d[perm_b],
                                            metric=metric, p=p)
    G = coo_matrix((G_sorted, (perm_a[indices[:, 0]], perm_b[indices[:, 1]])),
                   shape=(a.shape[0], b.shape[0]))
    if dense:
        G = G.toarray()
    if log:
        log = {'cost': cost}
        return G, log
    return G


def emd2_1d(x_a, x_b, a=None, b=None, metric='sqeuclidean', p=1., dense=True,
            log=False):
    r"""Solves the Earth Movers distance problem between 1d measures and returns
    the loss


    .. math::
        \gamma = arg\min_\gamma \sum_i \sum_j \gamma_{ij} d(x_a[i], x_b[j])

        s.t. \gamma 1 = a,
             \gamma^T 1= b,
             \gamma\geq 0
    where :

    - d is the metric
    - x_a and x_b are the samples
    - a and b are the sample weights

    When 'minkowski' is used as a metric, :math:`d(x, y) = |x - y|^p`.

    Uses the algorithm detailed in [1]_

    Parameters
    ----------
    x_a : (ns,) or (ns, 1) ndarray, float64
        Source dirac locations (on the real line)
    x_b : (nt,) or (ns, 1) ndarray, float64
        Target dirac locations (on the real line)
    a : (ns,) ndarray, float64, optional
        Source histogram (default is uniform weight)
    b : (nt,) ndarray, float64, optional
        Target histogram (default is uniform weight)
    metric: str, optional (default='sqeuclidean')
        Metric to be used. Only strings listed in :func:`ot.dist` are accepted.
        Due to implementation details, this function runs faster when
        `'sqeuclidean'`, `'minkowski'`, `'cityblock'`,  or `'euclidean'` metrics
        are used.
    p: float, optional (default=1.0)
         The p-norm to apply for if metric='minkowski'
    dense: boolean, optional (default=True)
        If True, returns math:`\gamma` as a dense ndarray of shape (ns, nt).
        Otherwise returns a sparse representation using scipy's `coo_matrix`
        format. Only used if log is set to True. Due to implementation details,
        this function runs faster when dense is set to False.
    log: boolean, optional (default=False)
        If True, returns a dictionary containing the transportation matrix.
        Otherwise returns only the loss.

    Returns
    -------
    loss: float
        Cost associated to the optimal transportation
    log: dict
        If input log is True, a dictionary containing the Optimal transportation
        matrix for the given parameters


    Examples
    --------

    Simple example with obvious solution. The function emd2_1d accepts lists and
    performs automatic conversion to numpy arrays

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> x_a = [2., 0.]
    >>> x_b = [0., 3.]
    >>> ot.emd2_1d(x_a, x_b, a, b)
    0.5
    >>> ot.emd2_1d(x_a, x_b)
    0.5

    References
    ----------

    .. [1]  Peyré, G., & Cuturi, M. (2017). "Computational Optimal
        Transport", 2018.

    See Also
    --------
    ot.lp.emd2 : EMD for multidimensional distributions
    ot.lp.emd_1d : EMD for 1d distributions (returns the transportation matrix
        instead of the cost)
    """
    # If we do not return G (log==False), then we should not to cast it to dense
    # (useless overhead)
    G, log_emd = emd_1d(x_a=x_a, x_b=x_b, a=a, b=b, metric=metric, p=p,
                        dense=dense and log, log=True)
    cost = log_emd['cost']
    if log:
        log_emd = {'G': G}
        return cost, log_emd
    return cost


def wasserstein_1d(x_a, x_b, a=None, b=None, p=1.):
    r"""Solves the p-Wasserstein distance problem between 1d measures and returns
    the distance

    .. math::
        \min_\gamma \left( \sum_i \sum_j \gamma_{ij} \|x_a[i] - x_b[j]\|^p \right)^{1/p}

        s.t. \gamma 1 = a,
             \gamma^T 1= b,
             \gamma\geq 0

    where :

    - x_a and x_b are the samples
    - a and b are the sample weights

    Uses the algorithm detailed in [1]_

    Parameters
    ----------
    x_a : (ns,) or (ns, 1) ndarray, float64
        Source dirac locations (on the real line)
    x_b : (nt,) or (ns, 1) ndarray, float64
        Target dirac locations (on the real line)
    a : (ns,) ndarray, float64, optional
        Source histogram (default is uniform weight)
    b : (nt,) ndarray, float64, optional
        Target histogram (default is uniform weight)
    p: float, optional (default=1.0)
         The order of the p-Wasserstein distance to be computed

    Returns
    -------
    dist: float
        p-Wasserstein distance


    Examples
    --------

    Simple example with obvious solution. The function wasserstein_1d accepts
    lists and performs automatic conversion to numpy arrays

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> x_a = [2., 0.]
    >>> x_b = [0., 3.]
    >>> ot.wasserstein_1d(x_a, x_b, a, b)
    0.5
    >>> ot.wasserstein_1d(x_a, x_b)
    0.5

    References
    ----------

    .. [1]  Peyré, G., & Cuturi, M. (2017). "Computational Optimal
        Transport", 2018.

    See Also
    --------
    ot.lp.emd_1d : EMD for 1d distributions
    """
    cost_emd = emd2_1d(x_a=x_a, x_b=x_b, a=a, b=b, metric='minkowski', p=p,
                       dense=False, log=False)
    return np.power(cost_emd, 1. / p)
