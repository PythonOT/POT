# -*- coding: utf-8 -*-
"""
Solvers for the original linear program OT problem.

"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import os
import multiprocessing
import sys

import numpy as np
import warnings

from . import cvx
from .cvx import barycenter
from .dmmot import dmmot_monge_1dgrid_loss, dmmot_monge_1dgrid_optimize

# import compiled emd
from .emd_wrap import emd_c, check_result, emd_1d_sorted
from .solver_1d import (emd_1d, emd2_1d, wasserstein_1d,
                        binary_search_circle, wasserstein_circle,
                        semidiscrete_wasserstein2_unif_circle)

from ..utils import dist, list_to_array
from ..utils import parmap
from ..backend import get_backend

__all__ = ['emd', 'emd2', 'barycenter', 'free_support_barycenter', 'cvx', ' emd_1d_sorted',
           'emd_1d', 'emd2_1d', 'wasserstein_1d', 'generalized_free_support_barycenter',
           'binary_search_circle', 'wasserstein_circle', 'semidiscrete_wasserstein2_unif_circle',
           'dmmot_monge_1dgrid_loss', 'dmmot_monge_1dgrid_optimize']


def check_number_threads(numThreads):
    """Checks whether or not the requested number of threads has a valid value.

    Parameters
    ----------
    numThreads : int or str
        The requested number of threads, should either be a strictly positive integer or "max" or None

    Returns
    -------
    numThreads : int
        Corrected number of threads
    """
    if (numThreads is None) or (isinstance(numThreads, str) and numThreads.lower() == 'max'):
        return -1
    if (not isinstance(numThreads, int)) or numThreads < 1:
        raise ValueError('numThreads should either be "max" or a strictly positive integer')
    return numThreads


def center_ot_dual(alpha0, beta0, a=None, b=None):
    r"""Center dual OT potentials w.r.t. their weights

    The main idea of this function is to find unique dual potentials
    that ensure some kind of centering/fairness. The main idea is to find dual potentials that lead to the same final objective value for both source and targets (see below for more details). It will help having
    stability when multiple calling of the OT solver with small changes.

    Basically we add another constraint to the potential that will not
    change the objective value but will ensure unicity. The constraint
    is the following:

    .. math::
        \alpha^T \mathbf{a} = \beta^T \mathbf{b}

    in addition to the OT problem constraints.

    since :math:`\sum_i a_i=\sum_j b_j` this can be solved by adding/removing
    a constant from both  :math:`\alpha_0` and :math:`\beta_0`.

    .. math::
        c &= \frac{\beta_0^T \mathbf{b} - \alpha_0^T \mathbf{a}}{\mathbf{1}^T \mathbf{b} + \mathbf{1}^T \mathbf{a}}

        \alpha &= \alpha_0 + c

        \beta &= \beta_0 + c

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
        This function is necessary because the C++ solver in `emd_c`
        discards all samples in the distributions with
        zeros weights. This means that while the primal variable (transport
        matrix) is exact, the solver only returns feasible dual potentials
        on the samples with weights different from zero.

    First we compute the constraints violations:

    .. math::
        \mathbf{V} = \alpha + \beta^T - \mathbf{M}

    Next we compute the max amount of violation per row (:math:`\alpha`) and
    columns (:math:`beta`)

    .. math::
        \mathbf{v^a}_i = \max_j \mathbf{V}_{i,j}

        \mathbf{v^b}_j = \max_i \mathbf{V}_{i,j}

    Finally we update the dual potential with 0 weights if a
    constraint is violated

    .. math::
        \alpha_i = \alpha_i - \mathbf{v^a}_i \quad \text{ if } \mathbf{a}_i=0 \text{ and } \mathbf{v^a}_i>0

        \beta_j = \beta_j - \mathbf{v^b}_j \quad \text{ if } \mathbf{b}_j=0 \text{ and } \mathbf{v^b}_j > 0

    In the end the dual potentials are centered using function
    :py:func:`ot.lp.center_ot_dual`.

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


def emd(a, b, M, numItermax=100000, log=False, center_dual=True, numThreads=1, check_marginals=True):
    r"""Solves the Earth Movers distance problem and returns the OT matrix


    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights

    .. warning:: Note that the :math:`\mathbf{M}` matrix in numpy needs to be a C-order
        numpy.array in float64 format. It will be converted if not in this
        format

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.

    .. note:: This function will cast the computed transport plan to the data type
        of the provided input with the following priority: :math:`\mathbf{a}`,
        then :math:`\mathbf{b}`, then :math:`\mathbf{M}` if marginals are not provided.
        Casting to an integer tensor might result in a loss of precision.
        If this behaviour is unwanted, please make sure to provide a
        floating point input.

    .. note:: An error will be raised if the vectors :math:`\mathbf{a}` and :math:`\mathbf{b}` do not sum to the same value.

    Uses the algorithm proposed in :ref:`[1] <references-emd>`.

    Parameters
    ----------
    a : (ns,) array-like, float
        Source histogram (uniform weight if empty list)
    b : (nt,) array-like, float
        Target histogram (uniform weight if empty list)
    M : (ns,nt) array-like, float
        Loss matrix (c-order array in numpy with type float64)
    numItermax : int, optional (default=100000)
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.
    log: bool, optional (default=False)
        If True, returns a dictionary containing the cost and dual variables.
        Otherwise returns only the optimal transportation matrix.
    center_dual: boolean, optional (default=True)
        If True, centers the dual potential using function
        :py:func:`ot.lp.center_ot_dual`.
    numThreads: int or "max", optional (default=1, i.e. OpenMP is not used)
        If compiled with OpenMP, chooses the number of threads to parallelize.
        "max" selects the highest number possible.
    check_marginals: bool, optional (default=True)
        If True, checks that the marginals mass are equal. If False, skips the
        check.


    Returns
    -------
    gamma: array-like, shape (ns, nt)
        Optimal transportation matrix for the given
        parameters
    log: dict, optional
        If input log is true, a dictionary containing the
        cost and dual variables and exit status


    Examples
    --------

    Simple example with obvious solution. The function emd accepts lists and
    perform automatic conversion to numpy arrays

    >>> import ot
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ot.emd(a, b, M)
    array([[0.5, 0. ],
           [0. , 0.5]])


    .. _references-emd:
    References
    ----------
    .. [1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W. (2011,
        December).  Displacement interpolation using Lagrangian mass transport.
        In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p. 158). ACM.

    See Also
    --------
    ot.bregman.sinkhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT
    """

    a, b, M = list_to_array(a, b, M)
    nx = get_backend(M, a, b)

    if len(a) != 0:
        type_as = a
    elif len(b) != 0:
        type_as = b
    else:
        type_as = M

    # if empty array given then use uniform distributions
    if len(a) == 0:
        a = nx.ones((M.shape[0],), type_as=type_as) / M.shape[0]
    if len(b) == 0:
        b = nx.ones((M.shape[1],), type_as=type_as) / M.shape[1]

    # store original tensors
    a0, b0, M0 = a, b, M

    # convert to numpy
    M, a, b = nx.to_numpy(M, a, b)

    # ensure float64
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64, order='C')

    # if empty array given then use uniform distributions
    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    assert (a.shape[0] == M.shape[0] and b.shape[0] == M.shape[1]), \
        "Dimension mismatch, check dimensions of M with a and b"

    # ensure that same mass
    if check_marginals:
        np.testing.assert_almost_equal(a.sum(0),
                                       b.sum(0), err_msg='a and b vector must have the same sum',
                                       decimal=6)
    b = b * a.sum() / b.sum()

    asel = a != 0
    bsel = b != 0

    numThreads = check_number_threads(numThreads)

    G, cost, u, v, result_code = emd_c(a, b, M, numItermax, numThreads)

    if center_dual:
        u, v = center_ot_dual(u, v, a, b)

    if np.any(~asel) or np.any(~bsel):
        u, v = estimate_dual_null_weights(u, v, a, b, M)

    result_code_string = check_result(result_code)
    if not nx.is_floating_point(type_as):
        warnings.warn(
            "Input histogram consists of integer. The transport plan will be "
            "casted accordingly, possibly resulting in a loss of precision. "
            "If this behaviour is unwanted, please make sure your input "
            "histogram consists of floating point elements.",
            stacklevel=2
        )
    if log:
        log = {}
        log['cost'] = cost
        log['u'] = nx.from_numpy(u, type_as=type_as)
        log['v'] = nx.from_numpy(v, type_as=type_as)
        log['warning'] = result_code_string
        log['result_code'] = result_code
        return nx.from_numpy(G, type_as=type_as), log
    return nx.from_numpy(G, type_as=type_as)


def emd2(a, b, M, processes=1,
         numItermax=100000, log=False, return_matrix=False,
         center_dual=True, numThreads=1, check_marginals=True):
    r"""Solves the Earth Movers distance problem and returns the loss

    .. math::
        \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.

    .. note:: This function will cast the computed transport plan and
        transportation loss to the data type of the provided input with the
        following priority: :math:`\mathbf{a}`, then :math:`\mathbf{b}`,
        then :math:`\mathbf{M}` if marginals are not provided.
        Casting to an integer tensor might result in a loss of precision.
        If this behaviour is unwanted, please make sure to provide a
        floating point input.

    .. note:: An error will be raised if the vectors :math:`\mathbf{a}` and :math:`\mathbf{b}` do not sum to the same value.

    Uses the algorithm proposed in :ref:`[1] <references-emd2>`.

    Parameters
    ----------
    a : (ns,) array-like, float64
        Source histogram (uniform weight if empty list)
    b : (nt,) array-like, float64
        Target histogram (uniform weight if empty list)
    M : (ns,nt) array-like, float64
        Loss matrix (for numpy c-order array with type float64)
    processes : int, optional (default=1)
        Nb of processes used for multiple emd computation (deprecated)
    numItermax : int, optional (default=100000)
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.
    log: boolean, optional (default=False)
        If True, returns a dictionary containing dual
        variables. Otherwise returns only the optimal transportation cost.
    return_matrix: boolean, optional (default=False)
        If True, returns the optimal transportation matrix in the log.
    center_dual: boolean, optional (default=True)
        If True, centers the dual potential using function
        :py:func:`ot.lp.center_ot_dual`.
    numThreads: int or "max", optional (default=1, i.e. OpenMP is not used)
        If compiled with OpenMP, chooses the number of threads to parallelize.
        "max" selects the highest number possible.
    check_marginals: bool, optional (default=True)
        If True, checks that the marginals mass are equal. If False, skips the
        check.


    Returns
    -------
    W: float, array-like
        Optimal transportation loss for the given parameters
    log: dict
        If input log is true, a dictionary containing dual
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


    .. _references-emd2:
    References
    ----------
    .. [1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W.
        (2011, December).  Displacement interpolation using Lagrangian mass
        transport. In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p.
        158). ACM.

    See Also
    --------
    ot.bregman.sinkhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT
    """

    a, b, M = list_to_array(a, b, M)
    nx = get_backend(M, a, b)

    if len(a) != 0:
        type_as = a
    elif len(b) != 0:
        type_as = b
    else:
        type_as = M

    # if empty array given then use uniform distributions
    if len(a) == 0:
        a = nx.ones((M.shape[0],), type_as=type_as) / M.shape[0]
    if len(b) == 0:
        b = nx.ones((M.shape[1],), type_as=type_as) / M.shape[1]

    # store original tensors
    a0, b0, M0 = a, b, M

    # convert to numpy
    M, a, b = nx.to_numpy(M, a, b)

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64, order='C')

    assert (a.shape[0] == M.shape[0] and b.shape[0] == M.shape[1]), \
        "Dimension mismatch, check dimensions of M with a and b"

    # ensure that same mass
    if check_marginals:
        np.testing.assert_almost_equal(a.sum(0),
                                       b.sum(0, keepdims=True), err_msg='a and b vector must have the same sum',
                                       decimal=6)
    b = b * a.sum(0) / b.sum(0, keepdims=True)

    asel = a != 0

    numThreads = check_number_threads(numThreads)

    if log or return_matrix:
        def f(b):
            bsel = b != 0

            G, cost, u, v, result_code = emd_c(a, b, M, numItermax, numThreads)

            if center_dual:
                u, v = center_ot_dual(u, v, a, b)

            if np.any(~asel) or np.any(~bsel):
                u, v = estimate_dual_null_weights(u, v, a, b, M)

            result_code_string = check_result(result_code)
            log = {}
            if not nx.is_floating_point(type_as):
                warnings.warn(
                    "Input histogram consists of integer. The transport plan will be "
                    "casted accordingly, possibly resulting in a loss of precision. "
                    "If this behaviour is unwanted, please make sure your input "
                    "histogram consists of floating point elements.",
                    stacklevel=2
                )
            G = nx.from_numpy(G, type_as=type_as)
            if return_matrix:
                log['G'] = G
            log['u'] = nx.from_numpy(u, type_as=type_as)
            log['v'] = nx.from_numpy(v, type_as=type_as)
            log['warning'] = result_code_string
            log['result_code'] = result_code
            cost = nx.set_gradients(nx.from_numpy(cost, type_as=type_as),
                                    (a0, b0, M0), (log['u'] - nx.mean(log['u']),
                                                   log['v'] - nx.mean(log['v']), G))
            return [cost, log]
    else:
        def f(b):
            bsel = b != 0
            G, cost, u, v, result_code = emd_c(a, b, M, numItermax, numThreads)

            if center_dual:
                u, v = center_ot_dual(u, v, a, b)

            if np.any(~asel) or np.any(~bsel):
                u, v = estimate_dual_null_weights(u, v, a, b, M)

            if not nx.is_floating_point(type_as):
                warnings.warn(
                    "Input histogram consists of integer. The transport plan will be "
                    "casted accordingly, possibly resulting in a loss of precision. "
                    "If this behaviour is unwanted, please make sure your input "
                    "histogram consists of floating point elements.",
                    stacklevel=2
                )
            G = nx.from_numpy(G, type_as=type_as)
            cost = nx.set_gradients(nx.from_numpy(cost, type_as=type_as),
                                    (a0, b0, M0), (nx.from_numpy(u - np.mean(u), type_as=type_as),
                                                   nx.from_numpy(v - np.mean(v), type_as=type_as), G))

            check_result(result_code)
            return cost

    if len(b.shape) == 1:
        return f(b)
    nb = b.shape[1]

    if processes > 1:
        warnings.warn(
            "The 'processes' parameter has been deprecated. "
            "Multiprocessing should be done outside of POT."
        )
    res = list(map(f, [b[:, i].copy() for i in range(nb)]))

    return res


def free_support_barycenter(measures_locations, measures_weights, X_init, b=None, weights=None, numItermax=100,
                            stopThr=1e-7, verbose=False, log=None, numThreads=1):
    r"""
    Solves the free support (locations of the barycenters are optimized, not the weights) Wasserstein barycenter problem (i.e. the weighted Frechet mean for the 2-Wasserstein distance), formally:

    .. math::
        \min_\mathbf{X} \quad \sum_{i=1}^N w_i W_2^2(\mathbf{b}, \mathbf{X}, \mathbf{a}_i, \mathbf{X}_i)

    where :

    - :math:`w \in \mathbb{(0, 1)}^{N}`'s are the barycenter weights and sum to one
    - `measure_weights` denotes the :math:`\mathbf{a}_i \in \mathbb{R}^{k_i}`: empirical measures weights (on simplex)
    - `measures_locations` denotes the :math:`\mathbf{X}_i \in \mathbb{R}^{k_i, d}`: empirical measures atoms locations
    - :math:`\mathbf{b} \in \mathbb{R}^{k}` is the desired weights vector of the barycenter

    This problem is considered in :ref:`[20] <references-free-support-barycenter>` (Algorithm 2).
    There are two differences with the following codes:

    - we do not optimize over the weights
    - we do not do line search for the locations updates, we use i.e. :math:`\theta = 1` in
      :ref:`[20] <references-free-support-barycenter>` (Algorithm 2). This can be seen as a discrete
      implementation of the fixed-point algorithm of
      :ref:`[43] <references-free-support-barycenter>` proposed in the continuous setting.

    Parameters
    ----------
    measures_locations : list of N (k_i,d) array-like
        The discrete support of a measure supported on :math:`k_i` locations of a `d`-dimensional space
        (:math:`k_i` can be different for each element of the list)
    measures_weights : list of N (k_i,) array-like
        Numpy arrays where each numpy array has :math:`k_i` non-negatives values summing to one
        representing the weights of each discrete input measure

    X_init : (k,d) array-like
        Initialization of the support locations (on `k` atoms) of the barycenter
    b : (k,) array-like
        Initialization of the weights of the barycenter (non-negatives, sum to 1)
    weights : (N,) array-like
        Initialization of the coefficients of the barycenter (non-negatives, sum to 1)

    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    numThreads: int or "max", optional (default=1, i.e. OpenMP is not used)
        If compiled with OpenMP, chooses the number of threads to parallelize.
        "max" selects the highest number possible.


    Returns
    -------
    X : (k,d) array-like
        Support locations (on k atoms) of the barycenter


    .. _references-free-support-barycenter:

    References
    ----------
    .. [20] Cuturi, Marco, and Arnaud Doucet. "Fast computation of Wasserstein barycenters." International Conference on Machine Learning. 2014.

    .. [43] Álvarez-Esteban, Pedro C., et al. "A fixed-point approach to barycenters in Wasserstein space." Journal of Mathematical Analysis and Applications 441.2 (2016): 744-762.

    """

    nx = get_backend(*measures_locations, *measures_weights, X_init)

    iter_count = 0

    N = len(measures_locations)
    k = X_init.shape[0]
    d = X_init.shape[1]
    if b is None:
        b = nx.ones((k,), type_as=X_init) / k
    if weights is None:
        weights = nx.ones((N,), type_as=X_init) / N

    X = X_init

    log_dict = {}
    displacement_square_norms = []

    displacement_square_norm = stopThr + 1.

    while (displacement_square_norm > stopThr and iter_count < numItermax):

        T_sum = nx.zeros((k, d), type_as=X_init)

        for (measure_locations_i, measure_weights_i, weight_i) in zip(measures_locations, measures_weights, weights):
            M_i = dist(X, measure_locations_i)
            T_i = emd(b, measure_weights_i, M_i, numThreads=numThreads)
            T_sum = T_sum + weight_i * 1. / b[:, None] * nx.dot(T_i, measure_locations_i)

        displacement_square_norm = nx.sum((T_sum - X) ** 2)
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


def generalized_free_support_barycenter(X_list, a_list, P_list, n_samples_bary, Y_init=None, b=None, weights=None,
                                        numItermax=100, stopThr=1e-7, verbose=False, log=None, numThreads=1, eps=0):
    r"""
    Solves the free support generalized Wasserstein barycenter problem: finding a barycenter (a discrete measure with
    a fixed amount of points of uniform weights) whose respective projections fit the input measures.
    More formally:

    .. math::
        \min_\gamma \quad \sum_{i=1}^p w_i W_2^2(\nu_i, \mathbf{P}_i\#\gamma)

    where :

    - :math:`\gamma = \sum_{l=1}^n b_l\delta_{y_l}` is the desired barycenter with each :math:`y_l \in \mathbb{R}^d`
    - :math:`\mathbf{b} \in \mathbb{R}^{n}` is the desired weights vector of the barycenter
    - The input measures are :math:`\nu_i = \sum_{j=1}^{k_i}a_{i,j}\delta_{x_{i,j}}`
    - The :math:`\mathbf{a}_i \in \mathbb{R}^{k_i}` are the respective empirical measures weights (on the simplex)
    - The :math:`\mathbf{X}_i \in \mathbb{R}^{k_i, d_i}` are the respective empirical measures atoms locations
    - :math:`w = (w_1, \cdots w_p)` are the barycenter coefficients (on the simplex)
    - Each :math:`\mathbf{P}_i \in \mathbb{R}^{d, d_i}`, and :math:`P_i\#\nu_i = \sum_{j=1}^{k_i}a_{i,j}\delta_{P_ix_{i,j}}`

    As show by :ref:`[42] <references-generalized-free-support-barycenter>`,
    this problem can be re-written as a Wasserstein Barycenter problem,
    which we solve using the free support method :ref:`[20] <references-generalized-free-support-barycenter>`
    (Algorithm 2).

    Parameters
    ----------
    X_list : list of p (k_i,d_i) array-like
        Discrete supports of the input measures: each consists of :math:`k_i` locations of a `d_i`-dimensional space
        (:math:`k_i` can be different for each element of the list)
    a_list : list of p (k_i,) array-like
        Measure weights: each element is a vector (k_i) on the simplex
    P_list : list of p (d_i,d) array-like
        Each :math:`P_i` is a linear map :math:`\mathbb{R}^{d} \rightarrow \mathbb{R}^{d_i}`
    n_samples_bary : int
        Number of barycenter points
    Y_init : (n_samples_bary,d) array-like
        Initialization of the support locations (on `k` atoms) of the barycenter
    b : (n_samples_bary,) array-like
        Initialization of the weights of the barycenter measure (on the simplex)
    weights : (p,) array-like
        Initialization of the coefficients of the barycenter (on the simplex)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    numThreads: int or "max", optional (default=1, i.e. OpenMP is not used)
        If compiled with OpenMP, chooses the number of threads to parallelize.
        "max" selects the highest number possible.
    eps: Stability coefficient for the change of variable matrix inversion
        If the :math:`\mathbf{P}_i^T` matrices don't span :math:`\mathbb{R}^d`, the problem is ill-defined and a matrix
        inversion will fail. In this case one may set eps=1e-8 and get a solution anyway (which may make little sense)


    Returns
    -------
    Y : (n_samples_bary,d) array-like
        Support locations (on n_samples_bary atoms) of the barycenter


    .. _references-generalized-free-support-barycenter:
    References
    ----------
    .. [20] Cuturi, M. and Doucet, A.. "Fast computation of Wasserstein barycenters." International Conference on Machine Learning. 2014.

    .. [42] Delon, J., Gozlan, N., and Saint-Dizier, A.. Generalized Wasserstein barycenters between probability measures living on different subspaces. arXiv preprint arXiv:2105.09755, 2021.

    """
    nx = get_backend(*X_list, *a_list, *P_list)
    d = P_list[0].shape[1]
    p = len(X_list)

    if weights is None:
        weights = nx.ones(p, type_as=X_list[0]) / p

    # variable change matrix to reduce the problem to a Wasserstein Barycenter (WB)
    A = eps * nx.eye(d, type_as=X_list[0])  # if eps nonzero: will force the invertibility of A
    for (P_i, lambda_i) in zip(P_list, weights):
        A = A + lambda_i * P_i.T @ P_i
    B = nx.inv(nx.sqrtm(A))

    Z_list = [x @ Pi @ B.T for (x, Pi) in zip(X_list, P_list)]  # change of variables -> (WB) problem on Z

    if Y_init is None:
        Y_init = nx.randn(n_samples_bary, d, type_as=X_list[0])

    if b is None:
        b = nx.ones(n_samples_bary, type_as=X_list[0]) / n_samples_bary  # not optimized

    out = free_support_barycenter(Z_list, a_list, Y_init, b, numItermax=numItermax,
                                  stopThr=stopThr, verbose=verbose, log=log, numThreads=numThreads)

    if log:  # unpack
        Y, log_dict = out
    else:
        Y = out
        log_dict = None
    Y = Y @ B.T  # return to the Generalized WB formulation

    if log:
        return Y, log_dict
    else:
        return Y
