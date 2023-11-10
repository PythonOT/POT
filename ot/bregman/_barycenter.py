# -*- coding: utf-8 -*-
"""
Bregman projections solvers for entropic regularized wasserstein barycenters
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#         Hicham Janati <hicham.janati100@gmail.com>
#         Ievgen Redko <ievgen.redko@univ-st-etienne.fr>
#
# License: MIT License

import warnings
import numpy as np

from ..utils import dist, list_to_array, unif
from ..backend import get_backend

from ._utils import geometricBar, geometricMean, projR, projC
from ._sinkhorn import sinkhorn


def barycenter(A, M, reg, weights=None, method="sinkhorn", numItermax=10000,
               stopThr=1e-4, verbose=False, log=False, warn=True, **kwargs):
    r"""Compute the entropic regularized wasserstein barycenter of distributions :math:`\mathbf{A}`

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein
      distance (see :py:func:`ot.bregman.sinkhorn`)
      if `method` is `sinkhorn` or `sinkhorn_stabilized` or `sinkhorn_log`.
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix
      :math:`\mathbf{A}`
    - `reg` and :math:`\mathbf{M}` are respectively the regularization term and
      the cost matrix for OT

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling
    algorithm as proposed in :ref:`[3] <references-barycenter>`

    Parameters
    ----------
    A : array-like, shape (dim, n_hists)
        `n_hists` training distributions :math:`\mathbf{a}_i` of size `dim`
    M : array-like, shape (dim, dim)
        loss matrix for OT
    reg : float
        Regularization term > 0
    method : str (optional)
        method used for the solver either 'sinkhorn' or 'sinkhorn_stabilized' or 'sinkhorn_log'
    weights : array-like, shape (n_hists,)
        Weights of each histogram :math:`\mathbf{a}_i` on the simplex (barycentric coodinates)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.


    Returns
    -------
    a : (dim,) array-like
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-barycenter:
    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015).
        Iterative Bregman projections for regularized transportation problems.
        SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    """

    if method.lower() == 'sinkhorn':
        return barycenter_sinkhorn(A, M, reg, weights=weights,
                                   numItermax=numItermax,
                                   stopThr=stopThr, verbose=verbose, log=log,
                                   warn=warn,
                                   **kwargs)
    elif method.lower() == 'sinkhorn_stabilized':
        return barycenter_stabilized(A, M, reg, weights=weights,
                                     numItermax=numItermax,
                                     stopThr=stopThr, verbose=verbose,
                                     log=log, warn=warn, **kwargs)
    elif method.lower() == 'sinkhorn_log':
        return _barycenter_sinkhorn_log(A, M, reg, weights=weights,
                                        numItermax=numItermax,
                                        stopThr=stopThr, verbose=verbose,
                                        log=log, warn=warn, **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def barycenter_sinkhorn(A, M, reg, weights=None, numItermax=1000,
                        stopThr=1e-4, verbose=False, log=False, warn=True):
    r"""Compute the entropic regularized wasserstein barycenter of distributions :math:`\mathbf{A}`

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance
      (see :py:func:`ot.bregman.sinkhorn`)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix
      :math:`\mathbf{A}`
    - `reg` and :math:`\mathbf{M}` are respectively the regularization term and
      the cost matrix for OT

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in :ref:`[3]<references-barycenter-sinkhorn>`.

    Parameters
    ----------
    A : array-like, shape (dim, n_hists)
        `n_hists` training distributions :math:`\mathbf{a}_i` of size `dim`
    M : array-like, shape (dim, dim)
        loss matrix for OT
    reg : float
        Regularization term > 0
    weights : array-like, shape (n_hists,)
        Weights of each histogram :math:`\mathbf{a}_i` on the simplex (barycentric coodinates)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.


    Returns
    -------
    a : (dim,) array-like
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-barycenter-sinkhorn:
    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015).
    Iterative Bregman projections for regularized transportation problems.
    SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    """

    A, M = list_to_array(A, M)

    nx = get_backend(A, M)

    if weights is None:
        weights = nx.ones((A.shape[1],), type_as=A) / A.shape[1]
    else:
        assert (len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    K = nx.exp(-M / reg)

    err = 1

    UKv = nx.dot(K, (A.T / nx.sum(K, axis=0)).T)

    u = (geometricMean(UKv) / UKv.T).T

    for ii in range(numItermax):

        UKv = u * nx.dot(K.T, A / nx.dot(K, u))
        u = (u.T * geometricBar(weights, UKv)).T / UKv

        if ii % 10 == 1:
            err = nx.sum(nx.std(UKv, axis=1))

            # log and verbose print
            if log:
                log['err'].append(err)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        log['niter'] = ii
        return geometricBar(weights, UKv), log
    else:
        return geometricBar(weights, UKv)


def free_support_sinkhorn_barycenter(measures_locations, measures_weights, X_init, reg, b=None, weights=None,
                                     numItermax=100, numInnerItermax=1000, stopThr=1e-7, verbose=False, log=None,
                                     **kwargs):
    r"""
    Solves the free support (locations of the barycenters are optimized, not the weights) regularized Wasserstein barycenter problem (i.e. the weighted Frechet mean for the 2-Sinkhorn divergence), formally:

    .. math::
        \min_\mathbf{X} \quad \sum_{i=1}^N w_i W_{reg}^2(\mathbf{b}, \mathbf{X}, \mathbf{a}_i, \mathbf{X}_i)

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
    - at each iteration, instead of solving an exact OT problem, we use the Sinkhorn algorithm for calculating the
      transport plan in :ref:`[20] <references-free-support-barycenter>` (Algorithm 2).

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
    reg : float
        Regularization term >0
    b : (k,) array-like
        Initialization of the weights of the barycenter (non-negatives, sum to 1)
    weights : (N,) array-like
        Initialization of the coefficients of the barycenter (non-negatives, sum to 1)

    numItermax : int, optional
        Max number of iterations
    numInnerItermax : int, optional
        Max number of iterations when calculating the transport plans with Sinkhorn
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    X : (k,d) array-like
        Support locations (on k atoms) of the barycenter

    See Also
    --------
    ot.bregman.sinkhorn : Entropic regularized OT solver
    ot.lp.free_support_barycenter : Barycenter solver based on Linear Programming

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
            T_i = sinkhorn(b, measure_weights_i, M_i, reg=reg,
                           numItermax=numInnerItermax, **kwargs)
            T_sum = T_sum + weight_i * 1. / \
                b[:, None] * nx.dot(T_i, measure_locations_i)

        displacement_square_norm = nx.sum((T_sum - X) ** 2)
        if log:
            displacement_square_norms.append(displacement_square_norm)

        X = T_sum

        if verbose:
            print('iteration %d, displacement_square_norm=%f\n',
                  iter_count, displacement_square_norm)

        iter_count += 1

    if log:
        log_dict['displacement_square_norms'] = displacement_square_norms
        return X, log_dict
    else:
        return X


def _barycenter_sinkhorn_log(A, M, reg, weights=None, numItermax=1000,
                             stopThr=1e-4, verbose=False, log=False, warn=True):
    r"""Compute the entropic wasserstein barycenter in log-domain
    """

    A, M = list_to_array(A, M)
    dim, n_hists = A.shape

    nx = get_backend(A, M)

    if nx.__name__ in ("jax", "tf"):
        raise NotImplementedError(
            "Log-domain functions are not yet implemented"
            " for Jax and tf. Use numpy or torch arrays instead."
        )

    if weights is None:
        weights = nx.ones(n_hists, type_as=A) / n_hists
    else:
        assert (len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    M = - M / reg
    logA = nx.log(A + 1e-15)
    log_KU, G = nx.zeros((2, *logA.shape), type_as=A)
    err = 1
    for ii in range(numItermax):
        log_bar = nx.zeros(dim, type_as=A)
        for k in range(n_hists):
            f = logA[:, k] - nx.logsumexp(M + G[None, :, k], axis=1)
            log_KU[:, k] = nx.logsumexp(M + f[:, None], axis=0)
            log_bar = log_bar + weights[k] * log_KU[:, k]

        if ii % 10 == 1:
            err = nx.exp(G + log_KU).std(axis=1).sum()

            # log and verbose print
            if log:
                log['err'].append(err)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))

        G = log_bar[:, None] - log_KU

    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        log['niter'] = ii
        return nx.exp(log_bar), log
    else:
        return nx.exp(log_bar)


def barycenter_stabilized(A, M, reg, tau=1e10, weights=None, numItermax=1000,
                          stopThr=1e-4, verbose=False, log=False, warn=True):
    r"""Compute the entropic regularized wasserstein barycenter of distributions :math:`\mathbf{A}` with stabilization.

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein
      distance (see :py:func:`ot.bregman.sinkhorn`)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix
      :math:`\mathbf{A}`
    - `reg` and :math:`\mathbf{M}` are respectively the regularization term and
      the cost matrix for OT

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling
    algorithm as proposed in :ref:`[3] <references-barycenter-stabilized>`

    Parameters
    ----------
    A : array-like, shape (dim, n_hists)
        `n_hists` training distributions :math:`\mathbf{a}_i` of size `dim`
    M : array-like, shape (dim, dim)
        loss matrix for OT
    reg : float
        Regularization term > 0
    tau : float
        threshold for max value in :math:`\mathbf{u}` or :math:`\mathbf{v}`
        for log scaling
    weights : array-like, shape (n_hists,)
        Weights of each histogram :math:`\mathbf{a}_i` on the simplex (barycentric coodinates)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.


    Returns
    -------
    a : (dim,) array-like
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-barycenter-stabilized:
    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015).
        Iterative Bregman projections for regularized transportation problems.
        SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    """

    A, M = list_to_array(A, M)

    nx = get_backend(A, M)

    dim, n_hists = A.shape
    if weights is None:
        weights = nx.ones((n_hists,), type_as=M) / n_hists
    else:
        assert (len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    u = nx.ones((dim, n_hists), type_as=M) / dim
    v = nx.ones((dim, n_hists), type_as=M) / dim

    K = nx.exp(-M / reg)

    err = 1.
    alpha = nx.zeros((dim,), type_as=M)
    beta = nx.zeros((dim,), type_as=M)
    q = nx.ones((dim,), type_as=M) / dim
    for ii in range(numItermax):
        qprev = q
        Kv = nx.dot(K, v)
        u = A / Kv
        Ktu = nx.dot(K.T, u)
        q = geometricBar(weights, Ktu)
        Q = q[:, None]
        v = Q / Ktu
        absorbing = False
        if nx.any(u > tau) or nx.any(v > tau):
            absorbing = True
            alpha += reg * nx.log(nx.max(u, 1))
            beta += reg * nx.log(nx.max(v, 1))
            K = nx.exp((alpha[:, None] + beta[None, :] - M) / reg)
            v = nx.ones(tuple(v.shape), type_as=v)
        Kv = nx.dot(K, v)
        if (nx.any(Ktu == 0.)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % ii)
            q = qprev
            break
        if (ii % 10 == 0 and not absorbing) or ii == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.max(nx.abs(u * Kv - A))
            if log:
                log['err'].append(err)
            if err < stopThr:
                break
            if verbose:
                if ii % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))

    else:
        if warn:
            warnings.warn("Stabilized Sinkhorn did not converge." +
                          "Try a larger entropy `reg`" +
                          "Or a larger absorption threshold `tau`.")
    if log:
        log['niter'] = ii
        log['logu'] = nx.log(u + 1e-16)
        log['logv'] = nx.log(v + 1e-16)
        return q, log
    else:
        return q


def barycenter_debiased(A, M, reg, weights=None, method="sinkhorn", numItermax=10000,
                        stopThr=1e-4, verbose=False, log=False, warn=True, **kwargs):
    r"""Compute the debiased Sinkhorn barycenter of distributions A

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i S_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`S_{reg}(\cdot,\cdot)` is the debiased Sinkhorn divergence
      (see :py:func:`ot.bregman.empirical_sinkhorn_divergence`)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix
      :math:`\mathbf{A}`
    - `reg` and :math:`\mathbf{M}` are respectively the regularization term and
      the cost matrix for OT

    The algorithm used for solving the problem is the debiased Sinkhorn
    algorithm as proposed in :ref:`[37] <references-barycenter-debiased>`

    Parameters
    ----------
    A : array-like, shape (dim, n_hists)
        `n_hists` training distributions :math:`\mathbf{a}_i` of size `dim`
    M : array-like, shape (dim, dim)
        loss matrix for OT
    reg : float
        Regularization term > 0
    method : str (optional)
        method used for the solver either 'sinkhorn' or 'sinkhorn_log'
    weights : array-like, shape (n_hists,)
        Weights of each histogram :math:`\mathbf{a}_i` on the simplex (barycentric coodinates)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.


    Returns
    -------
    a : (dim,) array-like
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-barycenter-debiased:
    References
    ----------
    .. [37] Janati, H., Cuturi, M., Gramfort, A. Proceedings of the 37th International
        Conference on Machine Learning, PMLR 119:4692-4701, 2020
    """

    if method.lower() == 'sinkhorn':
        return _barycenter_debiased(A, M, reg, weights=weights,
                                    numItermax=numItermax,
                                    stopThr=stopThr, verbose=verbose, log=log,
                                    warn=warn, **kwargs)
    elif method.lower() == 'sinkhorn_log':
        return _barycenter_debiased_log(A, M, reg, weights=weights,
                                        numItermax=numItermax,
                                        stopThr=stopThr, verbose=verbose,
                                        log=log, warn=warn, **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def _barycenter_debiased(A, M, reg, weights=None, numItermax=1000,
                         stopThr=1e-4, verbose=False, log=False, warn=True):
    r"""Compute the debiased sinkhorn barycenter of distributions A.
    """

    A, M = list_to_array(A, M)

    nx = get_backend(A, M)

    if weights is None:
        weights = nx.ones((A.shape[1],), type_as=A) / A.shape[1]
    else:
        assert (len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    K = nx.exp(-M / reg)

    err = 1

    UKv = nx.dot(K, (A.T / nx.sum(K, axis=0)).T)

    u = (geometricMean(UKv) / UKv.T).T
    c = nx.ones(A.shape[0], type_as=A)
    bar = nx.ones(A.shape[0], type_as=A)

    for ii in range(numItermax):
        bold = bar
        UKv = nx.dot(K, A / nx.dot(K, u))
        bar = c * geometricBar(weights, UKv)
        u = bar[:, None] / UKv
        c = (c * bar / nx.dot(K, c)) ** 0.5

        if ii % 10 == 9:
            err = abs(bar - bold).max() / max(bar.max(), 1.)

            # log and verbose print
            if log:
                log['err'].append(err)

            # debiased Sinkhorn does not converge monotonically
            # guarantee a few iterations are done before stopping
            if err < stopThr and ii > 20:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        log['niter'] = ii
        return bar, log
    else:
        return bar


def _barycenter_debiased_log(A, M, reg, weights=None, numItermax=1000,
                             stopThr=1e-4, verbose=False, log=False,
                             warn=True):
    r"""Compute the debiased sinkhorn barycenter in log domain.
     """

    A, M = list_to_array(A, M)
    dim, n_hists = A.shape

    nx = get_backend(A, M)
    if nx.__name__ in ("jax", "tf"):
        raise NotImplementedError(
            "Log-domain functions are not yet implemented"
            " for Jax and TF. Use numpy or torch arrays instead."
        )

    if weights is None:
        weights = nx.ones(n_hists, type_as=A) / n_hists
    else:
        assert (len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    M = - M / reg
    logA = nx.log(A + 1e-15)
    log_KU, G = nx.zeros((2, *logA.shape), type_as=A)
    c = nx.zeros(dim, type_as=A)
    err = 1
    for ii in range(numItermax):
        log_bar = nx.zeros(dim, type_as=A)
        for k in range(n_hists):
            f = logA[:, k] - nx.logsumexp(M + G[None, :, k], axis=1)
            log_KU[:, k] = nx.logsumexp(M + f[:, None], axis=0)
            log_bar += weights[k] * log_KU[:, k]
        log_bar += c
        if ii % 10 == 1:
            err = nx.exp(G + log_KU).std(axis=1).sum()

            # log and verbose print
            if log:
                log['err'].append(err)

            if err < stopThr and ii > 20:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))

        G = log_bar[:, None] - log_KU
        for _ in range(10):
            c = 0.5 * (c + log_bar - nx.logsumexp(M + c[:, None], axis=0))

    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        log['niter'] = ii
        return nx.exp(log_bar), log
    else:
        return nx.exp(log_bar)


def jcpot_barycenter(Xs, Ys, Xt, reg, metric='sqeuclidean', numItermax=100,
                     stopThr=1e-6, verbose=False, log=False, warn=True, **kwargs):
    r'''Joint OT and proportion estimation for multi-source target shift as
    proposed in :ref:`[27] <references-jcpot-barycenter>`

    The function solves the following optimization problem:

    .. math::

        \mathbf{h} = \mathop{\arg \min}_{\mathbf{h}} \quad \sum_{k=1}^{K} \lambda_k
                    W_{reg}((\mathbf{D}_2^{(k)} \mathbf{h})^T, \mathbf{a})

        s.t. \ \forall k, \mathbf{D}_1^{(k)} \gamma_k \mathbf{1}_n= \mathbf{h}

    where :

    - :math:`\lambda_k` is the weight of `k`-th source domain
    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance
      (see :py:func:`ot.bregman.sinkhorn`)
    - :math:`\mathbf{D}_2^{(k)}` is a matrix of weights related to `k`-th source domain
      defined as in [p. 5, :ref:`27 <references-jcpot-barycenter>`], its expected shape
      is :math:`(n_k, C)` where :math:`n_k` is the number of elements in the `k`-th source
      domain and `C` is the number of classes
    - :math:`\mathbf{h}` is a vector of estimated proportions in the target domain of size `C`
    - :math:`\mathbf{a}` is a uniform vector of weights in the target domain of size `n`
    - :math:`\mathbf{D}_1^{(k)}` is a matrix of class assignments defined as in
      [p. 5, :ref:`27 <references-jcpot-barycenter>`], its expected shape is :math:`(n_k, C)`

    The problem consist in solving a Wasserstein barycenter problem to estimate
    the proportions :math:`\mathbf{h}` in the target domain.

    The algorithm used for solving the problem is the Iterative Bregman projections algorithm
    with two sets of marginal constraints related to the unknown vector
    :math:`\mathbf{h}` and uniform target distribution.

    Parameters
    ----------
    Xs : list of K array-like(nsk,d)
        features of all source domains' samples
    Ys : list of K array-like(nsk,)
        labels of all source domains' samples
    Xt : array-like (nt,d)
        samples in the target domain
    reg : float
        Regularization term > 0
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on relative change in the barycenter (>0)
    verbose : bool, optional (default=False)
        Controls the verbosity of the optimization algorithm
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.

    Returns
    -------
    h : (C,) array-like
        proportion estimation in the target domain
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-jcpot-barycenter:
    References
    ----------

    .. [27] Ievgen Redko, Nicolas Courty, Rémi Flamary, Devis Tuia
        "Optimal transport for multi-source domain adaptation under target shift",
        International Conference on Artificial Intelligence and Statistics (AISTATS), 2019.
    '''

    Xs = list_to_array(*Xs)
    Ys = list_to_array(*Ys)
    Xt = list_to_array(Xt)

    nx = get_backend(*Xs, *Ys, Xt)

    nbclasses = len(nx.unique(Ys[0]))
    nbdomains = len(Xs)

    # log dictionary
    if log:
        log = {'niter': 0, 'err': [], 'M': [], 'D1': [], 'D2': [], 'gamma': []}

    K = []
    M = []
    D1 = []
    D2 = []

    # For each source domain, build cost matrices M, Gibbs kernels K and corresponding matrices D_1 and D_2
    for d in range(nbdomains):
        dom = {}
        nsk = Xs[d].shape[0]  # get number of elements for this domain
        dom['nbelem'] = nsk
        classes = nx.unique(Ys[d])  # get number of classes for this domain

        # format classes to start from 0 for convenience
        if nx.min(classes) != 0:
            Ys[d] -= nx.min(classes)
            classes = nx.unique(Ys[d])

        # build the corresponding D_1 and D_2 matrices
        Dtmp1 = np.zeros((nbclasses, nsk))
        Dtmp2 = np.zeros((nbclasses, nsk))

        for c in classes:
            nbelemperclass = float(nx.sum(Ys[d] == c))
            if nbelemperclass != 0:
                Dtmp1[int(c), nx.to_numpy(Ys[d] == c)] = 1.
                Dtmp2[int(c), nx.to_numpy(Ys[d] == c)] = 1. / (nbelemperclass)
        D1.append(nx.from_numpy(Dtmp1, type_as=Xs[0]))
        D2.append(nx.from_numpy(Dtmp2, type_as=Xs[0]))

        # build the cost matrix and the Gibbs kernel
        Mtmp = dist(Xs[d], Xt, metric=metric)
        M.append(Mtmp)

        Ktmp = nx.exp(-Mtmp / reg)
        K.append(Ktmp)

    # uniform target distribution
    a = nx.from_numpy(unif(Xt.shape[0]), type_as=Xs[0])

    err = 1
    old_bary = nx.ones((nbclasses,), type_as=Xs[0])

    for ii in range(numItermax):

        bary = nx.zeros((nbclasses,), type_as=Xs[0])

        # update coupling matrices for marginal constraints w.r.t. uniform target distribution
        for d in range(nbdomains):
            K[d] = projC(K[d], a)
            other = nx.sum(K[d], axis=1)
            bary += nx.log(nx.dot(D1[d], other)) / nbdomains

        bary = nx.exp(bary)

        # update coupling matrices for marginal constraints w.r.t. unknown proportions based on [Prop 4., 27]
        for d in range(nbdomains):
            new = nx.dot(D2[d].T, bary)
            K[d] = projR(K[d], new)

        err = nx.norm(bary - old_bary)

        old_bary = bary

        if log:
            log['err'].append(err)

        if err < stopThr:
            break
        if verbose:
            if ii % 200 == 0:
                print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
    else:
        if warn:
            warnings.warn("Algorithm did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    bary = bary / nx.sum(bary)

    if log:
        log['niter'] = ii
        log['M'] = M
        log['D1'] = D1
        log['D2'] = D2
        log['gamma'] = K
        return bary, log
    else:
        return bary
