# -*- coding: utf-8 -*-
"""
Regularized Unbalanced OT solvers
"""

# Author: Hicham Janati <hicham.janati@inria.fr>
# License: MIT License

from __future__ import division
import warnings
import numpy as np
from scipy.special import logsumexp

# from .utils import unif, dist


def sinkhorn_unbalanced(a, b, M, reg, reg_m, method='sinkhorn', numItermax=1000,
                        stopThr=1e-6, verbose=False, log=False, **kwargs):
    r"""
    Solve the unbalanced entropic regularization optimal transport problem
    and return the OT plan

    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma) + reg_m KL(\gamma 1, a) + reg_m KL(\gamma^T 1, b)

        s.t.
             \gamma\geq 0
    where :

    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization
        term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target unbalanced distributions
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized
        Sinkhorn-Knopp matrix scaling algorithm as proposed in [10, 23]_


    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Unnormalized histogram of dimension dim_a
    b : np.ndarray (dim_b,) or np.ndarray (dim_b, n_hists)
        One or multiple unnormalized histograms of dimension dim_b
        If many, compute all the OT distances (a, b_i)
    M : np.ndarray (dim_a, dim_b)
        loss matrix
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    method : str
        method used for the solver either 'sinkhorn',  'sinkhorn_stabilized' or
        'sinkhorn_reg_scaling', see those function for specific parameters
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    if n_hists == 1:
        gamma : (dim_a x dim_b) ndarray
            Optimal transportation matrix for the given parameters
        log : dict
            log dictionary returned only if `log` is `True`
    else:
        ot_distance : (n_hists,) ndarray
            the OT distance between `a` and each of the histograms `b_i`
        log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn_unbalanced(a, b, M, 1, 1)
    array([[0.51122823, 0.18807035],
           [0.18807035, 0.51122823]])


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal
        Transport, Advances in Neural Information Processing Systems
        (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for
        Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint
        arXiv:1607.05816.

    .. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. :
        Learning with a Wasserstein Loss,  Advances in Neural Information
        Processing Systems (NIPS) 2015


    See Also
    --------
    ot.unbalanced.sinkhorn_knopp_unbalanced : Unbalanced Classic Sinkhorn [10]
    ot.unbalanced.sinkhorn_stabilized_unbalanced:
        Unbalanced Stabilized sinkhorn [9][10]
    ot.unbalanced.sinkhorn_reg_scaling_unbalanced:
        Unbalanced Sinkhorn with epslilon scaling [9][10]

    """

    if method.lower() == 'sinkhorn':
        return sinkhorn_knopp_unbalanced(a, b, M, reg, reg_m,
                                         numItermax=numItermax,
                                         stopThr=stopThr, verbose=verbose,
                                         log=log, **kwargs)

    elif method.lower() == 'sinkhorn_stabilized':
        return sinkhorn_stabilized_unbalanced(a, b, M, reg, reg_m,
                                              numItermax=numItermax,
                                              stopThr=stopThr,
                                              verbose=verbose,
                                              log=log, **kwargs)
    elif method.lower() in ['sinkhorn_reg_scaling']:
        warnings.warn('Method not implemented yet. Using classic Sinkhorn Knopp')
        return sinkhorn_knopp_unbalanced(a, b, M, reg, reg_m,
                                         numItermax=numItermax,
                                         stopThr=stopThr, verbose=verbose,
                                         log=log, **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn_unbalanced2(a, b, M, reg, reg_m, method='sinkhorn',
                         numItermax=1000, stopThr=1e-6, verbose=False,
                         log=False, **kwargs):
    r"""
    Solve the entropic regularization unbalanced optimal transport problem and
    return the loss

    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma) + reg_m KL(\gamma 1, a) + reg_m KL(\gamma^T 1, b)

        s.t.
             \gamma\geq 0
    where :

    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
        :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target unbalanced distributions
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in [10, 23]_


    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Unnormalized histogram of dimension dim_a
    b : np.ndarray (dim_b,) or np.ndarray (dim_b, n_hists)
        One or multiple unnormalized histograms of dimension dim_b
        If many, compute all the OT distances (a, b_i)
    M : np.ndarray (dim_a, dim_b)
        loss matrix
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    method : str
        method used for the solver either 'sinkhorn',  'sinkhorn_stabilized' or
        'sinkhorn_reg_scaling', see those function for specific parameters
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    ot_distance : (n_hists,) ndarray
        the OT distance between `a` and each of the histograms `b_i`
    log : dict
        log dictionary returned only if `log` is `True`

    Examples
    --------

    >>> import ot
    >>> a=[.5, .10]
    >>> b=[.5, .5]
    >>> M=[[0., 1.],[1., 0.]]
    >>> ot.unbalanced.sinkhorn_unbalanced2(a, b, M, 1., 1.)
    array([0.31912866])



    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal
        Transport, Advances in Neural Information Processing Systems
        (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for
        Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint
        arXiv:1607.05816.

    .. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. :
        Learning with a Wasserstein Loss,  Advances in Neural Information
        Processing Systems (NIPS) 2015

    See Also
    --------
    ot.unbalanced.sinkhorn_knopp : Unbalanced Classic Sinkhorn [10]
    ot.unbalanced.sinkhorn_stabilized: Unbalanced Stabilized sinkhorn [9][10]
    ot.unbalanced.sinkhorn_reg_scaling: Unbalanced Sinkhorn with epslilon scaling [9][10]

    """
    b = np.asarray(b, dtype=np.float64)
    if len(b.shape) < 2:
        b = b[:, None]
    if method.lower() == 'sinkhorn':
        return sinkhorn_knopp_unbalanced(a, b, M, reg, reg_m,
                                         numItermax=numItermax,
                                         stopThr=stopThr, verbose=verbose,
                                         log=log, **kwargs)

    elif method.lower() == 'sinkhorn_stabilized':
        return sinkhorn_stabilized_unbalanced(a, b, M, reg, reg_m,
                                              numItermax=numItermax,
                                              stopThr=stopThr,
                                              verbose=verbose,
                                              log=log, **kwargs)
    elif method.lower() in ['sinkhorn_reg_scaling']:
        warnings.warn('Method not implemented yet. Using classic Sinkhorn Knopp')
        return sinkhorn_knopp_unbalanced(a, b, M, reg, reg_m,
                                         numItermax=numItermax,
                                         stopThr=stopThr, verbose=verbose,
                                         log=log, **kwargs)
    else:
        raise ValueError('Unknown method %s.' % method)


def sinkhorn_knopp_unbalanced(a, b, M, reg, reg_m, numItermax=1000,
                              stopThr=1e-6, verbose=False, log=False, **kwargs):
    r"""
    Solve the entropic regularization unbalanced optimal transport problem and return the loss

    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma) + \reg_m KL(\gamma 1, a) + \reg_m KL(\gamma^T 1, b)

        s.t.
             \gamma\geq 0
    where :

    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target unbalanced distributions
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized Sinkhorn-Knopp matrix scaling algorithm as proposed in [10, 23]_


    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Unnormalized histogram of dimension dim_a
    b : np.ndarray (dim_b,) or np.ndarray (dim_b, n_hists)
        One or multiple unnormalized histograms of dimension dim_b
        If many, compute all the OT distances (a, b_i)
    M : np.ndarray (dim_a, dim_b)
        loss matrix
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    if n_hists == 1:
        gamma : (dim_a x dim_b) ndarray
            Optimal transportation matrix for the given parameters
        log : dict
            log dictionary returned only if `log` is `True`
    else:
        ot_distance : (n_hists,) ndarray
            the OT distance between `a` and each of the histograms `b_i`
        log : dict
            log dictionary returned only if `log` is `True`
    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.],[1., 0.]]
    >>> ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M, 1., 1.)
    array([[0.51122823, 0.18807035],
           [0.18807035, 0.51122823]])

    References
    ----------

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint
        arXiv:1607.05816.

    .. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. :
        Learning with a Wasserstein Loss,  Advances in Neural Information
        Processing Systems (NIPS) 2015

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = np.ones(dim_a, dtype=np.float64) / dim_a
    if len(b) == 0:
        b = np.ones(dim_b, dtype=np.float64) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = np.ones((dim_a, 1)) / dim_a
        v = np.ones((dim_b, n_hists)) / dim_b
        a = a.reshape(dim_a, 1)
    else:
        u = np.ones(dim_a) / dim_a
        v = np.ones(dim_b) / dim_b

    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    fi = reg_m / (reg_m + reg)

    err = 1.

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = K.dot(v)
        u = (a / Kv) ** fi
        Ktu = K.T.dot(u)
        v = (b / Ktu) ** fi

        if (np.any(Ktu == 0.)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            break

        err_u = abs(u - uprev).max() / max(abs(u).max(), abs(uprev).max(), 1.)
        err_v = abs(v - vprev).max() / max(abs(v).max(), abs(vprev).max(), 1.)
        err = 0.5 * (err_u + err_v)
        if log:
            log['err'].append(err)
            if verbose:
                if i % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['logu'] = np.log(u + 1e-300)
        log['logv'] = np.log(v + 1e-300)

    if n_hists:  # return only loss
        res = np.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u[:, None] * K * v[None, :], log
        else:
            return u[:, None] * K * v[None, :]


def sinkhorn_stabilized_unbalanced(a, b, M, reg, reg_m, tau=1e5, numItermax=1000,
                                   stopThr=1e-6, verbose=False, log=False,
                                   **kwargs):
    r"""
    Solve the entropic regularization unbalanced optimal transport
    problem and return the loss

    The function solves the following optimization problem using log-domain
    stabilization as proposed in [10]:

    .. math::
        W = \min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma) + reg_m KL(\gamma 1, a) + reg_m KL(\gamma^T 1, b)

        s.t.
             \gamma\geq 0
    where :

    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization
        term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target unbalanced distributions
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in [10, 23]_


    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Unnormalized histogram of dimension dim_a
    b : np.ndarray (dim_b,) or np.ndarray (dim_b, n_hists)
        One or multiple unnormalized histograms of dimension dim_b
        If many, compute all the OT distances (a, b_i)
    M : np.ndarray (dim_a, dim_b)
        loss matrix
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    tau : float
        thershold for max value in u or v for log scaling
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    if n_hists == 1:
        gamma : (dim_a x dim_b) ndarray
            Optimal transportation matrix for the given parameters
        log : dict
            log dictionary returned only if `log` is `True`
    else:
        ot_distance : (n_hists,) ndarray
            the OT distance between `a` and each of the histograms `b_i`
        log : dict
            log dictionary returned only if `log` is `True`
    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.],[1., 0.]]
    >>> ot.unbalanced.sinkhorn_stabilized_unbalanced(a, b, M, 1., 1.)
    array([[0.51122823, 0.18807035],
           [0.18807035, 0.51122823]])

    References
    ----------

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

    .. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. :
        Learning with a Wasserstein Loss,  Advances in Neural Information
        Processing Systems (NIPS) 2015

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = np.ones(dim_a, dtype=np.float64) / dim_a
    if len(b) == 0:
        b = np.ones(dim_b, dtype=np.float64) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = np.ones((dim_a, n_hists)) / dim_a
        v = np.ones((dim_b, n_hists)) / dim_b
        a = a.reshape(dim_a, 1)
    else:
        u = np.ones(dim_a) / dim_a
        v = np.ones(dim_b) / dim_b

    # print(reg)
    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    fi = reg_m / (reg_m + reg)

    cpt = 0
    err = 1.
    alpha = np.zeros(dim_a)
    beta = np.zeros(dim_b)
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v

        Kv = K.dot(v)
        f_alpha = np.exp(- alpha / (reg + reg_m))
        f_beta = np.exp(- beta / (reg + reg_m))

        if n_hists:
            f_alpha = f_alpha[:, None]
            f_beta = f_beta[:, None]
        u = ((a / (Kv + 1e-16)) ** fi) * f_alpha
        Ktu = K.T.dot(u)
        v = ((b / (Ktu + 1e-16)) ** fi) * f_beta
        absorbing = False
        if (u > tau).any() or (v > tau).any():
            absorbing = True
            if n_hists:
                alpha = alpha + reg * np.log(np.max(u, 1))
                beta = beta + reg * np.log(np.max(v, 1))
            else:
                alpha = alpha + reg * np.log(np.max(u))
                beta = beta + reg * np.log(np.max(v))
            K = np.exp((alpha[:, None] + beta[None, :] -
                        M) / reg)
            v = np.ones_like(v)
        Kv = K.dot(v)

        if (np.any(Ktu == 0.)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % cpt)
            u = uprev
            v = vprev
            break
        if (cpt % 10 == 0 and not absorbing) or cpt == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = abs(u - uprev).max() / max(abs(u).max(), abs(uprev).max(),
                                             1.)
            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1

    if err > stopThr:
        warnings.warn("Stabilized Unbalanced Sinkhorn did not converge." +
                      "Try a larger entropy `reg` or a lower mass `reg_m`." +
                      "Or a larger absorption threshold `tau`.")
    if n_hists:
        logu = alpha[:, None] / reg + np.log(u)
        logv = beta[:, None] / reg + np.log(v)
    else:
        logu = alpha / reg + np.log(u)
        logv = beta / reg + np.log(v)
    if log:
        log['logu'] = logu
        log['logv'] = logv
    if n_hists:  # return only loss
        res = logsumexp(np.log(M + 1e-100)[:, :, None] + logu[:, None, :] +
                        logv[None, :, :] - M[:, :, None] / reg, axis=(0, 1))
        res = np.exp(res)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix
        ot_matrix = np.exp(logu[:, None] + logv[None, :] - M / reg)
        if log:
            return ot_matrix, log
        else:
            return ot_matrix


def barycenter_unbalanced_stabilized(A, M, reg, reg_m, weights=None, tau=1e3,
                                     numItermax=1000, stopThr=1e-6,
                                     verbose=False, log=False):
    r"""Compute the entropic unbalanced wasserstein barycenter of A with stabilization.

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i Wu_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`Wu_{reg}(\cdot,\cdot)` is the unbalanced entropic regularized
        Wasserstein distance (see ot.unbalanced.sinkhorn_unbalanced)
    - :math:`\mathbf{a}_i` are training distributions in the columns of
        matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and
        the cost matrix for OT
    - reg_mis the marginal relaxation hyperparameter
        The algorithm used for solving the problem is the generalized
        Sinkhorn-Knopp matrix scaling algorithm as proposed in [10]_

    Parameters
    ----------
    A : np.ndarray (dim, n_hists)
        `n_hists` training distributions a_i of dimension dim
    M : np.ndarray (dim, dim)
        ground metric matrix for OT.
    reg : float
        Entropy regularization term > 0
    reg_m : float
        Marginal relaxation term > 0
    tau : float
        Stabilization threshold for log domain absorption.
    weights : np.ndarray (n_hists,) optional
        Weight of each distribution (barycentric coodinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    a : (dim,) ndarray
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré,
        G. (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint
        arXiv:1607.05816.


    """
    dim, n_hists = A.shape
    if weights is None:
        weights = np.ones(n_hists) / n_hists
    else:
        assert(len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    fi = reg_m / (reg_m + reg)

    u = np.ones((dim, n_hists)) / dim
    v = np.ones((dim, n_hists)) / dim

    # print(reg)
    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    fi = reg_m / (reg_m + reg)

    cpt = 0
    err = 1.
    alpha = np.zeros(dim)
    beta = np.zeros(dim)
    q = np.ones(dim) / dim
    for i in range(numItermax):
        qprev = q.copy()
        Kv = K.dot(v)
        f_alpha = np.exp(- alpha / (reg + reg_m))
        f_beta = np.exp(- beta / (reg + reg_m))
        f_alpha = f_alpha[:, None]
        f_beta = f_beta[:, None]
        u = ((A / (Kv + 1e-16)) ** fi) * f_alpha
        Ktu = K.T.dot(u)
        q = (Ktu ** (1 - fi)) * f_beta
        q = q.dot(weights) ** (1 / (1 - fi))
        Q = q[:, None]
        v = ((Q / (Ktu + 1e-16)) ** fi) * f_beta
        absorbing = False
        if (u > tau).any() or (v > tau).any():
            absorbing = True
            alpha = alpha + reg * np.log(np.max(u, 1))
            beta = beta + reg * np.log(np.max(v, 1))
            K = np.exp((alpha[:, None] + beta[None, :] -
                        M) / reg)
            v = np.ones_like(v)
        Kv = K.dot(v)
        if (np.any(Ktu == 0.)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % cpt)
            q = qprev
            break
        if (i % 10 == 0 and not absorbing) or i == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = abs(q - qprev).max() / max(abs(q).max(),
                                             abs(qprev).max(), 1.)
            if log:
                log['err'].append(err)
            if verbose:
                if i % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(i, err))
            if err < stopThr:
                break

    if err > stopThr:
        warnings.warn("Stabilized Unbalanced Sinkhorn did not converge." +
                      "Try a larger entropy `reg` or a lower mass `reg_m`." +
                      "Or a larger absorption threshold `tau`.")
    if log:
        log['niter'] = i
        log['logu'] = np.log(u + 1e-300)
        log['logv'] = np.log(v + 1e-300)
        return q, log
    else:
        return q


def barycenter_unbalanced_sinkhorn(A, M, reg, reg_m, weights=None,
                                   numItermax=1000, stopThr=1e-6,
                                   verbose=False, log=False):
    r"""Compute the entropic unbalanced wasserstein barycenter of A.

     The function solves the following optimization problem with a

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i Wu_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`Wu_{reg}(\cdot,\cdot)` is the unbalanced entropic regularized
    Wasserstein distance (see ot.unbalanced.sinkhorn_unbalanced)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix
    :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and
    the cost matrix for OT
    - reg_mis the marginal relaxation hyperparameter
    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in [10]_

    Parameters
    ----------
    A : np.ndarray (dim, n_hists)
        `n_hists` training distributions a_i of dimension dim
    M : np.ndarray (dim, dim)
        ground metric matrix for OT.
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    weights : np.ndarray (n_hists,) optional
        Weight of each distribution (barycentric coodinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    a : (dim,) ndarray
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
        (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprin
        arXiv:1607.05816.


    """
    dim, n_hists = A.shape
    if weights is None:
        weights = np.ones(n_hists) / n_hists
    else:
        assert(len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    K = np.exp(- M / reg)

    fi = reg_m / (reg_m + reg)

    v = np.ones((dim, n_hists))
    u = np.ones((dim, 1))
    q = np.ones(dim)
    err = 1.

    for i in range(numItermax):
        uprev = u.copy()
        vprev = v.copy()
        qprev = q.copy()

        Kv = K.dot(v)
        u = (A / Kv) ** fi
        Ktu = K.T.dot(u)
        q = ((Ktu ** (1 - fi)).dot(weights))
        q = q ** (1 / (1 - fi))
        Q = q[:, None]
        v = (Q / Ktu) ** fi

        if (np.any(Ktu == 0.)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            q = qprev
            break
            # compute change in barycenter
        err = abs(q - qprev).max()
        err /= max(abs(q).max(), abs(qprev).max(), 1.)
        if log:
            log['err'].append(err)
        # if barycenter did not change + at least 10 iterations - stop
        if err < stopThr and i > 10:
            break

        if verbose:
            if i % 10 == 0:
                print(
                    '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(i, err))

    if log:
        log['niter'] = i
        log['logu'] = np.log(u + 1e-300)
        log['logv'] = np.log(v + 1e-300)
        return q, log
    else:
        return q


def barycenter_unbalanced(A, M, reg, reg_m, method="sinkhorn", weights=None,
                          numItermax=1000, stopThr=1e-6,
                          verbose=False, log=False, **kwargs):
    r"""Compute the entropic unbalanced wasserstein barycenter of A.

     The function solves the following optimization problem with a

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i Wu_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`Wu_{reg}(\cdot,\cdot)` is the unbalanced entropic regularized
    Wasserstein distance (see ot.unbalanced.sinkhorn_unbalanced)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix
    :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and
    the cost matrix for OT
    - reg_mis the marginal relaxation hyperparameter
    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in [10]_

    Parameters
    ----------
    A : np.ndarray (dim, n_hists)
        `n_hists` training distributions a_i of dimension dim
    M : np.ndarray (dim, dim)
        ground metric matrix for OT.
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    weights : np.ndarray (n_hists,) optional
        Weight of each distribution (barycentric coodinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    a : (dim,) ndarray
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
        (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprin
        arXiv:1607.05816.

    """

    if method.lower() == 'sinkhorn':
        return barycenter_unbalanced_sinkhorn(A, M, reg, reg_m,
                                              weights=weights,
                                              numItermax=numItermax,
                                              stopThr=stopThr, verbose=verbose,
                                              log=log, **kwargs)

    elif method.lower() == 'sinkhorn_stabilized':
        return barycenter_unbalanced_stabilized(A, M, reg, reg_m,
                                                weights=weights,
                                                numItermax=numItermax,
                                                stopThr=stopThr,
                                                verbose=verbose,
                                                log=log, **kwargs)
    elif method.lower() in ['sinkhorn_reg_scaling']:
        warnings.warn('Method not implemented yet. Using classic Sinkhorn Knopp')
        return barycenter_unbalanced(A, M, reg, reg_m,
                                     weights=weights,
                                     numItermax=numItermax,
                                     stopThr=stopThr, verbose=verbose,
                                     log=log, **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)
