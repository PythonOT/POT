# -*- coding: utf-8 -*-
"""
Regularized Unbalanced OT solvers
"""

# Author: Hicham Janati <hicham.janati@inria.fr>
#         Laetitia Chapel <laetitia.chapel@univ-ubs.fr>
# License: MIT License

from __future__ import division
import warnings

import numpy as np
from scipy.optimize import minimize, Bounds

from .backend import get_backend
from .utils import list_to_array
# from .utils import unif, dist


def sinkhorn_unbalanced(a, b, M, reg, reg_m, method='sinkhorn', numItermax=1000,
                        stopThr=1e-6, verbose=False, log=False, **kwargs):
    r"""
    Solve the unbalanced entropic regularization optimal transport problem
    and return the OT plan

    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \ \langle \gamma, \mathbf{M} \rangle_F + \mathrm{reg}\cdot\Omega(\gamma) +
        \mathrm{reg_m} \cdot \mathrm{KL}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{KL}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10, 25] <references-sinkhorn-unbalanced>`


    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,) or array-like (dim_b, n_hists)
        One or multiple unnormalized histograms of dimension `dim_b`.
        If many, compute all the OT distances :math:`(\mathbf{a}, \mathbf{b}_i)_i`
    M : array-like (dim_a, dim_b)
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
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    if n_hists == 1:
        - gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
        - log : dict
            log dictionary returned only if `log` is `True`
    else:
        - ot_distance : (n_hists,) array-like
            the OT distance between :math:`\mathbf{a}` and each of the histograms :math:`\mathbf{b}_i`
        - log : dict
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


    .. _references-sinkhorn-unbalanced:
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
    ot.unbalanced.sinkhorn_knopp_unbalanced : Unbalanced Classic Sinkhorn :ref:`[10] <references-sinkhorn-unbalanced>`
    ot.unbalanced.sinkhorn_stabilized_unbalanced:
        Unbalanced Stabilized sinkhorn :ref:`[9, 10] <references-sinkhorn-unbalanced>`
    ot.unbalanced.sinkhorn_reg_scaling_unbalanced:
        Unbalanced Sinkhorn with epslilon scaling :ref:`[9, 10] <references-sinkhorn-unbalanced>`

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
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F + \mathrm{reg}\cdot\Omega(\gamma) +
        \mathrm{reg_m} \cdot \mathrm{KL}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{KL}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma\geq 0
    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10, 25] <references-sinkhorn-unbalanced2>`


    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,) or array-like (dim_b, n_hists)
        One or multiple unnormalized histograms of dimension `dim_b`.
        If many, compute all the OT distances :math:`(\mathbf{a}, \mathbf{b}_i)_i`
    M : array-like (dim_a, dim_b)
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
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    ot_distance : (n_hists,) array-like
        the OT distance between :math:`\mathbf{a}` and each of the histograms :math:`\mathbf{b}_i`
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


    .. _references-sinkhorn-unbalanced2:
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
    ot.unbalanced.sinkhorn_knopp : Unbalanced Classic Sinkhorn :ref:`[10] <references-sinkhorn-unbalanced2>`
    ot.unbalanced.sinkhorn_stabilized: Unbalanced Stabilized sinkhorn :ref:`[9, 10] <references-sinkhorn-unbalanced2>`
    ot.unbalanced.sinkhorn_reg_scaling: Unbalanced Sinkhorn with epslilon scaling :ref:`[9, 10] <references-sinkhorn-unbalanced2>`

    """
    b = list_to_array(b)
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
    Solve the entropic regularization unbalanced optimal transport problem and
    return the OT plan

    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F + \mathrm{reg}\cdot\Omega(\gamma) +
        \mathrm{reg_m} \cdot \mathrm{KL}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{KL}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10, 25] <references-sinkhorn-knopp-unbalanced>`


    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,) or array-like (dim_b, n_hists)
        One or multiple unnormalized histograms of dimension `dim_b`
        If many, compute all the OT distances (a, b_i)
    M : array-like (dim_a, dim_b)
        loss matrix
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    if n_hists == 1:
        - gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
        - log : dict
            log dictionary returned only if `log` is `True`
    else:
        - ot_distance : (n_hists,) array-like
            the OT distance between :math:`\mathbf{a}` and each of the histograms :math:`\mathbf{b}_i`
        - log : dict
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


    .. _references-sinkhorn-knopp-unbalanced:
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
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = nx.ones((dim_a, 1), type_as=M) / dim_a
        v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
        a = a.reshape(dim_a, 1)
    else:
        u = nx.ones(dim_a, type_as=M) / dim_a
        v = nx.ones(dim_b, type_as=M) / dim_b

    K = nx.exp(M / (-reg))

    fi = reg_m / (reg_m + reg)

    err = 1.

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = nx.dot(K, v)
        u = (a / Kv) ** fi
        Ktu = nx.dot(K.T, u)
        v = (b / Ktu) ** fi

        if (nx.any(Ktu == 0.)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            break

        err_u = nx.max(nx.abs(u - uprev)) / max(
            nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.
        )
        err_v = nx.max(nx.abs(v - vprev)) / max(
            nx.max(nx.abs(v)), nx.max(nx.abs(vprev)), 1.
        )
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
        log['logu'] = nx.log(u + 1e-300)
        log['logv'] = nx.log(v + 1e-300)

    if n_hists:  # return only loss
        res = nx.einsum('ik,ij,jk,ij->k', u, K, v, M)
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
    stabilization as proposed in :ref:`[10] <references-sinkhorn-stabilized-unbalanced>`:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F + \mathrm{reg}\cdot\Omega(\gamma) +
        \mathrm{reg_m} \cdot \mathrm{KL}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{KL}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10, 25] <references-sinkhorn-stabilized-unbalanced>`


    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,) or array-like (dim_b, n_hists)
        One or multiple unnormalized histograms of dimension `dim_b`.
        If many, compute all the OT distances :math:`(\mathbf{a}, \mathbf{b}_i)_i`
    M : array-like (dim_a, dim_b)
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
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    if n_hists == 1:
        - gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
        - log : dict
            log dictionary returned only if `log` is `True`
    else:
        - ot_distance : (n_hists,) array-like
            the OT distance between :math:`\mathbf{a}` and each of the histograms :math:`\mathbf{b}_i`
        - log : dict
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


    .. _references-sinkhorn-stabilized-unbalanced:
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
    a, b, M = list_to_array(a, b, M)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
        v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
        a = a.reshape(dim_a, 1)
    else:
        u = nx.ones(dim_a, type_as=M) / dim_a
        v = nx.ones(dim_b, type_as=M) / dim_b

    # print(reg)
    K = nx.exp(-M / reg)

    fi = reg_m / (reg_m + reg)

    cpt = 0
    err = 1.
    alpha = nx.zeros(dim_a, type_as=M)
    beta = nx.zeros(dim_b, type_as=M)
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v

        Kv = nx.dot(K, v)
        f_alpha = nx.exp(- alpha / (reg + reg_m))
        f_beta = nx.exp(- beta / (reg + reg_m))

        if n_hists:
            f_alpha = f_alpha[:, None]
            f_beta = f_beta[:, None]
        u = ((a / (Kv + 1e-16)) ** fi) * f_alpha
        Ktu = nx.dot(K.T, u)
        v = ((b / (Ktu + 1e-16)) ** fi) * f_beta
        absorbing = False
        if nx.any(u > tau) or nx.any(v > tau):
            absorbing = True
            if n_hists:
                alpha = alpha + reg * nx.log(nx.max(u, 1))
                beta = beta + reg * nx.log(nx.max(v, 1))
            else:
                alpha = alpha + reg * nx.log(nx.max(u))
                beta = beta + reg * nx.log(nx.max(v))
            K = nx.exp((alpha[:, None] + beta[None, :] - M) / reg)
            v = nx.ones(v.shape, type_as=v)
        Kv = nx.dot(K, v)

        if (nx.any(Ktu == 0.)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % cpt)
            u = uprev
            v = vprev
            break
        if (cpt % 10 == 0 and not absorbing) or cpt == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.max(nx.abs(u - uprev)) / max(
                nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.
            )
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
        logu = alpha[:, None] / reg + nx.log(u)
        logv = beta[:, None] / reg + nx.log(v)
    else:
        logu = alpha / reg + nx.log(u)
        logv = beta / reg + nx.log(v)
    if log:
        log['logu'] = logu
        log['logv'] = logv
    if n_hists:  # return only loss
        res = nx.logsumexp(
            nx.log(M + 1e-100)[:, :, None]
            + logu[:, None, :]
            + logv[None, :, :]
            - M[:, :, None] / reg,
            axis=(0, 1)
        )
        res = nx.exp(res)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix
        ot_matrix = nx.exp(logu[:, None] + logv[None, :] - M / reg)
        if log:
            return ot_matrix, log
        else:
            return ot_matrix


def barycenter_unbalanced_stabilized(A, M, reg, reg_m, weights=None, tau=1e3,
                                     numItermax=1000, stopThr=1e-6,
                                     verbose=False, log=False):
    r"""Compute the entropic unbalanced wasserstein barycenter of :math:`\mathbf{A}` with stabilization.

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{u_{reg}}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{u_{reg}}(\cdot,\cdot)` is the unbalanced entropic regularized Wasserstein distance (see :py:func:`ot.unbalanced.sinkhorn_unbalanced`)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT
    - reg_mis the marginal relaxation hyperparameter

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10] <references-barycenter-unbalanced-stabilized>`

    Parameters
    ----------
    A : array-like (dim, n_hists)
        `n_hists` training distributions :math:`\mathbf{a}_i` of dimension `dim`
    M : array-like (dim, dim)
        ground metric matrix for OT.
    reg : float
        Entropy regularization term > 0
    reg_m : float
        Marginal relaxation term > 0
    tau : float
        Stabilization threshold for log domain absorption.
    weights : array-like (n_hists,) optional
        Weight of each distribution (barycentric coodinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    a : (dim,) array-like
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-barycenter-unbalanced-stabilized:
    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré,
        G. (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint
        arXiv:1607.05816.


    """
    A, M = list_to_array(A, M)
    nx = get_backend(A, M)

    dim, n_hists = A.shape
    if weights is None:
        weights = nx.ones(n_hists, type_as=A) / n_hists
    else:
        assert (len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    fi = reg_m / (reg_m + reg)

    u = nx.ones((dim, n_hists), type_as=A) / dim
    v = nx.ones((dim, n_hists), type_as=A) / dim

    # print(reg)
    K = nx.exp(-M / reg)

    fi = reg_m / (reg_m + reg)

    cpt = 0
    err = 1.
    alpha = nx.zeros(dim, type_as=A)
    beta = nx.zeros(dim, type_as=A)
    q = nx.ones(dim, type_as=A) / dim
    for i in range(numItermax):
        qprev = nx.copy(q)
        Kv = nx.dot(K, v)
        f_alpha = nx.exp(- alpha / (reg + reg_m))
        f_beta = nx.exp(- beta / (reg + reg_m))
        f_alpha = f_alpha[:, None]
        f_beta = f_beta[:, None]
        u = ((A / (Kv + 1e-16)) ** fi) * f_alpha
        Ktu = nx.dot(K.T, u)
        q = (Ktu ** (1 - fi)) * f_beta
        q = nx.dot(q, weights) ** (1 / (1 - fi))
        Q = q[:, None]
        v = ((Q / (Ktu + 1e-16)) ** fi) * f_beta
        absorbing = False
        if nx.any(u > tau) or nx.any(v > tau):
            absorbing = True
            alpha = alpha + reg * nx.log(nx.max(u, 1))
            beta = beta + reg * nx.log(nx.max(v, 1))
            K = nx.exp((alpha[:, None] + beta[None, :] - M) / reg)
            v = nx.ones(v.shape, type_as=v)
        Kv = nx.dot(K, v)
        if (nx.any(Ktu == 0.)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % cpt)
            q = qprev
            break
        if (i % 10 == 0 and not absorbing) or i == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.max(nx.abs(q - qprev)) / max(
                nx.max(nx.abs(q)), nx.max(nx.abs(qprev)), 1.
            )
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
        log['logu'] = nx.log(u + 1e-300)
        log['logv'] = nx.log(v + 1e-300)
        return q, log
    else:
        return q


def barycenter_unbalanced_sinkhorn(A, M, reg, reg_m, weights=None,
                                   numItermax=1000, stopThr=1e-6,
                                   verbose=False, log=False):
    r"""Compute the entropic unbalanced wasserstein barycenter of :math:`\mathbf{A}`.

     The function solves the following optimization problem with :math:`\mathbf{a}`

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{u_{reg}}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{u_{reg}}(\cdot,\cdot)` is the unbalanced entropic regularized Wasserstein distance (see :py:func:`ot.unbalanced.sinkhorn_unbalanced`)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT
    - reg_mis the marginal relaxation hyperparameter

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10] <references-barycenter-unbalanced-sinkhorn>`

    Parameters
    ----------
    A : array-like (dim, n_hists)
        `n_hists` training distributions :math:`\mathbf{a}_i` of dimension `dim`
    M : array-like (dim, dim)
        ground metric matrix for OT.
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    weights : array-like (n_hists,) optional
        Weight of each distribution (barycentric coodinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    a : (dim,) array-like
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-barycenter-unbalanced-sinkhorn:
    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
        (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprin
        arXiv:1607.05816.


    """
    A, M = list_to_array(A, M)
    nx = get_backend(A, M)

    dim, n_hists = A.shape
    if weights is None:
        weights = nx.ones(n_hists, type_as=A) / n_hists
    else:
        assert (len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    K = nx.exp(-M / reg)

    fi = reg_m / (reg_m + reg)

    v = nx.ones((dim, n_hists), type_as=A)
    u = nx.ones((dim, 1), type_as=A)
    q = nx.ones(dim, type_as=A)
    err = 1.

    for i in range(numItermax):
        uprev = nx.copy(u)
        vprev = nx.copy(v)
        qprev = nx.copy(q)

        Kv = nx.dot(K, v)
        u = (A / Kv) ** fi
        Ktu = nx.dot(K.T, u)
        q = nx.dot(Ktu ** (1 - fi), weights)
        q = q ** (1 / (1 - fi))
        Q = q[:, None]
        v = (Q / Ktu) ** fi

        if (nx.any(Ktu == 0.)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            q = qprev
            break
            # compute change in barycenter
        err = nx.max(nx.abs(q - qprev)) / max(
            nx.max(nx.abs(q)), nx.max(nx.abs(qprev)), 1.0
        )
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
        log['logu'] = nx.log(u + 1e-300)
        log['logv'] = nx.log(v + 1e-300)
        return q, log
    else:
        return q


def barycenter_unbalanced(A, M, reg, reg_m, method="sinkhorn", weights=None,
                          numItermax=1000, stopThr=1e-6,
                          verbose=False, log=False, **kwargs):
    r"""Compute the entropic unbalanced wasserstein barycenter of :math:`\mathbf{A}`.

     The function solves the following optimization problem with :math:`\mathbf{a}`

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{u_{reg}}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{u_{reg}}(\cdot,\cdot)` is the unbalanced entropic regularized Wasserstein distance (see :py:func:`ot.unbalanced.sinkhorn_unbalanced`)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT
    - reg_mis the marginal relaxation hyperparameter

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10] <references-barycenter-unbalanced>`

    Parameters
    ----------
    A : array-like (dim, n_hists)
        `n_hists` training distributions :math:`\mathbf{a}_i` of dimension `dim`
    M : array-like (dim, dim)
        ground metric matrix for OT.
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    weights : array-like (n_hists,) optional
        Weight of each distribution (barycentric coodinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    a : (dim,) array-like
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-barycenter-unbalanced:
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


def mm_unbalanced(a, b, M, reg_m, div='kl', G0=None, numItermax=1000,
                  stopThr=1e-15, verbose=False, log=False):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan.
    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma^T \mathbf{1}, \mathbf{b})
        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      unbalanced distributions
    - div is a divergence, either Kullback-Leibler or :math:`\ell_2` divergence

    The algorithm used for solving the problem is a maximization-
    minimization algorithm as proposed in :ref:`[41] <references-regpath>`

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
    M : array-like (dim_a, dim_b)
        loss matrix
    reg_m: float
        Marginal relaxation term > 0
    div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic)
    G0: array-like (dim_a, dim_b)
        Initialization of the transport matrix
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
    log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[1., 36.],[9., 4.]]
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'kl'), 2)
    array([[0.3 , 0.  ],
           [0.  , 0.07]])
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'l2'), 2)
    array([[0.25, 0.  ],
           [0.  , 0.  ]])


    .. _references-regpath:
    References
    ----------
    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.
    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.unbalanced.sinkhorn_unbalanced : Entropic regularized OT
    """
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if G0 is None:
        G = a[:, None] * b[None, :]
    else:
        G = G0

    if log:
        log = {'err': [], 'G': []}

    if div == 'kl':
        K = nx.exp(M / - reg_m / 2)
    elif div == 'l2':
        K = nx.maximum(a[:, None] + b[None, :] - M / reg_m / 2,
                       nx.zeros((dim_a, dim_b), type_as=M))
    else:
        warnings.warn("The div parameter should be either equal to 'kl' or \
                      'l2': it has been set to 'kl'.")
        div = 'kl'
        K = nx.exp(M / - reg_m / 2)

    for i in range(numItermax):
        Gprev = G

        if div == 'kl':
            u = nx.sqrt(a / (nx.sum(G, 1) + 1e-16))
            v = nx.sqrt(b / (nx.sum(G, 0) + 1e-16))
            G = G * K * u[:, None] * v[None, :]
        elif div == 'l2':
            Gd = nx.sum(G, 0, keepdims=True) + nx.sum(G, 1, keepdims=True) + 1e-16
            G = G * K / Gd

        err = nx.sqrt(nx.sum((G - Gprev) ** 2))
        if log:
            log['err'].append(err)
            log['G'].append(G)
        if verbose:
            print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['cost'] = nx.sum(G * M)
        return G, log
    else:
        return G


def mm_unbalanced2(a, b, M, reg_m, div='kl', G0=None, numItermax=1000,
                   stopThr=1e-15, verbose=False, log=False):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan.
    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      unbalanced distributions
    - :math:`\mathrm{div}` is a divergence, either Kullback-Leibler or :math:`\ell_2` divergence

    The algorithm used for solving the problem is a maximization-
    minimization algorithm as proposed in :ref:`[41] <references-regpath>`

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
    M : array-like (dim_a, dim_b)
        loss matrix
    reg_m: float
        Marginal relaxation term > 0
    div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic)
    G0: array-like (dim_a, dim_b)
        Initialization of the transport matrix
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    ot_distance : array-like
        the OT distance between :math:`\mathbf{a}` and :math:`\mathbf{b}`
    log : dict
        log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[1., 36.],[9., 4.]]
    >>> np.round(ot.unbalanced.mm_unbalanced2(a, b, M, 1, 'l2'),2)
    0.25
    >>> np.round(ot.unbalanced.mm_unbalanced2(a, b, M, 1, 'kl'),2)
    0.57

    References
    ----------
    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.
    See Also
    --------
    ot.lp.emd2 : Unregularized OT loss
    ot.unbalanced.sinkhorn_unbalanced2 : Entropic regularized OT loss
    """
    _, log_mm = mm_unbalanced(a, b, M, reg_m, div=div, G0=G0,
                              numItermax=numItermax, stopThr=stopThr,
                              verbose=verbose, log=True)

    if log:
        return log_mm['cost'], log_mm
    else:
        return log_mm['cost']


def _get_loss_unbalanced(a, b, M, reg, reg_m, reg_div='kl', regm_div='kl'):
    """
    return the loss function (scipy.optimize compatible) for regularized
    unbalanced OT
    """

    m, n = M.shape

    def kl(p, q):
        return np.sum(p * np.log(p / q + 1e-16))

    def reg_l2(G):
        return np.sum((G - a[:, None] * b[None, :])**2) / 2

    def grad_l2(G):
        return G - a[:, None] * b[None, :]

    def reg_kl(G):
        return kl(G, a[:, None] * b[None, :])

    def grad_kl(G):
        return np.log(G / (a[:, None] * b[None, :]) + 1e-16) + 1

    def reg_entropy(G):
        return kl(G, 1)

    def grad_entropy(G):
        return np.log(G + 1e-16) + 1

    if reg_div == 'kl':
        reg_fun = reg_kl
        grad_reg_fun = grad_kl
    elif reg_div == 'entropy':
        reg_fun = reg_entropy
        grad_reg_fun = grad_entropy
    else:
        reg_fun = reg_l2
        grad_reg_fun = grad_l2

    def marg_l2(G):
        return 0.5 * np.sum((G.sum(1) - a)**2) + 0.5 * np.sum((G.sum(0) - b)**2)

    def grad_marg_l2(G):
        return np.outer((G.sum(1) - a), np.ones(n)) + np.outer(np.ones(m), (G.sum(0) - b))

    def marg_kl(G):
        return kl(G.sum(1), a) + kl(G.sum(0), b)

    def grad_marg_kl(G):
        return np.outer(np.log(G.sum(1) / a + 1e-16) + 1, np.ones(n)) + np.outer(np.ones(m), np.log(G.sum(0) / b + 1e-16) + 1)

    if regm_div == 'kl':
        regm_fun = marg_kl
        grad_regm_fun = grad_marg_kl
    else:
        regm_fun = marg_l2
        grad_regm_fun = grad_marg_l2

    def _func(G):
        G = G.reshape((m, n))

        # compute loss
        val = np.sum(G * M) + reg * reg_fun(G) + reg_m * regm_fun(G)

        # compute gradient
        grad = M + reg * grad_reg_fun(G) + reg_m * grad_regm_fun(G)

        return val, grad.ravel()

    return _func


def lbfgsb_unbalanced(a, b, M, reg, reg_m, reg_div='kl', regm_div='kl', G0=None, numItermax=1000,
                      stopThr=1e-15, method='L-BFGS-B', verbose=False, log=False):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan using L-BFGS-B.
    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        + \mathrm{reg} \mathrm{div}(\gamma,\mathbf{a}\mathbf{b}^T)
        \mathrm{reg_m} \cdot \mathrm{div_m}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      unbalanced distributions
    - :math:`\mathrm{div}` is a divergence, either Kullback-Leibler or :math:`\ell_2` divergence

    The algorithm used for solving the problem is a L-BFGS-B from scipy.optimize

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
    M : array-like (dim_a, dim_b)
        loss matrix
    reg: float
        regularization term (>=0)
    reg_m: float
        Marginal relaxation term >= 0
    reg_div: string, optional
        Divergence used for regularization.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic)
    reg_div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic)
    G0: array-like (dim_a, dim_b)
        Initialization of the transport matrix
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    ot_distance : array-like
        the OT distance between :math:`\mathbf{a}` and :math:`\mathbf{b}`
    log : dict
        log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[1., 36.],[9., 4.]]
    >>> np.round(ot.unbalanced.mm_unbalanced2(a, b, M, 1, 'l2'),2)
    0.25
    >>> np.round(ot.unbalanced.mm_unbalanced2(a, b, M, 1, 'kl'),2)
    0.57

    References
    ----------
    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.
    See Also
    --------
    ot.lp.emd2 : Unregularized OT loss
    ot.unbalanced.sinkhorn_unbalanced2 : Entropic regularized OT loss
    """
    nx = get_backend(M, a, b)

    M0 = M
    # convert to humpy
    a, b, M = nx.to_numpy(a, b, M)

    if G0 is not None:
        G0 = nx.to_numpy(G0)
    else:
        G0 = np.zeros(M.shape)

    _func = _get_loss_unbalanced(a, b, M, reg, reg_m, reg_div, regm_div)

    res = minimize(_func, G0.ravel(), method=method, jac=True, bounds=Bounds(0, np.inf),
                   tol=stopThr, options=dict(maxiter=numItermax, disp=verbose))

    G = nx.from_numpy(res.x.reshape(M.shape), type_as=M0)

    if log:
        log = {'loss': nx.from_numpy(res.fun, type_as=M0), 'res': res}
        return G, log
    else:
        return G
