# -*- coding: utf-8 -*-
"""
Regularized Unbalanced OT
"""

# Author: Hicham Janati <hicham.janati@inria.fr>
# License: MIT License

from __future__ import division
import warnings
import numpy as np
# from .utils import unif, dist


def sinkhorn_unbalanced(a, b, M, reg, alpha, method='sinkhorn', numItermax=1000,
                        stopThr=1e-9, verbose=False, log=False, **kwargs):
    r"""
    Solve the unbalanced entropic regularization optimal transport problem and return the loss

    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma) + \\alpha KL(\gamma 1, a) + \\alpha KL(\gamma^T 1, b)

        s.t.
             \gamma\geq 0
    where :

    - M is the (ns, nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized Sinkhorn-Knopp matrix scaling algorithm as proposed in [10, 23]_


    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,) or np.ndarray (nt,n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    M : np.ndarray (ns, nt)
        loss matrix
    reg : float
        Entropy regularization term > 0
    alpha : float
        Marginal relaxation term > 0
    method : str
        method used for the solver either 'sinkhorn',  'sinkhorn_stabilized' or
        'sinkhorn_epsilon_scaling', see those function for specific parameters
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
    W : (nt) ndarray or float
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

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

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

    .. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. : Learning with a Wasserstein Loss,  Advances in Neural Information Processing Systems (NIPS) 2015


    See Also
    --------
    ot.unbalanced.sinkhorn_knopp_unbalanced : Unbalanced Classic Sinkhorn [10]
    ot.unbalanced.sinkhorn_stabilized_unbalanced: Unbalanced Stabilized sinkhorn [9][10]
    ot.unbalanced.sinkhorn_epsilon_scaling_unbalanced: Unbalanced Sinkhorn with epslilon scaling [9][10]

    """

    if method.lower() == 'sinkhorn':
        def sink():
            return sinkhorn_knopp_unbalanced(a, b, M, reg, alpha,
                                             numItermax=numItermax,
                                             stopThr=stopThr, verbose=verbose,
                                             log=log, **kwargs)

    elif method.lower() in ['sinkhorn_stabilized', 'sinkhorn_epsilon_scaling']:
        warnings.warn('Method not implemented yet. Using classic Sinkhorn Knopp')

        def sink():
            return sinkhorn_knopp_unbalanced(a, b, M, reg, alpha,
                                             numItermax=numItermax,
                                             stopThr=stopThr, verbose=verbose,
                                             log=log, **kwargs)
    else:
        raise ValueError('Unknown method. Using classic Sinkhorn Knopp')

    return sink()


def sinkhorn_unbalanced2(a, b, M, reg, alpha, method='sinkhorn',
                         numItermax=1000, stopThr=1e-9, verbose=False,
                         log=False, **kwargs):
    r"""
    Solve the entropic regularization unbalanced optimal transport problem and return the loss

    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma) + \\alpha KL(\gamma 1, a) + \\alpha KL(\gamma^T 1, b)

        s.t.
             \gamma\geq 0
    where :

    - M is the (ns, nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized Sinkhorn-Knopp matrix scaling algorithm as proposed in [10, 23]_


    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,) or np.ndarray (nt, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    M : np.ndarray (ns,nt)
        loss matrix
    reg : float
        Entropy regularization term > 0
    alpha : float
        Marginal relaxation term > 0
    method : str
        method used for the solver either 'sinkhorn',  'sinkhorn_stabilized' or
        'sinkhorn_epsilon_scaling', see those function for specific parameters
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
    W : (nt) ndarray or float
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

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

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

    .. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. : Learning with a Wasserstein Loss,  Advances in Neural Information Processing Systems (NIPS) 2015

    See Also
    --------
    ot.unbalanced.sinkhorn_knopp : Unbalanced Classic Sinkhorn [10]
    ot.unbalanced.sinkhorn_stabilized: Unbalanced Stabilized sinkhorn [9][10]
    ot.unbalanced.sinkhorn_epsilon_scaling: Unbalanced Sinkhorn with epslilon scaling [9][10]

    """

    if method.lower() == 'sinkhorn':
        def sink():
            return sinkhorn_knopp_unbalanced(a, b, M, reg, alpha,
                                             numItermax=numItermax,
                                             stopThr=stopThr, verbose=verbose,
                                             log=log, **kwargs)

    elif method.lower() in ['sinkhorn_stabilized', 'sinkhorn_epsilon_scaling']:
        warnings.warn('Method not implemented yet. Using classic Sinkhorn Knopp')

        def sink():
            return sinkhorn_knopp_unbalanced(a, b, M, reg, alpha,
                                             numItermax=numItermax,
                                             stopThr=stopThr, verbose=verbose,
                                             log=log, **kwargs)
    else:
        raise ValueError('Unknown method. Using classic Sinkhorn Knopp')

    b = np.asarray(b, dtype=np.float64)
    if len(b.shape) < 2:
        b = b[:, None]

    return sink()


def sinkhorn_knopp_unbalanced(a, b, M, reg, alpha, numItermax=1000,
                              stopThr=1e-9, verbose=False, log=False, **kwargs):
    r"""
    Solve the entropic regularization unbalanced optimal transport problem and return the loss

    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma) + \\alpha KL(\gamma 1, a) + \\alpha KL(\gamma^T 1, b)

        s.t.
             \gamma\geq 0
    where :

    - M is the (ns, nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized Sinkhorn-Knopp matrix scaling algorithm as proposed in [10, 23]_


    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,) or np.ndarray (nt, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    M : np.ndarray (ns,nt)
        loss matrix
    reg : float
        Entropy regularization term > 0
    alpha : float
        Marginal relaxation term > 0
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
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

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

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

    .. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. : Learning with a Wasserstein Loss,  Advances in Neural Information Processing Systems (NIPS) 2015

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    n_a, n_b = M.shape

    if len(a) == 0:
        a = np.ones(n_a, dtype=np.float64) / n_a
    if len(b) == 0:
        b = np.ones(n_b, dtype=np.float64) / n_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = np.ones((n_a, 1)) / n_a
        v = np.ones((n_b, n_hists)) / n_b
        a = a.reshape(n_a, 1)
    else:
        u = np.ones(n_a) / n_a
        v = np.ones(n_b) / n_b

    # print(reg)
    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    # print(np.min(K))
    fi = alpha / (alpha + reg)

    cpt = 0
    err = 1.

    while (err > stopThr and cpt < numItermax):
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
            warnings.warn('Numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.sum((u - uprev)**2) / np.sum((u)**2) + \
                np.sum((v - vprev)**2) / np.sum((v)**2)
            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt += 1

    if log:
        log['u'] = u
        log['v'] = v

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


def barycenter_unbalanced(A, M, reg, alpha, weights=None, numItermax=1000,
                          stopThr=1e-4, verbose=False, log=False):
    r"""Compute the entropic regularized unbalanced wasserstein barycenter of distributions A

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i Wu_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`Wu_{reg}(\cdot,\cdot)` is the unbalanced entropic regularized Wasserstein distance (see ot.unbalanced.sinkhorn_unbalanced)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT
    - alpha is the marginal relaxation hyperparameter
    The algorithm used for solving the problem is the generalized Sinkhorn-Knopp matrix scaling algorithm as proposed in [10]_

    Parameters
    ----------
    A : np.ndarray (d,n)
        n training distributions a_i of size d
    M : np.ndarray (d,d)
        loss matrix   for OT
    reg : float
        Entropy regularization term > 0
    alpha : float
        Marginal relaxation term > 0
    weights : np.ndarray (n,)
        Weights of each histogram a_i on the simplex (barycentric coodinates)
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
    a : (d,) ndarray
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015). Iterative Bregman projections for regularized transportation problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.


    """
    p, n_hists = A.shape
    if weights is None:
        weights = np.ones(n_hists) / n_hists
    else:
        assert(len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    K = np.exp(- M / reg)

    fi = alpha / (alpha + reg)

    v = np.ones((p, n_hists)) / p
    u = np.ones((p, 1)) / p

    cpt = 0
    err = 1.

    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v

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
            warnings.warn('Numerical errors at iteration %s' % cpt)
            u = uprev
            v = vprev
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.sum((u - uprev) ** 2) / np.sum((u) ** 2) + \
                np.sum((v - vprev) ** 2) / np.sum((v) ** 2)
            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

    cpt += 1
    if log:
        log['niter'] = cpt
        log['u'] = u
        log['v'] = v
        return q, log
    else:
        return q
