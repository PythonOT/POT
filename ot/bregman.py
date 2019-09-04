# -*- coding: utf-8 -*-
"""
Bregman projections for regularized OT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#         Kilian Fatras <kilian.fatras@irisa.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Hicham Janati <hicham.janati@inria.fr>
#
# License: MIT License

import numpy as np
import warnings
from .utils import unif, dist


def sinkhorn(a, b, M, reg, method='sinkhorn', numItermax=1000,
             stopThr=1e-9, verbose=False, log=False, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [2]_


    Parameters
    ----------
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,) or ndarray, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    M : ndarray, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    method : str
        method used for the solver either 'sinkhorn', 'greenkhorn', 'sinkhorn_stabilized' or
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
    gamma : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.



    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT
    ot.bregman.sinkhorn_knopp : Classic Sinkhorn [2]
    ot.bregman.sinkhorn_stabilized: Stabilized sinkhorn [9][10]
    ot.bregman.sinkhorn_epsilon_scaling: Sinkhorn with epslilon scaling [9][10]

    """

    if method.lower() == 'sinkhorn':
        return sinkhorn_knopp(a, b, M, reg, numItermax=numItermax,
                              stopThr=stopThr, verbose=verbose, log=log,
                              **kwargs)
    elif method.lower() == 'greenkhorn':
        return greenkhorn(a, b, M, reg, numItermax=numItermax,
                          stopThr=stopThr, verbose=verbose, log=log)
    elif method.lower() == 'sinkhorn_stabilized':
        return sinkhorn_stabilized(a, b, M, reg, numItermax=numItermax,
                                   stopThr=stopThr, verbose=verbose,
                                   log=log, **kwargs)
    elif method.lower() == 'sinkhorn_epsilon_scaling':
        return sinkhorn_epsilon_scaling(a, b, M, reg,
                                        numItermax=numItermax,
                                        stopThr=stopThr, verbose=verbose,
                                        log=log, **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn2(a, b, M, reg, method='sinkhorn', numItermax=1000,
              stopThr=1e-9, verbose=False, log=False, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem and return the loss

    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [2]_


    Parameters
    ----------
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,) or ndarray, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    M : ndarray, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
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
    W : (n_hists) ndarray or float
        Optimal transportation loss for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn2(a, b, M, 1)
    array([0.26894142])



    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

       [21] Altschuler J., Weed J., Rigollet P. : Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration, Advances in Neural Information Processing Systems (NIPS) 31, 2017



    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT
    ot.bregman.sinkhorn_knopp : Classic Sinkhorn [2]
    ot.bregman.greenkhorn : Greenkhorn [21]
    ot.bregman.sinkhorn_stabilized: Stabilized sinkhorn [9][10]
    ot.bregman.sinkhorn_epsilon_scaling: Sinkhorn with epslilon scaling [9][10]

    """
    b = np.asarray(b, dtype=np.float64)
    if len(b.shape) < 2:
        b = b[:, None]
    if method.lower() == 'sinkhorn':
        return sinkhorn_knopp(a, b, M, reg, numItermax=numItermax,
                              stopThr=stopThr, verbose=verbose, log=log,
                              **kwargs)
    elif method.lower() == 'sinkhorn_stabilized':
        return sinkhorn_stabilized(a, b, M, reg, numItermax=numItermax,
                                   stopThr=stopThr, verbose=verbose, log=log,
                                   **kwargs)
    elif method.lower() == 'sinkhorn_epsilon_scaling':
        return sinkhorn_epsilon_scaling(a, b, M, reg, numItermax=numItermax,
                                        stopThr=stopThr, verbose=verbose,
                                        log=log, **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn_knopp(a, b, M, reg, numItermax=1000,
                   stopThr=1e-9, verbose=False, log=False, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [2]_


    Parameters
    ----------
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,) or ndarray, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    M : ndarray, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
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
    gamma : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    # init data
    dim_a = len(a)
    dim_b = len(b)

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
    else:
        u = np.ones(dim_a) / dim_a
        v = np.ones(dim_b) / dim_b

    # print(reg)

    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    # print(np.min(K))
    tmp2 = np.empty(b.shape, dtype=M.dtype)

    Kp = (1 / a).reshape(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v

        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1. / np.dot(Kp, v)

        if (np.any(KtransposeU == 0)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                np.einsum('ik,ij,jk->jk', u, K, v, out=tmp2)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                np.einsum('i,ij,j->j', u, K, v, out=tmp2)
            err = np.linalg.norm(tmp2 - b)  # violation of marginal
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
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
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))


def greenkhorn(a, b, M, reg, numItermax=10000, stopThr=1e-9, verbose=False,
               log=False):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The algorithm used is based on the paper

    Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration
        by Jason Altschuler, Jonathan Weed, Philippe Rigollet
        appeared at NIPS 2017

    which is a stochastic version of the Sinkhorn-Knopp algorithm [2].

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (histograms, both sum to 1)



    Parameters
    ----------
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,) or ndarray, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    M : ndarray, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.bregman.greenkhorn(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
       [22] J. Altschuler, J.Weed, P. Rigollet : Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration, Advances in Neural Information Processing Systems (NIPS) 31, 2017


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    dim_a = a.shape[0]
    dim_b = b.shape[0]

    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty_like(M)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    u = np.full(dim_a, 1. / dim_a)
    v = np.full(dim_b, 1. / dim_b)
    G = u[:, np.newaxis] * K * v[np.newaxis, :]

    viol = G.sum(1) - a
    viol_2 = G.sum(0) - b
    stopThr_val = 1

    if log:
        log = dict()
        log['u'] = u
        log['v'] = v

    for i in range(numItermax):
        i_1 = np.argmax(np.abs(viol))
        i_2 = np.argmax(np.abs(viol_2))
        m_viol_1 = np.abs(viol[i_1])
        m_viol_2 = np.abs(viol_2[i_2])
        stopThr_val = np.maximum(m_viol_1, m_viol_2)

        if m_viol_1 > m_viol_2:
            old_u = u[i_1]
            u[i_1] = a[i_1] / (K[i_1, :].dot(v))
            G[i_1, :] = u[i_1] * K[i_1, :] * v

            viol[i_1] = u[i_1] * K[i_1, :].dot(v) - a[i_1]
            viol_2 += (K[i_1, :].T * (u[i_1] - old_u) * v)

        else:
            old_v = v[i_2]
            v[i_2] = b[i_2] / (K[:, i_2].T.dot(u))
            G[:, i_2] = u * K[:, i_2] * v[i_2]
            #aviol = (G@one_m - a)
            #aviol_2 = (G.T@one_n - b)
            viol += (-old_v + v[i_2]) * K[:, i_2] * u
            viol_2[i_2] = v[i_2] * K[:, i_2].dot(u) - b[i_2]

            #print('b',np.max(abs(aviol -viol)),np.max(abs(aviol_2 - viol_2)))

        if stopThr_val <= stopThr:
            break
    else:
        print('Warning: Algorithm did not converge')

    if log:
        log['u'] = u
        log['v'] = v

    if log:
        return G, log
    else:
        return G


def sinkhorn_stabilized(a, b, M, reg, numItermax=1000, tau=1e3, stopThr=1e-9,
                        warmstart=None, verbose=False, print_period=20,
                        log=False, **kwargs):
    r"""
    Solve the entropic regularization OT problem with log stabilization

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (histograms, both sum to 1)


    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in [2]_ but with the log stabilization
    proposed in [10]_ an defined in [9]_ (Algo 3.1) .


    Parameters
    ----------
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,)
        samples in the target domain
    M : ndarray, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    tau : float
        thershold for max value in u or v for log scaling
    warmstart : tible of vectors
        if given then sarting values for alpha an beta log scalings
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
    gamma : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ot.bregman.sinkhorn_stabilized(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    # test if multiple target
    if len(b.shape) > 1:
        n_hists = b.shape[1]
        a = a[:, np.newaxis]
    else:
        n_hists = 0

    # init data
    dim_a = len(a)
    dim_b = len(b)

    cpt = 0
    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = np.zeros(dim_a), np.zeros(dim_b)
    else:
        alpha, beta = warmstart

    if n_hists:
        u = np.ones((dim_a, n_hists)) / dim_a
        v = np.ones((dim_b, n_hists)) / dim_b
    else:
        u, v = np.ones(dim_a) / dim_a, np.ones(dim_b) / dim_b

    def get_K(alpha, beta):
        """log space computation"""
        return np.exp(-(M - alpha.reshape((dim_a, 1))
                        - beta.reshape((1, dim_b))) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return np.exp(-(M - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b)))
                      / reg + np.log(u.reshape((dim_a, 1))) + np.log(v.reshape((1, dim_b))))

    # print(np.min(K))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1
    while loop:

        uprev = u
        vprev = v

        # sinkhorn update
        v = b / (np.dot(K.T, u) + 1e-16)
        u = a / (np.dot(K, v) + 1e-16)

        # remove numerical problems and store them in K
        if np.abs(u).max() > tau or np.abs(v).max() > tau:
            if n_hists:
                alpha, beta = alpha + reg * \
                    np.max(np.log(u), 1), beta + reg * np.max(np.log(v))
            else:
                alpha, beta = alpha + reg * np.log(u), beta + reg * np.log(v)
                if n_hists:
                    u, v = np.ones((dim_a, n_hists)) / dim_a, np.ones((dim_b, n_hists)) / dim_b
                else:
                    u, v = np.ones(dim_a) / dim_a, np.ones(dim_b) / dim_b
            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                err_u = abs(u - uprev).max()
                err_u /= max(abs(u).max(), abs(uprev).max(), 1.)
                err_v = abs(v - vprev).max()
                err_v /= max(abs(v).max(), abs(vprev).max(), 1.)
                err = 0.5 * (err_u + err_v)
            else:
                transp = get_Gamma(alpha, beta, u, v)
                err = np.linalg.norm((np.sum(transp, axis=0) - b))
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % (print_period * 20) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        cpt = cpt + 1

    if log:
        if n_hists:
            alpha = alpha[:, None]
            beta = beta[:, None]
        logu = alpha / reg + np.log(u)
        logv = beta / reg + np.log(v)
        log['logu'] = logu
        log['logv'] = logv
        log['alpha'] = alpha + reg * np.log(u)
        log['beta'] = beta + reg * np.log(v)
        log['warmstart'] = (log['alpha'], log['beta'])
        if n_hists:
            res = np.zeros((n_hists))
            for i in range(n_hists):
                res[i] = np.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
            return res, log

        else:
            return get_Gamma(alpha, beta, u, v), log
    else:
        if n_hists:
            res = np.zeros((n_hists))
            for i in range(n_hists):
                res[i] = np.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
            return res
        else:
            return get_Gamma(alpha, beta, u, v)


def sinkhorn_epsilon_scaling(a, b, M, reg, numItermax=100, epsilon0=1e4,
                             numInnerItermax=100, tau=1e3, stopThr=1e-9,
                             warmstart=None, verbose=False, print_period=10,
                             log=False, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem with log
    stabilization and epsilon scaling.

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (histograms, both sum to 1)


    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in [2]_ but with the log stabilization
    proposed in [10]_ and the log scaling proposed in [9]_ algorithm 3.2


    Parameters
    ----------
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,)
        samples in the target domain
    M : ndarray, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    tau : float
        thershold for max value in u or v for log scaling
    warmstart : tuple of vectors
        if given then sarting values for alpha an beta log scalings
    numItermax : int, optional
        Max number of iterations
    numInnerItermax : int, optional
        Max number of iterationsin the inner slog stabilized sinkhorn
    epsilon0 : int, optional
        first epsilon regularization value (then exponential decrease to reg)
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.bregman.sinkhorn_epsilon_scaling(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    # init data
    dim_a = len(a)
    dim_b = len(b)

    # nrelative umerical precision with 64 bits
    numItermin = 35
    numItermax = max(numItermin, numItermax)  # ensure that last velue is exact

    cpt = 0
    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = np.zeros(dim_a), np.zeros(dim_b)
    else:
        alpha, beta = warmstart

    def get_K(alpha, beta):
        """log space computation"""
        return np.exp(-(M - alpha.reshape((dim_a, 1))
                        - beta.reshape((1, dim_b))) / reg)

    # print(np.min(K))
    def get_reg(n):  # exponential decreasing
        return (epsilon0 - reg) * np.exp(-n) + reg

    loop = 1
    cpt = 0
    err = 1
    while loop:

        regi = get_reg(cpt)

        G, logi = sinkhorn_stabilized(a, b, M, regi,
                                      numItermax=numInnerItermax, stopThr=1e-9,
                                      warmstart=(alpha, beta), verbose=False,
                                      print_period=20, tau=tau, log=True)

        alpha = logi['alpha']
        beta = logi['beta']

        if cpt >= numItermax:
            loop = False

        if cpt % (print_period) == 0:  # spsion nearly converged
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = G
            err = np.linalg.norm(
                (np.sum(transp, axis=0) - b))**2 + np.linalg.norm((np.sum(transp, axis=1) - a))**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % (print_period * 10) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        if err <= stopThr and cpt > numItermin:
            loop = False

        cpt = cpt + 1
    # print('err=',err,' cpt=',cpt)
    if log:
        log['alpha'] = alpha
        log['beta'] = beta
        log['warmstart'] = (log['alpha'], log['beta'])
        return G, log
    else:
        return G


def geometricBar(weights, alldistribT):
    """return the weighted geometric mean of distributions"""
    assert(len(weights) == alldistribT.shape[1])
    return np.exp(np.dot(np.log(alldistribT), weights.T))


def geometricMean(alldistribT):
    """return the  geometric mean of distributions"""
    return np.exp(np.mean(np.log(alldistribT), axis=1))


def projR(gamma, p):
    """return the KL projection on the row constrints """
    return np.multiply(gamma.T, p / np.maximum(np.sum(gamma, axis=1), 1e-10)).T


def projC(gamma, q):
    """return the KL projection on the column constrints """
    return np.multiply(gamma, q / np.maximum(np.sum(gamma, axis=0), 1e-10))


def barycenter(A, M, reg, weights=None, method="sinkhorn", numItermax=10000,
               stopThr=1e-4, verbose=False, log=False, **kwargs):
    r"""Compute the entropic regularized wasserstein barycenter of distributions A

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance (see ot.bregman.sinkhorn)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [3]_

    Parameters
    ----------
    A : ndarray, shape (dim, n_hists)
        n_hists training distributions a_i of size dim
    M : ndarray, shape (dim, dim)
        loss matrix for OT
    reg : float
        Regularization term > 0
    method : str (optional)
        method used for the solver either 'sinkhorn' or 'sinkhorn_stabilized'
    weights : ndarray, shape (n_hists,)
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
    a : (dim,) ndarray
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015). Iterative Bregman projections for regularized transportation problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    """

    if method.lower() == 'sinkhorn':
        return barycenter_sinkhorn(A, M, reg, numItermax=numItermax,
                                   stopThr=stopThr, verbose=verbose, log=log,
                                   **kwargs)
    elif method.lower() == 'sinkhorn_stabilized':
        return barycenter_stabilized(A, M, reg, numItermax=numItermax,
                                     stopThr=stopThr, verbose=verbose,
                                     log=log, **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def barycenter_sinkhorn(A, M, reg, weights=None, numItermax=1000,
                        stopThr=1e-4, verbose=False, log=False):
    r"""Compute the entropic regularized wasserstein barycenter of distributions A

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance (see ot.bregman.sinkhorn)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [3]_

    Parameters
    ----------
    A : ndarray, shape (dim, n_hists)
        n_hists training distributions a_i of size dim
    M : ndarray, shape (dim, dim)
        loss matrix for OT
    reg : float
        Regularization term > 0
    weights : ndarray, shape (n_hists,)
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
    a : (dim,) ndarray
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015). Iterative Bregman projections for regularized transportation problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    """

    if weights is None:
        weights = np.ones(A.shape[1]) / A.shape[1]
    else:
        assert(len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    # M = M/np.median(M) # suggested by G. Peyre
    K = np.exp(-M / reg)

    cpt = 0
    err = 1

    UKv = np.dot(K, np.divide(A.T, np.sum(K, axis=0)).T)
    u = (geometricMean(UKv) / UKv.T).T

    while (err > stopThr and cpt < numItermax):
        cpt = cpt + 1
        UKv = u * np.dot(K, np.divide(A, np.dot(K, u)))
        u = (u.T * geometricBar(weights, UKv)).T / UKv

        if cpt % 10 == 1:
            err = np.sum(np.std(UKv, axis=1))

            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

    if log:
        log['niter'] = cpt
        return geometricBar(weights, UKv), log
    else:
        return geometricBar(weights, UKv)


def barycenter_stabilized(A, M, reg, tau=1e10, weights=None, numItermax=1000,
                          stopThr=1e-4, verbose=False, log=False):
    r"""Compute the entropic regularized wasserstein barycenter of distributions A
        with stabilization.

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance (see ot.bregman.sinkhorn)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [3]_

    Parameters
    ----------
    A : ndarray, shape (dim, n_hists)
        n_hists training distributions a_i of size dim
    M : ndarray, shape (dim, dim)
        loss matrix for OT
    reg : float
        Regularization term > 0
    tau : float
        thershold for max value in u or v for log scaling
    weights : ndarray, shape (n_hists,)
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
    a : (dim,) ndarray
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015). Iterative Bregman projections for regularized transportation problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    """

    dim, n_hists = A.shape
    if weights is None:
        weights = np.ones(n_hists) / n_hists
    else:
        assert(len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    u = np.ones((dim, n_hists)) / dim
    v = np.ones((dim, n_hists)) / dim

    # print(reg)
    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    cpt = 0
    err = 1.
    alpha = np.zeros(dim)
    beta = np.zeros(dim)
    q = np.ones(dim) / dim
    while (err > stopThr and cpt < numItermax):
        qprev = q
        Kv = K.dot(v)
        u = A / (Kv + 1e-16)
        Ktu = K.T.dot(u)
        q = geometricBar(weights, Ktu)
        Q = q[:, None]
        v = Q / (Ktu + 1e-16)
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
        if (cpt % 10 == 0 and not absorbing) or cpt == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = abs(u * Kv - A).max()
            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1
    if err > stopThr:
        warnings.warn("Stabilized Unbalanced Sinkhorn did not converge." +
                      "Try a larger entropy `reg`" +
                      "Or a larger absorption threshold `tau`.")
    if log:
        log['niter'] = cpt
        log['logu'] = np.log(u + 1e-16)
        log['logv'] = np.log(v + 1e-16)
        return q, log
    else:
        return q


def convolutional_barycenter2d(A, reg, weights=None, numItermax=10000,
                               stopThr=1e-9, stabThr=1e-30, verbose=False,
                               log=False):
    r"""Compute the entropic regularized wasserstein barycenter of distributions A
    where A is a collection of 2D images.

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance (see ot.bregman.sinkhorn)
    - :math:`\mathbf{a}_i` are training distributions (2D images) in the mast two dimensions of matrix :math:`\mathbf{A}`
    - reg is the regularization strength scalar value

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [21]_

    Parameters
    ----------
    A : ndarray, shape (n_hists, width, height)
        n distributions (2D images) of size width x height
    reg : float
        Regularization term >0
    weights : ndarray, shape (n_hists,)
        Weights of each image on the simplex (barycentric coodinates)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (> 0)
    stabThr : float, optional
        Stabilization threshold to avoid numerical precision issue
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    a : ndarray, shape (width, height)
        2D Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------

    .. [21] Solomon, J., De Goes, F., Peyré, G., Cuturi, M., Butscher, A., Nguyen, A. & Guibas, L. (2015).
    Convolutional wasserstein distances: Efficient optimal transportation on geometric domains
    ACM Transactions on Graphics (TOG), 34(4), 66


    """

    if weights is None:
        weights = np.ones(A.shape[0]) / A.shape[0]
    else:
        assert(len(weights) == A.shape[0])

    if log:
        log = {'err': []}

    b = np.zeros_like(A[0, :, :])
    U = np.ones_like(A)
    KV = np.ones_like(A)

    cpt = 0
    err = 1

    # build the convolution operator
    t = np.linspace(0, 1, A.shape[1])
    [Y, X] = np.meshgrid(t, t)
    xi1 = np.exp(-(X - Y)**2 / reg)

    def K(x):
        return np.dot(np.dot(xi1, x), xi1)

    while (err > stopThr and cpt < numItermax):

        bold = b
        cpt = cpt + 1

        b = np.zeros_like(A[0, :, :])
        for r in range(A.shape[0]):
            KV[r, :, :] = K(A[r, :, :] / np.maximum(stabThr, K(U[r, :, :])))
            b += weights[r] * np.log(np.maximum(stabThr, U[r, :, :] * KV[r, :, :]))
        b = np.exp(b)
        for r in range(A.shape[0]):
            U[r, :, :] = b / np.maximum(stabThr, KV[r, :, :])

        if cpt % 10 == 1:
            err = np.sum(np.abs(bold - b))
            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

    if log:
        log['niter'] = cpt
        log['U'] = U
        return b, log
    else:
        return b


def unmix(a, D, M, M0, h0, reg, reg0, alpha, numItermax=1000,
          stopThr=1e-3, verbose=False, log=False):
    r"""
    Compute the unmixing of an observation with a given dictionary using Wasserstein distance

    The function solve the following optimization problem:

    .. math::
       \mathbf{h} = arg\min_\mathbf{h}  (1- \\alpha) W_{M,reg}(\mathbf{a},\mathbf{Dh})+\\alpha W_{M0,reg0}(\mathbf{h}_0,\mathbf{h})


    where :

    - :math:`W_{M,reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance with M loss matrix (see ot.bregman.sinkhorn)
    - :math: `\mathbf{D}` is a dictionary of `n_atoms` atoms of dimension `dim_a`, its expected shape is `(dim_a, n_atoms)`
    - :math:`\mathbf{h}` is the estimated unmixing of dimension `n_atoms`
    - :math:`\mathbf{a}` is an observed distribution of dimension `dim_a`
    - :math:`\mathbf{h}_0` is a prior on `h` of dimension `dim_prior`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix (dim_a, dim_a) for OT data fitting
    - reg0 and :math:`\mathbf{M0}` are respectively the regularization term and the cost matrix (dim_prior, n_atoms) regularization
    - :math:`\\alpha`weight data fitting and regularization

    The optimization problem is solved suing the algorithm described in [4]


    Parameters
    ----------
    a : ndarray, shape (dim_a)
        observed distribution (histogram, sums to 1)
    D : ndarray, shape (dim_a, n_atoms)
        dictionary matrix
    M : ndarray, shape (dim_a, dim_a)
        loss matrix
    M0 : ndarray, shape (n_atoms, dim_prior)
        loss matrix
    h0 : ndarray, shape (n_atoms,)
        prior on the estimated unmixing h
    reg : float
        Regularization term >0 (Wasserstein data fitting)
    reg0 : float
        Regularization term >0 (Wasserstein reg with h0)
    alpha : float
        How much should we trust the prior ([0,1])
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
    h : ndarray, shape (n_atoms,)
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------

    .. [4] S. Nakhostin, N. Courty, R. Flamary, D. Tuia, T. Corpetti, Supervised planetary unmixing with optimal transport, Whorkshop on Hyperspectral Image and Signal Processing : Evolution in Remote Sensing (WHISPERS), 2016.

    """

    # M = M/np.median(M)
    K = np.exp(-M / reg)

    # M0 = M0/np.median(M0)
    K0 = np.exp(-M0 / reg0)
    old = h0

    err = 1
    cpt = 0
    # log = {'niter':0, 'all_err':[]}
    if log:
        log = {'err': []}

    while (err > stopThr and cpt < numItermax):
        K = projC(K, a)
        K0 = projC(K0, h0)
        new = np.sum(K0, axis=1)
        # we recombine the current selection from dictionnary
        inv_new = np.dot(D, new)
        other = np.sum(K, axis=1)
        # geometric interpolation
        delta = np.exp(alpha * np.log(other) + (1 - alpha) * np.log(inv_new))
        K = projR(K, delta)
        K0 = np.dot(np.diag(np.dot(D.T, delta / inv_new)), K0)

        err = np.linalg.norm(np.sum(K0, axis=1) - old)
        old = new
        if log:
            log['err'].append(err)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err))

        cpt = cpt + 1

    if log:
        log['niter'] = cpt
        return np.sum(K0, axis=1), log
    else:
        return np.sum(K0, axis=1)


def empirical_sinkhorn(X_s, X_t, reg, a=None, b=None, metric='sqeuclidean',
                       numIterMax=10000, stopThr=1e-9, verbose=False,
                       log=False, **kwargs):
    r'''
    Solve the entropic regularization optimal transport problem and return the
    OT matrix from empirical data

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - :math:`M` is the (n_samples_a, n_samples_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`a` and :math:`b` are source and target weights (sum to 1)


    Parameters
    ----------
    X_s : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    X_t : ndarray, shape (n_samples_b, dim)
        samples in the target domain
    reg : float
        Regularization term >0
    a : ndarray, shape (n_samples_a,)
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,)
        samples weights in the target domain
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
    gamma : ndarray, shape (n_samples_a, n_samples_b)
        Regularized optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_samples_a = 2
    >>> n_samples_b = 2
    >>> reg = 0.1
    >>> X_s = np.reshape(np.arange(n_samples_a), (n_samples_a, 1))
    >>> X_t = np.reshape(np.arange(0, n_samples_b), (n_samples_b, 1))
    >>> empirical_sinkhorn(X_s, X_t, reg, verbose=False)  # doctest: +NORMALIZE_WHITESPACE
    array([[4.99977301e-01,  2.26989344e-05],
           [2.26989344e-05,  4.99977301e-01]])


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.
    '''

    if a is None:
        a = unif(np.shape(X_s)[0])
    if b is None:
        b = unif(np.shape(X_t)[0])

    M = dist(X_s, X_t, metric=metric)

    if log:
        pi, log = sinkhorn(a, b, M, reg, numItermax=numIterMax, stopThr=stopThr, verbose=verbose, log=True, **kwargs)
        return pi, log
    else:
        pi = sinkhorn(a, b, M, reg, numItermax=numIterMax, stopThr=stopThr, verbose=verbose, log=False, **kwargs)
        return pi


def empirical_sinkhorn2(X_s, X_t, reg, a=None, b=None, metric='sqeuclidean', numIterMax=10000, stopThr=1e-9, verbose=False, log=False, **kwargs):
    r'''
    Solve the entropic regularization optimal transport problem from empirical
    data and return the OT loss


    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - :math:`M` is the (n_samples_a, n_samples_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`a` and :math:`b` are source and target weights (sum to 1)


    Parameters
    ----------
    X_s : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    X_t : ndarray, shape (n_samples_b, dim)
        samples in the target domain
    reg : float
        Regularization term >0
    a : ndarray, shape (n_samples_a,)
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,)
        samples weights in the target domain
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
    gamma : ndarray, shape (n_samples_a, n_samples_b)
        Regularized optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_samples_a = 2
    >>> n_samples_b = 2
    >>> reg = 0.1
    >>> X_s = np.reshape(np.arange(n_samples_a), (n_samples_a, 1))
    >>> X_t = np.reshape(np.arange(0, n_samples_b), (n_samples_b, 1))
    >>> empirical_sinkhorn2(X_s, X_t, reg, verbose=False)
    array([4.53978687e-05])


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.
    '''

    if a is None:
        a = unif(np.shape(X_s)[0])
    if b is None:
        b = unif(np.shape(X_t)[0])

    M = dist(X_s, X_t, metric=metric)

    if log:
        sinkhorn_loss, log = sinkhorn2(a, b, M, reg, numItermax=numIterMax, stopThr=stopThr, verbose=verbose, log=log, **kwargs)
        return sinkhorn_loss, log
    else:
        sinkhorn_loss = sinkhorn2(a, b, M, reg, numItermax=numIterMax, stopThr=stopThr, verbose=verbose, log=log, **kwargs)
        return sinkhorn_loss


def empirical_sinkhorn_divergence(X_s, X_t, reg, a=None, b=None, metric='sqeuclidean', numIterMax=10000, stopThr=1e-9, verbose=False, log=False, **kwargs):
    r'''
    Compute the sinkhorn divergence loss from empirical data

    The function solves the following optimization problems and return the
    sinkhorn divergence :math:`S`:

    .. math::

        W &= \min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        W_a &= \min_{\gamma_a} <\gamma_a,M_a>_F + reg\cdot\Omega(\gamma_a)

        W_b &= \min_{\gamma_b} <\gamma_b,M_b>_F + reg\cdot\Omega(\gamma_b)

        S &= W - 1/2 * (W_a + W_b)

    .. math::
        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0

             \gamma_a 1 = a

             \gamma_a^T 1= a

             \gamma_a\geq 0

             \gamma_b 1 = b

             \gamma_b^T 1= b

             \gamma_b\geq 0
    where :

    - :math:`M` (resp. :math:`M_a, M_b`) is the (n_samples_a, n_samples_b) metric cost matrix (resp (n_samples_a, n_samples_a) and (n_samples_b, n_samples_b))
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`a` and :math:`b` are source and target weights (sum to 1)


    Parameters
    ----------
    X_s : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    X_t : ndarray, shape (n_samples_b, dim)
        samples in the target domain
    reg : float
        Regularization term >0
    a : ndarray, shape (n_samples_a,)
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,)
        samples weights in the target domain
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
    gamma : ndarray, shape (n_samples_a, n_samples_b)
        Regularized optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------
    >>> n_samples_a = 2
    >>> n_samples_b = 4
    >>> reg = 0.1
    >>> X_s = np.reshape(np.arange(n_samples_a), (n_samples_a, 1))
    >>> X_t = np.reshape(np.arange(0, n_samples_b), (n_samples_b, 1))
    >>> empirical_sinkhorn_divergence(X_s, X_t, reg)  # doctest: +ELLIPSIS
    array([1.499...])


    References
    ----------
    .. [23] Aude Genevay, Gabriel Peyré, Marco Cuturi, Learning Generative Models with Sinkhorn Divergences,  Proceedings of the Twenty-First International Conference on Artficial Intelligence and Statistics, (AISTATS) 21, 2018
    '''
    if log:
        sinkhorn_loss_ab, log_ab = empirical_sinkhorn2(X_s, X_t, reg, a, b, metric=metric, numIterMax=numIterMax, stopThr=1e-9, verbose=verbose, log=log, **kwargs)

        sinkhorn_loss_a, log_a = empirical_sinkhorn2(X_s, X_s, reg, a, b, metric=metric, numIterMax=numIterMax, stopThr=1e-9, verbose=verbose, log=log, **kwargs)

        sinkhorn_loss_b, log_b = empirical_sinkhorn2(X_t, X_t, reg, a, b, metric=metric, numIterMax=numIterMax, stopThr=1e-9, verbose=verbose, log=log, **kwargs)

        sinkhorn_div = sinkhorn_loss_ab - 1 / 2 * (sinkhorn_loss_a + sinkhorn_loss_b)

        log = {}
        log['sinkhorn_loss_ab'] = sinkhorn_loss_ab
        log['sinkhorn_loss_a'] = sinkhorn_loss_a
        log['sinkhorn_loss_b'] = sinkhorn_loss_b
        log['log_sinkhorn_ab'] = log_ab
        log['log_sinkhorn_a'] = log_a
        log['log_sinkhorn_b'] = log_b

        return max(0, sinkhorn_div), log

    else:
        sinkhorn_loss_ab = empirical_sinkhorn2(X_s, X_t, reg, a, b, metric=metric, numIterMax=numIterMax, stopThr=1e-9, verbose=verbose, log=log, **kwargs)

        sinkhorn_loss_a = empirical_sinkhorn2(X_s, X_s, reg, a, b, metric=metric, numIterMax=numIterMax, stopThr=1e-9, verbose=verbose, log=log, **kwargs)

        sinkhorn_loss_b = empirical_sinkhorn2(X_t, X_t, reg, a, b, metric=metric, numIterMax=numIterMax, stopThr=1e-9, verbose=verbose, log=log, **kwargs)

        sinkhorn_div = sinkhorn_loss_ab - 1 / 2 * (sinkhorn_loss_a + sinkhorn_loss_b)
        return max(0, sinkhorn_div)
