# -*- coding: utf-8 -*-
"""
Bregman projections solvers for entropic regularized OT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#         Kilian Fatras <kilian.fatras@irisa.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Hicham Janati <hicham.janati100@gmail.com>
#         Mokhtar Z. Alaya <mokhtarzahdi.alaya@gmail.com>
#         Alexander Tong <alexander.tong@yale.edu>
#         Ievgen Redko <ievgen.redko@univ-st-etienne.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

import warnings

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from ot.utils import unif, dist, list_to_array
from .backend import get_backend


def sinkhorn(a, b, M, reg, method='sinkhorn', numItermax=1000,
             stopThr=1e-9, verbose=False, log=False, warn=True,
             **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends.

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in :ref:`[2] <references-sinkhorn>`

    **Choosing a Sinkhorn solver**

    By default and when using a regularization parameter that is not too small
    the default sinkhorn solver should be enough. If you need to use a small
    regularization to get sharper OT matrices, you should use the
    :py:func:`ot.bregman.sinkhorn_stabilized` solver that will avoid numerical
    errors. This last solver can be very slow in practice and might not even
    converge to a reasonable OT matrix in a finite time. This is why
    :py:func:`ot.bregman.sinkhorn_epsilon_scaling` that relies on iterating the value
    of the regularization (and using warm start) sometimes leads to better
    solutions. Note that the greedy version of the sinkhorn
    :py:func:`ot.bregman.greenkhorn` can also lead to a speedup and the screening
    version of the sinkhorn :py:func:`ot.bregman.screenkhorn` aim at providing  a
    fast approximation of the Sinkhorn problem. For use of GPU and gradient
    computation with small number of iterations we strongly recommend the
    :py:func:`ot.bregman.sinkhorn_log` solver that will no need to check for
    numerical problems.


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or ndarray, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    method : str
        method used for the solver either 'sinkhorn','sinkhorn_log',
        'greenkhorn', 'sinkhorn_stabilized' or 'sinkhorn_epsilon_scaling', see
        those function for specific parameters
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
    gamma : array-like, shape (dim_a, dim_b)
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

    .. _references-sinkhorn:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information Processing
        Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms
        for Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems.
        arXiv preprint arXiv:1607.05816.

    .. [34] Feydy, J., Séjourné, T., Vialard, F. X., Amari, S. I., Trouvé,
        A., & Peyré, G. (2019, April). Interpolating between optimal transport
        and MMD using Sinkhorn divergences. In The 22nd International Conference
        on Artificial Intelligence and Statistics (pp. 2681-2690). PMLR.


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT
    ot.bregman.sinkhorn_knopp : Classic Sinkhorn :ref:`[2] <references-sinkhorn>`
    ot.bregman.sinkhorn_stabilized: Stabilized sinkhorn
        :ref:`[9] <references-sinkhorn>` :ref:`[10] <references-sinkhorn>`
    ot.bregman.sinkhorn_epsilon_scaling: Sinkhorn with epslilon scaling
        :ref:`[9] <references-sinkhorn>` :ref:`[10] <references-sinkhorn>`

    """

    if method.lower() == 'sinkhorn':
        return sinkhorn_knopp(a, b, M, reg, numItermax=numItermax,
                              stopThr=stopThr, verbose=verbose, log=log,
                              warn=warn,
                              **kwargs)
    elif method.lower() == 'sinkhorn_log':
        return sinkhorn_log(a, b, M, reg, numItermax=numItermax,
                            stopThr=stopThr, verbose=verbose, log=log,
                            warn=warn,
                            **kwargs)
    elif method.lower() == 'greenkhorn':
        return greenkhorn(a, b, M, reg, numItermax=numItermax,
                          stopThr=stopThr, verbose=verbose, log=log,
                          warn=warn)
    elif method.lower() == 'sinkhorn_stabilized':
        return sinkhorn_stabilized(a, b, M, reg, numItermax=numItermax,
                                   stopThr=stopThr, verbose=verbose,
                                   log=log, warn=warn,
                                   **kwargs)
    elif method.lower() == 'sinkhorn_epsilon_scaling':
        return sinkhorn_epsilon_scaling(a, b, M, reg,
                                        numItermax=numItermax,
                                        stopThr=stopThr, verbose=verbose,
                                        log=log, warn=warn,
                                        **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn2(a, b, M, reg, method='sinkhorn', numItermax=1000,
              stopThr=1e-9, verbose=False, log=False, warn=False, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem and return the loss

    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends.

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in :ref:`[2] <references-sinkhorn2>`


    **Choosing a Sinkhorn solver**

    By default and when using a regularization parameter that is not too small
    the default sinkhorn solver should be enough. If you need to use a small
    regularization to get sharper OT matrices, you should use the
    :py:func:`ot.bregman.sinkhorn_log` solver that will avoid numerical
    errors. This last solver can be very slow in practice and might not even
    converge to a reasonable OT matrix in a finite time. This is why
    :py:func:`ot.bregman.sinkhorn_epsilon_scaling` that relies on iterating the value
    of the regularization (and using warm start) sometimes leads to better
    solutions. Note that the greedy version of the sinkhorn
    :py:func:`ot.bregman.greenkhorn` can also lead to a speedup and the screening
    version of the sinkhorn :py:func:`ot.bregman.screenkhorn` aim a providing  a
    fast approximation of the Sinkhorn problem. For use of GPU and gradient
    computation with small number of iterations we strongly recommend the
    :py:func:`ot.bregman.sinkhorn_log` solver that will no need to check for
    numerical problems.

    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or ndarray, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    method : str
        method used for the solver either 'sinkhorn','sinkhorn_log',
        'sinkhorn_stabilized', see those function for specific parameters
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
    W : (n_hists) float/array-like
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
    0.26894142136999516


    .. _references-sinkhorn2:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of
        Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms
        for Entropy Regularized Transport Problems.
        arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems.
        arXiv preprint arXiv:1607.05816.

    .. [21] Altschuler J., Weed J., Rigollet P. : Near-linear time approximation
        algorithms for optimal transport via Sinkhorn iteration,
        Advances in Neural Information Processing Systems (NIPS) 31, 2017

    .. [34] Feydy, J., Séjourné, T., Vialard, F. X., Amari, S. I.,
        Trouvé, A., & Peyré, G. (2019, April).
        Interpolating between optimal transport and MMD using Sinkhorn
        divergences. In The 22nd International Conference on Artificial
        Intelligence and Statistics (pp. 2681-2690). PMLR.


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT
    ot.bregman.sinkhorn_knopp : Classic Sinkhorn :ref:`[2] <references-sinkhorn2>`
    ot.bregman.greenkhorn : Greenkhorn :ref:`[21] <references-sinkhorn2>`
    ot.bregman.sinkhorn_stabilized: Stabilized sinkhorn
        :ref:`[9] <references-sinkhorn2>` :ref:`[10] <references-sinkhorn2>`
    """

    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    if len(b.shape) < 2:
        if method.lower() == 'sinkhorn':
            res = sinkhorn_knopp(a, b, M, reg, numItermax=numItermax,
                                 stopThr=stopThr, verbose=verbose, log=log,
                                 **kwargs)
        elif method.lower() == 'sinkhorn_log':
            res = sinkhorn_log(a, b, M, reg, numItermax=numItermax,
                               stopThr=stopThr, verbose=verbose, log=log,
                               **kwargs)
        elif method.lower() == 'sinkhorn_stabilized':
            res = sinkhorn_stabilized(a, b, M, reg, numItermax=numItermax,
                                      stopThr=stopThr, verbose=verbose, log=log,
                                      **kwargs)
        else:
            raise ValueError("Unknown method '%s'." % method)
        if log:
            return nx.sum(M * res[0]), res[1]
        else:
            return nx.sum(M * res)

    else:

        if method.lower() == 'sinkhorn':
            return sinkhorn_knopp(a, b, M, reg, numItermax=numItermax,
                                  stopThr=stopThr, verbose=verbose, log=log,
                                  **kwargs)
        elif method.lower() == 'sinkhorn_log':
            return sinkhorn_log(a, b, M, reg, numItermax=numItermax,
                                stopThr=stopThr, verbose=verbose, log=log,
                                **kwargs)
        elif method.lower() == 'sinkhorn_stabilized':
            return sinkhorn_stabilized(a, b, M, reg, numItermax=numItermax,
                                       stopThr=stopThr, verbose=verbose, log=log,
                                       **kwargs)
        else:
            raise ValueError("Unknown method '%s'." % method)


def sinkhorn_knopp(a, b, M, reg, numItermax=1000, stopThr=1e-9,
                   verbose=False, log=False, warn=True,
                   **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp
    matrix scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-knopp>`


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or array-like, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
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
    gamma : array-like, shape (dim_a, dim_b)
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


    .. _references-sinkhorn-knopp:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(b) == 0:
        b = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

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
    else:
        u = nx.ones(dim_a, type_as=M) / dim_a
        v = nx.ones(dim_b, type_as=M) / dim_b

    K = nx.exp(M / (-reg))

    Kp = (1 / a).reshape(-1, 1) * K

    err = 1
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = nx.dot(K.T, u)
        v = b / KtransposeU
        u = 1. / nx.dot(Kp, v)

        if (nx.any(KtransposeU == 0)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Warning: numerical errors at iteration %d' % ii)
            u = uprev
            v = vprev
            break
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                tmp2 = nx.einsum('ik,ij,jk->jk', u, K, v)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = nx.einsum('i,ij,j->j', u, K, v)
            err = nx.norm(tmp2 - b)  # violation of marginal
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
        log['u'] = u
        log['v'] = v

    if n_hists:  # return only loss
        res = nx.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))


def sinkhorn_log(a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False,
                 log=False, warn=True, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem in log space
    and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm  :ref:`[2] <references-sinkhorn-log>` with the
    implementation from :ref:`[34] <references-sinkhorn-log>`


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or array-like, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
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
    gamma : array-like, shape (dim_a, dim_b)
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


    .. _references-sinkhorn-log:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of
        Optimal Transport, Advances in Neural Information Processing
        Systems (NIPS) 26, 2013

    .. [34] Feydy, J., Séjourné, T., Vialard, F. X., Amari, S. I.,
        Trouvé, A., & Peyré, G. (2019, April). Interpolating between
        optimal transport and MMD using Sinkhorn divergences. In The
        22nd International Conference on Artificial Intelligence and
        Statistics (pp. 2681-2690). PMLR.


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(b) == 0:
        b = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if n_hists:  # we do not want to use tensors sor we do a loop

        lst_loss = []
        lst_u = []
        lst_v = []

        for k in range(n_hists):
            res = sinkhorn_log(a, b[:, k], M, reg, numItermax=numItermax,
                               stopThr=stopThr, verbose=verbose, log=log, **kwargs)

            if log:
                lst_loss.append(nx.sum(M * res[0]))
                lst_u.append(res[1]['log_u'])
                lst_v.append(res[1]['log_v'])
            else:
                lst_loss.append(nx.sum(M * res))
        res = nx.stack(lst_loss)
        if log:
            log = {'log_u': nx.stack(lst_u, 1),
                   'log_v': nx.stack(lst_v, 1), }
            log['u'] = nx.exp(log['log_u'])
            log['v'] = nx.exp(log['log_v'])
            return res, log
        else:
            return res

    else:

        if log:
            log = {'err': []}

        Mr = - M / reg

        # we assume that no distances are null except those of the diagonal of
        # distances

        u = nx.zeros(dim_a, type_as=M)
        v = nx.zeros(dim_b, type_as=M)

        def get_logT(u, v):
            if n_hists:
                return Mr[:, :, None] + u + v
            else:
                return Mr + u[:, None] + v[None, :]

        loga = nx.log(a)
        logb = nx.log(b)

        err = 1
        for ii in range(numItermax):

            v = logb - nx.logsumexp(Mr + u[:, None], 0)
            u = loga - nx.logsumexp(Mr + v[None, :], 1)

            if ii % 10 == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations

                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = nx.sum(nx.exp(get_logT(u, v)), 0)
                err = nx.norm(tmp2 - b)  # violation of marginal
                if log:
                    log['err'].append(err)

                if verbose:
                    if ii % 200 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(ii, err))
                if err < stopThr:
                    break
        else:
            if warn:
                warnings.warn("Sinkhorn did not converge. You might want to "
                              "increase the number of iterations `numItermax` "
                              "or the regularization parameter `reg`.")

        if log:
            log['niter'] = ii
            log['log_u'] = u
            log['log_v'] = v
            log['u'] = nx.exp(u)
            log['v'] = nx.exp(v)

            return nx.exp(get_logT(u, v)), log

        else:
            return nx.exp(get_logT(u, v))


def greenkhorn(a, b, M, reg, numItermax=10000, stopThr=1e-9, verbose=False,
               log=False, warn=True):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The algorithm used is based on the paper :ref:`[22] <references-greenkhorn>`
    which is a stochastic version of the Sinkhorn-Knopp
    algorithm :ref:`[2] <references-greenkhorn>`

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or array-like, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.

    Returns
    -------
    gamma : array-like, shape (dim_a, dim_b)
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


    .. _references-greenkhorn:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013

    .. [22] J. Altschuler, J.Weed, P. Rigollet : Near-linear time
        approximation algorithms for optimal transport via Sinkhorn
        iteration, Advances in Neural Information Processing
        Systems (NIPS) 31, 2017


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)
    if nx.__name__ in ("jax", "tf"):
        raise TypeError("JAX or TF arrays have been received. Greenkhorn is not "
                        "compatible with  neither JAX nor TF")

    if len(a) == 0:
        a = nx.ones((M.shape[0],), type_as=M) / M.shape[0]
    if len(b) == 0:
        b = nx.ones((M.shape[1],), type_as=M) / M.shape[1]

    dim_a = a.shape[0]
    dim_b = b.shape[0]

    K = nx.exp(-M / reg)

    u = nx.full((dim_a,), 1. / dim_a, type_as=K)
    v = nx.full((dim_b,), 1. / dim_b, type_as=K)
    G = u[:, None] * K * v[None, :]

    viol = nx.sum(G, axis=1) - a
    viol_2 = nx.sum(G, axis=0) - b
    stopThr_val = 1
    if log:
        log = dict()
        log['u'] = u
        log['v'] = v

    for ii in range(numItermax):
        i_1 = nx.argmax(nx.abs(viol))
        i_2 = nx.argmax(nx.abs(viol_2))
        m_viol_1 = nx.abs(viol[i_1])
        m_viol_2 = nx.abs(viol_2[i_2])
        stopThr_val = nx.maximum(m_viol_1, m_viol_2)

        if m_viol_1 > m_viol_2:
            old_u = u[i_1]
            new_u = a[i_1] / nx.dot(K[i_1, :], v)
            G[i_1, :] = new_u * K[i_1, :] * v

            viol[i_1] = nx.dot(new_u * K[i_1, :], v) - a[i_1]
            viol_2 += (K[i_1, :].T * (new_u - old_u) * v)
            u[i_1] = new_u
        else:
            old_v = v[i_2]
            new_v = b[i_2] / nx.dot(K[:, i_2].T, u)
            G[:, i_2] = u * K[:, i_2] * new_v
            # aviol = (G@one_m - a)
            # aviol_2 = (G.T@one_n - b)
            viol += (-old_v + new_v) * K[:, i_2] * u
            viol_2[i_2] = new_v * nx.dot(K[:, i_2], u) - b[i_2]
            v[i_2] = new_v

        if stopThr_val <= stopThr:
            break
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")

    if log:
        log["n_iter"] = ii
        log['u'] = u
        log['v'] = v

    if log:
        return G, log
    else:
        return G


def sinkhorn_stabilized(a, b, M, reg, numItermax=1000, tau=1e3, stopThr=1e-9,
                        warmstart=None, verbose=False, print_period=20,
                        log=False, warn=True, **kwargs):
    r"""
    Solve the entropic regularization OT problem with log stabilization

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)


    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-stabilized>`
    but with the log stabilization
    proposed in :ref:`[10] <references-sinkhorn-stabilized>` an defined in
    :ref:`[9] <references-sinkhorn-stabilized>` (Algo 3.1) .


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,)
        samples in the target domain
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    tau : float
        threshold for max value in :math:`\mathbf{u}` or :math:`\mathbf{v}`
        for log scaling
    warmstart : table of vectors
        if given then starting values for alpha and beta log scalings
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
    gamma : array-like, shape (dim_a, dim_b)
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


    .. _references-sinkhorn-stabilized:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of
        Optimal Transport, Advances in Neural Information Processing
        Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms
        for Entropy Regularized Transport Problems.
        arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems.
        arXiv preprint arXiv:1607.05816.


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    if len(a) == 0:
        a = nx.ones((M.shape[0],), type_as=M) / M.shape[0]
    if len(b) == 0:
        b = nx.ones((M.shape[1],), type_as=M) / M.shape[1]

    # test if multiple target
    if len(b.shape) > 1:
        n_hists = b.shape[1]
        a = a[:, None]
    else:
        n_hists = 0

    # init data
    dim_a = len(a)
    dim_b = len(b)

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = nx.zeros(dim_a, type_as=M), nx.zeros(dim_b, type_as=M)
    else:
        alpha, beta = warmstart

    if n_hists:
        u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
        v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
    else:
        u, v = nx.ones(dim_a, type_as=M), nx.ones(dim_b, type_as=M)
        u /= dim_a
        v /= dim_b

    def get_K(alpha, beta):
        """log space computation"""
        return nx.exp(-(M - alpha.reshape((dim_a, 1))
                        - beta.reshape((1, dim_b))) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return nx.exp(-(M - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b)))
                      / reg + nx.log(u.reshape((dim_a, 1))) + nx.log(v.reshape((1, dim_b))))

    K = get_K(alpha, beta)
    transp = K
    err = 1
    for ii in range(numItermax):

        uprev = u
        vprev = v

        # sinkhorn update
        v = b / (nx.dot(K.T, u))
        u = a / (nx.dot(K, v))

        # remove numerical problems and store them in K
        if nx.max(nx.abs(u)) > tau or nx.max(nx.abs(v)) > tau:
            if n_hists:
                alpha, beta = alpha + reg * nx.max(nx.log(u), 1), beta + reg * nx.max(nx.log(v))
            else:
                alpha, beta = alpha + reg * nx.log(u), beta + reg * nx.log(v)
                if n_hists:
                    u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
                    v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
                else:
                    u = nx.ones(dim_a, type_as=M) / dim_a
                    v = nx.ones(dim_b, type_as=M) / dim_b
            K = get_K(alpha, beta)

        if ii % print_period == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                err_u = nx.max(nx.abs(u - uprev))
                err_u /= max(nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.0)
                err_v = nx.max(nx.abs(v - vprev))
                err_v /= max(nx.max(nx.abs(v)), nx.max(nx.abs(vprev)), 1.0)
                err = 0.5 * (err_u + err_v)
            else:
                transp = get_Gamma(alpha, beta, u, v)
                err = nx.norm(nx.sum(transp, axis=0) - b)
            if log:
                log['err'].append(err)

            if verbose:
                if ii % (print_period * 20) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))

        if err <= stopThr:
            break

        if nx.any(nx.isnan(u)) or nx.any(nx.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %d' % ii)
            u = uprev
            v = vprev
            break
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        if n_hists:
            alpha = alpha[:, None]
            beta = beta[:, None]
        logu = alpha / reg + nx.log(u)
        logv = beta / reg + nx.log(v)
        log["n_iter"] = ii
        log['logu'] = logu
        log['logv'] = logv
        log['alpha'] = alpha + reg * nx.log(u)
        log['beta'] = beta + reg * nx.log(v)
        log['warmstart'] = (log['alpha'], log['beta'])
        if n_hists:
            res = nx.stack([
                nx.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
                for i in range(n_hists)
            ])
            return res, log

        else:
            return get_Gamma(alpha, beta, u, v), log
    else:
        if n_hists:
            res = nx.stack([
                nx.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
                for i in range(n_hists)
            ])
            return res
        else:
            return get_Gamma(alpha, beta, u, v)


def sinkhorn_epsilon_scaling(a, b, M, reg, numItermax=100, epsilon0=1e4,
                             numInnerItermax=100, tau=1e3, stopThr=1e-9,
                             warmstart=None, verbose=False, print_period=10,
                             log=False, warn=True, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem with log
    stabilization and epsilon scaling.
    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-epsilon-scaling>`
    but with the log stabilization
    proposed in :ref:`[10] <references-sinkhorn-epsilon-scaling>` and the log scaling
    proposed in :ref:`[9] <references-sinkhorn-epsilon-scaling>` algorithm 3.2

    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,)
        samples in the target domain
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    tau : float
        threshold for max value in :math:`\mathbf{u}` or :math:`\mathbf{b}`
        for log scaling
    warmstart : tuple of vectors
        if given then starting values for alpha and beta log scalings
    numItermax : int, optional
        Max number of iterations
    numInnerItermax : int, optional
        Max number of iterations in the inner slog stabilized sinkhorn
    epsilon0 : int, optional
        first epsilon regularization value (then exponential decrease to reg)
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
    gamma : array-like, shape (dim_a, dim_b)
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

    .. _references-sinkhorn-epsilon-scaling:
    References
    ----------
    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal
        Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for
        Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT
    """

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    if len(a) == 0:
        a = nx.ones((M.shape[0],), type_as=M) / M.shape[0]
    if len(b) == 0:
        b = nx.ones((M.shape[1],), type_as=M) / M.shape[1]

    # init data
    dim_a = len(a)
    dim_b = len(b)

    # nrelative umerical precision with 64 bits
    numItermin = 35
    numItermax = max(numItermin, numItermax)  # ensure that last velue is exact

    ii = 0
    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = nx.zeros(dim_a, type_as=M), nx.zeros(dim_b, type_as=M)
    else:
        alpha, beta = warmstart

    # print(np.min(K))
    def get_reg(n):  # exponential decreasing
        return (epsilon0 - reg) * np.exp(-n) + reg

    err = 1
    for ii in range(numItermax):

        regi = get_reg(ii)

        G, logi = sinkhorn_stabilized(a, b, M, regi,
                                      numItermax=numInnerItermax, stopThr=1e-9,
                                      warmstart=(alpha, beta), verbose=False,
                                      print_period=20, tau=tau, log=True)

        alpha = logi['alpha']
        beta = logi['beta']

        if ii % (print_period) == 0:  # spsion nearly converged
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = G
            err = nx.norm(nx.sum(transp, axis=0) - b) ** 2 + nx.norm(nx.sum(transp, axis=1) - a) ** 2
            if log:
                log['err'].append(err)

            if verbose:
                if ii % (print_period * 10) == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))

        if err <= stopThr and ii > numItermin:
            break
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        log['alpha'] = alpha
        log['beta'] = beta
        log['warmstart'] = (log['alpha'], log['beta'])
        log['niter'] = ii
        return G, log
    else:
        return G


def geometricBar(weights, alldistribT):
    """return the weighted geometric mean of distributions"""
    weights, alldistribT = list_to_array(weights, alldistribT)
    nx = get_backend(weights, alldistribT)
    assert (len(weights) == alldistribT.shape[1])
    return nx.exp(nx.dot(nx.log(alldistribT), weights.T))


def geometricMean(alldistribT):
    """return the  geometric mean of distributions"""
    alldistribT = list_to_array(alldistribT)
    nx = get_backend(alldistribT)
    return nx.exp(nx.mean(nx.log(alldistribT), axis=1))


def projR(gamma, p):
    """return the KL projection on the row constrints """
    gamma, p = list_to_array(gamma, p)
    nx = get_backend(gamma, p)
    return (gamma.T * p / nx.maximum(nx.sum(gamma, axis=1), 1e-10)).T


def projC(gamma, q):
    """return the KL projection on the column constrints """
    gamma, q = list_to_array(gamma, q)
    nx = get_backend(gamma, q)
    return gamma * q / nx.maximum(nx.sum(gamma, axis=0), 1e-10)


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

        UKv = u * nx.dot(K, A / nx.dot(K, u))
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
        log['logu'] = np.log(u + 1e-16)
        log['logv'] = np.log(v + 1e-16)
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


def convolutional_barycenter2d(A, reg, weights=None, method="sinkhorn", numItermax=10000,
                               stopThr=1e-4, verbose=False, log=False,
                               warn=True, **kwargs):
    r"""Compute the entropic regularized wasserstein barycenter of distributions :math:`\mathbf{A}`
    where :math:`\mathbf{A}` is a collection of 2D images.

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein
      distance (see :py:func:`ot.bregman.sinkhorn`)
    - :math:`\mathbf{a}_i` are training distributions (2D images) in the mast two dimensions
      of matrix :math:`\mathbf{A}`
    - `reg` is the regularization strength scalar value

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm
    as proposed in :ref:`[21] <references-convolutional-barycenter-2d>`

    Parameters
    ----------
    A : array-like, shape (n_hists, width, height)
        `n` distributions (2D images) of size `width` x `height`
    reg : float
        Regularization term >0
    weights : array-like, shape (n_hists,)
        Weights of each image on the simplex (barycentric coodinates)
    method : string, optional
        method used for the solver either 'sinkhorn' or 'sinkhorn_log'
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    stabThr : float, optional
        Stabilization threshold to avoid numerical precision issue
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.

    Returns
    -------
    a : array-like, shape (width, height)
        2D Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-convolutional-barycenter-2d:
    References
    ----------

    .. [21] Solomon, J., De Goes, F., Peyré, G., Cuturi, M., Butscher,
        A., Nguyen, A. & Guibas, L. (2015).     Convolutional wasserstein distances:
        Efficient optimal transportation on geometric domains. ACM Transactions
        on Graphics (TOG), 34(4), 66

    .. [37] Janati, H., Cuturi, M., Gramfort, A. Proceedings of the 37th
        International Conference on Machine Learning, PMLR 119:4692-4701, 2020
    """

    if method.lower() == 'sinkhorn':
        return _convolutional_barycenter2d(A, reg, weights=weights,
                                           numItermax=numItermax,
                                           stopThr=stopThr, verbose=verbose,
                                           log=log, warn=warn,
                                           **kwargs)
    elif method.lower() == 'sinkhorn_log':
        return _convolutional_barycenter2d_log(A, reg, weights=weights,
                                               numItermax=numItermax,
                                               stopThr=stopThr, verbose=verbose,
                                               log=log, warn=warn,
                                               **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def _convolutional_barycenter2d(A, reg, weights=None, numItermax=10000,
                                stopThr=1e-9, stabThr=1e-30, verbose=False,
                                log=False, warn=True):
    r"""Compute the entropic regularized wasserstein barycenter of distributions A
    where A is a collection of 2D images.
    """

    A = list_to_array(A)

    nx = get_backend(A)

    if weights is None:
        weights = nx.ones((A.shape[0],), type_as=A) / A.shape[0]
    else:
        assert (len(weights) == A.shape[0])

    if log:
        log = {'err': []}

    bar = nx.ones(A.shape[1:], type_as=A)
    bar /= nx.sum(bar)
    U = nx.ones(A.shape, type_as=A)
    V = nx.ones(A.shape, type_as=A)
    err = 1

    # build the convolution operator
    # this is equivalent to blurring on horizontal then vertical directions
    t = nx.linspace(0, 1, A.shape[1])
    [Y, X] = nx.meshgrid(t, t)
    K1 = nx.exp(-(X - Y) ** 2 / reg)

    t = nx.linspace(0, 1, A.shape[2])
    [Y, X] = nx.meshgrid(t, t)
    K2 = nx.exp(-(X - Y) ** 2 / reg)

    def convol_imgs(imgs):
        kx = nx.einsum("...ij,kjl->kil", K1, imgs)
        kxy = nx.einsum("...ij,klj->kli", K2, kx)
        return kxy

    KU = convol_imgs(U)
    for ii in range(numItermax):
        V = bar[None] / KU
        KV = convol_imgs(V)
        U = A / KV
        KU = convol_imgs(U)
        bar = nx.exp(
            nx.sum(weights[:, None, None] * nx.log(KU + stabThr), axis=0)
        )
        if ii % 10 == 9:
            err = nx.sum(nx.std(V * KU, axis=0))
            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if ii % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
            if err < stopThr:
                break

    else:
        if warn:
            warnings.warn("Convolutional Sinkhorn did not converge. "
                          "Try a larger number of iterations `numItermax` "
                          "or a larger entropy `reg`.")
    if log:
        log['niter'] = ii
        log['U'] = U
        return bar, log
    else:
        return bar


def _convolutional_barycenter2d_log(A, reg, weights=None, numItermax=10000,
                                    stopThr=1e-4, stabThr=1e-30, verbose=False,
                                    log=False, warn=True):
    r"""Compute the entropic regularized wasserstein barycenter of distributions A
    where A is a collection of 2D images in log-domain.
    """

    A = list_to_array(A)

    nx = get_backend(A)
    if nx.__name__ in ("jax", "tf"):
        raise NotImplementedError(
            "Log-domain functions are not yet implemented"
            " for Jax and TF. Use numpy or torch arrays instead."
        )

    n_hists, width, height = A.shape

    if weights is None:
        weights = nx.ones((n_hists,), type_as=A) / n_hists
    else:
        assert (len(weights) == n_hists)

    if log:
        log = {'err': []}

    err = 1
    # build the convolution operator
    # this is equivalent to blurring on horizontal then vertical directions
    t = nx.linspace(0, 1, width)
    [Y, X] = nx.meshgrid(t, t)
    M1 = - (X - Y) ** 2 / reg

    t = nx.linspace(0, 1, height)
    [Y, X] = nx.meshgrid(t, t)
    M2 = - (X - Y) ** 2 / reg

    def convol_img(log_img):
        log_img = nx.logsumexp(M1[:, :, None] + log_img[None], axis=1)
        log_img = nx.logsumexp(M2[:, :, None] + log_img.T[None], axis=1).T
        return log_img

    logA = nx.log(A + stabThr)
    log_KU, G, F = nx.zeros((3, *logA.shape), type_as=A)
    err = 1
    for ii in range(numItermax):
        log_bar = nx.zeros((width, height), type_as=A)
        for k in range(n_hists):
            f = logA[k] - convol_img(G[k])
            log_KU[k] = convol_img(f)
            log_bar = log_bar + weights[k] * log_KU[k]

        if ii % 10 == 9:
            err = nx.exp(G + log_KU).std(axis=0).sum()
            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if ii % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
            if err < stopThr:
                break
        G = log_bar[None, :, :] - log_KU

    else:
        if warn:
            warnings.warn("Convolutional Sinkhorn did not converge. "
                          "Try a larger number of iterations `numItermax` "
                          "or a larger entropy `reg`.")
    if log:
        log['niter'] = ii
        return nx.exp(log_bar), log
    else:
        return nx.exp(log_bar)


def convolutional_barycenter2d_debiased(A, reg, weights=None, method="sinkhorn",
                                        numItermax=10000, stopThr=1e-3,
                                        verbose=False, log=False, warn=True,
                                        **kwargs):
    r"""Compute the debiased sinkhorn barycenter of distributions :math:`\mathbf{A}`
    where :math:`\mathbf{A}` is a collection of 2D images.

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i S_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`S_{reg}(\cdot,\cdot)` is the debiased entropic regularized Wasserstein
      distance (see :py:func:`ot.bregman.barycenter_debiased`)
    - :math:`\mathbf{a}_i` are training distributions (2D images) in the mast two
      dimensions of matrix :math:`\mathbf{A}`
    - `reg` is the regularization strength scalar value

    The algorithm used for solving the problem is the debiased Sinkhorn scaling
    algorithm as proposed in :ref:`[37] <references-convolutional-barycenter2d-debiased>`

    Parameters
    ----------
    A : array-like, shape (n_hists, width, height)
        `n` distributions (2D images) of size `width` x `height`
    reg : float
        Regularization term >0
    weights : array-like, shape (n_hists,)
        Weights of each image on the simplex (barycentric coodinates)
    method : string, optional
        method used for the solver either 'sinkhorn' or 'sinkhorn_log'
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    stabThr : float, optional
        Stabilization threshold to avoid numerical precision issue
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.


    Returns
    -------
    a : array-like, shape (width, height)
        2D Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-convolutional-barycenter2d-debiased:
    References
    ----------

    .. [37] Janati, H., Cuturi, M., Gramfort, A. Proceedings of the 37th International
        Conference on Machine Learning, PMLR 119:4692-4701, 2020
    """

    if method.lower() == 'sinkhorn':
        return _convolutional_barycenter2d_debiased(A, reg, weights=weights,
                                                    numItermax=numItermax,
                                                    stopThr=stopThr, verbose=verbose,
                                                    log=log, warn=warn,
                                                    **kwargs)
    elif method.lower() == 'sinkhorn_log':
        return _convolutional_barycenter2d_debiased_log(A, reg, weights=weights,
                                                        numItermax=numItermax,
                                                        stopThr=stopThr, verbose=verbose,
                                                        log=log, warn=warn,
                                                        **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def _convolutional_barycenter2d_debiased(A, reg, weights=None, numItermax=10000,
                                         stopThr=1e-3, stabThr=1e-15, verbose=False,
                                         log=False, warn=True):
    r"""Compute the debiased barycenter of 2D images via sinkhorn convolutions.
    """

    A = list_to_array(A)
    n_hists, width, height = A.shape

    nx = get_backend(A)

    if weights is None:
        weights = nx.ones((n_hists,), type_as=A) / n_hists
    else:
        assert (len(weights) == n_hists)

    if log:
        log = {'err': []}

    bar = nx.ones((width, height), type_as=A)
    bar /= width * height
    U = nx.ones(A.shape, type_as=A)
    V = nx.ones(A.shape, type_as=A)
    c = nx.ones(A.shape[1:], type_as=A)
    err = 1

    # build the convolution operator
    # this is equivalent to blurring on horizontal then vertical directions
    t = nx.linspace(0, 1, width)
    [Y, X] = nx.meshgrid(t, t)
    K1 = nx.exp(-(X - Y) ** 2 / reg)

    t = nx.linspace(0, 1, height)
    [Y, X] = nx.meshgrid(t, t)
    K2 = nx.exp(-(X - Y) ** 2 / reg)

    def convol_imgs(imgs):
        kx = nx.einsum("...ij,kjl->kil", K1, imgs)
        kxy = nx.einsum("...ij,klj->kli", K2, kx)
        return kxy

    KU = convol_imgs(U)
    for ii in range(numItermax):
        V = bar[None] / KU
        KV = convol_imgs(V)
        U = A / KV
        KU = convol_imgs(U)
        bar = c * nx.exp(
            nx.sum(weights[:, None, None] * nx.log(KU + stabThr), axis=0)
        )

        for _ in range(10):
            c = (c * bar / nx.squeeze(convol_imgs(c[None]))) ** 0.5

        if ii % 10 == 9:
            err = nx.sum(nx.std(V * KU, axis=0))
            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if ii % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))

            # debiased Sinkhorn does not converge monotonically
            # guarantee a few iterations are done before stopping
            if err < stopThr and ii > 20:
                break
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        log['niter'] = ii
        log['U'] = U
        return bar, log
    else:
        return bar


def _convolutional_barycenter2d_debiased_log(A, reg, weights=None, numItermax=10000,
                                             stopThr=1e-3, stabThr=1e-30, verbose=False,
                                             log=False, warn=True):
    r"""Compute the debiased barycenter of 2D images in log-domain.
     """

    A = list_to_array(A)
    n_hists, width, height = A.shape
    nx = get_backend(A)
    if nx.__name__ in ("jax", "tf"):
        raise NotImplementedError(
            "Log-domain functions are not yet implemented"
            " for Jax and TF. Use numpy or torch arrays instead."
        )
    if weights is None:
        weights = nx.ones((n_hists,), type_as=A) / n_hists
    else:
        assert (len(weights) == A.shape[0])

    if log:
        log = {'err': []}

    err = 1
    # build the convolution operator
    # this is equivalent to blurring on horizontal then vertical directions
    t = nx.linspace(0, 1, width)
    [Y, X] = nx.meshgrid(t, t)
    M1 = - (X - Y) ** 2 / reg

    t = nx.linspace(0, 1, height)
    [Y, X] = nx.meshgrid(t, t)
    M2 = - (X - Y) ** 2 / reg

    def convol_img(log_img):
        log_img = nx.logsumexp(M1[:, :, None] + log_img[None], axis=1)
        log_img = nx.logsumexp(M2[:, :, None] + log_img.T[None], axis=1).T
        return log_img

    logA = nx.log(A + stabThr)
    log_bar, c = nx.zeros((2, width, height), type_as=A)
    log_KU, G, F = nx.zeros((3, *logA.shape), type_as=A)
    err = 1
    for ii in range(numItermax):
        log_bar = nx.zeros((width, height), type_as=A)
        for k in range(n_hists):
            f = logA[k] - convol_img(G[k])
            log_KU[k] = convol_img(f)
            log_bar = log_bar + weights[k] * log_KU[k]
        log_bar += c
        for _ in range(10):
            c = 0.5 * (c + log_bar - convol_img(c))

        if ii % 10 == 9:
            err = nx.sum(nx.std(nx.exp(G + log_KU), axis=0))
            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if ii % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
            if err < stopThr and ii > 20:
                break
        G = log_bar[None, :, :] - log_KU

    else:
        if warn:
            warnings.warn("Convolutional Sinkhorn did not converge. "
                          "Try a larger number of iterations `numItermax` "
                          "or a larger entropy `reg`.")
    if log:
        log['niter'] = ii
        return nx.exp(log_bar), log
    else:
        return nx.exp(log_bar)


def unmix(a, D, M, M0, h0, reg, reg0, alpha, numItermax=1000,
          stopThr=1e-3, verbose=False, log=False, warn=True):
    r"""
    Compute the unmixing of an observation with a given dictionary using Wasserstein distance

    The function solve the following optimization problem:

    .. math::

       \mathbf{h} = \mathop{\arg \min}_\mathbf{h} \quad
       (1 - \alpha)  W_{\mathbf{M}, \mathrm{reg}}(\mathbf{a}, \mathbf{Dh}) +
       \alpha W_{\mathbf{M_0}, \mathrm{reg}_0}(\mathbf{h}_0, \mathbf{h})


    where :

    - :math:`W_{M,reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance
      with :math:`\mathbf{M}` loss matrix (see :py:func:`ot.bregman.sinkhorn`)
    - :math:`\mathbf{D}` is a dictionary of `n_atoms` atoms of dimension `dim_a`,
      its expected shape is `(dim_a, n_atoms)`
    - :math:`\mathbf{h}` is the estimated unmixing of dimension `n_atoms`
    - :math:`\mathbf{a}` is an observed distribution of dimension `dim_a`
    - :math:`\mathbf{h}_0` is a prior on :math:`\mathbf{h}` of dimension `dim_prior`
    - `reg` and :math:`\mathbf{M}` are respectively the regularization term and the
      cost matrix (`dim_a`, `dim_a`) for OT data fitting
    - `reg`:math:`_0` and :math:`\mathbf{M_0}` are respectively the regularization
      term and the cost matrix (`dim_prior`, `n_atoms`) regularization
    - :math:`\alpha` weight data fitting and regularization

    The optimization problem is solved following the algorithm described
    in :ref:`[4] <references-unmix>`


    Parameters
    ----------
    a : array-like, shape (dim_a)
        observed distribution (histogram, sums to 1)
    D : array-like, shape (dim_a, n_atoms)
        dictionary matrix
    M : array-like, shape (dim_a, dim_a)
        loss matrix
    M0 : array-like, shape (n_atoms, dim_prior)
        loss matrix
    h0 : array-like, shape (n_atoms,)
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
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.

    Returns
    -------
    h : array-like, shape (n_atoms,)
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-unmix:
    References
    ----------

    .. [4] S. Nakhostin, N. Courty, R. Flamary, D. Tuia, T. Corpetti,
        Supervised planetary unmixing with optimal transport, Whorkshop
        on Hyperspectral Image and Signal Processing :
        Evolution in Remote Sensing (WHISPERS), 2016.
    """

    a, D, M, M0, h0 = list_to_array(a, D, M, M0, h0)

    nx = get_backend(a, D, M, M0, h0)

    # M = M/np.median(M)
    K = nx.exp(-M / reg)

    # M0 = M0/np.median(M0)
    K0 = nx.exp(-M0 / reg0)
    old = h0

    err = 1
    # log = {'niter':0, 'all_err':[]}
    if log:
        log = {'err': []}

    for ii in range(numItermax):
        K = projC(K, a)
        K0 = projC(K0, h0)
        new = nx.sum(K0, axis=1)
        # we recombine the current selection from dictionnary
        inv_new = nx.dot(D, new)
        other = nx.sum(K, axis=1)
        # geometric interpolation
        delta = nx.exp(alpha * nx.log(other) + (1 - alpha) * nx.log(inv_new))
        K = projR(K, delta)
        K0 = nx.dot(D.T, delta / inv_new)[:, None] * K0
        err = nx.norm(nx.sum(K0, axis=1) - old)
        old = new
        if log:
            log['err'].append(err)

        if verbose:
            if ii % 200 == 0:
                print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(ii, err))
        if err < stopThr:
            break
    else:
        if warn:
            warnings.warn("Unmixing algorithm did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        log['niter'] = ii
        return nx.sum(K0, axis=1), log
    else:
        return nx.sum(K0, axis=1)


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


def empirical_sinkhorn(X_s, X_t, reg, a=None, b=None, metric='sqeuclidean',
                       numIterMax=10000, stopThr=1e-9, isLazy=False, batchSize=100, verbose=False,
                       log=False, warn=True, **kwargs):
    r'''
    Solve the entropic regularization optimal transport problem and return the
    OT matrix from empirical data

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`n_samples_a`, `n_samples_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)


    Parameters
    ----------
    X_s : array-like, shape (n_samples_a, dim)
        samples in the source domain
    X_t : array-like, shape (n_samples_b, dim)
        samples in the target domain
    reg : float
        Regularization term >0
    a : array-like, shape (n_samples_a,)
        samples weights in the source domain
    b : array-like, shape (n_samples_b,)
        samples weights in the target domain
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    isLazy: boolean, optional
        If True, then only calculate the cost matrix by block and return
        the dual potentials only (to save memory). If False, calculate full
        cost matrix and return outputs of sinkhorn function.
    batchSize: int or tuple of 2 int, optional
        Size of the batches used to compute the sinkhorn update without memory overhead.
        When a tuple is provided it sets the size of the left/right batches.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.


    Returns
    -------
    gamma : array-like, shape (n_samples_a, n_samples_b)
        Regularized optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_samples_a = 2
    >>> n_samples_b = 2
    >>> reg = 0.1
    >>> X_s = np.reshape(np.arange(n_samples_a, dtype=np.float64), (n_samples_a, 1))
    >>> X_t = np.reshape(np.arange(0, n_samples_b, dtype=np.float64), (n_samples_b, 1))
    >>> empirical_sinkhorn(X_s, X_t, reg=reg, verbose=False)  # doctest: +NORMALIZE_WHITESPACE
    array([[4.99977301e-01,  2.26989344e-05],
           [2.26989344e-05,  4.99977301e-01]])


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal
        Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for
        Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.
    '''

    X_s, X_t = list_to_array(X_s, X_t)

    nx = get_backend(X_s, X_t)

    ns, nt = X_s.shape[0], X_t.shape[0]
    if a is None:
        a = nx.from_numpy(unif(ns), type_as=X_s)
    if b is None:
        b = nx.from_numpy(unif(nt), type_as=X_s)

    if isLazy:
        if log:
            dict_log = {"err": []}

        log_a, log_b = nx.log(a), nx.log(b)
        f, g = nx.zeros((ns,), type_as=a), nx.zeros((nt,), type_as=a)

        if isinstance(batchSize, int):
            bs, bt = batchSize, batchSize
        elif isinstance(batchSize, tuple) and len(batchSize) == 2:
            bs, bt = batchSize[0], batchSize[1]
        else:
            raise ValueError("Batch size must be in integer or a tuple of two integers")

        range_s, range_t = range(0, ns, bs), range(0, nt, bt)

        lse_f = nx.zeros((ns,), type_as=a)
        lse_g = nx.zeros((nt,), type_as=a)

        X_s_np = nx.to_numpy(X_s)
        X_t_np = nx.to_numpy(X_t)

        for i_ot in range(numIterMax):

            lse_f_cols = []
            for i in range_s:
                M = dist(X_s_np[i:i + bs, :], X_t_np, metric=metric)
                M = nx.from_numpy(M, type_as=a)
                lse_f_cols.append(
                    nx.logsumexp(g[None, :] - M / reg, axis=1)
                )
            lse_f = nx.concatenate(lse_f_cols, axis=0)
            f = log_a - lse_f

            lse_g_cols = []
            for j in range_t:
                M = dist(X_s_np, X_t_np[j:j + bt, :], metric=metric)
                M = nx.from_numpy(M, type_as=a)
                lse_g_cols.append(
                    nx.logsumexp(f[:, None] - M / reg, axis=0)
                )
            lse_g = nx.concatenate(lse_g_cols, axis=0)
            g = log_b - lse_g

            if (i_ot + 1) % 10 == 0:
                m1_cols = []
                for i in range_s:
                    M = dist(X_s_np[i:i + bs, :], X_t_np, metric=metric)
                    M = nx.from_numpy(M, type_as=a)
                    m1_cols.append(
                        nx.sum(nx.exp(f[i:i + bs, None] + g[None, :] - M / reg), axis=1)
                    )
                m1 = nx.concatenate(m1_cols, axis=0)
                err = nx.sum(nx.abs(m1 - a))
                if log:
                    dict_log["err"].append(err)

                if verbose and (i_ot + 1) % 100 == 0:
                    print("Error in marginal at iteration {} = {}".format(i_ot + 1, err))

                if err <= stopThr:
                    break
        else:
            if warn:
                warnings.warn("Sinkhorn did not converge. You might want to "
                              "increase the number of iterations `numItermax` "
                              "or the regularization parameter `reg`.")
        if log:
            dict_log["u"] = f
            dict_log["v"] = g
            return (f, g, dict_log)
        else:
            return (f, g)

    else:
        M = dist(X_s, X_t, metric=metric)
        if log:
            pi, log = sinkhorn(a, b, M, reg, numItermax=numIterMax, stopThr=stopThr,
                               verbose=verbose, log=True, **kwargs)
            return pi, log
        else:
            pi = sinkhorn(a, b, M, reg, numItermax=numIterMax, stopThr=stopThr,
                          verbose=verbose, log=False, **kwargs)
            return pi


def empirical_sinkhorn2(X_s, X_t, reg, a=None, b=None, metric='sqeuclidean',
                        numIterMax=10000, stopThr=1e-9, isLazy=False,
                        batchSize=100, verbose=False, log=False, warn=True, **kwargs):
    r'''
    Solve the entropic regularization optimal transport problem from empirical
    data and return the OT loss


    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`n_samples_a`, `n_samples_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)


    Parameters
    ----------
    X_s : array-like, shape (n_samples_a, dim)
        samples in the source domain
    X_t : array-like, shape (n_samples_b, dim)
        samples in the target domain
    reg : float
        Regularization term >0
    a : array-like, shape (n_samples_a,)
        samples weights in the source domain
    b : array-like, shape (n_samples_b,)
        samples weights in the target domain
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    isLazy: boolean, optional
        If True, then only calculate the cost matrix by block and return
        the dual potentials only (to save memory). If False, calculate
        full cost matrix and return outputs of sinkhorn function.
    batchSize: int or tuple of 2 int, optional
        Size of the batches used to compute the sinkhorn update without memory overhead.
        When a tuple is provided it sets the size of the left/right batches.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.


    Returns
    -------
    W : (n_hists) array-like or float
        Optimal transportation loss for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_samples_a = 2
    >>> n_samples_b = 2
    >>> reg = 0.1
    >>> X_s = np.reshape(np.arange(n_samples_a, dtype=np.float64), (n_samples_a, 1))
    >>> X_t = np.reshape(np.arange(0, n_samples_b, dtype=np.float64), (n_samples_b, 1))
    >>> b = np.full((n_samples_b, 3), 1/n_samples_b)
    >>> empirical_sinkhorn2(X_s, X_t, b=b, reg=reg, verbose=False)
    array([4.53978687e-05, 4.53978687e-05, 4.53978687e-05])


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling
        Algorithms for Entropy Regularized Transport Problems.
        arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems.
        arXiv preprint arXiv:1607.05816.
    '''

    X_s, X_t = list_to_array(X_s, X_t)

    nx = get_backend(X_s, X_t)

    ns, nt = X_s.shape[0], X_t.shape[0]
    if a is None:
        a = nx.from_numpy(unif(ns), type_as=X_s)
    if b is None:
        b = nx.from_numpy(unif(nt), type_as=X_s)

    if isLazy:
        if log:
            f, g, dict_log = empirical_sinkhorn(X_s, X_t, reg, a, b, metric,
                                                numIterMax=numIterMax,
                                                stopThr=stopThr,
                                                isLazy=isLazy,
                                                batchSize=batchSize,
                                                verbose=verbose, log=log,
                                                warn=warn)
        else:
            f, g = empirical_sinkhorn(X_s, X_t, reg, a, b, metric,
                                      numIterMax=numIterMax, stopThr=stopThr,
                                      isLazy=isLazy, batchSize=batchSize,
                                      verbose=verbose, log=log,
                                      warn=warn)

        bs = batchSize if isinstance(batchSize, int) else batchSize[0]
        range_s = range(0, ns, bs)

        loss = 0

        X_s_np = nx.to_numpy(X_s)
        X_t_np = nx.to_numpy(X_t)

        for i in range_s:
            M_block = dist(X_s_np[i:i + bs, :], X_t_np, metric=metric)
            M_block = nx.from_numpy(M_block, type_as=a)
            pi_block = nx.exp(f[i:i + bs, None] + g[None, :] - M_block / reg)
            loss += nx.sum(M_block * pi_block)

        if log:
            return loss, dict_log
        else:
            return loss

    else:
        M = dist(nx.to_numpy(X_s), nx.to_numpy(X_t), metric=metric)
        M = nx.from_numpy(M, type_as=a)

        if log:
            sinkhorn_loss, log = sinkhorn2(a, b, M, reg, numItermax=numIterMax,
                                           stopThr=stopThr, verbose=verbose, log=log,
                                           warn=warn, **kwargs)
            return sinkhorn_loss, log
        else:
            sinkhorn_loss = sinkhorn2(a, b, M, reg, numItermax=numIterMax,
                                      stopThr=stopThr, verbose=verbose, log=log,
                                      warn=warn, **kwargs)
            return sinkhorn_loss


def empirical_sinkhorn_divergence(X_s, X_t, reg, a=None, b=None, metric='sqeuclidean',
                                  numIterMax=10000, stopThr=1e-9,
                                  verbose=False, log=False, warn=True,
                                  **kwargs):
    r'''
    Compute the sinkhorn divergence loss from empirical data

    The function solves the following optimization problems and return the
    sinkhorn divergence :math:`S`:

    .. math::

        W &= \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot\Omega(\gamma)

        W_a &= \min_{\gamma_a} \quad \langle \gamma_a, \mathbf{M_a} \rangle_F +
        \mathrm{reg} \cdot\Omega(\gamma_a)

        W_b &= \min_{\gamma_b} \quad \langle \gamma_b, \mathbf{M_b} \rangle_F +
        \mathrm{reg} \cdot\Omega(\gamma_b)

        S &= W - \frac{W_a + W_b}{2}

    .. math::
        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

             \gamma_a \mathbf{1} &= \mathbf{a}

             \gamma_a^T \mathbf{1} &= \mathbf{a}

             \gamma_a &\geq 0

             \gamma_b \mathbf{1} &= \mathbf{b}

             \gamma_b^T \mathbf{1} &= \mathbf{b}

             \gamma_b &\geq 0
    where :

    - :math:`\mathbf{M}` (resp. :math:`\mathbf{M_a}`, :math:`\mathbf{M_b}`)
      is the (`n_samples_a`, `n_samples_b`) metric cost matrix
      (resp (`n_samples_a, n_samples_a`) and (`n_samples_b`, `n_samples_b`))
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)


    Parameters
    ----------
    X_s : array-like, shape (n_samples_a, dim)
        samples in the source domain
    X_t : array-like, shape (n_samples_b, dim)
        samples in the target domain
    reg : float
        Regularization term >0
    a : array-like, shape (n_samples_a,)
        samples weights in the source domain
    b : array-like, shape (n_samples_b,)
        samples weights in the target domain
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
    W : (1,) array-like
        Optimal transportation symmetrized loss for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------
    >>> n_samples_a = 2
    >>> n_samples_b = 4
    >>> reg = 0.1
    >>> X_s = np.reshape(np.arange(n_samples_a, dtype=np.float64), (n_samples_a, 1))
    >>> X_t = np.reshape(np.arange(0, n_samples_b, dtype=np.float64), (n_samples_b, 1))
    >>> empirical_sinkhorn_divergence(X_s, X_t, reg)  # doctest: +ELLIPSIS
    1.499887176049052


    References
    ----------
    .. [23] Aude Genevay, Gabriel Peyré, Marco Cuturi, Learning Generative
        Models with Sinkhorn Divergences,  Proceedings of the Twenty-First
        International Conference on Artficial Intelligence and Statistics,
        (AISTATS) 21, 2018
    '''
    if log:
        sinkhorn_loss_ab, log_ab = empirical_sinkhorn2(X_s, X_t, reg, a, b, metric=metric,
                                                       numIterMax=numIterMax,
                                                       stopThr=1e-9, verbose=verbose,
                                                       log=log, warn=warn, **kwargs)

        sinkhorn_loss_a, log_a = empirical_sinkhorn2(X_s, X_s, reg, a, a, metric=metric,
                                                     numIterMax=numIterMax,
                                                     stopThr=1e-9, verbose=verbose,
                                                     log=log, warn=warn, **kwargs)

        sinkhorn_loss_b, log_b = empirical_sinkhorn2(X_t, X_t, reg, b, b, metric=metric,
                                                     numIterMax=numIterMax,
                                                     stopThr=1e-9, verbose=verbose,
                                                     log=log, warn=warn, **kwargs)

        sinkhorn_div = sinkhorn_loss_ab - 0.5 * (sinkhorn_loss_a + sinkhorn_loss_b)

        log = {}
        log['sinkhorn_loss_ab'] = sinkhorn_loss_ab
        log['sinkhorn_loss_a'] = sinkhorn_loss_a
        log['sinkhorn_loss_b'] = sinkhorn_loss_b
        log['log_sinkhorn_ab'] = log_ab
        log['log_sinkhorn_a'] = log_a
        log['log_sinkhorn_b'] = log_b

        return max(0, sinkhorn_div), log

    else:
        sinkhorn_loss_ab = empirical_sinkhorn2(X_s, X_t, reg, a, b, metric=metric,
                                               numIterMax=numIterMax, stopThr=1e-9,
                                               verbose=verbose, log=log,
                                               warn=warn, **kwargs)

        sinkhorn_loss_a = empirical_sinkhorn2(X_s, X_s, reg, a, a, metric=metric,
                                              numIterMax=numIterMax, stopThr=1e-9,
                                              verbose=verbose, log=log,
                                              warn=warn, **kwargs)

        sinkhorn_loss_b = empirical_sinkhorn2(X_t, X_t, reg, b, b, metric=metric,
                                              numIterMax=numIterMax, stopThr=1e-9,
                                              verbose=verbose, log=log,
                                              warn=warn, **kwargs)

        sinkhorn_div = sinkhorn_loss_ab - 0.5 * (sinkhorn_loss_a + sinkhorn_loss_b)
        return max(0, sinkhorn_div)


def screenkhorn(a, b, M, reg, ns_budget=None, nt_budget=None, uniform=False,
                restricted=True, maxiter=10000, maxfun=10000, pgtol=1e-09,
                verbose=False, log=False):
    r"""
    Screening Sinkhorn Algorithm for Regularized Optimal Transport

    The function solves an approximate dual of Sinkhorn divergence :ref:`[2]
    <references-screenkhorn>` which is written as the following optimization problem:

    .. math::

        (\mathbf{u}, \mathbf{v}) = \mathop{\arg \min}_{\mathbf{u}, \mathbf{v}} \quad
        \mathbf{1}_{ns}^T \mathbf{B}(\mathbf{u}, \mathbf{v}) \mathbf{1}_{nt} -
        \langle \kappa \mathbf{u}, \mathbf{a} \rangle -
        \langle \frac{1}{\kappa} \mathbf{v}, \mathbf{b} \rangle

    where:

    .. math::

        \mathbf{B}(\mathbf{u}, \mathbf{v}) = \mathrm{diag}(e^\mathbf{u}) \mathbf{K} \mathrm{diag}(e^\mathbf{v}) \text{, with } \mathbf{K} = e^{-\mathbf{M} / \mathrm{reg}} \text{ and}

    .. math::

        s.t. \ e^{u_i} &\geq \epsilon / \kappa, \forall i \in \{1, \ldots, ns\}

             e^{v_j} &\geq \epsilon \kappa, \forall j \in \{1, \ldots, nt\}

    The parameters `kappa` and `epsilon` are determined w.r.t the couple number
    budget of points (`ns_budget`, `nt_budget`), see Equation (5)
    in :ref:`[26] <references-screenkhorn>`


    Parameters
    ----------
    a: array-like, shape=(ns,)
        samples weights in the source domain
    b: array-like, shape=(nt,)
        samples weights in the target domain
    M: array-like, shape=(ns, nt)
        Cost matrix
    reg: `float`
        Level of the entropy regularisation
    ns_budget: `int`, default=None
        Number budget of points to be kept in the source domain.
        If it is None then 50% of the source sample points will be kept
    nt_budget: `int`, default=None
        Number budget of points to be kept in the target domain.
        If it is None then 50% of the target sample points will be kept
    uniform: `bool`, default=False
        If `True`, the source and target distribution are supposed to be uniform,
        i.e., :math:`a_i = 1 / ns` and :math:`b_j = 1 / nt`
    restricted : `bool`, default=True
         If `True`, a warm-start initialization for the  L-BFGS-B solver
         using a restricted Sinkhorn algorithm with at most 5 iterations
    maxiter: `int`, default=10000
      Maximum number of iterations in LBFGS solver
    maxfun: `int`, default=10000
      Maximum number of function evaluations in LBFGS solver
    pgtol: `float`, default=1e-09
      Final objective function accuracy in LBFGS solver
    verbose: `bool`, default=False
        If `True`, display informations about the cardinals of the active sets
        and the parameters kappa and epsilon


    .. admonition:: Dependency

        To gain more efficiency, :py:func:`ot.bregman.screenkhorn` needs to call the "Bottleneck"
        package (https://pypi.org/project/Bottleneck/) in the screening pre-processing step.

        If Bottleneck isn't installed, the following error message appears:

        "Bottleneck module doesn't exist. Install it from https://pypi.org/project/Bottleneck/"


    Returns
    -------
    gamma : array-like, shape=(ns, nt)
        Screened optimal transportation matrix for the given parameters

    log : `dict`, default=False
      Log dictionary return only if log==True in parameters


    .. _references-screenkhorn:
    References
    -----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport,
        Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [26] Alaya M. Z., Bérar M., Gasso G., Rakotomamonjy A. (2019).
        Screening Sinkhorn Algorithm for Regularized Optimal Transport (NIPS) 33, 2019

    """
    # check if bottleneck module exists
    try:
        import bottleneck
    except ImportError:
        warnings.warn(
            "Bottleneck module is not installed. Install it from"
            " https://pypi.org/project/Bottleneck/ for better performance.")
        bottleneck = np

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)
    if nx.__name__ in ("jax", "tf"):
        raise TypeError("JAX or TF arrays have been received but screenkhorn is not "
                        "compatible with neither JAX nor TF.")

    ns, nt = M.shape

    # by default, we keep only 50% of the sample data points
    if ns_budget is None:
        ns_budget = int(np.floor(0.5 * ns))
    if nt_budget is None:
        nt_budget = int(np.floor(0.5 * nt))

    # calculate the Gibbs kernel
    K = nx.exp(-M / reg)

    def projection(u, epsilon):
        u = nx.maximum(u, epsilon)
        return u

    # ----------------------------------------------------------------------------------------------------------------#
    #                                          Step 1: Screening pre-processing                                       #
    # ----------------------------------------------------------------------------------------------------------------#

    if ns_budget == ns and nt_budget == nt:
        # full number of budget points (ns, nt) = (ns_budget, nt_budget)
        Isel = nx.from_numpy(np.ones(ns, dtype=bool))
        Jsel = nx.from_numpy(np.ones(nt, dtype=bool))
        epsilon = 0.0
        kappa = 1.0

        cst_u = 0.
        cst_v = 0.

        bounds_u = [(0.0, np.inf)] * ns
        bounds_v = [(0.0, np.inf)] * nt

        a_I = a
        b_J = b
        K_IJ = K
        K_IJc = []
        K_IcJ = []

        vec_eps_IJc = nx.zeros((nt,), type_as=M)
        vec_eps_IcJ = nx.zeros((ns,), type_as=M)

    else:
        # sum of rows and columns of K
        K_sum_cols = nx.sum(K, axis=1)
        K_sum_rows = nx.sum(K, axis=0)

        if uniform:
            if ns / ns_budget < 4:
                aK_sort = nx.sort(K_sum_cols)
                epsilon_u_square = a[0] / aK_sort[ns_budget - 1]
            else:
                aK_sort = nx.from_numpy(
                    bottleneck.partition(nx.to_numpy(K_sum_cols), ns_budget - 1)[ns_budget - 1],
                    type_as=M
                )
                epsilon_u_square = a[0] / aK_sort

            if nt / nt_budget < 4:
                bK_sort = nx.sort(K_sum_rows)
                epsilon_v_square = b[0] / bK_sort[nt_budget - 1]
            else:
                bK_sort = nx.from_numpy(
                    bottleneck.partition(nx.to_numpy(K_sum_rows), nt_budget - 1)[nt_budget - 1],
                    type_as=M
                )
                epsilon_v_square = b[0] / bK_sort
        else:
            aK = a / K_sum_cols
            bK = b / K_sum_rows

            aK_sort = nx.flip(nx.sort(aK), axis=0)
            epsilon_u_square = aK_sort[ns_budget - 1]

            bK_sort = nx.flip(nx.sort(bK), axis=0)
            epsilon_v_square = bK_sort[nt_budget - 1]

        # active sets I and J (see Lemma 1 in [26])
        Isel = a >= epsilon_u_square * K_sum_cols
        Jsel = b >= epsilon_v_square * K_sum_rows

        if nx.sum(Isel) != ns_budget:
            if uniform:
                aK = a / K_sum_cols
                aK_sort = nx.flip(nx.sort(aK), axis=0)
            epsilon_u_square = nx.mean(aK_sort[ns_budget - 1:ns_budget + 1])
            Isel = a >= epsilon_u_square * K_sum_cols
            ns_budget = nx.sum(Isel)

        if nx.sum(Jsel) != nt_budget:
            if uniform:
                bK = b / K_sum_rows
                bK_sort = nx.flip(nx.sort(bK), axis=0)
            epsilon_v_square = nx.mean(bK_sort[nt_budget - 1:nt_budget + 1])
            Jsel = b >= epsilon_v_square * K_sum_rows
            nt_budget = nx.sum(Jsel)

        epsilon = (epsilon_u_square * epsilon_v_square) ** (1 / 4)
        kappa = (epsilon_v_square / epsilon_u_square) ** (1 / 2)

        if verbose:
            print("epsilon = %s\n" % epsilon)
            print("kappa = %s\n" % kappa)
            print('Cardinality of selected points: |Isel| = %s \t |Jsel| = %s \n'
                  % (sum(Isel), sum(Jsel)))

        # Ic, Jc: complementary of the active sets I and J
        Ic = ~Isel
        Jc = ~Jsel

        K_IJ = K[np.ix_(Isel, Jsel)]
        K_IcJ = K[np.ix_(Ic, Jsel)]
        K_IJc = K[np.ix_(Isel, Jc)]

        K_min = nx.min(K_IJ)
        if K_min == 0:
            K_min = float(np.finfo(float).tiny)

        # a_I, b_J, a_Ic, b_Jc
        a_I = a[Isel]
        b_J = b[Jsel]
        if not uniform:
            a_I_min = nx.min(a_I)
            a_I_max = nx.max(a_I)
            b_J_max = nx.max(b_J)
            b_J_min = nx.min(b_J)
        else:
            a_I_min = a_I[0]
            a_I_max = a_I[0]
            b_J_max = b_J[0]
            b_J_min = b_J[0]

        # box constraints in L-BFGS-B (see Proposition 1 in [26])
        bounds_u = [(max(a_I_min / ((nt - nt_budget) * epsilon + nt_budget * (b_J_max / (
                    ns * epsilon * kappa * K_min))), epsilon / kappa), a_I_max / (nt * epsilon * K_min))] * ns_budget

        bounds_v = [(
            max(b_J_min / ((ns - ns_budget) * epsilon + ns_budget * (kappa * a_I_max / (nt * epsilon * K_min))),
                epsilon * kappa), b_J_max / (ns * epsilon * K_min))] * nt_budget

        # pre-calculated constants for the objective
        vec_eps_IJc = epsilon * kappa * nx.sum(
            K_IJc * nx.ones((nt - nt_budget,), type_as=M)[None, :],
            axis=1
        )
        vec_eps_IcJ = (epsilon / kappa) * nx.sum(
            nx.ones((ns - ns_budget,), type_as=M)[:, None] * K_IcJ,
            axis=0
        )

    # initialisation
    u0 = nx.full((ns_budget,), 1. / ns_budget + epsilon / kappa, type_as=M)
    v0 = nx.full((nt_budget,), 1. / nt_budget + epsilon * kappa, type_as=M)

    # pre-calculed constants for Restricted Sinkhorn (see Algorithm 1 in supplementary of [26])
    if restricted:
        if ns_budget != ns or nt_budget != nt:
            cst_u = kappa * epsilon * nx.sum(K_IJc, axis=1)
            cst_v = epsilon * nx.sum(K_IcJ, axis=0) / kappa

        for _ in range(5):  # 5 iterations
            K_IJ_v = nx.dot(K_IJ.T, u0) + cst_v
            v0 = b_J / (kappa * K_IJ_v)
            KIJ_u = nx.dot(K_IJ, v0) + cst_u
            u0 = (kappa * a_I) / KIJ_u

        u0 = projection(u0, epsilon / kappa)
        v0 = projection(v0, epsilon * kappa)

    else:
        u0 = u0
        v0 = v0

    def restricted_sinkhorn(usc, vsc, max_iter=5):
        """
        Restricted Sinkhorn Algorithm as a warm-start initialized pointfor L-BFGS-B)
        """
        for _ in range(max_iter):
            K_IJ_v = nx.dot(K_IJ.T, usc) + cst_v
            vsc = b_J / (kappa * K_IJ_v)
            KIJ_u = nx.dot(K_IJ, vsc) + cst_u
            usc = (kappa * a_I) / KIJ_u

        usc = projection(usc, epsilon / kappa)
        vsc = projection(vsc, epsilon * kappa)

        return usc, vsc

    def screened_obj(usc, vsc):
        part_IJ = (
            nx.dot(nx.dot(usc, K_IJ), vsc)
            - kappa * nx.dot(a_I, nx.log(usc))
            - (1. / kappa) * nx.dot(b_J, nx.log(vsc))
        )
        part_IJc = nx.dot(usc, vec_eps_IJc)
        part_IcJ = nx.dot(vec_eps_IcJ, vsc)
        psi_epsilon = part_IJ + part_IJc + part_IcJ
        return psi_epsilon

    def screened_grad(usc, vsc):
        # gradients of Psi_(kappa,epsilon) w.r.t u and v
        grad_u = nx.dot(K_IJ, vsc) + vec_eps_IJc - kappa * a_I / usc
        grad_v = nx.dot(K_IJ.T, usc) + vec_eps_IcJ - (1. / kappa) * b_J / vsc
        return grad_u, grad_v

    def bfgspost(theta):
        u = theta[:ns_budget]
        v = theta[ns_budget:]
        # objective
        f = screened_obj(u, v)
        # gradient
        g_u, g_v = screened_grad(u, v)
        g = nx.concatenate([g_u, g_v], axis=0)
        return nx.to_numpy(f), nx.to_numpy(g)

    # ----------------------------------------------------------------------------------------------------------------#
    #                                           Step 2: L-BFGS-B solver                                              #
    # ----------------------------------------------------------------------------------------------------------------#

    u0, v0 = restricted_sinkhorn(u0, v0)
    theta0 = nx.concatenate([u0, v0], axis=0)

    bounds = bounds_u + bounds_v  # constraint bounds

    def obj(theta):
        return bfgspost(nx.from_numpy(theta, type_as=M))

    theta, _, _ = fmin_l_bfgs_b(func=obj,
                                x0=theta0,
                                bounds=bounds,
                                maxfun=maxfun,
                                pgtol=pgtol,
                                maxiter=maxiter)
    theta = nx.from_numpy(theta, type_as=M)

    usc = theta[:ns_budget]
    vsc = theta[ns_budget:]

    usc_full = nx.full((ns,), epsilon / kappa, type_as=M)
    vsc_full = nx.full((nt,), epsilon * kappa, type_as=M)
    usc_full[Isel] = usc
    vsc_full[Jsel] = vsc

    if log:
        log = {}
        log['u'] = usc_full
        log['v'] = vsc_full
        log['Isel'] = Isel
        log['Jsel'] = Jsel

    gamma = usc_full[:, None] * K * vsc_full[None, :]
    gamma = gamma / nx.sum(gamma)

    if log:
        return gamma, log
    else:
        return gamma
