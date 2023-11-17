# -*- coding: utf-8 -*-
"""
Bregman projections solvers for entropic regularized OT for empirical distributions
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Kilian Fatras <kilian.fatras@irisa.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

import warnings

from ..utils import dist, list_to_array, unif, LazyTensor
from ..backend import get_backend

from ._sinkhorn import sinkhorn, sinkhorn2


def get_sinkhorn_lazytensor(X_a, X_b, f, g, metric='sqeuclidean', reg=1e-1, nx=None):
    r""" Get a LazyTensor of Sinkhorn solution from the dual potentials

    The returned LazyTensor is
    :math:`\mathbf{T} = exp(  \mathbf{f} \mathbf{1}_b^\top + \mathbf{1}_a \mathbf{g}^\top - \mathbf{C}/reg)`, where :math:`\mathbf{C}` is the pairwise metric matrix between samples :math:`\mathbf{X}_a` and :math:`\mathbf{X}_b`.

    Parameters
    ----------
    X_a : array-like, shape (n_samples_a, dim)
        samples in the source domain
    X_b : array-like, shape (n_samples_b, dim)
        samples in the target domain
    f : array-like, shape (n_samples_a,)
        First dual potentials (log space)
    g : array-like, shape (n_samples_b,)
        Second dual potentials (log space)
    metric : str, default='sqeuclidean'
        Metric used for the cost matrix computation
    reg : float, default=1e-1
        Regularization term >0
    nx : Backend(), default=None
        Numerical backend used


    Returns
    -------
    T : LazyTensor
        Sinkhorn solution tensor
    """

    if nx is None:
        nx = get_backend(X_a, X_b, f, g)

    shape = (X_a.shape[0], X_b.shape[0])

    def func(i, j, X_a, X_b, f, g, metric, reg):
        C = dist(X_a[i], X_b[j], metric=metric)
        return nx.exp(f[i, None] + g[None, j] - C / reg)

    T = LazyTensor(shape, func, X_a=X_a, X_b=X_b, f=f, g=g, metric=metric, reg=reg)

    return T


def empirical_sinkhorn(X_s, X_t, reg, a=None, b=None, metric='sqeuclidean',
                       numIterMax=10000, stopThr=1e-9, isLazy=False, batchSize=100, verbose=False,
                       log=False, warn=True, warmstart=None, **kwargs):
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
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)


    Returns
    -------
    gamma : array-like, shape (n_samples_a, n_samples_b)
        Regularized optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import numpy as np
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
        if warmstart is None:
            f, g = nx.zeros((ns,), type_as=a), nx.zeros((nt,), type_as=a)
        else:
            f, g = warmstart

        if isinstance(batchSize, int):
            bs, bt = batchSize, batchSize
        elif isinstance(batchSize, tuple) and len(batchSize) == 2:
            bs, bt = batchSize[0], batchSize[1]
        else:
            raise ValueError(
                "Batch size must be in integer or a tuple of two integers")

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
                        nx.sum(nx.exp(f[i:i + bs, None] +
                                      g[None, :] - M / reg), axis=1)
                    )
                m1 = nx.concatenate(m1_cols, axis=0)
                err = nx.sum(nx.abs(m1 - a))
                if log:
                    dict_log["err"].append(err)

                if verbose and (i_ot + 1) % 100 == 0:
                    print("Error in marginal at iteration {} = {}".format(
                        i_ot + 1, err))

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
            dict_log["niter"] = i_ot
            dict_log["lazy_plan"] = get_sinkhorn_lazytensor(X_s, X_t, f, g, metric, reg)
            return (f, g, dict_log)
        else:
            return (f, g)

    else:
        M = dist(X_s, X_t, metric=metric)
        if log:
            pi, log = sinkhorn(a, b, M, reg, numItermax=numIterMax, stopThr=stopThr,
                               verbose=verbose, log=True, warmstart=warmstart, **kwargs)
            return pi, log
        else:
            pi = sinkhorn(a, b, M, reg, numItermax=numIterMax, stopThr=stopThr,
                          verbose=verbose, log=False, warmstart=warmstart, **kwargs)
            return pi


def empirical_sinkhorn2(X_s, X_t, reg, a=None, b=None, metric='sqeuclidean',
                        numIterMax=10000, stopThr=1e-9, isLazy=False, batchSize=100,
                        verbose=False, log=False, warn=True, warmstart=None, **kwargs):
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

    and returns :math:`\langle \gamma^*, \mathbf{M} \rangle_F` (without
    the entropic contribution).


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
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

    Returns
    -------
    W : (n_hists) array-like or float
        Optimal transportation loss for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import numpy as np
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
                                                warn=warn,
                                                warmstart=warmstart)
        else:
            f, g = empirical_sinkhorn(X_s, X_t, reg, a, b, metric,
                                      numIterMax=numIterMax,
                                      stopThr=stopThr,
                                      isLazy=isLazy, batchSize=batchSize,
                                      verbose=verbose, log=log,
                                      warn=warn,
                                      warmstart=warmstart)

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
        M = dist(X_s, X_t, metric=metric)

        if log:
            sinkhorn_loss, log = sinkhorn2(a, b, M, reg, numItermax=numIterMax,
                                           stopThr=stopThr, verbose=verbose, log=log,
                                           warn=warn, warmstart=warmstart, **kwargs)
            return sinkhorn_loss, log
        else:
            sinkhorn_loss = sinkhorn2(a, b, M, reg, numItermax=numIterMax,
                                      stopThr=stopThr, verbose=verbose, log=log,
                                      warn=warn, warmstart=warmstart, **kwargs)
            return sinkhorn_loss


def empirical_sinkhorn_divergence(X_s, X_t, reg, a=None, b=None, metric='sqeuclidean',
                                  numIterMax=10000, stopThr=1e-9, verbose=False,
                                  log=False, warn=True, warmstart=None, **kwargs):
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

    and returns :math:`\langle \gamma^*, \mathbf{M} \rangle_F -(\langle \gamma^*_a, \mathbf{M_a} \rangle_F + \langle
    \gamma^*_b , \mathbf{M_b} \rangle_F)/2`.

    .. note: The current implementation does not account for the entropic contributions and thus differs from the
    Sinkhorn divergence as introduced in the literature. The possibility to account for the entropic contributions
    will be provided in a future release.


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
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

    Returns
    -------
    W : (1,) array-like
        Optimal transportation symmetrized loss for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import numpy as np
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
        International Conference on Artificial Intelligence and Statistics,
        (AISTATS) 21, 2018
    '''
    X_s, X_t = list_to_array(X_s, X_t)

    nx = get_backend(X_s, X_t)
    if warmstart is None:
        warmstart_a, warmstart_b = None, None
    else:
        u, v = warmstart
        warmstart_a = (u, u)
        warmstart_b = (v, v)

    if log:
        sinkhorn_loss_ab, log_ab = empirical_sinkhorn2(X_s, X_t, reg, a, b, metric=metric,
                                                       numIterMax=numIterMax, stopThr=stopThr,
                                                       verbose=verbose, log=log, warn=warn,
                                                       warmstart=warmstart, **kwargs)

        sinkhorn_loss_a, log_a = empirical_sinkhorn2(X_s, X_s, reg, a, a, metric=metric,
                                                     numIterMax=numIterMax, stopThr=stopThr,
                                                     verbose=verbose, log=log, warn=warn,
                                                     warmstart=warmstart_a, **kwargs)

        sinkhorn_loss_b, log_b = empirical_sinkhorn2(X_t, X_t, reg, b, b, metric=metric,
                                                     numIterMax=numIterMax, stopThr=stopThr,
                                                     verbose=verbose, log=log, warn=warn,
                                                     warmstart=warmstart_b, **kwargs)

        sinkhorn_div = sinkhorn_loss_ab - 0.5 * \
            (sinkhorn_loss_a + sinkhorn_loss_b)

        log = {}
        log['sinkhorn_loss_ab'] = sinkhorn_loss_ab
        log['sinkhorn_loss_a'] = sinkhorn_loss_a
        log['sinkhorn_loss_b'] = sinkhorn_loss_b
        log['log_sinkhorn_ab'] = log_ab
        log['log_sinkhorn_a'] = log_a
        log['log_sinkhorn_b'] = log_b

        return nx.maximum(0, sinkhorn_div), log

    else:
        sinkhorn_loss_ab = empirical_sinkhorn2(X_s, X_t, reg, a, b, metric=metric,
                                               numIterMax=numIterMax, stopThr=stopThr,
                                               verbose=verbose, log=log, warn=warn,
                                               warmstart=warmstart, **kwargs)

        sinkhorn_loss_a = empirical_sinkhorn2(X_s, X_s, reg, a, a, metric=metric,
                                              numIterMax=numIterMax, stopThr=stopThr,
                                              verbose=verbose, log=log, warn=warn,
                                              warmstart=warmstart_a, **kwargs)

        sinkhorn_loss_b = empirical_sinkhorn2(X_t, X_t, reg, b, b, metric=metric,
                                              numIterMax=numIterMax, stopThr=stopThr,
                                              verbose=verbose, log=log, warn=warn,
                                              warmstart=warmstart_b, **kwargs)

        sinkhorn_div = sinkhorn_loss_ab - 0.5 * \
            (sinkhorn_loss_a + sinkhorn_loss_b)
        return nx.maximum(0, sinkhorn_div)
