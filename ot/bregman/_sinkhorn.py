# -*- coding: utf-8 -*-
"""
Bregman projections solvers for entropic regularized OT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Alexander Tong <alexander.tong@yale.edu>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

import warnings

import numpy as np
from ..utils import list_to_array
from ..backend import get_backend


def sinkhorn(
    a,
    b,
    M,
    reg,
    method="sinkhorn",
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
    **kwargs,
):
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
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

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
    ot.bregman.sinkhorn_epsilon_scaling: Sinkhorn with epsilon scaling
        :ref:`[9] <references-sinkhorn>` :ref:`[10] <references-sinkhorn>`

    """

    if method.lower() == "sinkhorn":
        return sinkhorn_knopp(
            a,
            b,
            M,
            reg,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            warn=warn,
            warmstart=warmstart,
            **kwargs,
        )
    elif method.lower() == "sinkhorn_log":
        return sinkhorn_log(
            a,
            b,
            M,
            reg,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            warn=warn,
            warmstart=warmstart,
            **kwargs,
        )
    elif method.lower() == "greenkhorn":
        return greenkhorn(
            a,
            b,
            M,
            reg,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            warn=warn,
            warmstart=warmstart,
        )
    elif method.lower() == "sinkhorn_stabilized":
        return sinkhorn_stabilized(
            a,
            b,
            M,
            reg,
            numItermax=numItermax,
            stopThr=stopThr,
            warmstart=warmstart,
            verbose=verbose,
            log=log,
            warn=warn,
            **kwargs,
        )
    elif method.lower() == "sinkhorn_epsilon_scaling":
        return sinkhorn_epsilon_scaling(
            a,
            b,
            M,
            reg,
            numItermax=numItermax,
            stopThr=stopThr,
            warmstart=warmstart,
            verbose=verbose,
            log=log,
            warn=warn,
            **kwargs,
        )
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn2(
    a,
    b,
    M,
    reg,
    method="sinkhorn",
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=False,
    warmstart=None,
    **kwargs,
):
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

    and returns :math:`\langle \gamma^*, \mathbf{M} \rangle_F` (without
    the entropic contribution).

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
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

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
        if method.lower() == "sinkhorn":
            res = sinkhorn_knopp(
                a,
                b,
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warn=warn,
                warmstart=warmstart,
                **kwargs,
            )
        elif method.lower() == "sinkhorn_log":
            res = sinkhorn_log(
                a,
                b,
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warn=warn,
                warmstart=warmstart,
                **kwargs,
            )
        elif method.lower() == "sinkhorn_stabilized":
            res = sinkhorn_stabilized(
                a,
                b,
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                warmstart=warmstart,
                verbose=verbose,
                log=log,
                warn=warn,
                **kwargs,
            )
        else:
            raise ValueError("Unknown method '%s'." % method)
        if log:
            return nx.sum(M * res[0]), res[1]
        else:
            return nx.sum(M * res)

    else:
        if method.lower() == "sinkhorn":
            return sinkhorn_knopp(
                a,
                b,
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warn=warn,
                warmstart=warmstart,
                **kwargs,
            )
        elif method.lower() == "sinkhorn_log":
            return sinkhorn_log(
                a,
                b,
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warn=warn,
                warmstart=warmstart,
                **kwargs,
            )
        elif method.lower() == "sinkhorn_stabilized":
            return sinkhorn_stabilized(
                a,
                b,
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                warmstart=warmstart,
                verbose=verbose,
                log=log,
                warn=warn,
                **kwargs,
            )
        else:
            raise ValueError("Unknown method '%s'." % method)


def sinkhorn_knopp(
    a,
    b,
    M,
    reg,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
    **kwargs,
):
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
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

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
        log = {"err": []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if n_hists:
            u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
            v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
        else:
            u = nx.ones(dim_a, type_as=M) / dim_a
            v = nx.ones(dim_b, type_as=M) / dim_b
    else:
        u, v = nx.exp(warmstart[0]), nx.exp(warmstart[1])

    K = nx.exp(M / (-reg))

    Kp = (1 / a).reshape(-1, 1) * K

    err = 1
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = nx.dot(K.T, u)
        v = b / KtransposeU
        u = 1.0 / nx.dot(Kp, v)

        if (
            nx.any(KtransposeU == 0)
            or nx.any(nx.isnan(u))
            or nx.any(nx.isnan(v))
            or nx.any(nx.isinf(u))
            or nx.any(nx.isinf(v))
        ):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn("Warning: numerical errors at iteration %d" % ii)
            u = uprev
            v = vprev
            break
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                tmp2 = nx.einsum("ik,ij,jk->jk", u, K, v)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = nx.einsum("i,ij,j->j", u, K, v)
            err = nx.norm(tmp2 - b)  # violation of marginal
            if log:
                log["err"].append(err)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(ii, err))
    else:
        if warn:
            warnings.warn(
                "Sinkhorn did not converge. You might want to "
                "increase the number of iterations `numItermax` "
                "or the regularization parameter `reg`."
            )
    if log:
        log["niter"] = ii
        log["u"] = u
        log["v"] = v

    if n_hists:  # return only loss
        res = nx.einsum("ik,ij,jk,ij->k", u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix
        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))


def sinkhorn_log(
    a,
    b,
    M,
    reg,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
    **kwargs,
):
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
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

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

    # in case of multiple histograms
    if n_hists > 1 and warmstart is None:
        warmstart = [None] * n_hists

    if n_hists:  # we do not want to use tensors sor we do a loop
        lst_loss = []
        lst_u = []
        lst_v = []

        for k in range(n_hists):
            res = sinkhorn_log(
                a,
                b[:, k],
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warmstart=warmstart[k],
                **kwargs,
            )

            if log:
                lst_loss.append(nx.sum(M * res[0]))
                lst_u.append(res[1]["log_u"])
                lst_v.append(res[1]["log_v"])
            else:
                lst_loss.append(nx.sum(M * res))
        res = nx.stack(lst_loss)
        if log:
            log = {
                "log_u": nx.stack(lst_u, 1),
                "log_v": nx.stack(lst_v, 1),
            }
            log["u"] = nx.exp(log["log_u"])
            log["v"] = nx.exp(log["log_v"])
            return res, log
        else:
            return res

    else:
        if log:
            log = {"err": []}

        Mr = -M / reg

        # we assume that no distances are null except those of the diagonal of
        # distances
        if warmstart is None:
            u = nx.zeros(dim_a, type_as=M)
            v = nx.zeros(dim_b, type_as=M)
        else:
            u, v = warmstart

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
                    log["err"].append(err)

                if verbose:
                    if ii % 200 == 0:
                        print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                    print("{:5d}|{:8e}|".format(ii, err))
                if err < stopThr:
                    break
        else:
            if warn:
                warnings.warn(
                    "Sinkhorn did not converge. You might want to "
                    "increase the number of iterations `numItermax` "
                    "or the regularization parameter `reg`."
                )

        if log:
            log["niter"] = ii
            log["log_u"] = u
            log["log_v"] = v
            log["u"] = nx.exp(u)
            log["v"] = nx.exp(v)

            return nx.exp(get_logT(u, v)), log

        else:
            return nx.exp(get_logT(u, v))


def greenkhorn(
    a,
    b,
    M,
    reg,
    numItermax=10000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
):
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
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

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
        raise TypeError(
            "JAX or TF arrays have been received. Greenkhorn is not "
            "compatible with  neither JAX nor TF"
        )

    if len(a) == 0:
        a = nx.ones((M.shape[0],), type_as=M) / M.shape[0]
    if len(b) == 0:
        b = nx.ones((M.shape[1],), type_as=M) / M.shape[1]

    dim_a = a.shape[0]
    dim_b = b.shape[0]

    K = nx.exp(-M / reg)

    if warmstart is None:
        u = nx.full((dim_a,), 1.0 / dim_a, type_as=K)
        v = nx.full((dim_b,), 1.0 / dim_b, type_as=K)
    else:
        u, v = nx.exp(warmstart[0]), nx.exp(warmstart[1])
    G = u[:, None] * K * v[None, :]

    viol = nx.sum(G, axis=1) - a
    viol_2 = nx.sum(G, axis=0) - b
    stopThr_val = 1
    if log:
        log = dict()
        log["u"] = u
        log["v"] = v

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
            viol_2 += K[i_1, :].T * (new_u - old_u) * v
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
            warnings.warn(
                "Sinkhorn did not converge. You might want to "
                "increase the number of iterations `numItermax` "
                "or the regularization parameter `reg`."
            )

    if log:
        log["n_iter"] = ii
        log["u"] = u
        log["v"] = v

    if log:
        return G, log
    else:
        return G


def sinkhorn_stabilized(
    a,
    b,
    M,
    reg,
    numItermax=1000,
    tau=1e3,
    stopThr=1e-9,
    warmstart=None,
    verbose=False,
    print_period=20,
    log=False,
    warn=True,
    **kwargs,
):
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
        log = {"err": []}

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
        return nx.exp(-(M - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b))) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return nx.exp(
            -(M - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b))) / reg
            + nx.log(u.reshape((dim_a, 1)))
            + nx.log(v.reshape((1, dim_b)))
        )

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
                alpha, beta = (
                    alpha + reg * nx.max(nx.log(u), 1),
                    beta + reg * nx.max(nx.log(v)),
                )
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
                log["err"].append(err)

            if verbose:
                if ii % (print_period * 20) == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(ii, err))

        if err <= stopThr:
            break

        if nx.any(nx.isnan(u)) or nx.any(nx.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn("Numerical errors at iteration %d" % ii)
            u = uprev
            v = vprev
            break
    else:
        if warn:
            warnings.warn(
                "Sinkhorn did not converge. You might want to "
                "increase the number of iterations `numItermax` "
                "or the regularization parameter `reg`."
            )
    if log:
        if n_hists:
            alpha = alpha[:, None]
            beta = beta[:, None]
        logu = alpha / reg + nx.log(u)
        logv = beta / reg + nx.log(v)
        log["n_iter"] = ii
        log["logu"] = logu
        log["logv"] = logv
        log["alpha"] = alpha + reg * nx.log(u)
        log["beta"] = beta + reg * nx.log(v)
        log["warmstart"] = (log["alpha"], log["beta"])
        if n_hists:
            res = nx.stack(
                [
                    nx.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
                    for i in range(n_hists)
                ]
            )
            return res, log

        else:
            return get_Gamma(alpha, beta, u, v), log
    else:
        if n_hists:
            res = nx.stack(
                [
                    nx.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
                    for i in range(n_hists)
                ]
            )
            return res
        else:
            return get_Gamma(alpha, beta, u, v)


def sinkhorn_epsilon_scaling(
    a,
    b,
    M,
    reg,
    numItermax=100,
    epsilon0=1e4,
    numInnerItermax=100,
    tau=1e3,
    stopThr=1e-9,
    warmstart=None,
    verbose=False,
    print_period=10,
    log=False,
    warn=True,
    **kwargs,
):
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
        log = {"err": []}

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

        G, logi = sinkhorn_stabilized(
            a,
            b,
            M,
            regi,
            numItermax=numInnerItermax,
            stopThr=stopThr,
            warmstart=(alpha, beta),
            verbose=False,
            print_period=20,
            tau=tau,
            log=True,
        )

        alpha = logi["alpha"]
        beta = logi["beta"]

        if ii % (print_period) == 0:  # spsion nearly converged
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = G
            err = (
                nx.norm(nx.sum(transp, axis=0) - b) ** 2
                + nx.norm(nx.sum(transp, axis=1) - a) ** 2
            )
            if log:
                log["err"].append(err)

            if verbose:
                if ii % (print_period * 10) == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(ii, err))

        if err <= stopThr and ii > numItermin:
            break
    else:
        if warn:
            warnings.warn(
                "Sinkhorn did not converge. You might want to "
                "increase the number of iterations `numItermax` "
                "or the regularization parameter `reg`."
            )
    if log:
        log["alpha"] = alpha
        log["beta"] = beta
        log["warmstart"] = (log["alpha"], log["beta"])
        log["niter"] = ii
        return G, log
    else:
        return G
