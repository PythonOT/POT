# -*- coding: utf-8 -*-
"""
Batch operations for linear optimal transport.
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Paul Krzakala <paul.krzakala@gmail.com>
#
# License: MIT License

from ..backend import get_backend
from ..utils import OTResult
from ._utils import (
    bregman_log_projection_batch,
    bregman_projection_batch,
    entropy_batch,
)


def dist_lp_batch(X, Y, p=2, q=1, nx=None):
    r"""Computes the cost matrix for a batch of samples using the Lp norm.

    .. math::
        M_{bij} = ( \sum_{d} (x_{bid} - y_{bjd})^p )^{q/p} = ||x_{bi} - y_{bj}||_p^q

    Parameters
    ----------
    X : array-like, shape (B, ns, d)
        Samples from source distribution
    Y : array-like, shape (B, nt, d)
        Samples from target distribution
    p : float, optional
        The order of the norm to use. Default is 2 (Euclidean distance).
    q : float, optional
        If None, use the Lp norm. If specified, it computes the Lp norm raised to the power of q.
    nx : backend, optional
        Backend to use for computations. If None, it will be inferred from the inputs.

    Returns
    -------
    M : array-like, shape (B, ns, nt)
        Cost matrix where M[bij] is the cost between sample i in batch b and sample j in batch b.
    """

    if nx is None:
        nx = get_backend(X, Y)
    M = nx.abs(X[:, :, None] - Y[:, None, :]) ** p
    M = M.sum(axis=-1)
    if q != p:
        M = M ** (q / p)
    return M


def dist_euclidean_batch(X, Y, squared=True, nx=None):
    r"""Computes the squared Euclidean cost matrix for a batch of samples.

    .. math::
        M_{bij} = \sum_{d} (x_{bid} - y_{bjd})^2 = ||x_{bi} - y_{bj}||_2^2

    Parameters
    ----------
    X : array-like, shape (B, ns, d)
        Samples from source distribution
    Y : array-like, shape (B, nt, d)
        Samples from target distribution
    squared : bool, optional
        If True, returns the squared Euclidean distance. Default is True.
    nx : backend, optional
        Backend to use for computations. If None, it will be inferred from the inputs.

    Returns
    -------
    M : array-like, shape (B, ns, nt)
        Cost matrix where M[bij] is the cost between sample i in batch b and sample j in batch b.
    """

    if nx is None:
        nx = get_backend(X, Y)
    XX = nx.sum(X**2, axis=-1, keepdims=True)
    YY = nx.sum(Y**2, axis=-1, keepdims=True)
    YY = nx.transpose(YY, axes=(0, 2, 1))
    M = XX + YY - 2 * nx.einsum("bid,bjd->bij", X, Y)
    if not squared:
        M = nx.sqrt(M)
    return M


def dist_kl_batch(X, Y, logits_X=False, nx=None, eps=1e-10):
    r"""Computes the KL divergence cost matrix for a batch of samples.

    .. math::
        M_{bij} = \sum_{d} y_{bjd} * log(y_{bjd}/X_{bid}) = KL(y_{bj} || x_{bi})

    Parameters
    ----------
    X : array-like, shape (B, ns, d)
        Samples from source distribution
    Y : array-like, shape (B, nt, d)
        Samples from target distribution
    logits_X : bool, optional
        If True, X is assumed to be in log space (logits). Default is False.
    nx : backend, optional
        Backend to use for computations. If None, it will be inferred from the inputs.

    Returns
    -------
    M : array-like, shape (B, ns, nt)
        Cost matrix where M[bij] is the cost between sample i in batch b and sample j in batch b.
    """

    if nx is None:
        nx = get_backend(X, Y)
    entr_y = nx.sum(Y * nx.log(Y + eps), axis=-1)  # B x m
    if logits_X:
        M = entr_y[:, None, :] - Y[:, None, :] * X[:, :, None]
    else:
        M = entr_y[:, None, :] - nx.sum(
            Y[:, None, :] * nx.log(X + eps)[:, :, None], axis=-1
        )
    return M


def loss_linear_batch(M, T, nx=None):
    r"""Computes the linear optimal transport loss given a batch cost matrices and transport plans.

    .. math::

        L(T, M)_b =  \langle T_b, M_b \rangle_F

    Parameters
    ----------
    M : array-like, shape (B, ns, nt)
        Cost matrix
    T : array-like, shape (B, ns, nt)
        Transport plan
    Returns
    -------
    loss : array-like, shape (B,)
        Loss value for each batch element
    See Also
    --------
    ot.batch.dist_batch : batched cost matrix computation for computing M.
    ot.batch.solve_batch : solver for computing the optimal T.
    """
    if nx is None:
        nx = get_backend(M, T)
    return nx.sum(M * T, axis=(1, 2))


def loss_linear_samples_batch(X, Y, T, metric="l2"):
    r"""Computes the linear optimal transport loss given samples and transport plan. This is the equivalent of
    calling `dist_batch` and then `loss_linear_batch`.

    Parameters
    ----------
    X : array-like, shape (B, ns, d)
        Samples from source distribution
    Y : array-like, shape (B, nt, d)
        Samples from target distribution
    T : array-like, shape (B, ns, nt)
        Transport plan
    metric : str, optional
            'sqeuclidean', 'euclidean', 'minkowski' or 'kl'
    Returns
    -------
    loss : array-like, shape (B,)
        Loss value for each batch element

    See Also
    --------
    ot.batch.dist_batch : batched cost matrix computation for computing M.
    ot.batch.solve_batch : solver for computing the optimal T.
    """
    M = dist_batch(X, Y, metric=metric)
    return loss_linear_batch(M, T)


def dist_batch(
    X1,
    X2=None,
    metric="sqeuclidean",
    p=2,
    nx=None,
):
    r"""Batched version of ot.dist, use it to compute many distance matrices in parallel.

    Parameters
    ----------

    X1 : array-like, shape (b,n1,d)
        `b` matrices with `n1` samples of size `d`
    X2 : array-like, shape (b,n2,d), optional
        `b` matrices with `n2` samples of size `d` (if None then :math:`\mathbf{X_2} = \mathbf{X_1}`)
    metric : str, optional
        'sqeuclidean', 'euclidean', 'minkowski' or 'kl'
    p : float, optional
        p-norm for the Minkowski metrics. Default value is 2.
    nx : Backend, optional
        Backend to perform computations on. If omitted, the backend defaults to that of `x1`.

    Returns
    -------

    M : array-like, shape (`b`, `n1`, `n2`)
        distance matrix computed with given metric

    Examples
    --------
    >>> import numpy as np
    >>> from ot.batch import dist_batch
    >>> X1 = np.random.randn(5, 10, 3)
    >>> X2 = np.random.randn(5, 15, 3)
    >>> M = dist_batch(X1, X2, metric="euclidean")
    >>> M.shape
    (5, 10, 15)

    See Also
    --------
    ot.dist : equivalent non-batched function.
    """
    X2 = X2 if X2 is not None else X1
    metric = metric.lower()
    if metric == "sqeuclidean":
        M = dist_euclidean_batch(X1, X2, squared=True, nx=nx)
    elif metric == "euclidean":
        M = dist_euclidean_batch(X1, X2, squared=False, nx=nx)
    elif metric == "minkowski":
        M = dist_lp_batch(X1, X2, p=p, q=1, nx=nx)
    elif metric == "kl":
        M = dist_kl_batch(X1, X2, logits_X=False, nx=nx)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return M


def solve_batch(
    M,
    reg,
    a=None,
    b=None,
    max_iter=1000,
    tol=1e-5,
    solver="log_sinkhorn",
    reg_type="entropy",
    grad="envelope",
):
    r"""Batched version of ot.solve, use it to solve many entropic OT problems in parallel.

    Parameters
    ----------
    M : array-like, shape (B, ns, nt)
        Cost matrix
    reg : float
        Regularization parameter for entropic regularization
    a : array-like, shape (B, ns)
        Source distribution (optional). If None, uniform distribution is used.
    b : array-like, shape (B, nt)
        Target distribution (optional). If None, uniform distribution is used.
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence
    solver: str
        Solver to use, either 'log_sinkhorn' or 'sinkhorn'. Default is "log_sinkhorn" which is more stable.
    reg_type : str, optional
        Type of regularization :math:`R`  either "KL", or "entropy". Default is "entropy".
    grad : str, optional
        Type of gradient computation, either or 'autodiff', 'envelope' or 'last_step' used only for
        Sinkhorn solver. By default 'autodiff' provides gradients wrt all
        outputs (`plan, value, value_linear`) but with important memory cost.
        'envelope' provides gradients only for `value` and and other outputs are
        detached. This is useful for memory saving when only the value is needed. 'last_step' provides
        gradients only for the last iteration of the Sinkhorn solver, but provides gradient for both the OT plan and the objective values.
        'detach' does not compute the gradients for the Sinkhorn solver.

    Returns
    -------
    res : OTResult()
        Result of the optimization problem. The information can be obtained as follows:

        - res.plan : OT plan :math:`\mathbf{T}`
        - res.potentials : OT dual potentials
        - res.value : Optimal value of the optimization problem
        - res.value_linear : Linear OT loss with the optimal OT plan

        See :any:`OTResult` for more information.

    Examples
    --------
    >>> import numpy as np
    >>> from ot.batch import solve_batch, dist_batch
    >>> X = np.random.randn(5, 10, 3)  # 5 batches of 10 samples in 3D
    >>> Y = np.random.randn(5, 15, 3)  # 5 batches of 15 samples in 3D
    >>> M = dist_batch(X, Y, metric="euclidean")  # Compute cost matrices
    >>> reg = 0.1
    >>> result = solve_batch(M, reg)
    >>> result.plan.shape  # Optimal transport plans for each batch
    (5, 10, 15)
    >>> result.value.shape  # Optimal transport values for each batch
    (5,)

    See Also
    --------
    ot.batch.dist_batch : batched cost matrix computation for computing M.
    ot.solve : non-batched version of the OT solver.
    """

    nx = get_backend(a, b, M)

    B, n, m = M.shape

    if a is None:
        a = nx.ones((B, n)) / n
    if b is None:
        b = nx.ones((B, m)) / m

    if solver == "log_sinkhorn":
        K = -M / reg
        out = bregman_log_projection_batch(
            K, a, b, nx=nx, max_iter=max_iter, tol=tol, grad=grad
        )
    elif solver == "sinkhorn":
        K = nx.exp(-M / reg)
        out = bregman_projection_batch(
            K, a, b, nx=nx, max_iter=max_iter, tol=tol, grad=grad
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    T = out["T"]

    if grad is None or grad == "detach":
        T = nx.detach(T)
        M = nx.detach(M)
    elif grad == "envelope":
        T = nx.detach(T)

    value_linear = loss_linear_batch(M, T)
    if reg_type.lower() == "entropy":
        entr = -entropy_batch(T, nx=nx)
        value = value_linear + reg * entr
    elif reg_type.lower() == "kl":
        ref = nx.einsum("bi,bj->bij", a, b)
        kl = nx.sum(T * nx.log(T / ref + 1e-16), axis=(1, 2))
        value = value_linear + reg * kl
    log = {"n_iter": out["n_iters"]}

    res = OTResult(
        value=value,
        value_linear=value_linear,
        potentials=out["potentials"],
        plan=T,
        backend=nx,
        log=log,
    )

    return res


def solve_sample_batch(
    X_a,
    X_b,
    reg,
    a=None,
    b=None,
    metric="sqeuclidean",
    p=2,
    max_iter=1000,
    tol=1e-5,
    solver="log_sinkhorn",
    reg_type="entropy",
    grad="envelope",
):
    r"""Batched version of ot.solve, use it to solve many entropic OT problems in parallel.

    Parameters
    ----------
    M : array-like, shape (B, ns, nt)
        Cost matrix
    reg : float
        Regularization parameter for entropic regularization
    metric : str, optional
        'sqeuclidean', 'euclidean', 'minkowski' or 'kl'
    p : float, optional
        p-norm for the Minkowski metrics. Default value is 2.
    a : array-like, shape (B, ns)
        Source distribution (optional). If None, uniform distribution is used.
    b : array-like, shape (B, nt)
        Target distribution (optional). If None, uniform distribution is used.
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence
    solver: str
        Solver to use, either 'log_sinkhorn' or 'sinkhorn'. Default is "log_sinkhorn" which is more stable.
    reg_type : str, optional
        Type of regularization :math:`R`  either "KL", or "entropy". Default is "entropy".
    grad : str, optional
        Type of gradient computation, either or 'autodiff', 'envelope' or 'last_step' used only for
        Sinkhorn solver. By default 'autodiff' provides gradients wrt all
        outputs (`plan, value, value_linear`) but with important memory cost.
        'envelope' provides gradients only for `value` and and other outputs are
        detached. This is useful for memory saving when only the value is needed. 'last_step' provides
        gradients only for the last iteration of the Sinkhorn solver, but provides gradient for both the OT plan and the objective values.
        'detach' does not compute the gradients for the Sinkhorn solver.

    Returns
    -------
    res : OTResult()
        Result of the optimization problem. The information can be obtained as follows:

        - res.plan : OT plan :math:`\mathbf{T}`
        - res.potentials : OT dual potentials
        - res.value : Optimal value of the optimization problem
        - res.value_linear : Linear OT loss with the optimal OT plan

        See :any:`OTResult` for more information.

    See Also
    --------
    ot.batch.solve_batch : solver for computing the optimal T from arbitrary cost matrix M.
    """

    M = dist_batch(X_a, X_b, metric=metric, p=p)
    return solve_batch(
        M,
        reg,
        a=a,
        b=b,
        max_iter=max_iter,
        tol=tol,
        solver=solver,
        reg_type=reg_type,
        grad=grad,
    )
