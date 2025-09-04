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
from ._utils import bregman_log_batch, bregman_batch


def cost_matrix_lp_batch(X, Y, p=2, q=1, nx=None):
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


def cost_matrix_l2_batch(X, Y, squared=True, nx=None):
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


def cost_matrix_kl_batch(X, Y, logits_X=False, nx=None):
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
    entr_y = nx.sum(Y * nx.log(Y + 1e-10), axis=-1)  # B x m
    if logits_X:
        M = entr_y[:, None, :] - Y[:, None, :] + nx.log(X + 1e-10)[:, :, None]
    else:
        M = entr_y[:, None, :] - Y[:, None, :] * X[:, :, None]
    return M


def loss_linear_batch(M, T, nx=None):
    r"""Computes the linear optimal transport loss given cost matrix and transport plan.

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
    """
    return (M * T).sum((1, 2))


def loss_linear_samples_batch(X, Y, T, metric="l2"):
    r"""Computes the linear optimal transport loss given samples and transport plan.

    Parameters
    ----------
    X : array-like, shape (B, ns, d)
        Samples from source distribution
    Y : array-like, shape (B, nt, d)
        Samples from target distribution
    T : array-like, shape (B, ns, nt)
        Transport plan
    metric : str, optional
        Metric to use for computing the cost matrix. Can be a string in ['l2','kl']
        or a float for the Lp norm,
        or a callable function that takes X and Y as inputs and returns the cost matrix.

    Returns
    -------
    loss : array-like, shape (B,)
        Loss value for each batch element
    """

    if callable(metric):
        M = metric(X, Y)
    elif metric == "l2":
        M = cost_matrix_l2_batch(X, Y)
    elif metric == "kl":
        M = cost_matrix_kl_batch(X, Y)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return loss_linear_batch(M, T)


def entropy_batch(T, nx=None):
    """Computes the entropy of the transport plan T."""
    if nx is None:
        nx = get_backend(T)
    return -nx.sum(T * nx.log(T + 1e-10), axis=(1, 2))


def solve_batch(
    M,
    a=None,
    b=None,
    epsilon=1e-3,
    max_iter=1000,
    tol=1e-5,
    log_dual=True,
    grad="detach",
):
    r"""Solves a batch of linear optimal transport problems using Bregman projections.

    .. math::
        \mathop{\min}_T \quad \langle T, \mathbf{M} \rangle_F +
        \mathrm{\epsilon}\cdot\Omega(T)

        s.t. \ T \mathbf{1} &= \mathbf{a}

             T^T \mathbf{1} &= \mathbf{b}

             T &\geq 0

    Parameters
    ----------
    M : array-like, shape (B, ns, nt)
        Cost matrix
    epsilon : float
        Regularization parameter
    a : array-like, shape (B, ns)
        Source distribution (optional). If None, uniform distribution is used.
    b : array-like, shape (B, nt)
        Target distribution (optional). If None, uniform distribution is used.
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence
    log_dual : bool
        If True, performs Bregman projection in log space. This is more stable.
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
    """

    nx = get_backend(a, b, M)

    if log_dual:
        K = -M / epsilon
        out = bregman_log_batch(K, a, b, nx=nx, max_iter=max_iter, tol=tol, grad=grad)
    else:
        K = nx.exp(-M / epsilon)
        out = bregman_batch(K, a, b, nx=nx, max_iter=max_iter, tol=tol, grad=grad)

    T = out["T"]

    if grad is None or grad == "detach":
        T = nx.detach(T)
        M = nx.detach(M)
    elif grad == "envelope":
        T = nx.detach(T)

    entr = entropy_batch(T, nx=nx)
    value_linear = loss_linear_batch(M, T)
    value = value_linear + epsilon * entr
    log = {"n_iter": out["n_iters"]}

    res = OTResult(
        value=value,
        value_linear=value_linear,
        plan=T,
        backend=nx,
        log=log,
    )

    return res
