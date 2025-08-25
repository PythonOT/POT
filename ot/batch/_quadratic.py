# -*- coding: utf-8 -*-
"""
Batch operations for quadratic optimal transport.
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Paul Krzakala <paul.krzakala@gmail.com>
#
# License: MIT License

from ..utils import OTResult
from ot.backend import get_backend
from ot.batch._linear import loss_linear_batch
from ot.batch._utils import grad_enabled, bmv, bop, bregman_log_batch
from abc import ABC, abstractmethod


def transpose(C, nx=None):
    if nx is None:
        nx = get_backend(C)
    return nx.transpose(C, (0, 2, 1)) if C.ndim == 3 else nx.transpose(C, (0, 2, 1, 3))


class QuadraticMetric(ABC):
    """
    Gromov-Wasserstein writes as:
        GW(T,C1,C2) = sum_ijkl T_ik T_jl l(C1_ij, C2_kl) = < LxT, T >
    Where L is a cost tensor L[i,j,k,l] = l(C1_ij, C2_kl).

    For loss function of form l(a,b) = f1(a) + f2(b) - < h1(a), h2(b) >
    The tensor product LxT can be computed fast using tensor_product [12].

    Typical use:
    L = metric.cost_tensor(C1, C2, T)
    loss = loss_quadratic_batch(L, T)

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """

    @abstractmethod
    def f1(self, C1, nx=None):
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def f2(self, C2, nx=None):
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def h1(self, C1, nx=None):
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def h2(self, C2, nx=None):
        raise NotImplementedError("Subclasses should implement this method")

    def cost_tensor(self, a, b, C1, C2, symmetric=True, nx=None):
        """
        "Compute" the cost tensor L for the quadratic OT problem.
        The full tensor is not stored, only the matrices required for fast computation of the gradient of < LxT, T> (which we denote \nabla_T L).
        Note that < LxT, T> = 0.5 < \nabla_T L, T >

        The most general formula is:
            \nabla_T L = (f1(C1) + f1(C1)^T) T 1 1^T
                        + 1 1^T T (f2(C2) + f2(C2)^T)^T
                        + h1(C1) T h2(C2)^T
                        + h1(C1)^T T h2(C2)

        If we assume T 1 = a and T^T 1 = b (i.e. the transport plan satisfies the marginal constraints), then some terms can be simplified:
            \nabla_T L = const
                        + h1(C1) T h2(C2)^T
                        + h1(C1)^T T h2(C2)
            where
            const = (f1(C1) + f1(C1)^T) a 1^T + 1 b^T (f2(C2) + f2(C2)^T)^T

        If we also assume that C1 and C2 are symmetric, then this further simplifies to:
            0.5 \nabla_T L = const - h1(C1) T h2(C2)^T
            where
            const = f1(C1) a 1^T + 1 b^T f2(C2)^T
        """
        if nx is None:
            nx = get_backend(C1, C2)

        fC1 = self.f1(C1, nx=nx)
        fC2 = self.f2(C2, nx=nx)
        if not symmetric:
            fC1 = 0.5 * (fC1 + transpose(fC1, nx=nx))
            fC2 = 0.5 * (fC2 + transpose(fC2, nx=nx))
        hC1 = self.h1(C1, nx=nx)
        hC2 = self.h2(C2, nx=nx)

        constC = compute_const_from_marginals(fC1, fC2, a, b, nx=nx)

        L = {"constC": constC, "hC1": hC1, "hC2": hC2, "fC1": fC1, "fC2": fC2}

        return L


class QuadraticEuclidean(QuadraticMetric):
    """
    Euclidean distance
        l(a,b) = ||a-b||^2

    Note: can be use for C1 and C2 of shape (B, N, N) or (B, N, N, d) where d is the dimension of the features.
    """

    def f1(self, C1, nx=None):
        if nx is None:
            nx = get_backend(C1)
        if C1.ndim == 4:
            return nx.sum(C1**2, axis=-1)
        else:
            return C1**2

    def f2(self, C2, nx=None):
        if nx is None:
            nx = get_backend(C2)
        if C2.ndim == 4:
            return nx.sum(C2**2, axis=-1)
        else:
            return C2**2

    def h1(self, C1, nx=None):
        if C1.ndim == 3:
            C1 = nx.unsqueeze(C1, -1)
        return 2 * C1

    def h2(self, C2, nx=None):
        if C2.ndim == 3:
            C2 = nx.unsqueeze(C2, -1)
        return C2


class QuadraticKL(QuadraticMetric):
    """
    KL divergence
        l(a,b) = sum_i a_i log(a_i/b_i)
    Expect x and y to be probability distributions
    If logits is True, x is expected to be logits (unnormalized log probabilities)
    l(x, y) = sum_i y_i * (log(y_i) - x_i)
    """

    def __init__(self, logits=False):
        self.logits = logits

    def f1(self, C1, nx=None):
        return nx.zeros(C1.shape, type_as=C1)

    def f2(self, C2, nx=None):
        assert C2.ndim == 3, "C2 must be a nxnxd tensor"
        if nx is None:
            nx = get_backend(C2)
        fC2 = C2 * nx.log(C2 + 1e-15)  # Avoid log(0)
        return fC2.sum(axis=-1)

    def h1(self, C1, nx=None):
        return C1 if self.logits else nx.log(C1 + 1e-15)

    def h2(self, C2, nx=None):
        return C2


def get_quadratic_metric(metric_name):
    if metric_name == "L2":
        return QuadraticEuclidean()
    elif metric_name == "KL":
        return QuadraticKL()
    elif isinstance(metric_name, QuadraticMetric):
        return metric_name
    else:
        raise ValueError(f"Unknown metric type: {metric_name}")


def detach_cost_tensor(L, nx=None):
    """
    Detach the cost tensor L to avoid gradients.
    """
    if nx is None:
        nx = get_backend(L["constC"], L["hC1"], L["hC2"])
    L_detached = {}
    for key, value in L.items():
        L_detached[key] = nx.detach(value)
    return L_detached


def compute_const_from_T(fC1, fC2, T, nx=None):
    """
    Compute the constant term f1(C1) a 1^T + 1 b^T f2(C2)^T
    where a and b are the marginals of T.
    """
    return compute_const_from_marginals(fC1, fC2, T.sum(axis=2), T.sum(axis=1), nx=nx)


def compute_const_from_marginals(fC1, fC2, a, b, nx=None):
    """
    Compute the constant term f1(C1) a 1^T + 1 b^T f2(C2)^T
    """
    if nx is None:
        nx = get_backend(fC1, fC2, a, b)
    fC1a = bmv(fC1, a, nx=nx)
    fC2b = bmv(fC2, b, nx=nx)
    constC = fC1a[:, :, None] + fC2b[:, None, :]
    return constC


def tensor_product(L, T, nx=None, recompute_const=False, symmetric=True):
    """
    Compute the tensor product LxT for the cost tensor L and transport plan T.
    The formula is:
        LxT = const - hC1 T hC2^T
        const = < fC1 a 1^T + 1 (fC2 b)^T

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """

    if nx is None:
        nx = get_backend(T)

    if recompute_const:
        const = compute_const_from_T(L["fC1"], L["fC2"], T, nx=nx)
    else:
        const = L["constC"]

    hC1 = L["hC1"]
    hC2 = L["hC2"]

    dot = nx.einsum("bijd,bjk->bikd", hC1, T)
    dot = nx.einsum("bikd,bjkd->bijd", dot, hC2)
    dot = dot.sum(axis=-1)

    if not symmetric:
        dot_t = nx.einsum("bijd,bjk->bikd", transpose(hC1), T)
        dot_t = nx.einsum("bikd,bjkd->bijd", dot_t, transpose(hC2))
        dot_t = dot_t.sum(axis=-1)
        dot = (dot + dot_t) / 2  # Average the two symmetric terms

    return const - dot


def loss_quadratic_batch(L, T, recompute_const=False, symmetric=True, nx=None):
    """
    Computes the quadratic cost < LxT, T > where L is the cost tensor
    and T is the transport plan.
    """
    LT = tensor_product(
        L, T, nx=nx, recompute_const=recompute_const, symmetric=symmetric
    )
    return (LT * T).sum((1, 2))


def solve_gromov_batch(
    C1,
    C2,
    a=None,
    b=None,
    loss="L2",
    symmetric=None,
    M=None,
    alpha=None,
    epsilon=1e-2,
    T_init=None,
    max_iter=50,
    tol=1e-5,
    max_iter_inner=50,
    tol_inner=1e-5,
    grad="detach",
):
    r"""Solves the quadratic optimal transport problem proximal gradient.

    .. math::
        \min_{\mathbf{T}\geq 0} \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ T \mathbf{1} &= \mathbf{a}

             T^T \mathbf{1} &= \mathbf{b}

             T &\geq 0

    If :math:`M` and :math:`\alpha` are given, solves the more general fused-gromov-wasserstein problem:

    .. math::
        \min_{\mathbf{T}\geq 0} (1-\alpha) \sum_{i,j} M_{i,j} \mathbf{T}_{i,j}  + \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ T \mathbf{1} &= \mathbf{a}

             T^T \mathbf{1} &= \mathbf{b}

             T &\geq 0

    Writing the objective as :math:`(1-\alpha) < M, T > + \alpha < L \otimes T, T >`, the solver uses proximal gradient descent i.e. each iteration is:

    .. math::
        T_k+1 = Minimize < M_k, T > + \epsilon KL(T || T_k) where M_k = (1 - \alpha)  M + 2 * \alpha L \otimes T_k

    i.e.

    .. math::
        T_k+1 = Minimize < M_k - \epsilon * \log(T_k), T > - \epsilon H(T)

    Parameters
    ----------
    C1 : array-like, shape (B, n, n, d) or (B, n, n)
        Samples affinity matrices from source distribution
    C2 : array-like, shape (B, n, n, d) or (B, n, n)
        Samples affinity matrices from target distribution
    a : array-like, shape (B, n), optional
        Marginal distribution of the source samples. If None, uniform distribution is used.
    b : array-like, shape (B, m), optional
        Marginal distribution of the target samples. If None, uniform distribution is used.
    loss : str, optional
        Type of loss function, can be 'L2' or 'KL' or a QuadraticMetric instance.
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    M : array-like, shape (dim_a, dim_b), optional
        Linear cost matrix for Fused Gromov-Wasserstein (default is None).
    alpha : float, optional
        Weight the quadratic term (alpha*Gromov) and the linear term
        ((1-alpha)*Wass) in the Fused Gromov-Wasserstein problem. Not used for
        Gromov problem (when M is not provided). By default ``alpha=None``
        corresponds to ``alpha=1`` for Gromov problem (``M==None``) and
        ``alpha=0.5`` for Fused Gromov-Wasserstein problem (``M!=None``)
    epsilon : float, optional
        Regularization parameter for proximal gradient descent. Default is 1e-2.
    T_init : array-like, shape (B, n, m), optional
        Initial transport plan. If None, it is initialized to uniform distribution.
    max_iter : int, optional
        Maximum number of iterations for the proximal gradient descent. Default is 50.
    tol : float, optional
        Tolerance for convergence of the proximal gradient descent. Default is 1e-5.
    max_iter_inner : int, optional
        Maximum number of iterations for the inner Bregman projection. Default is 50.
    tol_inner : float, optional
        Tolerance for convergence of the inner Bregman projection. Default is 1e-5.
    grad : str, optional
        Type of gradient computation, either or 'autodiff', 'envelope' or 'last_step' used only for
        Sinkhorn solver. By default 'autodiff' provides gradients wrt all
        outputs (`plan, value, value_linear`) but with important memory cost.
        'envelope' provides gradients only for `value` and and other outputs are
        detached. This is useful for memory saving when only the value is needed. 'last_step' provides
        gradients only for the last iteration of the Sinkhorn solver, but provides gradient for both the OT plan and the objective values.
        'detach' does not compute the gradients for the Sinkhorn solver.
    assume_inner_convergence : bool, optional
        If True, assumes that the inner Bregman projection always converged i.e. the transport plan satisfies the marginal constraints.
        This enables faster computations of the tensor product but might results in inaccurate results (e.g. negative values of the loss).
        Default is True.
    Returns
    -------
    res : OTResult()
        Result of the optimization problem. The information can be obtained as follows:

        - res.plan : OT plan :math:`\mathbf{T}`
        - res.potentials : OT dual potentials
        - res.value : Optimal value of the optimization problem
        - res.value_linear : Linear OT loss with the optimal OT plan
        - res.value_quad : Quadratic OT loss with the optimal OT plan

        See :any:`OTResult` for more information.

    Examples
    --------
    >>> from ot.batch import quadratic_solver_batch
    >>> import numpy as np
    >>> a = np.ones((B, ns)) / ns
    >>> b = np.ones((B, nt)) / nt
    >>> C1 = np.random.rand(B, ns, ns)
    >>> C2 = np.random.rand(B, nt, nt)
    >>> M = np.random.rand(B, ns, nt)
    >>> res = quadratic_solver_batch(C1=C1, C2=C2, a=a, b=b, M=M, alpha=0.5, epsilon=0.01)
    >>> res.plan.shape
    (B, ns, nt)
    >>> res.value.shape
    (B,)

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    -- "Scalable Gromov-Wasserstein learning for graph partitioning and matching." Advances in neural information processing systems, 2019.
    """

    # -------------- Setup -------------- #

    nx = get_backend(a, b, M, C1, C2, T_init)
    B, n, m = (C1.shape[0], C1.shape[1], C2.shape[1])

    if a is None:
        a = nx.ones((B, n), type_as=C1) / n
    if b is None:
        b = nx.ones((B, m), type_as=C2) / m

    if symmetric is None:
        symmetric = nx.allclose(C1, transpose(C1, nx=nx), atol=1e-10) and nx.allclose(
            C2, transpose(C2, nx=nx), atol=1e-10
        )

    # -------------- Get cost_tensor (quadratic part) -------------- #

    # Case 1: C1, C2, metric_quadratic are provided -> compute cost_tensor
    # Case 2: Error

    if C1 is not None and C2 is not None and loss is not None:
        metric_quadratic = (
            loss if isinstance(loss, QuadraticMetric) else get_quadratic_metric(loss)
        )
        L = metric_quadratic.cost_tensor(a, b, C1, C2, symmetric=symmetric, nx=nx)
    else:
        raise ValueError("C1, C2, and loss must be provided")

    # -------------- Get cost_matrix (linear part) -------------- #

    if M is None and alpha is None:
        M = nx.zeros((B, n, m), type_as=C1)
        alpha = 1.0  # Gromov problem
    elif M is not None and alpha is None:
        raise ValueError(
            "If M is provided, alpha must also be provided for Fused Gromov-Wasserstein problem"
        )
    elif M is None and alpha is not None:
        raise ValueError(
            "If alpha is provided, M must also be provided for Fused Gromov-Wasserstein problem"
        )
    elif M is not None and alpha is not None:
        if M.shape[0] != B or M.shape[1] != n or M.shape[2] != m:
            raise ValueError(
                f"Shape of M {M.shape} does not match the batch size {B} and dimensions {n}, {m}"
            )
        if alpha < 0 or alpha > 1:
            raise ValueError(
                "Alpha must be in [0, 1] for Fused Gromov-Wasserstein problem"
            )

    # -------------- Solver -------------- #

    if T_init is None:
        const = nx.sqrt(nx.sum(a, axis=1) * nx.sum(b, axis=1))
        T_init = bop(a, b, nx=nx) / const[:, None, None]  # B x n x m

    grad_inner = "autodiff" if grad == "autodiff" else "detach"

    with grad_enabled(nx, grad == "autodiff"):
        T = T_init
        log_T = nx.log(T + 1e-15)  # Avoid log(0)
        for iter in range(max_iter):
            T_prev = T
            Mk = (1 - alpha) * M + 2 * alpha * tensor_product(
                L, T, nx=nx, recompute_const=False, symmetric=symmetric
            )
            K = -Mk / epsilon + log_T
            out = bregman_log_batch(
                K, a, b, nx=nx, max_iter=max_iter_inner, tol=tol_inner, grad=grad_inner
            )
            T = out["T"]
            log_T = out["log_T"]
            max_err = nx.max(nx.mean(nx.abs(T_prev - T), axis=(1, 2)))
            if max_err < tol:
                break

    if grad is None or grad == "detach":
        T = nx.detach(T)
        M = nx.detach(M)
        L = detach_cost_tensor(L, nx=nx)
    elif grad == "envelope":
        T = nx.detach(T)
    else:
        raise ValueError(f"Unknown gradient mode: {grad}")

    value_linear = loss_linear_batch(M, T, nx=nx)
    value_quadratic = loss_quadratic_batch(
        L, T, nx=nx, recompute_const=True, symmetric=symmetric
    )  # Always recompute const for accurate value
    value = (1 - alpha) * value_linear + alpha * value_quadratic
    log = {"n_iter": iter}

    res = OTResult(
        value=value,
        value_linear=value_linear
        * (1 - alpha),  # Weight the linear value for consistency with ot.solve_gromov
        value_quad=value_quadratic * alpha,  # idem
        plan=T,
        backend=nx,
        log=log,
    )

    return res
