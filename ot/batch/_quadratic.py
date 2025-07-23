from ot.backend import get_backend
from ot.batch._linear import LinearMetric, get_linear_metric, cost_linear
from ot.batch._utils import grad_enabled, bmv, bop, bregman_log_batch


class QuadraticMetric:
    """
    Gromov-Wasserstein writes as:
        GW(T,C1,C2) = sum_ijkl T_ik T_jl l(C1_ij, C2_kl) = < LxT, T >
    Where L is a cost tensor L[i,j,k,l] = l(C1_ij, C2_kl).

    For loss function of form l(a,b) = f1(a) + f2(b) - < h1(a), h2(b) >
    The tensor product LxT can be computed fast using tensor_product [12].

    Typical use:
    L = metric.cost_tensor(C1, C2, T)
    cost = cost_quadratic(L, T)

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """

    def f1(self, C1, nx=None):
        raise NotImplementedError("Subclasses should implement this method")

    def f2(self, C2, nx=None):
        raise NotImplementedError("Subclasses should implement this method")

    def h1(self, C1, nx=None):
        raise NotImplementedError("Subclasses should implement this method")

    def h2(self, C2, nx=None):
        raise NotImplementedError("Subclasses should implement this method")

    def cost_tensor(self, a, b, C1, C2, nx=None):
        if nx is None:
            nx = get_backend(C1, C2)

        B, N1, N2 = C1.shape[0], C1.shape[1], C2.shape[1]

        fC1 = self.f1(C1, nx=nx)
        fC2 = self.f2(C2, nx=nx)
        hC1 = self.h1(C1, nx=nx)
        hC2 = self.h2(C2, nx=nx)

        fC1a = bmv(fC1, a, nx=nx)  # nx.einsum('bij,bj->bi', fC1, a)
        constC1 = bop(
            fC1a, nx.ones((B, N2), type_as=b), nx=nx
        )  # nx.einsum('bi,bj->bij', fC1a, nx.ones((B, N2), type_as=b))

        fC2b = bmv(fC2, b, nx=nx)
        constC2 = bop(nx.ones((B, N1), type_as=a), fC2b, nx=nx)

        constC = constC1 + constC2

        L = {"constC": constC, "hC1": hC1, "hC2": hC2}

        return L


def detach_cost_tensor(L, nx=None):
    """
    Detach the cost tensor L to avoid gradients.
    """
    if nx is None:
        nx = get_backend(L["constC"], L["hC1"], L["hC2"])

    constC = nx.detach(L["constC"])
    hC1 = nx.detach(L["hC1"])
    hC2 = nx.detach(L["hC2"])

    return {"constC": constC, "hC1": hC1, "hC2": hC2}


def tensor_product(L, T, nx=None):
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

    constC = L["constC"]
    hC1 = L["hC1"]
    hC2 = L["hC2"]

    dot = nx.einsum("bijd,bjk->bikd", hC1, T)
    dot = nx.einsum("bikd,bjkd->bijd", dot, hC2)
    dot = dot.sum(
        axis=-1
    )  # Handle the case when C1 and C2 are 3D tensors i.e. there are "edge features"

    return constC - dot


def cost_quadratic(L, T, nx=None):
    """
    Computes the quadratic cost < LxT, T > where L is the cost tensor
    and T is the transport plan.
    """
    LT = tensor_product(L, T, nx=nx)
    return (LT * T).sum((1, 2))


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
            C1 = C1.unsqueeze(-1)
        return 2 * C1

    def h2(self, C2, nx=None):
        if C2.ndim == 3:
            C2 = C2.unsqueeze(-1)
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
    if metric_name == "euclidean":
        return QuadraticEuclidean()
    elif isinstance(metric_name, QuadraticMetric):
        return metric_name
    else:
        raise ValueError(f"Unknown metric type: {metric_name}")


def quadratic_solver_batch(
    epsilon,  # B or float
    C1,  # Bxnxd or None
    C2,  # Bxmxd or None
    a=None,  # Bxn or None
    b=None,  # Bxm or None
    X=None,  # Bxnxd or None
    Y=None,  # Bxmxd or None
    metric_linear=None,  # p, kl LinearMetric
    M=None,  # Bxn or None
    metric_quadratic=None,  # euclidean or QuadraticMetrice
    alpha=0.5,  # float
    T_init=None,  # Bxnxm or None
    grad=None,  #'autodiff', 'envelope', 'last_step' or 'detach'
    max_iter=None,  # int or None
    tol=None,  # float or None
    max_iter_inner=None,  # int or None
    tol_inner=None,  # float or None
):  # bool
    """
    Minimize (1-alpha) * < M, T > + alpha * < L x T, T >
    The solver uses proximal gradient descent i.e. each iteration is:
        T_k+1 = Minimize < M_k, T > + epsilon KL(T || T_k) where M_k = (1 - alpha) *  M + 2 * alpha * L x T_k
        i.e.
        T_k+1 = Minimize < M_k - epsilon * log(T_k), T > - epsilon H(T)

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    -- "Scalable Gromov-Wasserstein learning for graph partitioning and matching." Advances in neural information processing systems, 2019.
    """

    # -------------- Get backend -------------- #

    nx = get_backend(a, b, M, C1, C2, T_init)
    B, n, m = (C1.shape[0], C1.shape[1], C2.shape[1])

    if a is None:
        a = nx.ones((B, n), type_as=C1) / n
    if b is None:
        b = nx.ones((B, m), type_as=C2) / m

    # -------------- Get cost_tensor (quadratic part) -------------- #

    # Case 1: C1, C2, metric_quadratic are provided -> compute cost_tensor
    # Case 2: Error

    if C1 is not None and C2 is not None and metric_quadratic is not None:
        metric_quadratic = (
            metric_quadratic
            if isinstance(metric_quadratic, QuadraticMetric)
            else get_quadratic_metric(metric_quadratic)
        )
        L = metric_quadratic.cost_tensor(a, b, C1, C2)
    else:
        raise ValueError("C1, C2, and metric_quadratic must be provided")

    # -------------- Get cost_matrix (linear part) -------------- #

    # Case 1: cost_matrix is provided -> use it directly
    # Case 2: X, T, metric_linear are all set to None -> set cost_matrix to 0
    # Case 3: X, Y, metric_linear are provided -> use them to compute cost_matrix
    # Case 4: Error

    if M is not None:
        pass
    elif X is None and Y is None and metric_linear is None:
        M = nx.zeros((B, n, m), type_as=C1)
    elif X is not None and Y is not None and metric_linear is not None:
        metric_linear = (
            metric_linear
            if isinstance(metric_linear, LinearMetric)
            else get_linear_metric(metric_linear)
        )
        nx = get_backend(a, b, X, Y, T_init)
        M = metric_linear.cost_matrix(X, Y, nx)
    else:
        raise ValueError(
            "If cost matrix is not provided, samples X, Y, and metric_linear must be provided (or all set to None)."
        )

    # -------------- Solver -------------- #

    if T_init is None:
        cst = nx.sqrt(nx.sum(a, axis=1) * nx.sum(b, axis=1))
        T_init = bop(a, b, nx=nx) / cst[:, None, None]  # B x n x m

    grad_inner = "autodiff" if grad == "autodiff" else "detach"

    with grad_enabled(nx, grad == "autodiff"):
        T = T_init
        log_T = nx.log(T + 1e-15)  # Avoid log(0)
        for _ in range(max_iter):
            T_prev = T
            Mk = (1 - alpha) * M + 2 * alpha * tensor_product(L, T, nx=nx)
            K = -Mk / epsilon + log_T
            out = bregman_log_batch(
                K, a, b, nx=nx, max_iter=max_iter_inner, tol=tol_inner, grad=grad_inner
            )
            T = out["T"]
            log_T = out["log_T"]
            max_err = nx.max(nx.abs(T_prev - T))
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

    cost_M = cost_linear(M, T, nx=nx)
    cost_L = cost_quadratic(L, T, nx=nx)
    cost_total = (1 - alpha) * cost_M + alpha * cost_L

    out = {"T": T, "cost_linear": cost_M, "cost_quadratic": cost_L, "cost": cost_total}

    return out
