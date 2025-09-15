from ot.backend import get_backend


def entropy_batch(T, nx=None, eps=1e-16):
    r"""Computes the entropy of a batch of transport plans T.

    .. math::
        H(T)_b = - \sum_{i,j} T_{b,i,j} \log(T_{b,i,j})

    Parameters
    ----------

    T : array-like, shape (b,n1,n2)
        `b` transport of size `n1` x `n2`.
    eps : float, optional
        Small constant to avoid numerical issues with log(0). Default is 1e-16.
    nx : Backend, optional
        Backend to perform computations on. If omitted, the backend defaults to that of `T`.
    """
    if nx is None:
        nx = get_backend(T)
    return -nx.sum(T * nx.log(T + eps), axis=(1, 2))


def norm_batch(u, p=2, nx=None):
    """Computes the lp norm of a batch of vectors u."""
    if nx is None:
        nx = get_backend(u)
    return nx.sum(u**p, axis=1) ** (1 / p)


def bmv(A, b, nx):
    """
    Batched matrix-vector multiplication for tensor A and vector b.
    """
    return nx.einsum("bij,bj->bi", A, b)


def bop(a, b, nx):
    """
    Batched outer product for vectors a and b.
    """
    return nx.einsum("bi,bj->bij", a, b)


def bregman_projection_batch(
    K, a=None, b=None, nx=None, max_iter=10000, tol=1e-5, grad="detach"
):
    r"""
    Apply Bregman projection to a batch of affinity matrices :math:`\mathbf{K}`.

    The function solves the following optimization problem:

    .. math::
        \begin{aligned}
            \mathbf{T} = \mathop{\arg \min}_\mathbf{T} \quad & \text{KL}(\mathbf{T} \| \mathbf{K}) \\
            \text{s.t.} \quad & \mathbf{T} \mathbf{1} = \mathbf{a} \\
            & \mathbf{T}^T \mathbf{1} = \mathbf{b} \\
            & \mathbf{T} \geq 0
        \end{aligned}

    This is equivalent to:

    .. math::
        \begin{aligned}
            \mathbf{T} = \mathop{\arg \max}_\mathbf{T} \quad & \langle \mathbf{T},  \log(\mathbf{K}) \rangle_F \\
            \text{s.t.} \quad & \mathbf{T} \mathbf{1} = \mathbf{a} \\
            & \mathbf{T}^T \mathbf{1} = \mathbf{b} \\
            & \mathbf{T} \geq 0
        \end{aligned}

    The optimal solution has the form :math:`\mathbf{T} = \text{diag}(\mathbf{f}) \mathbf{K} \text{diag}(\mathbf{g})`,
    where the dual variables :math:`\mathbf{u}` and :math:`\mathbf{v}` are found iteratively using:

    .. math::
        \mathbf{f}^{(k+1)} = \frac{\mathbf{a}}{\sum \mathbf{K} \mathbf{g}^{(k)}}
        
        \mathbf{g}^{(k+1)} = \frac{\mathbf{b}}{\sum \mathbf{K}^T \mathbf{f}^{(k)}}

    Parameters
    ----------
    K : array-like, shape (B, n, m)
        Affinity matrix for each problem in the batch.
    a : array-like, shape (B, n), optional
        Source distribution for each problem. If None, uniform distribution is used.
    b : array-like, shape (B, m), optional
        Target distribution for each problem. If None, uniform distribution is used.
    nx : backend object, optional
        Numerical backend to use for computations. If None, the default backend is used.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence. The solver stops when the maximum change in
        the dual variables is below this value.
    grad : str, optional
        Gradient computation mode: 'detach', 'autodiff', or 'last_step'.

    Returns
    -------
    dict
        Dictionary containing:
        - 'T' : array-like, shape (B, n, m)
            Optimal transport plan for each problem.
        - 'potentials' : tuple of array-like, shapes ((B, n), (B, m))
            Scaling factors (f, g).
        - 'n_iters' : int
            Number of iterations performed.


    Examples
    --------
    >>> import numpy as np
    >>> from ot.batch import bregman_projection_batch
    >>> # Create batch of affinity matrices
    >>> K = np.random.randn(5, 10, 15)  # 5 problems, 10x15 cost matrices
    >>> result = bregman_projection_batch(K)
    >>> T = result['T']  # Shape (5, 10, 15)

    See Also
    --------
    ot.batch.bregman_log_projection_batch : Bregman projection in the log-domain.
    """
    if nx is None:
        nx = get_backend(a, b, K)

    B, n, m = K.shape

    if a is None:
        a = nx.ones((B, n)) / n
    if b is None:
        b = nx.ones((B, m)) / m

    if grad == "detach":
        K = nx.detach(K)
    elif grad == "last_step":
        K_, K = K.clone(), nx.detach(K)

    f = nx.ones((B, n))  # a / nx.sum(K, axis=2)
    g = nx.ones((B, m))  # b / nx.sum(K, axis=1)

    for n_iters in range(max_iter):
        f = a / nx.sum(K * g[:, None, :], axis=2)
        g = b / nx.sum(K * f[:, :, None], axis=1)
        if n_iters % 10 == 0:
            T = K * f[:, :, None] * g[:, None, :]
            marginal = nx.sum(T, axis=2)
            err = nx.max(norm_batch(marginal - a))
            if err < tol:
                break

    if grad == "last_step":
        summand_f = nx.sum(K_ * g[:, None, :], axis=2)
        f = a / summand_f
        summand_g = nx.sum(K_ * f[:, :, None], axis=1)
        g = b / summand_g

    T = K * f[:, :, None] * g[:, None, :]
    potentials = (f, g)

    out = {"T": T, "n_iters": n_iters, "potentials": potentials}

    return out


def bregman_log_projection_batch(
    K, a=None, b=None, nx=None, max_iter=10000, tol=1e-5, grad="detach"
):
    r"""
    Apply Bregman projection to a batch of affinity matrices :math:`\exp(\mathbf{K})`.

    The function solves the following optimization problem:

    .. math::
        \begin{aligned}
            \mathbf{T} = \mathop{\arg \min}_\mathbf{T} \quad & \text{KL}(\mathbf{T} \| \exp(\mathbf{K})) \\
            \text{s.t.} \quad & \mathbf{T} \mathbf{1} = \mathbf{a} \\
            & \mathbf{T}^T \mathbf{1} = \mathbf{b} \\
            & \mathbf{T} \geq 0
        \end{aligned}

    This is equivalent to:

    .. math::
        \begin{aligned}
            \mathbf{T} = \mathop{\arg \max}_\mathbf{T} \quad & \langle \mathbf{T}, \mathbf{K} \rangle_F \\
            \text{s.t.} \quad & \mathbf{T} \mathbf{1} = \mathbf{a} \\
            & \mathbf{T}^T \mathbf{1} = \mathbf{b} \\
            & \mathbf{T} \geq 0
        \end{aligned}

    The optimal solution has the form :math:`\mathbf{T} = \text{diag}(\exp(\mathbf{u})) \exp(\mathbf{K}) \text{diag}(\exp(\mathbf{v}))`,
    where the dual variables :math:`\mathbf{u}` and :math:`\mathbf{v}` are found iteratively using:

    .. math::
        \mathbf{u}^{(k+1)} = \log(\mathbf{a}) - \text{LSE}(\mathbf{K} + \mathbf{v}^{(k)})

        \mathbf{v}^{(k+1)} = \log(\mathbf{b}) - \text{LSE}(\mathbf{K}^T + \mathbf{u}^{(k)})

    where LSE denotes the log-sum-exp operation. The iterations are performed using
    the log-sum-exp trick to avoid numerical issues.

    Parameters
    ----------
    K : array-like, shape (B, n, m)
        Affinity matrix for each problem in the batch.
    a : array-like, shape (B, n), optional
        Source distribution for each problem. If None, uniform distribution is used.
    b : array-like, shape (B, m), optional
        Target distribution for each problem. If None, uniform distribution is used.
    nx : backend object, optional
        Numerical backend to use for computations. If None, the default backend is used.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence. The solver stops when the maximum change in
        the dual variables is below this value.
    grad : str, optional
        Gradient computation mode: 'detach', 'autodiff', or 'last_step'.

    Returns
    -------
    dict
        Dictionary containing:
        - 'T' : array-like, shape (B, n, m)
            Optimal transport plan for each problem.
        - 'log_T' : array-like, shape (B, n, m)
            Logarithm of the optimal transport plan for each problem.
        - 'potentials' : tuple of array-like, shapes ((B, n), (B, m))
            Log-scaling factors (u, v).
        - 'n_iters' : int
            Number of iterations performed.

    Examples
    --------
    >>> import numpy as np
    >>> from ot.batch import bregman_log_projection_batch
    >>> # Create batch of affinity matrices
    >>> K = np.random.randn(5, 10, 15)  # 5 problems, 10x15 cost matrices
    >>> result = bregman_log_projection_batch(K)
    >>> T = result['T']  # Shape (5, 10, 15)

    See Also
    --------
    ot.batch.bregman_projection_batch : standard Bregman projection.
    """

    if nx is None:
        nx = get_backend(a, b, K)

    B, n, m = K.shape

    if a is None:
        a = nx.ones((B, n)) / n
    if b is None:
        b = nx.ones((B, m)) / m

    u = nx.zeros((B, n), type_as=K)  # u = nx.log(a) - nx.logsumexp(K, axis=2).squeeze()
    v = nx.zeros((B, m), type_as=K)  # v = nx.log(b) - nx.logsumexp(K, axis=1).squeeze()

    loga = nx.log(a)
    logb = nx.log(b)

    if grad == "detach":
        K = nx.detach(K)
    elif grad == "last_step":
        K_, K = K.clone(), nx.detach(K)

    for n_iters in range(max_iter):
        u = loga - nx.logsumexp(K + v[:, None, :], axis=2)
        v = logb - nx.logsumexp(K + u[:, :, None], axis=1)

        # Check convergence once every 10 iterations
        if n_iters % 10 == 0:
            T = nx.exp(K + u[:, :, None] + v[:, None, :])
            marginal = nx.sum(T, axis=2)
            err = nx.max(norm_batch(marginal - a))
            if err < tol:
                break

    if grad == "last_step":
        summand_u = K_ + v[:, None, :]
        u = loga - nx.logsumexp(summand_u, axis=2)
        summand_v = K_ + u[:, :, None]
        v = logb - nx.logsumexp(summand_v, axis=1)

    log_T = K + u[:, :, None] + v[:, None, :]
    T = nx.exp(log_T)
    log_potentials = (u, v)

    return {"T": T, "log_T": log_T, "n_iters": n_iters, "potentials": log_potentials}
