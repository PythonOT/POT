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
from ot.batch._utils import bmv, bop, bregman_log_projection_batch
from ot.utils import list_to_array


def tensor_batch(
    a, b, C1, C2, symmetric=True, nx=None, loss="sqeuclidean", logits=None
):
    r"""
    Compute the Gromov-Wasserstein cost tensor for a batch of problems.

    The Gromov-Wasserstein distance can be expressed as:

    .. math::
        \text{GW}(\mathbf{T}, \mathbf{C}_1, \mathbf{C}_2) = \sum_{ijkl} T_{ik} T_{jl} \ell(C_{1,ij}, C_{2,kl}) = \langle \mathcal{L} \times \mathbf{T}, \mathbf{T} \rangle

    where :math:`\mathcal{L}` is a 4D tensor with elements :math:`\mathcal{L}[i,j,k,l] = \ell(C_{1,ij}, C_{2,kl})`.

    For loss functions of the form :math:`\ell(a,b) = f_1(a) + f_2(b) - \langle h_1(a), h_2(b) \rangle`,
    the tensor product :math:`\mathcal{L} \times \mathbf{T}` can be computed efficiently without explicitly computing :math:`\mathcal{L}` [12].

    This function precomputes all matrices that implicitly define `\mathcal{L}` for various loss functions.

    Parameters
    ----------
    a : array-like, shape (B, n)
        Source distributions for each problem in the batch.
    b : array-like, shape (B, m)
        Target distributions for each problem in the batch.
    C1 : array-like, shape (B, n, n) or (B, n, n, d)
        Source cost matrices for each problem. Can be a 3D array for scalar costs or a 4D array for vector-valued costs (edge features).
    C2 : array-like, shape (B, m, m) or (B, n, n, d)
        Target cost matrices for each problem. Can be a 3D array for scalar costs or a 4D array for vector-valued costs (edge features).
    symmetric : bool, optional
        Whether the cost matrices are symmetric. Default is True.
    nx : backend object, optional
        Numerical backend to use for computations. If None, the default backend is used.
    loss : str, optional
        Loss function to use. Supported values: 'sqeuclidean', 'kl'.
        Default is 'sqeuclidean'.
    logits : bool, optional
        For KL divergence, whether inputs are logits (unnormalized log probabilities).
        If True, inputs are treated as logits. Default is None.

    Returns
    -------
    dict
        Dictionary containing:
        - constC : array-like, shape (B, n, m)
            Constant term in the tensor product.
        - hC1 : array-like, shape (B, n, n, d) or (B, n, n)
        - hC2 : array-like, shape (B, m, m, d) or (B, m, m)
        - fC1 : array-like, shape (B, n, n)
        - fC2 : array-like, shape (B, m, m)


    Supported loss functions:
    ------------------------------

    **Squared Euclidean loss**:

    .. math::
        \ell(a, b) = \|a - b\|_2^2 = \sum_i (a_i - b_i)^2

    **KL divergence**:

    .. math::
        \ell(a, b) = \sum_i a_i \log\left(\frac{a_i}{b_i}\right)

    If ``logits=True``, the entries of C1 are treated as logits (unnormalized log probabilities)
    and the loss becomes:

    .. math::
        \ell(x, y) = \sum_i y_i (\log(y_i) - x_i)

    Examples
    --------
    >>> import numpy as np
    >>> from ot.batch import tensor_batch
    >>> # Create batch of cost matrices
    >>> C1 = np.random.rand(3, 5, 5)  # 3 problems, 5x5 source matrices
    >>> C2 = np.random.rand(3, 4, 4)  # 3 problems, 4x4 target matrices
    >>> a = np.ones((3, 5)) / 5  # Uniform source distributions
    >>> b = np.ones((3, 4)) / 4  # Uniform target distributions
    >>> L = tensor_batch(a, b, C1, C2, loss='sqeuclidean')

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """

    if nx is None:
        nx = get_backend(C1)

    loss = loss.lower()

    if loss == "sqeuclidean" or loss == "l2":

        def f1(C1):
            if C1.ndim == 4:
                return nx.sum(C1**2, axis=-1)
            else:
                return C1**2

        def f2(C2):
            if C2.ndim == 4:
                return nx.sum(C2**2, axis=-1)
            else:
                return C2**2

        def h1(C1):
            if C1.ndim == 3:
                C1 = nx.unsqueeze(C1, -1)
            return 2 * C1

        def h2(C2):
            if C2.ndim == 3:
                C2 = nx.unsqueeze(C2, -1)
            return C2

    elif loss == "kl":
        assert logits in [
            True,
            False,
        ], "logits must be either True or False for KL loss"

        def f1(C1):
            return nx.zeros((C1.shape[0], C1.shape[1], C1.shape[2]), type_as=C1)

        def f2(C2):
            assert C2.ndim == 4, "C2 must be a bxnxnxd tensor"
            fC2 = C2 * nx.log(C2 + 1e-15)  # Avoid log(0)
            return nx.sum(fC2, axis=-1)

        def h1(C1):
            return C1 if logits else nx.log(C1 + 1e-15)

        def h2(C2):
            return C2

    return compute_tensor_batch(f1, f2, h1, h2, a, b, C1, C2, symmetric=symmetric)


def div_between_product_batch(mu, nu, alpha, beta, divergence, nx=None):
    r"""Fast computation of the Bregman divergence between batches of product measures.
    Only support for Kullback-Leibler and half-squared L2 divergences.

    For half-squared L2 divergence:

    .. math::
        \frac{1}{2} || \mu \otimes \nu, \alpha \otimes \beta ||^2
        = \frac{1}{2} \Big[ ||\alpha||^2 ||\beta||^2 + ||\mu||^2 ||\nu||^2 - 2 \langle \alpha, \mu \rangle \langle \beta, \nu \rangle \Big]

    For Kullback-Leibler divergence:

    .. math::
        KL(\mu \otimes \nu, \alpha \otimes \beta)
        = m(\mu) * KL(\nu, \beta) + m(\nu) * KL(\mu, \alpha) + (m(\mu) - m(\alpha)) * (m(\nu) - m(\beta))

    where:

    - :math:`\mu` and :math:`\alpha` are two measures having the same shape.
    - :math:`\nu` and :math:`\beta` are two measures having the same shape.
    - :math:`m` denotes the mass of the measure

    Parameters
    ----------
    mu : array-like, shape (B, ...)
        First factor of each product measure in the batch.
    nu : array-like, shape (B, ...)
        Second factor of each product measure in the batch.
    alpha : array-like, shape (B, ...)
        Reference factor with the same shape as `mu`.
    beta : array-like, shape (B, ...)
        Reference factor with the same shape as `nu`.
    divergence : string, default = "kl"
        Bregman divergence, either "kl" (Kullback-Leibler divergence) or "l2" (half-squared L2 divergence)
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    ----------
    Bregman divergence between two product measures for each problem in the batch.
    """

    if nx is None:
        nx = get_backend(mu, nu, alpha, beta)

    axis_mu = tuple(range(1, mu.ndim)) if mu.ndim > 1 else 0
    axis_nu = tuple(range(1, nu.ndim)) if nu.ndim > 1 else 0
    axis_alpha = tuple(range(1, alpha.ndim)) if alpha.ndim > 1 else 0
    axis_beta = tuple(range(1, beta.ndim)) if beta.ndim > 1 else 0

    if divergence == "kl":
        m_mu = nx.sum(mu, axis=axis_mu)
        m_nu = nx.sum(nu, axis=axis_nu)
        m_alpha = nx.sum(alpha, axis=axis_alpha)
        m_beta = nx.sum(beta, axis=axis_beta)
        const = (m_mu - m_alpha) * (m_nu - m_beta)
        res = (
            m_nu * nx.kl_div(mu, alpha, mass=True, axis=axis_mu)
            + m_mu * nx.kl_div(nu, beta, mass=True, axis=axis_nu)
            + const
        )

    elif divergence == "l2":
        res = (
            nx.sum(alpha**2, axis=axis_alpha) * nx.sum(beta**2, axis=axis_beta)
            - 2 * nx.sum(alpha * mu, axis=axis_mu) * nx.sum(beta * nu, axis=axis_nu)
            + nx.sum(mu**2, axis=axis_mu) * nx.sum(nu**2, axis=axis_nu)
        ) / 2

    return res


def loss_quadratic_batch(L, T, recompute_const=False, symmetric=True, nx=None):
    r"""
    Computes the gromov-wasserstein cost given a cost tensor and transport plan. Batched version.

    Parameters
    ----------
    L : dict
        Cost tensor as returned by `tensor_batch`.
    T : array-like, shape (B, n, m)
        Transport plan.
    recompute_const : bool, optional
        Whether to recompute the constant term. Default is False. This should be set to True if T does not satisfy the marginal constraints.
    symmetric : bool, optional
        Whether to use symmetric version. Default is True.
    nx : module, optional
        Backend to use. Default is None.

    Examples
    --------
    >>> import numpy as np
    >>> from ot.batch import tensor_batch, loss_quadratic_batch
    >>> # Create batch of cost matrices
    >>> C1 = np.random.rand(3, 5, 5)  # 3 problems, 5x5 source matrices
    >>> C2 = np.random.rand(3, 4, 4)  # 3 problems, 4x4 target matrices
    >>> a = np.ones((3, 5)) / 5  # Uniform source distributions
    >>> b = np.ones((3, 4)) / 4  # Uniform target distributions
    >>> L = tensor_batch(a, b, C1, C2, loss='sqeuclidean')
    >>> # Use the uniform transport plan for testing
    >>> T = np.ones((3, 5, 4)) / (5 * 4)
    >>> loss = loss_quadratic_batch(L, T, recompute_const=True)
    >>> loss.shape
    (3,)

    See Also
    --------
    ot.batch.tensor_batch : From computing the cost tensor L.
    ot.batch.solve_gromov_batch : For finding the optimal transport plan T.
    """
    if nx is None:
        nx = get_backend(T)
    LT = tensor_product_batch(
        L, T, nx=nx, recompute_const=recompute_const, symmetric=symmetric
    )
    return nx.sum(LT * T, axis=(1, 2))


def loss_quadratic_samples_batch(
    a,
    b,
    C1,
    C2,
    T,
    M=None,
    alpha=None,
    unbalanced=None,
    unbalanced_type="kl",
    loss="sqeuclidean",
    symmetric=True,
    nx=None,
    logits=None,
    recompute_const=False,
):
    r"""
    Computes the gromov-wasserstein for samples C1, C2 and transport plan. Batched version.

    Parameters
    ----------
    a : array-like, shape (B, n)
        Source distributions.
    b : array-like, shape (B, m)
        Target distributions.
    C1 : array-like, shape (B, n, n) or (B, n, n, d)
        Source cost matrices.
    C2 : array-like, shape (B, m, m) or (B, n, n, d)
        Target cost matrices.
    T : array-like, shape (B, n, m)
        Transport plan.
    M : array-like, shape (B, n, m)
        Cost matrix between features across domains (default is None).
    alpha : float, array-like or list (B,) optional
        Weight the quadratic term (alpha*Gromov) and the linear term
        ((1-alpha)*Wass) in the Fused Gromov-Wasserstein problem. Not used for
        Gromov problem (when M is not provided). By default ``alpha=None``
        corresponds to ``alpha=1`` for Gromov problem (``M==None``) and
        ``alpha=0.5`` for Fused Gromov-Wasserstein problem (``M!=None``).
        If alpha is a scalar, it is used for all problems in the batch.
    unbalanced : float array-like or list(B,) optional
        Unbalanced penalization weight :math:`\lambda_u`. If unbalanced is a scalar, it is used for all problems in the batch.
    unbalanced_type : string, optional
        Type of unbalanced penalization function, either "kl" (Kullback-Leibler divergence) or "l2" (half-squared L2 divergence)
    loss : str, optional
        Loss function to use. Supported values: 'sqeuclidean', 'kl'.
        Default is 'sqeuclidean'.
    symmetric : bool, optional
        Whether to use symmetric version. Default is True.
    nx : module, optional
        Backend to use. Default is None.
    logits : bool, optional
        For KL divergence, whether inputs are logits (unnormalized log probabilities).
        If True, inputs are treated as logits. Default is None.
    recompute_const : bool, optional
        Whether to recompute the constant term. Default is False. This should be set to True if T does not satisfy the marginal constraints.
        Will be set to True if unbalanced is not None.


    Examples
    --------
    >>> import numpy as np
    >>> from ot.batch import loss_quadratic_samples_batch
    >>> # Create batch of cost matrices
    >>> C1 = np.random.rand(3, 5, 5)  # 3 problems, 5x5 source matrices
    >>> C2 = np.random.rand(3, 4, 4)  # 3 problems, 4x4 target matrices
    >>> a = np.ones((3, 5)) / 5  # Uniform source distributions
    >>> b = np.ones((3, 4)) / 4  # Uniform target distributions
    >>> # Use the uniform transport plan for testing
    >>> T = np.ones((3, 5, 4)) / (5 * 4)
    >>> loss = loss_quadratic_samples_batch(a, b, C1, C2, T, recompute_const=True)
    >>> loss.shape
    (3,)

    See Also
    --------
    ot.batch.tensor_batch : From computing the cost tensor L.
    ot.batch.solve_gromov_batch : For finding the optimal transport plan T.
    """
    if nx is None:
        nx = get_backend(T)

    if isinstance(loss, str) and loss in ["sqeuclidean", "kl", "l2"]:
        L = tensor_batch(
            a, b, C1, C2, symmetric=symmetric, nx=nx, loss=loss, logits=logits
        )
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    if unbalanced is not None:
        recompute_const = True

    quadratic = loss_quadratic_batch(
        L, T, recompute_const=recompute_const, symmetric=symmetric, nx=nx
    )

    if unbalanced is None and M is None:
        return quadratic

    B = T.shape[0]

    if unbalanced is not None:
        if unbalanced_type is None:
            raise ValueError(
                "unbalanced_type must be specified if unbalanced is not None"
            )

        unbalanced_type = unbalanced_type.lower()

        if unbalanced_type not in ["kl", "l2"]:
            raise ValueError(
                f"Unknown unbalanced_type: {unbalanced_type}, expected 'kl' or 'l2'"
            )

        if isinstance(unbalanced, list):
            unbalanced = list_to_array(unbalanced, nx=nx)

        if hasattr(unbalanced, "ndim") and unbalanced.ndim > 0:
            if unbalanced.ndim != 1 or unbalanced.shape[0] != B:
                raise ValueError(
                    f"If reg_marginals is not a scalar, it must have shape ({B},), got {unbalanced.shape}"
                )

        T1 = nx.sum(T, 2)
        T2 = nx.sum(T, 1)
        unbalanced_term = div_between_product_batch(
            T1,
            T2,
            a,
            b,
            divergence=unbalanced_type,
            nx=nx,
        )

    if M is not None:
        if alpha is None:
            alpha = 0.5
        if isinstance(alpha, list):
            alpha = list_to_array(alpha, nx=nx)
        if hasattr(alpha, "ndim") and alpha.ndim > 0:
            if alpha.ndim != 1 or alpha.shape[0] != B:
                raise ValueError(
                    f"If alpha is not a scalar, it must have shape ({B},), got {alpha.shape}"
                )
        linear = loss_linear_batch(M, T, nx=nx)

    if M is not None and unbalanced is not None:
        return (1 - alpha) * linear + alpha * quadratic + unbalanced * unbalanced_term

    elif M is not None and unbalanced is None:
        return (1 - alpha) * linear + alpha * quadratic

    else:
        return quadratic + unbalanced * unbalanced_term


def solve_gromov_batch(
    C1,
    C2,
    reg=1e-2,
    a=None,
    b=None,
    loss="sqeuclidean",
    symmetric=None,
    M=None,
    alpha=None,
    T_init=None,
    max_iter=50,
    tol=1e-5,
    max_iter_inner=50,
    tol_inner=1e-5,
    grad="envelope",
    logits=None,
):
    r"""
    Solves a batch of Gromov-Wasserstein optimal transport problems using proximal gradient [12, 81].
    For each problem in the batch, solves:

    .. math::
        \begin{aligned}
            \min_{\mathbf{T} \geq 0} \quad & \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} \\
            \text{s.t.} \quad & \mathbf{T} \mathbf{1} = \mathbf{a} \\
            & \mathbf{T}^T \mathbf{1} = \mathbf{b} \\
            & \mathbf{T} \geq 0
        \end{aligned}

    If :math:`\mathbf{M}` and :math:`\alpha` are given, solves the more general fused Gromov-Wasserstein problem:

    .. math::
        \begin{aligned}
            \min_{\mathbf{T} \geq 0} \quad & (1-\alpha) \sum_{i,j} M_{i,j} \mathbf{T}_{i,j} + \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} \\
            \text{s.t.} \quad & \mathbf{T} \mathbf{1} = \mathbf{a} \\
            & \mathbf{T}^T \mathbf{1} = \mathbf{b} \\
            & \mathbf{T} \geq 0
        \end{aligned}

    Writing the objective as :math:`(1-\alpha) \langle \mathbf{M}, \mathbf{T} \rangle + \alpha \langle \mathcal{L} \otimes \mathbf{T}, \mathbf{T} \rangle`,
    the solver uses proximal gradient descent where each iteration is:

    .. math::
        \begin{aligned}
            \mathbf{T}_{k+1} = \mathop{\arg \min}_{\mathbf{T} \geq 0} \quad & \langle \mathbf{M}_k, \mathbf{T} \rangle + \epsilon \, \text{KL}(\mathbf{T} \| \mathbf{T}_k) \\
            \text{where} \quad & \mathbf{M}_k = (1 - \alpha) \mathbf{M} + 2 \alpha \mathcal{L} \otimes \mathbf{T}_k
        \end{aligned}

    This can be rewritten as:

    .. math::
        \begin{aligned}
            \mathbf{T}_{k+1} = \mathop{\arg \min}_{\mathbf{T} \geq 0} \quad & \langle \mathbf{M}_k - \epsilon \log(\mathbf{T}_k), \mathbf{T} \rangle - \epsilon H(\mathbf{T})
        \end{aligned}

    where :math:`H(\mathbf{T})` is the entropy of :math:`\mathbf{T}`. Thus each iteration can be solved using the Bregman projection solver implemented in `bregman_log_projection_batch`. 
    
    Note that the inner optimization problem does not need to be solved exactly. In practice it sufficient to set `max_iter_inner` to a low value (e.g. 20) and/or `tol_inner` to a high value (e.g. 1e-2).

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
        Type of loss function, can be 'sqeuclidean' or 'kl' or a QuadraticMetric instance.
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
        Type of gradient computation, either or 'autodiff', 'envelope' or 'detach'. 'autodiff' provides gradients wrt all
        outputs (`plan, value, value_linear`) but with important memory cost.
        'envelope' provides gradients only for (`value, value_linear`)`.  `detach`` is the fastest option but
        provides no gradients. Default is 'detach'.
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

    See Also
    --------
    ot.batch.tensor_batch : From computing the cost tensor L.
    ot.solve_gromov : Non-batched solver for Gromov-Wasserstein. Note that the non-batched solver uses a different algorithm (conditional gradient) and might not give the same results.
    
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    .. [81] Xu, H., Luo, D., & Carin, L. (2019). "Scalable Gromov-Wasserstein learning for graph partitioning and matching."
        Advances in neural information processing systems (NeurIPS). 2019.
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

    if isinstance(loss, str):
        L = tensor_batch(
            a, b, C1, C2, symmetric=symmetric, nx=nx, loss=loss, logits=logits
        )
    else:
        raise ValueError(f"Unknown loss function: {loss}")

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

    T = T_init
    log_T = nx.log(T + 1e-15)  # Avoid log(0)
    for iter in range(max_iter):
        T_prev = T
        Mk = (1 - alpha) * M + 2 * alpha * tensor_product_batch(
            L, T, nx=nx, recompute_const=False, symmetric=symmetric
        )
        K = -Mk / reg + log_T
        out = bregman_log_projection_batch(
            K, a, b, nx=nx, max_iter=max_iter_inner, tol=tol_inner, grad=grad_inner
        )
        T = out["T"]
        log_T = out["log_T"]
        max_err = nx.max(nx.mean(nx.abs(T_prev - T), axis=(1, 2)))
        if max_err < tol:
            break

    if grad == "detach":
        T = nx.detach(T)
        M = nx.detach(M)
        L = detach_cost_tensor(L, nx=nx)
    elif grad == "envelope":
        T = nx.detach(T)

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
        potentials=out["potentials"],
        log=log,
    )

    return res


### --------------------- Utility functions for quadratic OT --------------------- ###


def compute_tensor_batch(f1, f2, h1, h2, a, b, C1, C2, symmetric=True):
    """
    Gromov-Wasserstein writes as:
        GW(T,C1,C2) = sum_ijkl T_ik T_jl l(C1_ij, C2_kl) = < LxT, T >
    Where L is a cost tensor L[i,j,k,l] = l(C1_ij, C2_kl).

    For loss function of form l(a,b) = f1(a) + f2(b) - < h1(a), h2(b) >
    The tensor product LxT can be computed fast using tensor_product [12].

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """

    fC1 = f1(C1)
    fC2 = f2(C2)
    if not symmetric:
        fC1 = 0.5 * (fC1 + transpose(fC1))
        fC2 = 0.5 * (fC2 + transpose(fC2))
    hC1 = h1(C1)
    hC2 = h2(C2)

    constC = compute_const_from_marginals(fC1, fC2, a, b)

    L = {"constC": constC, "hC1": hC1, "hC2": hC2, "fC1": fC1, "fC2": fC2}

    return L


def tensor_product_batch(L, T, nx=None, recompute_const=False, symmetric=True):
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
        const = compute_const_from_marginals(
            L["fC1"], L["fC2"], nx.sum(T, axis=2), nx.sum(T, axis=1), nx=nx
        )
    else:
        const = L["constC"]

    hC1 = L["hC1"]
    hC2 = L["hC2"]

    dot = nx.einsum("bijd,bjk->bikd", hC1, T)
    dot = nx.einsum("bikd,bjkd->bijd", dot, hC2)
    dot = nx.sum(dot, axis=-1)

    if not symmetric:
        dot_t = nx.einsum("bijd,bjk->bikd", transpose(hC1), T)
        dot_t = nx.einsum("bikd,bjkd->bijd", dot_t, transpose(hC2))
        dot_t = nx.sum(dot_t, axis=-1)
        dot = (dot + dot_t) / 2  # Average the two symmetric terms

    return const - dot


def transpose(C, nx=None):
    if nx is None:
        nx = get_backend(C)
    return nx.transpose(C, (0, 2, 1)) if C.ndim == 3 else nx.transpose(C, (0, 2, 1, 3))


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
