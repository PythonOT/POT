# -*- coding: utf-8 -*-
"""
Low rank Solvers
"""

# Author: Yessin Moakher <yessin.moakher@polytechnique.edu>
#
# License: MIT License

from ..utils import list_to_array
from ..backend import get_backend
from ..bregman import sinkhorn
from ..unbalanced import sinkhorn_unbalanced


def _initialize_couplings(a, b, r, nx, reg_init=1, random_state=42):
    """Initialize the couplings Q, R, T for the Factor Relaxation algorithm."""

    n = a.shape[0]
    m = b.shape[0]

    nx.seed(seed=random_state)
    M_Q = nx.rand(n, r, type_as=a)
    M_R = nx.rand(m, r, type_as=a)
    M_T = nx.rand(r, r, type_as=a)

    g_Q, g_R = (
        nx.full(r, 1 / r, type_as=a),
        nx.full(r, 1 / r, type_as=a),
    )  # Shape (r,) and (r,)

    Q = sinkhorn(a, g_Q, M_Q, reg_init, method="sinkhorn_log")
    R = sinkhorn(b, g_R, M_R, reg_init, method="sinkhorn_log")
    T = sinkhorn(
        nx.dot(Q.T, nx.ones(n, type_as=a)),
        nx.dot(R.T, nx.ones(m, type_as=a)),
        M_T,
        reg_init,
        method="sinkhorn_log",
    )

    return Q, R, T


def _compute_gradient_Q(M, Q, R, X, g_Q, nx):
    """Compute the gradient of the loss with respect to Q."""

    n = Q.shape[0]

    term1 = nx.dot(
        nx.dot(M, R), X.T
    )  # The order of multiplications is important because r<<min{n,m}
    term2 = nx.diag(nx.dot(nx.dot(term1.T, Q), nx.diag(1 / g_Q))).reshape(1, -1)
    term3 = nx.dot(nx.ones((n, 1), type_as=M), term2)
    grad_Q = term1 - term3

    return grad_Q


def _compute_gradient_R(M, Q, R, X, g_R, nx):
    """Compute the gradient of the loss with respect to R."""

    m = R.shape[0]

    term1 = nx.dot(nx.dot(M.T, Q), X)
    term2 = nx.diag(nx.dot(nx.diag(1 / g_R), nx.dot(R.T, term1))).reshape(1, -1)
    term3 = nx.dot(nx.ones((m, 1), type_as=M), term2)
    grad_R = term1 - term3

    return grad_R


def _compute_gradient_T(Q, R, M, g_Q, g_R, nx):
    """Compute the gradient of the loss with respect to T."""

    term_1 = nx.dot(nx.dot(Q.T, M), R)
    return nx.dot(nx.dot(nx.diag(1 / g_Q), term_1), nx.diag(1 / g_R))


def _compute_distance(Q_new, R_new, T_new, Q, R, T, nx):
    """Compute the distance between the new and the old couplings."""

    return (
        nx.sum((Q_new - Q) ** 2) + nx.sum((R_new - R) ** 2) + nx.sum((T_new - T) ** 2)
    )


def solve_balanced_FRLC(
    a,
    b,
    M,
    r,
    tau,
    gamma,
    stopThr=1e-5,
    numItermax=1000,
    log=False,
):
    r"""
    Solve the low-rank balanced optimal transport problem using Factor Relaxation
    with Latent Coupling and return the OT matrix.

    The function solves the following optimization problem:

    .. math::
        \textbf{P} = \mathop{\arg \min}_P \quad \langle \textbf{P}, \mathbf{M} \rangle_F

            \text{s.t.}  \textbf{P} = \textbf{Q} \operatorname{diag}(1/g_Q)\textbf{T}\operatorname{diag}(1/g_R)\textbf{R}^T

                \textbf{Q} \in \Pi_{a,\cdot}, \quad \textbf{R} \in \Pi_{b,\cdot}, \quad \textbf{T} \in \Pi_{g_Q,g_R}

                \textbf{Q}  \in \mathbb{R}^+_{n,r},\textbf{R} \in \mathbb{R}^+_{m,r},\textbf{T} \in \mathbb{R}^+_{r,r}

    where:

    - :math:`\mathbf{M}` is the given cost matrix.
    - :math:`g_Q := \mathbf{Q}^T 1_n, \quad g_R := \mathbf{R}^T 1_m`.
    - :math:`\Pi_a, \cdot := \left\{ \mathbf{P} \mid \mathbf{P} 1_m = a \right\}, \quad \Pi_{\cdot, b} := \left\{ \mathbf{P} \mid \mathbf{P}^T 1_n = b \right\}, \quad \Pi_{a,b} := \Pi_{a, \cdot} \cap \Pi_{\cdot, b}`.


    Parameters
    ----------
    a : array-like, shape (n,)
        samples weights in the source domain
    b : array-like, shape (m,)
        samples in the target domain
    M : array-like, shape (n, m)
        loss matrix
    r : int
        Rank constraint for the transport plan P.
    tau : float
        Regularization parameter controlling the relaxation of the inner marginals.
    gamma : float
        Step size (learning rate) for the coordinate mirror descent algorithm.
    numItermax : int, optional
        Max number of iterations for the mirror descent optimization.
    stopThr : float, optional
        Stop threshold on error (>0)
    log : bool, optional
        Print cost value at each iteration.

    Returns
    -------
    P : array-like, shape (n, m)
        The computed low-rank optimal transportion matrix.

    References
    ----------
    [1] Halmos, P., Liu, X., Gold, J., & Raphael, B. (2024). Low-Rank Optimal Transport through Factor Relaxation with Latent Coupling.
    In Proceedings of the Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024).
    """

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    n, m = M.shape

    ones_n, ones_m = (
        nx.ones(n, type_as=M),
        nx.ones(m, type_as=M),
    )  # Shape (n,) and (m,)

    Q, R, T = _initialize_couplings(a, b, r, nx)  # Shape (n,r), (m,r), (r,r)
    g_Q, g_R = nx.dot(Q.T, ones_n), nx.dot(R.T, ones_m)  # Shape (r,) and (r,)
    X = nx.dot(nx.dot(nx.diag(1 / g_Q), T), nx.diag(1 / g_R))  # Shape (r,r)

    for i in range(numItermax):
        grad_Q = _compute_gradient_Q(M, Q, R, X, g_Q, nx)  # Shape (n,r)
        grad_R = _compute_gradient_R(M, Q, R, X, g_R, nx)  # Shape (m,r)

        gamma_k = gamma / max(
            nx.max(nx.abs(grad_Q)), nx.max(nx.abs(grad_R))
        )  # l-inf normalization

        # We can parallelize the calculation of Q_new and R_new
        Q_new = sinkhorn_unbalanced(
            a=a,
            b=g_Q,
            M=grad_Q,
            reg=1 / gamma_k,
            reg_m=[float("inf"), tau],
            method="sinkhorn_stabilized",
        )

        R_new = sinkhorn_unbalanced(
            a=b,
            b=g_R,
            M=grad_R,
            reg=1 / gamma_k,
            reg_m=[float("inf"), tau],
            method="sinkhorn_stabilized",
        )

        g_Q = nx.dot(Q_new.T, ones_n)
        g_R = nx.dot(R_new.T, ones_m)

        grad_T = _compute_gradient_T(Q_new, R_new, M, g_Q, g_R, nx)  # Shape (r, r)

        gamma_T = gamma / nx.max(nx.abs(grad_T))

        T_new = sinkhorn(
            g_R, g_Q, grad_T, reg=1 / gamma_T, method="sinkhorn_log"
        )  # Shape (r, r)

        X_new = nx.dot(nx.dot(nx.diag(1 / g_Q), T_new), nx.diag(1 / g_R))  # Shape (r,r)

        if log:
            print(f"iteration {i} ", nx.sum(M * nx.dot(nx.dot(Q_new, X_new), R_new.T)))

        if (
            _compute_distance(Q_new, R_new, T_new, Q, R, T, nx)
            < gamma_k * gamma_k * stopThr
        ):
            return nx.dot(nx.dot(Q_new, X_new), R_new.T)  # Shape (n, m)

        Q, R, T, X = Q_new, R_new, T_new, X_new


if __name__ == "__main__":
    import torch

    grid_size = 4
    torch.manual_seed(42)
    x_vals = torch.linspace(0, 3, grid_size)
    y_vals = torch.linspace(0, 3, grid_size)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
    source_points = torch.stack([X.ravel(), Y.ravel()], dim=-1)  # (16, 2)
    a = torch.ones(len(source_points)) / len(source_points)  # Uniform distribution

    # Generate Target Distribution (Gaussian Samples)
    mean = torch.tensor([2.0, 2.0])
    cov = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    target_points = torch.distributions.MultivariateNormal(
        mean, covariance_matrix=cov
    ).sample((len(source_points),))  # (16, 2)
    b = torch.ones(len(target_points)) / len(target_points)  # Uniform distribution

    # Compute Cost Matrix (Squared Euclidean Distance)
    C = torch.cdist(source_points, target_points, p=2) ** 2

    # Solve OT problem (assuming you have PyTorch versions of these functions)
    print(type(a.numpy()))
    P = solve_balanced_FRLC(
        a.to(torch.float64),
        b.to(torch.float64),
        C.to(torch.float64),
        10,
        tau=1e2,
        gamma=1e2,
        stopThr=1e-7,
        numItermax=100,
        log=True,
    )
    P = sinkhorn(a, b, C, reg=1)
    print(torch.sum(P * C))
