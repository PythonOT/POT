"""
Low rank OT solvers
"""

# Author: Laurène David <laurene.david@ip-paris.fr>
#
# License: MIT License


import warnings
from .utils import unif, get_lowrank_lazytensor
from .backend import get_backend


def compute_lr_sqeuclidean_matrix(X_s, X_t, nx=None):
    """
    Compute the low rank decomposition of a squared euclidean distance matrix.
    This function won't work for any other distance metric.

    See "Section 3.5, proposition 1"

    Parameters
    ----------
    X_s : array-like, shape (n_samples_a, dim)
        samples in the source domain
    X_t : array-like, shape (n_samples_b, dim)
        samples in the target domain
    nx : POT backend, default none


    Returns
    ----------
    M1 : array-like, shape (n_samples_a, dim+2)
        First low rank decomposition of the distance matrix
    M2 : array-like, shape (n_samples_b, dim+2)
        Second low rank decomposition of the distance matrix


    References
    ----------
    .. [65] Scetbon, M., Cuturi, M., & Peyré, G. (2021).
    "Low-rank Sinkhorn factorization". In International Conference on Machine Learning.
    """

    if nx is None:
        nx = get_backend(X_s, X_t)

    ns = X_s.shape[0]
    nt = X_t.shape[0]

    # First low rank decomposition of the cost matrix (A)
    array1 = nx.reshape(nx.sum(X_s**2, 1), (-1, 1))
    array2 = nx.reshape(nx.ones(ns, type_as=X_s), (-1, 1))
    M1 = nx.concatenate((array1, array2, -2 * X_s), axis=1)

    # Second low rank decomposition of the cost matrix (B)
    array1 = nx.reshape(nx.ones(nt, type_as=X_s), (-1, 1))
    array2 = nx.reshape(nx.sum(X_t**2, 1), (-1, 1))
    M2 = nx.concatenate((array1, array2, X_t), axis=1)

    return M1, M2


def _LR_Dysktra(eps1, eps2, eps3, p1, p2, alpha, stopThr, numItermax, warn, nx=None):
    """
    Implementation of the Dykstra algorithm for the Low Rank sinkhorn OT solver.
    This function is specific to lowrank_sinkhorn.

    Parameters
    ----------
    eps1 : array-like, shape (n_samples_a, r)
        First input parameter of the Dykstra algorithm
    eps2 : array-like, shape (n_samples_b, r)
        Second input parameter of the Dykstra algorithm
    eps3 : array-like, shape (r,)
        Third input parameter of the Dykstra algorithm
    p1 : array-like, shape (n_samples_a,)
        Samples weights in the source domain (same as "a" in lowrank_sinkhorn)
    p2 : array-like, shape (n_samples_b,)
        Samples weights in the target domain (same as "b" in lowrank_sinkhorn)
    alpha: int
        Lower bound for the weight vector g (same as "alpha" in lowrank_sinkhorn)
    stopThr : float
        Stop threshold on error
    numItermax : int
        Max number of iterations
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    nx : default None
        POT backend


    Returns
    ----------
    Q : array-like, shape (n_samples_a, r)
        Dykstra update of the first low-rank matrix decomposition Q
    R: array-like, shape (n_samples_b, r)
        Dykstra update of the Second low-rank matrix decomposition R
    g : array-like, shape (r, )
        Dykstra update of the weight vector g


    References
    ----------
    .. [65] Scetbon, M., Cuturi, M., & Peyré, G. (2021).
    "Low-rank Sinkhorn factorization". In International Conference on Machine Learning.

    """

    # POT backend if None
    if nx is None:
        nx = get_backend(eps1, eps2, eps3, p1, p2)

    # ----------------- Initialisation of Dykstra algorithm -----------------
    r = len(eps3)  # rank
    g_ = nx.copy(eps3)  # \tilde{g}
    q3_1, q3_2 = nx.ones(r, type_as=p1), nx.ones(r, type_as=p1)  # q^{(3)}_1, q^{(3)}_2
    v1_, v2_ = nx.ones(r, type_as=p1), nx.ones(r, type_as=p1)  # \tilde{v}^{(1)}, \tilde{v}^{(2)}
    q1, q2 = nx.ones(r, type_as=p1), nx.ones(r, type_as=p1)  # q^{(1)}, q^{(2)}
    err = 1  # initial error

    # --------------------- Dykstra algorithm -------------------------

    # See Section 3.3 - "Algorithm 2 LR-Dykstra" in paper

    for ii in range(numItermax):
        if err > stopThr:
            # Compute u^{(1)} and u^{(2)}
            u1 = p1 / nx.dot(eps1, v1_)
            u2 = p2 / nx.dot(eps2, v2_)

            # Compute g, g^{(3)}_1 and update \tilde{g}
            g = nx.maximum(alpha, g_ * q3_1)
            q3_1 = (g_ * q3_1) / g
            g_ = nx.copy(g)

            # Compute new value of g with \prod
            prod1 = (v1_ * q1) * nx.dot(eps1.T, u1)
            prod2 = (v2_ * q2) * nx.dot(eps2.T, u2)
            g = (g_ * q3_2 * prod1 * prod2) ** (1 / 3)

            # Compute v^{(1)} and v^{(2)}
            v1 = g / nx.dot(eps1.T, u1)
            v2 = g / nx.dot(eps2.T, u2)

            # Compute q^{(1)}, q^{(2)} and q^{(3)}_2
            q1 = (v1_ * q1) / v1
            q2 = (v2_ * q2) / v2
            q3_2 = (g_ * q3_2) / g

            # Update values of \tilde{v}^{(1)}, \tilde{v}^{(2)} and \tilde{g}
            v1_, v2_ = nx.copy(v1), nx.copy(v2)
            g_ = nx.copy(g)

            # Compute error
            err1 = nx.sum(nx.abs(u1 * (eps1 @ v1) - p1))
            err2 = nx.sum(nx.abs(u2 * (eps2 @ v2) - p2))
            err = err1 + err2

        else:
            break

    else:
        if warn:
            warnings.warn(
                "Sinkhorn did not converge. You might want to "
                "increase the number of iterations `numItermax` "
            )

    # Compute low rank matrices Q, R
    Q = u1[:, None] * eps1 * v1[None, :]
    R = u2[:, None] * eps2 * v2[None, :]

    return Q, R, g


def lowrank_sinkhorn(X_s, X_t, a=None, b=None, reg=0, rank=None, alpha=None,
                     numItermax=1000, stopThr=1e-9, warn=True, log=False):
    r"""
    Solve the entropic regularization optimal transport problem under low-nonnegative rank constraints.

    The function solves the following optimization problem:

    .. math::
        \mathop{\inf_{(Q,R,g) \in \mathcal{C(a,b,r)}}} \langle C, Q\mathrm{diag}(1/g)R^T \rangle -
            \mathrm{reg} \cdot H((Q,R,g))

    where :
    - :math:`C` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`H((Q,R,g))` is the values of the three respective entropies evaluated for each term.
    - :math: `Q` and `R` are the low-rank matrix decomposition of the OT plan
    - :math: `g` is the weight vector for the low-rank decomposition of the OT plan
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (histograms, both sum to 1)
    - :math: `r` is the rank of the OT plan
    - :math: `\mathcal{C(a,b,r)}` are the low-rank couplings of the OT problem


    Parameters
    ----------
    X_s : array-like, shape (n_samples_a, dim)
        samples in the source domain
    X_t : array-like, shape (n_samples_b, dim)
        samples in the target domain
    a : array-like, shape (n_samples_a,)
        samples weights in the source domain
    b : array-like, shape (n_samples_b,)
        samples weights in the target domain
    reg : float, optional
        Regularization term >0
    rank: int, optional. Default is None. (>0)
        Nonnegative rank of the OT plan. If None, min(ns, nt) is considered.
    alpha: int, optional. Default is None. (>0 and <1/r)
        Lower bound for the weight vector g. If None, 1e-10 is considered
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    log : bool, optional
        record log if True


    Returns
    -------
    lazy_plan : LazyTensor()
        OT plan in a LazyTensor object of shape (shape_plan)
        See :any:`LazyTensor` for more information.
    value : float
        Optimal value of the optimization problem
    value_linear : float
        Linear OT loss with the optimal OT
    Q : array-like, shape (n_samples_a, r)
        First low-rank matrix decomposition of the OT plan
    R: array-like, shape (n_samples_b, r)
        Second low-rank matrix decomposition of the OT plan
    g : array-like, shape (r, )
        Weight vector for the low-rank decomposition of the OT plan


    References
    ----------
    .. [65] Scetbon, M., Cuturi, M., & Peyré, G (2021).
        "Low-Rank Sinkhorn Factorization" arXiv preprint arXiv:2103.04737.

    """

    # POT backend
    nx = get_backend(X_s, X_t)
    ns, nt = X_s.shape[0], X_t.shape[0]

    # Initialize weights a, b
    if a is None:
        a = unif(ns, type_as=X_s)
    if b is None:
        b = unif(nt, type_as=X_t)

    # Compute rank (see Section 3.1, def 1)
    r = rank
    if rank is None:
        r = min(ns, nt)

    if alpha is None:
        alpha = 1e-10

    # Dykstra algorithm won't converge if 1/rank < alpha (alpha is the lower bound for 1/rank)
    # (see "Section 3.2: The Low-rank OT Problem (LOT)" in the paper)
    if 1 / r < alpha:
        raise ValueError("alpha ({a}) should be smaller than 1/rank ({r}) for the Dykstra algorithm to converge.".format(
            a=alpha, r=1 / rank))

    if r <= 0:
        raise ValueError("The rank parameter cannot have a negative value")

    # Low rank decomposition of the sqeuclidean cost matrix (A, B)
    M1, M2 = compute_lr_sqeuclidean_matrix(X_s, X_t, nx=None)

    # Compute gamma (see "Section 3.4, proposition 4" in the paper)
    L = nx.sqrt(
        3 * (2 / (alpha**4)) * ((nx.norm(M1) * nx.norm(M2)) ** 2) +
        (reg + (2 / (alpha**3)) * (nx.norm(M1) * nx.norm(M2))) ** 2
    )
    gamma = 1 / (2 * L)

    # Initialize the low rank matrices Q, R, g
    Q = nx.ones((ns, r), type_as=a)
    R = nx.ones((nt, r), type_as=a)
    g = nx.ones(r, type_as=a)
    k = 100

    # -------------------------- Low rank algorithm ------------------------------
    # see "Section 3.3, Algorithm 3 LOT" in the paper

    for ii in range(k):
        # Compute the C*R dot matrix using the lr decomposition of C
        CR_ = nx.dot(M2.T, R)
        CR = nx.dot(M1, CR_)

        # Compute the C.t * Q dot matrix using the lr decomposition of C
        CQ_ = nx.dot(M1.T, Q)
        CQ = nx.dot(M2, CQ_)

        diag_g = (1 / g)[None, :]

        eps1 = nx.exp(-gamma * (CR * diag_g) - ((gamma * reg) - 1) * nx.log(Q))
        eps2 = nx.exp(-gamma * (CQ * diag_g) - ((gamma * reg) - 1) * nx.log(R))
        omega = nx.diag(nx.dot(Q.T, CR))
        eps3 = nx.exp(gamma * omega / (g**2) - (gamma * reg - 1) * nx.log(g))

        Q, R, g = _LR_Dysktra(
            eps1, eps2, eps3, a, b, alpha, stopThr, numItermax, warn, nx
        )
        Q = Q + 1e-16
        R = R + 1e-16

    # ----------------- Compute lazy_plan, value and value_linear  ------------------
    # see "Section 3.2: The Low-rank OT Problem" in the paper

    # Compute lazy plan (using LazyTensor class)
    lazy_plan = get_lowrank_lazytensor(Q, R, 1 / g)

    # Compute value_linear (using trace formula)
    v1 = nx.dot(Q.T, M1)
    v2 = nx.dot(R, (v1.T * diag_g).T)
    value_linear = nx.sum(nx.diag(nx.dot(M2.T, v2)))

    # Compute value with entropy reg (entropy of Q, R, g must be computed separatly, see "Section 3.2" in the paper)
    reg_Q = nx.sum(Q * nx.log(Q + 1e-16))  # entropy for Q
    reg_g = nx.sum(g * nx.log(g + 1e-16))  # entropy for g
    reg_R = nx.sum(R * nx.log(R + 1e-16))  # entropy for R
    value = value_linear + reg * (reg_Q + reg_g + reg_R)

    if log:
        dict_log = dict()
        dict_log["value"] = value
        dict_log["value_linear"] = value_linear
        dict_log["lazy_plan"] = lazy_plan

        return Q, R, g, dict_log

    return Q, R, g
