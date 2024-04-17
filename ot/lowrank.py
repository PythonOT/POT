"""
Low rank OT solvers
"""

# Author: Laurène David <laurene.david@ip-paris.fr>
#
# License: MIT License


import warnings
from .utils import unif, dist, get_lowrank_lazytensor
from .backend import get_backend
from .bregman import sinkhorn

# test if sklearn is installed for linux-minimal-deps
try:
    import sklearn.cluster
    sklearn_import = True
except ImportError:
    sklearn_import = False


def _init_lr_sinkhorn(X_s, X_t, a, b, rank, init, reg_init, random_state, nx=None):
    """
    Implementation of different initialization strategies for the low rank sinkhorn solver (Q ,R, g).
    This function is specific to lowrank_sinkhorn.

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
    rank : int
        Nonnegative rank of the OT plan.
    init : str
        Initialization strategy for Q, R and g. 'random', 'trivial' or 'kmeans'
    reg_init : float, optional.
        Regularization term for a 'kmeans' init.
    random_state : int, optional.
        Random state for a "random" or 'kmeans' init strategy
    nx : optional, Default is None
        POT backend


    Returns
    ---------
    Q : array-like, shape (n_samples_a, r)
        Init for the first low-rank matrix decomposition of the OT plan (Q)
    R: array-like, shape (n_samples_b, r)
        Init for the second low-rank matrix decomposition of the OT plan (R)
    g : array-like, shape (r, )
        Init for the weight vector of the low-rank decomposition of the OT plan (g)


    References
    -----------
    .. [65] Scetbon, M., Cuturi, M., & Peyré, G. (2021).
        "Low-rank Sinkhorn factorization". In International Conference on Machine Learning.

    """

    if nx is None:
        nx = get_backend(X_s, X_t, a, b)

    ns = X_s.shape[0]
    nt = X_t.shape[0]
    r = rank

    if init == "random":
        nx.seed(seed=random_state)

        # Init g
        g = nx.abs(nx.randn(r, type_as=X_s)) + 1
        g = g / nx.sum(g)

        # Init Q
        Q = nx.abs(nx.randn(ns, r, type_as=X_s)) + 1
        Q = (Q.T * (a / nx.sum(Q, axis=1))).T

        # Init R
        R = nx.abs(nx.randn(nt, rank, type_as=X_s)) + 1
        R = (R.T * (b / nx.sum(R, axis=1))).T

    if init == "deterministic":
        # Init g
        g = nx.ones(rank) / rank

        lambda_1 = min(nx.min(a), nx.min(g), nx.min(b)) / 2
        a1 = nx.arange(start=1, stop=ns + 1, type_as=X_s)
        a1 = a1 / nx.sum(a1)
        a2 = (a - lambda_1 * a1) / (1 - lambda_1)

        b1 = nx.arange(start=1, stop=nt + 1, type_as=X_s)
        b1 = b1 / nx.sum(b1)
        b2 = (b - lambda_1 * b1) / (1 - lambda_1)

        g1 = nx.arange(start=1, stop=rank + 1, type_as=X_s)
        g1 = g1 / nx.sum(g1)
        g2 = (g - lambda_1 * g1) / (1 - lambda_1)

        # Init Q
        Q1 = lambda_1 * nx.dot(a1[:, None], nx.reshape(g1, (1, -1)))
        Q2 = (1 - lambda_1) * nx.dot(a2[:, None], nx.reshape(g2, (1, -1)))
        Q = Q1 + Q2

        # Init R
        R1 = lambda_1 * nx.dot(b1[:, None], nx.reshape(g1, (1, -1)))
        R2 = (1 - lambda_1) * nx.dot(b2[:, None], nx.reshape(g2, (1, -1)))
        R = R1 + R2

    if init == "kmeans":
        if sklearn_import:
            # Init g
            g = nx.ones(rank, type_as=X_s) / rank

            # Init Q
            kmeans_Xs = sklearn.cluster.KMeans(n_clusters=rank, random_state=random_state, n_init="auto")
            kmeans_Xs.fit(X_s)
            Z_Xs = nx.from_numpy(kmeans_Xs.cluster_centers_)
            C_Xs = dist(X_s, Z_Xs)  # shape (ns, rank)
            C_Xs = C_Xs / nx.max(C_Xs)
            Q = sinkhorn(a, g, C_Xs, reg=reg_init, numItermax=10000, stopThr=1e-3)

            # Init R
            kmeans_Xt = sklearn.cluster.KMeans(n_clusters=rank, random_state=random_state, n_init="auto")
            kmeans_Xt.fit(X_t)
            Z_Xt = nx.from_numpy(kmeans_Xt.cluster_centers_)
            C_Xt = dist(X_t, Z_Xt)  # shape (nt, rank)
            C_Xt = C_Xt / nx.max(C_Xt)
            R = sinkhorn(b, g, C_Xt, reg=reg_init, numItermax=10000, stopThr=1e-3)

        else:
            raise ImportError("Scikit-learn should be installed to use the 'kmeans' init.")

    return Q, R, g


def compute_lr_sqeuclidean_matrix(X_s, X_t, rescale_cost, nx=None):
    """
    Compute the low rank decomposition of a squared euclidean distance matrix.
    This function won't work for other distance metrics.

    See "Section 3.5, proposition 1"

    Parameters
    ----------
    X_s : array-like, shape (n_samples_a, dim)
        samples in the source domain
    X_t : array-like, shape (n_samples_b, dim)
        samples in the target domain
    rescale_cost : bool
        Rescale the low rank factorization of the sqeuclidean cost matrix
    nx : default None
        POT backend


    Returns
    ----------
    M1 : array-like, shape (n_samples_a, dim+2)
        First low rank decomposition of the distance matrix
    M2 : array-like, shape (n_samples_b, dim+2)
        Second low rank decomposition of the distance matrix


    References
    -----------
    .. [65] Scetbon, M., Cuturi, M., & Peyré, G. (2021).
        "Low-rank Sinkhorn factorization". In International Conference on Machine Learning.
    """

    if nx is None:
        nx = get_backend(X_s, X_t)

    ns = X_s.shape[0]
    nt = X_t.shape[0]

    # First low rank decomposition of the cost matrix (A)
    array1 = nx.reshape(nx.sum(X_s**2, 1), (-1, 1))
    array2 = nx.ones((ns, 1), type_as=X_s)
    M1 = nx.concatenate((array1, array2, -2 * X_s), axis=1)

    # Second low rank decomposition of the cost matrix (B)
    array1 = nx.ones((nt, 1), type_as=X_s)
    array2 = nx.reshape(nx.sum(X_t**2, 1), (-1, 1))
    M2 = nx.concatenate((array1, array2, X_t), axis=1)

    if rescale_cost is True:
        M1 = M1 / nx.sqrt(nx.max(M1))
        M2 = M2 / nx.sqrt(nx.max(M2))

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
        "Low-rank Sinkhorn Factorization". In International Conference on Machine Learning.

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
                "Dykstra did not converge. You might want to "
                "increase the number of iterations `numItermax` "
            )

    # Compute low rank matrices Q, R
    Q = u1[:, None] * eps1 * v1[None, :]
    R = u2[:, None] * eps2 * v2[None, :]

    return Q, R, g


def lowrank_sinkhorn(X_s, X_t, a=None, b=None, reg=0, rank=None, alpha=1e-10, rescale_cost=True,
                     init="random", reg_init=1e-1, seed_init=49, gamma_init="rescale",
                     numItermax=2000, stopThr=1e-7, warn=True, log=False):
    r"""
    Solve the entropic regularization optimal transport problem under low-nonnegative rank constraints
    on the couplings.

    The function solves the following optimization problem:

    .. math::
        \mathop{\inf_{(\mathbf{Q},\mathbf{R},\mathbf{g}) \in \mathcal{C}(\mathbf{a},\mathbf{b},r)}} \langle \mathbf{C}, \mathbf{Q}\mathrm{diag}(1/\mathbf{g})\mathbf{R}^\top \rangle -
            \mathrm{reg} \cdot H((\mathbf{Q}, \mathbf{R}, \mathbf{g}))

    where :

    - :math:`\mathbf{C}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`H((\mathbf{Q}, \mathbf{R}, \mathbf{g}))` is the values of the three respective entropies evaluated for each term.
    - :math:`\mathbf{Q}` and :math:`\mathbf{R}` are the low-rank matrix decomposition of the OT plan
    - :math:`\mathbf{g}` is the weight vector for the low-rank decomposition of the OT plan
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (histograms, both sum to 1)
    - :math:`r` is the rank of the OT plan
    - :math:`\mathcal{C}(\mathbf{a}, \mathbf{b}, r)` are the low-rank couplings of the OT problem


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
    rank : int, optional. Default is None. (>0)
        Nonnegative rank of the OT plan. If None, min(ns, nt) is considered.
    alpha : int, optional. Default is 1e-10. (>0 and <1/r)
        Lower bound for the weight vector g.
    rescale_cost : bool, optional. Default is False
        Rescale the low rank factorization of the sqeuclidean cost matrix
    init : str, optional. Default is 'random'.
        Initialization strategy for the low rank couplings. 'random', 'deterministic' or 'kmeans'
    reg_init : float, optional. Default is 1e-1. (>0)
        Regularization term for a 'kmeans' init. If None, 1 is considered.
    seed_init : int, optional. Default is 49. (>0)
        Random state for a 'random' or 'kmeans' init strategy.
    gamma_init : str, optional. Default is "rescale".
        Initialization strategy for gamma. 'rescale', or 'theory'
        Gamma is a constant that scales the convergence criterion of the Mirror Descent
        optimization scheme used to compute the low-rank couplings (Q, R and g)
    numItermax : int, optional. Default is 2000.
        Max number of iterations for the Dykstra algorithm
    stopThr : float, optional. Default is 1e-7.
        Stop threshold on error (>0) in Dykstra
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    log : bool, optional
        record log if True


    Returns
    ---------
    Q : array-like, shape (n_samples_a, r)
        First low-rank matrix decomposition of the OT plan
    R: array-like, shape (n_samples_b, r)
        Second low-rank matrix decomposition of the OT plan
    g : array-like, shape (r, )
        Weight vector for the low-rank decomposition of the OT
    log : dict (lazy_plan, value and value_linear)
        log dictionary return only if log==True in parameters


    References
    ----------
    .. [65] Scetbon, M., Cuturi, M., & Peyré, G. (2021).
        "Low-rank Sinkhorn Factorization". In International Conference on Machine Learning.

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
    else:
        r = min(ns, nt, rank)

    if r <= 0:
        raise ValueError("The rank parameter cannot have a negative value")

    # Dykstra won't converge if 1/rank < alpha (see Section 3.2)
    if 1 / r < alpha:
        raise ValueError("alpha ({a}) should be smaller than 1/rank ({r}) for the Dykstra algorithm to converge.".format(
            a=alpha, r=1 / rank))

    # Low rank decomposition of the sqeuclidean cost matrix
    M1, M2 = compute_lr_sqeuclidean_matrix(X_s, X_t, rescale_cost, nx)

    # Initialize the low rank matrices Q, R, g
    Q, R, g = _init_lr_sinkhorn(X_s, X_t, a, b, r, init, reg_init, seed_init, nx=nx)

    # Gamma initialization
    if gamma_init == "theory":
        L = nx.sqrt(
            3 * (2 / (alpha**4)) * ((nx.norm(M1) * nx.norm(M2)) ** 2) +
            (reg + (2 / (alpha**3)) * (nx.norm(M1) * nx.norm(M2))) ** 2
        )
        gamma = 1 / (2 * L)

    if gamma_init not in ["rescale", "theory"]:
        raise (NotImplementedError('Not implemented gamma_init="{}"'.format(gamma_init)))

    # -------------------------- Low rank algorithm ------------------------------
    # see "Section 3.3, Algorithm 3 LOT"

    for ii in range(100):
        # Compute C*R dot using the lr decomposition of C
        CR = nx.dot(M2.T, R)
        CR_ = nx.dot(M1, CR)
        diag_g = (1 / g)[None, :]
        CR_g = CR_ * diag_g

        # Compute C.T * Q using the lr decomposition of C
        CQ = nx.dot(M1.T, Q)
        CQ_ = nx.dot(M2, CQ)
        CQ_g = CQ_ * diag_g

        # Compute omega
        omega = nx.diag(nx.dot(Q.T, CR_))

        # Rescale gamma at each iteration
        if gamma_init == "rescale":
            norm_1 = nx.max(nx.abs(CR_ * diag_g + reg * nx.log(Q))) ** 2
            norm_2 = nx.max(nx.abs(CQ_ * diag_g + reg * nx.log(R))) ** 2
            norm_3 = nx.max(nx.abs(-omega * diag_g)) ** 2
            gamma = 10 / max(norm_1, norm_2, norm_3)

        eps1 = nx.exp(-gamma * CR_g - ((gamma * reg) - 1) * nx.log(Q))
        eps2 = nx.exp(-gamma * CQ_g - ((gamma * reg) - 1) * nx.log(R))
        eps3 = nx.exp((gamma * omega / (g**2)) - (gamma * reg - 1) * nx.log(g))

        # LR Dykstra algorithm
        Q, R, g = _LR_Dysktra(
            eps1, eps2, eps3, a, b, alpha, stopThr, numItermax, warn, nx
        )
        Q = Q + 1e-16
        R = R + 1e-16
        g = g + 1e-16

    # ----------------- Compute lazy_plan, value and value_linear  ------------------
    # see "Section 3.2: The Low-rank OT Problem" in the paper

    # Compute lazy plan (using LazyTensor class)
    lazy_plan = get_lowrank_lazytensor(Q, R, 1 / g)

    # Compute value_linear (using trace formula)
    v1 = nx.dot(Q.T, M1)
    v2 = nx.dot(R, (v1.T * diag_g).T)
    value_linear = nx.sum(nx.diag(nx.dot(M2.T, v2)))

    # Compute value with entropy reg (see "Section 3.2" in the paper)
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
