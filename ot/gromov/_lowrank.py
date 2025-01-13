"""
Low rank Gromov-Wasserstein solver
"""

# Author: Laurène David <laurene.david@ip-paris.fr>
#
# License: MIT License

import warnings
from ..utils import unif, get_lowrank_lazytensor
from ..backend import get_backend
from ..lowrank import compute_lr_sqeuclidean_matrix, _init_lr_sinkhorn, _LR_Dysktra


def _flat_product_operator(X, nx=None):
    r"""
    Implementation of the flattened out-product operator.

    This function is used in low rank gromov wasserstein to compute the low rank decomposition of
    a cost matrix's squared hadamard product (page 6 in paper).

    Parameters
    ----------
    X: array-like, shape (n_samples, n_col)
        Input matrix for operator

    nx: default None
        POT backend

    Returns
    ----------
    X_flat: array-like, shape (n_samples, n_col**2)
        Matrix with flattened out-product operator applied on each row

    References
    ----------
    .. [67] Scetbon, M., Peyré, G. & Cuturi, M. (2022).
        "Linear-Time GromovWasserstein Distances using Low Rank Couplings and Costs".
        In International Conference on Machine Learning (ICML), 2022.

    """

    if nx is None:
        nx = get_backend(X)

    n = X.shape[0]
    x1 = X[0, :][:, None]
    X_flat = nx.dot(x1, x1.T).flatten()[:, None]

    for i in range(1, n):
        x = X[i, :][:, None]
        x_out = nx.dot(x, x.T).flatten()[:, None]
        X_flat = nx.concatenate((X_flat, x_out), axis=1)

    X_flat = X_flat.T

    return X_flat


def lowrank_gromov_wasserstein_samples(
    X_s,
    X_t,
    a=None,
    b=None,
    reg=0,
    rank=None,
    alpha=1e-10,
    gamma_init="rescale",
    rescale_cost=True,
    cost_factorized_Xs=None,
    cost_factorized_Xt=None,
    stopThr=1e-4,
    numItermax=1000,
    stopThr_dykstra=1e-3,
    numItermax_dykstra=10000,
    seed_init=49,
    warn=True,
    warn_dykstra=False,
    log=False,
):
    r"""
    Solve the entropic regularization Gromov-Wasserstein transport problem under low-nonnegative rank constraints
    on the couplings and cost matrices.

    Squared euclidean distance matrices are considered for the target and source distributions.

    The function solves the following optimization problem:

    .. math::
        \mathop{\min_{(Q,R,g) \in \mathcal{C(a,b,r)}}} \mathcal{Q}_{A,B}(Q\mathrm{diag}(1/g)R^T) -
            \epsilon \cdot H((Q,R,g))

    where :

    - :math:`A` is the (`dim_a`, `dim_a`) square pairwise cost matrix of the source domain.
    - :math:`B` is the (`dim_a`, `dim_a`) square pairwise cost matrix of the target domain.
    - :math:`\mathcal{Q}_{A,B}` is quadratic objective function of the Gromov Wasserstein plan.
    - :math:`Q` and `R` are the low-rank matrix decomposition of the Gromov-Wasserstein plan.
    - :math:`g` is the weight vector for the low-rank decomposition of the Gromov-Wasserstein plan.
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (histograms, both sum to 1).
    - :math:`r` is the rank of the Gromov-Wasserstein plan.
    - :math:`\mathcal{C(a,b,r)}` are the low-rank couplings of the OT problem.
    - :math:`H((Q,R,g))` is the values of the three respective entropies evaluated for each term.


    Parameters
    ----------
    X_s : array-like, shape (n_samples_a, dim_Xs)
        Samples in the source domain
    X_t : array-like, shape (n_samples_b, dim_Xt)
        Samples in the target domain
    a : array-like, shape (n_samples_a,), optional
        Samples weights in the source domain
        If let to its default value None, uniform distribution is taken.
    b : array-like, shape (n_samples_b,), optional
        Samples weights in the target domain
        If let to its default value None, uniform distribution is taken.
    reg : float, optional
        Regularization term >=0
    rank : int, optional. Default is None. (>0)
        Nonnegative rank of the OT plan. If None, min(ns, nt) is considered.
    alpha : int, optional. Default is 1e-10. (>0 and <1/r)
        Lower bound for the weight vector g.
    rescale_cost : bool, optional. Default is False
        Rescale the low rank factorization of the sqeuclidean cost matrix
    seed_init : int, optional. Default is 49. (>0)
        Random state for the 'random' initialization of low rank couplings
    gamma_init : str, optional. Default is "rescale".
        Initialization strategy for gamma. 'rescale', or 'theory'
        Gamma is a constant that scales the convergence criterion of the Mirror Descent
        optimization scheme used to compute the low-rank couplings (Q, R and g)
    numItermax : int, optional. Default is 1000.
        Max number of iterations for Low Rank GW
    stopThr : float, optional. Default is 1e-4.
        Stop threshold on error (>0) for Low Rank GW
        The error is the sum of Kullback Divergences computed for each low rank
        coupling (Q, R and g) and scaled using gamma.
    numItermax_dykstra : int, optional. Default is 2000.
        Max number of iterations for the Dykstra algorithm
    stopThr_dykstra : float, optional. Default is 1e-7.
        Stop threshold on error (>0) in Dykstra
    cost_factorized_Xs: tuple, optional. Default is None
        Tuple with two pre-computed low rank decompositions (A1, A2) of the source cost
        matrix. Both matrices should have a shape of (n_samples_a, dim_Xs + 2).
        If None, the low rank cost matrices will be computed as sqeuclidean cost matrices.
    cost_factorized_Xt: tuple, optional. Default is None
        Tuple with two pre-computed low rank decompositions (B1, B2) of the target cost
        matrix. Both matrices should have a shape of (n_samples_b, dim_Xt + 2).
        If None, the low rank cost matrices will be computed as sqeuclidean cost matrices.
    warn : bool, optional
        if True, raises a warning if the low rank GW algorithm doesn't convergence.
    warn_dykstra: bool, optional
        if True, raises a warning if the Dykstra algorithm doesn't convergence.
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
    .. [67] Scetbon, M., Peyré, G. & Cuturi, M. (2022).
        "Linear-Time GromovWasserstein Distances using Low Rank Couplings and Costs".
        In International Conference on Machine Learning (ICML), 2022.

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
        raise ValueError(
            "alpha ({a}) should be smaller than 1/rank ({r}) for the Dykstra algorithm to converge.".format(
                a=alpha, r=1 / rank
            )
        )

    if cost_factorized_Xs is not None:
        A1, A2 = cost_factorized_Xs
    else:
        A1, A2 = compute_lr_sqeuclidean_matrix(X_s, X_s, rescale_cost, nx=nx)

    if cost_factorized_Xt is not None:
        B1, B2 = cost_factorized_Xt
    else:
        B1, B2 = compute_lr_sqeuclidean_matrix(X_t, X_t, rescale_cost, nx=nx)

    # Initial values for LR couplings (Q, R, g) with LOT
    Q, R, g = _init_lr_sinkhorn(
        X_s, X_t, a, b, r, init="random", random_state=seed_init, reg_init=None, nx=nx
    )

    # Gamma initialization
    if gamma_init == "theory":
        L = (27 * nx.norm(A1) * nx.norm(A2)) / alpha**4
        gamma = 1 / (2 * L)

    if gamma_init not in ["rescale", "theory"]:
        raise (
            NotImplementedError('Not implemented gamma_init="{}"'.format(gamma_init))
        )

    # initial value of error
    err = 1

    for ii in range(numItermax):
        Q_prev = Q
        R_prev = R
        g_prev = g

        if err > stopThr:
            # Compute cost matrices
            C1 = nx.dot(A2.T, Q * (1 / g)[None, :])
            C1 = -4 * nx.dot(A1, C1)
            C2 = nx.dot(R.T, B1)
            C2 = nx.dot(C2, B2.T)
            diag_g = (1 / g)[None, :]

            # Compute C*R dot using the lr decomposition of C
            CR = nx.dot(C2, R)
            CR = nx.dot(C1, CR)
            CR_g = CR * diag_g

            # Compute C.T * Q using the lr decomposition of C
            CQ = nx.dot(C1.T, Q)
            CQ = nx.dot(C2.T, CQ)
            CQ_g = CQ * diag_g

            # Compute omega
            omega = nx.diag(nx.dot(Q.T, CR))

            # Rescale gamma at each iteration
            if gamma_init == "rescale":
                norm_1 = nx.max(nx.abs(CR_g + reg * nx.log(Q))) ** 2
                norm_2 = nx.max(nx.abs(CQ_g + reg * nx.log(R))) ** 2
                norm_3 = nx.max(nx.abs(-omega * (diag_g**2))) ** 2
                gamma = 10 / max(norm_1, norm_2, norm_3)

            K1 = nx.exp(-gamma * CR_g - ((gamma * reg) - 1) * nx.log(Q))
            K2 = nx.exp(-gamma * CQ_g - ((gamma * reg) - 1) * nx.log(R))
            K3 = nx.exp((gamma * omega / (g**2)) - (gamma * reg - 1) * nx.log(g))

            # Update couplings with LR Dykstra algorithm
            Q, R, g = _LR_Dysktra(
                K1,
                K2,
                K3,
                a,
                b,
                alpha,
                stopThr_dykstra,
                numItermax_dykstra,
                warn_dykstra,
                nx,
            )

            # Update error with kullback-divergence
            err_1 = ((1 / gamma) ** 2) * (nx.kl_div(Q, Q_prev) + nx.kl_div(Q_prev, Q))
            err_2 = ((1 / gamma) ** 2) * (nx.kl_div(R, R_prev) + nx.kl_div(R_prev, R))
            err_3 = ((1 / gamma) ** 2) * (nx.kl_div(g, g_prev) + nx.kl_div(g_prev, g))
            err = err_1 + err_2 + err_3

            # fix divide by zero
            Q = Q + 1e-16
            R = R + 1e-16
            g = g + 1e-16

        else:
            break

    else:
        if warn:
            warnings.warn(
                "Low Rank GW did not converge. You might want to "
                "increase the number of iterations `numItermax` "
            )

    # Update low rank costs
    C1 = nx.dot(A2.T, Q * (1 / g)[None, :])
    C1 = -4 * nx.dot(A1, C1)
    C2 = nx.dot(R.T, B1)
    C2 = nx.dot(C2, B2.T)

    # Compute lazy plan (using LazyTensor class)
    lazy_plan = get_lowrank_lazytensor(Q, R, 1 / g)

    # Compute value_quad
    A1_, A2_ = _flat_product_operator(A1, nx), _flat_product_operator(A2, nx)
    B1_, B2_ = _flat_product_operator(B1, nx), _flat_product_operator(B2, nx)

    x_ = nx.dot(A1_, nx.dot(A2_.T, a))
    y_ = nx.dot(B1_, nx.dot(B2_.T, b))
    c1 = nx.dot(x_, a) + nx.dot(y_, b)

    G = nx.dot(C1, nx.dot(C2, R))
    G = nx.dot(Q.T, G * diag_g)
    value_quad = c1 + nx.trace(G) / 2

    if reg != 0:
        reg_Q = nx.sum(Q * nx.log(Q + 1e-16))  # entropy for Q
        reg_g = nx.sum(g * nx.log(g + 1e-16))  # entropy for g
        reg_R = nx.sum(R * nx.log(R + 1e-16))  # entropy for R
        value = value_quad + reg * (reg_Q + reg_g + reg_R)
    else:
        value = value_quad

    if log:
        dict_log = dict()
        dict_log["value"] = value
        dict_log["value_quad"] = value_quad
        dict_log["lazy_plan"] = lazy_plan

        return Q, R, g, dict_log

    return Q, R, g
