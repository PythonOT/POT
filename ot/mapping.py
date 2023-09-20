# -*- coding: utf-8 -*-
"""
Optimal Transport maps and variants
"""

# Author: Eloi Tanguy <eloi.tanguy@u-paris.fr>
#
# License: MIT License

from .backend import get_backend
from .lp import emd
import numpy as np
from .utils import dist, unif


def nearest_brenier_potential(X, V, X_classes=None, a=None, b=None, strongly_convex_constant=.6,
                              gradient_lipschitz_constant=1.4, its=100, log=False, seed=None):
    r"""
    Computes optimal values and gradients at X for a strongly convex potential :math:`\\varphi` with Lipschitz gradients
    on the partitions defined by `X_classes`, where :math:`\\varphi` is optimal such that
    :math:`\\nabla \\varphi \#\\mu \\approx \\nu`, given samples :math:`X = x_1, \\cdots, x_n \\sim \\mu` and
    :math:`V = v_1, \\cdots, v_n \\sim \\nu`. Finding such a potential that has the desired regularity on the
    partition :math:`(E_k)_{k \in [K]}` (given by the classes `X_classes`) is equivalent to finding optimal values
    `phi` for the :math:`\\varphi(x_i)` and its gradients :math:`\\nabla \\varphi(x_i)` (variable`G`).
    In practice, these optimal values are found by solving the following problem

    .. math::
        \\text{min} \\sum_{i,j}\\pi_{i,j}\|g_i - v_j\|_2^2

         g_1,\\cdots, g_n \in \mathbb{R}^d,\; \\varphi_1, \\cdots, \\varphi_n \in \mathbb{R},\; \pi \in \Pi(a, b)

         \\text{s.t.}\ \\forall k \in [K],\; \\forall i,j \in I_k:

		\\varphi_i-\\varphi_j-\langle g_j, x_i-x_j\\rangle \geq c_1\|g_i - g_j\|_2^2 +
		c_2\|x_i-x_j\|_2^2 - c_3\langle g_j-g_i, x_j -x_i \\rangle.

    The constants :math:`c_1, c_2, c_3` only depend on `strongly_convex_constant` and `gradient_lipschitz_constant`.
    The constraint :math:`\pi \in \Pi(a, b)` denotes the fact that the matrix :math:`\pi` belong to the OT polytope
    of marginals a and b. :math:`I_k` is the subset of :math:`[n]` of the i such that :math:`x_i` is in the
    partition (or class) :math:`E_k`, i.e. `X_classes[i] == k`.

    This problem is solved by alternating over the variable :math:`\pi` and the variables :math:`\\varphi_i, g_i`.
    For :math:`\pi`, the problem is the standard discrete OT problem, and for :math:`\\varphi_i, g_i`, the
    problem is a convex QCQP solved using :code:`cvxpy` (ECOS solver).

    Parameters
    ----------
    X: array-like (n, d)
        reference points used to compute the optimal values phi and G
    V: array-like (n, d)
        values of the gradients at the reference points X
    X_classes : array-like (n,), optional
        classes of the reference points, defaults to a single class
    a: array-like (n,), optional
        weights for the reference points X, defaults to uniform
    b: array-like (n,), optional
        weights for the target points V, defaults to uniform
    strongly_convex_constant : float, optional
        constant for the strong convexity of the input potential phi, defaults to 0.6
    gradient_lipschitz_constant : float, optional
        constant for the Lipschitz property of the input gradient G, defaults to 1.4
    its: int, optional
        number of iterations, defaults to 100
    pbar: bool, optional
        if True show a progress bar, defaults to False
    log : bool, optional
        record log if true
    seed: int or RandomState or None, optional
        Seed used for random number generator

    Returns
    -------
    phi : array-like (n,)
        optimal values of the potential at the points X
    G : array-like (n, d)
        optimal values of the gradients at the points X
    log : dict, optional
        If input log is true, a dictionary containing the values of the variables at each iteration, as well
        as solver information

    References
    ----------

    .. [58] François-Pierre Paty, Alexandre d’Aspremont, and Marco Cuturi. Regularity as regularization:
            Smooth and strongly convex brenier potentials in optimal transport. In International Conference
            on Artificial Intelligence and Statistics, pages 1222–1232. PMLR, 2020.

    """
    assert X.shape == V.shape, f"point shape should be the same as value shape, yet {X.shape} != {V.shape}"
    if X_classes is not None and a is None and b is None:
        nx = get_backend(X, V, X_classes)
    if X_classes is None and a is not None and b is None:
        nx = get_backend(X, V, a)
    else:
        nx = get_backend(X, V)
    assert 0 <= strongly_convex_constant <= gradient_lipschitz_constant, "incompatible regularity assumption"
    n, d = X.shape
    if X_classes is not None:
        assert X_classes.size == n, "incorrect number of class items"
    else:
        X_classes = nx.zeros(n)
    if a is None:
        a = ot.unif(n)
    if b is None:
        b = ot.unif(n)
    assert a.size == b.size == n, 'incorrect measure weight sizes'

    if isinstance(seed, np.random.RandomState) and str(nx) == 'numpy':
        G = np.random.randn(n, d)
    else:
        if seed is not None:
            nx.seed(seed)
        G = nx.randn(n, d)

    phi = None
    log_dict = {
        'G_list': [],
        'phi_list': [],
        'its': []
    }

    for _ in range(its):  # alternate optimisation iterations
        cost_matrix = dist(G, V)
        # optimise the plan
        plan = emd(a, b, cost_matrix)
        # optimise the values phi and the gradients G
        out = solve_nearest_brenier_potential_qcqp(plan, X, X_classes, V,
                                                   strongly_convex_constant, gradient_lipschitz_constant, log)
        if not log:
            phi, G = out
        else:
            phi, G, it_log_dict = out
            log_dict['its'].append(it_log_dict)
            log_dict['G_list'].append(G)
            log_dict['phi_list'].append(phi)

    if not log:
        return phi, G
    return phi, G, log_dict


def qcqp_constants(strongly_convex_constant, gradient_lipschitz_constant):
    r"""
    Handy function computing the constants for the Nearest Brenier Potential QCQP problems

    Parameters
    ----------
    strongly_convex_constant : float
    gradient_lipschitz_constant : float

    Returns
    -------
    c1 : float
    c2 : float
    c3 : float

    """
    c = 1 / (2 * (1 - strongly_convex_constant / gradient_lipschitz_constant))
    c1 = c / gradient_lipschitz_constant
    c2 = strongly_convex_constant * c
    c3 = 2 * strongly_convex_constant * c / gradient_lipschitz_constant
    return c1, c2, c3


def solve_nearest_brenier_potential_qcqp(plan, X, X_classes, V, strongly_convex_constant=0.6,
                                         gradient_lipschitz_constant=1.4, log=False):
    r"""
    Solves the QCQP problem from `nearest_brenier_potential`, using the method from :ref:`[58]`.

    Parameters
    ----------
    plan : array-like (n, n)
        fixed OT plan matrix
    X: array-like (n, d)
        reference points used to compute the optimal values phi and G
    X_classes : array-like (n,)
        classes of the reference points
    V: array-like (n, d)
        values of the gradients at the reference points X
    strongly_convex_constant : float, optional
        constant for the strong convexity of the input potential phi, defaults to 0.6
    gradient_lipschitz_constant : float, optional
        constant for the Lipschitz property of the input gradient G, defaults to 1.4
    log : bool, optional
        record log if true

    Returns
    -------
    phi : array-like (n,)
        optimal values of the potential at the points X
    G : array-like (n, d)
        optimal values of the gradients at the points X
    log : dict, optional
        If input log is true, a dictionary containing solver information

    References
    ----------

    .. [58] François-Pierre Paty, Alexandre d’Aspremont, and Marco Cuturi. Regularity as regularization:
            Smooth and strongly convex brenier potentials in optimal transport. In International Conference
            on Artificial Intelligence and Statistics, pages 1222–1232. PMLR, 2020.

    """
    try:
        import cvxpy as cvx
    except ImportError:
        print('Please install CVXPY to use this function')
    assert X.shape == V.shape, f"point shape should be the same as value shape, yet {X.shape} != {V.shape}"
    assert 0 <= strongly_convex_constant <= gradient_lipschitz_constant, "incompatible regularity assumption"
    n, d = X.shape
    assert X_classes.size == n, "incorrect number of class items"
    assert plan.shape == (n, n), f'plan should be of shape {(n, n)} but is of shape {plan.shape}'
    phi = cvx.Variable(n)
    G = cvx.Variable((n, d))
    constraints = []
    cost = 0
    for i in range(n):
        for j in range(n):
            cost += cvx.sum_squares(G[i, :] - V[j, :]) * plan[i, j]
    objective = cvx.Minimize(cost)  # OT cost
    c1, c2, c3 = qcqp_constants(strongly_convex_constant, gradient_lipschitz_constant)

    for k in np.unique(X_classes):  # constraints for the convex interpolation
        for i in np.where(X_classes == k)[0]:
            for j in np.where(X_classes == k)[0]:
                constraints += [
                    phi[i] >= phi[j] + G[j].T @ (X[i] - X[j]) + c1 * cvx.sum_squares(G[i] - G[j]) \
                    + c2 * cvx.sum_squares(X[i] - X[j]) - c3 * (G[j] - G[i]).T @ (X[j] - X[i])
                ]
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=cvx.ECOS)

    if not log:
        return phi.value, G.value
    log_dict = {
        'solve_time': problem.solver_stats.solve_time,
        'setup_time': problem.solver_stats.setup_time,
        'num_iters': problem.solver_stats.num_iters,
        'status': problem.status,
        'value': problem.value
    }
    return phi.value, G.value, log_dict


def bounding_potentials_from_point_values(X, X_classes, phi, G, Y, Y_classes, strongly_convex_constant=0.6,
                                          gradient_lipschitz_constant=1.4, log=False):
    r"""
    Compute the values of the lower and upper bounding potentials at the input points Y, using the potential optimal
    values phi at X and their gradients G at X. The 'lower' potential corresponds to the method from :ref:`[58]`,
    Equation 2, while the bounding property and 'upper' potential come from :ref:`[59]`, Theorem 3.14 (taking into
    account the fact that this theorem's statement has a min instead of a max, which is a typo).

    If :math:`I_k` is the subset of :math:`[n]` of the i such that :math:`x_i` is in the partition (or class)
    :math:`E_k`, for each :math:`y \in E_k`, this function solves the convex QCQP problems,
    respectively for l: 'lower' and u: 'upper:

    .. math::
        (\\varphi_{l}(x), \\nabla \\varphi_l(x)) = \\text{argmin}\ t,

        t\in \mathbb{R},\; g\in \mathbb{R}^d,

		\\text{s.t.} \\forall j \in I_k,\; t-\\varphi_j - \langle g_j, y-x_j \\rangle \geq c_1\|g - g_j\|_2^2
		+ c_2\|y-x_j\|_2^2 - c_3\langle g_j-g, x_j -y \\rangle.

    .. math::
        (\\varphi_{u}(x), \\nabla \\varphi_u(x)) = \\text{argmax}\ t,

        t\in \mathbb{R},\; g\in \mathbb{R}^d,

		\\text{s.t.} \\forall i \in I_k,\; \\varphi_i^* -t - \langle g, x_i-y \\rangle \geq c_1\|g_i - g\|_2^2
		+ c_2\|x_i-y\|_2^2 - c_3\langle g-g_i, y -x_i \\rangle.

    The constants :math:`c_1, c_2, c_3` only depend on `strongly_convex_constant` and `gradient_lipschitz_constant`.

    Parameters
    ----------
    X : array-like (n, d)
        reference points used to compute the optimal values phi and G
    X_classes : array-like (n,)
        classes of the reference points
    phi : array-like (n,)
        optimal values of the potential at the points X
    G : array-like (n, d)
        optimal values of the gradients at the points X
    Y : array-like (m, d)
        input points
    Y_classes : array_like (m)
        classes of the input points
    strongly_convex_constant : float, optional
        constant for the strong convexity of the input potential phi, defaults to 0.6
    gradient_lipschitz_constant : float, optional
        constant for the Lipschitz property of the input gradient G, defaults to 1.4
    log : bool, optional
        record log if true

    Returns
    -------
        phi_lu: array-like (2, m)
            values of the lower and upper bounding potentials at Y
        G_lu: array-like (2, m, d)
            gradients of the lower and upper bounding potentials at Y
        log : dict, optional
            If input log is true, a dictionary containing solver information

    References
    ----------

    .. [58] François-Pierre Paty, Alexandre d’Aspremont, and Marco Cuturi. Regularity as regularization:
            Smooth and strongly convex brenier potentials in optimal transport. In International Conference
            on Artificial Intelligence and Statistics, pages 1222–1232. PMLR, 2020.

    .. [59] Adrien B Taylor. Convex interpolation and performance estimation of first-order methods for
            convex optimization. PhD thesis, Catholic University of Louvain, Louvain-la-Neuve, Belgium,
            2017.

    """
    try:
        import cvxpy as cvx
    except ImportError:
        print('Please install CVXPY to use this function')
    m, d = Y.shape
    assert Y_classes.size == m, 'wrong number of class items for Y'
    assert X.shape[1] == d, f'incompatible dimensions between X: {X.shape} and Y: {Y.shape}'
    n, _ = X.shape
    assert X_classes.size == n, 'wrong number of class items for X'
    c1, c2, c3 = qcqp_constants(strongly_convex_constant, gradient_lipschitz_constant)
    phi_lu = np.zeros((2, m))
    G_lu = np.zeros((2, m, d))
    log_dict = {}

    for y_idx in range(m):
        log_item = {}
        # lower bound
        phi_l_y = cvx.Variable(1)
        G_l_y = cvx.Variable(d)
        objective = cvx.Minimize(phi_l_y)
        constraints = []
        k = Y_classes[y_idx]
        for j in np.where(X_classes == k)[0]:
            constraints += [
                phi_l_y >= phi[j] + G[j].T @ (Y[y_idx] - X[j]) + c1 * cvx.sum_squares(G_l_y - G[j]) \
                + c2 * cvx.sum_squares(Y[y_idx] - X[j]) - c3 * (G[j] - G_l_y).T @ (X[j] - Y[y_idx])
            ]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=cvx.ECOS)
        phi_lu[0, y_idx] = phi_l_y.value
        G_lu[0, y_idx] = G_l_y.value
        if log:
            log_item['l'] = {
                'solve_time': problem.solver_stats.solve_time,
                'setup_time': problem.solver_stats.setup_time,
                'num_iters': problem.solver_stats.num_iters,
                'status': problem.status,
                'value': problem.value
            }

        # upper bound
        phi_u_y = cvx.Variable(1)
        G_u_y = cvx.Variable(d)
        objective = cvx.Maximize(phi_u_y)
        constraints = []
        for i in np.where(X_classes == k)[0]:
            constraints += [
                phi[i] >= phi_u_y + G_u_y.T @ (X[i] - Y[y_idx]) + c1 * cvx.sum_squares(G[i] - G_u_y) \
                + c2 * cvx.sum_squares(X[i] - Y[y_idx]) - c3 * (G_u_y - G[i]).T @ (Y[y_idx] - X[i])
            ]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=cvx.ECOS)
        phi_lu[1, y_idx] = phi_u_y.value
        G_lu[1, y_idx] = G_u_y.value
        if log:
            log_item['u'] = {
                'solve_time': problem.solver_stats.solve_time,
                'setup_time': problem.solver_stats.setup_time,
                'num_iters': problem.solver_stats.num_iters,
                'status': problem.status,
                'value': problem.value
            }
            log_dict[y_idx] = log_item

    if not log:
        return phi_lu, G_lu
    return phi_lu, G_lu, log_dict

