# -*- coding: utf-8 -*-
"""
Optimal Transport maps and variants
"""

# Author: Eloi Tanguy <eloi.tanguy@u-paris.fr>
#
# License: MIT License

from .backend import get_backend, to_numpy
from .lp import emd
import numpy as np
from .utils import dist, unif


def nearest_brenier_potential_fit(X, V, X_classes=None, a=None, b=None, strongly_convex_constant=.6,
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

    Accepts any compatible backend, but will perform the QCQP optimisation on Numpy arrays, and convert back at the end.

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
    try:
        import cvxpy as cvx
    except ImportError:
        print('Please install CVXPY to use this function')
        return
    assert X.shape == V.shape, f"point shape should be the same as value shape, yet {X.shape} != {V.shape}"
    nx = get_backend(X, V, X_classes, a, b)
    assert 0 <= strongly_convex_constant <= gradient_lipschitz_constant, "incompatible regularity assumption"
    X, V = to_numpy(X), to_numpy(V)
    n, d = X.shape
    if X_classes is not None:
        assert X_classes.size == n, "incorrect number of class items"
    else:
        X_classes = nx.zeros(n)
    if a is None:
        a = unif(n, type_as=X)
    if b is None:
        b = unif(n, type_as=X)
    assert a.size == b.size == n, 'incorrect measure weight sizes'

    if isinstance(seed, np.random.RandomState):
        G_val = np.random.randn(n, d)
    else:
        if seed is not None:
            np.random.seed(seed)
        G_val = np.random.randn(n, d)

    phi_val = None
    log_dict = {
        'G_list': [],
        'phi_list': [],
        'its': []
    }

    for _ in range(its):  # alternate optimisation iterations
        cost_matrix = dist(G_val, V)
        # optimise the plan
        plan = emd(a, b, cost_matrix)

        # optimise the values phi and the gradients G
        phi = cvx.Variable(n)
        G = cvx.Variable((n, d))
        constraints = []
        cost = 0
        for i in range(n):
            for j in range(n):
                cost += cvx.sum_squares(G[i, :] - V[j, :]) * plan[i, j]
        objective = cvx.Minimize(cost)  # OT cost
        c1, c2, c3 = ssnb_qcqp_constants(strongly_convex_constant, gradient_lipschitz_constant)

        for k in nx.unique(X_classes):  # constraints for the convex interpolation
            for i in nx.where(X_classes == k)[0]:
                for j in nx.where(X_classes == k)[0]:
                    constraints += [
                        phi[i] >= phi[j] + G[j].T @ (X[i] - X[j]) + c1 * cvx.sum_squares(G[i] - G[j]) \
                        + c2 * cvx.sum_squares(X[i] - X[j]) - c3 * (G[j] - G[i]).T @ (X[j] - X[i])
                    ]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=cvx.ECOS)
        phi_val, G_val = phi.value, G.value
        it_log_dict = {
            'solve_time': problem.solver_stats.solve_time,
            'setup_time': problem.solver_stats.setup_time,
            'num_iters': problem.solver_stats.num_iters,
            'status': problem.status,
            'value': problem.value
        }
        if log:
            log_dict['its'].append(it_log_dict)
            log_dict['G_list'].append(G_val)
            log_dict['phi_list'].append(phi_val)

    # convert back to backend
    phi_val = nx.from_numpy(phi_val)
    G_val = nx.from_numpy(G_val)
    if not log:
        return phi_val, G_val
    return phi_val, G_val, log_dict


def ssnb_qcqp_constants(strongly_convex_constant, gradient_lipschitz_constant):
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


def nearest_brenier_potential_predict_bounds(X, phi, G, Y, X_classes=None, Y_classes=None,
                                             strongly_convex_constant=0.6, gradient_lipschitz_constant=1.4, log=False):
    r"""
    Compute the values of the lower and upper bounding potentials at the input points Y, using the potential optimal
    values phi at X and their gradients G at X. The 'lower' potential corresponds to the method from :ref:`[58]`,
    Equation 2, while the bounding property and 'upper' potential come from :ref:`[59]`, Theorem 3.14 (taking into
    account the fact that this theorem's statement has a min instead of a max, which is a typo).

    If :math:`I_k` is the subset of :math:`[n]` of the i such that :math:`x_i` is in the partition (or class)
    :math:`E_k`, for each :math:`y \in E_k`, this function solves the convex QCQP problems,
    respectively for l: 'lower' and u: 'upper':

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
    X_classes : array-like (n,), optional
        classes of the reference points, defaults to a single class
    Y_classes : array_like (m,), optional
        classes of the input points, defaults to a single class
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
        return
    nx = get_backend(X, phi, G, Y)
    X = to_numpy(X)
    phi = to_numpy(phi)
    G = to_numpy(G)
    Y = to_numpy(Y)
    m, d = Y.shape
    assert Y_classes.size == m, 'wrong number of class items for Y'
    assert X.shape[1] == d, f'incompatible dimensions between X: {X.shape} and Y: {Y.shape}'
    n, _ = X.shape
    if X_classes is not None:
        assert X_classes.size == n, "incorrect number of class items"
    else:
        X_classes = nx.zeros(n)
    assert X_classes.size == n, 'wrong number of class items for X'
    c1, c2, c3 = ssnb_qcqp_constants(strongly_convex_constant, gradient_lipschitz_constant)
    phi_lu = nx.zeros((2, m))
    G_lu = nx.zeros((2, m, d))
    log_dict = {}

    for y_idx in range(m):
        log_item = {}
        # lower bound
        phi_l_y = cvx.Variable(1)
        G_l_y = cvx.Variable(d)
        objective = cvx.Minimize(phi_l_y)
        constraints = []
        k = Y_classes[y_idx]
        for j in nx.where(X_classes == k)[0]:
            constraints += [
                phi_l_y >= phi[j] + G[j].T @ (Y[y_idx] - X[j]) + c1 * cvx.sum_squares(G_l_y - G[j]) \
                + c2 * cvx.sum_squares(Y[y_idx] - X[j]) - c3 * (G[j] - G_l_y).T @ (X[j] - Y[y_idx])
            ]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=cvx.ECOS)
        phi_lu[0, y_idx] = nx.from_numpy(phi_l_y.value, type_as=X)
        G_lu[0, y_idx] = nx.from_numpy(G_l_y.value, type_as=X)
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
        for i in nx.where(X_classes == k)[0]:
            constraints += [
                phi[i] >= phi_u_y + G_u_y.T @ (X[i] - Y[y_idx]) + c1 * cvx.sum_squares(G[i] - G_u_y) \
                + c2 * cvx.sum_squares(X[i] - Y[y_idx]) - c3 * (G_u_y - G[i]).T @ (Y[y_idx] - X[i])
            ]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=cvx.ECOS)
        phi_lu[1, y_idx] = nx.from_numpy(phi_u_y.value, type_as=X)
        G_lu[1, y_idx] = nx.from_numpy(G_u_y.value, type_as=X)
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
