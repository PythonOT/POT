# -*- coding: utf-8 -*-
"""
Optimal Transport maps and variants

.. warning::
    Note that by default the module is not imported in :mod:`ot`. In order to
    use it you need to explicitly import :mod:`ot.mapping`
"""

# Author: Eloi Tanguy <eloi.tanguy@math.cnrs.fr>
#         Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

from .backend import get_backend, to_numpy
from .lp import emd
import numpy as np

from .optim import cg
from .utils import dist, unif, list_to_array, kernel, dots


def nearest_brenier_potential_fit(
    X,
    V,
    X_classes=None,
    a=None,
    b=None,
    strongly_convex_constant=0.6,
    gradient_lipschitz_constant=1.4,
    its=100,
    log=False,
    init_method="barycentric",
    solver=None,
):
    r"""
    Computes optimal values and gradients at X for a strongly convex potential :math:`\varphi` with Lipschitz gradients
    on the partitions defined by `X_classes`, where :math:`\varphi` is optimal such that
    :math:`\nabla \varphi \#\mu \approx \nu`, given samples :math:`X = x_1, \cdots, x_n \sim \mu` and
    :math:`V = v_1, \cdots, v_n \sim \nu`. Finding such a potential that has the desired regularity on the
    partition :math:`(E_k)_{k \in [K]}` (given by the classes `X_classes`) is equivalent to finding optimal values
    `phi` for the :math:`\varphi(x_i)` and its gradients :math:`\nabla \varphi(x_i)` (variable`G`).
    In practice, these optimal values are found by solving the following problem

    .. math::
        :nowrap:

        \begin{gather*}
        \text{min} \sum_{i,j}\pi_{i,j}\|g_i - v_j\|_2^2 \\
        g_1,\cdots, g_n \in \mathbb{R}^d,\; \varphi_1, \cdots, \varphi_n \in \mathbb{R},\; \pi \in \Pi(a, b) \\
        \text{s.t.}\ \forall k \in [K],\; \forall i,j \in I_k: \\
        \varphi_i-\varphi_j-\langle g_j, x_i-x_j\rangle \geq c_1\|g_i - g_j\|_2^2
        + c_2\|x_i-x_j\|_2^2 - c_3\langle g_j-g_i, x_j -x_i \rangle.
        \end{gather*}

    The constants :math:`c_1, c_2, c_3` only depend on `strongly_convex_constant` and `gradient_lipschitz_constant`.
    The constraint :math:`\pi \in \Pi(a, b)` denotes the fact that the matrix :math:`\pi` belong to the OT polytope
    of marginals a and b. :math:`I_k` is the subset of :math:`[n]` of the i such that :math:`x_i` is in the
    partition (or class) :math:`E_k`, i.e. `X_classes[i] == k`.

    This problem is solved by alternating over the variable :math:`\pi` and the variables :math:`\varphi_i, g_i`.
    For :math:`\pi`, the problem is the standard discrete OT problem, and for :math:`\varphi_i, g_i`, the
    problem is a convex QCQP solved using :code:`cvxpy` (ECOS solver).

    Accepts any compatible backend, but will perform the QCQP optimisation on Numpy arrays, and convert back at the end.

    .. warning:: This function requires the CVXPY library
    .. warning:: Accepts any backend but will convert to Numpy then back to the backend.

    Parameters
    ----------
    X : array-like (n, d)
        reference points used to compute the optimal values phi and G
    V : array-like (n, d)
        values of the gradients at the reference points X
    X_classes : array-like (n,), optional
        classes of the reference points, defaults to a single class
    a : array-like (n,), optional
        weights for the reference points X, defaults to uniform
    b : array-like (n,), optional
        weights for the target points V, defaults to uniform
    strongly_convex_constant : float, optional
        constant for the strong convexity of the input potential phi, defaults to 0.6
    gradient_lipschitz_constant : float, optional
        constant for the Lipschitz property of the input gradient G, defaults to 1.4
    its: int, optional
        number of iterations, defaults to 100
    log : bool, optional
        record log if true
    init_method : str, optional
        'target' initialises G=V, 'barycentric' initialises at the image of X by
        the barycentric projection
    solver : str, optional
        The CVXPY solver to use

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

    See Also
    --------
    ot.mapping.nearest_brenier_potential_predict_bounds : Predicting SSNB images on new source data
    ot.da.NearestBrenierPotential : BaseTransport wrapper for SSNB

    """
    try:
        import cvxpy as cvx
    except ImportError:
        print("Please install CVXPY to use this function")
        return
    assert (
        X.shape == V.shape
    ), f"point shape should be the same as value shape, yet {X.shape} != {V.shape}"
    nx = get_backend(X, V, X_classes, a, b)
    X, V = to_numpy(X), to_numpy(V)
    n, d = X.shape
    if X_classes is not None:
        X_classes = to_numpy(X_classes)
        assert X_classes.size == n, "incorrect number of class items"
    else:
        X_classes = np.zeros(n)
    a = unif(n) if a is None else nx.to_numpy(a)
    b = unif(n) if b is None else nx.to_numpy(b)
    assert a.shape[-1] == b.shape[-1] == n, "incorrect measure weight sizes"

    assert init_method in [
        "target",
        "barycentric",
    ], f"Unsupported initialization method '{init_method}'"
    if init_method == "target":
        G_val = V
    else:  # Init G_val with barycentric projection
        G_val = emd(a, b, dist(X, V)) @ V / a.reshape(n, 1)
    phi_val = None
    log_dict = {"G_list": [], "phi_list": [], "its": []}

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
        c1, c2, c3 = _ssnb_qcqp_constants(
            strongly_convex_constant, gradient_lipschitz_constant
        )

        for k in np.unique(X_classes):  # constraints for the convex interpolation
            for i in np.where(X_classes == k)[0]:
                for j in np.where(X_classes == k)[0]:
                    constraints += [
                        phi[i]
                        >= phi[j]
                        + G[j].T @ (X[i] - X[j])
                        + c1 * cvx.sum_squares(G[i] - G[j])
                        + c2 * cvx.sum_squares(X[i] - X[j])
                        - c3 * (G[j] - G[i]).T @ (X[j] - X[i])
                    ]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=solver)
        phi_val, G_val = phi.value, G.value
        it_log_dict = {
            "solve_time": problem.solver_stats.solve_time,
            "setup_time": problem.solver_stats.setup_time,
            "num_iters": problem.solver_stats.num_iters,
            "status": problem.status,
            "value": problem.value,
        }
        if log:
            log_dict["its"].append(it_log_dict)
            log_dict["G_list"].append(G_val)
            log_dict["phi_list"].append(phi_val)

    # convert back to backend
    phi_val = nx.from_numpy(phi_val)
    G_val = nx.from_numpy(G_val)
    if not log:
        return phi_val, G_val
    return phi_val, G_val, log_dict


def _ssnb_qcqp_constants(strongly_convex_constant, gradient_lipschitz_constant):
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
    assert (
        0 < strongly_convex_constant < gradient_lipschitz_constant
    ), "incompatible regularity assumption"
    c = 1 / (2 * (1 - strongly_convex_constant / gradient_lipschitz_constant))
    c1 = c / gradient_lipschitz_constant
    c2 = strongly_convex_constant * c
    c3 = 2 * strongly_convex_constant * c / gradient_lipschitz_constant
    return c1, c2, c3


def nearest_brenier_potential_predict_bounds(
    X,
    phi,
    G,
    Y,
    X_classes=None,
    Y_classes=None,
    strongly_convex_constant=0.6,
    gradient_lipschitz_constant=1.4,
    log=False,
    solver=None,
):
    r"""
    Compute the values of the lower and upper bounding potentials at the input points Y, using the potential optimal
    values phi at X and their gradients G at X. The 'lower' potential corresponds to the method from :ref:`[58]`,
    Equation 2, while the bounding property and 'upper' potential come from :ref:`[59]`, Theorem 3.14 (taking into
    account the fact that this theorem's statement has a min instead of a max, which is a typo). Both potentials are
    optimal for the SSNB problem.

    If :math:`I_k` is the subset of :math:`[n]` of the i such that :math:`x_i` is in the partition (or class)
    :math:`E_k`, for each :math:`y \in E_k`, this function solves the convex QCQP problems,
    respectively for l: 'lower' and u: 'upper':

    .. math::
        :nowrap:

        \begin{gather*}
        (\varphi_{l}(x), \nabla \varphi_l(x)) = \text{argmin}\ t, \\
        t\in \mathbb{R},\; g\in \mathbb{R}^d, \\
        \text{s.t.} \forall j \in I_k,\; t-\varphi_j - \langle g_j, y-x_j \rangle \geq c_1\|g - g_j\|_2^2
        + c_2\|y-x_j\|_2^2 - c_3\langle g_j-g, x_j -y \rangle.
        \end{gather*}

    .. math::
        :nowrap:

        \begin{gather*}
        (\varphi_{u}(x), \nabla \varphi_u(x)) = \text{argmax}\ t, \\
        t\in \mathbb{R},\; g\in \mathbb{R}^d, \\
        \text{s.t.} \forall i \in I_k,\; \varphi_i^* -t - \langle g, x_i-y \rangle \geq c_1\|g_i - g\|_2^2
        + c_2\|x_i-y\|_2^2 - c_3\langle g-g_i, y -x_i \rangle.
        \end{gather*}

    The constants :math:`c_1, c_2, c_3` only depend on `strongly_convex_constant` and `gradient_lipschitz_constant`.

    .. warning:: This function requires the CVXPY library
    .. warning:: Accepts any backend but will convert to Numpy then back to the backend.

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
    solver : str, optional
        The CVXPY solver to use

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

    See Also
    --------
    ot.mapping.nearest_brenier_potential_fit : Fitting the SSNB on source and target data
    ot.da.NearestBrenierPotential : BaseTransport wrapper for SSNB

    """
    try:
        import cvxpy as cvx
    except ImportError:
        print("Please install CVXPY to use this function")
        return
    nx = get_backend(X, phi, G, Y)
    X = to_numpy(X)
    phi = to_numpy(phi)
    G = to_numpy(G)
    Y = to_numpy(Y)
    m, d = Y.shape
    if Y_classes is not None:
        Y_classes = to_numpy(Y_classes)
        assert Y_classes.size == m, "wrong number of class items for Y"
    else:
        Y_classes = np.zeros(m)
    assert (
        X.shape[1] == d
    ), f"incompatible dimensions between X: {X.shape} and Y: {Y.shape}"
    n, _ = X.shape
    if X_classes is not None:
        X_classes = to_numpy(X_classes)
        assert X_classes.size == n, "incorrect number of class items"
    else:
        X_classes = np.zeros(n)
    assert X_classes.size == n, "wrong number of class items for X"
    c1, c2, c3 = _ssnb_qcqp_constants(
        strongly_convex_constant, gradient_lipschitz_constant
    )
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
                phi_l_y
                >= phi[j]
                + G[j].T @ (Y[y_idx] - X[j])
                + c1 * cvx.sum_squares(G_l_y - G[j])
                + c2 * cvx.sum_squares(Y[y_idx] - X[j])
                - c3 * (G[j] - G_l_y).T @ (X[j] - Y[y_idx])
            ]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=solver)
        phi_lu[0, y_idx] = phi_l_y.value
        G_lu[0, y_idx] = G_l_y.value
        if log:
            log_item["l"] = {
                "solve_time": problem.solver_stats.solve_time,
                "setup_time": problem.solver_stats.setup_time,
                "num_iters": problem.solver_stats.num_iters,
                "status": problem.status,
                "value": problem.value,
            }

        # upper bound
        phi_u_y = cvx.Variable(1)
        G_u_y = cvx.Variable(d)
        objective = cvx.Maximize(phi_u_y)
        constraints = []
        for i in np.where(X_classes == k)[0]:
            constraints += [
                phi[i]
                >= phi_u_y
                + G_u_y.T @ (X[i] - Y[y_idx])
                + c1 * cvx.sum_squares(G[i] - G_u_y)
                + c2 * cvx.sum_squares(X[i] - Y[y_idx])
                - c3 * (G_u_y - G[i]).T @ (Y[y_idx] - X[i])
            ]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=solver)
        phi_lu[1, y_idx] = phi_u_y.value
        G_lu[1, y_idx] = G_u_y.value
        if log:
            log_item["u"] = {
                "solve_time": problem.solver_stats.solve_time,
                "setup_time": problem.solver_stats.setup_time,
                "num_iters": problem.solver_stats.num_iters,
                "status": problem.status,
                "value": problem.value,
            }
            log_dict[y_idx] = log_item

    phi_lu, G_lu = nx.from_numpy(phi_lu), nx.from_numpy(G_lu)
    if not log:
        return phi_lu, G_lu
    return phi_lu, G_lu, log_dict


def joint_OT_mapping_linear(
    xs,
    xt,
    mu=1,
    eta=0.001,
    bias=False,
    verbose=False,
    verbose2=False,
    numItermax=100,
    numInnerItermax=10,
    stopInnerThr=1e-6,
    stopThr=1e-5,
    log=False,
    **kwargs,
):
    r"""Joint OT and linear mapping estimation as proposed in
    :ref:`[8] <references-joint-OT-mapping-linear>`.

    The function solves the following optimization problem:

    .. math::
        \min_{\gamma,L}\quad \|L(\mathbf{X_s}) - n_s\gamma \mathbf{X_t} \|^2_F +
          \mu \langle \gamma, \mathbf{M} \rangle_F + \eta \|L - \mathbf{I}\|^2_F

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) squared euclidean cost matrix between samples in
      :math:`\mathbf{X_s}` and :math:`\mathbf{X_t}` (scaled by :math:`n_s`)
    - :math:`L` is a :math:`d\times d` linear operator that approximates the barycentric
      mapping
    - :math:`\mathbf{I}` is the identity matrix (neutral linear mapping)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are uniform source and target weights

    The problem consist in solving jointly an optimal transport matrix
    :math:`\gamma` and a linear mapping that fits the barycentric mapping
    :math:`n_s\gamma \mathbf{X_t}`.

    One can also estimate a mapping with constant bias (see supplementary
    material of :ref:`[8] <references-joint-OT-mapping-linear>`) using the bias optional argument.

    The algorithm used for solving the problem is the block coordinate
    descent that alternates between updates of :math:`\mathbf{G}` (using conditional gradient)
    and the update of :math:`\mathbf{L}` using a classical least square solver.


    Parameters
    ----------
    xs : array-like (ns,d)
        samples in the source domain
    xt : array-like (nt,d)
        samples in the target domain
    mu : float,optional
        Weight for the linear OT loss (>0)
    eta : float, optional
        Regularization term  for the linear mapping L (>0)
    bias : bool,optional
        Estimate linear mapping with constant bias
    numItermax : int, optional
        Max number of BCD iterations
    stopThr : float, optional
        Stop threshold on relative loss decrease (>0)
    numInnerItermax : int, optional
        Max number of iterations (inner CG solver)
    stopInnerThr : float, optional
        Stop threshold on error (inner CG solver) (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns, nt) array-like
        Optimal transportation matrix for the given parameters
    L : (d, d) array-like
        Linear mapping matrix ((:math:`d+1`, `d`) if bias)
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-joint-OT-mapping-linear:
    References
    ----------
    .. [8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
        "Mapping estimation for discrete optimal transport",
        Neural Information Processing Systems (NIPS), 2016.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    xs, xt = list_to_array(xs, xt)
    nx = get_backend(xs, xt)

    ns, nt, d = xs.shape[0], xt.shape[0], xt.shape[1]

    if bias:
        xs1 = nx.concatenate((xs, nx.ones((ns, 1), type_as=xs)), axis=1)
        xstxs = nx.dot(xs1.T, xs1)
        Id = nx.eye(d + 1, type_as=xs)
        Id[-1] = 0
        I0 = Id[:, :-1]

        def sel(x):
            return x[:-1, :]
    else:
        xs1 = xs
        xstxs = nx.dot(xs1.T, xs1)
        Id = nx.eye(d, type_as=xs)
        I0 = Id

        def sel(x):
            return x

    if log:
        log = {"err": []}

    a = unif(ns, type_as=xs)
    b = unif(nt, type_as=xt)
    M = dist(xs, xt) * ns
    G = emd(a, b, M)

    vloss = []

    def loss(L, G):
        """Compute full loss"""
        return (
            nx.sum((nx.dot(xs1, L) - ns * nx.dot(G, xt)) ** 2)
            + mu * nx.sum(G * M)
            + eta * nx.sum(sel(L - I0) ** 2)
        )

    def solve_L(G):
        """solve L problem with fixed G (least square)"""
        xst = ns * nx.dot(G, xt)
        return nx.solve(xstxs + eta * Id, nx.dot(xs1.T, xst) + eta * I0)

    def solve_G(L, G0):
        """Update G with CG algorithm"""
        xsi = nx.dot(xs1, L)

        def f(G):
            return nx.sum((xsi - ns * nx.dot(G, xt)) ** 2)

        def df(G):
            return -2 * ns * nx.dot(xsi - ns * nx.dot(G, xt), xt.T)

        G = cg(
            a,
            b,
            M,
            1.0 / mu,
            f,
            df,
            G0=G0,
            numItermax=numInnerItermax,
            stopThr=stopInnerThr,
        )
        return G

    L = solve_L(G)

    vloss.append(loss(L, G))

    if verbose:
        print(
            "{:5s}|{:12s}|{:8s}".format("It.", "Loss", "Delta loss") + "\n" + "-" * 32
        )
        print("{:5d}|{:8e}|{:8e}".format(0, vloss[-1], 0))

    # init loop
    if numItermax > 0:
        loop = 1
    else:
        loop = 0
    it = 0

    while loop:
        it += 1

        # update G
        G = solve_G(L, G)

        # update L
        L = solve_L(G)

        vloss.append(loss(L, G))

        if it >= numItermax:
            loop = 0

        if abs(vloss[-1] - vloss[-2]) / abs(vloss[-2]) < stopThr:
            loop = 0

        if verbose:
            if it % 20 == 0:
                print(
                    "{:5s}|{:12s}|{:8s}".format("It.", "Loss", "Delta loss")
                    + "\n"
                    + "-" * 32
                )
            print(
                "{:5d}|{:8e}|{:8e}".format(
                    it, vloss[-1], (vloss[-1] - vloss[-2]) / abs(vloss[-2])
                )
            )
    if log:
        log["loss"] = vloss
        return G, L, log
    else:
        return G, L


def joint_OT_mapping_kernel(
    xs,
    xt,
    mu=1,
    eta=0.001,
    kerneltype="gaussian",
    sigma=1,
    bias=False,
    verbose=False,
    verbose2=False,
    numItermax=100,
    numInnerItermax=10,
    stopInnerThr=1e-6,
    stopThr=1e-5,
    log=False,
    **kwargs,
):
    r"""Joint OT and nonlinear mapping estimation with kernels as proposed in
    :ref:`[8] <references-joint-OT-mapping-kernel>`.

    The function solves the following optimization problem:

    .. math::
        \min_{\gamma, L\in\mathcal{H}}\quad \|L(\mathbf{X_s}) -
        n_s\gamma \mathbf{X_t}\|^2_F + \mu \langle \gamma, \mathbf{M} \rangle_F +
        \eta \|L\|^2_\mathcal{H}

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0


    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) squared euclidean cost matrix between samples in
      :math:`\mathbf{X_s}` and :math:`\mathbf{X_t}` (scaled by :math:`n_s`)
    - :math:`L` is a :math:`n_s \times d` linear operator on a kernel matrix that
      approximates the barycentric mapping
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are uniform source and target weights

    The problem consist in solving jointly an optimal transport matrix
    :math:`\gamma` and the nonlinear mapping that fits the barycentric mapping
    :math:`n_s\gamma \mathbf{X_t}`.

    One can also estimate a mapping with constant bias (see supplementary
    material of :ref:`[8] <references-joint-OT-mapping-kernel>`) using the bias optional argument.

    The algorithm used for solving the problem is the block coordinate
    descent that alternates between updates of :math:`\mathbf{G}` (using conditional gradient)
    and the update of :math:`\mathbf{L}` using a classical kernel least square solver.


    Parameters
    ----------
    xs : array-like (ns,d)
        samples in the source domain
    xt : array-like (nt,d)
        samples in the target domain
    mu : float,optional
        Weight for the linear OT loss (>0)
    eta : float, optional
        Regularization term  for the linear mapping L (>0)
    kerneltype : str,optional
        kernel used by calling function :py:func:`ot.utils.kernel` (gaussian by default)
    sigma : float, optional
        Gaussian kernel bandwidth.
    bias : bool,optional
        Estimate linear mapping with constant bias
    verbose : bool, optional
        Print information along iterations
    verbose2 : bool, optional
        Print information along iterations
    numItermax : int, optional
        Max number of BCD iterations
    numInnerItermax : int, optional
        Max number of iterations (inner CG solver)
    stopInnerThr : float, optional
        Stop threshold on error (inner CG solver) (>0)
    stopThr : float, optional
        Stop threshold on relative loss decrease (>0)
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns, nt) array-like
        Optimal transportation matrix for the given parameters
    L : (ns, d) array-like
        Nonlinear mapping matrix ((:math:`n_s+1`, `d`) if bias)
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-joint-OT-mapping-kernel:
    References
    ----------
    .. [8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
        "Mapping estimation for discrete optimal transport",
        Neural Information Processing Systems (NIPS), 2016.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    xs, xt = list_to_array(xs, xt)
    nx = get_backend(xs, xt)

    ns, nt = xs.shape[0], xt.shape[0]

    K = kernel(xs, xs, method=kerneltype, sigma=sigma)
    if bias:
        K1 = nx.concatenate((K, nx.ones((ns, 1), type_as=xs)), axis=1)
        Id = nx.eye(ns + 1, type_as=xs)
        Id[-1] = 0
        Kp = nx.eye(ns + 1, type_as=xs)
        Kp[:ns, :ns] = K

        # ls regu
        # K0 = K1.T.dot(K1)+eta*I
        # Kreg=I

        # RKHS regul
        K0 = nx.dot(K1.T, K1) + eta * Kp
        Kreg = Kp

    else:
        K1 = K
        Id = nx.eye(ns, type_as=xs)

        # ls regul
        # K0 = K1.T.dot(K1)+eta*I
        # Kreg=I

        # proper kernel ridge
        K0 = K + eta * Id
        Kreg = K

    if log:
        log = {"err": []}

    a = unif(ns, type_as=xs)
    b = unif(nt, type_as=xt)
    M = dist(xs, xt) * ns
    G = emd(a, b, M)

    vloss = []

    def loss(L, G):
        """Compute full loss"""
        return (
            nx.sum((nx.dot(K1, L) - ns * nx.dot(G, xt)) ** 2)
            + mu * nx.sum(G * M)
            + eta * nx.trace(dots(L.T, Kreg, L))
        )

    def solve_L_nobias(G):
        """solve L problem with fixed G (least square)"""
        xst = ns * nx.dot(G, xt)
        return nx.solve(K0, xst)

    def solve_L_bias(G):
        """solve L problem with fixed G (least square)"""
        xst = ns * nx.dot(G, xt)
        return nx.solve(K0, nx.dot(K1.T, xst))

    def solve_G(L, G0):
        """Update G with CG algorithm"""
        xsi = nx.dot(K1, L)

        def f(G):
            return nx.sum((xsi - ns * nx.dot(G, xt)) ** 2)

        def df(G):
            return -2 * ns * nx.dot(xsi - ns * nx.dot(G, xt), xt.T)

        G = cg(
            a,
            b,
            M,
            1.0 / mu,
            f,
            df,
            G0=G0,
            numItermax=numInnerItermax,
            stopThr=stopInnerThr,
        )
        return G

    if bias:
        solve_L = solve_L_bias
    else:
        solve_L = solve_L_nobias

    L = solve_L(G)

    vloss.append(loss(L, G))

    if verbose:
        print(
            "{:5s}|{:12s}|{:8s}".format("It.", "Loss", "Delta loss") + "\n" + "-" * 32
        )
        print("{:5d}|{:8e}|{:8e}".format(0, vloss[-1], 0))

    # init loop
    if numItermax > 0:
        loop = 1
    else:
        loop = 0
    it = 0

    while loop:
        it += 1

        # update G
        G = solve_G(L, G)

        # update L
        L = solve_L(G)

        vloss.append(loss(L, G))

        if it >= numItermax:
            loop = 0

        if abs(vloss[-1] - vloss[-2]) / abs(vloss[-2]) < stopThr:
            loop = 0

        if verbose:
            if it % 20 == 0:
                print(
                    "{:5s}|{:12s}|{:8s}".format("It.", "Loss", "Delta loss")
                    + "\n"
                    + "-" * 32
                )
            print(
                "{:5d}|{:8e}|{:8e}".format(
                    it, vloss[-1], (vloss[-1] - vloss[-2]) / abs(vloss[-2])
                )
            )
    if log:
        log["loss"] = vloss
        return G, L, log
    else:
        return G, L
