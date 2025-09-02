# -*- coding: utf-8 -*-
"""
OT Barycenter Solvers
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Eloi Tanguy <eloi.tanguy@math.cnrs.fr>
#
# License: MIT License

from ..backend import get_backend
from ..utils import dist
from ._network_simplex import emd, emd2

import numpy as np
import scipy as sp
import scipy.sparse as sps

try:
    import cvxopt  # for cvxopt barycenter solver
    from cvxopt import solvers, matrix, spmatrix
except ImportError:
    cvxopt = False


def scipy_sparse_to_spmatrix(A):
    """Efficient conversion from scipy sparse matrix to cvxopt sparse matrix"""
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP


def barycenter(A, M, weights=None, verbose=False, log=False, solver="highs-ipm"):
    r"""Compute the Wasserstein barycenter of distributions A

     The function solves the following optimization problem [16]:

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i W_{1}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_1(\cdot,\cdot)` is the Wasserstein distance (see ot.emd.sinkhorn)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`

    The linear program is solved using the interior point solver from scipy.optimize.
    If cvxopt solver if installed it can use cvxopt

    Note that this problem do not scale well (both in memory and computational time).

    Parameters
    ----------
    A : np.ndarray (d,n)
        n training distributions a_i of size d
    M : np.ndarray (d,d)
        loss matrix for OT
    reg : float
        Regularization term >0
    weights : np.ndarray (n,)
        Weights of each histogram a_i on the simplex (barycentric coordinates)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    solver : string, optional
        the solver used, default 'interior-point' use the lp solver from
        scipy.optimize. None, or 'glpk' or 'mosek' use the solver from cvxopt.

    Returns
    -------
    a : (d,) ndarray
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [16] Agueh, M., & Carlier, G. (2011). Barycenters in the Wasserstein space. SIAM Journal on Mathematical Analysis, 43(2), 904-924.


    """

    if weights is None:
        weights = np.ones(A.shape[1]) / A.shape[1]
    else:
        assert len(weights) == A.shape[1]

    n_distributions = A.shape[1]
    n = A.shape[0]

    n2 = n * n
    c = np.zeros((0))
    b_eq1 = np.zeros((0))
    for i in range(n_distributions):
        c = np.concatenate((c, M.ravel() * weights[i]))
        b_eq1 = np.concatenate((b_eq1, A[:, i]))
    c = np.concatenate((c, np.zeros(n)))

    lst_idiag1 = [sps.kron(sps.eye(n), np.ones((1, n))) for i in range(n_distributions)]
    #  row constraints
    A_eq1 = sps.hstack(
        (sps.block_diag(lst_idiag1), sps.coo_matrix((n_distributions * n, n)))
    )

    # columns constraints
    lst_idiag2 = []
    lst_eye = []
    for i in range(n_distributions):
        if i == 0:
            lst_idiag2.append(sps.kron(np.ones((1, n)), sps.eye(n)))
            lst_eye.append(-sps.eye(n))
        else:
            lst_idiag2.append(sps.kron(np.ones((1, n)), sps.eye(n - 1, n)))
            lst_eye.append(-sps.eye(n - 1, n))

    A_eq2 = sps.hstack((sps.block_diag(lst_idiag2), sps.vstack(lst_eye)))
    b_eq2 = np.zeros((A_eq2.shape[0]))

    # full problem
    A_eq = sps.vstack((A_eq1, A_eq2))
    b_eq = np.concatenate((b_eq1, b_eq2))

    if not cvxopt or solver in ["interior-point", "highs", "highs-ipm", "highs-ds"]:
        # cvxopt not installed or interior point

        if solver is None:
            solver = "interior-point"

        options = {"disp": verbose}
        sol = sp.optimize.linprog(
            c, A_eq=A_eq, b_eq=b_eq, method=solver, options=options
        )
        x = sol.x
        b = x[-n:]

    else:
        h = np.zeros((n_distributions * n2 + n))
        G = -sps.eye(n_distributions * n2 + n)

        sol = solvers.lp(
            matrix(c),
            scipy_sparse_to_spmatrix(G),
            matrix(h),
            A=scipy_sparse_to_spmatrix(A_eq),
            b=matrix(b_eq),
            solver=solver,
        )

        x = np.array(sol["x"])
        b = x[-n:].ravel()

    if log:
        return b, sol
    else:
        return b


def free_support_barycenter(
    measures_locations,
    measures_weights,
    X_init,
    b=None,
    weights=None,
    numItermax=100,
    stopThr=1e-7,
    verbose=False,
    log=None,
    numThreads=1,
):
    r"""
    Solves the free support (locations of the barycenters are optimized, not the weights) Wasserstein barycenter problem (i.e. the weighted Frechet mean for the 2-Wasserstein distance), formally:

    .. math::
        \min_\mathbf{X} \quad \sum_{i=1}^N w_i W_2^2(\mathbf{b}, \mathbf{X}, \mathbf{a}_i, \mathbf{X}_i)

    where :

    - :math:`w \in \mathbb{(0, 1)}^{N}`'s are the barycenter weights and sum to one
    - `measure_weights` denotes the :math:`\mathbf{a}_i \in \mathbb{R}^{k_i}`: empirical measures weights (on simplex)
    - `measures_locations` denotes the :math:`\mathbf{X}_i \in \mathbb{R}^{k_i, d}`: empirical measures atoms locations
    - :math:`\mathbf{b} \in \mathbb{R}^{k}` is the desired weights vector of the barycenter

    This problem is considered in :ref:`[20] <references-free-support-barycenter>` (Algorithm 2).
    There are two differences with the following codes:

    - we do not optimize over the weights
    - we do not do line search for the locations updates, we use i.e. :math:`\theta = 1` in
      :ref:`[20] <references-free-support-barycenter>` (Algorithm 2). This can be seen as a discrete
      implementation of the fixed-point algorithm of
      :ref:`[43] <references-free-support-barycenter>` proposed in the continuous setting.

    Parameters
    ----------
    measures_locations : list of N (k_i,d) array-like
        The discrete support of a measure supported on :math:`k_i` locations of a `d`-dimensional space
        (:math:`k_i` can be different for each element of the list)
    measures_weights : list of N (k_i,) array-like
        Numpy arrays where each numpy array has :math:`k_i` non-negatives values summing to one
        representing the weights of each discrete input measure
    X_init : (k,d) array-like
        Initialization of the support locations (on `k` atoms) of the barycenter
    b : (k,) array-like
        Initialization of the weights of the barycenter (non-negatives, sum to 1)
    weights : (N,) array-like
        Initialization of the coefficients of the barycenter (non-negatives, sum to 1)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    numThreads: int or "max", optional (default=1, i.e. OpenMP is not used)
        If compiled with OpenMP, chooses the number of threads to parallelize.
        "max" selects the highest number possible.

    Returns
    -------
    X : (k,d) array-like
        Support locations (on k atoms) of the barycenter

    .. _references-free-support-barycenter:
    References
    ----------
    .. [20] Cuturi, Marco, and Arnaud Doucet. "Fast computation of Wasserstein barycenters."
        International Conference on Machine Learning. 2014.

    .. [43] Álvarez-Esteban, Pedro C., et al. "A fixed-point approach to barycenters in Wasserstein space."
        Journal of Mathematical Analysis and Applications 441.2 (2016): 744-762.

    """

    nx = get_backend(*measures_locations, *measures_weights, X_init)

    iter_count = 0

    N = len(measures_locations)
    k = X_init.shape[0]
    d = X_init.shape[1]
    if b is None:
        b = nx.ones((k,), type_as=X_init) / k
    if weights is None:
        weights = nx.ones((N,), type_as=X_init) / N

    X = X_init

    log_dict = {}
    displacement_square_norms = []

    displacement_square_norm = stopThr + 1.0

    while displacement_square_norm > stopThr and iter_count < numItermax:
        T_sum = nx.zeros((k, d), type_as=X_init)

        for measure_locations_i, measure_weights_i, weight_i in zip(
            measures_locations, measures_weights, weights
        ):
            M_i = dist(X, measure_locations_i)
            T_i = emd(b, measure_weights_i, M_i, numThreads=numThreads)
            T_sum = T_sum + weight_i * 1.0 / b[:, None] * nx.dot(
                T_i, measure_locations_i
            )

        displacement_square_norm = nx.sum((T_sum - X) ** 2)
        if log:
            displacement_square_norms.append(displacement_square_norm)

        X = T_sum

        if verbose:
            print(
                "iteration %d, displacement_square_norm=%f\n",
                iter_count,
                displacement_square_norm,
            )

        iter_count += 1

    if log:
        log_dict["displacement_square_norms"] = displacement_square_norms
        return X, log_dict
    else:
        return X


def generalized_free_support_barycenter(
    X_list,
    a_list,
    P_list,
    n_samples_bary,
    Y_init=None,
    b=None,
    weights=None,
    numItermax=100,
    stopThr=1e-7,
    verbose=False,
    log=None,
    numThreads=1,
    eps=0,
):
    r"""
    Solves the free support generalized Wasserstein barycenter problem: finding a barycenter (a discrete measure with
    a fixed amount of points of uniform weights) whose respective projections fit the input measures.
    More formally:

    .. math::
        \min_\gamma \quad \sum_{i=1}^p w_i W_2^2(\nu_i, \mathbf{P}_i\#\gamma)

    where :

    - :math:`\gamma = \sum_{l=1}^n b_l\delta_{y_l}` is the desired barycenter with each :math:`y_l \in \mathbb{R}^d`
    - :math:`\mathbf{b} \in \mathbb{R}^{n}` is the desired weights vector of the barycenter
    - The input measures are :math:`\nu_i = \sum_{j=1}^{k_i}a_{i,j}\delta_{x_{i,j}}`
    - The :math:`\mathbf{a}_i \in \mathbb{R}^{k_i}` are the respective empirical measures weights (on the simplex)
    - The :math:`\mathbf{X}_i \in \mathbb{R}^{k_i, d_i}` are the respective empirical measures atoms locations
    - :math:`w = (w_1, \cdots w_p)` are the barycenter coefficients (on the simplex)
    - Each :math:`\mathbf{P}_i \in \mathbb{R}^{d, d_i}`, and :math:`P_i\#\nu_i = \sum_{j=1}^{k_i}a_{i,j}\delta_{P_ix_{i,j}}`

    As show by :ref:`[42] <references-generalized-free-support-barycenter>`,
    this problem can be re-written as a Wasserstein Barycenter problem,
    which we solve using the free support method :ref:`[20] <references-generalized-free-support-barycenter>`
    (Algorithm 2).

    Parameters
    ----------
    X_list : list of p (k_i,d_i) array-like
        Discrete supports of the input measures: each consists of :math:`k_i` locations of a `d_i`-dimensional space
        (:math:`k_i` can be different for each element of the list)
    a_list : list of p (k_i,) array-like
        Measure weights: each element is a vector (k_i) on the simplex
    P_list : list of p (d_i,d) array-like
        Each :math:`P_i` is a linear map :math:`\mathbb{R}^{d} \rightarrow \mathbb{R}^{d_i}`
    n_samples_bary : int
        Number of barycenter points
    Y_init : (n_samples_bary,d) array-like
        Initialization of the support locations (on `k` atoms) of the barycenter
    b : (n_samples_bary,) array-like
        Initialization of the weights of the barycenter measure (on the simplex)
    weights : (p,) array-like
        Initialization of the coefficients of the barycenter (on the simplex)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    numThreads: int or "max", optional (default=1, i.e. OpenMP is not used)
        If compiled with OpenMP, chooses the number of threads to parallelize.
        "max" selects the highest number possible.
    eps: Stability coefficient for the change of variable matrix inversion
        If the :math:`\mathbf{P}_i^T` matrices don't span :math:`\mathbb{R}^d`, the problem is ill-defined and a matrix
        inversion will fail. In this case one may set eps=1e-8 and get a solution anyway (which may make little sense)


    Returns
    -------
    Y : (n_samples_bary,d) array-like
        Support locations (on n_samples_bary atoms) of the barycenter


    .. _references-generalized-free-support-barycenter:
    References
    ----------
    .. [20] Cuturi, M. and Doucet, A.. "Fast computation of Wasserstein barycenters."
        International Conference on Machine Learning. 2014.

    .. [42] Delon, J., Gozlan, N., and Saint-Dizier, A.. Generalized Wasserstein barycenters
        between probability measures living on different subspaces.
        arXiv preprint arXiv:2105.09755, 2021.

    """
    nx = get_backend(*X_list, *a_list, *P_list)
    d = P_list[0].shape[1]
    p = len(X_list)

    if weights is None:
        weights = nx.ones(p, type_as=X_list[0]) / p

    # variable change matrix to reduce the problem to a Wasserstein Barycenter (WB)
    A = eps * nx.eye(
        d, type_as=X_list[0]
    )  # if eps nonzero: will force the invertibility of A
    for P_i, lambda_i in zip(P_list, weights):
        A = A + lambda_i * P_i.T @ P_i
    B = nx.inv(nx.sqrtm(A))

    Z_list = [
        x @ Pi @ B.T for (x, Pi) in zip(X_list, P_list)
    ]  # change of variables -> (WB) problem on Z

    if Y_init is None:
        Y_init = nx.randn(n_samples_bary, d, type_as=X_list[0])

    if b is None:
        b = nx.ones(n_samples_bary, type_as=X_list[0]) / n_samples_bary  # not optimized

    out = free_support_barycenter(
        Z_list,
        a_list,
        Y_init,
        b,
        numItermax=numItermax,
        stopThr=stopThr,
        verbose=verbose,
        log=log,
        numThreads=numThreads,
    )

    if log:  # unpack
        Y, log_dict = out
    else:
        Y = out
        log_dict = None
    Y = Y @ B.T  # return to the Generalized WB formulation

    if log:
        return Y, log_dict
    else:
        return Y


def ot_barycenter_energy(measure_locations, measure_weights, X, a, cost_list, nx=None):
    r"""
    Computes the energy of the OT barycenter functional for a given barycenter
    support `X` and weights `a`: .. math::
        V(X, a) = \sum_{k=1}^K w_k \mathcal{T}_{c_k}(X, a, Y_k, b_k),

    where: - :math:`X` (n, d) is the barycenter support, - :math:`a` (n) is the
    barycenter weights, - :math:`Y_k` (m_k, d_k) is the k-th measure support
      (`measure_locations[k]`),
    - :math:`b_k` (m_k) is the k-th measure weights (`measure_weights[k]`),
    - :math:`c_k: \mathbb{R}^{n\times d}\times\mathbb{R}^{m_k\times d_k}
         \rightarrow \mathbb{R}_+^{n\times m_k}` is the k-th cost function
         (which computes the pairwise cost matrix)
    - :math:`\mathcal{T}_{c_k}(X, a, Y_k, b)` is the OT cost between the
      barycenter measure and the k-th measure with respect to the cost
      :math:`c_k`.
    The function computes :math:`V(X, a)` as defined above.

    Parameters
    ----------
    measure_locations : list of array-like
        List of K arrays of measure positions, each of shape (m_k, d_k).
    measure_weights : list of array-like
        List of K arrays of measure weights, each of shape (m_k).
    X : array-like
        Array of shape (n, d) representing barycenter points.
    a : array-like
        Array of shape (n,) representing barycenter weights.
    cost_list : list of callable or callable
        List of K cost functions :math:`c_k: \mathbb{R}^{n\times
        d}\times\mathbb{R}^{m_k\times d_k} \rightarrow \mathbb{R}_+^{n\times
        m_k}`. If cost_list is a single callable, the same cost is used K times.
    nx : backend, optional
        The backend to use.

    Returns
    -------
    V : float
        The value of the OT barycenter functional :math:`V(X, a)`.

    References
    ----------
    .. [77] Tanguy, Eloi and Delon, Julie and Gozlan, Nathaël (2024). Computing
        barycenters of Measures for Generic Transport Costs. arXiv preprint
        2501.04016 (2024)

    See Also
    --------
    ot.lp.free_support_barycenter_generic_costs : Free support solver for the
    associated barycenter problem.
    """
    if nx is None:
        nx = get_backend(*measure_locations, *measure_weights, X, a)
    K = len(measure_locations)
    if callable(cost_list):
        cost_list = [cost_list] * K
    V = 0
    for k in range(K):
        C_k = cost_list[k](X, measure_locations[k])
        V += emd2(a, measure_weights[k], C_k)
    return V


def free_support_barycenter_generic_costs(
    measure_locations,
    measure_weights,
    X_init,
    cost_list,
    ground_bary=None,
    a=None,
    numItermax=100,
    method="L2_barycentric_proj",
    stopThr=1e-5,
    log=False,
    ground_bary_lr=1e-2,
    ground_bary_numItermax=100,
    ground_bary_stopThr=1e-5,
    ground_bary_solver="SGD",
    clean_measure=False,
):
    r"""
    Solves the OT barycenter problem [77] for generic costs using the fixed
    point algorithm, iterating the ground barycenter function B on transport
    plans between the current barycenter and the measures.

    The problem finds an optimal barycenter support `X` of given size (n, d)
    (enforced by the initialisation), minimising a sum of pairwise transport
    costs for the costs :math:`c_k`:

    .. math::
        \min_{X} \sum_{k=1}^K \mathcal{T}_{c_k}(X, a, Y_k, b_k),

    where:

    - :math:`X` (n, d) is the barycenter support,
    - :math:`a` (n) is the (fixed) barycenter weights,
    - :math:`Y_k` (m_k, d_k) is the k-th measure support
      (`measure_locations[k]`),
    - :math:`b_k` (m_k) is the k-th measure weights (`measure_weights[k]`),
    - :math:`c_k: \mathbb{R}^{n\times d}\times\mathbb{R}^{m_k\times d_k}\rightarrow \mathbb{R}_+^{n\times m_k}` is the k-th cost function (which computes the pairwise cost matrix)
    - :math:`\mathcal{T}_{c_k}(X, a, Y_k, b)` is the OT cost between the barycenter measure and the k-th measure with respect to the cost :math:`c_k`:

    .. math::
        \mathcal{T}_{c_k}(X, a, Y_k, b_k) = \min_\pi \quad \langle \pi, c_k(X, Y_k) \rangle_F

        s.t. \ \pi \mathbf{1} = \mathbf{a}

             \pi^T \mathbf{1} = \mathbf{b_k}

             \pi \geq 0

    in other words, :math:`\mathcal{T}_{c_k}(X, a, Y_k, b_k)` is `ot.emd2(a, b_k, c_k(X, Y_k))`.

    The function :math:`B:\mathbb{R}^{n\times d_1}\times
    \cdots\times\mathbb{R}^{n\times d_K} \longrightarrow \mathbb{R}^{n\times d}`
    is an input to the solver. `B` computes solutions of the following
    minimisation problem given :math:`(Y_1, \cdots, Y_K) \in \mathbb{R}^{n\times
    d_1}\times\cdots\times\mathbb{R}^{n\times d_K}` (broadcasted along `n`):

    .. math::
        B(y_1, \cdots, y_K) = \mathrm{argmin}_{x \in \mathbb{R}^d} \sum_{k=1}^K c_k(x, y_k),

    The input function B takes a list of K arrays of shape (n, d_k) and returns
    an array of shape (n, d). For certain costs, :math:`B` can be computed
    explicitly, or through a numerical solver.

    This function implements two algorithms:

    - Algorithm 2 from [77] when `method=true_fixed_point` is used, which may
      increase the support size of the barycenter at each iteration, with a
      maximum final size of :math:`N_0 + T\sum_k n_k - TK` for T iterations and
      an initial support size of :math:`N_0`. The computation of the iterates is
      done using the North West Corner multi-marginal gluing method. This method
      has convergence guarantees [77].

    - Algorithm 3 from [77] when `method=L2_barycentric_proj` is used, which is
      a heuristic simplification which fixes the weights and support size of the
      barycenter by performing barycentric projections of the pair-wise OT
      matrices. This method is substantially faster than the first one, but does
      not have convergence guarantees. (Default)

    The implemented methods ([77] Algorithms 2 and 3), generalises [20] and [43]
    to general costs and includes convergence guarantees, including for discrete
    measures.

    Parameters
    ----------
    measure_locations : list of array-like
        List of K arrays of measure positions, each of shape (m_k, d_k).
    measure_weights : list of array-like
        List of K arrays of measure weights, each of shape (m_k).
    X_init : array-like
        Array of shape (n, d) representing initial barycenter points.
    cost_list : list of callable or callable
        List of K cost functions :math:`c_k: \mathbb{R}^{n\times
        d}\times\mathbb{R}^{m_k\times d_k} \rightarrow \mathbb{R}_+^{n\times
        m_k}`. If cost_list is a single callable, the same cost is used K times.
    ground_bary : callable or None, optional
        Function List(array(n, d_k)) -> array(n, d) accepting a list of K arrays
        of shape :math:`(n\times d_K)`, computing the ground barycenters
        (broadcasted over n). If not provided, done with Adam on PyTorch
        (requires PyTorch backend), inefficiently using the cost functions in
        `cost_list`.
    a : array-like, optional
        Array of shape (n,) representing weights of the barycenter
        measure.Defaults to uniform.
    numItermax : int, optional
        Maximum number of iterations (default is 100).
    method : str, optional
        Barycentre method: 'L2_barycentric_proj' (default) for Euclidean
        barycentric projection, or 'true_fixed_point' for iterates using the
        North West Corner multi-marginal gluing method.
    stopThr : float, optional
        If :math:`W_2^2(a_t, X_t, a_{t+1}, X_{t+1}) < \mathrm{stopThr} \times
        \frac{1}{n}\|X_t\|_2^2`, terminate (default is 1e-5).
    log : bool, optional
        Whether to return the log dictionary (default is False).
    ground_bary_lr : float, optional
        Learning rate for the ground barycenter solver (if auto is used).
    ground_bary_numItermax : int, optional
        Maximum number of iterations for the ground barycenter solver (if auto
        is used).
    ground_bary_stopThr : float, optional
        Stop threshold for the ground barycenter solver (if auto is used): stop
        if the energy decreases less than this value.
    ground_bary_solver : str, optional
        Solver for auto ground bary solver (torch SGD or Adam). Default is
        "SGD".
    clean_measure : bool, optional
        For method=='true_fixed_point', whether to clean the discrete measure
        (X, a) at each iteration to remove duplicate points and sum their
        weights (default is False).

    Returns
    -------
    X : array-like
        Array of shape (n, d) representing barycenter points.
    log_dict : list of array-like, optional
        log containing the exit status, list of iterations and list of
        displacements if log is True.

    References
    ----------
    .. [77] Tanguy, Eloi and Delon, Julie and Gozlan, Nathaël (2024). Computing
        barycenters of Measures for Generic Transport Costs. arXiv preprint
        2501.04016 (2024)

    .. [20] Cuturi, Marco, and Arnaud Doucet. "Fast computation of Wasserstein
        barycenters." International Conference on Machine Learning. 2014.

    .. [43] Álvarez-Esteban, Pedro C., et al. "A fixed-point approach to
        barycenters in Wasserstein space." Journal of Mathematical Analysis and
        Applications 441.2 (2016): 744-762.

    .. note:: For the case of the L2 cost :math:`c_k(x, y) = \|x-y\|_2^2`,
        the ground barycenter is simply the Euclidean barycenter, i.e.
        :math:`B(y_1, \cdots, y_K) = \sum_k w_k y_k`. In this case, we recover
        the free-support algorithm from [20].

    See Also
    --------
    ot.lp.free_support_barycenter : Free support solver for the case where
    :math:`c_k(x,y) = \lambda_k\|x-y\|_2^2`.

    ot.lp.generalized_free_support_barycenter : Free support solver for the case
    where :math:`c_k(x,y) = \|P_kx-y\|_2^2` with :math:`P_k` linear.

    ot.lp.NorthWestMMGluing : gluing method used in the `true_fixed_point`
    method.
    """
    assert method in [
        "L2_barycentric_proj",
        "true_fixed_point",
    ], "Method must be 'L2_barycentric_proj' or 'true_fixed_point'"
    nx = get_backend(X_init, measure_locations[0])
    K = len(measure_locations)
    n = X_init.shape[0]
    if a is None:
        a = nx.ones(n, type_as=X_init) / n
    if callable(cost_list):  # use the given cost for all K pairs
        cost_list = [cost_list] * K
    auto_ground_bary = False

    if ground_bary is None:
        auto_ground_bary = True
        assert str(nx) == "torch", (
            f"Backend {str(nx)} is not compatible with ground_bary=None, it"
            "must be provided if not using PyTorch backend"
        )
        try:
            import torch
            from torch.optim import Adam, SGD

            def ground_bary(y, x_init):
                x = x_init.clone().detach().requires_grad_(True)
                solver = Adam if ground_bary_solver == "Adam" else SGD
                opt = solver([x], lr=ground_bary_lr)
                loss_prev = None
                for i in range(ground_bary_numItermax):
                    opt.zero_grad()
                    # inefficient cost computation but compatible
                    # with the choice of cost_list[k] giving the cost matrix
                    loss = torch.sum(
                        torch.stack(
                            [torch.diag(cost_list[k](x, y[k])) for k in range(K)]
                        )
                    )
                    loss.backward()
                    opt.step()
                    if i == 0:
                        diff = ground_bary_stopThr + 1.0
                    else:
                        diff = loss_prev - loss.item()
                    loss_prev = loss.item()
                    if diff < ground_bary_stopThr:
                        break
                return x.detach()

        except ImportError:
            raise ImportError("PyTorch is required to use ground_bary=None")

    X_list = [X_init] if log else []  # store the iterations
    a_list = [a] if log and method == "true_fixed_point" else []
    X = X_init
    diff_list = []  # store energy differences
    V_list = []  # store energy values
    exit_status = "Max iterations reached"

    for i in range(numItermax):
        pi_list = [  # compute the pairwise transport plans
            emd(a, measure_weights[k], cost_list[k](X, measure_locations[k]))
            for k in range(K)
        ]
        if i == 0:  # compute initial energy
            V = ot_barycenter_energy(
                measure_locations, measure_weights, X, a, cost_list, nx=nx
            )
            if log:
                V_list.append(V)

        if method == "L2_barycentric_proj":
            a_next = a  # barycentre weights are fixed
            Y_perm = [
                (1 / a[:, None]) * pi_list[k] @ measure_locations[k] for k in range(K)
            ]  # L2 barycentric projection of pi_k
            if auto_ground_bary:  # use previous position as initialization
                X_next = ground_bary(Y_perm, X)
            else:
                X_next = ground_bary(Y_perm)

        elif method == "true_fixed_point":
            # North West Corner gluing of pi_k
            J, a_next = NorthWestMMGluing(pi_list, nx=nx)
            # J is a (N, K) array of indices, w is a (N,) array of weights
            # Each Y_perm[k] is a (N, d_k) array of some points in Y_list[k]
            Y_perm = [measure_locations[k][J[:, k]] for k in range(K)]
            # warm start impossible due to possible size mismatch
            X_next = ground_bary(Y_perm)

            if clean_measure and method == "true_fixed_point":
                # clean the discrete measure (X, a) to remove duplicates
                X_next, a_next = _clean_discrete_measure(X_next, a_next, nx=nx)

        V_next = ot_barycenter_energy(
            measure_locations, measure_weights, X_next, a_next, cost_list, nx=nx
        )
        diff = V - V_next
        if log:
            X_list.append(X_next)
            V_list.append(V_next)
            if method == "true_fixed_point":
                a_list.append(a_next)
            diff_list.append(diff)

        X = X_next
        a = a_next
        V = V_next

        if diff < stopThr:
            exit_status = "Stationary Point"
            break

    if log:
        log_dict = {
            "X_list": X_list,
            "exit_status": exit_status,
            "a_list": a_list,
            "diff_list": diff_list,
            "V_list": V_list,
        }
        if method == "true_fixed_point":
            return X, a, log_dict
        else:
            return X, log_dict

    if method == "true_fixed_point":
        return X, a
    else:
        return X


def _to_int_array(x, nx=None):
    """
    Converts an array to an integer type array.
    """
    if nx is None:
        nx = get_backend(x)
    if str(nx) == "numpy":
        return x.astype(int)

    if str(nx) == "torch":
        return x.to(int)

    if str(nx) == "jax":
        return x.astype(int)

    if str(nx) == "cupy":
        return x.astype(int)

    if str(nx) == "tf":
        import tensorflow as tf

        return tf.cast(x, tf.int32)


def NorthWestMMGluing(pi_list, a=None, log=False, nx=None):
    r"""
    Glue transport plans :math:`(\pi_1, ..., \pi_K)` which have a common first
    marginal using the (multi-marginal) North-West Corner method. Writing the
    marginals of each :math:`\pi_k\in \mathbb{R}^{n\times n_l}` as :math:`a \in
    \mathbb{R}^n` and :math:`b_k \in \mathbb{R}^{n_k}`, the output represents a
    particular K-marginal transport plan :math:`\rho \in
    \mathbb{R}^{n_1\times\cdots\times n_K}` whose k-th marginal is :math:`b_k`.
    This K-plan is such that there exists a K+1-marginal transport plan
    :math:`\gamma \in \mathbb{R}^{n\times n_1 \times \cdots \times n_K}` such
    that :math:`\sum_i\gamma_{i,j_1,\cdots,j_K} = \rho_{j_1, \cdots, j_K}` and
    with Einstein summation convention, :math:`\gamma_{i, j_1, \cdots, j_K} =
    [\pi_k]_{i, j_k}` for all :math:`k=1,\cdots,K`.

    Instead of outputting the full K-multi-marginal plan :math:`\rho`, this
    function provides an array `J` of shape (N, K) where each `J[i]` is of the
    form `(J[i, 1], ..., J[i, K])` with each `J[i, k]` between 0 and
    :math:`n_k-1`, and a weight vector `w` of size N, such that the K-plan
    :math:`\rho` writes:

    .. math::
        \rho_{j_1, \cdots, j_K} = \mathbb{1}\left(\exists i \text{ s.t. } (j_1, \cdots, j_K) = (J[i, 1], \cdots, J[i, K])\right)\ w_i.

    This representation is useful for its memory efficiency, as it avoids
    storing the full K-marginal plan.

    If `log=True`, the function computes the full K+1-marginal transport plan
    :math:`\gamma` and stores it in log_dict['gamma']. Note that this option is
    extremely costly in memory.

    Parameters
    ----------
    pi_list : list of arrays (n, n_k)
        List of transport plans.
    log : bool, optional
        If True, return a log dictionary (computationally expensive).
    a : array (n,), optional
        The common first marginal of each transport plan. If None is provided,
        it is computed as the first marginal of the first transport plan.
    nx : backend, optional
        The backend to use. If None is provided, the backend of the first
        transport plan is used.

    Returns
    -------
    J : array (N, K)
        The indices (J[i, 1], ..., J[i, K]) of the K-plan :math:`\rho`.
    w : array (N,)
        The weights w_i of the K-plan :math:`\rho`.
    log_dict : dict, optional
        If log=True, a dictionary containing the full K+1-marginal transport
        plan under the key 'gamma'.
    """
    if nx is None:
        nx = get_backend(pi_list[0])
    if a is None:
        a = nx.sum(pi_list[0], axis=1)  # common first marginal a in Delta_n
    nk_list = [pi.shape[1] for pi in pi_list]  # list of n_k
    K = len(pi_list)
    n = pi_list[0].shape[0]  # number of points in the first marginal
    gamma = None

    log_dict = {}
    if log:  # n x n_1 x ... x n_K tensor
        gamma = nx.zeros([n] + nk_list, type_as=pi_list[0])

    gamma_weights = {}  # dict of (j_1, ..., j_K) : weight
    P_list = [nx.copy(pi) for pi in pi_list]  # copy of the transport plans

    # jjs is a list of K lists of size m_k
    # checks if each jj_idx[k] is < m_k
    # this is to avoid over-shooting the while loop due to numerical
    # imprecision in the conditions "x > 0"
    def jj_idx_in_range(jj_idx, jjs):
        out = True
        for k in range(K):
            out = out and jj_idx[k] < len(jjs[k])
        return out

    for i in range(n):
        # jjs[k] is the list of indices j in [0, n_k - 1] such that Pk[i, j] >0
        jjs = [nx.to_numpy(nx.where(P[i, :] > 0)[0]) for P in P_list]
        # list [0, ..., 0] of size K for use with jjs: current indices in jjs
        jj_idx = [0] * K
        u = a[i]  # mass at i, will decrease to 0 as we fill gamma[i, :]

        # while there is mass to add to gamma[i, :]
        while u > 0 and jj_idx_in_range(jj_idx, jjs):
            # current multi-index j_1 ... j_K
            jj = tuple(jjs[k][jj_idx[k]] for k in range(K))
            # min transport plan value: min_k pi_k[i, j_k]
            v = nx.min(nx.stack([P_list[k][i, jj[k]] for k in range(K)]))
            if log:  # assign mass v to gamma[i, j_1, ..., j_K]
                gamma[(i,) + jj] = v
            if jj in gamma_weights:
                gamma_weights[jj] += v
            else:
                gamma_weights[jj] = v
            u -= v  # at i, we u-v mass left to assign
            for k in range(K):  # update plan copies Pk
                P_list[k][i, jj[k]] -= v  # Pk[i, j_k] has v less mass left
                if P_list[k][i, jj[k]] == 0:
                    # move to next index in jjs[k] if Pk[i, j_k] is empty
                    jj_idx[k] += 1

    log_dict["gamma"] = gamma
    J = list(gamma_weights.keys())  # list of multi-indices (j_1, ..., j_K)
    J = _to_int_array(nx.from_numpy(np.array(J), type_as=pi_list[0]), nx=nx)
    w = nx.stack(list(gamma_weights.values()))
    if log:
        return J, w, log_dict
    return J, w


def _clean_discrete_measure(X, a, tol=1e-10, nx=None):
    r"""
    Simplifies a discrete measure by consolidating duplicate points and summing
    their weights. Given a discrete measure with support X (n, d) and weights a
    (n), returns a points Y (m, d) and weights b (m) such that Y is the set of
    unique points in X and b is the sum of weights in a for each point in Y

    Parameters
    ----------
    X : array-like
        Array of shape (n, d) representing the support points of the discrete
        measure.
    a : array-like
        Array of shape (n,) representing the weights associated with the support
        points.
    tol : float, optional
        Tolerance for determining uniqueness of points in `X`. Points closer
        than `tol` are considered identical. Default is 1e-10.
    nx : backend, optional
        The backend to use. If None is provided, the backend of `X` is used

    Returns
    -------
    Y : array-like
        Array of shape (m, d) representing the unique support points of the
        discrete measure.
    b : array-like
        Array of shape (m,) representing the summed weights for each unique
        point in `Y`.
    """
    nx = get_backend(X, a)
    D = dist(X, X)
    # each D[I[k], J[k]] < tol so X[I[k]] = X[J[k]]
    idxI, idxJ = nx.where(D < tol)
    idxI = nx.to_numpy(idxI)
    idxJ = nx.to_numpy(idxJ)
    # keep only the cases I[k] <= J[k] to avoid pairs (i, j) (j, i) with i != j
    mask = idxI <= idxJ
    idxI, idxJ = idxI[mask], idxJ[mask]
    X_idx_to_Y_idx = {}  # X[i] = Y[X_idx_to_Y_idx[i]]
    # indices of unique points in X, at the end, Y := X[unique_X_idx]
    unique_X_idx = []

    b = []
    for i, j in zip(idxI, idxJ):
        if i not in X_idx_to_Y_idx:  # i is a new point
            unique_X_idx.append(i)
            X_idx_to_Y_idx[i] = len(unique_X_idx) - 1
            b.append(a[i])

        else:  # i is not new, check if j is known
            if j not in X_idx_to_Y_idx:
                b[X_idx_to_Y_idx[i]] += a[j]
                X_idx_to_Y_idx[j] = X_idx_to_Y_idx[i]

    # create the unique points array Y
    Y = X[tuple(unique_X_idx), :]
    b = nx.from_numpy(np.array(b), type_as=X)
    return Y, b
