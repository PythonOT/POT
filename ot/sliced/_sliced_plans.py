# -*- coding: utf-8 -*-
"""
Sliced Wasserstein plans approximation solvers: min-pivot sliced (=sliced
Wasserstein Generalized Geodesic) and expected sliced.
"""

# Author: Eloi Tanguy <eloi.tanguy@math.cnrs.fr>
#         Laetitia Chapel <laetitia.chapel@irisa.fr>
#
# License: MIT License

import warnings

from ..backend import get_backend
from ..utils import list_to_array, sparse_ot_dist, dist
from ._utils import get_random_projections
from ..lp import wasserstein_1d
from collections import namedtuple


def sliced_plans(
    X_s,
    X_t,
    a=None,
    b=None,
    metric="sqeuclidean",
    p=1,
    thetas=None,
    warm_theta=None,
    n_proj=None,
    log=False,
):
    r"""
    Computes all the permutations that sort the projections of two `(ns, nt)`
    datasets `X_s` and `X_t` on the directions `thetas`.
    Each permutation `perm[:, k]` is such that each :math:`X_s[i, :]` is matched
    to `X_t[perm[i, k], :]` when projected on `thetas[k, :]`.

    Parameters
    ----------
    X_s : array-like, shape (ns, d)
        The first set of vectors.
    X_t : array-like, shape (nt, d)
        The second set of vectors.
    a : ndarray of float64, shape (ns,), optional
        Source histogram (default is uniform weight)
    b : ndarray of float64, shape (nt,), optional
        Target histogram (default is uniform weight)
    metric: str, optional (default='sqeuclidean')
        Metric to be used. Only works with either of the strings
        `'sqeuclidean'`, `'minkowski'`, `'cityblock'`,  or `'euclidean'`.
    p: float, optional (default=1.0)
         The p-norm to apply for if metric='minkowski'
    thetas : array-like, shape (n_proj, d), optional
        The projection directions. If None, random directions will be
        generated.
        Default is None.
    warm_theta : array-like, shape (d,), optional
        A direction to add to the set of directions. Default is None.
        Default is False.
    n_proj : int, optional
        The number of projection directions. Required if thetas is None.
    log : bool, optional
        If True, returns additional logging information. Default is False.

    Returns
    -------
    plan: list of dictionaries
        List of the optimal transport plans as a list of dictionaries containing
        the rows, cols and data of the non-zero elements of the transportation matrix.
    costs : list of float
        The cost associated to each projection.
    log_dict : dict, optional
        A dictionary containing intermediate computations for logging purposes.
        Returned only if `log` is True.
    """

    X_s, X_t = list_to_array(X_s, X_t)

    if a is not None and b is not None and thetas is None:
        nx = get_backend(X_s, X_t, a, b)
    elif a is not None and b is not None and thetas is not None:
        nx = get_backend(X_s, X_t, a, b, thetas)
    elif a is None and b is None and thetas is not None:
        nx = get_backend(X_s, X_t, thetas)
    else:
        nx = get_backend(X_s, X_t)

    assert X_s.ndim == 2, f"X_s must be a 2d array, got {X_s.ndim}d array instead"
    assert X_t.ndim == 2, f"X_t must be a 2d array, got {X_t.ndim}d array instead"

    assert metric in ("minkowski", "euclidean", "cityblock", "sqeuclidean"), (
        "Sliced plans work only with metrics "
        + "from the following list: "
        + "`['sqeuclidean', 'minkowski', 'cityblock', 'euclidean']`"
    )

    assert (
        X_s.shape[1] == X_t.shape[1]
    ), f"X_s ({X_s.shape}) and X_t ({X_t.shape}) must have the same number of columns"

    if metric == "euclidean":
        p = 2
    elif metric == "cityblock":
        p = 1

    d = X_s.shape[1]
    n = X_s.shape[0]
    m = X_t.shape[0]

    if a is None:
        a = nx.ones(n) / n
    if b is None:
        b = nx.ones(m) / m

    is_perm = False
    if n == m:
        if (a == a.sum() / n).all() and (b == a.sum() / n).all():
            is_perm = True

    if thetas is None:  # create thetas (n_proj, d)
        assert n_proj is not None, "n_proj must be specified if thetas is None"
        thetas = get_random_projections(d, n_proj, backend=nx, type_as=X_s).T
        if warm_theta is not None:
            thetas = nx.concatenate([thetas, warm_theta[:, None].T], axis=0)
    else:
        n_proj = thetas.shape[0]

    # project on each theta: (n or m, d) -> (n or m, n_proj)
    Xs_theta = X_s @ thetas.T  # shape (n, n_proj)
    Xt_theta = X_t @ thetas.T  # shape (m, n_proj)

    if is_perm:  # we compute maps (permutations)
        # sigma[:, i_proj] is a permutation sorting Xs_theta[:, i_proj]
        sigma = nx.argsort(Xs_theta, axis=0)  # (n, n_proj)
        tau = nx.argsort(Xt_theta, axis=0)  # (m, n_proj)
        costs = [
            sparse_ot_dist(X_s, X_t, sigma[:, k], tau[:, k], metric=metric, p=p)
            for k in range(n_proj)
        ]

        PlanTuple = namedtuple("PlanTuple", ["data", "rows", "cols"])
        plans = [
            PlanTuple(
                data=a,
                rows=sigma[:, k],
                cols=tau[:, k],
            )
            for k in range(n_proj)
        ]
    else:  # we compute plans
        _, plans = wasserstein_1d(
            Xs_theta, Xt_theta, a, b, p, require_sort=True, return_plans="coo_tuple"
        )
        costs = [
            sparse_ot_dist(
                X_s,
                X_t,
                plans[k].rows,
                plans[k].cols,
                plans[k].data,
                metric=metric,
                p=p,
            )
            for k in range(n_proj)
        ]

    if log:
        log_dict = {"X_theta": Xs_theta, "Y_theta": Xt_theta, "thetas": thetas}
        return plans, nx.stack(costs), log_dict
    else:
        return plans, nx.stack(costs)


def min_pivot_sliced(
    X_s,
    X_t,
    a=None,
    b=None,
    thetas=None,
    metric="sqeuclidean",
    p=2,
    n_proj=None,
    dense=True,
    log=False,
    warm_theta=None,
):
    r"""
    Computes the cost and permutation associated to the min-Pivot Sliced
    Discrepancy (introduced as SWGG in [83] and studied further in [84]). Given
    the supports `X_s` and `X_t` of two discrete uniform measures with `ns` and `nt`
    atoms in dimension `d`, the min-Pivot Sliced Discrepancy goes through
    `n_proj` different projections of the measures on random directions, and
    retains the couplings that yields the lowest cost between `X_s` and `X_t`
    (compared in :math:`\mathbb{R}^d`). When `ns=nt`, it gives

    .. math::
        \mathrm{min\text{-}PS}_p^p(X_s, X_t) \approx
        \min_{k \in [1, n_{\mathrm{proj}}]} \left(
        \frac{1}{n_s} \sum_{i=1}^{n_s} \|X_{s,i} - X_{t,\sigma_k(i)}\|_2^p \right),

    where :math:`\sigma_k` is a permutation such that ordering the projections
    on the axis `thetas[k, :]` matches :math:`X_s[i, :]` to :math:`X_t[\sigma_k(i), :]`.

    .. note::
        The computation ignores potential ambiguities in the projections: if
        two points from a same measure have the same projection on a direction,
        then multiple sorting permutations are possible. To avoid combinatorial
        explosion, only one permutation is retained: this strays from theory in
        pathological cases.

    .. warning::
        Tensorflow and jax only returns dense plans, as they do not support well sparse matrices.

    Parameters
    ----------
    X_s : array-like, shape (ns, d)
        The first set of vectors.
    X_t : array-like, shape (nt, d)
        The second set of vectors.
    a : ndarray of float64, shape (ns,), optional
        Source histogram (default is uniform weight)
    b : ndarray of float64, shape (nt,), optional
        Target histogram (default is uniform weight)
    thetas : array-like, shape (n_proj, d), optional
        The projection directions. If None, random directions will be generated
        Default is None.
    metric: str, optional (default='sqeuclidean')
        Metric to be used. Only works with either of the strings
        `'sqeuclidean'`, `'minkowski'`, `'cityblock'`,  or `'euclidean'`.
    p: float, optional (default=1.0)
         The p-norm to apply for if metric='minkowski'
    n_proj : int, optional
        The number of projection directions. Required if thetas is None.
    dense: boolean, optional (default=True)
        If True, returns :math:`\gamma` as a dense ndarray of shape (ns, nt).
        Otherwise returns a sparse representation using scipy's `coo_matrix`
        format.
    log : bool, optional
        If True, returns additional logging information. Default is False.
    warm_theta : array-like, shape (d,), optional
        A theta to add to the list of thetas. Default is None.

    Returns
    -------
    plan : ndarray, shape (ns, nt) or coo_matrix if dense is False
        Optimal transportation matrix for the given parameters.
    cost : float
        The cost associated to the optimal permutation.
    log_dict : dict, optional
        A dictionary containing intermediate computations for logging purposes.
        Returned only if `log` is True.

    References
    ----------
    .. [83] Mahey, G., Chapel, L., Gasso, G., Bonet, C., & Courty, N. (2023).
            Fast Optimal Transport through Sliced Generalized Wasserstein
            Geodesics. Advances in Neural Information Processing Systems, 36,
            35350–35385.

    .. [84] Tanguy, E., Chapel, L., Delon, J. (2025). Sliced Optimal Transport
            Plans. arXiv preprint 2506.03661.

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> x=np.array([[3.,3.], [1.,1.]])
    >>> y=np.array([[2.,2.5], [3.,2.]])
    >>> thetas=np.array([[1, 0], [0, 1]])
    >>> plan, cost = ot.min_pivot_sliced(x, y, thetas=thetas)
    >>> plan
    array([[0. , 0.5],
           [0.5, 0. ]])
    >>> cost
    2.125
    """

    X_s, X_t = list_to_array(X_s, X_t)

    if a is not None and b is not None and thetas is None:
        nx = get_backend(X_s, X_t, a, b)
    elif a is not None and b is not None and thetas is not None:
        nx = get_backend(X_s, X_t, a, b, thetas)
    elif a is None and b is None and thetas is not None:
        nx = get_backend(X_s, X_t, thetas)
    else:
        nx = get_backend(X_s, X_t)
    assert X_s.ndim == 2, f"X_s must be a 2d array, got {X_s.ndim}d array instead"
    assert X_t.ndim == 2, f"X_t must be a 2d array, got {X_t.ndim}d array instead"

    assert (
        X_s.shape[1] == X_t.shape[1]
    ), f"X_s ({X_s.shape}) and X_t ({X_t.shape}) must have the same number of columns"

    if str(nx) in ["tf", "jax"] and not dense:
        dense = True
        warnings.warn("JAX and TF do not support sparse matrices, converting to dense")

    log_dict = {}
    plans, costs, log_dict_plans = sliced_plans(
        X_s,
        X_t,
        a,
        b,
        metric,
        p,
        thetas,
        n_proj=n_proj,
        warm_theta=warm_theta,
        log=True,
    )

    pos_min = nx.argmin(costs)
    cost = costs[pos_min]
    plan = plans[pos_min]

    if log:
        log_dict = {
            "thetas": log_dict_plans["thetas"],
            "costs": costs,
            "min_theta": log_dict_plans["thetas"][pos_min],
            "X_min_theta": log_dict_plans["X_theta"][:, pos_min],
            "Y_min_theta": log_dict_plans["Y_theta"][:, pos_min],
        }

    # get the plan from the indices of the non-zero entries of the sparse plan
    plan = nx.coo_matrix(
        plan.data,
        plan.rows,
        plan.cols,
        shape=(X_s.shape[0], X_t.shape[0]),
        type_as=X_s,
    )

    if dense:
        plan = nx.todense(plan)

    if log:
        return plan, cost, log_dict
    else:
        return plan, cost


def expected_sliced(
    X_s,
    X_t,
    a=None,
    b=None,
    thetas=None,
    metric="sqeuclidean",
    p=2,
    n_proj=None,
    dense=True,
    log=False,
    beta=0.0,
):
    r"""
    Computes the Expected Sliced cost and plan between two  datasets `X_s` and
    `X_t` of shapes `(ns, d)` and `(nt, d)`. Given a set of `n_proj` projection
    directions, the expected sliced plan is obtained by averaging the `n_proj`
    1d optimal transport plans between the projections of `X_s` and `X_t` on each
    direction. Expected Sliced was introduced in [85] and further studied in
    [84].

    .. note::
        The computation ignores potential ambiguities in the projections: if
        two points from a same measure have the same projection on a direction,
        then multiple sorting permutations are possible. To avoid combinatorial
        explosion, only one permutation is retained: this strays from theory in
        pathological cases.

    .. warning::
        Tensorflow and jax only returns dense plans, as they do not support well sparse matrices.

    Parameters
    ----------
    X_s : array-like, shape (ns, d)
        The first set of vectors.
    X_t : array-like, shape (nt, d)
        The second set of vectors.
    a : ndarray of float64, shape (ns,), optional
        Source histogram (default is uniform weight)
    b : ndarray of float64, shape (nt,), optional
        Target histogram (default is uniform weight)
    thetas : array-like, optional
        An array-like of shape (n_proj, d) representing the projection directions.
        If None, random directions will be generated. Default is None.
    metric: str, optional (default='sqeuclidean')
        Metric to be used. Only works with either of the strings
        `'sqeuclidean'`, `'minkowski'`, `'cityblock'`,  or `'euclidean'`.
    p: float, optional (default=2)
            The p-norm to apply for if metric='minkowski'
    n_proj : int, optional
        The number of projection directions. Required if thetas is None.
    dense: boolean, optional (default=True)
        If True, returns :math:`\gamma` as a dense ndarray of shape (ns, nt).
        Otherwise returns a sparse representation using scipy's `coo_matrix`
        format.
    log : bool, optional
        If True, returns additional logging information. Default is False.
    beta : float, optional
        Inverse-temperature parameter which weights each projection's
        contribution to the expected plan. Default is 0 (uniform weighting).

    Returns
    -------
    plan : ndarray, shape (ns, nt) or coo_matrix if dense is False
        Optimal transportation matrix for the given parameters.
    cost : float
        The cost associated to the optimal permutation.
    log_dict : dict, optional
        A dictionary containing intermediate computations for logging purposes.
        Returned only if `log` is True.

    References
    ----------
    .. [84] Tanguy, E., Chapel, L., Delon, J. (2025). Sliced Optimal Transport
            Plans. arXiv preprint 2506.03661.
    .. [85] Liu, X., Diaz Martin, R., Bai Y., Shahbazi A., Thorpe M., Aldroubi
            A., Kolouri, S. (2024). Expected Sliced Transport Plans.
            International Conference on Learning Representations.

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> x=np.array([[3.,3.], [1.,1.]])
    >>> y=np.array([[2.,2.5], [3.,2.]])
    >>> thetas=np.array([[1, 0], [0, 1]])
    >>> plan, cost = ot.expected_sliced(x, y, thetas=thetas)
    >>> plan
    array([[0.25, 0.25],
           [0.25, 0.25]])
    >>> cost
    2.625
    """

    X_s, X_t = list_to_array(X_s, X_t)
    if a is not None and b is not None and thetas is None:
        nx = get_backend(X_s, X_t, a, b)
    elif a is not None and b is not None and thetas is not None:
        nx = get_backend(X_s, X_t, a, b, thetas)
    elif a is None and b is None and thetas is not None:
        nx = get_backend(X_s, X_t, thetas)
    else:
        nx = get_backend(X_s, X_t)

    assert X_s.ndim == 2, f"X_s must be a 2d array, got {X_s.ndim}d array instead"
    assert X_t.ndim == 2, f"X_t must be a 2d array, got {X_t.ndim}d array instead"

    assert (
        X_s.shape[1] == X_t.shape[1]
    ), f"X_s ({X_s.shape}) and X_t ({X_t.shape}) must have the same number of columns"

    if str(nx) in ["tf", "jax"] and not dense:
        dense = True
        warnings.warn("JAX and TF do not support sparse matrices, converting to dense")

    n = X_s.shape[0]
    m = X_t.shape[0]

    log_dict = {}
    plans, costs, log_dict_plans = sliced_plans(
        X_s, X_t, a, b, metric, p, thetas, n_proj=n_proj, log=True
    )

    if beta != 0.0:  # computing the temperature weighting
        log_factors = -beta * list_to_array(costs)
        weights = nx.exp(log_factors - nx.logsumexp(log_factors))
        cost = nx.sum(list_to_array(costs) * weights)
    else:  # uniform weights
        if n_proj is None:
            n_proj = thetas.shape[0]
        weights = nx.ones(n_proj) / n_proj

    weights_e = nx.concatenate([plans[i].data * weights[i] for i in range(len(plans))])
    Xs_idx = nx.concatenate([plans[i].rows for i in range(len(plans))])
    Xt_idx = nx.concatenate([plans[i].cols for i in range(len(plans))])

    plan = nx.coo_matrix(weights_e, Xs_idx, Xt_idx, shape=(n, m), type_as=weights)

    if dense:
        plan = nx.todense(plan)

    if beta == 0.0:
        if dense:
            cost = nx.sum(plan * dist(X_s, X_t, metric=metric, p=p))
        else:
            cost = plan.multiply(dist(X_s, X_t, metric=metric, p=p)).sum()
    if log:
        log_dict = {
            "thetas": log_dict_plans["thetas"],
            "costs": costs,
            "G": plans,
            "weights": weights,
        }
        return plan, cost, log_dict
    else:
        return plan, cost
