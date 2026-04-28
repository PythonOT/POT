import numpy as np
import warnings

from ..utils import list_to_array
from ..backend import get_backend
from .bsp_wrap import bsp_solve_c, merge_bijections_c


def compute_bspot_bijection(
    X, Y, n_plans=64, lp_power=2, initial_plan=None, gaussian=True
):
    r"""

    This solver provides a good and fast approximation of the combinatorial problem of finding
    a bijection between two point clouds that minimizes the transport cost:

    .. math::
        \min_{\sigma \in S_n}  \sum_{i=1}^n \|X_i - Y_{\sigma(i)}\|_p^p

    To do so, it generates :math:`n_{plans}` random bijective BSP matchings, merges them together to obtain a bijection of low transport cost.
    Log-linear complexity in the number of points.

    .. note:: There is no guarantee on the quality of the returned bijection, but the method is highly scalable on the CPU.
        Worst cases are obtained between point clouds that are very similar (e.g. two samples from the same distribution),
        where the solver can get stuck in local minima, but works well when the point clouds are very different.
        The method also works best for the standard squared euclidean cost (lp_power=2), as this cost enables
        efficient BSP construction strategy (with a cubic dependence on the dimension, this feature is disabled
        for dimensions larger than 64).

    Parameters
    ----------
    X : array-like, shape (n_samples, dimension)
    Y : array-like, shape (n_samples, dimension)
    n_plans : int
        The number of BSP Matchings used to compute the final bijection.
    lp_power : int, optional
        The power of the ground metric (default 2 for squared euclidean, -1 for infinity norm).
    initial_plan : array-like, shape (n_samples,), optional
        Bijection to use for initializing merging (optional).
    gaussian : bool, optional
        If true then uses the Gaussian slicing heuristic to improve matching quality.
        Comes with a cubic complexity with dimension, set at true by default.

    Returns
    -------
    cost : float
        The transport cost of the final bijection.
    plan : array-like, shape (n_samples,)
        The final bijection, stored as a permutation (e.g. a list of numbers) such that X[i] is assigned to Y[plan[i]].
    plans : list of array
        The intermediary bijections used to compute the final one.


    """

    nx = get_backend(X)

    X_np = nx.to_numpy(X)
    Y_np = nx.to_numpy(Y)

    X_np = np.asarray(X_np, dtype=np.float64)
    Y_np = np.asarray(Y_np, dtype=np.float64)

    if X_np.shape != Y_np.shape:
        raise ValueError("X and Y must have the same shape")

    if X_np.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")

    if initial_plan is not None:
        initial_plan = list_to_array(initial_plan)
        if initial_plan.shape != (X.shape[0],):
            raise ValueError(
                "initial_plan must have shape (n,) where n is the number of points"
            )

    cost_inner, plan, plans = bsp_solve_c(
        X_np, Y_np, n_plans, lp_power, initial_plan, gaussian
    )

    plan = nx.from_numpy(plan)

    if lp_power == -1:
        # infinity norm per pair of points, then mean over all pairs
        cost = nx.sum(nx.max(nx.abs(X - Y[plan]), axis=1))
    else:
        cost = nx.sum(nx.abs(X - Y[plan]) ** lp_power)

    cost = cost / X_np.shape[0]

    # add warning if cost_inner and cost differ significantly
    if not np.isclose(nx.to_numpy(cost), cost_inner, rtol=1e-3):
        warnings.warn(
            "Cost computed from plan differs from cost returned by solver. Cost from plan: {}, Cost from solver: {}".format(
                cost, cost_inner
            )
        )

    return cost, plan, plans


def merge_bijections(X, Y, plans, lp_power=2):
    r"""
    Merge several bijections between two point clouds to obtain a new one with low transport cost.
    The new bijection is guaranteed to have a transport cost no greater than the cost of any of the input bijections.
    Based on simple local/global swapping strategy, with a linear complexity in the number of points.

    Parameters
    ----------
    X : array-like, shape (n_samples, dimension)
    Y : array-like, shape (n_samples, dimension)
    plans : list of array-like, shape (n_samples,)
        The bijections to merge, stored as permutations (e.g. a list of numbers)
    lp_power : int, optional
        The power of the ground metric (default 2 for squared euclidean).
        If lp_power is -1, the infinity norm is used.

    Returns
    -------
    cost : float
        The transport cost of the merged bijection.
    plan : array-like, shape (n_samples,)
        The merged bijection, stored as a permutation (e.g. a list of numbers) such that X[i] is assigned to Y[plan[i]].
    """

    nx = get_backend(X)

    X_np = nx.to_numpy(X)
    Y_np = nx.to_numpy(Y)

    X_np = np.asarray(X_np, dtype=np.float64)
    Y_np = np.asarray(Y_np, dtype=np.float64)

    if X_np.shape != Y_np.shape:
        raise ValueError("X and Y must have the same shape")

    if X_np.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")

    plans = np.asarray(plans, dtype=np.int32)

    cost_inner, plan = merge_bijections_c(X_np, Y_np, plans, lp_power)

    plan = nx.from_numpy(plan)

    if lp_power == -1:
        # infinity norm per pair of points, then mean over all pairs
        cost = nx.sum(nx.max(nx.abs(X - Y[plan]), axis=1))
    else:
        cost = nx.sum(nx.abs(X - Y[plan]) ** lp_power)

    cost = cost / X_np.shape[0]

    # add warning if cost_inner and cost differ significantly
    if not np.isclose(nx.to_numpy(cost), cost_inner, rtol=1e-3):
        warnings.warn(
            "Cost computed from plan differs from cost returned by solver. Cost from plan: {}, Cost from solver: {}".format(
                cost, cost_inner
            )
        )

    return cost, plan
