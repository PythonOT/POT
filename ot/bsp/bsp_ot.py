import numpy as np
import warnings

from ..utils import list_to_array
from ..backend import get_backend
from .bsp_wrap import bsp_solve_c, merge_bijections_c


def bsp_solve(X, Y, n_plans=64, lp_power=2, initial_plan=None):
    """

    Builds nb_plans BSP Matchings and merges them in a single bijection.

        cost,plan,plans = bsp_solve(X,Y,n_plans)

    where :

    - X and Y are the input point clouds
    - n_plans is the number of BSP Matchings used to compute the final bijection
    - lp_power is the power of the ground metric (default 2 for squared euclidean)
    - initial_plan bijection to use for initializing merging (optional)

    Returns the transport cost of the final bijection, the final bijection, and the intermediary ones

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

    cost_inner, plan, plans = bsp_solve_c(X_np, Y_np, n_plans, lp_power, initial_plan)

    plan = nx.from_numpy(plan)

    if lp_power == -1:
        # infinity norm per pair of points, then mean over all pairs
        cost = nx.sum(nx.max(nx.abs(X - Y[plan]), axis=1))
    else:
        cost = nx.sum(nx.abs(X - Y[plan]) ** lp_power)

    cost = cost / X_np.shape[0]

    # add warning if cost_inner and cost differ significantly
    if not np.isclose(nx.to_numpy(cost), cost_inner, rtol=1e-5):
        warnings.warn(
            "Cost computed from plan differs from cost returned by solver. Cost from plan: {}, Cost from solver: {}".format(
                cost, cost_inner
            )
        )

    return cost, plan, plans


def merge_bijections(X, Y, plans, lp_power=2):
    """
        Merges transport bijections

    where :

    - X and Y are the input point clouds
    - plans input bijections
    - lp_power is the power of the ground metric (default 2 for squared euclidean)

    Returns the merged bijection and its transport cost.
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
    if not np.isclose(nx.to_numpy(cost), cost_inner, rtol=1e-5):
        warnings.warn(
            "Cost computed from plan differs from cost returned by solver. Cost from plan: {}, Cost from solver: {}".format(
                cost, cost_inner
            )
        )

    return cost, plan
