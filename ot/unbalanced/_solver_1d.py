# -*- coding: utf-8 -*-
"""
1D Unbalanced OT solvers
"""

# Author:
#
# License: MIT License

from ..backend import get_backend
from ..utils import get_parameter_pair
from ..lp.solver_1d import emd_1d_dual, emd_1d_dual_backprop


def rescale_potentials(f, g, a, b, rho1, rho2, nx):
    r"""
    TODO
    """
    tau = (rho1 * rho2) / (rho1 + rho2)
    num = nx.logsumexp(-f / rho1 + nx.log(a))
    denom = nx.logsumexp(-g / rho2 + nx.log(b))
    transl = tau * (num - denom)
    return transl


def uot_1d(
    u_values,
    v_values,
    reg_m,
    u_weights=None,
    v_weights=None,
    p=1,
    require_sort=True,
    numItermax=10,
    stopThr=1e-6,
    mode="icdf",
    log=False,
):
    r"""
    TODO, TOTEST, seems not very stable?

    Solves the 1D unbalanced OT problem with KL regularization.
    The function implements the Frank-Wolfe algorithm to solve the dual problem,
    as proposed in [73].

    TODO: add math equation

    Parameters
    ----------
    u_values: array-like, shape (n, ...)
        locations of the first empirical distribution
    v_values: array-like, shape (m, ...)
        locations of the second empirical distribution
    reg_m: float or indexable object of length 1 or 2
        Marginal relaxation term.
        If :math:`\mathrm{reg_{m}}` is a scalar or an indexable object of length 1,
        then the same :math:`\mathrm{reg_{m}}` is applied to both marginal relaxations.
        The balanced OT can be recovered using :math:`\mathrm{reg_{m}}=float("inf")`.
        For semi-relaxed case, use either
        :math:`\mathrm{reg_{m}}=(float("inf"), scalar)` or
        :math:`\mathrm{reg_{m}}=(scalar, float("inf"))`.
        If :math:`\mathrm{reg_{m}}` is an array,
        it must have the same backend as input arrays `(a, b, M)`.
    u_weights: array-like, shape (n, ...), optional
        weights of the first empirical distribution, if None then uniform weights are used
    v_weights: array-like, shape (m, ...), optional
        weights of the second empirical distribution, if None then uniform weights are used
    p: int, optional
        order of the ground metric used, should be at least 1, default is 1
    require_sort: bool, optional
        sort the distributions atoms locations, if False we will consider they have been sorted prior to being passed to
        the function, default is True
    numItermax: int, optional
    mode: str, optional
        "icdf" for inverse CDF, "backprop" for backpropagation mode.
        Default is "icdf".
    log: bool, optional

    Returns
    -------
    f: array-like shape (n, ...)
        First dual potential
    g: array-like shape (m, ...)
        Second dual potential
    loss: float/array-like, shape (...)
        the batched EMD

    References
    ---------
    .. [73] Séjourné, T., Vialard, F. X., & Peyré, G. (2022).
       Faster unbalanced optimal transport: Translation invariant sinkhorn and 1-d frank-wolfe.
       In International Conference on Artificial Intelligence and Statistics (pp. 4995-5021). PMLR.
    """
    assert mode in ["backprop", "icdf"]

    if u_weights is not None and v_weights is not None:
        nx = get_backend(u_values, v_values, u_weights, v_weights)
    else:
        nx = get_backend(u_values, v_values)

    reg_m1, reg_m2 = get_parameter_pair(reg_m)

    n = u_values.shape[0]
    m = v_values.shape[0]

    # Init weights or broadcast if necessary
    if u_weights is None:
        u_weights = nx.full(u_values.shape, 1.0 / n, type_as=u_values)
    elif u_weights.ndim != u_values.ndim:
        u_weights = nx.repeat(u_weights[..., None], u_values.shape[-1], -1)

    if v_weights is None:
        v_weights = nx.full(v_values.shape, 1.0 / m, type_as=v_values)
    elif v_weights.ndim != v_values.ndim:
        v_weights = nx.repeat(v_weights[..., None], v_values.shape[-1], -1)

    # Sort w.r.t. support if not already done
    if require_sort:
        u_sorter = nx.argsort(u_values, 0)
        u_rev_sorter = nx.argsort(u_sorter, 0)
        u_values_sorted = nx.take_along_axis(u_values, u_sorter, 0)

        v_sorter = nx.argsort(v_values, 0)
        v_rev_sorter = nx.argsort(v_sorter, 0)
        v_values_sorted = nx.take_along_axis(v_values, v_sorter, 0)

        u_weights_sorted = nx.take_along_axis(u_weights, u_sorter, 0)
        v_weights_sorted = nx.take_along_axis(v_weights, v_sorter, 0)

    f = nx.zeros(u_weights.shape, type_as=u_weights)
    g = nx.zeros(v_weights.shape, type_as=v_weights)

    for i in range(numItermax):
        transl = rescale_potentials(
            f, g, u_weights_sorted, v_weights_sorted, reg_m1, reg_m2, nx
        )

        f = f + transl
        g = g - transl

        u_reweighted = u_weights_sorted * nx.exp(-f / reg_m1)
        v_reweighted = v_weights_sorted * nx.exp(-g / reg_m2)

        if mode == "icdf":
            fd, gd, loss = emd_1d_dual(
                u_values_sorted,
                v_values_sorted,
                u_weights=u_reweighted,
                v_weights=v_reweighted,
                p=p,
                require_sort=False,
            )
        elif mode == "backprop":
            fd, gd, loss = emd_1d_dual_backprop(
                u_values_sorted,
                v_values_sorted,
                u_weights=u_reweighted,
                v_weights=v_reweighted,
                p=p,
                require_sort=False,
            )

        t = 2.0 / (2.0 + i)
        f = f + t * (fd - f)
        g = g + t * (gd - g)

    if require_sort:
        f = nx.take_along_axis(f, u_rev_sorter, 0)
        g = nx.take_along_axis(g, v_rev_sorter, 0)
        u_reweighted = nx.take_along_axis(u_reweighted, u_rev_sorter, 0)
        v_reweighted = nx.take_along_axis(v_reweighted, v_rev_sorter, 0)

    uot_loss = (
        loss
        + reg_m1 * nx.kl_div(u_reweighted, u_weights)
        + reg_m2 * nx.kl_div(v_reweighted, v_weights)
    )

    if log:
        dico = {"f": f, "g": g}
        return u_reweighted, v_reweighted, uot_loss, dico
    return u_reweighted, v_reweighted, uot_loss
