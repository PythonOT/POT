# -*- coding: utf-8 -*-
"""
1D Unbalanced OT solvers
"""

# Author: Clément Bonet <clement.bonet.mapp@polytechnique.edu>
#
# License: MIT License

from ..backend import get_backend
from ..utils import get_parameter_pair
from ..lp.solver_1d import emd_1d_dual_backprop, wasserstein_1d


def rescale_potentials(f, g, a, b, rho1, rho2, nx):
    r"""
    Find the optimal :math: `\lambda` in the translation invariant dual of UOT
    with KL regularization and returns it, see Proposition 2 in :ref:`[73] <references-uot>`.

    Parameters
    ----------
    f: array-like, shape (n, ...)
        first dual potential
    g: array-like, shape (m, ...)
        second dual potential
    a: array-like, shape (n, ...)
        weights of the first empirical distribution
    b: array-like, shape (m, ...)
        weights of the second empirical distribution
    rho1: float
        Marginal relaxation term for the first marginal
    rho2: float
        Marginal relaxation term for the second marginal
    nx: module
        backend module

    Returns
    -------
    transl: array-like, shape (...)
        optimal translation

    .. _references-uot:
    References
    ----------
    .. [73] Séjourné, T., Vialard, F. X., & Peyré, G. (2022).
       Faster unbalanced optimal transport: Translation invariant sinkhorn and 1-d frank-wolfe.
       In International Conference on Artificial Intelligence and Statistics (pp. 4995-5021). PMLR.
    """
    if rho1 == float("inf") and rho2 == float("inf"):
        return nx.zeros(shape=nx.sum(f, axis=0).shape, type_as=f)

    elif rho1 == float("inf"):
        tau = rho2
        denom = nx.logsumexp(-g / rho2 + nx.log(b), axis=0)
        num = nx.log(nx.sum(a, axis=0))

    elif rho2 == float("inf"):
        tau = rho1
        num = nx.logsumexp(-f / rho1 + nx.log(a), axis=0)
        denom = nx.log(nx.sum(b, axis=0))

    else:
        tau = (rho1 * rho2) / (rho1 + rho2)
        num = nx.logsumexp(-f / rho1 + nx.log(a), axis=0)
        denom = nx.logsumexp(-g / rho2 + nx.log(b), axis=0)

    transl = tau * (num - denom)

    return transl


def get_reweighted_marginal_uot(
    f, g, u_weights_sorted, v_weights_sorted, reg_m1, reg_m2, nx
):
    r"""
    One step of the FW algorithm for the 1D UOT problem with KL regularization.
    This function computes the reweighted marginals given the current dual potentials.
    It returns the current potentials, and the reweighted marginals (normalized by the mass so that they sum to 1).

    Parameters
    ----------
    f: array-like, shape (n, ...)
        first dual potential
    g: array-like, shape (m, ...)
        second dual potential
    u_weights_sorted: array-like, shape (n, ...)
        weights of the first empirical distribution, sorted w.r.t. the support
    v_weights_sorted: array-like, shape (m, ...)
        weights of the second empirical distribution, sorted w.r.t. the support
    reg_m1: float
        Marginal relaxation term for the first marginal
    reg_m2: float
        Marginal relaxation term for the second marginal
    nx: module
        backend module

    Returns
    -------
    f: array-like, shape (n, ...)
        first dual potential
    g: array-like, shape (m, ...)
        second dual potential
    u_rescaled: array-like, shape (n, ...)
        reweighted first marginal, normalized by the mass
    v_rescaled: array-like, shape (m, ...)
        reweighted second marginal, normalized by the mass
    full_mass: array-like, shape (...)
        mass of the reweighted marginals
    """
    transl = rescale_potentials(
        f, g, u_weights_sorted, v_weights_sorted, reg_m1, reg_m2, nx
    )

    f = f + transl[None]
    g = g - transl[None]

    if reg_m1 != float("inf"):
        u_reweighted = u_weights_sorted * nx.exp(-f / reg_m1)
    else:
        u_reweighted = u_weights_sorted

    if reg_m2 != float("inf"):
        v_reweighted = v_weights_sorted * nx.exp(-g / reg_m2)
    else:
        v_reweighted = v_weights_sorted

    full_mass = nx.sum(u_reweighted, axis=0)

    # Normalize weights
    u_rescaled = u_reweighted / nx.sum(u_reweighted, axis=0, keepdims=True)
    v_rescaled = v_reweighted / nx.sum(v_reweighted, axis=0, keepdims=True)

    return f, g, u_rescaled, v_rescaled, full_mass


def uot_1d(
    u_values,
    v_values,
    reg_m,
    u_weights=None,
    v_weights=None,
    p=2,
    require_sort=True,
    numItermax=10,
    returnCost="linear",
    log=False,
):
    r"""
    Solves the 1D unbalanced OT problem with KL regularization.
    The function implements the Frank-Wolfe algorithm to solve the dual problem,
    as proposed in :ref:`[73] <references-uot>`.

    The unbalanced OT problem reads

    .. math::
        \mathrm{UOT}_p^p(\mu,\nu) = \min_{\gamma \in \mathcal{M}_{+}(\mathbb{R}\times\mathbb{R})} W_p^p(\pi^1_\#\gamma,\pi^2_\#\gamma) + \mathrm{reg_{m}}_1 \mathrm{KL}(\pi^1_\#\gamma|\mu) + \mathrm{reg_{m}}_2 \mathrm{KL}(\pi^2_\#\gamma|\nu).

    .. warning:: This function only works in pytorch or jax as it uses autodifferentiation to compute the potentials. It is not maintained in jax.

    Parameters
    ----------
    u_values: array-like, shape (n, ...)
        locations of the first empirical distribution
    v_values: array-like, shape (m, ...)
        locations of the second empirical distribution
    reg_m: float or indexable object of length 1 or 2
        Marginal relaxation term.
        If `reg_m` is a scalar or an indexable object of length 1,
        then the same `reg_m` is applied to both marginal relaxations.
        The balanced OT can be recovered using `reg_m=float("inf")`.
        For semi-relaxed case, use either `reg_m=(float("inf"), scalar)` or `reg_m=(scalar, float("inf"))`.
        If `reg_m` is an array, it must have the same backend as input arrays `(u_values, v_values)`.
    u_weights: array-like, shape (n, ...), optional
        weights of the first empirical distribution, if None then uniform weights are used
    v_weights: array-like, shape (m, ...), optional
        weights of the second empirical distribution, if None then uniform weights are used
    p: int, optional
        order of the ground metric used, should be at least 1, default is 2
    require_sort: bool, optional
        sort the distributions atoms locations, if False we will consider they have been sorted prior to being passed to
        the function, default is True
    numItermax: int, optional
    returnCost: string, optional (default = "linear")
        If `returnCost` = "linear", then return the linear part of the unbalanced OT loss.
        If `returnCost` = "total", then return the total unbalanced OT loss.
    log: bool, optional

    Returns
    -------
    u_reweighted: array-like shape (n, ...)
        First marginal reweighted
    v_reweighted: array-like shape (m, ...)
        Second marginal reweighted
    loss: float/array-like, shape (...)
        The batched 1D UOT
    log: dict, optional
        If `log` is True, then returns a dictionary containing the dual potentials, the total cost and the linear cost.


    .. _references-uot:
    References
    ---------
    .. [73] Séjourné, T., Vialard, F. X., & Peyré, G. (2022).
       Faster unbalanced optimal transport: Translation invariant sinkhorn and 1-d frank-wolfe.
       In International Conference on Artificial Intelligence and Statistics (pp. 4995-5021). PMLR.
    """
    nx = get_backend(u_values, v_values, u_weights, v_weights)

    assert nx.__name__ in ["torch", "jax"], "Function only valid in torch and jax"

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
    fd = nx.zeros(u_weights.shape, type_as=u_weights)
    g = nx.zeros(v_weights.shape, type_as=v_weights)
    gd = nx.zeros(v_weights.shape, type_as=v_weights)

    for i in range(numItermax):
        # FW steps
        f, g, u_rescaled, v_rescaled, _ = get_reweighted_marginal_uot(
            f, g, u_weights_sorted, v_weights_sorted, reg_m1, reg_m2, nx
        )

        fd, gd, loss = emd_1d_dual_backprop(
            u_values_sorted,
            v_values_sorted,
            u_weights=u_rescaled,
            v_weights=v_rescaled,
            p=p,
            require_sort=False,
        )

        t = 2.0 / (2.0 + i)
        f = f + t * (fd - f)
        g = g + t * (gd - g)

    f, g, u_rescaled, v_rescaled, full_mass = get_reweighted_marginal_uot(
        f, g, u_weights_sorted, v_weights_sorted, reg_m1, reg_m2, nx
    )

    loss = wasserstein_1d(
        u_values_sorted,
        v_values_sorted,
        u_rescaled,
        v_rescaled,
        p=p,
        require_sort=False,
    )

    if require_sort:
        f = nx.take_along_axis(f, u_rev_sorter, 0)
        g = nx.take_along_axis(g, v_rev_sorter, 0)
        u_reweighted = nx.take_along_axis(u_rescaled, u_rev_sorter, 0) * full_mass
        v_reweighted = nx.take_along_axis(v_rescaled, v_rev_sorter, 0) * full_mass

    # rescale OT loss
    linear_loss = loss * full_mass

    if reg_m1 == float("inf") and reg_m2 == float("inf"):
        uot_loss = linear_loss
    elif reg_m1 == float("inf"):
        uot_loss = linear_loss + reg_m2 * nx.kl_div(v_reweighted, v_weights, mass=True)
    elif reg_m2 == float("inf"):
        uot_loss = linear_loss + reg_m1 * nx.kl_div(u_reweighted, u_weights, mass=True)
    else:
        uot_loss = (
            linear_loss
            + reg_m1 * nx.kl_div(u_reweighted, u_weights, mass=True, axis=0)
            + reg_m2 * nx.kl_div(v_reweighted, v_weights, mass=True, axis=0)
        )

    if returnCost == "linear":
        out_loss = linear_loss
    elif returnCost == "total":
        out_loss = uot_loss

    if log:
        dico = {"f": f, "g": g, "total_cost": uot_loss, "linear_cost": linear_loss}
        return u_reweighted, v_reweighted, out_loss, dico
    return u_reweighted, v_reweighted, out_loss
