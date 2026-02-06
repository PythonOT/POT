# -*- coding: utf-8 -*-
"""
Sliced Unbalanced OT solvers
"""

# Author: Clément Bonet <clement.bonet.mapp@polytechnique.edu>
#
# License: MIT License

from ..backend import get_backend
from ..utils import get_parameter_pair, list_to_array
from ..sliced import get_random_projections
from ._solver_1d import rescale_potentials, uot_1d
from ..lp.solver_1d import emd_1d_dual_backprop, wasserstein_1d


def sliced_unbalanced_ot(
    X_s,
    X_t,
    reg_m,
    a=None,
    b=None,
    n_projections=50,
    p=2,
    projections=None,
    seed=None,
    numItermax=10,
    log=False,
):
    r"""
    Compute SUOT

    TODO

    This function only works in pytorch or jax.

    Parameters
    ----------
    X_s : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    X_t : ndarray, shape (n_samples_b, dim)
        samples in the target domain
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
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,), optional
        samples weights in the target domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    p: float, optional, by default =2
        Power p used for computing the sliced Wasserstein
    projections: shape (dim, n_projections), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    numItermax: int, optional
    log: bool, optional
        if True, returns the projections used and their associated UOTs and reweighted marginals.

    Returns
    -------
    loss: float/array-like, shape (...)
        SUOT

    References
    ----------
    [82] Bonet, C., Nadjahi, K., Séjourné, T., Fatras, K., & Courty, N. (2025).
    Slicing Unbalanced Optimal Transport. Transactions on Machine Learning Research
    """
    X_s, X_t = list_to_array(X_s, X_t)

    if a is not None and b is not None and projections is None:
        nx = get_backend(X_s, X_t, a, b)
    elif a is not None and b is not None and projections is not None:
        nx = get_backend(X_s, X_t, a, b, projections)
    elif a is None and b is None and projections is not None:
        nx = get_backend(X_s, X_t, projections)
    else:
        nx = get_backend(X_s, X_t)

    assert nx.__name__ in ["torch", "jax"], "Function only valid in torch and jax"

    n = X_s.shape[0]
    m = X_t.shape[0]

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(
                X_s.shape[1], X_t.shape[1]
            )
        )

    if a is None:
        a = nx.full(n, 1 / n, type_as=X_s)
    if b is None:
        b = nx.full(m, 1 / m, type_as=X_s)

    d = X_s.shape[1]

    if projections is None:
        projections = get_random_projections(
            d, n_projections, seed, backend=nx, type_as=X_s
        )
    else:
        n_projections = projections.shape[1]

    X_s_projections = nx.dot(X_s, projections)  # shape (n, n_projs)
    X_t_projections = nx.dot(X_t, projections)

    a_reweighted, b_reweighted, projected_uot = uot_1d(
        X_s_projections,
        X_t_projections,
        reg_m,
        a,
        b,
        p,
        require_sort=True,
        numItermax=numItermax,
    )

    res = nx.mean(projected_uot) ** (1.0 / p)

    if log:
        dico = {
            "projection": projections,
            "projected_uots": projected_uot,
            "a_reweighted": a_reweighted,
            "b_reweighted": b_reweighted,
        }
        return res, dico

    return res


def unbalanced_sliced_ot(
    X_s,
    X_t,
    reg_m,
    a=None,
    b=None,
    n_projections=50,
    p=2,
    projections=None,
    seed=None,
    numItermax=10,
    mode="backprop",
    stochastic_proj=False,
    log=False,
):
    r"""
    Compute USOT

    TODO

    This function only works in pytorch or jax.

    Parameters
    ----------
    X_s : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    X_t : ndarray, shape (n_samples_b, dim)
        samples in the target domain
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
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,), optional
        samples weights in the target domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    p: float, optional, by default =2
        Power p used for computing the sliced Wasserstein
    projections: shape (dim, n_projections), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    numItermax: int, optional
    mode: str, optional
        "icdf" for inverse CDF, "backprop" for backpropagation mode.
        Default is "icdf".
    stochastic_proj: bool, default False
    log: bool, optional
        if True, sliced_wasserstein_distance returns the projections used and their associated EMD.

    Returns
    -------
    a_reweighted: array-like shape (n, ...)
        First marginal reweighted
    b_reweighted: array-like shape (m, ...)
        Second marginal reweighted
    loss: float/array-like, shape (...)
        USOT

    References
    ----------
    [82] Bonet, C., Nadjahi, K., Séjourné, T., Fatras, K., & Courty, N. (2025).
    Slicing Unbalanced Optimal Transport. Transactions on Machine Learning Research
    """
    X_s, X_t = list_to_array(X_s, X_t)

    if a is not None and b is not None and projections is None:
        nx = get_backend(X_s, X_t, a, b)
    elif a is not None and b is not None and projections is not None:
        nx = get_backend(X_s, X_t, a, b, projections)
    elif a is None and b is None and projections is not None:
        nx = get_backend(X_s, X_t, projections)
    else:
        nx = get_backend(X_s, X_t)

    assert nx.__name__ in ["torch", "jax"], "Function only valid in torch and jax"

    reg_m1, reg_m2 = get_parameter_pair(reg_m)

    n = X_s.shape[0]
    m = X_t.shape[0]

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(
                X_s.shape[1], X_t.shape[1]
            )
        )

    if a is None:
        a = nx.full(n, 1 / n, type_as=X_s)
    if b is None:
        b = nx.full(m, 1 / m, type_as=X_s)

    d = X_s.shape[1]

    if projections is None and not stochastic_proj:
        projections = get_random_projections(
            d, n_projections, seed, backend=nx, type_as=X_s
        )
    else:
        n_projections = projections.shape[1]

    if not stochastic_proj:
        X_s_projections = nx.dot(X_s, projections).T  # shape (n_projs, n)
        X_t_projections = nx.dot(X_t, projections).T

        X_s_sorter = nx.argsort(X_s_projections, -1)
        X_s_rev_sorter = nx.argsort(X_s_sorter, -1)
        X_s_sorted = nx.take_along_axis(X_s_projections, X_s_sorter, -1)

        X_t_sorter = nx.argsort(X_t_projections, -1)
        X_t_rev_sorter = nx.argsort(X_t_sorter, -1)
        X_t_sorted = nx.take_along_axis(X_t_projections, X_t_sorter, -1)

    # Initialize potentials - WARNING: They correspond to non-sorted samples
    f = nx.zeros(a.shape, type_as=a)
    g = nx.zeros(b.shape, type_as=b)

    for i in range(numItermax):
        # Output FW descent direction
        # translate potentials
        transl = rescale_potentials(f, g, a, b, reg_m1, reg_m2, nx)

        f = f + transl
        g = g - transl

        # If stochastic version then sample new directions and re-sort data
        # TODO: add functions to sample and project
        if stochastic_proj:
            projections = get_random_projections(
                d, n_projections, seed, backend=nx, type_as=X_s
            )

            X_s_projections = nx.dot(X_s, projections)
            X_t_projections = nx.dot(X_t, projections)

            X_s_sorter = nx.argsort(X_s_projections, -1)
            X_s_rev_sorter = nx.argsort(X_s_sorter, -1)
            X_s_sorted = nx.take_along_axis(X_s_projections, X_s_sorter, -1)

            X_t_sorter = nx.argsort(X_t_projections, -1)
            X_t_rev_sorter = nx.argsort(X_t_sorter, -1)
            X_t_sorted = nx.take_along_axis(X_t_projections, X_t_sorter, -1)

        # update measures
        a_reweighted = (a * nx.exp(-f / reg_m1))[..., X_s_sorter]
        b_reweighted = (b * nx.exp(-g / reg_m2))[..., X_t_sorter]

        full_mass = nx.sum(a_reweighted, axis=1)

        # normalize the weights for compatibility with wasserstein_1d
        a_reweighted = a_reweighted / nx.sum(a_reweighted, axis=1, keepdims=True)
        b_reweighted = b_reweighted / nx.sum(b_reweighted, axis=1, keepdims=True)

        fd, gd, loss = emd_1d_dual_backprop(
            X_s_sorted.T,
            X_t_sorted.T,
            u_weights=a_reweighted.T,
            v_weights=b_reweighted.T,
            p=p,
            require_sort=False,
        )
        fd, gd = fd.T, gd.T

        # default step for FW
        t = 2.0 / (2.0 + i)

        f = f + t * (nx.mean(nx.take_along_axis(fd, X_s_rev_sorter, 1), axis=0) - f)
        g = g + t * (nx.mean(nx.take_along_axis(gd, X_t_rev_sorter, 1), axis=0) - g)

    ot_loss = wasserstein_1d(
        X_s_sorted,
        X_t_sorted,
        u_weights=a_reweighted.T,
        v_weights=b_reweighted.T,
        p=p,
        require_sort=False,
    )
    sot_loss = nx.mean(ot_loss * full_mass)

    a_reweighted, b_reweighted = a * nx.exp(-f / reg_m1), b * nx.exp(-g / reg_m2)

    uot_loss = (
        sot_loss
        + reg_m1 * nx.kl_div(a_reweighted, a, mass=True)
        + reg_m2 * nx.kl_div(b_reweighted, b, mass=True)
    )

    if log:
        return a_reweighted, b_reweighted, uot_loss, {"projections": projections}

    return a_reweighted, b_reweighted, uot_loss
