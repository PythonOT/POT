# -*- coding: utf-8 -*-
"""
Sliced Wasserstein distances on the Sphere solvers.
"""

# Author: Nicolas Courty <ncourty@irisa.fr>
# Author: Clément Bonet <clement.bonet.mapp@polytechnique.edu>
#
# License: MIT License

from ..backend import get_backend
from ._utils import get_projections_sphere, projection_sphere_to_circle
from ..lp import (
    wasserstein_circle,
    semidiscrete_wasserstein2_unif_circle,
    linear_circular_ot,
)


def sliced_wasserstein_sphere(
    X_s,
    X_t,
    a=None,
    b=None,
    n_projections=50,
    p=2,
    projections=None,
    seed=None,
    log=False,
):
    r"""
    Compute the spherical sliced-Wasserstein discrepancy.

    .. math::
        SSW_p(\mu,\nu) = \left(\int_{\mathbb{V}_{d,2}} W_p^p(P^U_\#\mu, P^U_\#\nu)\ \mathrm{d}\sigma(U)\right)^{\frac{1}{p}}

    where:

    - :math:`P^U_\# \mu` stands for the pushforwards of the projection :math:`\forall x\in S^{d-1},\ P^U(x) = \frac{U^Tx}{\|U^Tx\|_2}`

    The function runs on backend but tensorflow and jax are not supported.

    Parameters
    ----------
    X_s: ndarray, shape (n_samples_a, dim)
        Samples in the source domain
    X_t: ndarray, shape (n_samples_b, dim)
        Samples in the target domain
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,), optional
        samples weights in the target domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    p: float, optional (default=2)
        Power p used for computing the spherical sliced Wasserstein
    projections: shape (n_projections, dim, 2), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    log: bool, optional
        if True, sliced_wasserstein_sphere returns the projections used and their associated EMD.

    Returns
    -------
    cost: float
        Spherical Sliced Wasserstein Cost
    log: dict, optional
        log dictionary return only if log==True in parameters

    Examples
    --------
    >>> import ot
    >>> n_samples_a = 20
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> X = X / np.sqrt(np.sum(X**2, -1, keepdims=True))
    >>> ot.sliced_wasserstein_sphere(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0

    References
    ----------
    .. [46] Bonet, C., Berg, P., Courty, N., Septier, F., Drumetz, L., & Pham, M. T. (2023). Spherical sliced-wasserstein. International Conference on Learning Representations.
    """
    d = X_s.shape[-1]

    nx = get_backend(X_s, X_t, a, b)

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(
                X_s.shape[1], X_t.shape[1]
            )
        )
    if nx.any(nx.abs(nx.sum(X_s**2, axis=-1) - 1) > 10 ** (-4)):
        raise ValueError("X_s is not on the sphere.")
    if nx.any(nx.abs(nx.sum(X_t**2, axis=-1) - 1) > 10 ** (-4)):
        raise ValueError("X_t is not on the sphere.")

    if projections is None:
        projections = get_projections_sphere(
            d, n_projections, seed=seed, backend=nx, type_as=X_s
        )

    Xps_coords, _ = projection_sphere_to_circle(
        X_s, n_projections=n_projections, projections=projections, seed=seed, backend=nx
    )

    Xpt_coords, _ = projection_sphere_to_circle(
        X_t, n_projections=n_projections, projections=projections, seed=seed, backend=nx
    )

    projected_emd = wasserstein_circle(
        Xps_coords.T, Xpt_coords.T, u_weights=a, v_weights=b, p=p
    )
    res = nx.mean(projected_emd) ** (1 / p)

    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res


def sliced_wasserstein_sphere_unif(
    X_s, a=None, n_projections=50, projections=None, seed=None, log=False
):
    r"""Compute the 2-spherical sliced wasserstein w.r.t. a uniform distribution.

    .. math::
        SSW_2(\mu_n, \nu)

    where

    - :math:`\mu_n=\sum_{i=1}^n \alpha_i \delta_{x_i}`
    - :math:`\nu=\mathrm{Unif}(S^{d-1})`

    Parameters
    ----------
    X_s: ndarray, shape (n_samples_a, dim)
        Samples in the source domain
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    projections: shape (n_projections, dim, 2), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    log: bool, optional
        if True, sliced_wasserstein_distance returns the projections used and their associated EMD.

    Returns
    -------
    cost: float
        Spherical Sliced Wasserstein Cost
    log: dict, optional
        log dictionary return only if log==True in parameters

    Examples
    ---------
    >>> import ot
    >>> np.random.seed(42)
    >>> x0 = np.random.randn(500,3)
    >>> x0 = x0 / np.sqrt(np.sum(x0**2, -1, keepdims=True))
    >>> ssw = ot.sliced_wasserstein_sphere_unif(x0, seed=42)
    >>> np.allclose(ot.sliced_wasserstein_sphere_unif(x0, seed=42), 0.01734, atol=1e-3)
    True

    References:
    -----------
    .. [46] Bonet, C., Berg, P., Courty, N., Septier, F., Drumetz, L., & Pham, M. T. (2023). Spherical sliced-wasserstein. International Conference on Learning Representations.
    """
    d = X_s.shape[-1]

    nx = get_backend(X_s, a)

    if nx.any(nx.abs(nx.sum(X_s**2, axis=-1) - 1) > 10 ** (-4)):
        raise ValueError("X_s is not on the sphere.")

    if projections is None:
        projections = get_projections_sphere(
            d, n_projections, seed=seed, backend=nx, type_as=X_s
        )

    Xps_coords, _ = projection_sphere_to_circle(
        X_s, n_projections=n_projections, projections=projections, seed=seed, backend=nx
    )

    projected_emd = semidiscrete_wasserstein2_unif_circle(Xps_coords.T, u_weights=a)
    res = nx.mean(projected_emd) ** (1 / 2)

    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res


def linear_sliced_wasserstein_sphere(
    X_s,
    X_t=None,
    a=None,
    b=None,
    n_projections=50,
    projections=None,
    seed=None,
    log=False,
):
    r"""Computes the linear spherical sliced wasserstein distance from :ref:`[79] <references-lssot>`.

    General loss returned:

    .. math::
        \mathrm{LSSOT}_2(\mu, \nu) = \left(\int_{\mathbb{V}_{d,2}} \mathrm{LCOT}_2^2(P^U_\#\mu, P^U_\#\nu)\ \mathrm{d}\sigma(U)\right)^{\frac12},

    where :math:`\mu,\nu\in\mathcal{P}(S^{d-1})` are two probability measures on the sphere, :math:`\mathrm{LCOT}_2` is the linear circular optimal transport distance,
    and :math:`P^U_\# \mu` stands for the pushforwards of the projection :math:`\forall x\in S^{d-1},\ P^U(x) = \frac{U^Tx}{\|U^Tx\|_2}`.

    Parameters
    ----------
    X_s: ndarray, shape (n_samples_a, dim)
        Samples in the source domain
    X_t: ndarray, shape (n_samples_b, dim), optional
        Samples in the target domain. If None, computes the distance against
        the uniform distribution on the sphere.
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,), optional
        samples weights in the target domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    projections: shape (n_projections, dim, 2), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    log: bool, optional
        if True, linear_sliced_wasserstein_sphere returns the projections used
        and their associated LCOT.

    Returns
    -------
    cost: float
        Linear Spherical Sliced Wasserstein Cost
    log: dict, optional
        log dictionary return only if log==True in parameters

    Examples
    ---------
    >>> import ot
    >>> n_samples_a = 20
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> X = X / np.sqrt(np.sum(X**2, -1, keepdims=True))
    >>> ot.linear_sliced_wasserstein_sphere(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0


    .. _references-lssot:
    References
    ----------
    .. [79] Liu, X., Bai, Y., Martín, R. D., Shi, K., Shahbazi, A., Landman,
       B. A., Chang, C., & Kolouri, S. (2025). Linear Spherical Sliced Optimal
       Transport: A Fast Metric for Comparing Spherical Data. International
       Conference on Learning Representations.
    """
    d = X_s.shape[-1]

    nx = get_backend(X_s, X_t, a, b)

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} \
            respectively given".format(X_s.shape[1], X_t.shape[1])
        )
    if nx.any(nx.abs(nx.sum(X_s**2, axis=-1) - 1) > 10 ** (-4)):
        raise ValueError("X_s is not on the sphere.")
    if nx.any(nx.abs(nx.sum(X_t**2, axis=-1) - 1) > 10 ** (-4)):
        raise ValueError("X_t is not on the sphere.")

    if projections is None:
        projections = get_projections_sphere(
            d, n_projections, seed=seed, backend=nx, type_as=X_s
        )

    Xps_coords, _ = projection_sphere_to_circle(
        X_s, n_projections=n_projections, projections=projections, seed=seed, backend=nx
    )

    if X_t is not None:
        Xpt_coords, _ = projection_sphere_to_circle(
            X_t,
            n_projections=n_projections,
            projections=projections,
            seed=seed,
            backend=nx,
        )

    projected_lcot = linear_circular_ot(
        Xps_coords.T, Xpt_coords.T, u_weights=a, v_weights=b
    )
    res = nx.mean(projected_lcot) ** (1 / 2)

    if log:
        return res, {"projections": projections, "projected_emds": projected_lcot}
    return res
