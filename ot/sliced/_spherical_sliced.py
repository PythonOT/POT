# -*- coding: utf-8 -*-
"""
Sliced Wasserstein distances on the Sphere solvers.
"""

# Author: Nicolas Courty <ncourty@irisa.fr>
# Author: Clément Bonet <clement.bonet.mapp@polytechnique.edu>
# Author: continuousml <continuousml@gmail.com>
#
# License: MIT License

import numpy as np

from ..backend import get_backend
from ._utils import (
    get_projections_sphere,
    get_random_projections,
    get_random_rotations,
    projection_sphere_to_circle,
    projection_sphere_to_ball,
)
from ..lp import (
    wasserstein_circle,
    semidiscrete_wasserstein2_unif_circle,
    linear_circular_ot,
    wasserstein_1d,
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
    >>> import numpy as np
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
    >>> import numpy as np
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
    >>> import numpy as np
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


def stereographic_sliced_wasserstein_sphere(
    X_s,
    X_t,
    a=None,
    b=None,
    n_projections=50,
    p=2,
    projections=None,
    n_rotations=0,
    rotations=None,
    eps=1e-6,
    seed=None,
    log=False,
):
    r"""Computes the stereographic spherical sliced Wasserstein distance from :ref:`[93] <references-s3w>`.

    General loss returned:

    .. math::
        S3W_p(\mu,\nu) = \left(\int_{\mathbb{S}^{d-2}} W_p^p(\theta_\# (\tfrac{1}{\pi}h_1\circ\phi_\epsilon)_\#\mu, \theta_\# (\tfrac{1}{\pi}h_1\circ\phi_\epsilon)_\#\nu)\ \mathrm{d}\sigma(\theta)\right)^{\frac{1}{p}}

    where :math:`\mu,\nu\in\mathcal{P}(S^{d-1})` are two probability measures on the
    sphere, :math:`\theta_\# \mu` stands for the pushforwards of the projection
    :math:`X \in \mathbb{R}^{d-1} \mapsto \langle \theta, X \rangle`,
    :math:`\phi_\epsilon` is the stereographic projection
    :math:`\phi(x) = \frac{2 x_{1:d-1}}{1-x_d}` restricted to the sphere without the
    :math:`\epsilon`-cap around the north pole (points with :math:`x_d > 1-\epsilon`
    are first mapped to the circle :math:`x_d = 1-\epsilon`), and
    :math:`h_1(x) = \mathrm{arccos}\left(\frac{1-\|x\|^2}{1+\|x\|^2}\right)\frac{x}{\|x\|}`
    is the injective defining function of :ref:`[93] <references-s3w>`, rescaled by
    :math:`\frac{1}{\pi}` to map the sphere to the unit ball.

    If ``n_rotations >= 1`` or ``rotations`` is provided, computes instead a
    Monte-Carlo approximation of the rotationally invariant extension

    .. math::
        RI\text{-}S3W_p(\mu,\nu) = \int_{\mathrm{SO}(d)} S3W_p(R_\#\mu, R_\#\nu)\ \mathrm{d}\omega(R)

    where :math:`\omega` is the normalized Haar measure on :math:`\mathrm{SO}(d)`.
    The generation cost of the rotations can be amortized over several calls by
    pregenerating a pool of rotations with
    :any:`ot.sliced.get_random_rotations` and passing a random subset of it as
    ``rotations`` at each call (ARI-S3W :ref:`[93] <references-s3w>`).

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
        Power p used for computing the stereographic spherical sliced Wasserstein
    projections: shape (dim-1, n_projections), optional
        Projection matrix (n_projections and seed are not used in this case)
    n_rotations : int, optional (default=0)
        Number of rotations used for the Monte-Carlo approximation of
        :math:`RI\text{-}S3W_p`. If 0, no rotation is applied and
        :math:`S3W_p` is computed.
    rotations: shape (n_rotations, dim, dim), optional
        Rotation matrices (n_rotations is not used in this case)
    eps: float, optional (default=1e-6)
        Size of the cap around the north pole excluded from the stereographic
        projection to ensure numerical stability
    seed: int or RandomState or None, optional
        Seed used for random number generator
    log: bool, optional
        if True, stereographic_sliced_wasserstein_sphere returns the projections
        and rotations used and the associated EMDs.

    Returns
    -------
    cost: float
        Stereographic Spherical Sliced Wasserstein Cost
    log: dict, optional
        log dictionary return only if log==True in parameters

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> n_samples_a = 20
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> X = X / np.sqrt(np.sum(X**2, -1, keepdims=True))
    >>> ot.stereographic_sliced_wasserstein_sphere(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0


    .. _references-s3w:
    References
    ----------
    .. [93] Tran, H., Bai, Y., Kothapalli, A., Shahbazi, A., Liu, X.,
       Diaz Martin, R., & Kolouri, S. (2024). Stereographic Spherical Sliced
       Wasserstein Distances. International Conference on Machine Learning.
    """
    d = X_s.shape[-1]

    nx = get_backend(X_s, X_t, a, b, projections, rotations)

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
        projections = get_random_projections(
            d - 1, n_projections, seed=seed, backend=nx, type_as=X_s
        )
        if seed is not None and not isinstance(seed, np.random.RandomState):
            # draw the rotations from the stream advanced by the projections
            seed = None
    else:
        n_projections = projections.shape[1]

    if rotations is None and n_rotations > 0:
        rotations = get_random_rotations(
            d, n_rotations, seed=seed, backend=nx, type_as=X_s
        )
    elif rotations is not None:
        n_rotations = rotations.shape[0]

    if rotations is not None:
        Xps = nx.einsum("kij, nj -> kni", rotations, X_s)
        Xpt = nx.einsum("kij, nj -> kni", rotations, X_t)
    else:
        n_rotations = 1
        Xps = X_s[None, :, :]
        Xpt = X_t[None, :, :]

    Xps = projection_sphere_to_ball(Xps, eps=eps, backend=nx)
    Xpt = projection_sphere_to_ball(Xpt, eps=eps, backend=nx)

    Xps = nx.reshape(
        nx.einsum("kni, il -> nkl", Xps, projections),
        (X_s.shape[0], n_rotations * n_projections),
    )
    Xpt = nx.reshape(
        nx.einsum("kni, il -> nkl", Xpt, projections),
        (X_t.shape[0], n_rotations * n_projections),
    )

    projected_emd = nx.reshape(
        wasserstein_1d(Xps, Xpt, a, b, p=p), (n_rotations, n_projections)
    )
    res = nx.mean(nx.mean(projected_emd, axis=-1) ** (1.0 / p))

    if log:
        return res, {
            "projections": projections,
            "rotations": rotations,
            "projected_emds": projected_emd,
        }
    return res
