"""
Sliced OT Distances

"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty   <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

import numpy as np
from .backend import get_backend, NumpyBackend
from .utils import list_to_array, get_coordinate_circle
from .lp import (
    wasserstein_circle,
    semidiscrete_wasserstein2_unif_circle,
    linear_circular_ot,
)


def get_random_projections(d, n_projections, seed=None, backend=None, type_as=None):
    r"""
    Generates n_projections samples from the uniform on the unit sphere of dimension :math:`d-1`: :math:`\mathcal{U}(\mathcal{S}^{d-1})`

    Parameters
    ----------
    d : int
        dimension of the space
    n_projections : int
        number of samples requested
    seed: int or RandomState, optional
        Seed used for numpy random number generator
    backend:
        Backend to use for random generation

    Returns
    -------
    out: ndarray, shape (d, n_projections)
        The uniform unit vectors on the sphere

    Examples
    --------
    >>> n_projections = 100
    >>> d = 5
    >>> projs = get_random_projections(d, n_projections)
    >>> np.allclose(np.sum(np.square(projs), 0), 1.)  # doctest: +NORMALIZE_WHITESPACE
    True

    """

    if backend is None:
        nx = NumpyBackend()
    else:
        nx = backend

    if isinstance(seed, np.random.RandomState) and str(nx) == "numpy":
        projections = seed.randn(d, n_projections)
    else:
        if seed is not None:
            nx.seed(seed)
        projections = nx.randn(d, n_projections, type_as=type_as)

    projections = projections / nx.sqrt(nx.sum(projections**2, 0, keepdims=True))
    return projections


def sliced_wasserstein_distance(
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
    Computes a Monte-Carlo approximation of the p-Sliced Wasserstein distance

    .. math::
        \mathcal{SWD}_p(\mu, \nu) = \underset{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}{\mathbb{E}}\left(\mathcal{W}_p^p(\theta_\# \mu, \theta_\# \nu)\right)^{\frac{1}{p}}


    where :

    - :math:`\theta_\# \mu` stands for the pushforwards of the projection :math:`X \in \mathbb{R}^d \mapsto \langle \theta, X \rangle`


    Parameters
    ----------
    X_s : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    X_t : ndarray, shape (n_samples_b, dim)
        samples in the target domain
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,), optional
        samples weights in the target domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    p: float, optional
        Power p used for computing the sliced Wasserstein
    projections: shape (dim, n_projections), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    log: bool, optional
        if True, sliced_wasserstein_distance returns the projections used and their associated EMD.

    Returns
    -------
    cost: float
        Sliced Wasserstein Cost
    log : dict, optional
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_samples_a = 20
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> sliced_wasserstein_distance(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0

    References
    ----------

    .. [31] Bonneel, Nicolas, et al. "Sliced and radon wasserstein barycenters of measures." Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45
    """
    from .lp import wasserstein_1d

    X_s, X_t = list_to_array(X_s, X_t)

    if a is not None and b is not None and projections is None:
        nx = get_backend(X_s, X_t, a, b)
    elif a is not None and b is not None and projections is not None:
        nx = get_backend(X_s, X_t, a, b, projections)
    elif a is None and b is None and projections is not None:
        nx = get_backend(X_s, X_t, projections)
    else:
        nx = get_backend(X_s, X_t)

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

    X_s_projections = nx.dot(X_s, projections)
    X_t_projections = nx.dot(X_t, projections)

    projected_emd = wasserstein_1d(X_s_projections, X_t_projections, a, b, p=p)

    res = (nx.sum(projected_emd) / n_projections) ** (1.0 / p)
    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res


def max_sliced_wasserstein_distance(
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
    Computes a Monte-Carlo approximation of the max p-Sliced Wasserstein distance

    .. math::
        \mathcal{Max-SWD}_p(\mu, \nu) = \underset{\theta _in
        \mathcal{U}(\mathbb{S}^{d-1})}{\max} [\mathcal{W}_p^p(\theta_\#
        \mu, \theta_\# \nu)]^{\frac{1}{p}}

    where :

    - :math:`\theta_\# \mu` stands for the pushforwards of the projection :math:`\mathbb{R}^d \ni X \mapsto \langle \theta, X \rangle`


    Parameters
    ----------
    X_s : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    X_t : ndarray, shape (n_samples_b, dim)
        samples in the target domain
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,), optional
        samples weights in the target domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    p: float, optional =
        Power p used for computing the sliced Wasserstein
    projections: shape (dim, n_projections), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    log: bool, optional
        if True, sliced_wasserstein_distance returns the projections used and their associated EMD.

    Returns
    -------
    cost: float
        Sliced Wasserstein Cost
    log : dict, optional
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_samples_a = 20
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> sliced_wasserstein_distance(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0

    References
    ----------

    .. [35] Deshpande, I., Hu, Y. T., Sun, R., Pyrros, A., Siddiqui, N., Koyejo, S., ... & Schwing, A. G. (2019). Max-sliced wasserstein distance and its use for gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10648-10656).
    """
    from .lp import wasserstein_1d

    X_s, X_t = list_to_array(X_s, X_t)

    if a is not None and b is not None and projections is None:
        nx = get_backend(X_s, X_t, a, b)
    elif a is not None and b is not None and projections is not None:
        nx = get_backend(X_s, X_t, a, b, projections)
    elif a is None and b is None and projections is not None:
        nx = get_backend(X_s, X_t, projections)
    else:
        nx = get_backend(X_s, X_t)

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

    X_s_projections = nx.dot(X_s, projections)
    X_t_projections = nx.dot(X_t, projections)

    projected_emd = wasserstein_1d(X_s_projections, X_t_projections, a, b, p=p)

    res = nx.max(projected_emd) ** (1.0 / p)
    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res


def get_projections_sphere(d, n_projections, seed=None, backend=None, type_as=None):
    r"""
    Generates n_projections samples from the uniform distribution on the Stiefel manifold of dimension :math:`d\times 2`: :math:`\mathbb{V}_{d,2}=\{X \in \mathbb{R}^{d\times 2}, X^TX=I_2\}`

    Parameters
    ----------
    d : int
        dimension of the space
    n_projections : int
        number of samples requested
    seed: int or RandomState, optional
        Seed used for numpy random number generator
    backend:
        Backend to use for random generation
    type_as: optional
        Type to use for random generation

    Returns
    -------
    out: ndarray, shape (n_projections, d, 2)

    Examples
    --------
    >>> n_projections = 100
    >>> d = 5
    >>> projs = get_projections_sphere(d, n_projections)
    >>> np.allclose(np.einsum("nij, nik -> njk", projs, projs), np.eye(2))  # doctest: +NORMALIZE_WHITESPACE
    True
    """
    if backend is None:
        nx = NumpyBackend()
    else:
        nx = backend

    if isinstance(seed, np.random.RandomState) and str(nx) == "numpy":
        Z = seed.randn(n_projections, d, 2)
    else:
        if seed is not None:
            nx.seed(seed)
        Z = nx.randn(n_projections, d, 2, type_as=type_as)

    projections, _ = nx.qr(Z)
    return projections


def projection_sphere_to_circle(
    x, n_projections=50, projections=None, seed=None, backend=None
):
    r"""
    Projection of :math:`x\in S^{d-1}` on circles using coordinates on [0,1[.

    To get the projection on the circle, we use the following formula:
    .. math::
        P^U(x) = \frac{U^Tx}{\|U^Tx\|_2}

    where :math:`U` is a random matrix sampled from the uniform distribution on the Stiefel manifold of dimension :math:`d\times 2`: :math:`\mathbb{V}_{d,2}=\{X \in \mathbb{R}^{d\times 2}, X^TX=I_2\}`
    and :math:`x` is a point on the sphere. Then, we apply the function get_coordinate_circle to get the coordinates on :math:`[0,1[`.

    Parameters
    ----------
    x : ndarray, shape (n_samples, dim)
        samples on the sphere
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    projections: shape (n_projections, dim, 2), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    backend:
        Backend to use for random generation

    Returns
    -------
    Xp_coords: ndarray, shape (n_projections, n_samples)
        Coordinates of the projections on the circle
    """
    if backend is None:
        nx = get_backend(x)
    else:
        nx = backend

    n, d = x.shape

    if projections is None:
        projections = get_projections_sphere(
            d, n_projections, seed=seed, backend=nx, type_as=x
        )

    # Projection on S^1
    # Projection on plane
    Xp = nx.einsum("ikj, lk -> ilj", projections, x)

    # Projection on sphere
    Xp = Xp / nx.sqrt(nx.sum(Xp**2, -1, keepdims=True))

    # Get coordinates on [0,1[
    Xp_coords = nx.reshape(
        get_coordinate_circle(nx.reshape(Xp, (-1, 2))), (n_projections, n)
    )

    return Xp_coords, projections


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
    >>> n_samples_a = 20
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> X = X / np.sqrt(np.sum(X**2, -1, keepdims=True))
    >>> sliced_wasserstein_sphere(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0

    References
    ----------
    .. [46] Bonet, C., Berg, P., Courty, N., Septier, F., Drumetz, L., & Pham, M. T. (2023). Spherical sliced-wasserstein. International Conference on Learning Representations.
    """
    d = X_s.shape[-1]

    if a is not None and b is not None:
        nx = get_backend(X_s, X_t, a, b)
    else:
        nx = get_backend(X_s, X_t)

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
    >>> np.random.seed(42)
    >>> x0 = np.random.randn(500,3)
    >>> x0 = x0 / np.sqrt(np.sum(x0**2, -1, keepdims=True))
    >>> ssw = sliced_wasserstein_sphere_unif(x0, seed=42)
    >>> np.allclose(sliced_wasserstein_sphere_unif(x0, seed=42), 0.01734, atol=1e-3)
    True

    References:
    -----------
    .. [46] Bonet, C., Berg, P., Courty, N., Septier, F., Drumetz, L., & Pham, M. T. (2023). Spherical sliced-wasserstein. International Conference on Learning Representations.
    """
    d = X_s.shape[-1]

    if a is not None:
        nx = get_backend(X_s, a)
    else:
        nx = get_backend(X_s)

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
        Samples in the target domain. If None, computes the distance against the uniform distribution on the sphere.
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
        if True, linear_sliced_wasserstein_sphere returns the projections used and their associated LCOT.

    Returns
    -------
    cost: float
        Linear Spherical Sliced Wasserstein Cost
    log: dict, optional
        log dictionary return only if log==True in parameters

    Examples
    ---------
    >>> n_samples_a = 20
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> X = X / np.sqrt(np.sum(X**2, -1, keepdims=True))
    >>> linear_sliced_wasserstein_sphere(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0


    .. _references-lssot:
    References
    ----------
    .. [79] Liu, X., Bai, Y., Martín, R. D., Shi, K., Shahbazi, A., Landman, B. A., Chang, C., & Kolouri, S. (2025). Linear Spherical Sliced Optimal Transport: A Fast Metric for Comparing Spherical Data. International Conference on Learning Representations.
    """
    d = X_s.shape[-1]

    if a is not None and b is not None:
        nx = get_backend(X_s, X_t, a, b)
    else:
        nx = get_backend(X_s, X_t)

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
