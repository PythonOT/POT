"""
Sliced OT Distances
"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty   <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@polytechnique.edu>
#         Eloi Tanguy <eloi.tanguy@math.cnrs.fr>
#
# License: MIT License

import numpy as np
from .backend import get_backend, NumpyBackend
from .utils import list_to_array, get_coordinate_circle, dist
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


def sliced_permutations(X, Y, thetas=None, n_proj=None, log=False, backend=None):
    r"""
    Computes all the permutations that sort the projections of two `(n, d)`
    datasets `X` and `Y` on the directions `thetas`.
    Each permutation `perm[:, k]` is such that each `X[i, :]` is matched
    to `Y[perm[i, k], :]` when projected on `thetas[k, :]`.

    Parameters
    ----------
    X : array-like, shape (n, d)
        The first set of vectors.
    Y : array-like, shape (n, d)
        The second set of vectors.
    thetas : array-like, shape (n_proj, d), optional
        The projection directions. If None, random directions will be generated.
        Default is None.
    n_proj : int, optional
        The number of projection directions. Required if thetas is None.
    log : bool, optional
        If True, returns additional logging information. Default is False.
    backend : ot.backend, optional
        Backend to use for computations. If None, the backend is inferred from the input arrays. Default is None.

    Returns
    -------
    perm : array-like, shape (n, n_proj)
        All sliced permutations.
    log_dict : dict, optional
        A dictionary containing intermediate computations for logging purposes.
        Returned only if `log` is True.
    """
    assert (
        X.shape == Y.shape
    ), f"X ({X.shape}) and Y ({Y.shape}) must have the same shape"
    nx = get_backend(X, Y) if backend is None else backend
    d = X.shape[1]
    do_draw_thetas = thetas is None
    if do_draw_thetas:  # create thetas (n_proj, d)
        thetas = get_random_projections(d, n_proj, backend=nx).T

    # project on each theta: (n, d) -> (n, n_proj)
    X_theta = X @ thetas.T  # shape (n, n_proj)
    Y_theta = Y @ thetas.T  # shape (n, n_proj)

    # sigma[:, i_proj] is a permutation sorting X_theta[:, i_proj]
    sigma = nx.argsort(X_theta, axis=0)  # (n, n_proj)
    tau = nx.argsort(Y_theta, axis=0)  # (n, n_proj)

    # perm[:, i_proj] is tau[:, i_proj] o sigma[:, i_proj]^{-1}
    perm = nx.take_along_axis(tau, nx.argsort(sigma, axis=0), axis=0)  # (n, n_proj)

    if log:
        log_dict = {
            "X_theta": X_theta,
            "Y_theta": Y_theta,
            "sigma": sigma,
            "tau": tau,
            "perm": perm,
        }
        if do_draw_thetas:
            log_dict["thetas"] = thetas
        return perm, log_dict
    else:
        return perm


def min_pivot_sliced(
    X, Y, thetas=None, order=2, n_proj=None, log=False, warm_perm=None
):
    r"""
    Computes the cost and permutation associated to the min-Pivot Sliced
    Discrepancy (introduced as SWGG in [81] and studied further in [82]). Given
    the supports `X` and `Y` of two discrete uniform measures with `n` atoms in
    dimension `d`, the min-Pivot Sliced Discrepancy goes through `n_proj`
    different projections of the measures on random directions, and retains the
    permutation that yields the lowest cost between `X` and `Y` (compared
    in :math:`\mathbb{R}^d`).

    .. math::
        \mathrm{min\text{-}PS}_p^p(X, Y) \approx
        \min_{k \in [1, n_{\mathrm{proj}}]} \left(
        \frac{1}{n} \sum_{i=1}^n \|X_i - Y_{\sigma_k(i)}\|_2^p \right),

    where :math:`\sigma_k` is a permutation such that ordering the projections
    on the axis `thetas[k, :]` matches `X[i, :]` to `Y[\sigma_k(i), :]`.

    .. note::
        The computation ignores potential ambiguities in the projections: if two points from a same measure have the same projection on a direction, then multiple sorting permutations are possible. To avoid combinatorial explosion, only one permutation is retained: this strays from theory in pathological cases.

    Parameters
    ----------
    X : array-like, shape (n, d)
        The first set of vectors.
    Y : array-like, shape (n, d)
        The second set of vectors.
    thetas : array-like, shape (n_proj, d), optional
        The projection directions. If None, random directions will be generated. Default is None.
    order : int, optional
        Power to elevate the norm. Default is 2.
    n_proj : int, optional
        The number of projection directions. Required if thetas is None.
    log : bool, optional
        If True, returns additional logging information. Default is False.
    warm_perm : array-like, shape (n,), optional
        A permutation to add to the permutation list. Default is None.

    Returns
    -------
    perm : array-like, shape (n,)
        The permutation that minimizes the cost.
    min_cost : float
        The minimum cost corresponding to the optimal permutation.
    log_dict : dict, optional
        A dictionary containing intermediate computations for logging purposes.
        Returned only if `log` is True.

    References
    ----------
    .. [81] Mahey, G., Chapel, L., Gasso, G., Bonet, C., & Courty, N. (2023). Fast Optimal Transport through Sliced Generalized Wasserstein Geodesics. Advances in Neural Information Processing Systems, 36, 35350–35385.

    .. [82] Tanguy, E., Chapel, L., Delon, J. (2025). Sliced Optimal Transport Plans. arXiv preprint 2506.03661.
    """
    assert (
        X.shape == Y.shape
    ), f"X ({X.shape}) and Y ({Y.shape}) must have the same shape"
    n = X.shape[0]
    nx = get_backend(X, Y)
    log_dict = {}

    if log:
        perm, log_dict = sliced_permutations(
            X, Y, thetas=thetas, n_proj=n_proj, log=True, backend=nx
        )
    else:
        perm = sliced_permutations(
            X, Y, thetas=thetas, n_proj=n_proj, log=False, backend=nx
        )

    # add the 'warm perm' to permutations to test
    if warm_perm is not None:
        perm = nx.concatenate([perm, warm_perm[:, None]], axis=1)
        if log:
            log_dict["perm"] = perm

    min_cost = None
    idx_min_cost = None
    costs = []

    for k in range(perm.shape[-1]):
        cost = nx.sum(nx.abs(X - Y[perm[:, k]]) ** order) / n
        if min_cost is None or cost < min_cost:
            min_cost = cost
            idx_min_cost = k
        if log:
            costs.append(cost)

    min_perm = perm[:, idx_min_cost]

    if log:
        log_dict["costs"] = costs
        log_dict["idx_min_cost"] = idx_min_cost
        return min_perm, min_cost, log_dict
    else:
        return min_perm, min_cost


def expected_sliced(X, Y, thetas=None, n_proj=None, order=2, log=False, beta=0.0):
    r"""
    Computes the Expected Sliced cost and plan between two `(n, d)`
    datasets `X` and `Y`. Given a set of `n_proj` projection directions,
    the expected sliced plan is obtained by averaging the `n_proj` 1d optimal
    transport plans between the projections of `X` and `Y` on each direction.
    Expected Sliced was introduced in [83] and further studied in [82].

    .. note::
        The computation ignores potential ambiguities in the projections: if two points from a same measure have the same projection on a direction, then multiple sorting permutations are possible. To avoid combinatorial explosion, only one permutation is retained: this strays from theory in pathological cases.

    Parameters
    ----------
    X : torch.Tensor
        A tensor of shape (n, d) representing the first set of vectors.
    Y : torch.Tensor
        A tensor of shape (n, d) representing the second set of vectors.
    thetas : torch.Tensor, optional
        A tensor of shape (n_proj, d) representing the projection directions.
        If None, random directions will be generated. Default is None.
    n_proj : int, optional
        The number of projection directions. Required if thetas is None.
    order : int, optional
        Power to elevate the norm. Default is 2.
    log : bool, optional
        If True, returns additional logging information. Default is False.
    beta : float, optional
        Inverse-temperature parameter which weights each projection's contribution to the expected plan. Default is 0 (uniform weighting).

    Returns
    -------
    plan : torch.Tensor
        A tensor of shape (n_proj, n, n) representing the expected sliced plan.
    log_dict : dict, optional
        A dictionary containing intermediate computations for logging purposes.
        Returned only if `log` is True.

    References
    ----------
    .. [82] Tanguy, E., Chapel, L., Delon, J. (2025). Sliced Optimal Transport Plans. arXiv preprint 2506.03661.

    .. [83] Liu, X., Diaz Martin, R., Bai Y., Shahbazi A., Thorpe M., Aldroubi A., Kolouri, S. (2024). Expected Sliced Transport Plans. International Conference on Learning Representations.
    """
    assert (
        X.shape == Y.shape
    ), f"X ({X.shape}) and Y ({Y.shape}) must have the same shape"
    nx = get_backend(X, Y)
    n = X.shape[0]
    log_dict = {}
    if log:
        perm, log_dict = sliced_permutations(
            X, Y, thetas=thetas, n_proj=n_proj, log=log, backend=nx
        )
    else:
        perm = sliced_permutations(
            X, Y, thetas=thetas, n_proj=n_proj, log=log, backend=nx
        )
    plan = nx.zeros((n, n), type_as=X)
    n_proj = perm.shape[1]
    range_array = nx.arange(n, type_as=X)

    if beta != 0.0:  # computing the temperature weighting
        log_factors = nx.zeros(n_proj, type_as=X)  # for beta weighting
        for k in range(n_proj):
            cost_k = nx.sum(nx.abs(X - Y[perm[:, k]]) ** order) / n
            log_factors[k] = -beta * cost_k
        weights = nx.exp(log_factors - nx.logsumexp(log_factors))

    else:  # uniform weights
        weights = nx.ones(n_proj, type_as=X) / n_proj

    for k in range(n_proj):  # populating the expected plan
        # 1 / n is because is a permutation of [1, n]
        plan[range_array, perm[:, k]] += (1 / n) * weights[k]

    cost = (dist(X, Y, p=order) * plan).sum()

    if log:
        return plan, cost, log_dict
    else:
        return plan, cost
