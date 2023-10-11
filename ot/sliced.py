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
from .lp import wasserstein_circle, semidiscrete_wasserstein2_unif_circle


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

    if isinstance(seed, np.random.RandomState) and str(nx) == 'numpy':
        projections = seed.randn(d, n_projections)
    else:
        if seed is not None:
            nx.seed(seed)
        projections = nx.randn(d, n_projections, type_as=type_as)

    projections = projections / nx.sqrt(nx.sum(projections**2, 0, keepdims=True))
    return projections


def sliced_wasserstein_distance(X_s, X_t, a=None, b=None, n_projections=50, p=2,
                                projections=None, seed=None, log=False):
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
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(X_s.shape[1],
                                                                                                      X_t.shape[1]))

    if a is None:
        a = nx.full(n, 1 / n, type_as=X_s)
    if b is None:
        b = nx.full(m, 1 / m, type_as=X_s)

    d = X_s.shape[1]

    if projections is None:
        projections = get_random_projections(d, n_projections, seed, backend=nx, type_as=X_s)
    else:
        n_projections = projections.shape[1]

    X_s_projections = nx.dot(X_s, projections)
    X_t_projections = nx.dot(X_t, projections)

    projected_emd = wasserstein_1d(X_s_projections, X_t_projections, a, b, p=p)

    res = (nx.sum(projected_emd) / n_projections) ** (1.0 / p)
    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res


def max_sliced_wasserstein_distance(X_s, X_t, a=None, b=None, n_projections=50, p=2,
                                    projections=None, seed=None, log=False):
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
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(X_s.shape[1],
                                                                                                      X_t.shape[1]))

    if a is None:
        a = nx.full(n, 1 / n, type_as=X_s)
    if b is None:
        b = nx.full(m, 1 / m, type_as=X_s)

    d = X_s.shape[1]

    if projections is None:
        projections = get_random_projections(d, n_projections, seed, backend=nx, type_as=X_s)

    X_s_projections = nx.dot(X_s, projections)
    X_t_projections = nx.dot(X_t, projections)

    projected_emd = wasserstein_1d(X_s_projections, X_t_projections, a, b, p=p)

    res = nx.max(projected_emd) ** (1.0 / p)
    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res


def sliced_wasserstein_sphere(X_s, X_t, a=None, b=None, n_projections=50,
                              p=2, projections=None, seed=None, log=False):
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
    if a is not None and b is not None:
        nx = get_backend(X_s, X_t, a, b)
    else:
        nx = get_backend(X_s, X_t)

    n, d = X_s.shape
    m, _ = X_t.shape

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(X_s.shape[1],
                                                                                                      X_t.shape[1]))
    if nx.any(nx.abs(nx.sum(X_s**2, axis=-1) - 1) > 10**(-4)):
        raise ValueError("X_s is not on the sphere.")
    if nx.any(nx.abs(nx.sum(X_t**2, axis=-1) - 1) > 10**(-4)):
        raise ValueError("X_t is not on the sphere.")

    if projections is None:
        # Uniforms and independent samples on the Stiefel manifold V_{d,2}
        if isinstance(seed, np.random.RandomState) and str(nx) == 'numpy':
            Z = seed.randn(n_projections, d, 2)
        else:
            if seed is not None:
                nx.seed(seed)
            Z = nx.randn(n_projections, d, 2, type_as=X_s)

        projections, _ = nx.qr(Z)
    else:
        n_projections = projections.shape[0]

    # Projection on S^1
    # Projection on plane
    Xps = nx.einsum("ikj, lk -> ilj", projections, X_s)
    Xpt = nx.einsum("ikj, lk -> ilj", projections, X_t)

    # Projection on sphere
    Xps = Xps / nx.sqrt(nx.sum(Xps**2, -1, keepdims=True))
    Xpt = Xpt / nx.sqrt(nx.sum(Xpt**2, -1, keepdims=True))

    # Get coordinates on [0,1[
    Xps_coords = nx.reshape(get_coordinate_circle(nx.reshape(Xps, (-1, 2))), (n_projections, n))
    Xpt_coords = nx.reshape(get_coordinate_circle(nx.reshape(Xpt, (-1, 2))), (n_projections, m))

    projected_emd = wasserstein_circle(Xps_coords.T, Xpt_coords.T, u_weights=a, v_weights=b, p=p)
    res = nx.mean(projected_emd) ** (1 / p)

    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res


def sliced_wasserstein_sphere_unif(X_s, a=None, n_projections=50, seed=None, log=False):
    r"""Compute the 2-spherical sliced wasserstein w.r.t. a uniform distribution.

    .. math::
        SSW_2(\mu_n, \nu)

    where

    - :math:`\mu_n=\sum_{i=1}^n \alpha_i \delta_{x_i}`
    - :math:`\nu=\mathrm{Unif}(S^1)`

    Parameters
    ----------
    X_s: ndarray, shape (n_samples_a, dim)
        Samples in the source domain
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
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
    if a is not None:
        nx = get_backend(X_s, a)
    else:
        nx = get_backend(X_s)

    n, d = X_s.shape

    if nx.any(nx.abs(nx.sum(X_s**2, axis=-1) - 1) > 10**(-4)):
        raise ValueError("X_s is not on the sphere.")

    # Uniforms and independent samples on the Stiefel manifold V_{d,2}
    if isinstance(seed, np.random.RandomState) and str(nx) == 'numpy':
        Z = seed.randn(n_projections, d, 2)
    else:
        if seed is not None:
            nx.seed(seed)
        Z = nx.randn(n_projections, d, 2, type_as=X_s)

    projections, _ = nx.qr(Z)

    # Projection on S^1
    # Projection on plane
    Xps = nx.einsum("ikj, lk -> ilj", projections, X_s)

    # Projection on sphere
    Xps = Xps / nx.sqrt(nx.sum(Xps**2, -1, keepdims=True))

    # Get coordinates on [0,1[
    Xps_coords = nx.reshape(get_coordinate_circle(nx.reshape(Xps, (-1, 2))), (n_projections, n))

    projected_emd = semidiscrete_wasserstein2_unif_circle(Xps_coords.T, u_weights=a)
    res = nx.mean(projected_emd) ** (1 / 2)

    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res
