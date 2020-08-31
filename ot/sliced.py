"""
Sliced Wasserstein Distance.

"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#
# License: MIT License


import numpy as np


def get_random_projections(n_projections, d, seed=None):
    r"""
    Generates n_projections samples from the uniform on the unit sphere of dimension d-1: :math:`\mathcal{U}(\mathcal{S}^{d-1})`

    Parameters
    ----------
    n_projections : int
        number of samples requested
    d : int
        dimension of the space
    seed: int or RandomState, optional
        Seed used for numpy random number generator

    Returns
    -------
    out: ndarray, shape (n_projections, d)
        The uniform unit vectors on the sphere

    Examples
    --------
    >>> n_projections = 100
    >>> d = 5
    >>> projs = get_random_projections(n_projections, d)
    >>> np.allclose(np.sum(np.square(projs), 1), 1.)  # doctest: +NORMALIZE_WHITESPACE
    True

    """

    if not isinstance(seed, np.random.RandomState):
        random_state = np.random.RandomState(seed)
    else:
        random_state = seed

    projections = random_state.normal(0., 1., [n_projections, d])
    norm = np.linalg.norm(projections, ord=2, axis=1, keepdims=True)
    projections = projections / norm
    return projections


def sliced_wasserstein_distance(X_s, X_t, a=None, b=None, n_projections=50, seed=None, log=False):
    r"""
    Computes a Monte-Carlo approximation of the 2-Sliced Wasserstein distance

    .. math::
        \mathcal{SWD}_2(\mu, \nu) = \underset{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}{\mathbb{E}}[\mathcal{W}_2^2(\theta_\# \mu, \theta_\# \nu)]^{\frac{1}{2}}

    where :

    - :math:`\theta_\# \mu` stands for the pushforwars of the projection :math:`\mathbb{R}^d \ni X \mapsto \langle \theta, X \rangle`


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
    seed: int or RandomState or None, optional
        Seed used for numpy random number generator
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
    >>> reg = 0.1
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> sliced_wasserstein_distance(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0

    References
    ----------

    .. [31] Bonneel, Nicolas, et al. "Sliced and radon wasserstein barycenters of measures." Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45
    """
    from .lp import emd2_1d

    X_s = np.asanyarray(X_s)
    X_t = np.asanyarray(X_t)

    n = X_s.shape[0]
    m = X_t.shape[0]

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(X_s.shape[1],
                                                                                                      X_t.shape[1]))

    if a is None:
        a = np.full(n, 1 / n)
    if b is None:
        b = np.full(m, 1 / m)

    d = X_s.shape[1]

    projections = get_random_projections(n_projections, d, seed)

    X_s_projections = np.dot(projections, X_s.T)
    X_t_projections = np.dot(projections, X_t.T)

    if log:
        projected_emd = np.empty(n_projections)
    else:
        projected_emd = None

    res = 0.

    for i, (X_s_proj, X_t_proj) in enumerate(zip(X_s_projections, X_t_projections)):
        emd = emd2_1d(X_s_proj, X_t_proj, a, b, log=False, dense=False)
        if projected_emd is not None:
            projected_emd[i] = emd
        res += emd

    res = (res / n_projections) ** 0.5
    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res
