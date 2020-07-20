"""
Sliced Wasserstein Distance.

"""

# Author: Adrien Corenflos <adrien.corenflos@gmail.com>
#
# License: MIT License


import numpy as np


def _random_projections(n_projections, dimension, random_state):
    """Samples n_projections times dimension normal distributions"""
    projections = random_state.normal(0., 1., [n_projections, dimension])
    norm = np.linalg.norm(projections, ord=2, axis=1, keepdims=True)
    projections = projections / norm
    return projections


def sliced(X_s, X_t, a=None, b=None, n_projections=50, seed=None):
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
    a : ndarray, shape (n_samples_a,)
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,)
        samples weights in the target domain
    n_projections : int
        Number of projections used for the Monte-Carlo approximation
    seed: int or RandomState or None
        Seed used for numpy random number generator

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
    >>> sliced(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.

    References
    ----------

    .. [1] S. Kolouri et al., Generalized Sliced Wasserstein Distances, Advances in Neural Information Processing Systems (NIPS) 33, 2019
    """
    from .lp import emd2_1d

    X_s = np.asanyarray(X_s)
    X_t = np.asanyarray(X_t)

    n = X_s.shape[0]
    m = X_t.shape[0]

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            f"X_s and X_t must have the same number of dimensions {X_s.shape[1]} and {X_t.shape[1]} respectively given")

    if a is None:
        a = np.full(n, 1 / n)
    if b is None:
        b = np.full(m, 1 / m)

    d = X_s.shape[1]

    if not isinstance(seed, np.random.RandomState):
        random_state = np.random.RandomState(seed)
        projections = _random_projections(n_projections, d, random_state)
    else:
        projections = _random_projections(n_projections, d, seed)

    res = 0.
    for projection in projections:
        X_s_proj = X_s @ projection
        X_t_proj = X_t @ projection
        res += emd2_1d(X_s_proj, X_t_proj, a, b, log=False, dense=False)
    return (res / n_projections) ** 0.5
