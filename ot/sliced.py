"""
Sliced OT Distances

"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty   <ncourty@irisa.fr>
# License: MIT License


import numpy as np
from ot.backend import get_backend


def quantile_function(qs, cws, xs):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    r""" Computes the quantile function of an empirical distribution

    Parameters
    ----------
    qs: array-like, shape (n,)
        Quantiles at which the quantile function is evaluated
    cws: array-like, shape (m, ...)
        cumulative weights of the 1D empirical distribution, if batched, must be similar to xs
    xs: array-like, shape (n, ...)
        locations of the 1D empirical distribution, batched against the `xs.ndim - 1` first dimensions

    Returns
    -------
    q: array-like, shape (..., n)
        The quantiles of the distribution
    """
    nx = get_backend(qs, cws)
    n = xs.shape[0]
    if nx.__name__ == 'torch':
        # this is to ensure the best performance for torch searchsorted
        # and avoid a warninng related to non-contiguous arrays
        cws = cws.T.contiguous()
        qs = qs.T.contiguous()
    else:
        cws = cws.T
        qs = qs.T
    idx = nx.searchsorted(cws, qs).T
    return nx.take_along_axis(xs, nx.clip(idx, 0, n - 1), axis=0)


def emd1D(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    """
    Computes the 1 dimensional OT loss [15] between two (batched) empirical distributions
    ..math:
        ot_{loss} &= \int_0^1 |cdf_u^{-1}(q)  cdf_v^{-1}(q)|^p dq

    It is formally the p-Wasserstein distance raised to the power p.
    We do so in a vectorized way by first building the individual quantile functions then integrating them.

    Parameters
    ----------
    u_values: array-like, shape (n, ...)
        locations of the first empirical distribution
    v_values: array-like, shape (m, ...)
        locations of the second empirical distribution
    u_weights: array-like, shape (n, ...), optional
        weights of the first empirical distribution, if None then uniform weights are used
    v_weights: array-like, shape (m, ...), optional
        weights of the second empirical distribution, if None then uniform weights are used
    p: int, optional
        order of the ground metric used, should be at least 1 (see [2, Chap. 2], default is 1
    require_sort: bool, optional
        sort the distributions atoms locations, if False we will consider they have been sorted prior to being passed to
        the function, default is True

    Returns
    -------
    cost: float/array-like, shape (...)
        the batched EMD

    Examples
    --------
    Simple example:
    >>> from scipy.stats import wasserstein_distance
    >>> import ot
    >>> import torch
    >>> from ot.sliced import emd1D_nx
    >>> np.random.seed(0)
    >>> num=1000
    >>> x = torch.linspace(0,5,num)
    >>> rho_u = torch.abs(torch.randn(num))
    >>> rho_u /= torch.sum(rho_u)
    >>> rho_v = torch.abs(torch.randn(num))
    >>> rho_v /= torch.sum(rho_v)
    >>> print(emd1D(x, x, rho_u, rho_v, p=1))
    >>> print(emd1D(x.numpy(), x.numpy(), rho_u.numpy(), rho_v.numpy(), p=1))
    >>> print('for comparison')
    >>> print(wasserstein_distance(x.numpy(),x.numpy(),rho_u.numpy(),rho_v.numpy()))
    tensor(0.0410)
    0.04097907
    Scipy Wasserstein_distance:
    0.04097760462819538

    References
    ----------
    .. [15] Peyré, G., & Cuturi, M. (2018). [Computational Optimal Transport](https://arxiv.org/pdf/1803.00567.pdf)

    """

    assert p >= 1, "The OT loss is only valid for p>=1, {p} was given".format(p=p)

    if u_weights is not None and v_weights is not None:
        nx = get_backend(u_values, v_values, u_weights, v_weights)
    else:
        nx = get_backend(u_values, v_values)

    n = u_values.shape[0]
    m = v_values.shape[0]

    if u_weights is None:
        u_weights = nx.full(u_values.shape, 1. / n)
    elif u_weights.ndim != u_values.ndim:
        u_weights = nx.repeat(u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = nx.full(v_values.shape, 1. / m)
    elif v_weights.ndim != v_values.ndim:
        v_weights = nx.repeat(v_weights[..., None], v_values.shape[-1], -1)

    if require_sort:
        u_sorter = nx.argsort(u_values, 0)
        u_values = nx.take_along_axis(u_values, u_sorter, 0)

        v_sorter = nx.argsort(v_values, 0)
        v_values = nx.take_along_axis(v_values, v_sorter, 0)

        u_weights = nx.take_along_axis(u_weights, u_sorter, 0)
        v_weights = nx.take_along_axis(v_weights, v_sorter, 0)

    u_cumweights = nx.cumsum(u_weights, 0)
    v_cumweights = nx.cumsum(v_weights, 0)

    qs = nx.sort(nx.concatenate((u_cumweights, v_cumweights), 0), 0)
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)
    qs = nx.zero_pad(qs, pad_with=[(1, 0)] + (qs.ndim - 1) * [(0, 0)])
    delta = qs[1:, ...] - qs[:-1, ...]
    diff_quantiles = nx.abs(u_quantiles - v_quantiles)

    if p == 1:
        return nx.sum(delta * nx.abs(diff_quantiles), axis=0)
    return nx.sum(delta * nx.power(diff_quantiles, p), axis=0)


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
