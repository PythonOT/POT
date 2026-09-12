# -*- coding: utf-8 -*-
"""
Sliced Wasserstein distances solvers: sliced and max-sliced.
"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty   <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

from ..backend import get_backend
from ..utils import list_to_array, apply_scaler
from ._utils import get_random_projections, get_projections_spiral
from ..lp import wasserstein_1d


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
    scaler=None,
    sampling_slices="uniform",
):
    r"""
    Computes a Monte-Carlo approximation of the p-Sliced Wasserstein distance

    .. math::
        \mathcal{SWD}_p(\mu, \nu) = \underset{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}{\mathbb{E}}\left(\mathcal{W}_p^p(\theta_\# \mu, \theta_\# \nu)\right)^{\frac{1}{p}}


    where :

    - :math:`\theta_\# \mu` stands for the pushforwards of the projection :math:`X \in \mathbb{R}^d \mapsto \langle \theta, X \rangle`

    By default, the projection directions :math:`\theta` are sampled uniformly
    at random. Setting ``sampling_slices`` to ``"spiral_qmc"`` or ``"randomized_spiral_qmc"`` instead
    uses Quasi-Monte Carlo point sets on the sphere (generalized spiral
    points), which can reduce the approximation error for a given
    ``n_projections`` [95]. These two options are
    only implemented for ``dim == 3``.

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
        Projection matrix (n_projections, seed and sampling_slices are not
        used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator. Ignored if
        ``sampling_slices="spiral_qmc"`` (the deterministic point set does not
        depend on a seed).
    log: bool, optional
        if True, sliced_wasserstein_distance returns the projections used and their associated EMD.
    scaler: None, object with .transform(), or callable, optional
        Preprocessing applied to X_s and X_t before computing the distance.
        Useful for normalizing inputs when features have very different scales.

        - ``None`` : no preprocessing (default)
        - Object with ``.transform()`` method : e.g. an :class:`ot.utils.DataScaler`
          fitted on a representative sample. This is the recommended way to get
          stable, consistent normalization across multiple calls (e.g. when
          using SWD as a loss in mini-batch training).
        - Callable : any function, lambda, or PyTorch transform applied
          directly as ``scaler(X_s)`` and ``scaler(X_t)``.

        See :class:`ot.utils.DataScaler` for a backend-aware scaler that supports
        joint fitting on multiple distributions.
    sampling_slices: str, optional
        Method used to sample the projection directions when ``projections``
        is not provided directly. One of:

        - ``"uniform"`` (default): directions sampled uniformly at random on
          the sphere (Monte Carlo).
        - ``"spiral_qmc"``: deterministic Quasi-Sliced Wasserstein directions via
          generalized spiral points. Only implemented for ``dim == 3``.
        - ``"randomized_spiral_qmc"``: Randomized Quasi-Sliced Wasserstein -- the same spiral
          point set as ``"spiral_qmc"``, with a random rotation applied, giving an
          unbiased estimator suitable for stochastic optimization. Only
          implemented for ``dim == 3``.

    Returns
    -------
    cost: float
        Sliced Wasserstein Cost
    log : dict, optional
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import numpy as np
    >>> n_samples_a = 20
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> sliced_wasserstein_distance(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0

    References
    ----------

    .. [31] Bonneel, Nicolas, et al. "Sliced and radon wasserstein barycenters of measures." Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45
    .. [95] Nguyen, K., Bariletto, N., & Ho, N. (2024). "Quasi-Monte Carlo for 3D Sliced Wasserstein." International Conference on Learning Representations (ICLR).
    .. [96] Rakhmanov, E. A., Saff, E. B., & Zhou, Y. M. (1994). "Minimal Discrete Energy on the Sphere." Mathematical Research Letters, 1(6), 647-662.
    """

    X_s, X_t = list_to_array(X_s, X_t)

    nx = get_backend(X_s, X_t, a, b, projections)

    X_s, X_t = apply_scaler(X_s, X_t, scaler)

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

    randomized = sampling_slices.startswith("randomized_")
    method = sampling_slices.removeprefix("randomized_")

    if projections is None:
        if sampling_slices == "uniform":
            projections = get_random_projections(
                d, n_projections, seed, backend=nx, type_as=X_s
            )
        elif method == "spiral_qmc":
            projections = get_projections_spiral(
                d,
                n_projections,
                randomized=randomized,
                seed=seed,
                backend=nx,
                type_as=X_s,
            )
        else:
            raise ValueError(
                f"Unknown sampling_slices method '{sampling_slices}', "
                "must be one of 'uniform', 'spiral_qmc', 'randomized_spiral_qmc'"
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
    scaler=None,
):
    r"""
    Computes a Monte-Carlo approximation of the max p-Sliced Wasserstein distance

    .. math::
        \mathcal{Max-SWD}_p(\mu, \nu) = \underset{\theta \in
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
    scaler : None, object with .transform(), or callable, optional
        Preprocessing applied to X_s and X_t before computing the distance.
        Useful for normalizing inputs when features have very different scales.

        - ``None`` : no preprocessing (default)
        - Object with ``.transform()`` method : e.g. an :class:`ot.utils.DataScaler`
          fitted on a representative sample. This is the recommended way to get
          stable, consistent normalization across multiple calls (e.g. when
          using SWD as a loss in mini-batch training).
        - Callable : any function, lambda, or PyTorch transform applied
          directly as ``scaler(X_s)`` and ``scaler(X_t)``.

        See :class:`ot.utils.DataScaler` for a backend-aware scaler that supports
        joint fitting on multiple distributions.

    Returns
    -------
    cost: float
        Sliced Wasserstein Cost
    log : dict, optional
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import numpy as np
    >>> n_samples_a = 20
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> sliced_wasserstein_distance(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0

    References
    ----------

    .. [35] Deshpande, I., Hu, Y. T., Sun, R., Pyrros, A., Siddiqui, N., Koyejo, S., ... & Schwing, A. G. (2019). Max-sliced wasserstein distance and its use for gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10648-10656).
    """

    X_s, X_t = list_to_array(X_s, X_t)

    nx = get_backend(X_s, X_t, a, b, projections)

    X_s, X_t = apply_scaler(X_s, X_t, scaler)

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
