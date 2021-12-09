# -*- coding: utf-8 -*-
"""
Exact solvers for the 1D Wasserstein distance using cvxopt
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
# Author: Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
import warnings

from .emd_wrap import emd_1d_sorted
from ..backend import get_backend
from ..utils import list_to_array


def quantile_function(qs, cws, xs):
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


def wasserstein_1d(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    r"""
    Computes the 1 dimensional OT loss [15] between two (batched) empirical
    distributions

    .. math:
        OT_{loss} = \int_0^1 |cdf_u^{-1}(q)  cdf_v^{-1}(q)|^p dq

    It is formally the p-Wasserstein distance raised to the power p.
    We do so in a vectorized way by first building the individual quantile functions then integrating them.

    This function should be preferred to `emd_1d` whenever the backend is
    different to numpy, and when gradients over
    either sample positions or weights are required.

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

    References
    ----------
    .. [15] Peyré, G., & Cuturi, M. (2018). Computational Optimal Transport.

    """

    assert p >= 1, "The OT loss is only valid for p>=1, {p} was given".format(p=p)

    if u_weights is not None and v_weights is not None:
        nx = get_backend(u_values, v_values, u_weights, v_weights)
    else:
        nx = get_backend(u_values, v_values)

    n = u_values.shape[0]
    m = v_values.shape[0]

    if u_weights is None:
        u_weights = nx.full(u_values.shape, 1. / n, type_as=u_values)
    elif u_weights.ndim != u_values.ndim:
        u_weights = nx.repeat(u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = nx.full(v_values.shape, 1. / m, type_as=v_values)
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
    qs = nx.zero_pad(qs, pad_width=[(1, 0)] + (qs.ndim - 1) * [(0, 0)])
    delta = qs[1:, ...] - qs[:-1, ...]
    diff_quantiles = nx.abs(u_quantiles - v_quantiles)

    if p == 1:
        return nx.sum(delta * nx.abs(diff_quantiles), axis=0)
    return nx.sum(delta * nx.power(diff_quantiles, p), axis=0)


def emd_1d(x_a, x_b, a=None, b=None, metric='sqeuclidean', p=1., dense=True,
           log=False):
    r"""Solves the Earth Movers distance problem between 1d measures and returns
    the OT matrix


    .. math::
        \gamma = arg\min_\gamma \sum_i \sum_j \gamma_{ij} d(x_a[i], x_b[j])

        s.t. \gamma 1 = a,
             \gamma^T 1= b,
             \gamma\geq 0
    where :

    - d is the metric
    - x_a and x_b are the samples
    - a and b are the sample weights

    When 'minkowski' is used as a metric, :math:`d(x, y) = |x - y|^p`.

    Uses the algorithm detailed in [1]_

    Parameters
    ----------
    x_a : (ns,) or (ns, 1) ndarray, float64
        Source dirac locations (on the real line)
    x_b : (nt,) or (ns, 1) ndarray, float64
        Target dirac locations (on the real line)
    a : (ns,) ndarray, float64, optional
        Source histogram (default is uniform weight)
    b : (nt,) ndarray, float64, optional
        Target histogram (default is uniform weight)
    metric: str, optional (default='sqeuclidean')
        Metric to be used. Only strings listed in :func:`ot.dist` are accepted.
        Due to implementation details, this function runs faster when
        `'sqeuclidean'`, `'cityblock'`,  or `'euclidean'` metrics are used.
    p: float, optional (default=1.0)
         The p-norm to apply for if metric='minkowski'
    dense: boolean, optional (default=True)
        If True, returns math:`\gamma` as a dense ndarray of shape (ns, nt).
        Otherwise returns a sparse representation using scipy's `coo_matrix`
        format. Due to implementation details, this function runs faster when
        `'sqeuclidean'`, `'minkowski'`, `'cityblock'`,  or `'euclidean'` metrics
        are used.
    log: boolean, optional (default=False)
        If True, returns a dictionary containing the cost.
        Otherwise returns only the optimal transportation matrix.

    Returns
    -------
    gamma: (ns, nt) ndarray
        Optimal transportation matrix for the given parameters
    log: dict
        If input log is True, a dictionary containing the cost


    Examples
    --------

    Simple example with obvious solution. The function emd_1d accepts lists and
    performs automatic conversion to numpy arrays

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> x_a = [2., 0.]
    >>> x_b = [0., 3.]
    >>> ot.emd_1d(x_a, x_b, a, b)
    array([[0. , 0.5],
           [0.5, 0. ]])
    >>> ot.emd_1d(x_a, x_b)
    array([[0. , 0.5],
           [0.5, 0. ]])

    References
    ----------

    .. [1]  Peyré, G., & Cuturi, M. (2017). "Computational Optimal
        Transport", 2018.

    See Also
    --------
    ot.lp.emd : EMD for multidimensional distributions
    ot.lp.emd2_1d : EMD for 1d distributions (returns cost instead of the
        transportation matrix)
    """
    a, b, x_a, x_b = list_to_array(a, b, x_a, x_b)
    nx = get_backend(x_a, x_b)

    assert (x_a.ndim == 1 or x_a.ndim == 2 and x_a.shape[1] == 1), \
        "emd_1d should only be used with monodimensional data"
    assert (x_b.ndim == 1 or x_b.ndim == 2 and x_b.shape[1] == 1), \
        "emd_1d should only be used with monodimensional data"

    # if empty array given then use uniform distributions
    if a is None or a.ndim == 0 or len(a) == 0:
        a = nx.ones((x_a.shape[0],), type_as=x_a) / x_a.shape[0]
    if b is None or b.ndim == 0 or len(b) == 0:
        b = nx.ones((x_b.shape[0],), type_as=x_b) / x_b.shape[0]

    # ensure that same mass
    np.testing.assert_almost_equal(
        nx.to_numpy(nx.sum(a, axis=0)),
        nx.to_numpy(nx.sum(b, axis=0)),
        err_msg='a and b vector must have the same sum'
    )
    b = b * nx.sum(a) / nx.sum(b)

    x_a_1d = nx.reshape(x_a, (-1,))
    x_b_1d = nx.reshape(x_b, (-1,))
    perm_a = nx.argsort(x_a_1d)
    perm_b = nx.argsort(x_b_1d)

    G_sorted, indices, cost = emd_1d_sorted(
        nx.to_numpy(a[perm_a]).astype(np.float64),
        nx.to_numpy(b[perm_b]).astype(np.float64),
        nx.to_numpy(x_a_1d[perm_a]).astype(np.float64),
        nx.to_numpy(x_b_1d[perm_b]).astype(np.float64),
        metric=metric, p=p
    )

    G = nx.coo_matrix(
        G_sorted,
        perm_a[indices[:, 0]],
        perm_b[indices[:, 1]],
        shape=(a.shape[0], b.shape[0]),
        type_as=x_a
    )
    if dense:
        G = nx.todense(G)
    elif str(nx) == "jax":
        warnings.warn("JAX does not support sparse matrices, converting to dense")
    if log:
        log = {'cost': nx.from_numpy(cost, type_as=x_a)}
        return G, log
    return G


def emd2_1d(x_a, x_b, a=None, b=None, metric='sqeuclidean', p=1., dense=True,
            log=False):
    r"""Solves the Earth Movers distance problem between 1d measures and returns
    the loss


    .. math::
        \gamma = arg\min_\gamma \sum_i \sum_j \gamma_{ij} d(x_a[i], x_b[j])

        s.t. \gamma 1 = a,
             \gamma^T 1= b,
             \gamma\geq 0
    where :

    - d is the metric
    - x_a and x_b are the samples
    - a and b are the sample weights

    When 'minkowski' is used as a metric, :math:`d(x, y) = |x - y|^p`.

    Uses the algorithm detailed in [1]_

    Parameters
    ----------
    x_a : (ns,) or (ns, 1) ndarray, float64
        Source dirac locations (on the real line)
    x_b : (nt,) or (ns, 1) ndarray, float64
        Target dirac locations (on the real line)
    a : (ns,) ndarray, float64, optional
        Source histogram (default is uniform weight)
    b : (nt,) ndarray, float64, optional
        Target histogram (default is uniform weight)
    metric: str, optional (default='sqeuclidean')
        Metric to be used. Only strings listed in :func:`ot.dist` are accepted.
        Due to implementation details, this function runs faster when
        `'sqeuclidean'`, `'minkowski'`, `'cityblock'`,  or `'euclidean'` metrics
        are used.
    p: float, optional (default=1.0)
         The p-norm to apply for if metric='minkowski'
    dense: boolean, optional (default=True)
        If True, returns math:`\gamma` as a dense ndarray of shape (ns, nt).
        Otherwise returns a sparse representation using scipy's `coo_matrix`
        format. Only used if log is set to True. Due to implementation details,
        this function runs faster when dense is set to False.
    log: boolean, optional (default=False)
        If True, returns a dictionary containing the transportation matrix.
        Otherwise returns only the loss.

    Returns
    -------
    loss: float
        Cost associated to the optimal transportation
    log: dict
        If input log is True, a dictionary containing the Optimal transportation
        matrix for the given parameters


    Examples
    --------

    Simple example with obvious solution. The function emd2_1d accepts lists and
    performs automatic conversion to numpy arrays

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> x_a = [2., 0.]
    >>> x_b = [0., 3.]
    >>> ot.emd2_1d(x_a, x_b, a, b)
    0.5
    >>> ot.emd2_1d(x_a, x_b)
    0.5

    References
    ----------

    .. [1]  Peyré, G., & Cuturi, M. (2017). "Computational Optimal
        Transport", 2018.

    See Also
    --------
    ot.lp.emd2 : EMD for multidimensional distributions
    ot.lp.emd_1d : EMD for 1d distributions (returns the transportation matrix
        instead of the cost)
    """
    # If we do not return G (log==False), then we should not to cast it to dense
    # (useless overhead)
    G, log_emd = emd_1d(x_a=x_a, x_b=x_b, a=a, b=b, metric=metric, p=p,
                        dense=dense and log, log=True)
    cost = log_emd['cost']
    if log:
        log_emd = {'G': G}
        return cost, log_emd
    return cost
