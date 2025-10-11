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
    r"""Computes the quantile function of an empirical distribution

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
    if nx.__name__ == "torch":
        # this is to ensure the best performance for torch searchsorted
        # and avoid a warning related to non-contiguous arrays
        cws = cws.movedim(0, -1).contiguous()
        qs = qs.movedim(0, -1).contiguous()
    else:
        cws = cws.T
        qs = qs.T
    idx = nx.searchsorted(cws, qs).T
    return nx.take_along_axis(xs, nx.clip(idx, 0, n - 1), axis=0)


def wasserstein_1d(
    u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True
):
    r"""
    Computes the 1 dimensional OT loss [15] between two (batched) empirical
    distributions

    .. math:
        OT_{loss} = \int_0^1 |cdf_u^{-1}(q) - cdf_v^{-1}(q)|^p dq

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
        u_weights = nx.full(u_values.shape, 1.0 / n, type_as=u_values)
    elif u_weights.ndim != u_values.ndim:
        u_weights = nx.repeat(u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = nx.full(v_values.shape, 1.0 / m, type_as=v_values)
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
        return nx.sum(delta * diff_quantiles, axis=0)
    return nx.sum(delta * nx.power(diff_quantiles, p), axis=0)


def emd_1d(
    x_a,
    x_b,
    a=None,
    b=None,
    metric="sqeuclidean",
    p=1.0,
    dense=True,
    log=False,
    check_marginals=True,
):
    r"""Solves the Earth Movers distance problem between 1d measures and returns
    the OT matrix


    .. math::
        \gamma = arg\min_\gamma \sum_i \sum_j \gamma_{ij} d(x_a[i], x_b[j])

        s.t. \gamma 1 = a,
             \gamma^T 1= b,
             \gamma\geq 0

    where :

    - d is the metric
    - :math:`x_a` and :math:`x_b` are the samples
    - a and b are the sample weights

    This implementation only supports metrics
    of the form :math:`d(x, y) = |x - y|^p`.

    Uses the algorithm detailed in [1]_

    Parameters
    ----------
    x_a : ndarray of float64, shape (ns,) or (ns, 1)
        Source dirac locations (on the real line)
    x_b : ndarray of float64, shape (nt,) or (ns, 1)
        Target dirac locations (on the real line)
    a : ndarray of float64, shape (ns,), optional
        Source histogram (default is uniform weight)
    b : ndarray of float64, shape (nt,), optional
        Target histogram (default is uniform weight)
    metric: str, optional (default='sqeuclidean')
        Metric to be used. Only works with either of the strings
        `'sqeuclidean'`, `'minkowski'`, `'cityblock'`,  or `'euclidean'`.
    p: float, optional (default=1.0)
         The p-norm to apply for if metric='minkowski'
    dense: boolean, optional (default=True)
        If True, returns :math:`\gamma` as a dense ndarray of shape (ns, nt).
        Otherwise returns a sparse representation using scipy's `coo_matrix`
        format. Due to implementation details, this function runs faster when
        `'sqeuclidean'`, `'minkowski'`, `'cityblock'`,  or `'euclidean'` metrics
        are used.
    log: boolean, optional (default=False)
        If True, returns a dictionary containing the cost.
        Otherwise returns only the optimal transportation matrix.
    check_marginals: bool, optional (default=True)
        If True, checks that the marginals mass are equal. If False, skips the
        check.

    Returns
    -------
    gamma: ndarray, shape (ns, nt)
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
    x_a, x_b = list_to_array(x_a, x_b)
    nx = get_backend(x_a, x_b)
    if a is not None:
        a = list_to_array(a, nx=nx)
    if b is not None:
        b = list_to_array(b, nx=nx)

    assert (
        x_a.ndim == 1 or x_a.ndim == 2 and x_a.shape[1] == 1
    ), "emd_1d should only be used with monodimensional data"
    assert (
        x_b.ndim == 1 or x_b.ndim == 2 and x_b.shape[1] == 1
    ), "emd_1d should only be used with monodimensional data"
    if metric not in ["sqeuclidean", "minkowski", "cityblock", "euclidean"]:
        raise ValueError(
            "Solver for EMD in 1d only supports metrics "
            + "from the following list: "
            + "`['sqeuclidean', 'minkowski', 'cityblock', 'euclidean']`"
        )

    # if empty array given then use uniform distributions
    if a is None or a.ndim == 0 or len(a) == 0:
        a = nx.ones((x_a.shape[0],), type_as=x_a) / x_a.shape[0]
    if b is None or b.ndim == 0 or len(b) == 0:
        b = nx.ones((x_b.shape[0],), type_as=x_b) / x_b.shape[0]

    # ensure that same mass
    if check_marginals:
        np.testing.assert_almost_equal(
            nx.to_numpy(nx.sum(a, axis=0)),
            nx.to_numpy(nx.sum(b, axis=0)),
            err_msg="a and b vector must have the same sum",
            decimal=6,
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
        metric=metric,
        p=p,
    )

    G = nx.coo_matrix(
        G_sorted,
        perm_a[indices[:, 0]],
        perm_b[indices[:, 1]],
        shape=(a.shape[0], b.shape[0]),
        type_as=x_a,
    )
    if dense:
        G = nx.todense(G)
    elif str(nx) == "jax":
        warnings.warn("JAX does not support sparse matrices, converting to dense")
    if log:
        log = {"cost": nx.from_numpy(cost, type_as=x_a)}
        return G, log
    return G


def emd2_1d(
    x_a, x_b, a=None, b=None, metric="sqeuclidean", p=1.0, dense=True, log=False
):
    r"""Solves the Earth Movers distance problem between 1d measures and returns
    the loss


    .. math::
        \gamma = arg\min_\gamma \sum_i \sum_j \gamma_{ij} d(x_a[i], x_b[j])

        s.t. \gamma 1 = a,
             \gamma^T 1= b,
             \gamma\geq 0

    where :

    - d is the metric
    - :math:`x_a` and :math:`x_b` are the samples
    - a and b are the sample weights

    This implementation only supports metrics
    of the form :math:`d(x, y) = |x - y|^p`.

    Uses the algorithm detailed in [1]_

    Parameters
    ----------
    x_a : ndarray of float64, shape (ns,) or (ns, 1)
        Source dirac locations (on the real line)
    x_b : ndarray of float64, shape (nt,) or (ns, 1)
        Target dirac locations (on the real line)
    a : ndarray of float64, shape (ns,), optional
        Source histogram (default is uniform weight)
    b : ndarray of float64, shape (nt,), optional
        Target histogram (default is uniform weight)
    metric: str, optional (default='sqeuclidean')
        Metric to be used. Only works with either of the strings
        `'sqeuclidean'`, `'minkowski'`, `'cityblock'`,  or `'euclidean'`.
    p: float, optional (default=1.0)
         The p-norm to apply for if metric='minkowski'
    dense: boolean, optional (default=True)
        If True, returns :math:`\gamma` as a dense ndarray of shape (ns, nt).
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
    G, log_emd = emd_1d(
        x_a=x_a, x_b=x_b, a=a, b=b, metric=metric, p=p, dense=dense and log, log=True
    )
    cost = log_emd["cost"]
    if log:
        log_emd = {"G": G}
        return cost, log_emd
    return cost


def roll_cols(M, shifts):
    r"""
    Utils functions which allow to shift the order of each row of a 2d matrix

    Parameters
    ----------
    M : ndarray, shape (nr, nc)
        Matrix to shift
    shifts: int or ndarray, shape (nr,)

    Returns
    -------
    Shifted array

    Examples
    --------
    >>> M = np.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> roll_cols(M, 2)
    array([[2, 3, 1],
           [5, 6, 4],
           [8, 9, 7]])
    >>> roll_cols(M, np.array([[1],[2],[1]]))
    array([[3, 1, 2],
           [5, 6, 4],
           [9, 7, 8]])

    References
    ----------
    https://stackoverflow.com/questions/66596699/how-to-shift-columns-or-rows-in-a-tensor-with-different-offsets-in-pytorch
    """
    nx = get_backend(M)

    n_rows, n_cols = M.shape

    arange1 = nx.tile(
        nx.reshape(nx.arange(n_cols, type_as=shifts), (1, n_cols)), (n_rows, 1)
    )
    arange2 = (arange1 - shifts) % n_cols

    return nx.take_along_axis(M, arange2, 1)


def derivative_cost_on_circle(theta, u_values, v_values, u_cdf, v_cdf, p=2):
    r"""Computes the left and right derivative of the cost (Equation (6.3) and (6.4) of [1])

    Parameters
    ----------
    theta: array-like, shape (n_batch, n)
        Cuts on the circle
    u_values: array-like, shape (n_batch, n)
        locations of the first empirical distribution
    v_values: array-like, shape (n_batch, n)
        locations of the second empirical distribution
    u_cdf: array-like, shape (n_batch, n)
        cdf of the first empirical distribution
    v_cdf: array-like, shape (n_batch, n)
        cdf of the second empirical distribution
    p: float, optional = 2
        Power p used for computing the Wasserstein distance

    Returns
    -------
    dCp: array-like, shape (n_batch, 1)
        The batched right derivative
    dCm: array-like, shape (n_batch, 1)
        The batched left derivative

    References
    ---------
    .. [44] Delon, Julie, Julien Salomon, and Andrei Sobolevski. "Fast transport optimization for Monge costs on the circle." SIAM Journal on Applied Mathematics 70.7 (2010): 2239-2258.
    """
    nx = get_backend(theta, u_values, v_values, u_cdf, v_cdf)

    v_values = nx.copy(v_values)

    n = u_values.shape[-1]
    m_batch, m = v_values.shape

    v_cdf_theta = v_cdf - (theta - nx.floor(theta))

    mask_p = v_cdf_theta >= 0
    mask_n = v_cdf_theta < 0

    v_values[mask_n] += nx.floor(theta)[mask_n] + 1
    v_values[mask_p] += nx.floor(theta)[mask_p]

    if nx.any(mask_n) and nx.any(mask_p):
        v_cdf_theta[mask_n] += 1

    v_cdf_theta2 = nx.copy(v_cdf_theta)
    v_cdf_theta2[mask_n] = np.inf
    shift = -nx.argmin(v_cdf_theta2, axis=-1)

    v_cdf_theta = roll_cols(v_cdf_theta, nx.reshape(shift, (-1, 1)))
    v_values = roll_cols(v_values, nx.reshape(shift, (-1, 1)))
    v_values = nx.concatenate(
        [v_values, nx.reshape(v_values[:, 0], (-1, 1)) + 1], axis=1
    )

    if nx.__name__ == "torch":
        # this is to ensure the best performance for torch searchsorted
        # and avoid a warning related to non-contiguous arrays
        u_cdf = u_cdf.contiguous()
        v_cdf_theta = v_cdf_theta.contiguous()

    # quantiles of F_u evaluated in F_v^\theta
    u_index = nx.searchsorted(u_cdf, v_cdf_theta)
    u_icdf_theta = nx.take_along_axis(u_values, nx.clip(u_index, 0, n - 1), -1)

    # Deal with 1
    u_cdfm = nx.concatenate([u_cdf, nx.reshape(u_cdf[:, 0], (-1, 1)) + 1], axis=1)
    u_valuesm = nx.concatenate(
        [u_values, nx.reshape(u_values[:, 0], (-1, 1)) + 1], axis=1
    )

    if nx.__name__ == "torch":
        # this is to ensure the best performance for torch searchsorted
        # and avoid a warning related to non-contiguous arrays
        u_cdfm = u_cdfm.contiguous()
        v_cdf_theta = v_cdf_theta.contiguous()

    u_indexm = nx.searchsorted(u_cdfm, v_cdf_theta, side="right")
    u_icdfm_theta = nx.take_along_axis(u_valuesm, nx.clip(u_indexm, 0, n), -1)

    dCp = nx.sum(
        nx.power(nx.abs(u_icdf_theta - v_values[:, 1:]), p)
        - nx.power(nx.abs(u_icdf_theta - v_values[:, :-1]), p),
        axis=-1,
    )

    dCm = nx.sum(
        nx.power(nx.abs(u_icdfm_theta - v_values[:, 1:]), p)
        - nx.power(nx.abs(u_icdfm_theta - v_values[:, :-1]), p),
        axis=-1,
    )

    return dCp.reshape(-1, 1), dCm.reshape(-1, 1)


def ot_cost_on_circle(theta, u_values, v_values, u_cdf, v_cdf, p):
    r"""Computes the the cost (Equation (6.2) of [1])

    Parameters
    ----------
    theta: array-like, shape (n_batch, n)
        Cuts on the circle
    u_values: array-like, shape (n_batch, n)
        locations of the first empirical distribution
    v_values: array-like, shape (n_batch, n)
        locations of the second empirical distribution
    u_cdf: array-like, shape (n_batch, n)
        cdf of the first empirical distribution
    v_cdf: array-like, shape (n_batch, n)
        cdf of the second empirical distribution
    p: float, optional = 2
        Power p used for computing the Wasserstein distance

    Returns
    -------
    ot_cost: array-like, shape (n_batch,)
        OT cost evaluated at theta

    References
    ---------
    .. [44] Delon, Julie, Julien Salomon, and Andrei Sobolevski. "Fast transport optimization for Monge costs on the circle." SIAM Journal on Applied Mathematics 70.7 (2010): 2239-2258.
    """
    nx = get_backend(theta, u_values, v_values, u_cdf, v_cdf)

    v_values = nx.copy(v_values)

    m_batch, m = v_values.shape
    n_batch, n = u_values.shape

    v_cdf_theta = v_cdf - (theta - nx.floor(theta))

    mask_p = v_cdf_theta >= 0
    mask_n = v_cdf_theta < 0

    v_values[mask_n] += nx.floor(theta)[mask_n] + 1
    v_values[mask_p] += nx.floor(theta)[mask_p]

    if nx.any(mask_n) and nx.any(mask_p):
        v_cdf_theta[mask_n] += 1

    # Put negative values at the end
    v_cdf_theta2 = nx.copy(v_cdf_theta)
    v_cdf_theta2[mask_n] = np.inf
    shift = -nx.argmin(v_cdf_theta2, axis=-1)

    v_cdf_theta = roll_cols(v_cdf_theta, nx.reshape(shift, (-1, 1)))
    v_values = roll_cols(v_values, nx.reshape(shift, (-1, 1)))
    v_values = nx.concatenate(
        [v_values, nx.reshape(v_values[:, 0], (-1, 1)) + 1], axis=1
    )

    # Compute absciss
    cdf_axis = nx.sort(nx.concatenate((u_cdf, v_cdf_theta), -1), -1)
    cdf_axis_pad = nx.zero_pad(cdf_axis, pad_width=[(0, 0), (1, 0)])

    delta = cdf_axis_pad[..., 1:] - cdf_axis_pad[..., :-1]

    if nx.__name__ == "torch":
        # this is to ensure the best performance for torch searchsorted
        # and avoid a warning related to non-contiguous arrays
        u_cdf = u_cdf.contiguous()
        v_cdf_theta = v_cdf_theta.contiguous()
        cdf_axis = cdf_axis.contiguous()

    # Compute icdf
    u_index = nx.searchsorted(u_cdf, cdf_axis)
    u_icdf = nx.take_along_axis(u_values, u_index.clip(0, n - 1), -1)

    v_values = nx.concatenate(
        [v_values, nx.reshape(v_values[:, 0], (-1, 1)) + 1], axis=1
    )
    v_index = nx.searchsorted(v_cdf_theta, cdf_axis)
    v_icdf = nx.take_along_axis(v_values, v_index.clip(0, m), -1)

    if p == 1:
        ot_cost = nx.sum(delta * nx.abs(u_icdf - v_icdf), axis=-1)
    else:
        ot_cost = nx.sum(delta * nx.power(nx.abs(u_icdf - v_icdf), p), axis=-1)

    return ot_cost


def binary_search_circle(
    u_values,
    v_values,
    u_weights=None,
    v_weights=None,
    p=1,
    Lm=10,
    Lp=10,
    tm=-1,
    tp=1,
    eps=1e-6,
    require_sort=True,
    log=False,
):
    r"""Computes the Wasserstein distance on the circle using the Binary search algorithm proposed in [44].
    Samples need to be in :math:`S^1\cong [0,1[`. If they are on :math:`\mathbb{R}`,
    takes the value modulo 1.
    If the values are on :math:`S^1\subset\mathbb{R}^2`, it is required to first find the coordinates
    using e.g. the atan2 function.

    .. math::
        W_p^p(u,v) = \inf_{\theta\in\mathbb{R}}\int_0^1 |F_u^{-1}(q)  - (F_v-\theta)^{-1}(q)|^p\ \mathrm{d}q

    where:

    - :math:`F_u` and :math:`F_v` are respectively the cdfs of :math:`u` and :math:`v`

    For values :math:`x=(x_1,x_2)\in S^1`, it is required to first get their coordinates with

    .. math::
        u = \frac{\pi + \mathrm{atan2}(-x_2,-x_1)}{2\pi}

    using e.g. ot.utils.get_coordinate_circle(x)

    The function runs on backend but tensorflow and jax are not supported.

    Parameters
    ----------
    u_values : ndarray, shape (n, ...)
        samples in the source domain (coordinates on [0,1[)
    v_values : ndarray, shape (n, ...)
        samples in the target domain (coordinates on [0,1[)
    u_weights : ndarray, shape (n, ...), optional
        samples weights in the source domain
    v_weights : ndarray, shape (n, ...), optional
        samples weights in the target domain
    p : float, optional (default=1)
        Power p used for computing the Wasserstein distance
    Lm : int, optional
        Lower bound dC
    Lp : int, optional
        Upper bound dC
    tm: float, optional
        Lower bound theta
    tp: float, optional
        Upper bound theta
    eps: float, optional
        Stopping condition
    require_sort: bool, optional
        If True, sort the values.
    log: bool, optional
        If True, returns also the optimal theta

    Returns
    -------
    loss: float/array-like, shape (...)
        Batched cost associated to the optimal transportation
    log: dict, optional
        log dictionary returned only if log==True in parameters

    Examples
    --------
    >>> u = np.array([[0.2,0.5,0.8]])%1
    >>> v = np.array([[0.4,0.5,0.7]])%1
    >>> binary_search_circle(u.T, v.T, p=1)
    array([0.1])

    References
    ----------
    .. [44] Delon, Julie, Julien Salomon, and Andrei Sobolevski. "Fast transport optimization for Monge costs on the circle." SIAM Journal on Applied Mathematics 70.7 (2010): 2239-2258.
    .. Matlab Code: https://users.mccme.ru/ansobol/otarie/software.html
    """
    assert p >= 1, "The OT loss is only valid for p>=1, {p} was given".format(p=p)

    if u_weights is not None and v_weights is not None:
        nx = get_backend(u_values, v_values, u_weights, v_weights)
    else:
        nx = get_backend(u_values, v_values)

    n = u_values.shape[0]
    m = v_values.shape[0]

    if len(u_values.shape) == 1:
        u_values = nx.reshape(u_values, (n, 1))
    if len(v_values.shape) == 1:
        v_values = nx.reshape(v_values, (m, 1))

    if u_values.shape[1] != v_values.shape[1]:
        raise ValueError(
            "u and v must have the same number of batches {} and {} respectively given".format(
                u_values.shape[1], v_values.shape[1]
            )
        )

    u_values = u_values % 1
    v_values = v_values % 1

    if u_weights is None:
        u_weights = nx.full(u_values.shape, 1.0 / n, type_as=u_values)
    elif u_weights.ndim != u_values.ndim:
        u_weights = nx.repeat(u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = nx.full(v_values.shape, 1.0 / m, type_as=v_values)
    elif v_weights.ndim != v_values.ndim:
        v_weights = nx.repeat(v_weights[..., None], v_values.shape[-1], -1)

    if require_sort:
        u_sorter = nx.argsort(u_values, 0)
        u_values = nx.take_along_axis(u_values, u_sorter, 0)

        v_sorter = nx.argsort(v_values, 0)
        v_values = nx.take_along_axis(v_values, v_sorter, 0)

        u_weights = nx.take_along_axis(u_weights, u_sorter, 0)
        v_weights = nx.take_along_axis(v_weights, v_sorter, 0)

    u_cdf = nx.cumsum(u_weights, 0).T
    v_cdf = nx.cumsum(v_weights, 0).T

    u_values = u_values.T
    v_values = v_values.T

    L = max(Lm, Lp)

    tm = tm * nx.reshape(nx.ones((u_values.shape[0],), type_as=u_values), (-1, 1))
    tm = nx.tile(tm, (1, m))
    tp = tp * nx.reshape(nx.ones((u_values.shape[0],), type_as=u_values), (-1, 1))
    tp = nx.tile(tp, (1, m))
    tc = (tm + tp) / 2

    done = nx.zeros((u_values.shape[0], m))

    cpt = 0
    while nx.any(1 - done):
        cpt += 1

        dCp, dCm = derivative_cost_on_circle(tc, u_values, v_values, u_cdf, v_cdf, p)
        done = ((dCp * dCm) <= 0) * 1

        mask = ((tp - tm) < eps / L) * (1 - done)

        if nx.any(mask):
            # can probably be improved by computing only relevant values
            dCptp, dCmtp = derivative_cost_on_circle(
                tp, u_values, v_values, u_cdf, v_cdf, p
            )
            dCptm, dCmtm = derivative_cost_on_circle(
                tm, u_values, v_values, u_cdf, v_cdf, p
            )
            Ctm = ot_cost_on_circle(tm, u_values, v_values, u_cdf, v_cdf, p).reshape(
                -1, 1
            )
            Ctp = ot_cost_on_circle(tp, u_values, v_values, u_cdf, v_cdf, p).reshape(
                -1, 1
            )

            # Avoid warning raised when dCptm - dCmtp == 0, for which
            # tc is not updated as mask_end is False,
            # see Issue #738
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mask_end = mask * (nx.abs(dCptm - dCmtp) > 0.001)
                tc[mask_end > 0] = (
                    (Ctp - Ctm + tm * dCptm - tp * dCmtp) / (dCptm - dCmtp)
                )[mask_end > 0]
            done[nx.prod(mask, axis=-1) > 0] = 1
        elif nx.any(1 - done):
            tm[((1 - mask) * (dCp < 0)) > 0] = tc[((1 - mask) * (dCp < 0)) > 0]
            tp[((1 - mask) * (dCp >= 0)) > 0] = tc[((1 - mask) * (dCp >= 0)) > 0]
            tc[((1 - mask) * (1 - done)) > 0] = (
                tm[((1 - mask) * (1 - done)) > 0] + tp[((1 - mask) * (1 - done)) > 0]
            ) / 2

    w = ot_cost_on_circle(nx.detach(tc), u_values, v_values, u_cdf, v_cdf, p)

    if log:
        return w, {"optimal_theta": tc[:, 0]}
    return w


def wasserstein1_circle(
    u_values, v_values, u_weights=None, v_weights=None, require_sort=True
):
    r"""Computes the 1-Wasserstein distance on the circle using the level median [45].
    Samples need to be in :math:`S^1\cong [0,1[`. If they are on :math:`\mathbb{R}`,
    takes the value modulo 1.
    If the values are on :math:`S^1\subset\mathbb{R}^2`, first find the coordinates
    using e.g. the atan2 function.
    The function runs on backend but tensorflow and jax are not supported.

    .. math::
        W_1(u,v) = \int_0^1 |F_u(t)-F_v(t)-LevMed(F_u-F_v)|\ \mathrm{d}t

    Parameters
    ----------
    u_values : ndarray, shape (n, ...)
        samples in the source domain (coordinates on [0,1[)
    v_values : ndarray, shape (n, ...)
        samples in the target domain (coordinates on [0,1[)
    u_weights : ndarray, shape (n, ...), optional
        samples weights in the source domain
    v_weights : ndarray, shape (n, ...), optional
        samples weights in the target domain
    require_sort: bool, optional
        If True, sort the values.

    Returns
    -------
    loss: float/array-like, shape (...)
        Batched cost associated to the optimal transportation

    Examples
    --------
    >>> u = np.array([[0.2,0.5,0.8]])%1
    >>> v = np.array([[0.4,0.5,0.7]])%1
    >>> wasserstein1_circle(u.T, v.T)
    array([0.1])

    References
    ----------
    .. [45] Hundrieser, Shayan, Marcel Klatt, and Axel Munk. "The statistics of circular optimal transport." Directional Statistics for Innovative Applications: A Bicentennial Tribute to Florence Nightingale. Singapore: Springer Nature Singapore, 2022. 57-82.
    .. Code R: https://gitlab.gwdg.de/shundri/circularOT/-/tree/master/
    """

    if u_weights is not None and v_weights is not None:
        nx = get_backend(u_values, v_values, u_weights, v_weights)
    else:
        nx = get_backend(u_values, v_values)

    n = u_values.shape[0]
    m = v_values.shape[0]

    if len(u_values.shape) == 1:
        u_values = nx.reshape(u_values, (n, 1))
    if len(v_values.shape) == 1:
        v_values = nx.reshape(v_values, (m, 1))

    if u_values.shape[1] != v_values.shape[1]:
        raise ValueError(
            "u and v must have the same number of batchs {} and {} respectively given".format(
                u_values.shape[1], v_values.shape[1]
            )
        )

    u_values = u_values % 1
    v_values = v_values % 1

    if u_weights is None:
        u_weights = nx.full(u_values.shape, 1.0 / n, type_as=u_values)
    elif u_weights.ndim != u_values.ndim:
        u_weights = nx.repeat(u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = nx.full(v_values.shape, 1.0 / m, type_as=v_values)
    elif v_weights.ndim != v_values.ndim:
        v_weights = nx.repeat(v_weights[..., None], v_values.shape[-1], -1)

    if require_sort:
        u_sorter = nx.argsort(u_values, 0)
        u_values = nx.take_along_axis(u_values, u_sorter, 0)

        v_sorter = nx.argsort(v_values, 0)
        v_values = nx.take_along_axis(v_values, v_sorter, 0)

        u_weights = nx.take_along_axis(u_weights, u_sorter, 0)
        v_weights = nx.take_along_axis(v_weights, v_sorter, 0)

    # Code inspired from https://gitlab.gwdg.de/shundri/circularOT/-/tree/master/
    values_sorted, values_sorter = nx.sort2(nx.concatenate((u_values, v_values), 0), 0)

    cdf_diff = nx.cumsum(
        nx.take_along_axis(
            nx.concatenate((u_weights, -v_weights), 0), values_sorter, 0
        ),
        0,
    )
    cdf_diff_sorted, cdf_diff_sorter = nx.sort2(cdf_diff, axis=0)

    values_sorted = nx.zero_pad(values_sorted, pad_width=[(0, 1), (0, 0)], value=1)
    delta = values_sorted[1:, ...] - values_sorted[:-1, ...]
    weight_sorted = nx.take_along_axis(delta, cdf_diff_sorter, 0)

    sum_weights = nx.cumsum(weight_sorted, axis=0) - 0.5
    sum_weights[sum_weights < 0] = np.inf
    inds = nx.argmin(sum_weights, axis=0)

    levMed = nx.take_along_axis(cdf_diff_sorted, nx.reshape(inds, (1, -1)), 0)

    return nx.sum(delta * nx.abs(cdf_diff - levMed), axis=0)


def wasserstein_circle(
    u_values,
    v_values,
    u_weights=None,
    v_weights=None,
    p=1,
    Lm=10,
    Lp=10,
    tm=-1,
    tp=1,
    eps=1e-6,
    require_sort=True,
):
    r"""Computes the Wasserstein distance on the circle using either :ref:`[45] <references-wasserstein-circle>` for p=1 or
    the binary search algorithm proposed in :ref:`[44] <references-wasserstein-circle>` otherwise.
    Samples need to be in :math:`S^1\cong [0,1[`. If they are on :math:`\mathbb{R}`,
    takes the value modulo 1.
    If the values are on :math:`S^1\subset\mathbb{R}^2`, it requires to first find the coordinates
    using e.g. the atan2 function.

    General loss returned:

    .. math::
        OT_{loss} = \inf_{\theta\in\mathbb{R}}\int_0^1 |cdf_u^{-1}(q)  - (cdf_v-\theta)^{-1}(q)|^p\ \mathrm{d}q

    For p=1, [45]

    .. math::
        W_1(u,v) = \int_0^1 |F_u(t)-F_v(t)-LevMed(F_u-F_v)|\ \mathrm{d}t

    For values :math:`x=(x_1,x_2)\in S^1`, it is required to first get their coordinates with

    .. math::
        u = \frac{\pi + \mathrm{atan2}(-x_2,-x_1)}{2\pi}

    using e.g. ot.utils.get_coordinate_circle(x)

    The function runs on backend but tensorflow and jax are not supported.

    Parameters
    ----------
    u_values : ndarray, shape (n, ...)
        samples in the source domain (coordinates on [0,1[)
    v_values : ndarray, shape (n, ...)
        samples in the target domain (coordinates on [0,1[)
    u_weights : ndarray, shape (n, ...), optional
        samples weights in the source domain
    v_weights : ndarray, shape (n, ...), optional
        samples weights in the target domain
    p : float, optional (default=1)
        Power p used for computing the Wasserstein distance
    Lm : int, optional
        Lower bound dC. For p>1.
    Lp : int, optional
        Upper bound dC. For p>1.
    tm: float, optional
        Lower bound theta. For p>1.
    tp: float, optional
        Upper bound theta. For p>1.
    eps: float, optional
        Stopping condition. For p>1.
    require_sort: bool, optional
        If True, sort the values.

    Returns
    -------
    loss: float/array-like, shape (...)
        Batched cost associated to the optimal transportation

    Examples
    --------
    >>> u = np.array([[0.2,0.5,0.8]])%1
    >>> v = np.array([[0.4,0.5,0.7]])%1
    >>> wasserstein_circle(u.T, v.T)
    array([0.1])


    .. _references-wasserstein-circle:
    References
    ----------
    .. [44] Hundrieser, Shayan, Marcel Klatt, and Axel Munk. "The statistics of circular optimal transport." Directional Statistics for Innovative Applications: A Bicentennial Tribute to Florence Nightingale. Singapore: Springer Nature Singapore, 2022. 57-82.
    .. [45] Delon, Julie, Julien Salomon, and Andrei Sobolevski. "Fast transport optimization for Monge costs on the circle." SIAM Journal on Applied Mathematics 70.7 (2010): 2239-2258.
    """
    assert p >= 1, "The OT loss is only valid for p>=1, {p} was given".format(p=p)

    return binary_search_circle(
        u_values,
        v_values,
        u_weights,
        v_weights,
        p=p,
        Lm=Lm,
        Lp=Lp,
        tm=tm,
        tp=tp,
        eps=eps,
        require_sort=require_sort,
    )


def semidiscrete_wasserstein2_unif_circle(u_values, u_weights=None):
    r"""Computes the closed-form for the 2-Wasserstein distance between samples and a uniform distribution on :math:`S^1`
    Samples need to be in :math:`S^1\cong [0,1[`. If they are on :math:`\mathbb{R}`,
    takes the value modulo 1.
    If the values are on :math:`S^1\subset\mathbb{R}^2`, it is required to first find the coordinates
    using e.g. the atan2 function.

    .. math::
        W_2^2(\mu_n, \nu) = \sum_{i=1}^n \alpha_i x_i^2 - \left(\sum_{i=1}^n \alpha_i x_i\right)^2 + \sum_{i=1}^n \alpha_i x_i \left(1-\alpha_i-2\sum_{k=1}^{i-1}\alpha_k\right) + \frac{1}{12}

    where:

    - :math:`\nu=\mathrm{Unif}(S^1)` and :math:`\mu_n  = \sum_{i=1}^n \alpha_i \delta_{x_i}`

    For values :math:`x=(x_1,x_2)\in S^1`, it is required to first get their coordinates with

    .. math::
        u = \frac{\pi + \mathrm{atan2}(-x_2,-x_1)}{2\pi},

    using e.g. ot.utils.get_coordinate_circle(x).

    Parameters
    ----------
    u_values : ndarray, shape (n, ...)
        Samples
    u_weights : ndarray, shape (n, ...), optional
        samples weights in the source domain

    Returns
    -------
    loss: float/array-like, shape (...)
        Batched cost associated to the optimal transportation

    Examples
    --------
    >>> x0 = np.array([[0], [0.2], [0.4]])
    >>> semidiscrete_wasserstein2_unif_circle(x0)
    array([0.02111111])

    References
    ----------
    .. [46] Bonet, C., Berg, P., Courty, N., Septier, F., Drumetz, L., & Pham, M. T. (2023). Spherical sliced-wasserstein. International Conference on Learning Representations.
    """

    if u_weights is not None:
        nx = get_backend(u_values, u_weights)
    else:
        nx = get_backend(u_values)

    n = u_values.shape[0]

    u_values = u_values % 1

    if len(u_values.shape) == 1:
        u_values = nx.reshape(u_values, (n, 1))

    if u_weights is None:
        u_weights = nx.full(u_values.shape, 1.0 / n, type_as=u_values)
    elif u_weights.ndim != u_values.ndim:
        u_weights = nx.repeat(u_weights[..., None], u_values.shape[-1], -1)

    u_values = nx.sort(u_values, 0)
    u_cdf = nx.cumsum(u_weights, 0)
    u_cdf = nx.zero_pad(u_cdf, [(1, 0), (0, 0)])

    cpt1 = nx.sum(u_weights * u_values**2, axis=0)
    u_mean = nx.sum(u_weights * u_values, axis=0)

    ns = 1 - u_weights - 2 * u_cdf[:-1]
    cpt2 = nx.sum(u_values * u_weights * ns, axis=0)

    return cpt1 - u_mean**2 + cpt2 + 1 / 12


def linear_circular_embedding(x, u_values, u_weights=None, require_sort=True):
    r"""Returns the embedding :math:`\hat{\mu}(x)` of Linear Circular OT with reference
    :math:`\eta=\mathrm{Unif}(S^1)` evaluated in :math:`x`.

    For any :math:`x\in [0,1[`, the embedding is given by (see :ref:`[78] <references-lcot>`)

    .. math``
        \hat{\mu}(x) = F_{\mu}^{-1}\big(x - \int z\mathrm{d}\mu(z) + \frac12) - x.

    Parameters
    ----------
    x : ndary, shape (m,)
        Points in [0,1[ where to evaluate the embedding
    u_values : ndarray, shape (n, ...)
        samples in the source domain (coordinates on [0,1[)
    u_weights : ndarray, shape (n, ...), optional
        samples weights in the source domain

    Returns
    -------
    embedding: ndarray of shape (m, ...)
        Embedding evaluated at :math:`x`

    .. _references-lcot:
    References
    ----------
    .. [78] Martin, R. D., Medri, I., Bai, Y., Liu, X., Yan, K., Rohde, G. K., & Kolouri, S. (2024). LCOT: Linear Circular Optimal Transport. International Conference on Learning Representations.
    """
    if u_weights is not None:
        nx = get_backend(u_values, u_weights)
    else:
        nx = get_backend(u_values)

    n = u_values.shape[0]
    u_values = u_values % 1

    if len(u_values.shape) == 1:
        u_values = nx.reshape(u_values, (n, 1))

    if u_weights is None:
        u_weights = nx.full(u_values.shape, 1.0 / n, type_as=u_values)
    elif u_weights.ndim != u_values.ndim:
        u_weights = nx.repeat(u_weights[..., None], u_values.shape[-1], -1)

    if require_sort:
        u_sorter = nx.argsort(u_values, 0)
        u_values = nx.take_along_axis(u_values, u_sorter, 0)
        u_weights = nx.take_along_axis(u_weights, u_sorter, 0)

    u_cdf = nx.cumsum(u_weights, 0)
    u_cdf = nx.zero_pad(u_cdf, [(1, 0), (0, 0)])

    q_s = (
        x[:, None] - nx.sum(u_values * u_weights, axis=0)[None] + 0.5
    )  # shape (m, ...)

    u_quantiles = quantile_function(q_s % 1, u_cdf, u_values)

    return (u_quantiles - x[:, None]) % 1


def linear_circular_ot(u_values, v_values=None, u_weights=None, v_weights=None):
    r"""Computes the Linear Circular Optimal Transport distance from :ref:`[78] <references-lcot>` using :math:`\eta=\mathrm{Unif}(S^1)`
    as reference measure.
    Samples need to be in :math:`S^1\cong [0,1[`. If they are on :math:`\mathbb{R}`,
    takes the value modulo 1.
    If the values are on :math:`S^1\subset\mathbb{R}^2`, it is required to first find the coordinates
    using e.g. the atan2 function.

    General loss returned:

    .. math::
        \mathrm{LCOT}_2^2(\mu, \nu) = \int_0^1 d_{S^1}\big(\hat{\mu}(t), \hat{\nu}(t)\big)^2\ \mathrm{d}t

    where :math:`\hat{\mu}(x)=F_{\mu}^{-1}(x-\int z\mathrm{d}\mu(z)+\frac12) - x` for all :math:`x\in [0,1[`,
    and :math:`d_{S^1}(x,y)=\min(|x-y|, 1-|x-y|)` for :math:`x,y\in [0,1[`.

    Parameters
    ----------
    u_values : ndarray, shape (n, ...)
        samples in the source domain (coordinates on [0,1[)
    v_values : ndarray, shape (n, ...), optional
        samples in the target domain (coordinates on [0,1[), if None, compute distance against uniform distribution
    u_weights : ndarray, shape (n, ...), optional
        samples weights in the source domain
    v_weights : ndarray, shape (n, ...), optional
        samples weights in the target domain

    Returns
    -------
    loss: float/array-like, shape (...)
        Batched cost associated to the linear optimal transportation

    Examples
    --------
    >>> u = np.array([[0.2,0.5,0.8]])%1
    >>> v = np.array([[0.4,0.5,0.7]])%1
    >>> linear_circular_ot(u.T, v.T)
    array([0.0127])


    .. _references-lcot:
    References
    ----------
    .. [78] Martin, R. D., Medri, I., Bai, Y., Liu, X., Yan, K., Rohde, G. K., & Kolouri, S. (2024). LCOT: Linear Circular Optimal Transport. International Conference on Learning Representations.
    """
    if u_weights is not None:
        nx = get_backend(u_values, u_weights)
    else:
        nx = get_backend(u_values)

    n = u_values.shape[0]
    u_values = u_values % 1

    if len(u_values.shape) == 1:
        u_values = nx.reshape(u_values, (n, 1))

    if u_weights is None:
        u_weights = nx.full(u_values.shape, 1.0 / n, type_as=u_values)
    elif u_weights.ndim != u_values.ndim:
        u_weights = nx.repeat(u_weights[..., None], u_values.shape[-1], -1)

    unif_s1 = nx.linspace(0, 1, 101, type_as=u_values)[:-1]

    emb_u = linear_circular_embedding(unif_s1, u_values, u_weights)

    if v_values is None:
        dist_u = nx.minimum(nx.abs(emb_u), 1 - nx.abs(emb_u))
        return nx.mean(dist_u**2, axis=0)
    else:
        m = v_values.shape[0]
        if len(v_values.shape) == 1:
            v_values = nx.reshape(v_values, (m, 1))

        if u_values.shape[1] != v_values.shape[1]:
            raise ValueError(
                "u and v must have the same number of batchs {} and {} respectively given".format(
                    u_values.shape[1], v_values.shape[1]
                )
            )

    emb_v = linear_circular_embedding(unif_s1, v_values, v_weights)

    dist_uv = nx.minimum(nx.abs(emb_u - emb_v), 1 - nx.abs(emb_u - emb_v))
    return nx.mean(dist_uv**2, axis=0)
