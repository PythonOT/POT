# -*- coding: utf-8 -*-
"""
Various useful functions
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

from functools import reduce
import time

import numpy as np
from scipy.spatial.distance import cdist
import sys
import warnings
from inspect import signature
from .backend import get_backend, Backend, NumpyBackend, JaxBackend

__time_tic_toc = time.perf_counter()


def tic():
    r"""Python implementation of Matlab tic() function"""
    global __time_tic_toc
    __time_tic_toc = time.perf_counter()


def toc(message="Elapsed time : {} s"):
    r"""Python implementation of Matlab toc() function"""
    t = time.perf_counter()
    print(message.format(t - __time_tic_toc))
    return t - __time_tic_toc


def toq():
    r"""Python implementation of Julia toc() function"""
    t = time.perf_counter()
    return t - __time_tic_toc


def kernel(x1, x2, method="gaussian", sigma=1, **kwargs):
    r"""Compute kernel matrix"""

    nx = get_backend(x1, x2)

    if method.lower() in ["gaussian", "gauss", "rbf"]:
        K = nx.exp(-dist(x1, x2) / (2 * sigma**2))
    return K


def laplacian(x):
    r"""Compute Laplacian matrix"""
    nx = get_backend(x)
    L = nx.diag(nx.sum(x, axis=0)) - x
    return L


def list_to_array(*lst, nx=None):
    r"""Convert a list if in numpy format"""
    lst_not_empty = [a for a in lst if len(a) > 0 and not isinstance(a, list)]
    if nx is None:  # find backend
        if len(lst_not_empty) == 0:
            type_as = np.zeros(0)
            nx = get_backend(type_as)
        else:
            nx = get_backend(*lst_not_empty)
            type_as = lst_not_empty[0]
    else:
        if len(lst_not_empty) == 0:
            type_as = None
        else:
            type_as = lst_not_empty[0]
    if len(lst) > 1:
        return [
            nx.from_numpy(np.array(a), type_as=type_as) if isinstance(a, list) else a
            for a in lst
        ]
    else:
        if isinstance(lst[0], list):
            return nx.from_numpy(np.array(lst[0]), type_as=type_as)
        else:
            return lst[0]


def proj_simplex(v, z=1):
    r"""Compute the closest point (orthogonal projection) on the
    generalized `(n-1)`-simplex of a vector :math:`\mathbf{v}` wrt. to the Euclidean
    distance, thus solving:

    .. math::
        \mathcal{P}(w) \in \mathop{\arg \min}_\gamma \| \gamma - \mathbf{v} \|_2

        s.t. \ \gamma^T \mathbf{1} = z

             \gamma \geq 0

    If :math:`\mathbf{v}` is a 2d array, compute all the projections wrt. axis 0

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends.

    Parameters
    ----------
    v : {array-like}, shape (n, d)
    z : int, optional
        'size' of the simplex (each vectors sum to z, 1 by default)

    Returns
    -------
    h : ndarray, shape (`n`, `d`)
        Array of projections on the simplex
    """
    nx = get_backend(v)
    n = v.shape[0]
    if v.ndim == 1:
        d1 = 1
        v = v[:, None]
    else:
        d1 = 0
    d = v.shape[1]

    # sort u in ascending order
    u = nx.sort(v, axis=0)
    # take the descending order
    u = nx.flip(u, 0)
    cssv = nx.cumsum(u, axis=0) - z
    ind = nx.arange(n, type_as=v)[:, None] + 1
    cond = u - cssv / ind > 0
    rho = nx.sum(cond, 0)
    theta = cssv[rho - 1, nx.arange(d)] / rho
    w = nx.maximum(v - theta[None, :], nx.zeros(v.shape, type_as=v))
    if d1:
        return w[:, 0]
    else:
        return w


def projection_sparse_simplex(V, max_nz, z=1, axis=None, nx=None):
    r"""Projection of :math:`\mathbf{V}` onto the simplex with cardinality constraint (maximum number of non-zero elements) and then scaled by `z`.

    .. math::
        P\left(\mathbf{V}, \text{max_nz}, z\right) = \mathop{\arg \min}_{\substack{\mathbf{y} >= 0 \\ \sum_i \mathbf{y}_i = z} \\ ||p||_0 \le \text{max_nz}} \quad \|\mathbf{y} - \mathbf{V}\|^2

    Parameters
    ----------
    V: 1-dim or 2-dim ndarray
    max_nz: int
        Maximum number of non-zero elements in the projection.
        If `max_nz` is larger than the number of elements in `V`, then
        the projection is equivalent to `proj_simplex(V, z)`.
    z: float or array
        If array, len(z) must be compatible with :math:`\mathbf{V}`
    axis: None or int
        - axis=None: project :math:`\mathbf{V}` by :math:`P(\mathbf{V}.\mathrm{ravel}(), \text{max_nz}, z)`
        - axis=1: project each :math:`\mathbf{V}_i` by :math:`P(\mathbf{V}_i, \text{max_nz}, z_i)`
        - axis=0: project each :math:`\mathbf{V}_{:, j}` by :math:`P(\mathbf{V}_{:, j}, \text{max_nz}, z_j)`

    Returns
    -------
    projection: ndarray, shape :math:`\mathbf{V}`.shape

    References
    ----------
    .. [1] Sparse projections onto the simplex
        Anastasios Kyrillidis, Stephen Becker, Volkan Cevher and, Christoph Koch
        ICML 2013
        https://arxiv.org/abs/1206.1529
    """
    if nx is None:
        nx = get_backend(V)
    if V.ndim == 1:
        return projection_sparse_simplex(
            # V[nx.newaxis, :], max_nz, z, axis=1).ravel()
            V[None, :],
            max_nz,
            z,
            axis=1,
        ).ravel()

    if V.ndim > 2:
        raise ValueError("V.ndim must be <= 2")

    if axis == 1:
        # For each row of V, find top max_nz values; arrange the
        # corresponding column indices such that their values are
        # in a descending order.
        max_nz_indices = nx.argsort(V, axis=1)[:, -max_nz:]
        max_nz_indices = nx.flip(max_nz_indices, axis=1)

        row_indices = nx.arange(V.shape[0])
        row_indices = row_indices.reshape(-1, 1)
        print(row_indices.shape)
        # Extract the top max_nz values for each row
        # and then project to simplex.
        U = V[row_indices, max_nz_indices]
        z = nx.ones(len(U)) * z
        cssv = nx.cumsum(U, axis=1) - z[:, None]
        ind = nx.arange(max_nz) + 1
        cond = U - cssv / ind > 0
        # rho = nx.count_nonzero(cond, axis=1)
        rho = nx.sum(cond, axis=1)
        theta = cssv[nx.arange(len(U)), rho - 1] / rho
        nz_projection = nx.maximum(U - theta[:, None], 0)

        # Put the projection of max_nz_values to their original column indices
        # while keeping other values zero.
        sparse_projection = nx.zeros(V.shape, type_as=nz_projection)

        if isinstance(nx, JaxBackend):
            # in Jax, we need to use the `at` property of `jax.numpy.ndarray`
            # to do in-place array modifications.
            sparse_projection = sparse_projection.at[row_indices, max_nz_indices].set(
                nz_projection
            )
        else:
            sparse_projection[row_indices, max_nz_indices] = nz_projection
        return sparse_projection

    elif axis == 0:
        return projection_sparse_simplex(V.T, max_nz, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_sparse_simplex(V, max_nz, z, axis=1).ravel()


def unif(n, type_as=None):
    r"""
    Return a uniform histogram of length `n` (simplex).

    Parameters
    ----------
    n : int
        number of bins in the histogram
    type_as : array-like
        array of the same type of the expected output (numpy/pytorch/jax)

    Returns
    -------
    h : array-like, shape (n,)
        histogram of length `n` such that :math:`\forall i, \mathbf{h}_i = \frac{1}{n}`
    """
    if type_as is None:
        return np.ones((n,)) / n
    else:
        nx = get_backend(type_as)
        return nx.ones((n,), type_as=type_as) / n


def clean_zeros(a, b, M):
    r"""Remove all components with zeros weights in :math:`\mathbf{a}` and :math:`\mathbf{b}`"""
    M2 = M[a > 0, :][:, b > 0].copy()  # copy force c style matrix (froemd)
    a2 = a[a > 0]
    b2 = b[b > 0]
    return a2, b2, M2


def euclidean_distances(X, Y, squared=False, nx=None):
    r"""
    Considering the rows of :math:`\mathbf{X}` (and :math:`\mathbf{Y} = \mathbf{X}`) as vectors, compute the
    distance matrix between each pair of vectors.

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends.

    Parameters
    ----------
    X : array-like, shape (n_samples_1, n_features)
    Y : array-like, shape (n_samples_2, n_features)
    squared : boolean, optional
        Return squared Euclidean distances.

    Returns
    -------
    distances : array-like, shape (`n_samples_1`, `n_samples_2`)
    """
    if nx is None:
        nx = get_backend(X, Y)

    a2 = nx.einsum("ij,ij->i", X, X)
    b2 = nx.einsum("ij,ij->i", Y, Y)

    c = -2 * nx.dot(X, nx.transpose(Y))
    c += a2[:, None]
    c += b2[None, :]

    c = nx.maximum(c, 0)

    if not squared:
        c = nx.sqrt(c)

    if X is Y:
        c = c * (1 - nx.eye(X.shape[0], type_as=c))

    return c


def dist(
    x1,
    x2=None,
    metric="sqeuclidean",
    p=2,
    w=None,
    backend="auto",
    nx=None,
    use_tensor=False,
):
    r"""Compute distance between samples in :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends for the following metrics:
        'sqeuclidean', 'euclidean', 'cityblock', 'minkowski', 'cosine', 'correlation'.

    Parameters
    ----------

    x1 : array-like, shape (n1,d)
        matrix with `n1` samples of size `d`
    x2 : array-like, shape (n2,d), optional
        matrix with `n2` samples of size `d` (if None then :math:`\mathbf{x_2} = \mathbf{x_1}`)
    metric : str | callable, optional
        'sqeuclidean' or 'euclidean' on all backends. On numpy the function also
        accepts  from the scipy.spatial.distance.cdist function : 'braycurtis',
        'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
        'euclidean', 'hamming', 'jaccard', 'kulczynski1', 'mahalanobis',
        'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    p : float, optional
        p-norm for the Minkowski and the Weighted Minkowski metrics. Default value is 2.
    w : array-like, rank 1
        Weights for the weighted metrics.
    backend : str, optional
        Backend to use for the computation. If 'auto', the backend is
        automatically selected based on the input data. if 'scipy',
        the ``scipy.spatial.distance.cdist`` function is used (and gradients are
        detached).
    use_tensor : bool, optional
        If true use tensorized computation for the distance matrix which can
        cause memory issues for large datasets. Default is False and the
        parameter is used only for the 'cityblock' and 'minkowski' metrics.
    nx : Backend, optional
        Backend to perform computations on. If omitted, the backend defaults to that of `x1`.

    Returns
    -------

    M : array-like, shape (`n1`, `n2`)
        distance matrix computed with given metric

    """
    if nx is None:
        nx = get_backend(x1, x2)
    if x2 is None:
        x2 = x1
    if backend == "scipy":  # force scipy backend with cdist function
        x1 = nx.to_numpy(x1)
        x2 = nx.to_numpy(x2)
        if isinstance(metric, str) and metric.endswith("minkowski"):
            return nx.from_numpy(cdist(x1, x2, metric=metric, p=p, w=w))
        if w is not None:
            return nx.from_numpy(cdist(x1, x2, metric=metric, w=w))
        return nx.from_numpy(cdist(x1, x2, metric=metric))
    elif metric == "sqeuclidean":
        return euclidean_distances(x1, x2, squared=True, nx=nx)
    elif metric == "euclidean":
        return euclidean_distances(x1, x2, squared=False, nx=nx)
    elif metric == "cityblock":
        if use_tensor:
            return nx.sum(nx.abs(x1[:, None, :] - x2[None, :, :]), axis=2)
        else:
            M = 0.0
            for i in range(x1.shape[1]):
                M += nx.abs(x1[:, i][:, None] - x2[:, i][None, :])
            return M
    elif metric == "minkowski":
        if w is None:
            if use_tensor:
                return nx.power(
                    nx.sum(
                        nx.power(nx.abs(x1[:, None, :] - x2[None, :, :]), p), axis=2
                    ),
                    1 / p,
                )
            else:
                M = 0.0
                for i in range(x1.shape[1]):
                    M += nx.abs(x1[:, i][:, None] - x2[:, i][None, :]) ** p
                return M ** (1 / p)
        else:
            if use_tensor:
                return nx.power(
                    nx.sum(
                        w[None, None, :]
                        * nx.power(nx.abs(x1[:, None, :] - x2[None, :, :]), p),
                        axis=2,
                    ),
                    1 / p,
                )
            else:
                M = 0.0
                for i in range(x1.shape[1]):
                    M += w[i] * nx.abs(x1[:, i][:, None] - x2[:, i][None, :]) ** p
                return M ** (1 / p)
    elif metric == "cosine":
        nx1 = nx.sqrt(nx.einsum("ij,ij->i", x1, x1))
        nx2 = nx.sqrt(nx.einsum("ij,ij->i", x2, x2))
        return 1.0 - (nx.dot(x1, nx.transpose(x2)) / nx1[:, None] / nx2[None, :])
    elif metric == "correlation":
        x1 = x1 - nx.mean(x1, axis=1)[:, None]
        x2 = x2 - nx.mean(x2, axis=1)[:, None]
        nx1 = nx.sqrt(nx.einsum("ij,ij->i", x1, x1))
        nx2 = nx.sqrt(nx.einsum("ij,ij->i", x2, x2))
        return 1.0 - (nx.dot(x1, nx.transpose(x2)) / nx1[:, None] / nx2[None, :])
    else:
        if not get_backend(x1, x2).__name__ == "numpy":
            raise NotImplementedError()
        else:
            if isinstance(metric, str) and metric.endswith("minkowski"):
                return cdist(x1, x2, metric=metric, p=p, w=w)
            if w is not None:
                return cdist(x1, x2, metric=metric, w=w)
            return cdist(x1, x2, metric=metric)


def dist0(n, method="lin_square"):
    r"""Compute standard cost matrices of size (`n`, `n`) for OT problems

    Parameters
    ----------
    n : int
        Size of the cost matrix.
    method : str, optional
        Type of loss matrix chosen from:

        * 'lin_square' : linear sampling between 0 and `n-1`, quadratic loss

    Returns
    -------
    M : ndarray, shape (`n1`, `n2`)
        Distance matrix computed with given metric.
    """
    res = 0
    if method == "lin_square":
        x = np.arange(n, dtype=np.float64).reshape((n, 1))
        res = dist(x, x)
    return res


def cost_normalization(C, norm=None, return_value=False, value=None):
    r"""Apply normalization to the loss matrix

    Parameters
    ----------
    C : ndarray, shape (n1, n2)
        The cost matrix to normalize.
    norm : str
        Type of normalization from 'median', 'max', 'log', 'loglog'. Any
        other value do not normalize.

    Returns
    -------
    C : ndarray, shape (`n1`, `n2`)
        The input cost matrix normalized according to given norm.
    """

    nx = get_backend(C)

    if norm is None:
        pass
    elif norm == "median":
        if value is None:
            value = nx.median(C)
        C /= value
    elif norm == "max":
        if value is None:
            value = nx.max(C)
        C /= float(value)
    elif norm == "log":
        C = nx.log(1 + C)
    elif norm == "loglog":
        C = nx.log(1 + nx.log(1 + C))
    else:
        raise ValueError(
            "Norm %s is not a valid option.\n"
            "Valid options are:\n"
            "median, max, log, loglog" % norm
        )
    if return_value:
        return C, value
    else:
        return C


def dots(*args):
    r"""dots function for multiple matrix multiply"""
    nx = get_backend(*args)
    return reduce(nx.dot, args)


def is_all_finite(*args):
    r"""Tests element-wise for finiteness in all arguments."""
    nx = get_backend(*args)
    return all(not nx.any(~nx.isfinite(arg)) for arg in args)


def label_normalization(y, start=0, nx=None):
    r"""Transform labels to start at a given value

    Parameters
    ----------
    y : array-like, shape (n, )
        The vector of labels to be normalized.
    start : int
        Desired value for the smallest label in :math:`\mathbf{y}` (default=0)
    nx : Backend, optional
        Backend to perform computations on. If omitted, the backend defaults to that of `y`.

    Returns
    -------
    y : array-like, shape (`n1`, )
        The input vector of labels normalized according to given start value.
    """
    if nx is None:
        nx = get_backend(y)
    diff = nx.min(y) - start
    return y if diff == 0 else (y - diff)


def labels_to_masks(y, type_as=None, nx=None):
    r"""Transforms (n_samples,) vector of labels into a (n_samples, n_labels) matrix of masks.

    Parameters
    ----------
    y : array-like, shape (n_samples, )
        The vector of labels.
    type_as : array_like
        Array of the same type of the expected output.
    nx : Backend, optional
        Backend to perform computations on. If omitted, the backend defaults to that of `y`.

    Returns
    -------
    masks : array-like, shape (n_samples, n_labels)
        The (n_samples, n_labels) matrix of label masks.
    """
    if nx is None:
        nx = get_backend(y)
    if type_as is None:
        type_as = y
    labels_u, labels_idx = nx.unique(y, return_inverse=True)
    n_labels = labels_u.shape[0]
    masks = nx.eye(n_labels, type_as=type_as)[labels_idx]
    return masks


def parmap(f, X, nprocs="default"):
    r"""parallel map for multiprocessing.
    The function has been deprecated and only performs a regular map.
    """
    return list(map(f, X))


def check_params(**kwargs):
    r"""check_params: check whether some parameters are missing"""

    missing_params = []
    check = True

    for param in kwargs:
        if kwargs[param] is None:
            missing_params.append(param)

    if len(missing_params) > 0:
        print("POT - Warning: following necessary parameters are missing")
        for p in missing_params:
            print("\n", p)

        check = False

    return check


def check_random_state(seed):
    r"""Turn `seed` into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If `seed` is None, return the RandomState singleton used by np.random.
        If `seed` is an int, return a new RandomState instance seeded with `seed`.
        If `seed` is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "{} cannot be used to seed a numpy.random.RandomState instance".format(seed)
    )


def get_coordinate_circle(x):
    r"""For :math:`x\in S^1 \subset \mathbb{R}^2`, returns the coordinates in
    turn (in [0,1[).

    .. math::
        u = \frac{\pi + \mathrm{atan2}(-x_2,-x_1)}{2\pi}

    Parameters
    ----------
    x: ndarray, shape (n, 2)
        Samples on the circle with ambient coordinates

    Returns
    -------
    x_t: ndarray, shape (n,)
        Coordinates on [0,1[

    Examples
    --------
    >>> u = np.array([[0.2,0.5,0.8]]) * (2 * np.pi)
    >>> x1, y1 = np.cos(u), np.sin(u)
    >>> x = np.concatenate([x1, y1]).T
    >>> get_coordinate_circle(x)
    array([0.2, 0.5, 0.8])
    """
    nx = get_backend(x)
    x_t = (nx.atan2(-x[:, 1], -x[:, 0]) + np.pi) / (2 * np.pi)
    return x_t


def reduce_lazytensor(a, func, axis=None, nx=None, batch_size=100):
    """Reduce a LazyTensor along an axis with function fun using batches.

    When axis=None, reduce the LazyTensor to a scalar as a sum of fun over
    batches taken along dim.

    .. warning::
        This function works for tensor of any order but the reduction can be done
        only along the first two axis (or global). Also, in order to work, it requires that the slice of size `batch_size` along the axis to reduce (or axis 0 if `axis=None`) is can be computed and fits in memory.


    Parameters
    ----------
    a : LazyTensor
        LazyTensor to reduce
    func : callable
        Function to apply to the LazyTensor
    axis : int, optional
        Axis along which to reduce the LazyTensor. If None, reduce the
        LazyTensor to a scalar as a sum of fun over batches taken along axis 0.
        If 0 or 1 reduce the LazyTensor to a vector/matrix as a sum of fun over
        batches taken along axis.
    nx : Backend, optional
        Backend to use for the reduction
    batch_size : int, optional
        Size of the batches to use for the reduction (default=100)

    Returns
    -------
    res : array-like
        Result of the reduction

    """

    if nx is None:
        nx = get_backend(a[0:1])

    if axis is None:
        res = 0.0
        for i in range(0, a.shape[0], batch_size):
            res += func(a[i : i + batch_size])
        return res
    elif axis == 0:
        res = nx.zeros(a.shape[1:], type_as=a[0])
        if nx.__name__ in ["jax", "tf"]:
            lst = []
            for j in range(0, a.shape[1], batch_size):
                lst.append(func(a[:, j : j + batch_size], 0))
            return nx.concatenate(lst, axis=0)
        else:
            for j in range(0, a.shape[1], batch_size):
                res[j : j + batch_size] = func(a[:, j : j + batch_size], axis=0)
        return res
    elif axis == 1:
        if len(a.shape) == 2:
            shape = a.shape[0]
        else:
            shape = (a.shape[0], *a.shape[2:])
        res = nx.zeros(shape, type_as=a[0])
        if nx.__name__ in ["jax", "tf"]:
            lst = []
            for i in range(0, a.shape[0], batch_size):
                lst.append(func(a[i : i + batch_size], 1))
            return nx.concatenate(lst, axis=0)
        else:
            for i in range(0, a.shape[0], batch_size):
                res[i : i + batch_size] = func(a[i : i + batch_size], axis=1)
        return res

    else:
        raise (NotImplementedError("Only axis=None, 0 or 1 is implemented for now."))


def get_lowrank_lazytensor(Q, R, d=None, nx=None):
    """Get a low rank LazyTensor T=Q@R^T or T=Q@diag(d)@R^T

    Parameters
    ----------
    Q : ndarray, shape (n, r)
        First factor of the lowrank tensor
    R : ndarray, shape (m, r)
        Second factor of the lowrank tensor
    d : ndarray, shape (r,), optional
        Diagonal of the lowrank tensor
    nx : Backend, optional
        Backend to use for the reduction

    Returns
    -------
    T : LazyTensor
        Lowrank tensor T=Q@R^T or T=Q@diag(d)@R^T
    """

    if nx is None:
        nx = get_backend(Q, R, d)

    shape = (Q.shape[0], R.shape[0])

    if d is None:

        def func(i, j, Q, R):
            return nx.dot(Q[i], R[j].T)

        T = LazyTensor(shape, func, Q=Q, R=R)

    else:

        def func(i, j, Q, R, d):
            return nx.dot(Q[i] * d[None, :], R[j].T)

        T = LazyTensor(shape, func, Q=Q, R=R, d=d)

    return T


def get_parameter_pair(parameter):
    r"""Extract a pair of parameters from a given parameter
    Used in unbalanced OT and COOT solvers
    to handle marginal regularization and entropic regularization.

    Parameters
    ----------
    parameter : float or indexable object
    nx : backend object

    Returns
    -------
    param_1 : float
    param_2 : float
    """

    if isinstance(parameter, float) or isinstance(parameter, int):
        param_1, param_2 = parameter, parameter
    elif len(parameter) == 1:
        param_1, param_2 = parameter[0], parameter[0]
    else:
        if len(parameter) > 2:
            raise ValueError(
                "Parameter must be either a scalar, \
                             or an indexable object of length 1 or 2."
            )
        else:
            param_1, param_2 = parameter[0], parameter[1]

    return param_1, param_2


class deprecated(object):
    r"""Decorator to mark a function or class as deprecated.

    deprecated class from scikit-learn package
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/deprecation.py
    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.
    The optional extra argument will be appended to the deprecation message
    and the docstring.

    .. note::
        To use this with the default value for extra, use empty parentheses:

        >>> from ot.deprecation import deprecated  # doctest: +SKIP
        >>> @deprecated()  # doctest: +SKIP
        ... def some_function(): pass  # doctest: +SKIP

    Parameters
    ----------
    extra : str
        To be added to the deprecation messages.
    """

    # Adapted from http://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    def __init__(self, extra=""):
        self.extra = extra

    def __call__(self, obj):
        r"""Call method
        Parameters
        ----------
        obj : object
        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)

        cls.__init__ = wrapped

        wrapped.__name__ = "__init__"
        wrapped.__doc__ = self._update_doc(init.__doc__)
        wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        r"""Decorate function fun"""

        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        wrapped.__name__ = fun.__name__
        wrapped.__dict__ = fun.__dict__
        wrapped.__doc__ = self._update_doc(fun.__doc__)

        return wrapped

    def _update_doc(self, olddoc):
        newdoc = "DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            newdoc = "%s\n\n%s" % (newdoc, olddoc)
        return newdoc


def _is_deprecated(func):
    r"""Helper to check if func is wrapped by our deprecated decorator"""
    if sys.version_info < (3, 5):
        raise NotImplementedError("This is only available for python3.5 or above")
    closures = getattr(func, "__closure__", [])
    if closures is None:
        closures = []
    is_deprecated = "deprecated" in "".join(
        [c.cell_contents for c in closures if isinstance(c.cell_contents, str)]
    )
    return is_deprecated


class BaseEstimator(object):
    r"""Base class for most objects in POT

    Code adapted from sklearn BaseEstimator class

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    nx: Backend = None

    def _get_backend(self, *arrays):
        nx = get_backend(*[input_ for input_ in arrays if input_ is not None])
        if nx.__name__ in ("tf",):
            raise TypeError("Domain adaptation does not support TF backend.")
        self.nx = nx
        return nx

    @classmethod
    def _get_param_names(cls):
        r"""Get parameter names for the estimator"""

        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "POT estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        r"""Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and isinstance(w[0].category, DeprecationWarning):
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        r"""Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        # for key, value in iteritems(params):
        for key, value in params.items():
            split = key.split("__", 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for estimator %s. "
                        "Check the list of available parameters "
                        "with `estimator.get_params().keys()`." % (name, self)
                    )
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for estimator %s. "
                        "Check the list of available parameters "
                        "with `estimator.get_params().keys()`."
                        % (key, self.__class__.__name__)
                    )
                setattr(self, key, value)
        return self


class UndefinedParameter(Exception):
    r"""
    Aim at raising an Exception when a undefined parameter is called

    """

    pass


class OTResult:
    """Base class for OT results.

    Parameters
    ----------

    potentials : tuple of array-like, shape (`n1`, `n2`)
        Dual potentials, i.e. Lagrange multipliers for the marginal constraints.
        This pair of arrays has the same shape, numerical type
        and properties as the input weights "a" and "b".
    value : float, array-like
        Full transport cost, including possible regularization terms and
        quadratic term for Gromov Wasserstein solutions.
    value_linear : float, array-like
        The linear part of the transport cost, i.e. the product between the
        transport plan and the cost.
    value_quad : float, array-like
        The quadratic part of the transport cost for Gromov-Wasserstein
        solutions.
    plan : array-like, shape (`n1`, `n2`)
        Transport plan, encoded as a dense array.
    log : dict
        Dictionary containing potential information about the solver.
    backend : Backend
        Backend used to compute the results.
    sparse_plan : array-like, shape (`n1`, `n2`)
        Transport plan, encoded as a sparse array.
    lazy_plan : LazyTensor
        Transport plan, encoded as a symbolic POT or KeOps LazyTensor.
    status : int or str
        Status of the solver.
    batch_size : int
        Batch size used to compute the results/marginals for LazyTensor.

    Attributes
    ----------

    potentials : tuple of array-like, shape (`n1`, `n2`)
        Dual potentials, i.e. Lagrange multipliers for the marginal constraints.
        This pair of arrays has the same shape, numerical type
        and properties as the input weights "a" and "b".
    potential_a : array-like, shape (`n1`,)
        First dual potential, associated to the "source" measure "a".
    potential_b : array-like, shape (`n2`,)
        Second dual potential, associated to the "target" measure "b".
    value : float, array-like
        Full transport cost, including possible regularization terms and
        quadratic term for Gromov Wasserstein solutions.
    value_linear : float, array-like
        The linear part of the transport cost, i.e. the product between the
        transport plan and the cost.
    value_quad : float, array-like
        The quadratic part of the transport cost for Gromov-Wasserstein
        solutions.
    plan : array-like, shape (`n1`, `n2`)
        Transport plan, encoded as a dense array.
    sparse_plan : array-like, shape (`n1`, `n2`)
        Transport plan, encoded as a sparse array.
    lazy_plan : LazyTensor
        Transport plan, encoded as a symbolic POT or KeOps LazyTensor.
    marginals : tuple of array-like, shape (`n1`,), (`n2`,)
        Marginals of the transport plan: should be very close to "a" and "b"
        for balanced OT.
    marginal_a : array-like, shape (`n1`,)
        Marginal of the transport plan for the "source" measure "a".
    marginal_b : array-like, shape (`n2`,)
        Marginal of the transport plan for the "target" measure "b".

    """

    def __init__(
        self,
        potentials=None,
        value=None,
        value_linear=None,
        value_quad=None,
        plan=None,
        log=None,
        backend=None,
        sparse_plan=None,
        lazy_plan=None,
        status=None,
        batch_size=100,
    ):
        self._potentials = potentials
        self._value = value
        self._value_linear = value_linear
        self._value_quad = value_quad
        self._plan = plan
        self._log = log
        self._sparse_plan = sparse_plan
        self._lazy_plan = lazy_plan
        self._backend = backend if backend is not None else NumpyBackend()
        self._status = status
        self._batch_size = batch_size

        # I assume that other solvers may return directly
        # some primal objects?
        # In the code below, let's define the main quantities
        # that may be of interest to users.
        # An OT solver returns an object that inherits from OTResult
        # (e.g. SinkhornOTResult) and implements the relevant
        # methods (e.g. "plan" and "lazy_plan" but not "sparse_plan", etc.).
        # log is a dictionary containing potential information about the solver

    # Dual potentials --------------------------------------------

    def __repr__(self):
        s = "OTResult("
        if self._value is not None:
            s += "value={},".format(self._value)
        if self._value_linear is not None:
            s += "value_linear={},".format(self._value_linear)
        if self._plan is not None:
            s += "plan={}(shape={}),".format(
                self._plan.__class__.__name__, self._plan.shape
            )
        if self._lazy_plan is not None:
            s += "lazy_plan={}(shape={}),".format(
                self._lazy_plan.__class__.__name__, self._lazy_plan.shape
            )
        if s[-1] != "(":
            s = s[:-1] + ")"
        else:
            s = s + ")"
        return s

    @property
    def potentials(self):
        """Dual potentials, i.e. Lagrange multipliers for the marginal constraints.

        This pair of arrays has the same shape, numerical type
        and properties as the input weights "a" and "b".
        """
        return self._potentials

    @property
    def potential_a(self):
        """First dual potential, associated to the "source" measure "a"."""
        if self._potentials is not None:
            return self._potentials[0]
        else:
            return None

    @property
    def potential_b(self):
        """Second dual potential, associated to the "target" measure "b"."""
        if self._potentials is not None:
            return self._potentials[1]
        else:
            return None

    # Transport plan -------------------------------------------
    @property
    def plan(self):
        """Transport plan, encoded as a dense array."""
        # N.B.: We may catch out-of-memory errors and suggest
        # the use of lazy_plan or sparse_plan when appropriate.

        return self._plan

    @property
    def sparse_plan(self):
        """Transport plan, encoded as a sparse array."""
        if self._sparse_plan is not None:
            return self._sparse_plan
        elif self._plan is not None:
            return self._backend.tocsr(self._plan)
        else:
            return None

    @property
    def lazy_plan(self):
        """Transport plan, encoded as a symbolic KeOps LazyTensor."""
        return self._lazy_plan

    # Loss values --------------------------------

    @property
    def value(self):
        """Full transport cost, including possible regularization terms and
        quadratic term for Gromov Wasserstein solutions."""
        return self._value

    @property
    def value_linear(self):
        """The "minimal" transport cost, i.e. the product between the transport plan and the cost."""
        return self._value_linear

    @property
    def value_quad(self):
        """The quadratic part of the transport cost for Gromov-Wasserstein solutions."""
        return self._value_quad

    # Marginal constraints -------------------------
    @property
    def marginals(self):
        """Marginals of the transport plan: should be very close to "a" and "b"
        for balanced OT."""
        if self._plan is not None:
            return self.marginal_a, self.marginal_b
        else:
            return None

    @property
    def marginal_a(self):
        """First marginal of the transport plan, with the same shape as "a"."""
        if self._plan is not None:
            return self._backend.sum(self._plan, 1)
        elif self._lazy_plan is not None:
            lp = self._lazy_plan
            bs = self._batch_size
            nx = self._backend
            return reduce_lazytensor(lp, nx.sum, axis=1, nx=nx, batch_size=bs)
        else:
            return None

    @property
    def marginal_b(self):
        """Second marginal of the transport plan, with the same shape as "b"."""
        if self._plan is not None:
            return self._backend.sum(self._plan, 0)
        elif self._lazy_plan is not None:
            lp = self._lazy_plan
            bs = self._batch_size
            nx = self._backend
            return reduce_lazytensor(lp, nx.sum, axis=0, nx=nx, batch_size=bs)
        else:
            return None

    @property
    def status(self):
        """Optimization status of the solver."""
        return self._status

    @property
    def log(self):
        """Dictionary containing potential information about the solver."""
        return self._log

    # Barycentric mappings -------------------------
    # Return the displacement vectors as an array
    # that has the same shape as "xa"/"xb" (for samples)
    # or "a"/"b" * D (for images)?

    @property
    def a_to_b(self):
        """Displacement vectors from the first to the second measure."""
        raise NotImplementedError()

    @property
    def b_to_a(self):
        """Displacement vectors from the second to the first measure."""
        raise NotImplementedError()

    # # Wasserstein barycenters ----------------------
    # @property
    # def masses(self):
    #     """Masses for the Wasserstein barycenter."""
    #     raise NotImplementedError()

    # @property
    # def samples(self):
    #     """Sample locations for the Wasserstein barycenter."""
    #     raise NotImplementedError()

    # Miscellaneous --------------------------------

    @property
    def citation(self):
        """Appropriate citation(s) for this result, in plain text and BibTex formats."""

        # The string below refers to the POT library:
        # successor methods may concatenate the relevant references
        # to the original definitions, solvers and underlying numerical backends.
        return """POT library:

            POT Python Optimal Transport library, Journal of Machine Learning Research, 22(78):1−8, 2021.
            Website: https://pythonot.github.io/
            Rémi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, Aurélie Boisbunon, Stanislas Chambon, Laetitia Chapel, Adrien Corenflos, Kilian Fatras, Nemo Fournier, Léo Gautheron, Nathalie T.H. Gayraud, Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet, Antony Schutz, Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, Titouan Vayer;

            @article{flamary2021pot,
              author  = {R{\'e}mi Flamary and Nicolas Courty and Alexandre Gramfort and Mokhtar Z. Alaya and Aur{\'e}lie Boisbunon and Stanislas Chambon and Laetitia Chapel and Adrien Corenflos and Kilian Fatras and Nemo Fournier and L{\'e}o Gautheron and Nathalie T.H. Gayraud and Hicham Janati and Alain Rakotomamonjy and Ievgen Redko and Antoine Rolet and Antony Schutz and Vivien Seguy and Danica J. Sutherland and Romain Tavenard and Alexander Tong and Titouan Vayer},
              title   = {{POT}: {Python} {Optimal} {Transport}},
              journal = {Journal of Machine Learning Research},
              year    = {2021},
              volume  = {22},
              number  = {78},
              pages   = {1-8},
              url     = {http://jmlr.org/papers/v22/20-451.html}
            }
        """


class LazyTensor(object):
    """A lazy tensor is a tensor that is not stored in memory. Instead, it is
    defined by a function that computes its values on the fly from slices.

    Parameters
    ----------

    shape : tuple
        shape of the tensor
    getitem : callable
        function that computes the values of the indices/slices and tensors
        as arguments

    kwargs : dict
        named arguments for the function, those names will be used as attributed
        of the LazyTensor object

    Examples
    --------
    >>> import numpy as np
    >>> v = np.arange(5)
    >>> def getitem(i,j, v):
    ...     return v[i,None]+v[None,j]
    >>> T = LazyTensor((5,5),getitem, v=v)
    >>> T[1,2]
    array([3])
    >>> T[1,:]
    array([[1, 2, 3, 4, 5]])
    >>> T[:]
    array([[0, 1, 2, 3, 4],
           [1, 2, 3, 4, 5],
           [2, 3, 4, 5, 6],
           [3, 4, 5, 6, 7],
           [4, 5, 6, 7, 8]])

    """

    def __init__(self, shape, getitem, **kwargs):
        self._getitem = getitem
        self.shape = shape
        self.ndim = len(shape)
        self.kwargs = kwargs

        # set attributes for named arguments/arrays
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        k = []
        if isinstance(key, int) or isinstance(key, slice):
            k.append(key)
            for i in range(self.ndim - 1):
                k.append(slice(None))
        elif isinstance(key, tuple):
            k = list(key)
            for i in range(self.ndim - len(key)):
                k.append(slice(None))
        else:
            raise NotImplementedError(
                "Only integer, slice, and tuple indexing is supported"
            )

        return self._getitem(*k, **self.kwargs)

    def __repr__(self):
        return "LazyTensor(shape={},attributes=({}))".format(
            self.shape, ",".join(self.kwargs.keys())
        )


def proj_SDP(S, nx=None, vmin=0.0):
    """
    Project a symmetric matrix onto the space of symmetric matrices with
    eigenvalues larger or equal to `vmin`.

    Parameters
    ----------
    S : array_like (n, d, d) or (d, d)
        The input symmetric matrix or matrices.
    nx : module, optional
        The numerical backend module to use. If not provided, the backend will
        be fetched from the input matrix `S`.
    vmin : float, optional
        The minimum value for the eigenvalues. Eigenvalues below this value will
        be clipped to vmin.

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends.

    Returns
    -------
    P : ndarray (n, d, d) or (d, d)
        The projected symmetric positive definite matrix.

    """
    if nx is None:
        nx = get_backend(S)

    w, P = nx.eigh(S)
    w = nx.clip(w, vmin, None)

    if len(S.shape) == 2:  # input was (d, d)
        return P @ nx.diag(w) @ P.T

    else:  # input was (n, d, d): broadcasting
        Q = nx.einsum("ijk,ik->ijk", P, w)  # Q[i] = P[i] @ diag(w[i])
        # R[i] = Q[i] @ P[i].T
        return nx.einsum("ijk,ikl->ijl", Q, nx.transpose(P, (0, 2, 1)))


def exp_bures(Sigma, S, nx=None):
    r"""
    Exponential map in Bures-Wasserstein space at Sigma:

    .. math::
        \exp_\Sigma(S) = (I_d+S)\Sigma(I_d+S).

    Parameters
    ----------
    Sigma : array-like (d,d)
        SPD matrix
    S : array-like (d,d)
        Symmetric matrix
    nx : module, optional
        The numerical backend module to use. If not provided, the backend will
        be fetched from the input matrices `Sigma, S`.

    Returns
    -------
    P : array-like (d,d)
        SPD matrix obtained as the exponential map of S at Sigma
    """
    if nx is None:
        nx = get_backend(Sigma, S)
    d = S.shape[-1]
    Id = nx.eye(d, type_as=S)
    C = Id + S

    return nx.einsum("ij,jk,kl -> il", C, Sigma, C)


def check_number_threads(numThreads):
    """Checks whether or not the requested number of threads has a valid value.

    Parameters
    ----------
    numThreads : int or str
        The requested number of threads, should either be a strictly positive integer or "max" or None

    Returns
    -------
    numThreads : int
        Corrected number of threads
    """
    if (numThreads is None) or (
        isinstance(numThreads, str) and numThreads.lower() == "max"
    ):
        return -1
    if (not isinstance(numThreads, int)) or numThreads < 1:
        raise ValueError(
            'numThreads should either be "max" or a strictly positive integer'
        )
    return numThreads


def fun_to_numpy(fun, arr, nx, warn=True):
    """Convert a function to a numpy function.

    Parameters
    ----------
    fun : callable
        The function to convert.
    arr : array-like
        The input to test the function. Can be from any backend.
    nx : Backend
        The backend to use for the conversion.
    warn : bool, optional
        Whether to raise a warning if the function is not compatible with numpy.
        Default is True.
    Returns
    -------
    fun_numpy : callable
        The converted function.
    """
    if arr is None:
        raise ValueError("arr should not be None to test fun")

    nx_arr = get_backend(arr)
    if nx_arr.__name__ != "numpy":
        arr = nx.to_numpy(arr)
    try:
        fun(arr)
        return fun
    except BaseException:
        if warn:
            warnings.warn(
                "The callable function should be able to handle numpy arrays, a compatible function is created and comes with overhead"
            )

        def fun_numpy(x):
            return nx.to_numpy(fun(nx.from_numpy(x)))

        return fun_numpy
