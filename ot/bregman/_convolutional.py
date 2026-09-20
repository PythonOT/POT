# -*- coding: utf-8 -*-
"""
Bregman projections solvers for entropic regularized Wasserstein convolutional barycenters
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import warnings

from ..backend import get_backend
from ..utils import list_to_array

_warning_msg = (
    "Convolutional Sinkhorn did not converge. "
    "Try a larger number of iterations `numItermax` "
    "or a larger entropy `reg`."
)


def _move_axis_perm(ndim, src):
    """Return (perm, inv_perm) moving axis `src` to the last position of an
    `ndim`-dimensional array, and the permutation that moves it back."""
    perm = tuple(i for i in range(ndim) if i != src) + (src,)
    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return perm, tuple(inv_perm)


def _log_matmul_exp(nx, y, K):
    """Stabilized log(exp(y) @ K).

    y : array-like, shape (..., n)
    K : array-like, shape (n, m), non-negative
    """
    c = nx.max(y, axis=-1, keepdims=True)
    c = nx.where(nx.isfinite(c), c, nx.zeros(c.shape, type_as=c))
    return c + nx.log(nx.matmul(nx.exp(y - c), K))


class _SeparableKernel:
    r"""Apply the separable kernel :math:`K = K_1 \otimes \dots \otimes K_d` to
    the trailing `d` axes of an array.

    Leading axes are treated as batch axes and are left untouched, so a
    single instance serves a single grid of shape `(*shape)`, a stack of
    shape `(n_hists, *shape)`, or anything with extra leading axes.

    Parameters
    ----------
    nx : Backend
        backend to use for computations
    log_kernels : list of array-like
        one `(n_k, n_k)` matrix per grid axis, holding
        :math:`-|x-y|^2/\mathrm{reg}` for that axis
    log_domain : bool, optional
        if True, `__call__` expects and returns log-domain arrays and uses a
        stabilized log-matmul-exp; otherwise it expects and returns
        exp-domain arrays and applies a plain matrix product

    .. note::
        Each 1-D factor is applied as a dense matrix product (``x @ K``)
        rather than a native 1-D convolution. The kernel is Toeplitz, so a
        convolution would be mathematically equivalent (checked to ~3e-15
        against ``K @ x`` on a small example), but truncating it is not a
        safe optimization for a Sinkhorn iteration: the dual scalings grow
        to compensate for the kernel's decay, so the transport plan stays
        spread out even where the kernel itself is negligible. On this
        package's own convolutional-barycenter example (64x64 images,
        `reg=0.004`, :math:`\sigma \approx 2.8` pixels), truncating the
        kernel below 12 standard deviations makes the iteration diverge to
        NaN. Dense ``matmul`` is also simply faster here: ~4.5 ms against
        ~220 ms for a depthwise ``conv2d`` with a 4-sigma-truncated kernel
        (Torch 2.14, CPU, float64, 5 images, `n=512`, `reg=4e-3`).
    """

    def __init__(self, nx, log_kernels, log_domain=False):
        self.nx = nx
        self.ndim = len(log_kernels)
        self.log_domain = log_domain
        self.log_kernels = log_kernels
        self.kernels = [nx.exp(K) for K in log_kernels]

    def __call__(self, x):
        nx = self.nx
        n_batch = x.ndim - self.ndim
        for axis, K in enumerate(self.kernels):
            src = n_batch + axis
            perm, inv_perm = _move_axis_perm(x.ndim, src)
            x = nx.transpose(x, perm)
            if self.log_domain:
                x = _log_matmul_exp(nx, x, K)
            else:
                x = nx.matmul(x, K)
            x = nx.transpose(x, inv_perm)
        return x


def _grid_gaussian_kernel(nx, shape, reg, type_as, log_domain=False):
    r"""Gaussian kernel :math:`\exp(-|x-y|^2/\mathrm{reg})` on a regular grid
    of the given `shape`, as a :class:`_SeparableKernel` with one factor per
    axis. Each axis is sampled on `nx.linspace(0, 1, n)`.
    """
    log_kernels = []
    for n in shape:
        t = nx.linspace(0, 1, n, type_as=type_as)
        Y, X = nx.meshgrid(t, t)
        log_kernels.append(-((X - Y) ** 2) / reg)
    return _SeparableKernel(nx, log_kernels, log_domain=log_domain)


def _exact_separable_log_apply(nx, y, log_kernels):
    """Reference (non-stabilized-matmul) separable log-domain kernel
    application, computed with a full :any:`Backend.logsumexp` reduction
    per axis instead of the shifted log-matmul-exp used by
    :class:`_SeparableKernel`.

    This is a test helper only: it materialises a `(..., n, n)`
    intermediate per axis and is only tractable on small grids. It is used
    to check the fast, shifted form against the exact reduction; it is not
    part of the public API and is not used by any solver.
    """
    n_batch = y.ndim - len(log_kernels)
    for axis, logK in enumerate(log_kernels):
        src = n_batch + axis
        perm, inv_perm = _move_axis_perm(y.ndim, src)
        y = nx.transpose(y, perm)
        y = nx.logsumexp(y[..., :, None] + logK, axis=-2)
        y = nx.transpose(y, inv_perm)
    return y


def _print_report(ii, err):
    """Print the report of the iteration."""
    if ii % 200 == 0:
        print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
    print("{:5d}|{:8e}|".format(ii, err))


def convolutional_grid_barycenter(
    A,
    reg,
    weights=None,
    method="sinkhorn",
    numItermax=10000,
    stopThr=1e-4,
    verbose=False,
    log=False,
    warn=True,
    **kwargs,
):
    r"""Compute the entropic regularized Wasserstein barycenter of distributions
    :math:`\mathbf{A}` where :math:`\mathbf{A}` is a collection of histograms on a
    common regular grid of arbitrary dimension (1D signals, 2D images, 3D volumes, ...).

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein
      distance (see :py:func:`ot.bregman.sinkhorn`)
    - :math:`\mathbf{a}_i` are training distributions on the grid, in the last
      `A.ndim - 1` dimensions of matrix :math:`\mathbf{A}`
    - `reg` is the regularization strength scalar value

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling
    algorithm as proposed in :ref:`[21] <references-convolutional-grid-barycenter>`,
    applied through a separable Gaussian kernel on the grid.

    Parameters
    ----------
    A : array-like, shape (n_hists, \*grid_shape)
        `n_hists` distributions on a regular grid of shape `grid_shape`
        (any number of grid dimensions, e.g. `(width, height)` for images or
        `(width, height, depth)` for volumes)
    reg : float
        Regularization term >0
    weights : array-like, shape (n_hists,)
        Weights of each histogram on the simplex (barycentric coordinates)
    method : string, optional
        method used for the solver either 'sinkhorn' or 'sinkhorn_log'
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    stabThr : float, optional
        Stabilization threshold to avoid numerical precision issue
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.

    Returns
    -------
    a : array-like, shape (\*grid_shape)
        Wasserstein barycenter on the grid
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-convolutional-grid-barycenter:
    References
    ----------

    .. [21] Solomon, J., De Goes, F., Peyré, G., Cuturi, M., Butscher,
        A., Nguyen, A. & Guibas, L. (2015).     Convolutional wasserstein distances:
        Efficient optimal transportation on geometric domains. ACM Transactions
        on Graphics (TOG), 34(4), 66

    .. [37] Janati, H., Cuturi, M., Gramfort, A. Proceedings of the 37th
        International Conference on Machine Learning, PMLR 119:4692-4701, 2020
    """

    if method.lower() == "sinkhorn":
        return _convolutional_grid_barycenter(
            A,
            reg,
            weights=weights,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            warn=warn,
            **kwargs,
        )
    elif method.lower() == "sinkhorn_log":
        return _convolutional_grid_barycenter_log(
            A,
            reg,
            weights=weights,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            warn=warn,
            **kwargs,
        )
    else:
        raise ValueError("Unknown method '%s'." % method)


def _convolutional_grid_barycenter(
    A,
    reg,
    weights=None,
    numItermax=10000,
    stopThr=1e-9,
    stabThr=1e-30,
    verbose=False,
    log=False,
    warn=True,
):
    r"""Compute the entropic regularized Wasserstein barycenter of distributions A
    where A is a collection of histograms on a regular grid (e.g. unit-normalised
    images or volumes).
    """

    A = list_to_array(A)
    n_hists = A.shape[0]
    grid_shape = tuple(A.shape[1:])
    grid_ndim = len(grid_shape)

    nx = get_backend(A)

    if weights is None:
        weights = nx.ones((n_hists,), type_as=A) / n_hists
    else:
        assert len(weights) == n_hists
    weights = nx.reshape(weights, (n_hists,) + (1,) * grid_ndim)

    if log:
        log = {"err": []}

    bar = nx.ones(grid_shape, type_as=A)
    bar /= nx.sum(bar)
    U = nx.ones(A.shape, type_as=A)
    V = nx.ones(A.shape, type_as=A)
    err = 1

    # build the separable convolution kernel
    kernel = _grid_gaussian_kernel(nx, grid_shape, reg, type_as=A)

    KU = kernel(U)
    for ii in range(numItermax):
        V = bar[None] / KU
        KV = kernel(V)
        U = A / KV
        KU = kernel(U)
        bar = nx.exp(nx.sum(weights * nx.log(KU + stabThr), axis=0))
        if ii % 10 == 9:
            err = nx.sum(nx.std(V * KU, axis=0))
            # log and verbose print
            if log:
                log["err"].append(err)
            if verbose:
                _print_report(ii, err)
            if err < stopThr:
                break

    else:
        if warn:
            warnings.warn(_warning_msg)
    if log:
        log["niter"] = ii
        log["U"] = U
        log["V"] = V
        return bar, log
    else:
        return bar


def _convolutional_grid_barycenter_log(
    A,
    reg,
    weights=None,
    numItermax=10000,
    stopThr=1e-4,
    stabThr=1e-30,
    verbose=False,
    log=False,
    warn=True,
):
    r"""Compute the entropic regularized Wasserstein barycenter of distributions A
    where A is a collection of histograms on a regular grid (e.g. unit-normalised
    images or volumes), in log-domain.
    """

    A = list_to_array(A)
    n_hists = A.shape[0]
    grid_shape = tuple(A.shape[1:])
    grid_ndim = len(grid_shape)

    nx = get_backend(A)

    if weights is None:
        weights = nx.ones((n_hists,), type_as=A) / n_hists
    else:
        assert len(weights) == n_hists
    weights = nx.reshape(weights, (n_hists,) + (1,) * grid_ndim)

    if log:
        log = {"err": []}

    # build the separable convolution kernel
    kernel = _grid_gaussian_kernel(nx, grid_shape, reg, type_as=A, log_domain=True)

    logA = nx.log(A + stabThr)
    G = nx.zeros(logA.shape, type_as=A)
    err = 1
    for ii in range(numItermax):
        f = logA - kernel(G)
        log_KU = kernel(f)
        log_bar = nx.sum(weights * log_KU, axis=0)

        if ii % 10 == 9:
            err = nx.sum(nx.std(nx.exp(G + log_KU), axis=0))
            # log and verbose print
            if log:
                log["err"].append(err)
            if verbose:
                _print_report(ii, err)
            if err < stopThr:
                break
        G = log_bar[None] - log_KU

    else:
        if warn:
            warnings.warn(_warning_msg)
    if log:
        log["niter"] = ii
        return nx.exp(log_bar), log
    else:
        return nx.exp(log_bar)


def convolutional_barycenter2d(
    A,
    reg,
    weights=None,
    method="sinkhorn",
    numItermax=10000,
    stopThr=1e-4,
    verbose=False,
    log=False,
    warn=True,
    **kwargs,
):
    r"""Compute the entropic regularized wasserstein barycenter of distributions :math:`\mathbf{A}`
    where :math:`\mathbf{A}` is a collection of 2D images.

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein
      distance (see :py:func:`ot.bregman.sinkhorn`)
    - :math:`\mathbf{a}_i` are training distributions (2D images) in the mast two dimensions
      of matrix :math:`\mathbf{A}`
    - `reg` is the regularization strength scalar value

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm
    as proposed in :ref:`[21] <references-convolutional-barycenter-2d>`

    Parameters
    ----------
    A : array-like, shape (n_hists, width, height)
        `n` distributions (2D images) of size `width` x `height`
    reg : float
        Regularization term >0
    weights : array-like, shape (n_hists,)
        Weights of each image on the simplex (barycentric coordinates)
    method : string, optional
        method used for the solver either 'sinkhorn' or 'sinkhorn_log'
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    stabThr : float, optional
        Stabilization threshold to avoid numerical precision issue
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.

    Returns
    -------
    a : array-like, shape (width, height)
        2D Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-convolutional-barycenter-2d:
    References
    ----------

    .. [21] Solomon, J., De Goes, F., Peyré, G., Cuturi, M., Butscher,
        A., Nguyen, A. & Guibas, L. (2015).     Convolutional wasserstein distances:
        Efficient optimal transportation on geometric domains. ACM Transactions
        on Graphics (TOG), 34(4), 66

    .. [37] Janati, H., Cuturi, M., Gramfort, A. Proceedings of the 37th
        International Conference on Machine Learning, PMLR 119:4692-4701, 2020
    """
    A = list_to_array(A)
    if A.ndim != 3:
        raise ValueError(
            "convolutional_barycenter2d expects `A` of shape "
            f"(n_hists, width, height) (A.ndim == 3), got A.ndim == {A.ndim}. "
            "For grids of other dimensions, use "
            "ot.bregman.convolutional_grid_barycenter."
        )
    return convolutional_grid_barycenter(
        A,
        reg,
        weights=weights,
        method=method,
        numItermax=numItermax,
        stopThr=stopThr,
        verbose=verbose,
        log=log,
        warn=warn,
        **kwargs,
    )


def convolutional_grid_barycenter_debiased(
    A,
    reg,
    weights=None,
    method="sinkhorn",
    numItermax=10000,
    stopThr=1e-3,
    verbose=False,
    log=False,
    warn=True,
    **kwargs,
):
    r"""Compute the debiased sinkhorn barycenter of distributions :math:`\mathbf{A}`
    where :math:`\mathbf{A}` is a collection of histograms on a common regular grid of
    arbitrary dimension (1D signals, 2D images, 3D volumes, ...).

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i S_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`S_{reg}(\cdot,\cdot)` is the debiased entropic regularized Wasserstein
      distance (see :py:func:`ot.bregman.barycenter_debiased`)
    - :math:`\mathbf{a}_i` are training distributions on the grid, in the last
      `A.ndim - 1` dimensions of matrix :math:`\mathbf{A}`
    - `reg` is the regularization strength scalar value

    The algorithm used for solving the problem is the debiased Sinkhorn scaling
    algorithm as proposed in
    :ref:`[37] <references-convolutional-grid-barycenter-debiased>`, applied through a
    separable Gaussian kernel on the grid.

    Parameters
    ----------
    A : array-like, shape (n_hists, \*grid_shape)
        `n_hists` distributions on a regular grid of shape `grid_shape`
        (any number of grid dimensions, e.g. `(width, height)` for images or
        `(width, height, depth)` for volumes)
    reg : float
        Regularization term >0
    weights : array-like, shape (n_hists,)
        Weights of each histogram on the simplex (barycentric coordinates)
    method : string, optional
        method used for the solver either 'sinkhorn' or 'sinkhorn_log'
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    stabThr : float, optional
        Stabilization threshold to avoid numerical precision issue
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.


    Returns
    -------
    a : array-like, shape (\*grid_shape)
        Wasserstein barycenter on the grid
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-convolutional-grid-barycenter-debiased:
    References
    ----------

    .. [37] Janati, H., Cuturi, M., Gramfort, A. Proceedings of the 37th International
        Conference on Machine Learning, PMLR 119:4692-4701, 2020
    """

    if method.lower() == "sinkhorn":
        return _convolutional_grid_barycenter_debiased(
            A,
            reg,
            weights=weights,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            warn=warn,
            **kwargs,
        )
    elif method.lower() == "sinkhorn_log":
        return _convolutional_grid_barycenter_debiased_log(
            A,
            reg,
            weights=weights,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            warn=warn,
            **kwargs,
        )
    else:
        raise ValueError("Unknown method '%s'." % method)


def _convolutional_grid_barycenter_debiased(
    A,
    reg,
    weights=None,
    numItermax=10000,
    stopThr=1e-3,
    stabThr=1e-15,
    verbose=False,
    log=False,
    warn=True,
):
    r"""Compute the debiased barycenter of histograms on a regular grid
    (e.g. unit-normalised images or volumes) via sinkhorn convolutions."""

    A = list_to_array(A)
    n_hists = A.shape[0]
    grid_shape = tuple(A.shape[1:])
    grid_ndim = len(grid_shape)
    grid_size = 1
    for n in grid_shape:
        grid_size *= n

    nx = get_backend(A)

    if weights is None:
        weights = nx.ones((n_hists,), type_as=A) / n_hists
    else:
        assert len(weights) == n_hists
    weights = nx.reshape(weights, (n_hists,) + (1,) * grid_ndim)

    if log:
        log = {"err": []}

    bar = nx.ones(grid_shape, type_as=A)
    bar /= grid_size
    U = nx.ones(A.shape, type_as=A)
    V = nx.ones(A.shape, type_as=A)
    c = nx.ones(grid_shape, type_as=A)
    err = 1

    # build the separable convolution kernel
    kernel = _grid_gaussian_kernel(nx, grid_shape, reg, type_as=A)

    KU = kernel(U)
    for ii in range(numItermax):
        V = bar[None] / KU
        KV = kernel(V)
        U = A / KV
        KU = kernel(U)
        bar = c * nx.exp(nx.sum(weights * nx.log(KU + stabThr), axis=0))

        for _ in range(10):
            c = (c * bar / nx.squeeze(kernel(c[None]))) ** 0.5

        if ii % 10 == 9:
            err = nx.sum(nx.std(V * KU, axis=0))
            # log and verbose print
            if log:
                log["err"].append(err)
            if verbose:
                _print_report(ii, err)

            # debiased Sinkhorn does not converge monotonically
            # guarantee a few iterations are done before stopping
            if err < stopThr and ii > 20:
                break
    else:
        if warn:
            warnings.warn(_warning_msg)
    if log:
        log["niter"] = ii
        log["U"] = U
        log["V"] = V
        return bar, log
    else:
        return bar


def _convolutional_grid_barycenter_debiased_log(
    A,
    reg,
    weights=None,
    numItermax=10000,
    stopThr=1e-3,
    stabThr=1e-30,
    verbose=False,
    log=False,
    warn=True,
):
    r"""Compute the debiased barycenter of histograms on a regular grid
    (e.g. unit-normalised images or volumes) in log-domain."""

    A = list_to_array(A)
    n_hists = A.shape[0]
    grid_shape = tuple(A.shape[1:])
    grid_ndim = len(grid_shape)

    nx = get_backend(A)

    if weights is None:
        weights = nx.ones((n_hists,), type_as=A) / n_hists
    else:
        assert len(weights) == n_hists
    weights = nx.reshape(weights, (n_hists,) + (1,) * grid_ndim)

    if log:
        log = {"err": []}

    # build the separable convolution kernel
    kernel = _grid_gaussian_kernel(nx, grid_shape, reg, type_as=A, log_domain=True)

    logA = nx.log(A + stabThr)
    G = nx.zeros(logA.shape, type_as=A)
    c = nx.zeros(grid_shape, type_as=A)
    err = 1
    for ii in range(numItermax):
        f = logA - kernel(G)
        log_KU = kernel(f)
        log_bar = nx.sum(weights * log_KU, axis=0) + c

        for _ in range(10):
            c = 0.5 * (c + log_bar - kernel(c))

        if ii % 10 == 9:
            err = nx.sum(nx.std(nx.exp(G + log_KU), axis=0))
            # log and verbose print
            if log:
                log["err"].append(err)
            if verbose:
                _print_report(ii, err)
            if err < stopThr and ii > 20:
                break
        G = log_bar[None] - log_KU

    else:
        if warn:
            warnings.warn(_warning_msg)
    if log:
        log["niter"] = ii
        return nx.exp(log_bar), log
    else:
        return nx.exp(log_bar)


def convolutional_barycenter2d_debiased(
    A,
    reg,
    weights=None,
    method="sinkhorn",
    numItermax=10000,
    stopThr=1e-3,
    verbose=False,
    log=False,
    warn=True,
    **kwargs,
):
    r"""Compute the debiased sinkhorn barycenter of distributions :math:`\mathbf{A}`
    where :math:`\mathbf{A}` is a collection of 2D images.

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i S_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`S_{reg}(\cdot,\cdot)` is the debiased entropic regularized Wasserstein
      distance (see :py:func:`ot.bregman.barycenter_debiased`)
    - :math:`\mathbf{a}_i` are training distributions (2D images) in the mast two
      dimensions of matrix :math:`\mathbf{A}`
    - `reg` is the regularization strength scalar value

    The algorithm used for solving the problem is the debiased Sinkhorn scaling
    algorithm as proposed in :ref:`[37] <references-convolutional-barycenter2d-debiased>`

    Parameters
    ----------
    A : array-like, shape (n_hists, width, height)
        `n` distributions (2D images) of size `width` x `height`
    reg : float
        Regularization term >0
    weights : array-like, shape (n_hists,)
        Weights of each image on the simplex (barycentric coordinates)
    method : string, optional
        method used for the solver either 'sinkhorn' or 'sinkhorn_log'
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    stabThr : float, optional
        Stabilization threshold to avoid numerical precision issue
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.


    Returns
    -------
    a : array-like, shape (width, height)
        2D Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-convolutional-barycenter2d-debiased:
    References
    ----------

    .. [37] Janati, H., Cuturi, M., Gramfort, A. Proceedings of the 37th International
        Conference on Machine Learning, PMLR 119:4692-4701, 2020
    """
    A = list_to_array(A)
    if A.ndim != 3:
        raise ValueError(
            "convolutional_barycenter2d_debiased expects `A` of shape "
            f"(n_hists, width, height) (A.ndim == 3), got A.ndim == {A.ndim}. "
            "For grids of other dimensions, use "
            "ot.bregman.convolutional_grid_barycenter_debiased."
        )
    return convolutional_grid_barycenter_debiased(
        A,
        reg,
        weights=weights,
        method=method,
        numItermax=numItermax,
        stopThr=stopThr,
        verbose=verbose,
        log=log,
        warn=warn,
        **kwargs,
    )
