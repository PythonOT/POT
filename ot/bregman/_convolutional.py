# -*- coding: utf-8 -*-
"""
Bregman projections solvers for entropic regularized Wasserstein convolutional barycenters
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import warnings

from ..utils import list_to_array
from ..backend import get_backend


def convolutional_barycenter2d(A, reg, weights=None, method="sinkhorn", numItermax=10000,
                               stopThr=1e-4, verbose=False, log=False,
                               warn=True, **kwargs):
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
        Weights of each image on the simplex (barycentric coodinates)
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

    if method.lower() == 'sinkhorn':
        return _convolutional_barycenter2d(A, reg, weights=weights,
                                           numItermax=numItermax,
                                           stopThr=stopThr, verbose=verbose,
                                           log=log, warn=warn,
                                           **kwargs)
    elif method.lower() == 'sinkhorn_log':
        return _convolutional_barycenter2d_log(A, reg, weights=weights,
                                               numItermax=numItermax,
                                               stopThr=stopThr, verbose=verbose,
                                               log=log, warn=warn,
                                               **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def _convolutional_barycenter2d(A, reg, weights=None, numItermax=10000,
                                stopThr=1e-9, stabThr=1e-30, verbose=False,
                                log=False, warn=True):
    r"""Compute the entropic regularized wasserstein barycenter of distributions A
    where A is a collection of 2D images.
    """

    A = list_to_array(A)

    nx = get_backend(A)

    if weights is None:
        weights = nx.ones((A.shape[0],), type_as=A) / A.shape[0]
    else:
        assert (len(weights) == A.shape[0])

    if log:
        log = {'err': []}

    bar = nx.ones(A.shape[1:], type_as=A)
    bar /= nx.sum(bar)
    U = nx.ones(A.shape, type_as=A)
    V = nx.ones(A.shape, type_as=A)
    err = 1

    # build the convolution operator
    # this is equivalent to blurring on horizontal then vertical directions
    t = nx.linspace(0, 1, A.shape[1], type_as=A)
    [Y, X] = nx.meshgrid(t, t)
    K1 = nx.exp(-(X - Y) ** 2 / reg)

    t = nx.linspace(0, 1, A.shape[2], type_as=A)
    [Y, X] = nx.meshgrid(t, t)
    K2 = nx.exp(-(X - Y) ** 2 / reg)

    def convol_imgs(imgs):
        kx = nx.einsum("...ij,kjl->kil", K1, imgs)
        kxy = nx.einsum("...ij,klj->kli", K2, kx)
        return kxy

    KU = convol_imgs(U)
    for ii in range(numItermax):
        V = bar[None] / KU
        KV = convol_imgs(V)
        U = A / KV
        KU = convol_imgs(U)
        bar = nx.exp(
            nx.sum(weights[:, None, None] * nx.log(KU + stabThr), axis=0)
        )
        if ii % 10 == 9:
            err = nx.sum(nx.std(V * KU, axis=0))
            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if ii % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
            if err < stopThr:
                break

    else:
        if warn:
            warnings.warn("Convolutional Sinkhorn did not converge. "
                          "Try a larger number of iterations `numItermax` "
                          "or a larger entropy `reg`.")
    if log:
        log['niter'] = ii
        log['U'] = U
        return bar, log
    else:
        return bar


def _convolutional_barycenter2d_log(A, reg, weights=None, numItermax=10000,
                                    stopThr=1e-4, stabThr=1e-30, verbose=False,
                                    log=False, warn=True):
    r"""Compute the entropic regularized wasserstein barycenter of distributions A
    where A is a collection of 2D images in log-domain.
    """

    A = list_to_array(A)

    nx = get_backend(A)
    if nx.__name__ in ("jax", "tf"):
        raise NotImplementedError(
            "Log-domain functions are not yet implemented"
            " for Jax and TF. Use numpy or torch arrays instead."
        )

    n_hists, width, height = A.shape

    if weights is None:
        weights = nx.ones((n_hists,), type_as=A) / n_hists
    else:
        assert (len(weights) == n_hists)

    if log:
        log = {'err': []}

    err = 1
    # build the convolution operator
    # this is equivalent to blurring on horizontal then vertical directions
    t = nx.linspace(0, 1, width, type_as=A)
    [Y, X] = nx.meshgrid(t, t)
    M1 = - (X - Y) ** 2 / reg

    t = nx.linspace(0, 1, height, type_as=A)
    [Y, X] = nx.meshgrid(t, t)
    M2 = - (X - Y) ** 2 / reg

    def convol_img(log_img):
        log_img = nx.logsumexp(M1[:, :, None] + log_img[None], axis=1)
        log_img = nx.logsumexp(M2[:, :, None] + log_img.T[None], axis=1).T
        return log_img

    logA = nx.log(A + stabThr)
    log_KU, G, F = nx.zeros((3, *logA.shape), type_as=A)
    err = 1
    for ii in range(numItermax):
        log_bar = nx.zeros((width, height), type_as=A)
        for k in range(n_hists):
            f = logA[k] - convol_img(G[k])
            log_KU[k] = convol_img(f)
            log_bar = log_bar + weights[k] * log_KU[k]

        if ii % 10 == 9:
            err = nx.exp(G + log_KU).std(axis=0).sum()
            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if ii % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
            if err < stopThr:
                break
        G = log_bar[None, :, :] - log_KU

    else:
        if warn:
            warnings.warn("Convolutional Sinkhorn did not converge. "
                          "Try a larger number of iterations `numItermax` "
                          "or a larger entropy `reg`.")
    if log:
        log['niter'] = ii
        return nx.exp(log_bar), log
    else:
        return nx.exp(log_bar)


def convolutional_barycenter2d_debiased(A, reg, weights=None, method="sinkhorn",
                                        numItermax=10000, stopThr=1e-3,
                                        verbose=False, log=False, warn=True,
                                        **kwargs):
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
        Weights of each image on the simplex (barycentric coodinates)
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

    if method.lower() == 'sinkhorn':
        return _convolutional_barycenter2d_debiased(A, reg, weights=weights,
                                                    numItermax=numItermax,
                                                    stopThr=stopThr, verbose=verbose,
                                                    log=log, warn=warn,
                                                    **kwargs)
    elif method.lower() == 'sinkhorn_log':
        return _convolutional_barycenter2d_debiased_log(A, reg, weights=weights,
                                                        numItermax=numItermax,
                                                        stopThr=stopThr, verbose=verbose,
                                                        log=log, warn=warn,
                                                        **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def _convolutional_barycenter2d_debiased(A, reg, weights=None, numItermax=10000,
                                         stopThr=1e-3, stabThr=1e-15, verbose=False,
                                         log=False, warn=True):
    r"""Compute the debiased barycenter of 2D images via sinkhorn convolutions.
    """

    A = list_to_array(A)
    n_hists, width, height = A.shape

    nx = get_backend(A)

    if weights is None:
        weights = nx.ones((n_hists,), type_as=A) / n_hists
    else:
        assert (len(weights) == n_hists)

    if log:
        log = {'err': []}

    bar = nx.ones((width, height), type_as=A)
    bar /= width * height
    U = nx.ones(A.shape, type_as=A)
    V = nx.ones(A.shape, type_as=A)
    c = nx.ones(A.shape[1:], type_as=A)
    err = 1

    # build the convolution operator
    # this is equivalent to blurring on horizontal then vertical directions
    t = nx.linspace(0, 1, width, type_as=A)
    [Y, X] = nx.meshgrid(t, t)
    K1 = nx.exp(-(X - Y) ** 2 / reg)

    t = nx.linspace(0, 1, height, type_as=A)
    [Y, X] = nx.meshgrid(t, t)
    K2 = nx.exp(-(X - Y) ** 2 / reg)

    def convol_imgs(imgs):
        kx = nx.einsum("...ij,kjl->kil", K1, imgs)
        kxy = nx.einsum("...ij,klj->kli", K2, kx)
        return kxy

    KU = convol_imgs(U)
    for ii in range(numItermax):
        V = bar[None] / KU
        KV = convol_imgs(V)
        U = A / KV
        KU = convol_imgs(U)
        bar = c * nx.exp(
            nx.sum(weights[:, None, None] * nx.log(KU + stabThr), axis=0)
        )

        for _ in range(10):
            c = (c * bar / nx.squeeze(convol_imgs(c[None]))) ** 0.5

        if ii % 10 == 9:
            err = nx.sum(nx.std(V * KU, axis=0))
            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if ii % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))

            # debiased Sinkhorn does not converge monotonically
            # guarantee a few iterations are done before stopping
            if err < stopThr and ii > 20:
                break
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        log['niter'] = ii
        log['U'] = U
        return bar, log
    else:
        return bar


def _convolutional_barycenter2d_debiased_log(A, reg, weights=None, numItermax=10000,
                                             stopThr=1e-3, stabThr=1e-30, verbose=False,
                                             log=False, warn=True):
    r"""Compute the debiased barycenter of 2D images in log-domain.
     """

    A = list_to_array(A)
    n_hists, width, height = A.shape
    nx = get_backend(A)
    if nx.__name__ in ("jax", "tf"):
        raise NotImplementedError(
            "Log-domain functions are not yet implemented"
            " for Jax and TF. Use numpy or torch arrays instead."
        )
    if weights is None:
        weights = nx.ones((n_hists,), type_as=A) / n_hists
    else:
        assert (len(weights) == A.shape[0])

    if log:
        log = {'err': []}

    err = 1
    # build the convolution operator
    # this is equivalent to blurring on horizontal then vertical directions
    t = nx.linspace(0, 1, width, type_as=A)
    [Y, X] = nx.meshgrid(t, t)
    M1 = - (X - Y) ** 2 / reg

    t = nx.linspace(0, 1, height, type_as=A)
    [Y, X] = nx.meshgrid(t, t)
    M2 = - (X - Y) ** 2 / reg

    def convol_img(log_img):
        log_img = nx.logsumexp(M1[:, :, None] + log_img[None], axis=1)
        log_img = nx.logsumexp(M2[:, :, None] + log_img.T[None], axis=1).T
        return log_img

    logA = nx.log(A + stabThr)
    log_bar, c = nx.zeros((2, width, height), type_as=A)
    log_KU, G, F = nx.zeros((3, *logA.shape), type_as=A)
    err = 1
    for ii in range(numItermax):
        log_bar = nx.zeros((width, height), type_as=A)
        for k in range(n_hists):
            f = logA[k] - convol_img(G[k])
            log_KU[k] = convol_img(f)
            log_bar = log_bar + weights[k] * log_KU[k]
        log_bar += c
        for _ in range(10):
            c = 0.5 * (c + log_bar - convol_img(c))

        if ii % 10 == 9:
            err = nx.sum(nx.std(nx.exp(G + log_KU), axis=0))
            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if ii % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
            if err < stopThr and ii > 20:
                break
        G = log_bar[None, :, :] - log_KU

    else:
        if warn:
            warnings.warn("Convolutional Sinkhorn did not converge. "
                          "Try a larger number of iterations `numItermax` "
                          "or a larger entropy `reg`.")
    if log:
        log['niter'] = ii
        return nx.exp(log_bar), log
    else:
        return nx.exp(log_bar)
