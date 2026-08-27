# -*- coding: utf-8 -*-
"""
Useful functions for solvers for the (balanced) sliced transport problem.
"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty   <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@polytechnique.edu>
#         Eloi Tanguy <eloi.tanguy@math.cnrs.fr>
#         Laetitia Chapel <laetitia.chapel@irisa.fr>
#         Clément Bonet <clement.bonet.mapp@polytechnique.edu>
#
# License: MIT License

import numpy as np
from ..backend import get_backend, NumpyBackend
from ..utils import get_coordinate_circle


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
    type_as: type, optional
        Type of the returned array

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

    if isinstance(seed, np.random.RandomState) and str(nx) == "numpy":
        projections = seed.randn(d, n_projections)
    else:
        if seed is not None:
            nx.seed(seed)
        projections = nx.randn(d, n_projections, type_as=type_as)

    projections = projections / nx.sqrt(nx.sum(projections**2, 0, keepdims=True))
    return projections


def get_projections_sphere(d, n_projections, seed=None, backend=None, type_as=None):
    r"""
    Generates n_projections samples from the uniform distribution on the Stiefel manifold of dimension :math:`d\times 2`: :math:`\mathbb{V}_{d,2}=\{X \in \mathbb{R}^{d\times 2}, X^TX=I_2\}`

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
    type_as: optional
        Type to use for random generation

    Returns
    -------
    out: ndarray, shape (n_projections, d, 2)

    Examples
    --------
    >>> n_projections = 100
    >>> d = 5
    >>> projs = get_projections_sphere(d, n_projections)
    >>> np.allclose(np.einsum("nij, nik -> njk", projs, projs), np.eye(2))  # doctest: +NORMALIZE_WHITESPACE
    True
    """
    if backend is None:
        nx = NumpyBackend()
    else:
        nx = backend

    if isinstance(seed, np.random.RandomState) and str(nx) == "numpy":
        Z = seed.randn(n_projections, d, 2)
    else:
        if seed is not None:
            nx.seed(seed)
        Z = nx.randn(n_projections, d, 2, type_as=type_as)

    projections, _ = nx.qr(Z)
    return projections


def get_random_rotations(d, n_rotations, seed=None, backend=None, type_as=None):
    r"""
    Generates n_rotations samples from the uniform (Haar) distribution on the special orthogonal group :math:`\mathrm{SO}(d)=\{R \in \mathbb{R}^{d\times d}, R^TR=I_d, \mathrm{det}(R)=1\}`.

    The rotations are obtained from the QR factorization of Gaussian matrices,
    with the sign correction of :ref:`[94] <references-get-random-rotations>`
    and a sign flip of the first column of the matrices with negative
    determinant.

    Parameters
    ----------
    d : int
        dimension of the space
    n_rotations : int
        number of samples requested
    seed: int or RandomState, optional
        Seed used for numpy random number generator
    backend:
        Backend to use for random generation
    type_as: optional
        Type to use for random generation

    Returns
    -------
    out: ndarray, shape (n_rotations, d, d)

    Examples
    --------
    >>> n_rotations = 100
    >>> d = 5
    >>> rotations = get_random_rotations(d, n_rotations)
    >>> np.allclose(np.einsum("nij, nkj -> nik", rotations, rotations), np.eye(d))  # doctest: +NORMALIZE_WHITESPACE
    True
    >>> np.allclose(np.linalg.det(rotations), 1.)  # doctest: +NORMALIZE_WHITESPACE
    True


    .. _references-get-random-rotations:
    References
    ----------
    .. [94] Mezzadri, F. (2007). How to generate random matrices from the
       classical compact groups. Notices of the American Mathematical
       Society, 54(5), 592-604.
    """
    if backend is None:
        nx = NumpyBackend()
    else:
        nx = backend

    if isinstance(seed, np.random.RandomState) and str(nx) == "numpy":
        Z = seed.randn(n_rotations, d, d)
    else:
        if seed is not None:
            nx.seed(seed)
        Z = nx.randn(n_rotations, d, d, type_as=type_as)

    Q, R = nx.qr(Z)
    diagonal = nx.sum(R * nx.eye(d, type_as=R)[None, :, :], axis=-1)
    Q = Q * nx.sign(diagonal)[:, None, :]
    flip = nx.sign(nx.det(Q))
    rotations = nx.concatenate(
        (Q[:, :, :1] * flip[:, None, None], Q[:, :, 1:]), axis=-1
    )
    return rotations


def projection_sphere_to_circle(
    x, n_projections=50, projections=None, seed=None, backend=None
):
    r"""
    Projection of :math:`x\in S^{d-1}` on circles using coordinates on [0,1[.

    To get the projection on the circle, we use the following formula:

    .. math::
        P^U(x) = \frac{U^Tx}{\|U^Tx\|_2}

    where :math:`U` is a random matrix sampled from the uniform distribution on the Stiefel manifold of dimension :math:`d\times 2`: :math:`\mathbb{V}_{d,2}=\{X \in \mathbb{R}^{d\times 2}, X^TX=I_2\}`
    and :math:`x` is a point on the sphere. Then, we apply the function get_coordinate_circle to get the coordinates on :math:`[0,1[`.

    Parameters
    ----------
    x : ndarray, shape (n_samples, dim)
        samples on the sphere
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    projections: shape (n_projections, dim, 2), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    backend:
        Backend to use for random generation

    Returns
    -------
    Xp_coords: ndarray, shape (n_projections, n_samples)
        Coordinates of the projections on the circle
    """
    if backend is None:
        nx = get_backend(x)
    else:
        nx = backend

    n, d = x.shape

    if projections is None:
        projections = get_projections_sphere(
            d, n_projections, seed=seed, backend=nx, type_as=x
        )

    # Projection on S^1
    # Projection on plane
    Xp = nx.einsum("ikj, lk -> ilj", projections, x)

    # Projection on sphere
    Xp = Xp / nx.sqrt(nx.sum(Xp**2, -1, keepdims=True))

    # Get coordinates on [0,1[
    Xp_coords = nx.reshape(
        get_coordinate_circle(nx.reshape(Xp, (-1, 2))), (n_projections, n)
    )

    return Xp_coords, projections


def projection_sphere_to_ball(x, eps=1e-6, backend=None):
    r"""
    Projection of :math:`x\in S^{d-1}` on the unit ball of :math:`\mathbb{R}^{d-1}` with :math:`\frac{1}{\pi}h_1\circ\phi_\epsilon`.

    To get the projection on the ball, we use the following closed form:

    .. math::
        \frac{1}{\pi}(h_1\circ\phi_\epsilon)(x) = \frac{1}{\pi}\mathrm{arccos}\left(-\frac{3+5x_d}{5+3x_d}\right)\frac{x_{1:d-1}}{\|x_{1:d-1}\|}

    where :math:`\phi_\epsilon` is the stereographic projection
    :math:`\phi(x) = \frac{2 x_{1:d-1}}{1-x_d}` restricted to the sphere without the
    :math:`\epsilon`-cap around the north pole (points with :math:`x_d > 1-\epsilon`
    are first mapped to the circle :math:`x_d = 1-\epsilon`), and
    :math:`h_1(x) = \mathrm{arccos}\left(\frac{1-\|x\|^2}{1+\|x\|^2}\right)\frac{x}{\|x\|}`
    is the injective defining function of :ref:`[93] <references-s3w>`.

    Parameters
    ----------
    x : ndarray, shape (..., dim)
        samples on the sphere
    eps: float, optional (default=1e-6)
        Size of the cap around the north pole excluded from the stereographic
        projection to ensure numerical stability
    backend:
        Backend to use for the computations

    Returns
    -------
    Xp: ndarray, shape (..., dim-1)
        Images of the samples in the unit ball
    """
    if backend is None:
        nx = get_backend(x)
    else:
        nx = backend

    x_d = nx.clip(x[..., -1:], -1.0, 1.0 - eps)
    x_azimuth = x[..., :-1]
    norm2 = nx.sum(x_azimuth**2, axis=-1, keepdims=True)
    # the azimuth of the poles is arbitrary, fix it for reproducibility
    x_azimuth = nx.where(
        norm2 > 0, x_azimuth, nx.ones(x_azimuth.shape, type_as=x_azimuth)
    )
    norm2 = nx.sum(x_azimuth**2, axis=-1, keepdims=True)
    radius = nx.arccos(-(3.0 + 5.0 * x_d) / (5.0 + 3.0 * x_d)) / np.pi
    return radius * x_azimuth / nx.sqrt(norm2)
