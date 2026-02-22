# -*- coding: utf-8 -*-
"""
Spectral-Grassmann optimal transport for linear operators.

This module implements the Spectral-Grassmann Wasserstein framework for
comparing dynamical systems via their learned operator representations.

It provides tools to extract spectral "atoms" (eigenvalues and associated
eigenspaces) from linear operators and to compute an optimal transport metric
that combines a spectral term on eigenvalues with a Grassmannian term on
eigenspaces.
"""

# Author: Sienna O'Shea  <osheasienna@gmail.com>
#         Thibaut Germain<thibaut.germain.pro@gmail.com>
# License: MIT License

import ot
from ot.backend import get_backend

#####################################################################################################################################
#####################################################################################################################################
### NORMALISATION AND OPERATOR ATOMS  ###
#####################################################################################################################################
#####################################################################################################################################


def eigenvalue_cost_matrix(
    Ds, Dt, q=1, real_scale: float = 1.0, imag_scale: float = 1.0, nx=None
):
    """Compute pairwise eigenvalue distances for source and target domains.

    Parameters
    ----------
    Ds: array-like, shape (n_s,)
        Source eigenvalues.
    Dt: array-like, shape (n_t,)
        Target eigenvalues.
    real_scale: float, optional
        Scale factor for real parts, default 1.0.
    imag_scale: float, optional
        Scale factor for imaginary parts, default 1.0.

    Returns
    ----------
    C: np.ndarray, shape (n_s, n_t)
        Eigenvalue cost matrix.
    """
    if nx is None:
        nx = get_backend(Ds, Dt)

    Dsn = nx.real(Ds) * real_scale + 1j * nx.imag(Ds) * imag_scale
    Dtn = nx.real(Dt) * real_scale + 1j * nx.imag(Dt) * imag_scale
    prod = Dsn[:, None] - Dtn[None, :]
    prod = nx.real(prod * nx.conj(prod))
    return prod ** (q / 2)


def ot_plan(C, Ws=None, Wt=None, nx=None):
    """Compute the optimal transport plan for a given cost matrix and marginals.

    Parameters
    ----------
    C: array-like, shape (n, m)
        Cost matrix.
    Ws: array-like, shape (n,), optional
        Source distribution. If None, uses a uniform distribution.
    Wt: array-like, shape (m,), optional
        Target distribution. If None, uses a uniform distribution.

    Returns
    ----------
    P: np.ndarray, shape (n, m)
        Optimal transport plan.
    """
    if nx is None:
        nx = get_backend(C)

    n, m = C.shape

    if Ws is None:
        Ws = nx.ones((n,), type_as=C) / float(n)

    if Wt is None:
        Wt = nx.ones((m,), type_as=C) / float(m)

    Ws = Ws / nx.sum(Ws)
    Wt = Wt / nx.sum(Wt)

    C_real = nx.real(C)

    return ot.emd(Ws, Wt, C_real)


def _normalize_columns(A, nx, eps=1e-12):
    """Normalize the columns of an array with a backend-aware norm.

    Parameters
    ----------
    A: array-like, shape (d, n)
        Input array whose columns are normalized.
    nx: module
        Backend (NumPy-compatible) used for math operations.
    eps: float, optional
        Minimum norm value to avoid division by zero, default 1e-12.

    Returns
    ----------
    A_norm: array-like, shape (d, n)
        Column-normalized array.
    """
    nrm = nx.sqrt(nx.sum(A * nx.conj(A), axis=0, keepdims=True))
    nrm = nx.real(nrm)  # norm is real; avoid complex dtype for maximum (e.g. torch)
    nrm = nx.maximum(nrm, eps)
    return A / nrm


def _delta_matrix_1d(Rs, Ls, Rt, Lt, nx=None, eps=1e-12):
    """Compute the normalized inner-product delta matrix for eigenspaces.

    Parameters
    ----------
    Rs: array-like, shape (L, n_s)
        Source right eigenvectors.
    Ls: array-like, shape (L, n_s)
        Source left eigenvectors.
    Rt: array-like, shape (L, n_t)
        Target right eigenvectors.
    Lt: array-like, shape (L, n_t)
        Target left eigenvectors.
    nx: module, optional
        Backend (NumPy-compatible). If None, inferred from inputs.
    eps: float, optional
        Minimum norm value used in normalization, default 1e-12.

    Returns
    ----------
    delta: array-like, shape (n_s, n_t)
        Delta matrix with entries in [0, 1].
    """
    if nx is None:
        nx = get_backend(Rs, Ls, Rt, Lt)

    Rsn = _normalize_columns(Rs, nx=nx, eps=eps)
    Lsn = _normalize_columns(Ls, nx=nx, eps=eps)
    Rtn = _normalize_columns(Rt, nx=nx, eps=eps)
    Ltn = _normalize_columns(Lt, nx=nx, eps=eps)

    Cr = nx.dot(nx.conj(Rsn).T, Rtn)
    Cl = nx.dot(nx.conj(Lsn).T, Ltn)

    delta = nx.abs(Cr * Cl)
    delta = nx.clip(delta, 0.0, 1.0)
    return delta


def _grassmann_distance_squared(delta, grassman_metric="chordal", nx=None, eps=1e-300):
    """Compute squared Grassmannian distances from delta similarities.

    Parameters
    ----------
    delta: array-like
        Similarity values in [0, 1].
    grassman_metric: str, optional
        Metric type: "geodesic", "chordal", "procrustes", or "martin".
    nx: module, optional
        Backend (NumPy-compatible). If None, inferred from inputs.
    eps: float, optional
        Minimum value used for numerical stability, default 1e-300.

    Returns
    ----------
    dist2: array-like
        Squared Grassmannian distance(s).
    """
    if nx is None:
        nx = get_backend(delta)

    delta = nx.clip(delta, 0.0, 1.0)

    if grassman_metric == "geodesic":
        return nx.arccos(delta) ** 2
    if grassman_metric == "chordal":
        return 1.0 - delta**2
    if grassman_metric == "procrustes":
        return 2.0 * (1.0 - delta)
    if grassman_metric == "martin":
        return -nx.log(nx.clip(delta**2, eps, 1e300))
    raise ValueError(f"Unknown grassman_metric: {grassman_metric}")


#####################################################################################################################################
#####################################################################################################################################
### SPECTRAL-GRASSMANNIAN WASSERSTEIN METRIC ###
#####################################################################################################################################
#####################################################################################################################################
def cost(
    Ds,
    Rs,
    Ls,
    Dt,
    Rt,
    Lt,
    eta=0.5,
    p=2,
    q=1,
    grassman_metric="chordal",
    real_scale=1.0,
    imag_scale=1.0,
    nx=None,
):
    """Compute the SGOT cost matrix between two spectral decompositions.

    Parameters
    ----------
    Ds: array-like, shape (n_s,)
        Eigenvalues of operator T1.
    Rs: array-like, shape (L, n_s)
        Right eigenvectors of operator T1.
    Ls: array-like, shape (L, n_s)
        Left eigenvectors of operator T1.
    Dt: array-like, shape (n_t,)
        Eigenvalues of operator T2.
    Rt: array-like, shape (L, n_t)
        Right eigenvectors of operator T2.
    Lt: array-like, shape (L, n_t)
        Left eigenvectors of operator T2.
    eta: float, optional
        Weighting between spectral and Grassmann terms, default 0.5.
    p: int, optional
        Power for the OT cost, default 2.
    grassman_metric: str, optional
        Metric type: "geodesic", "chordal", "procrustes", or "martin".
    real_scale: float, optional
        Scale factor for real parts, default 1.0.
    imag_scale: float, optional
        Scale factor for imaginary parts, default 1.0.

    Returns
    ----------
    C: array-like, shape (n_s, n_t)
        SGOT cost matrix.
    """
    if nx is None:
        nx = get_backend(Ds, Rs, Ls, Dt, Rt, Lt)

    if Ds.ndim != 1:
        raise ValueError(f"cost() expects Ds to be 1D (n,), got shape {Ds.shape}")
    lam1 = Ds

    if Dt.ndim != 1:
        raise ValueError(f"cost() expects Dt to be 1D (n,), got shape {Dt.shape}")
    lam2 = Dt

    lam1 = nx.astype(lam1, "complex128")
    lam2 = nx.astype(lam2, "complex128")

    if Rs.shape != Ls.shape:
        raise ValueError(
            f"Rs and Ls must have the same shape, got {Rs.shape} and {Ls.shape}"
        )

    if Rt.shape != Lt.shape:
        raise ValueError(
            f"Rt and Lt must have the same shape, got {Rt.shape} and {Lt.shape}"
        )

    if Rs.shape[1] != lam1.shape[0]:
        raise ValueError(
            f"Number of source eigenvectors ({Rs.shape[1]}) must match "
            f"number of source eigenvalues ({lam1.shape[0]})"
        )

    if Rt.shape[1] != lam2.shape[0]:
        raise ValueError(
            f"Number of target eigenvectors ({Rt.shape[1]}) must match "
            f"number of target eigenvalues ({lam2.shape[0]})"
        )

    C_lambda = eigenvalue_cost_matrix(
        lam1, lam2, q=q, real_scale=real_scale, imag_scale=imag_scale, nx=nx
    )

    delta = _delta_matrix_1d(Rs, Ls, Rt, Lt, nx=nx)
    C_grass = _grassmann_distance_squared(delta, grassman_metric=grassman_metric, nx=nx)

    C2 = eta * C_lambda + (1.0 - eta) * C_grass
    C = C2 ** (p / 2.0)

    return C


def metric(
    Ds,
    Rs,
    Ls,
    Dt,
    Rt,
    Lt,
    eta=0.5,
    p=2,
    q=1,
    r=2,
    grassman_metric="chordal",
    real_scale=1.0,
    imag_scale=1.0,
    Ws=None,
    Wt=None,
    nx=None,
):
    """Compute the SGOT metric between two spectral decompositions.

    Parameters
    ----------
    Ds: array-like, shape (n_s,)
        Eigenvalues of operator T1.
    Rs: array-like, shape (L, n_s)
        Right eigenvectors of operator T1.
    Ls: array-like, shape (L, n_s)
        Left eigenvectors of operator T1.
    Dt: array-like, shape (n_t,)
        Eigenvalues of operator T2.
    Rt: array-like, shape (L, n_t)
        Right eigenvectors of operator T2.
    Lt: array-like, shape (L, n_t)
        Left eigenvectors of operator T2.
    eta: float, optional
        Weighting between spectral and Grassmann terms, default 0.5.
    p: int, optional
        Exponent defining the OT ground cost and Wasserstein order. The cost matrix is raised to the power p/2 and the OT objective
        is raised to the power 1/p. Default is 2.
    q: int, optional
        Exponent applied to the eigenvalue distance in the spectral term. Controls the geometry of the eigenvalue cost matrix.
        Default is 1.
    r: int, optional
        Outer root applied to the Wasserstein objective. Default is 2.
    grassman_metric: str, optional
        Metric type: "geodesic", "chordal", "procrustes", or "martin".
    real_scale: float, optional
        Scale factor for real parts, default 1.0.
    imag_scale: float, optional
        Scale factor for imaginary parts, default 1.0.
    Ws: array-like, shape (n_s,), optional
        Source distribution. If None, uses a uniform distribution.
    Wt: array-like, shape (n_t,), optional
        Target distribution. If None, uses a uniform distribution.

    Returns
    ----------
    dist: float
        SGOT metric value.
    """
    if nx is None:
        nx = get_backend(Ds, Rs, Ls, Dt, Rt, Lt)

    if Ds.ndim != 1:
        raise ValueError(f"metric() expects Ds to be 1D (n,), got shape {Ds.shape}")
    if Dt.ndim != 1:
        raise ValueError(f"metric() expects Dt to be 1D (n,), got shape {Dt.shape}")

    C = cost(
        Ds,
        Rs,
        Ls,
        Dt,
        Rt,
        Lt,
        eta=eta,
        p=p,
        q=q,
        grassman_metric=grassman_metric,
        real_scale=real_scale,
        imag_scale=imag_scale,
        nx=nx,
    )

    P = ot_plan(C, Ws=Ws, Wt=Wt, nx=nx)
    obj = float(nx.sum(C * P) ** (1.0 / p))
    return float(obj) ** (1.0 / float(r))
