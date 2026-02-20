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

import numpy as np
import ot
from ot.backend import get_backend

###
# Settings : (Ds,Rs,Ls) if primal, (Ds,Xs,prs,pls) if dual
###

#####################################################################################################################################
#####################################################################################################################################
### OT METRIC ###
#####################################################################################################################################
#####################################################################################################################################


def principal_grassman_matrix(Ps, Pt, eps: float = 1e-12, nx=None):
    """Compute the unitary Grassmann matrix for source and target domains.

    Parameters
    ----------
    Ps : array-like, shape (l, n_ds)
        Source domain data, with columns spanning the source subspace.
    Pt : array-like, shape (l, n_dt)
        Target domain data, with columns spanning the target subspace.
    eps : float, optional
        Minimum column norm used to avoid division by zero. Default is 1e-12.

    Returns
    -------
    C : np.ndarray, shape (n_ds, n_dt)
        Grassmann matrix between source and target subspaces.
    """
    if nx is None:
        nx = get_backend(Ps, Pt)

    Ps = nx.asarray(Ps)
    Pt = nx.asarray(Pt)

    ns = nx.sqrt(nx.sum(Ps * nx.conj(Ps), axis=0, keepdims=True))
    nt = nx.sqrt(nx.sum(Pt * nx.conj(Pt), axis=0, keepdims=True))

    ns = nx.clip(ns, eps, 1e300)
    nt = nx.clip(nt, eps, 1e300)

    Psn = Ps / ns
    Ptn = Pt / nt

    return nx.dot(nx.conj(Psn).T, Ptn)


def eigenvector_chordal_cost_matrix(Rs, Ls, Rt, Lt, nx=None):
    """Compute pairwise Grassmann matrices for source and target domains.

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

    Returns
    ----------
    C: np.ndarray, shape (n_s, n_t)
        Eigenvector chordal cost matrix.
    """
    if nx is None:
        nx = get_backend(Rs, Ls, Rt, Lt)

    Cr = principal_grassman_matrix(Rs, Rt, nx=nx)
    Cl = principal_grassman_matrix(Ls, Lt, nx=nx)

    prod = nx.real(Cr * Cl)
    prod = nx.clip(prod, 0.0, 1.0)
    return nx.sqrt(1.0 - prod)


def eigenvalue_cost_matrix(
    Ds, Dt, real_scale: float = 1.0, imag_scale: float = 1.0, nx=None
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

    Ds = nx.asarray(Ds)
    Dt = nx.asarray(Dt)
    Dsn = nx.real(Ds) * real_scale + 1j * nx.imag(Ds) * imag_scale
    Dtn = nx.real(Dt) * real_scale + 1j * nx.imag(Dt) * imag_scale
    return nx.abs(Dsn[:, None] - Dtn[None, :])


def chordal_cost_matrix(
    Ds, Rs, Ls, Dt, Rt, Lt, real_scale=1.0, imag_scale=1.0, alpha=0.5, p=2, nx=None
):
    """Compute the chordal cost matrix between source and target spectral decompositions.

    Parameters
    ----------
    Ds: array-like, shape (n_s,)
        Source eigenvalues.
    Rs: array-like, shape (L, n_s)
        Source right eigenvectors.
    Ls: array-like, shape (L, n_s)
        Source left eigenvectors.
    Dt: array-like, shape (n_t,)
        Target eigenvalues.
    Rt: array-like, shape (L, n_t)
        Target right eigenvectors.
    Lt: array-like, shape (L, n_t)
        Target left eigenvectors.
    real_scale: float, optional
        Scale factor for real parts, default 1.0.
    imag_scale: float, optional
        Scale factor for imaginary parts, default 1.0.
    alpha: float, optional
        Weighting factor for the eigenvalue cost, default 0.5.
    p: int, optional
        Power for the chordal distance, default 2.

    Returns
    ----------
    C: np.ndarray, shape (n_s, n_t)
        Chordal cost matrix.
    """
    if nx is None:
        nx = get_backend(Ds, Rs, Ls, Dt, Rt, Lt)
    CD = eigenvalue_cost_matrix(
        Ds, Dt, real_scale=real_scale, imag_scale=imag_scale, nx=nx
    )
    CC = eigenvector_chordal_cost_matrix(Rs, Ls, Rt, Lt, nx=nx)
    C = alpha * CD + (1.0 - alpha) * CC
    return C**p


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

    C = nx.asarray(C)
    n, m = C.shape

    if Ws is None:
        Ws = nx.ones((n,), dtype=C.dtype) / float(n)
    else:
        Ws = nx.asarray(Ws)

    if Wt is None:
        Wt = nx.ones((m,), dtype=C.dtype) / float(m)
    else:
        Wt = nx.asarray(Wt)

    Ws = Ws / nx.sum(Ws)
    Wt = Wt / nx.sum(Wt)

    C_real = nx.real(C)

    C_np = ot.backend.to_numpy(C_real)
    Ws_np = ot.backend.to_numpy(Ws)
    Wt_np = ot.backend.to_numpy(Wt)

    return ot.emd(Ws_np, Wt_np, C_np)


def ot_score(C, P, p: int = 2, nx=None):
    """Compute the OT score (distance) given a cost matrix and a transport plan.

    Parameters
    ----------
    C: array-like, shape (n, m)
        Cost matrix.
    P: array-like, shape (n, m)
        Transport plan.
    p: int, optional
        Power for the OT score, default 2.

    Returns
    ----------
    dist: float
        OT score (distance).
    """
    if nx is None:
        nx = get_backend(C)
    C = nx.asarray(C)
    P = nx.asarray(P)
    return float(nx.sum(C * P) ** (1.0 / p))


def chordal_metric(
    Ds,
    Rs,
    Ls,
    Dt,
    Rt,
    Lt,
    real_scale: float = 1.0,
    imag_scale: float = 1.0,
    alpha: float = 0.5,
    p: int = 2,
    nx=None,
):
    """Compute the chordal OT metric between two spectral decompositions.

    Parameters
    ----------
    Ds: array-like, shape (n_s,)
        Source eigenvalues.
    Rs: array-like, shape (L, n_s)
        Source right eigenvectors.
    Ls: array-like, shape (L, n_s)
        Source left eigenvectors.
    Dt: array-like, shape (n_t,)
        Target eigenvalues.
    Rt: array-like, shape (L, n_t)
        Target right eigenvectors.
    Lt: array-like, shape (L, n_t)
        Target left eigenvectors.
    real_scale: float, optional
        Scale factor for real parts, default 1.0.
    imag_scale: float, optional
        Scale factor for imaginary parts, default 1.0.
    alpha: float, optional
        Weighting factor for the eigenvalue cost, default 0.5.
    p: int, optional
        Power for the chordal distance, default 2.

    Returns
    ----------
    dist: float
        Chordal OT metric value.
    """
    if nx is None:
        nx = get_backend(Ds, Rs, Ls, Dt, Rt, Lt)

    C = chordal_cost_matrix(
        Ds,
        Rs,
        Ls,
        Dt,
        Rt,
        Lt,
        real_scale=real_scale,
        imag_scale=imag_scale,
        alpha=alpha,
        p=p,
        nx=nx,
    )
    P = ot_plan(C, nx=nx)
    return ot_score(C, P, p=p, nx=nx)


#####################################################################################################################################
#####################################################################################################################################
### NORMALISATION AND OPERATOR ATOMS  ###
#####################################################################################################################################
#####################################################################################################################################


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
    nrm = nx.clip(nrm, eps, 1e300)
    return A / nrm


def _delta_matrix_1d_hs(Rs, Ls, Rt, Lt, nx=None, eps=1e-12):
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

    Rs = nx.asarray(Rs)
    Ls = nx.asarray(Ls)
    Rt = nx.asarray(Rt)
    Lt = nx.asarray(Lt)

    Rsn = _normalize_columns(Rs, nx=nx, eps=eps)
    Lsn = _normalize_columns(Ls, nx=nx, eps=eps)
    Rtn = _normalize_columns(Rt, nx=nx, eps=eps)
    Ltn = _normalize_columns(Lt, nx=nx, eps=eps)

    Cr = nx.dot(nx.conj(Rsn).T, Rtn)
    Cl = nx.dot(nx.conj(Lsn).T, Ltn)

    delta = nx.abs(Cr * Cl)
    delta = nx.clip(delta, 0.0, 1.0)
    return delta


def _atoms_from_operator(T, r=None, sort_mode="closest_to_1"):
    """Extract dua; eigen-atoms from a square operator.

    Parameters
    ----------
    T: array-like, shape (d, d)
        Input linear operator.
    r: int, optional
        Number of modes to keep. If None, keep all modes.
    sort_mode: str, optional
        Eigenvalue sorting mode: "closest_to_1", "closest_to_0", or "largest_mag".

    Returns
    ----------
    D: np.ndarray, shape (r,)
        Selected eigenvalues.
    R: np.ndarray, shape (d, r)
        Corresponding right eigenvectors.
    L: np.ndarray, shape (d, r)
        Dual left eigenvectors.
    """
    nx = get_backend(T)
    T = nx.asarray(T)

    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError(f"T must be a square 2D array; got shape {T.shape}")

    d = int(T.shape[0])
    if r is None:
        r = d
    r = int(r)
    if not (1 <= r <= d):
        raise ValueError(f"r must be an integer in [1, {d}], got r={r}")

    T_np = ot.backend.to_numpy(T)
    evals_np, evecs_np = np.linalg.eig(T_np)

    if sort_mode == "closest_to_1":
        order = np.argsort(np.abs(evals_np - 1.0))
    elif sort_mode == "closest_to_0":
        order = np.argsort(np.abs(evals_np))
    elif sort_mode == "largest_mag":
        order = np.argsort(-np.abs(evals_np))
    else:
        raise ValueError(
            "sort_mode must be one of 'closest_to_1', 'closest_to_0', or 'largest_mag'"
        )

    idx = order[:r]
    D_np = evals_np[idx]
    R_np = evecs_np[:, idx]

    evalsL_np, evecsL_np = np.linalg.eig(T_np.conj().T)

    L_np = np.zeros((d, r), dtype=np.complex128)
    used = set()

    for i, lam in enumerate(D_np):
        targets = np.abs(evalsL_np - np.conj(lam))
        for j in np.argsort(targets):
            if j not in used:
                used.add(j)
                L_np[:, i] = evecsL_np[:, j]
                break

    if hasattr(nx, "from_numpy"):
        D = nx.from_numpy(D_np, type_as=T)
        R = nx.from_numpy(R_np, type_as=T)
        L = nx.from_numpy(L_np, type_as=T)
    else:
        D = nx.asarray(D_np)
        R = nx.asarray(R_np)
        L = nx.asarray(L_np)

    G = nx.dot(nx.conj(L).T, R)

    G_np = ot.backend.to_numpy(G)
    if np.linalg.matrix_rank(G_np) < r:
        raise ValueError("Dual normalization failed: L^* R is singular.")

    invG_H_np = np.linalg.inv(G_np).conj().T
    if hasattr(nx, "from_numpy"):
        invG_H = nx.from_numpy(invG_H_np, type_as=T)
    else:
        invG_H = nx.asarray(invG_H_np)

    L = nx.dot(L, invG_H)

    return D, R, L


#####################################################################################################################################
#####################################################################################################################################
### GRASSMANNIAN METRIC ###
#####################################################################################################################################
#####################################################################################################################################


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

    delta = nx.asarray(delta)
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
    grassman_metric="chordal",
    real_scale=1.0,
    imag_scale=1.0,
    nx=None,
):
    """Compute the SGOT cost matrix between two spectral decompositions.

    Parameters
    ----------
    Ds: array-like, shape (n_s,) or (n_s, n_s)
        Eigenvalues of operator T1 (or diagonal matrix).
    Rs: array-like, shape (L, n_s)
        Right eigenvectors of operator T1.
    Ls: array-like, shape (L, n_s)
        Left eigenvectors of operator T1.
    Dt: array-like, shape (n_t,) or (n_t, n_t)
        Eigenvalues of operator T2 (or diagonal matrix).
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

    Ds = nx.asarray(Ds)
    Dt = nx.asarray(Dt)
    if len(Ds.shape) == 2:
        lam1 = nx.diag(Ds)
    else:
        lam1 = Ds.reshape((-1,))
    if len(Dt.shape) == 2:
        lam2 = nx.diag(Dt)
    else:
        lam2 = Dt.reshape((-1,))

    lam1 = lam1.astype(complex)
    lam2 = lam2.astype(complex)

    lam1s = nx.real(lam1) * real_scale + 1j * nx.imag(lam1) * imag_scale
    lam2s = nx.real(lam2) * real_scale + 1j * nx.imag(lam2) * imag_scale
    C_lambda = nx.abs(lam1s[:, None] - lam2s[None, :]) ** 2

    delta = _delta_matrix_1d_hs(Rs, Ls, Rt, Lt, nx=nx)
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
    Ds: array-like, shape (n_s,) or (n_s, n_s)
        Eigenvalues of operator T1 (or diagonal matrix).
    Rs: array-like, shape (L, n_s)
        Right eigenvectors of operator T1.
    Ls: array-like, shape (L, n_s)
        Left eigenvectors of operator T1.
    Dt: array-like, shape (n_t,) or (n_t, n_t)
        Eigenvalues of operator T2 (or diagonal matrix).
    Rt: array-like, shape (L, n_t)
        Right eigenvectors of operator T2.
    Lt: array-like, shape (L, n_t)
        Left eigenvectors of operator T2.
    eta: float, optional
        Weighting between spectral and Grassmann terms, default 0.5.
    p: int, optional
        Power for the OT cost, default 2.
    q: int, optional
        Outer root applied to the OT objective, default 1.
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

    C = cost(
        Ds,
        Rs,
        Ls,
        Dt,
        Rt,
        Lt,
        eta=eta,
        p=p,
        grassman_metric=grassman_metric,
        real_scale=real_scale,
        imag_scale=imag_scale,
        nx=nx,
    )

    n, m = C.shape

    if Ws is None:
        Ws = nx.ones((n,), dtype=C.dtype) / float(n)
    else:
        Ws = nx.asarray(Ws)

    if Wt is None:
        Wt = nx.ones((m,), dtype=C.dtype) / float(m)
    else:
        Wt = nx.asarray(Wt)

    Ws = Ws / nx.sum(Ws)
    Wt = Wt / nx.sum(Wt)

    P = ot_plan(C, Ws=Ws, Wt=Wt, nx=nx)
    obj = ot_score(C, P, p=p, nx=nx)
    return float(obj) ** (1.0 / float(q))


def metric_from_operator(
    T1,
    T2,
    r=None,
    eta=0.5,
    p=2,
    q=1,
    grassman_metric="chordal",
    real_scale=1.0,
    imag_scale=1.0,
    Ws=None,
    Wt=None,
):
    """Compute the SGOT metric directly from two operators.

    Parameters
    ----------
    T1: array-like, shape (d, d)
        First operator.
    T2: array-like, shape (d, d)
        Second operator.
    r: int, optional
        Number of modes to keep. If None, keep all modes.
    eta: float, optional
        Weighting between spectral and Grassmann terms, default 0.5.
    p: int, optional
        Power for the OT cost, default 2.
    q: int, optional
        Outer root applied to the OT objective, default 1.
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
    Ds, Rs, Ls = _atoms_from_operator(T1, r=r, sort_mode="closest_to_1")
    Dt, Rt, Lt = _atoms_from_operator(T2, r=r, sort_mode="closest_to_1")

    return metric(
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
        Ws=Ws,
        Wt=Wt,
    )


def operator_estimator(
    X,
    Y=None,
    r=None,
    ref=1e-8,
    force_complex=False,
):
    """Estimate a linear operator from data.

    Parameters
    ----------
    X: array-like, shape (n_samples, d) or (d, n_samples)
        Input snapshot matrix.
    Y: array-like, shape like X, optional
        Output snapshot matrix. If None, uses a one-step shift of X.
    r: int, optional
        Rank for optional truncated SVD of the estimated operator.
    ref: float, optional
        Tikhonov regularization strength, default 1e-8.
    force_complex: bool, optional
        If True, cast inputs to complex dtype.

    Returns
    ----------
    T_hat: np.ndarray, shape (d, d)
        Estimated linear operator.
    """
    nx = get_backend(X, Y) if Y is not None else get_backend(X)

    X = nx.asarray(X)

    if Y is None:
        if X.ndim != 2 or int(X.shape[0]) < 2:
            raise ValueError("If Y is None, X must be 2D with at least 2 samples/rows.")
        X0 = X[:-1]
        Y0 = X[1:]
    else:
        Y = nx.asarray(Y)
        if tuple(X.shape) != tuple(Y.shape):
            raise ValueError(
                f"X and Y must have the same shape; got {X.shape} vs {Y.shape}"
            )
        X0, Y0 = X, Y

    if X0.shape[0] >= 1 and X0.shape[0] != X0.shape[1]:
        if X0.shape[0] >= X0.shape[1]:
            Xc = X0.T
            Yc = Y0.T
        else:
            Xc = X0
            Yc = Y0
    else:
        Xc = X0
        Yc = Y0

    if Xc.ndim != 2 or Yc.ndim != 2:
        raise ValueError("X and Y must be 2D arrays after processing.")

    d, n = int(Xc.shape[0]), int(Xc.shape[1])
    if tuple(Yc.shape) != (d, n):
        raise ValueError(
            f"After formatting, expected Y to have shape {(d, n)}, got {Yc.shape}"
        )

    if force_complex:
        Xc_np = ot.backend.to_numpy(Xc)  # explicit backend->NumPy copy
        Yc_np = ot.backend.to_numpy(Yc)
        Xc_np = Xc_np.astype(np.complex128, copy=False)
        Yc_np = Yc_np.astype(np.complex128, copy=False)
        if hasattr(nx, "from_numpy"):
            Xc = nx.from_numpy(Xc_np, type_as=Xc)
            Yc = nx.from_numpy(Yc_np, type_as=Yc)
        else:
            Xc = nx.asarray(Xc_np)
            Yc = nx.asarray(Yc_np)

    XXH = nx.dot(Xc, nx.conj(Xc).T)
    YXH = nx.dot(Yc, nx.conj(Xc).T)
    A = XXH + ref * nx.eye(d, type_as=XXH)

    AH = nx.conj(A).T
    BH = nx.conj(YXH).T

    AH_np = ot.backend.to_numpy(AH)  # explicit backend->NumPy copy
    BH_np = ot.backend.to_numpy(BH)
    Xsol_np = np.linalg.solve(AH_np, BH_np)

    if hasattr(nx, "from_numpy"):
        Xsol = nx.from_numpy(Xsol_np, type_as=YXH)
    else:
        Xsol = nx.asarray(Xsol_np)

    T_hat = nx.conj(Xsol).T

    if r is not None:
        r = int(r)
        if not (1 <= r <= d):
            raise ValueError(f"r must be in [1, {d}], got r={r}")

        T_np = ot.backend.to_numpy(T_hat)  # explicit backend->NumPy copy
        U, S, Vh = np.linalg.svd(T_np, full_matrices=False)
        T_np = (U[:, :r] * S[:r]) @ Vh[:r, :]

        if hasattr(nx, "from_numpy"):
            T_hat = nx.from_numpy(T_np, type_as=T_hat)
        else:
            T_hat = nx.asarray(T_np)

    return T_hat
