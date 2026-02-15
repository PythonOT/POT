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
from sklearn.utils.extmath import randomized_svd
from ot.backend import get_backend

###
# Settings : (Ds,Rs,Ls) if primal, (Ds,Xs,prs,pls) if dual
###

#####################################################################################################################################
#####################################################################################################################################
### PRINCIPAL ANGLE METRICS ###
#####################################################################################################################################
#####################################################################################################################################


def hs_metric(Ds, Rs, Ls, Dt, Rt, Lt, sampfreqs: int = 1, sampfreqt: int = 1):
    """Compute the Hilbert-Schmidt (Frobenius) distance between two operators.

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
    sampfreqs: int, optional, sampling frequency for the source operator with default 1
    sampfreqt: int, optional, sampling frequency for the target operator with default 1

    Returns
    ----------
    dist: float, Frobenius norm
    """
    Ts = Rs @ (np.exp(Ds / sampfreqs).reshape(-1, 1) * Ls.conj().T)
    Tt = Rt @ (np.exp(Dt / sampfreqt).reshape(-1, 1) * Lt.conj().T)
    C = Ts - Tt
    return np.linalg.norm(C, "fro")


def operator_metric(
    Ds,
    Rs,
    Ls,
    Dt,
    Rt,
    Lt,
    sampfreqs: int = 1,
    sampfreqt: int = 1,
    exact: bool = False,
    n_iter: int = 5,
    random_state: int = None,
):
    """Compute the spectral norm distance between two reconstructed operators.

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
    sampfreqs: int, optional
        Sampling frequency for the source operator with default 1
    sampfreqt: int, optional
        Sampling frequency for the target operator with default 1
    exact: bool, optional
    n_iter: int, optional
    random_state: int or None, optional

    Returns
    ----------
    dist: float
    """
    Ts = Rs @ (np.exp(Ds / sampfreqs).reshape(-1, 1) * Ls.conj().T)
    Tt = Rt @ (np.exp(Dt / sampfreqt).reshape(-1, 1) * Lt.conj().T)
    C = Ts - Tt
    if exact:
        return np.linalg.norm(C, 2)
    else:
        _, S, _ = randomized_svd(
            C.real, n_components=1, n_iter=n_iter, random_state=random_state
        )
    return S[0]


def principal_angles_via_svd(A, B):
    """Compute principal angles between two subspaces using SVD of QA^T QB.

    Parameters
        A: array-like, shape (d, p) whose columns span the first subspace
        B: array-like, shape (d, q) whose columns span the second subspace

    Returns
        angle: sorted principal angles (in radians), shape (min(p, q),)
    """
    QA, _ = np.linalg.qr(A, mode="reduced")
    QB, _ = np.linalg.qr(B, mode="reduced")
    C = QA.T @ QB
    # SVD of small matrix C
    _, S, _ = np.linalg.svd(C, full_matrices=False)
    S = np.clip(S, -1.0, 1.0)
    angles = np.arccos(S)
    return np.sort(angles)


def principal_angles_distance(
    Ds,
    Rs,
    Ls,
    Dt,
    Rt,
    Lt,
):
    """Compute a principal angles distance between two spectral decompositions.

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

    Returns
    -------
    dist: float
        Principal-angles distance between the two decompositions.
    """
    ns = Rs.shape[1]
    nt = Rt.shape[1]
    Ms = np.vstack(
        [(ls[:, None] * rs.conj()[None, :]).flatten() for ls, rs in zip(Ls.T, Rs.T)]
    ).T
    Mt = np.vstack(
        [(lt[:, None] * rt.conj()[None, :]).flatten() for lt, rt in zip(Lt.T, Rt.T)]
    ).T
    angles = principal_angles_via_svd(Ms, Mt)
    if angles.shape[0] != max(ns, nt):
        angles = np.hstack([angles, np.pi / 2 * np.ones(max(ns, nt) - angles.shape[0])])
    return np.sqrt(np.sum(angles**2))


#####################################################################################################################################
#####################################################################################################################################
### OT METRIC ###
#####################################################################################################################################
#####################################################################################################################################


def principal_grassman_matrix(Ps, Pt, eps: float = 1e-12):
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
    ns = np.linalg.norm(Ps, axis=0, keepdims=True)
    nt = np.linalg.norm(Pt, axis=0, keepdims=True)
    ns = np.maximum(ns, eps)
    nt = np.maximum(nt, eps)

    Psn = Ps / ns
    Ptn = Pt / nt

    C = Psn.conj().T @ Ptn
    return C


def eigenvector_chordal_cost_matrix(Rs, Ls, Rt, Lt):
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
    Cr = principal_grassman_matrix(Rs, Rt)
    Cl = principal_grassman_matrix(Ls, Lt)
    C = np.sqrt(1 - np.clip((Cr * Cl).real, a_min=0, a_max=1))
    return C


def eigenvalue_cost_matrix(Ds, Dt, real_scale: float = 1.0, imag_scale: float = 1.0):
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
    Dsn = Ds.real * real_scale + 1j * Ds.imag * imag_scale
    Dtn = Dt.real * real_scale + 1j * Dt.imag * imag_scale
    C = np.abs(Dsn[:, None] - Dtn[None, :])
    return C


def ChordalCostFunction(
    real_scale: float = 1.0, imag_scale: float = 1.0, alpha: float = 0.5, p: int = 2
):
    """Generate the chordal cost function.

    Parameters
    ----------
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
    cost_function: Chordal cost function.
    """

    def cost_function(Ds, Rs, Ls, Dt, Rt, Lt) -> np.ndarray:
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

        Returns
        ----------
        C: np.ndarray, shape (n_s, n_t)
            Chordal cost matrix.
        """
        CD = eigenvalue_cost_matrix(
            Ds, Dt, real_scale=real_scale, imag_scale=imag_scale
        )
        CC = eigenvector_chordal_cost_matrix(Rs, Ls, Rt, Lt)
        C = alpha * CD + (1 - alpha) * CC
        return C**p

    return cost_function


def ot_plan(C, Ws=None, Wt=None):
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
    if Ws is None:
        Ws = np.ones(C.shape[0]) / C.shape[0]
    if Wt is None:
        Wt = np.ones(C.shape[1]) / C.shape[1]
    return ot.emd(Ws, Wt, C)


def ot_score(C, P, p: int = 2) -> float:
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
    return np.sum(C * P) ** (1 / p)


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
    cost_fn = ChordalCostFunction(real_scale, imag_scale, alpha, p)
    C = cost_fn(Ds, Rs, Ls, Dt, Rt, Lt)
    P = ot_plan(C)
    return ot_score(C, P, p)


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
    nrm = nx.maximum(nrm, eps)
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
    delta = nx.minimum(nx.maximum(delta, 0.0), 1.0)
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
    T = np.asarray(T)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError(f"T must be a square 2D array; got shape {T.shape}")

    d = T.shape[0]
    if r is None:
        r = d
    r = int(r)
    if not (1 <= r <= d):
        raise ValueError(f"r must be an integer in [1, {d}], got r={r}")

    evals, evecs = np.linalg.eig(T)

    if sort_mode == "closest_to_1":
        order = np.argsort(np.abs(evals - 1.0))
    elif sort_mode == "closest_to_0":
        order = np.argsort(np.abs(evals))
    elif sort_mode == "largest_mag":
        order = np.argsort(-np.abs(evals))
    else:
        raise ValueError(
            "sort_mode must be one of 'closest_to_1', 'closest_to_0', or 'largest_mag'"
        )

    idx = order[:r]
    D = evals[idx]
    R = evecs[:, idx]

    evalsL, evecsL = np.linalg.eig(T.conj().T)

    L = np.zeros((d, r), dtype=complex)
    used = set()

    for i, lam in enumerate(D):
        targets = np.abs(evalsL - np.conj(lam))
        for j in np.argsort(targets):
            if j not in used:
                used.add(j)
                L[:, i] = evecsL[:, j]
                break

    G = L.conj().T @ R
    if np.linalg.matrix_rank(G) < r:
        raise ValueError("Dual normalization failed: L^* R is singular.")

    L = L @ np.linalg.inv(G).conj().T

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
    delta = nx.minimum(nx.maximum(delta, 0.0), 1.0)

    if grassman_metric == "geodesic":
        return nx.arccos(delta) ** 2
    if grassman_metric == "chordal":
        return 1.0 - delta**2
    if grassman_metric == "procrustes":
        return 2.0 * (1.0 - delta)
    if grassman_metric == "martin":
        return -nx.log(nx.maximum(delta**2, eps))
    raise ValueError(f"Unknown grassman_metric: {grassman_metric}")


#####################################################################################################################################
#####################################################################################################################################
### SPECTRAL-GRASSMANNIAN WASSERSTEIN METRIC ###
#####################################################################################################################################
#####################################################################################################################################
def cost(
    D1,
    R1,
    L1,
    D2,
    R2,
    L2,
    eta=0.5,
    p=2,
    grassman_metric="chordal",
    real_scale=1.0,
    imag_scale=1.0,
):
    """Compute the SGOT cost matrix between two spectral decompositions.

    Parameters
    ----------
    D1: array-like, shape (n_1,) or (n_1, n_1)
        Eigenvalues of operator T1 (or diagonal matrix).
    R1: array-like, shape (L, n_1)
        Right eigenvectors of operator T1.
    L1: array-like, shape (L, n_1)
        Left eigenvectors of operator T1.
    D2: array-like, shape (n_2,) or (n_2, n_2)
        Eigenvalues of operator T2 (or diagonal matrix).
    R2: array-like, shape (L, n_2)
        Right eigenvectors of operator T2.
    L2: array-like, shape (L, n_2)
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
    C: array-like, shape (n_1, n_2)
        SGOT cost matrix.
    """
    nx = get_backend(D1, R1, L1, D2, R2, L2)

    D1 = nx.asarray(D1)
    D2 = nx.asarray(D2)
    if len(D1.shape) == 2:
        lam1 = nx.diag(D1)
    else:
        lam1 = D1.reshape((-1,))
    if len(D2.shape) == 2:
        lam2 = nx.diag(D2)
    else:
        lam2 = D2.reshape((-1,))

    lam1 = lam1.astype(complex)
    lam2 = lam2.astype(complex)

    lam1s = nx.real(lam1) * real_scale + 1j * nx.imag(lam1) * imag_scale
    lam2s = nx.real(lam2) * real_scale + 1j * nx.imag(lam2) * imag_scale
    C_lambda = nx.abs(lam1s[:, None] - lam2s[None, :]) ** 2

    delta = _delta_matrix_1d_hs(R1, L1, R2, L2, nx=nx)
    C_grass = _grassmann_distance_squared(delta, grassman_metric=grassman_metric, nx=nx)

    C2 = eta * C_lambda + (1.0 - eta) * C_grass
    C = C2 ** (p / 2.0)

    return C


def metric(
    D1,
    R1,
    L1,
    D2,
    R2,
    L2,
    eta=0.5,
    p=2,
    q=1,
    grassman_metric="chordal",
    real_scale=1.0,
    imag_scale=1.0,
    Ws=None,
    Wt=None,
):
    """Compute the SGOT metric between two spectral decompositions.

    Parameters
    ----------
    D1: array-like, shape (n_1,) or (n_1, n_1)
        Eigenvalues of operator T1 (or diagonal matrix).
    R1: array-like, shape (L, n_1)
        Right eigenvectors of operator T1.
    L1: array-like, shape (L, n_1)
        Left eigenvectors of operator T1.
    D2: array-like, shape (n_2,) or (n_2, n_2)
        Eigenvalues of operator T2 (or diagonal matrix).
    R2: array-like, shape (L, n_2)
        Right eigenvectors of operator T2.
    L2: array-like, shape (L, n_2)
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
    Ws: array-like, shape (n_1,), optional
        Source distribution. If None, uses a uniform distribution.
    Wt: array-like, shape (n_2,), optional
        Target distribution. If None, uses a uniform distribution.

    Returns
    ----------
    dist: float
        SGOT metric value.
    """
    C = cost(
        D1,
        R1,
        L1,
        D2,
        R2,
        L2,
        eta=eta,
        p=p,
        grassman_metric=grassman_metric,
        real_scale=real_scale,
        imag_scale=imag_scale,
    )

    nx = get_backend(C)
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

    C_np = ot.backend.to_numpy(C)
    Ws_np = ot.backend.to_numpy(Ws)
    Wt_np = ot.backend.to_numpy(Wt)

    P = ot_plan(C_np, Ws=Ws_np, Wt=Wt_np)
    obj = ot_score(C_np, P, p=p)

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
    Ws: array-like, shape (n_1,), optional
        Source distribution. If None, uses a uniform distribution.
    Wt: array-like, shape (n_2,), optional
        Target distribution. If None, uses a uniform distribution.

    Returns
    ----------
    dist: float
        SGOT metric value.
    """
    D1, R1, L1 = _atoms_from_operator(T1, r=r, sort_mode="closest_to_1")
    D2, R2, L2 = _atoms_from_operator(T2, r=r, sort_mode="closest_to_1")

    return metric(
        D1,
        R1,
        L1,
        D2,
        R2,
        L2,
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
    X = np.asarray(X)

    if Y is None:
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("If Y is None, X must be 2D with at least 2 samples/rows.")
        X0 = X[:-1]
        Y0 = X[1:]
    else:
        Y = np.asarray(Y)
        if X.shape != Y.shape:
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

    d, n = Xc.shape
    if Yc.shape != (d, n):
        raise ValueError(
            f"After formatting, expected Y to have shape {(d, n)}, got {Yc.shape}"
        )

    if force_complex:
        Xc = Xc.astype(complex)
        Yc = Yc.astype(complex)

    XXH = Xc @ Xc.conj().T
    YXH = Yc @ Xc.conj().T
    A = XXH + ref * np.eye(d, dtype=XXH.dtype)

    T_hat = np.linalg.solve(A.T.conj(), YXH.T.conj()).T.conj()

    if r is not None:
        if not (1 <= r <= d):
            raise ValueError(f"r must be in [1, {d}], got r={r}")
        U, S, Vh = np.linalg.svd(T_hat, full_matrices=False)
        T_hat = (U[:, :r] * S[:r]) @ Vh[:r, :]

    return T_hat
