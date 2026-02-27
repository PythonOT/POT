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


def eigenvalue_cost_matrix(Ds, Dt, q=1, eigen_scaling=None, nx=None):
    """Compute pairwise eigenvalue distances for source and target domains.

    Parameters
    ----------
    Ds: array-like, shape (n_s,)
        Source eigenvalues.
    Dt: array-like, shape (n_t,)
        Target eigenvalues.
    eigen_scaling: None or array-like of length 2, optional
        Scaling (real_scale, imag_scale) applied to eigenvalues before computing
        distances. If None, defaults to (1.0, 1.0). Accepts tuple/list or
        array/tensor with two entries.

    Returns
    ----------
    C: np.ndarray, shape (n_s, n_t)
        Eigenvalue cost matrix.
    """
    if nx is None:
        nx = get_backend(Ds, Dt)

    if eigen_scaling is None:
        real_scale, imag_scale = 1.0, 1.0
    else:
        if isinstance(eigen_scaling, (tuple, list)):
            real_scale, imag_scale = eigen_scaling
        else:
            real_scale, imag_scale = eigen_scaling[0], eigen_scaling[1]

    Dsn = nx.real(Ds) * real_scale + 1j * nx.imag(Ds) * imag_scale
    Dtn = nx.real(Dt) * real_scale + 1j * nx.imag(Dt) * imag_scale
    prod = Dsn[:, None] - Dtn[None, :]
    prod = nx.real(prod * nx.conj(prod))
    return prod ** (q / 2)


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
        # Martin-type Grassmann metric: -log(delta^2) with lower clamp at eps.
        # We deliberately avoid any upper threshold to stay close to the
        # information-geometric interpretation in Germain et al. (2025).
        delta2 = nx.maximum(delta**2, eps)
        return -nx.log(delta2)
    raise ValueError(f"Unknown grassman_metric: {grassman_metric}")


#####################################################################################################################################
#####################################################################################################################################
### SPECTRAL-GRASSMANNIAN WASSERSTEIN METRIC ###
#####################################################################################################################################
#####################################################################################################################################
def sgot_cost_matrix(
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
    eigen_scaling=None,
    nx=None,
):
    r"""Compute the SGOT cost matrix between two spectral decompositions.

    This returns the discrete ground cost matrix used in the SGOT optimal transport
    objective. Each spectral atom is :math:`z_i=(\lambda_i, V_i)` where
    :math:`\lambda_i \in \mathbb{C}` is an eigenvalue and :math:`V_i` is the
    associated (bi-orthogonal) eigenspace point.

    .. math::
        C_2(i,j) \;=\; \eta\,C_\lambda(i,j) \;+\; (1-\eta)\,C_G(i,j),

    with spectral term

    .. math::
        C_\lambda(i,j) \;=\; \big|\lambda_i - \lambda'_j\big|^{q},

    and Grassmann term computed from a similarity score :math:`\delta_{ij}\in[0,1]`
    built from left/right eigenvectors

    .. math::
        \delta_{ij} \;=\; \left|\langle r_i, r'_j\rangle\,\langle \ell_i, \ell'_j\rangle\right|.

    Depending on ``grassman_metric``, the Grassmann contribution is:

    - ``"chordal"``:
    .. math::
        C_G(i,j) \;=\; 1 - \delta_{ij}^2
    - ``"geodesic"``:
    .. math::
        C_G(i,j) \;=\; \arccos(\delta_{ij})^2
    - ``"procrustes"``:
    .. math::
        C_G(i,j) \;=\; 2(1-\delta_{ij})
    - ``"martin"``:
    .. math::
        C_G(i,j) \;=\; -\log\!\left(\max(\delta_{ij}^2,\varepsilon)\right)

    Finally, we return a matrix suited for a :math:`p`-Wasserstein objective by
    treating :math:`C_2 \approx d^2` and outputting

    .. math::
        C(i,j) \;=\; \big(\operatorname{Re}(C_2(i,j))\big)^{p/2}.

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
        Exponent defining the OT ground cost. The returned cost is :math:`d^p` with
        :math:`d^2 \approx C_2`. Default is 2.
    q: int, optional
        Exponent applied to the eigenvalue distance in the spectral term.
        Default is 1.
    grassman_metric: str, optional
        Metric type: "geodesic", "chordal", "procrustes", or "martin".
    eigen_scaling: None or array-like of length 2, optional
        Scaling ``(real_scale, imag_scale)`` applied to eigenvalues before computing
        :math:`C_\lambda`. If provided, eigenvalues are transformed as
        :math:`\lambda \mapsto \alpha\operatorname{Re}(\lambda) + i\,\beta\operatorname{Im}(\lambda)`.
        If None, defaults to ``(1.0, 1.0)``. Accepts tuple/list or array/tensor with
        two entries.
    nx: module, optional
        Backend (NumPy-compatible). If None, inferred from inputs.

    Returns
    ----------
    C: array-like, shape (n_s, n_t)
        SGOT cost matrix :math:`C = d^p`.

    References
    ----------
    Germain et al., *Spectral-Grassmann Optimal Transport* (SGOT).
    """
    if nx is None:
        nx = get_backend(Ds, Rs, Ls, Dt, Rt, Lt)

    if Ds.ndim != 1 or Dt.ndim != 1:
        raise ValueError(
            f"sgot_cost_matrix() expects Ds, Dt 1D; "
            f"got Ds {getattr(Ds, 'shape', None)}, Dt {getattr(Dt, 'shape', None)}"
        )

    if Rs.shape != Ls.shape or Rt.shape != Lt.shape:
        raise ValueError(
            "Right/left eigenvector shapes must match; got "
            f"(Rs,Ls)=({Rs.shape},{Ls.shape}), (Rt,Lt)=({Rt.shape},{Lt.shape})"
        )

    if Rs.shape[1] != Ds.shape[0] or Rt.shape[1] != Dt.shape[0]:
        raise ValueError(
            "Eigenvector columns must match eigenvalues: "
            f"Rs {Rs.shape[1]} vs Ds {Ds.shape[0]}, "
            f"Rt {Rt.shape[1]} vs Dt {Dt.shape[0]}"
        )

    C_lambda = eigenvalue_cost_matrix(Ds, Dt, q=q, eigen_scaling=eigen_scaling, nx=nx)

    delta = _delta_matrix_1d(Rs, Ls, Rt, Lt, nx=nx)
    C_grass = _grassmann_distance_squared(delta, grassman_metric=grassman_metric, nx=nx)

    C2 = eta * C_lambda + (1.0 - eta) * C_grass
    C = nx.real(C2) ** (p / 2.0)

    return C


def _validate_sgot_metric_inputs(Ds, Dt):
    """Validate that eigenvalue inputs for SGOT metric are 1D."""
    Ds_shape = getattr(Ds, "shape", None)
    Dt_shape = getattr(Dt, "shape", None)
    Ds_ndim = getattr(Ds, "ndim", None)
    Dt_ndim = getattr(Dt, "ndim", None)
    if Ds_ndim != 1 or Dt_ndim != 1:
        raise ValueError(
            "sgot_metric() expects Ds and Dt to be 1D (n,), "
            f"got Ds shape {Ds_shape} and Dt shape {Dt_shape}"
        )


def sgot_metric(
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
    eigen_scaling=None,
    Ws=None,
    Wt=None,
    nx=None,
):
    r"""Compute the SGOT metric between two spectral decompositions.

    This function computes a discrete optimal transport problem between two measures
    over spectral atoms :math:`z_i=(\lambda_i, V_i)` and :math:`z'_j=(\lambda'_j, V'_j)`.
    Using the ground cost matrix :math:`C=d^p` returned by :func:`sgot_cost_matrix`,
    we solve:

    .. math::
        P^\star \in \arg\min_{P\in\Pi(W_s, W_t)} \langle C, P\rangle,

    and compute the associated :math:`p`-Wasserstein objective:

    .. math::
        \mathrm{obj} \;=\; \left(\sum_{i,j} C(i,j)\,P^\star_{ij}\right)^{1/p}.

    This implementation returns an additional outer root:

    .. math::
        \mathrm{SGOT} \;=\; \mathrm{obj}^{1/r}.

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
        Exponent defining the OT ground cost and Wasserstein order. The cost matrix
        is :math:`d^p` and the OT objective is raised to the power :math:`1/p`.
        Default is 2.
    q: int, optional
        Exponent applied to the eigenvalue distance in the spectral term.
        Default is 1.
    r: int, optional
        Outer root applied to the Wasserstein objective. Default is 2.
    grassman_metric: str, optional
        Metric type: "geodesic", "chordal", "procrustes", or "martin".
    eigen_scaling: None or array-like of length 2, optional
        Scaling ``(real_scale, imag_scale)`` applied to eigenvalues before computing
        the spectral part of the cost. If None, defaults to ``(1.0, 1.0)``.
    Ws: array-like, shape (n_s,), optional
        Source distribution. If None, uses a uniform distribution.
    Wt: array-like, shape (n_t,), optional
        Target distribution. If None, uses a uniform distribution.
    nx: module, optional
        Backend (NumPy-compatible). If None, inferred from inputs.

    Returns
    ----------
    dist: float
        SGOT metric value.

    References
    ----------
    Germain et al., *Spectral-Grassmann Optimal Transport* (SGOT).
    """
    if nx is None:
        nx = get_backend(Ds, Rs, Ls, Dt, Rt, Lt)

    _validate_sgot_metric_inputs(Ds, Dt)

    C = sgot_cost_matrix(
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
        eigen_scaling=eigen_scaling,
        nx=nx,
    )

    if Ws is None:
        Ws = nx.ones((C.shape[0],), type_as=C) / float(C.shape[0])
    if Wt is None:
        Wt = nx.ones((C.shape[1],), type_as=C) / float(C.shape[1])

    Ws = Ws / nx.sum(Ws)
    Wt = Wt / nx.sum(Wt)

    obj = ot.emd2(Ws, Wt, nx.real(C))
    obj = obj ** (1.0 / p)
    return obj ** (1.0 / float(r))
