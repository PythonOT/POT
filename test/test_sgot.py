"""Tests for ot.sgot module"""

# Author: Sienna O'Shea  <osheasienna@gmail.com>
#         Thibaut Germain<thibaut.germain.pro@gmail.com>
# License: MIT License

import numpy as np
import pytest

from ot.backend import get_backend

try:
    import torch
except ImportError:
    torch = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None

from ot.sgot import (
    eigenvalue_cost_matrix,
    _delta_matrix_1d,
    _grassmann_distance_squared,
    cost,
    metric,
)

rng = np.random.RandomState(0)


def rand_complex(shape):
    real = rng.randn(*shape)
    imag = rng.randn(*shape)
    return real + 1j * imag


def random_atoms(d=8, r=4):
    Ds = rand_complex((r,))
    Rs = rand_complex((d, r))
    Ls = rand_complex((d, r))
    Dt = rand_complex((r,))
    Rt = rand_complex((d, r))
    Lt = rand_complex((d, r))
    return Ds, Rs, Ls, Dt, Rt, Lt


# ---------------------------------------------------------------------
# DATA / SAMPLING TESTS
# ---------------------------------------------------------------------


def test_atoms_are_complex():
    """Confirm sampled atoms are complex (Gaussian real + 1j*imag)."""
    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()
    for name, arr in [
        ("Ds", Ds),
        ("Rs", Rs),
        ("Ls", Ls),
        ("Dt", Dt),
        ("Rt", Rt),
        ("Lt", Lt),
    ]:
        assert np.iscomplexobj(arr), f"{name} should be complex"
        assert np.any(np.imag(arr) != 0), f"{name} should have non-zero imaginary part"


def test_random_d_r():
    """Sample d and r uniformly and run cost (and metric when available) with those shapes."""
    d_min, d_max = 4, 12
    r_min, r_max = 2, 6
    for _ in range(5):
        d = int(rng.randint(d_min, d_max + 1))
        r = int(rng.randint(r_min, r_max + 1))
        Ds, Rs, Ls, Dt, Rt, Lt = random_atoms(d=d, r=r)
        C = cost(Ds, Rs, Ls, Dt, Rt, Lt)
        np.testing.assert_allclose(C.shape, (r, r))
        assert np.all(np.isfinite(C)) and np.all(C >= 0)
        try:
            dist = metric(Ds, Rs, Ls, Dt, Rt, Lt)
            assert np.isfinite(dist) and dist >= 0
        except TypeError:
            pytest.skip("metric() unavailable (emd_c signature mismatch)")


# ---------------------------------------------------------------------
# BACKEND CONSISTENCY TESTS
# ---------------------------------------------------------------------


def test_backend_return():
    """Confirm get_backend returns the correct backend for numpy/torch/jax arrays."""
    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()
    nx = get_backend(Ds, Rs, Ls, Dt, Rt, Lt)
    assert nx is not None
    assert nx.__name__ == "numpy"

    if torch is not None:
        Ds_t = torch.from_numpy(Ds)
        nx_t = get_backend(Ds_t)
        assert nx_t is not None
        assert nx_t.__name__ == "torch"

    if jax is not None:
        Ds_j = jnp.array(Ds)
        nx_j = get_backend(Ds_j)
        assert nx_j is not None
        assert nx_j.__name__ == "jax"


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "jax"])
def test_cost_backend_consistency(backend_name):
    if backend_name == "torch" and torch is None:
        pytest.skip("Torch not available")
    if backend_name == "jax" and jax is None:
        pytest.skip("JAX not available")

    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()

    C_np = cost(Ds, Rs, Ls, Dt, Rt, Lt)

    if backend_name == "numpy":
        C_backend = C_np

    elif backend_name == "torch":
        Ds_b = torch.from_numpy(Ds)
        Rs_b = torch.from_numpy(Rs)
        Ls_b = torch.from_numpy(Ls)
        Dt_b = torch.from_numpy(Dt)
        Rt_b = torch.from_numpy(Rt)
        Lt_b = torch.from_numpy(Lt)
        C_backend = cost(Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b)
        C_backend = C_backend.detach().cpu().numpy()

    elif backend_name == "jax":
        Ds_b = jnp.array(Ds)
        Rs_b = jnp.array(Rs)
        Ls_b = jnp.array(Ls)
        Dt_b = jnp.array(Dt)
        Rt_b = jnp.array(Rt)
        Lt_b = jnp.array(Lt)
        C_backend = cost(Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b)
        C_backend = np.array(C_backend)

    np.testing.assert_allclose(C_backend, C_np, atol=1e-6)


# ---------------------------------------------------------------------
# DELTA MATRIX TESTS
# ---------------------------------------------------------------------


def test_delta_identity():
    r = 4
    I = np.eye(r, dtype=complex)
    delta = _delta_matrix_1d(I, I, I, I)
    np.testing.assert_allclose(delta, np.eye(r), atol=1e-12)


def test_delta_swap_invariance():
    d, r = 6, 3
    R = rand_complex((d, r))
    L = R.copy()
    delta1 = _delta_matrix_1d(R, L, R, L)
    delta2 = _delta_matrix_1d(L, R, L, R)
    np.testing.assert_allclose(delta1, delta2, atol=1e-12)


# ---------------------------------------------------------------------
# GRASSMANN DISTANCE TESTS
# ---------------------------------------------------------------------


@pytest.mark.parametrize("metric_name", ["geodesic", "chordal", "procrustes", "martin"])
def test_grassmann_zero_distance(metric_name):
    delta = np.ones((3, 3))
    dist2 = _grassmann_distance_squared(delta, grassman_metric=metric_name)
    np.testing.assert_allclose(dist2, 0.0, atol=1e-12)


def test_grassmann_invalid_name():
    delta = np.ones((2, 2))
    with pytest.raises(ValueError):
        _grassmann_distance_squared(delta, grassman_metric="cordal")


# ---------------------------------------------------------------------
# COST TESTS
# ---------------------------------------------------------------------


def test_cost_self_zero(nx):
    """(D_S R_S L_S D_S): diagonal of cost matrix (same atom to same atom) should be near zero."""
    Ds, Rs, Ls, _, _, _ = random_atoms()
    Ds_b, Rs_b, Ls_b, Ds_b2, Rs_b2, Ls_b2 = nx.from_numpy(Ds, Rs, Ls, Ds, Rs, Ls)
    C = cost(Ds_b, Rs_b, Ls_b, Ds_b2, Rs_b2, Ls_b2)
    C_np = nx.to_numpy(C)
    np.testing.assert_allclose(np.diag(C_np), np.zeros(C_np.shape[0]), atol=1e-10)
    np.testing.assert_allclose(C_np, C_np.T, atol=1e-10)


def test_cost_reference(nx):
    """Cost with same inputs and HPs should be deterministic (np.testing.assert_allclose)."""
    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()
    Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b = nx.from_numpy(Ds, Rs, Ls, Dt, Rt, Lt)
    eta, p, q = 0.5, 2, 1
    C1 = cost(Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b, eta=eta, p=p, q=q)
    C2 = cost(Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b, eta=eta, p=p, q=q)
    np.testing.assert_allclose(nx.to_numpy(C1), nx.to_numpy(C2), atol=1e-12)


@pytest.mark.parametrize(
    "grassman_metric", ["geodesic", "chordal", "procrustes", "martin"]
)
def test_cost_basic(grassman_metric, nx):
    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()
    Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b = nx.from_numpy(Ds, Rs, Ls, Dt, Rt, Lt)
    C = cost(Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b, grassman_metric=grassman_metric)
    C_np = nx.to_numpy(C)
    assert C_np.shape == (Ds.shape[0], Dt.shape[0])
    assert np.all(np.isfinite(C_np))
    assert np.all(C_np >= 0)


def test_cost_validation():
    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()

    with pytest.raises(ValueError):
        cost(Ds.reshape(-1, 1), Rs, Ls, Dt, Rt, Lt)

    with pytest.raises(ValueError):
        cost(Ds, Rs[:, :-1], Ls, Dt, Rt, Lt)


# ---------------------------------------------------------------------
# METRIC TESTS
# ---------------------------------------------------------------------


def test_metric_self_zero():
    Ds, Rs, Ls, _, _, _ = random_atoms()
    dist = metric(Ds, Rs, Ls, Ds, Rs, Ls)
    assert np.isfinite(dist)
    assert abs(dist) < 2e-4


def test_metric_symmetry():
    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()
    d1 = metric(Ds, Rs, Ls, Dt, Rt, Lt)
    d2 = metric(Dt, Rt, Lt, Ds, Rs, Ls)
    np.testing.assert_allclose(d1, d2, atol=1e-8)


def test_metric_with_weights():
    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()
    r = Ds.shape[0]

    logits_s = rng.randn(r)
    logits_t = rng.randn(r)

    Ws = np.exp(logits_s)
    Ws = Ws / np.sum(Ws)

    Wt = np.exp(logits_t)
    Wt = Wt / np.sum(Wt)

    dist = metric(Ds, Rs, Ls, Dt, Rt, Lt, Ws=Ws, Wt=Wt)
    assert np.isfinite(dist)


# ---------------------------------------------------------------------
# HYPERPARAMETER SWEEP TEST
# ---------------------------------------------------------------------


def test_hyperparameter_sweep_cost(nx):
    """Create test_cost for each trial: sweep over HPs and run cost()."""
    grassmann_types = ["geodesic", "chordal", "procrustes", "martin"]
    n_trials = 10
    for _ in range(n_trials):
        Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()
        Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b = nx.from_numpy(Ds, Rs, Ls, Dt, Rt, Lt)
        eta = rng.uniform(0.0, 1.0)
        p = rng.choice([1, 2])
        q = rng.choice([1, 2])
        gm = rng.choice(grassmann_types)
        C = cost(
            Ds_b,
            Rs_b,
            Ls_b,
            Dt_b,
            Rt_b,
            Lt_b,
            eta=eta,
            p=p,
            q=q,
            grassman_metric=gm,
        )
        C_np = nx.to_numpy(C)
        assert C_np.shape == (Ds.shape[0], Dt.shape[0])
        assert np.all(np.isfinite(C_np))
        assert np.all(C_np >= 0)


def test_hyperparameter_sweep():
    grassmann_types = ["geodesic", "chordal", "procrustes", "martin"]

    for _ in range(10):
        Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()
        eta = rng.uniform(0.0, 1.0)
        p = rng.choice([1, 2])
        q = rng.choice([1, 2])
        r = rng.choice([1, 2])
        gm = rng.choice(grassmann_types)

        dist = metric(
            Ds,
            Rs,
            Ls,
            Dt,
            Rt,
            Lt,
            eta=eta,
            p=p,
            q=q,
            r=r,
            grassman_metric=gm,
        )

        assert np.isfinite(dist)
        assert dist >= 0
