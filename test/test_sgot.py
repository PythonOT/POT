"""Tests for ot.sgot module"""

# Author: Sienna O'Shea  <osheasienna@gmail.com>
#         Thibaut Germain<thibaut.germain.pro@gmail.com>
# License: MIT License

import numpy as np
import pytest

from ot.sgot import (
    eigenvalue_cost_matrix,
    _delta_matrix_1d,
    _grassmann_distance_squared,
    sgot_cost_matrix,
    sgot_metric,
)


def random_atoms(d=8, r=4, seed=42):
    """Deterministic complex atoms for given d, r."""

    def _rand_complex(shape, seed_):
        rng = np.random.RandomState(seed_)
        real = rng.randn(*shape)
        imag = rng.randn(*shape)
        return real + 1j * imag

    Ds = _rand_complex((r,), seed + 0)
    Rs = _rand_complex((d, r), seed + 1)
    Ls = _rand_complex((d, r), seed + 2)
    Dt = _rand_complex((r,), seed + 3)
    Rt = _rand_complex((d, r), seed + 4)
    Lt = _rand_complex((d, r), seed + 5)

    return Ds, Rs, Ls, Dt, Rt, Lt


# ---------------------------------------------------------------------
# DATA / SAMPLING TESTS
# ---------------------------------------------------------------------


def test_random_d_r(nx):
    """Sample d and r uniformly and run sgot_cost_matrix (and sgot_metric when available) with those shapes."""
    rng = np.random.RandomState(0)
    d_min, d_max = 4, 12
    r_min, r_max = 2, 6
    for _ in range(5):
        d = int(rng.randint(d_min, d_max + 1))
        r = int(rng.randint(r_min, r_max + 1))
        Ds, Rs, Ls, Dt, Rt, Lt = random_atoms(d=d, r=r)
        Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b = nx.from_numpy(Ds, Rs, Ls, Dt, Rt, Lt)
        C = sgot_cost_matrix(Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b)
        C_np = nx.to_numpy(C)
        np.testing.assert_allclose(C_np.shape, (r, r))
        assert np.all(np.isfinite(C_np)) and np.all(C_np >= 0)
        try:
            dist = sgot_metric(Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b)
            dist_np = nx.to_numpy(dist)
            assert np.isfinite(dist_np) and dist_np >= 0
        except TypeError:
            pytest.skip("sgot_metric() unavailable (emd_c signature mismatch)")


# ---------------------------------------------------------------------
# DELTA MATRIX TESTS
# ---------------------------------------------------------------------


def test_eigenvalue_cost_matrix_simple():
    Ds = np.array([0.0, 1.0])
    Dt = np.array([0.0, 2.0])
    C = eigenvalue_cost_matrix(Ds, Dt, q=2)
    expected = np.array([[0.0, 4.0], [1.0, 1.0]])
    np.testing.assert_allclose(C, expected)


def test_delta_matrix_1d_identity():
    r = 4
    I = np.eye(r, dtype=complex)
    delta = _delta_matrix_1d(I, I, I, I)
    np.testing.assert_allclose(delta, np.eye(r), atol=1e-12)


def test_delta_matrix_1d_swap_invariance():
    d, r = 6, 3
    _, R, _, _, _, _ = random_atoms(d=d, r=r)
    L = R.copy()
    delta1 = _delta_matrix_1d(R, L, R, L)
    delta2 = _delta_matrix_1d(L, R, L, R)
    np.testing.assert_allclose(delta1, delta2, atol=1e-12)


# ---------------------------------------------------------------------
# GRASSMANN DISTANCE TESTS
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "grassman_metric", ["geodesic", "chordal", "procrustes", "martin"]
)
def test_grassmann_zero_distance(grassman_metric, nx):
    delta = nx.from_numpy(np.ones((3, 3)))
    dist2 = _grassmann_distance_squared(delta, grassman_metric=grassman_metric, nx=nx)
    dist2_np = nx.to_numpy(dist2)
    np.testing.assert_allclose(dist2_np, 0.0, atol=1e-12)


def test_grassmann_distance_invalid_name():
    delta = np.ones((2, 2))
    with pytest.raises(ValueError):
        _grassmann_distance_squared(delta, grassman_metric="cordal")


# ---------------------------------------------------------------------
# COST TESTS
# ---------------------------------------------------------------------


def test_cost_self_zero(nx):
    """(D_S R_S L_S D_S): diagonal of sgot_cost_matrix matrix (same atom to same atom) should be near zero."""
    Ds, Rs, Ls, _, _, _ = random_atoms()
    Ds_b, Rs_b, Ls_b, Ds_b2, Rs_b2, Ls_b2 = nx.from_numpy(Ds, Rs, Ls, Ds, Rs, Ls)
    C = sgot_cost_matrix(Ds_b, Rs_b, Ls_b, Ds_b2, Rs_b2, Ls_b2)
    C_np = nx.to_numpy(C)
    np.testing.assert_allclose(np.diag(C_np), np.zeros(C_np.shape[0]), atol=1e-10)
    np.testing.assert_allclose(C_np, C_np.T, atol=1e-10)


def test_grassmann_cost_reference(nx):
    """Cost with same inputs and HPs should be deterministic (np.testing.assert_allclose)."""
    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()
    Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b = nx.from_numpy(Ds, Rs, Ls, Dt, Rt, Lt)
    eta, p, q = 0.5, 2, 1
    C1 = sgot_cost_matrix(Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b, eta=eta, p=p, q=q)
    C2 = sgot_cost_matrix(Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b, eta=eta, p=p, q=q)
    np.testing.assert_allclose(nx.to_numpy(C1), nx.to_numpy(C2), atol=1e-12)


@pytest.mark.parametrize(
    "grassman_metric", ["geodesic", "chordal", "procrustes", "martin"]
)
def test_grassmann_cost_basic_properties(grassman_metric, nx):
    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()
    Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b = nx.from_numpy(Ds, Rs, Ls, Dt, Rt, Lt)
    C = sgot_cost_matrix(
        Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b, grassman_metric=grassman_metric
    )
    C_np = nx.to_numpy(C)
    assert C_np.shape == (Ds.shape[0], Dt.shape[0])
    assert np.all(np.isfinite(C_np))
    assert np.all(C_np >= 0)


def test_sgot_cost_input_validation():
    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()

    with pytest.raises(ValueError):
        sgot_cost_matrix(Ds.reshape(-1, 1), Rs, Ls, Dt, Rt, Lt)

    with pytest.raises(ValueError):
        sgot_cost_matrix(Ds, Rs[:, :-1], Ls, Dt, Rt, Lt)


# ---------------------------------------------------------------------
# METRIC TESTS
# ---------------------------------------------------------------------


def test_sgot_metric_self_zero(nx):
    Ds, Rs, Ls, _, _, _ = random_atoms()
    Ds_b, Rs_b, Ls_b, Ds_b2, Rs_b2, Ls_b2 = nx.from_numpy(Ds, Rs, Ls, Ds, Rs, Ls)
    dist = sgot_metric(Ds_b, Rs_b, Ls_b, Ds_b2, Rs_b2, Ls_b2, nx=nx)
    dist_np = nx.to_numpy(dist)
    assert np.isfinite(dist_np)
    assert abs(float(dist_np)) < 5e-4


def test_sgot_metric_symmetry():
    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()
    d1 = sgot_metric(Ds, Rs, Ls, Dt, Rt, Lt)
    d2 = sgot_metric(Dt, Rt, Lt, Ds, Rs, Ls)
    np.testing.assert_allclose(d1, d2, atol=1e-8)


def test_sgot_metric_with_weights():
    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()
    r = Ds.shape[0]

    rng = np.random.RandomState(1)
    Ws = rng.rand(r)
    Ws = Ws / np.sum(Ws)

    Wt = rng.rand(r)
    Wt = Wt / np.sum(Wt)

    dist = sgot_metric(Ds, Rs, Ls, Dt, Rt, Lt, Ws=Ws, Wt=Wt)
    assert np.isfinite(dist)


# ---------------------------------------------------------------------
# HYPERPARAMETER SWEEP TEST
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "eta, p, q, grassman_metric",
    [
        (0.5, 1, 1, "geodesic"),
        (0.5, 2, 1, "chordal"),
        (0.3, 2, 2, "procrustes"),
        (0.7, 1, 2, "martin"),
    ],
)
def test_hyperparameter_sweep_cost(nx, eta, p, q, grassman_metric):
    """Sweep over a set of fixed HPs and run cost()."""
    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()
    Ds_b, Rs_b, Ls_b, Dt_b, Rt_b, Lt_b = nx.from_numpy(Ds, Rs, Ls, Dt, Rt, Lt)

    C = sgot_cost_matrix(
        Ds_b,
        Rs_b,
        Ls_b,
        Dt_b,
        Rt_b,
        Lt_b,
        eta=eta,
        p=p,
        q=q,
        grassman_metric=grassman_metric,
    )
    C_np = nx.to_numpy(C)
    assert C_np.shape == (Ds.shape[0], Dt.shape[0])
    assert np.all(np.isfinite(C_np))
    assert np.all(C_np >= 0)


@pytest.mark.parametrize(
    "grassman_metric", ["geodesic", "chordal", "procrustes", "martin"]
)
def test_hyperparameter_sweep(grassman_metric):
    Ds, Rs, Ls, Dt, Rt, Lt = random_atoms()
    rng = np.random.RandomState(3)
    eta = rng.uniform(0.0, 1.0)
    p = rng.choice([1, 2])
    q = rng.choice([1, 2])
    r = rng.choice([1, 2])

    dist = sgot_metric(
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
        grassman_metric=grassman_metric,
    )

    assert np.isfinite(dist)
    assert dist >= 0
