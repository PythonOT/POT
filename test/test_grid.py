"""Tests for the EMD-L1 grid solver (ot.lp.emd_grid_l1)"""

# Author: Tom Vercauteren
#
# License: MIT License

import itertools
from unittest import mock

import numpy as np
import pytest

import ot
from ot.lp import emd_grid_l1
from ot.lp._grid import _emd_grid_l1_1d_cost, _emd_grid_l1_1d_plan


def _grid_coords(shape):
    axes = [np.arange(s) for s in shape]
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(shape), -1).T


def _dense_cost_and_check(a, b, shape):
    coords = _grid_coords(shape)
    M = ot.dist(coords, coords, metric="cityblock")
    return ot.emd2(a, b, M)


def _check_plan(G, a, b, shape, cost, nx):
    """G must be a genuine coupling: nonnegative, row sums a, column sums b,
    and total mass-times-distance equal to the reported cost."""
    dense = nx.to_numpy(nx.todense(G))
    assert np.all(dense >= -1e-12)
    np.testing.assert_allclose(dense.sum(axis=1), a, atol=1e-8)
    np.testing.assert_allclose(dense.sum(axis=0), b, atol=1e-8)

    coords = _grid_coords(shape)
    M = ot.dist(coords, coords, metric="cityblock")
    np.testing.assert_allclose((dense * M).sum(), cost, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "shape",
    [
        (4,),
        (5,),
        (2, 3),
        (3, 4),
        (2, 2, 2),
        (2, 3, 2),
    ],
)
def test_emd_grid_l1_vs_dense(shape):
    """The grid solver must match the dense bipartite solver exactly."""
    rng = np.random.RandomState(42)
    n = int(np.prod(shape))
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(n)
    b /= b.sum()

    cost_dense = _dense_cost_and_check(a, b, shape)
    cost_grid = emd_grid_l1(a.reshape(shape), b.reshape(shape))

    np.testing.assert_allclose(cost_grid, cost_dense, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("shape", [(4,), (3, 3), (2, 3, 2)])
def test_emd_grid_l1_identical_histograms(shape):
    """Identical histograms cost 0; the plan, if requested, is the (trivially
    optimal) identity coupling, not an empty one."""
    n = int(np.prod(shape))
    rng = np.random.RandomState(0)
    a = rng.rand(n)
    a /= a.sum()
    A = a.reshape(shape)
    nx = ot.backend.NumpyBackend()

    cost, log = emd_grid_l1(A, A, return_plan=True, log=True)

    assert cost == 0.0
    _check_plan(log["G"], a, a, shape, cost, nx)
    np.testing.assert_allclose(nx.to_numpy(nx.todense(log["G"])).sum(), 1.0)


def test_emd_grid_l1_return_plan_default_false():
    """By default, no plan is computed or returned (cost-only fast path)."""
    shape = (3, 3)
    n = int(np.prod(shape))
    rng = np.random.RandomState(1)
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(n)
    b /= b.sum()
    A, B = a.reshape(shape), b.reshape(shape)

    cost_no_log = emd_grid_l1(A, B)
    cost, log = emd_grid_l1(A, B, log=True)

    assert cost == cost_no_log
    assert "G" not in log

    # Explicitly requesting the plan must add it to the log, with a matching
    # cost.
    cost_with_plan, log_with_plan = emd_grid_l1(A, B, return_plan=True, log=True)
    assert cost_with_plan == cost
    assert "G" in log_with_plan


def test_emd_grid_l1_one_hot():
    """Point mass moving between two corners costs their Manhattan distance."""
    shape = (2, 3, 4)
    A = np.zeros(shape)
    B = np.zeros(shape)
    A[0, 0, 0] = 1.0
    B[1, 2, 3] = 1.0

    cost = emd_grid_l1(A, B)
    np.testing.assert_allclose(cost, 1 + 2 + 3)


def test_emd_grid_l1_plan_marginals():
    """The transportation plan must be a genuine coupling: row sums A,
    column sums B, including the mass A and B already share at the same
    bin (which the underlying flow decomposition alone would miss)."""
    shape = (3, 4)
    n = int(np.prod(shape))
    rng = np.random.RandomState(7)
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(n)
    b /= b.sum()
    nx = ot.backend.NumpyBackend()

    cost, log = emd_grid_l1(
        a.reshape(shape), b.reshape(shape), return_plan=True, log=True
    )

    _check_plan(log["G"], a, b, shape, cost, nx)


def test_emd_grid_l1_mass_mismatch():
    shape = (2, 2)
    A = np.array([1.0, 0.0, 0.0, 0.0]).reshape(shape)
    B = np.array([0.0, 0.0, 0.0, 0.5]).reshape(shape)

    with pytest.raises(AssertionError):
        emd_grid_l1(A, B)

    # Explicitly skipping the check should not raise from the Python side.
    _cost, log = emd_grid_l1(A, B, check_marginals=False, log=True)
    assert log["result_code"] != 1  # not OPTIMAL: infeasible


def test_emd_grid_l1_shape_mismatch():
    a = np.ones(4) / 4
    b = np.ones(5) / 5
    with pytest.raises(ValueError):
        emd_grid_l1(a, b)  # different total number of elements

    # Same total number of elements, but a different grid shape/structure
    # must still raise: the grid geometry, not just the element count, has
    # to match.
    A = np.ones((2, 3)) / 6
    B = np.ones((3, 2)) / 6
    with pytest.raises(ValueError):
        emd_grid_l1(A, B)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_emd_grid_l1_random_grids_batch(ndim):
    """Broader randomized cross-check across many small grid shapes."""
    rng = np.random.RandomState(123)
    for extents in itertools.product(range(2, 4), repeat=ndim):
        n = int(np.prod(extents))
        a = rng.rand(n)
        a /= a.sum()
        b = rng.rand(n)
        b /= b.sum()

        cost_dense = _dense_cost_and_check(a, b, extents)
        cost_grid = emd_grid_l1(a.reshape(extents), b.reshape(extents))
        np.testing.assert_allclose(cost_grid, cost_dense, rtol=1e-6, atol=1e-8)


def test_emd_grid_l1_1d_uses_native_cost_path():
    """A 1D grid with return_plan=False must skip both the general C++
    solver and the O(n) merge that recovers the plan."""
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 1.0])

    with (
        mock.patch("ot.lp._grid.emd_c_grid_l1") as mocked_cpp,
        mock.patch(
            "ot.lp._grid._emd_grid_l1_1d_plan",
            wraps=_emd_grid_l1_1d_plan,
        ) as mocked_plan,
    ):
        cost = emd_grid_l1(a, b)
    mocked_cpp.assert_not_called()
    mocked_plan.assert_not_called()
    np.testing.assert_allclose(cost, 3.0)

    # Requesting the plan on a 1D grid must still avoid the C++ solver, but
    # does need the O(n) merge.
    with (
        mock.patch("ot.lp._grid.emd_c_grid_l1") as mocked_cpp,
        mock.patch(
            "ot.lp._grid._emd_grid_l1_1d_plan",
            wraps=_emd_grid_l1_1d_plan,
        ) as mocked_plan,
    ):
        emd_grid_l1(a, b, return_plan=True)
    mocked_cpp.assert_not_called()
    mocked_plan.assert_called_once()

    # A 2D grid, by contrast, must go through the general C++ solver.
    with mock.patch(
        "ot.lp._grid.emd_c_grid_l1", wraps=ot.lp._grid.emd_c_grid_l1
    ) as mocked_cpp:
        emd_grid_l1(a.reshape(2, 2), b.reshape(2, 2))
    mocked_cpp.assert_called_once()


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_emd_grid_l1_1d_vs_dense(n):
    """Cross-check the dedicated 1D path against the dense solver directly,
    beyond what test_emd_grid_l1_vs_dense already covers."""
    rng = np.random.RandomState(11)
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(n)
    b /= b.sum()

    cost_dense = _dense_cost_and_check(a, b, (n,))
    cost_grid = emd_grid_l1(a, b)
    np.testing.assert_allclose(cost_grid, cost_dense, rtol=1e-6, atol=1e-8)


def test_emd_grid_l1_1d_plan_marginals():
    n = 6
    rng = np.random.RandomState(5)
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(n)
    b /= b.sum()
    nx = ot.backend.NumpyBackend()

    cost, log = emd_grid_l1(a, b, return_plan=True, log=True)

    _check_plan(log["G"], a, b, (n,), cost, nx)


def test_emd_grid_l1_1d_identical_histograms():
    """Identical histograms cost 0 and the plan is the identity coupling,
    even though the underlying merge could in principle emit zero-mass
    entries at ties."""
    a = np.array([0.25, 0.25, 0.25, 0.25])
    nx = ot.backend.NumpyBackend()

    cost, log = emd_grid_l1(a, a, return_plan=True, log=True)

    assert cost == 0.0
    dense = nx.to_numpy(nx.todense(log["G"]))
    assert np.all(np.diag(dense) > 0)
    np.testing.assert_allclose(dense, np.diag(a))


def test_emd_grid_l1_1d_mass_mismatch():
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 0.5])

    with pytest.raises(AssertionError):
        emd_grid_l1(a, b)

    _cost, log = emd_grid_l1(a, b, check_marginals=False, log=True)
    assert log["result_code"] != 1  # not OPTIMAL: infeasible


def test_emd_grid_l1_1d_direct_cost_helper():
    nx = ot.backend.NumpyBackend()

    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 1.0])
    cost, result_code = _emd_grid_l1_1d_cost(a, b, nx)
    assert result_code == 1  # OPTIMAL
    np.testing.assert_allclose(cost, 3.0)

    # Negative values and mass mismatches must be reported as infeasible,
    # like the general (C++) path.
    a_neg = np.array([1.0, -0.1, 0.0, 0.0])
    _cost, result_code = _emd_grid_l1_1d_cost(a_neg, b, nx)
    assert result_code != 1

    a_mismatch = np.array([1.0, 0.0, 0.0, 0.0])
    b_mismatch = np.array([0.0, 0.0, 0.0, 0.5])
    _cost, result_code = _emd_grid_l1_1d_cost(a_mismatch, b_mismatch, nx)
    assert result_code != 1


def test_emd_grid_l1_1d_direct_plan_helper_negative_values():
    a = np.array([1.0, -0.1, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 0.9])
    nx = ot.backend.NumpyBackend()
    _sources, _targets, _values, cost, result_code = _emd_grid_l1_1d_plan(a, b, nx)
    assert result_code != 1  # not OPTIMAL: infeasible
    assert cost == 0.0


def test_emd_grid_l1_1d_direct_plan_helper_mass_mismatch():
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 0.5])
    nx = ot.backend.NumpyBackend()
    _sources, _targets, _values, cost, result_code = _emd_grid_l1_1d_plan(a, b, nx)
    assert result_code != 1  # not OPTIMAL: infeasible
    assert cost == 0.0


def test_emd_grid_l1_backends(nx):
    """Non-numpy inputs (e.g. torch, jax) must be accepted and the outputs
    returned in the same backend/dtype/device as the inputs, including a
    round trip through GPU arrays where the backend supports them."""
    shape = (2, 2, 2)
    n = int(np.prod(shape))
    rng = np.random.RandomState(0)
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(n)
    b /= b.sum()
    A, B = a.reshape(shape), b.reshape(shape)

    cost_np = emd_grid_l1(A, B)

    for tp in nx.__type_list__:
        Ab, Bb = nx.from_numpy(A, B, type_as=tp)

        cost_b = emd_grid_l1(Ab, Bb)
        nx.assert_same_dtype_device(tp, cost_b)
        np.testing.assert_allclose(nx.to_numpy(cost_b), cost_np, rtol=1e-6, atol=1e-8)

        cost_plan_b, log_b = emd_grid_l1(Ab, Bb, return_plan=True, log=True)
        nx.assert_same_dtype_device(tp, cost_plan_b)
        np.testing.assert_allclose(
            nx.to_numpy(cost_plan_b), cost_np, rtol=1e-6, atol=1e-8
        )
        _check_plan(log_b["G"], a, b, shape, nx.to_numpy(cost_plan_b), nx)


def test_emd_grid_l1_1d_backends_native_cost(nx):
    """The 1D, cost-only path must also work across backends, without a plan
    (and, for backends that support it, without ever leaving the device)."""
    a = np.array([0.3, 0.1, 0.2, 0.4])
    b = np.array([0.1, 0.2, 0.3, 0.4])

    cost_np = emd_grid_l1(a, b)

    for tp in nx.__type_list__:
        ab, bb = nx.from_numpy(a, b, type_as=tp)
        cost_b = emd_grid_l1(ab, bb)
        nx.assert_same_dtype_device(tp, cost_b)
        np.testing.assert_allclose(nx.to_numpy(cost_b), cost_np, rtol=1e-6, atol=1e-8)
