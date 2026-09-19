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
from ot.lp._grid import _emd_grid_l1_1d, _emd_grid_l1_1d_monotone_plan


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
    solver and the merge that recovers the plan."""
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 1.0])

    with (
        mock.patch("ot.lp._grid.emd_c_grid_l1") as mocked_cpp,
        mock.patch(
            "ot.lp._grid._emd_grid_l1_1d_monotone_plan",
            wraps=_emd_grid_l1_1d_monotone_plan,
        ) as mocked_plan,
    ):
        cost = emd_grid_l1(a, b)
    mocked_cpp.assert_not_called()
    mocked_plan.assert_not_called()
    np.testing.assert_allclose(cost, 3.0)

    # Requesting the plan on a 1D grid must still avoid the C++ solver, but
    # does need the merge.
    with (
        mock.patch("ot.lp._grid.emd_c_grid_l1") as mocked_cpp,
        mock.patch(
            "ot.lp._grid._emd_grid_l1_1d_monotone_plan",
            wraps=_emd_grid_l1_1d_monotone_plan,
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


def test_emd_grid_l1_1d_direct_helper_cost_only():
    nx = ot.backend.NumpyBackend()

    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 1.0])
    cost, alpha, sources, targets, values, result_code = _emd_grid_l1_1d(
        a, b, False, False, nx
    )
    assert result_code == 1  # OPTIMAL
    np.testing.assert_allclose(cost, 3.0)
    assert alpha is None
    assert sources is None and targets is None and values is None

    # return_alpha=True must compute it even without a plan.
    _cost, alpha, _sources, _targets, _values, result_code = _emd_grid_l1_1d(
        a, b, False, True, nx
    )
    assert alpha is not None

    # Negative values and mass mismatches must be reported as infeasible,
    # like the general (C++) path.
    a_neg = np.array([1.0, -0.1, 0.0, 0.0])
    _cost, _alpha, _sources, _targets, _values, result_code = _emd_grid_l1_1d(
        a_neg, b, False, False, nx
    )
    assert result_code != 1

    a_mismatch = np.array([1.0, 0.0, 0.0, 0.0])
    b_mismatch = np.array([0.0, 0.0, 0.0, 0.5])
    _cost, _alpha, _sources, _targets, _values, result_code = _emd_grid_l1_1d(
        a_mismatch, b_mismatch, False, False, nx
    )
    assert result_code != 1


def test_emd_grid_l1_1d_direct_helper_plan_negative_values():
    a = np.array([1.0, -0.1, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 0.9])
    nx = ot.backend.NumpyBackend()
    cost, alpha, sources, targets, values, result_code = _emd_grid_l1_1d(
        a, b, True, True, nx
    )
    assert result_code != 1  # not OPTIMAL: infeasible
    assert cost == 0.0
    assert alpha is not None
    assert sources is None and targets is None and values is None


def test_emd_grid_l1_1d_direct_helper_plan_mass_mismatch():
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 0.5])
    nx = ot.backend.NumpyBackend()
    cost, alpha, sources, targets, values, result_code = _emd_grid_l1_1d(
        a, b, True, True, nx
    )
    assert result_code != 1  # not OPTIMAL: infeasible
    assert cost == 0.0
    assert alpha is not None
    assert sources is None and targets is None and values is None


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


@pytest.mark.parametrize("shape", [(8,), (3, 4), (2, 3, 2)])
def test_emd_grid_l1_gradient_strong_duality(shape):
    """The (centred) node potentials alpha, beta=-alpha are the gradient of
    cost w.r.t. A, B, and must satisfy strong duality: for the optimal
    coupling, dot(alpha, A) + dot(beta, B) == cost."""
    n = int(np.prod(shape))
    rng = np.random.RandomState(42)
    for _trial in range(5):
        a = rng.rand(n)
        a /= a.sum()
        b = rng.rand(n)
        b /= b.sum()
        A, B = a.reshape(shape), b.reshape(shape)

        cost, log = emd_grid_l1(A, B, log=True)
        alpha = log["alpha"]
        beta = log["beta"]

        np.testing.assert_allclose(beta, -alpha, atol=1e-10)
        np.testing.assert_allclose(
            np.dot(alpha, a) + np.dot(beta, b), cost, rtol=1e-6, atol=1e-8
        )
        # Equivalent, since beta = -alpha.
        np.testing.assert_allclose(np.dot(alpha, a - b), cost, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("shape", [(8,), (3, 4), (2, 3, 2)])
def test_emd_grid_l1_gradient_finite_differences(shape):
    """alpha[i] - alpha[j] must match the centered finite difference of cost
    with respect to moving mass eps from bin j to bin i."""
    n = int(np.prod(shape))
    rng = np.random.RandomState(0)
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(n)
    b /= b.sum()
    A, B = a.reshape(shape), b.reshape(shape)

    _cost, log = emd_grid_l1(A, B, log=True)
    alpha = log["alpha"]

    eps = 1e-5
    pairs = {
        (int(i), int(j))
        for i, j in zip(rng.randint(0, n, size=8), rng.randint(0, n, size=8))
        if i != j
    }
    for i, j in pairs:
        a_plus = a.copy()
        a_plus[i] += eps
        a_plus[j] -= eps
        a_minus = a.copy()
        a_minus[i] -= eps
        a_minus[j] += eps

        cost_plus = emd_grid_l1(a_plus.reshape(shape), B)
        cost_minus = emd_grid_l1(a_minus.reshape(shape), B)
        fd = (cost_plus - cost_minus) / (2 * eps)

        np.testing.assert_allclose(fd, alpha[i] - alpha[j], atol=1e-4)


def test_emd_grid_l1_1d_gradient_matches_return_plan():
    """The gradient must not depend on whether return_plan is also
    requested: both the cost-only closed form and the with-plan branch
    dispatch to the same closed-form alpha."""
    n = 7
    rng = np.random.RandomState(9)
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(n)
    b /= b.sum()

    cost, log = emd_grid_l1(a, b, log=True)
    cost_p, log_p = emd_grid_l1(a, b, return_plan=True, log=True)

    np.testing.assert_allclose(cost, cost_p, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(log["alpha"], log_p["alpha"], atol=1e-10)
    np.testing.assert_allclose(log["beta"], log_p["beta"], atol=1e-10)
    np.testing.assert_allclose(
        np.dot(log_p["alpha"], a - b), cost_p, rtol=1e-6, atol=1e-8
    )


def test_emd_grid_l1_1d_gradient_infeasible_is_zero():
    """Matches the general (C++) path: no meaningful gradient when
    infeasible, so alpha/beta are exactly zero rather than nonsense."""
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 0.5])

    _cost, log = emd_grid_l1(a, b, check_marginals=False, log=True)
    assert log["result_code"] != 1  # not OPTIMAL: infeasible
    np.testing.assert_allclose(log["alpha"], 0.0)
    np.testing.assert_allclose(log["beta"], 0.0)


def test_emd_grid_l1_1d_gradient_backends(nx):
    """The 1D gradient must also be backend-native: dtype/device matching
    the inputs, computed without a CPU round-trip."""
    n = 6
    rng = np.random.RandomState(4)
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(n)
    b /= b.sum()

    cost_np, log_np = emd_grid_l1(a, b, log=True)

    for tp in nx.__type_list__:
        ab, bb = nx.from_numpy(a, b, type_as=tp)
        cost_b, log_b = emd_grid_l1(ab, bb, log=True)
        nx.assert_same_dtype_device(tp, log_b["alpha"])
        nx.assert_same_dtype_device(tp, log_b["beta"])
        np.testing.assert_allclose(
            nx.to_numpy(log_b["alpha"]), log_np["alpha"], rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(nx.to_numpy(cost_b), cost_np, rtol=1e-6, atol=1e-8)


def test_emd_grid_l1_grad_invalid_raises():
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        emd_grid_l1(a, b, grad="bogus")


def test_emd_grid_l1_1d_grad_none_omits_alpha_and_skips_computation():
    """grad=None on a 1D grid must skip the (non-free) O(n) gradient pass
    entirely, and log must not contain alpha/beta."""
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 1.0])

    with mock.patch("ot.lp._grid._emd_grid_l1_1d", wraps=_emd_grid_l1_1d) as mocked_1d:
        cost, log = emd_grid_l1(a, b, grad=None, log=True)
    mocked_1d.assert_called_once_with(a, b, False, False, mock.ANY)
    np.testing.assert_allclose(cost, 3.0)
    assert "alpha" not in log
    assert "beta" not in log

    # The default ('envelope') must still compute and return it.
    with mock.patch("ot.lp._grid._emd_grid_l1_1d", wraps=_emd_grid_l1_1d) as mocked_1d:
        _cost, log = emd_grid_l1(a, b, log=True)
    mocked_1d.assert_called_once_with(a, b, False, True, mock.ANY)
    assert "alpha" in log
    assert "beta" in log


def test_emd_grid_l1_1d_grad_none_without_log_unaffected():
    """grad only matters when log is requested; without log, behavior and
    return value must be unchanged."""
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 1.0])

    cost_default = emd_grid_l1(a, b)
    cost_grad_none = emd_grid_l1(a, b, grad=None)
    np.testing.assert_allclose(cost_default, cost_grad_none)


@pytest.mark.parametrize("shape", [(3, 4), (2, 2, 2)])
def test_emd_grid_l1_grad_none_multid_still_has_alpha(shape):
    """For d >= 2, alpha/beta are a free byproduct of the network-simplex
    solve, so grad=None must not remove them from log."""
    n = int(np.prod(shape))
    rng = np.random.RandomState(3)
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(n)
    b /= b.sum()
    A, B = a.reshape(shape), b.reshape(shape)

    cost, log = emd_grid_l1(A, B, grad=None, log=True)
    cost_default, log_default = emd_grid_l1(A, B, log=True)

    assert "alpha" in log
    assert "beta" in log
    np.testing.assert_allclose(cost, cost_default)
    np.testing.assert_allclose(log["alpha"], log_default["alpha"])
    np.testing.assert_allclose(log["beta"], log_default["beta"])
