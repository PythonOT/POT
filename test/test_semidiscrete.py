# -*- coding: utf-8 -*-
"""Tests for ``ot/semidiscrete.py``.

We rely on three small toy problems whose optimal semi-dual potential is
known in closed form, and check that the solver converges close to that
optimum. All tests run across every POT backend (via the ``nx`` fixture).
"""

# License: MIT License

import numpy as np
import pytest

from ot.semidiscrete import (
    atom_weights,
    c_transform,
    ot_map,
    solve_semidiscrete,
)

N_ITER = 2_000
BATCH_SIZE = 16
TOLERANCE = 0.05


# ---------------------------------------------------------------------
# Three toy problems with known optimal potentials.
# Each builder returns numpy arrays so we can lift them onto any backend.
# ---------------------------------------------------------------------


def regular_grid_problem():
    """Uniform source on ``[0, 1]^3``, 10 target atoms regularly placed on axis 0.

    By symmetry the optimal centred potential is zero on every atom.
    """
    m, d = 10, 3
    target = np.zeros((m, d))
    target[:, 0] = (np.arange(m) + 0.5) / m
    target[:, 1:] = 0.5
    weights = np.full(m, 1.0 / m)
    optimal = np.zeros(m)
    return target, weights, optimal, 1.0, d, "uniform_cube"


def nonuniform_weights_problem():
    """Uniform source on ``[0, 1]^3``, fixed support, nonuniform weights.

    The weights were computed by Monte Carlo so that the optimal centred
    potential is exactly ``optimal_potential`` (used as a regression target).
    """
    d = 3
    target = np.array(
        [
            [0.54488318, 0.4236548, 0.64589411],
            [0.77815675, 0.87001215, 0.97861834],
            [0.11827443, 0.63992102, 0.14335329],
            [0.94466892, 0.52184832, 0.41466194],
        ]
    )
    weights = np.array([0.3806643, 0.012264, 0.5503486, 0.0567231])
    optimal_potential = np.array([0.77423369, 0.61209572, 0.94374808, 0.6818203])
    return target, weights, optimal_potential, 1.0, d, "uniform_cube"


def shifted_1d_problem():
    r"""Uniform source on ``[delta, 1 + delta]``, atoms at ``k/m`` on the line.

    For this 1D problem with uniform target weights the optimal potential
    has the closed form

    .. math::
        g^*_j = j \left( \frac{1}{2 m^2} - \frac{\delta}{m} \right),
        \qquad j = 0, \dots, m - 1.

    See Appendix E.1 in "Stochastic Optimization in Semi-Discrete Optimal
    Transport: Convergence Analysis and Minimax Rate", Genans et al.
    (NeurIPS 2025).
    """
    m, delta = 10, 0.5
    target = np.linspace(1 / m, 1.0, m).reshape(m, 1)
    weights = np.full(m, 1.0 / m)
    optimal = np.zeros(m)
    for j in range(1, m):
        optimal[j] = optimal[j - 1] + 1 / (2 * m * m) - delta / m
    return target, weights, optimal, 1.0 + delta, 1, ("shifted_1d", delta)


ALL_PROBLEMS = [
    pytest.param(regular_grid_problem, id="regular_grid"),
    pytest.param(nonuniform_weights_problem, id="nonuniform_weights"),
    pytest.param(shifted_1d_problem, id="shifted_1d"),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_sampler(kind, d, nx, target):
    """Return a backend-aware sampler with a fixed numpy RNG inside."""
    rng = np.random.default_rng(0)
    if kind == "uniform_cube":

        def sampler(b):
            return nx.from_numpy(rng.random((b, d)), type_as=target)

    else:
        # ("shifted_1d", delta)
        delta = kind[1]

        def sampler(b):
            return nx.from_numpy(delta + rng.random((b, d)), type_as=target)

    return sampler


def centered_l2_error(estimated, reference):
    """L2 distance between the centred (mean-zero) potentials.

    The semi-dual is invariant to a global additive constant, so we factor that out.
    """
    estimated = estimated - estimated.mean()
    reference = reference - reference.mean()
    return float(np.linalg.norm(estimated - reference))


def lift(nx, target_np, weights_np):
    """Move numpy ``target`` and ``weights`` arrays onto backend ``nx``."""
    target = nx.from_numpy(target_np)
    weights = nx.from_numpy(weights_np, type_as=target)
    return target, weights


# ---------------------------------------------------------------------
# Convergence on every backend
# ---------------------------------------------------------------------


@pytest.mark.parametrize("build_problem", ALL_PROBLEMS)
def test_solve_converges(nx, build_problem):
    """Plain SGD (no decreasing reg) reaches the optimum on every toy problem."""
    target_np, weights_np, optimal, max_cost, d, kind = build_problem()
    target, weights = lift(nx, target_np, weights_np)
    sampler = make_sampler(kind, d, nx, target)

    g = solve_semidiscrete(
        target,
        sampler,
        target_weights=weights,
        n_iter=N_ITER,
        batch_size=BATCH_SIZE,
        decreasing_reg=False,
        proj_bound=max_cost,
    )
    err = centered_l2_error(nx.to_numpy(g), optimal)
    assert err < TOLERANCE, f"err={err:.4f}"


@pytest.mark.parametrize("build_problem", ALL_PROBLEMS)
def test_drag_converges(nx, build_problem):
    """DRAG (decreasing entropic reg) reaches the optimum on every toy problem."""
    target_np, weights_np, optimal, max_cost, d, kind = build_problem()
    target, weights = lift(nx, target_np, weights_np)
    sampler = make_sampler(kind, d, nx, target)

    g = solve_semidiscrete(
        target,
        sampler,
        target_weights=weights,
        n_iter=N_ITER,
        batch_size=BATCH_SIZE,
        decreasing_reg=True,
        proj_bound=max_cost,
    )
    err = centered_l2_error(nx.to_numpy(g), optimal)
    assert err < TOLERANCE, f"err={err:.4f}"


# ---------------------------------------------------------------------
# Entropic regime
# ---------------------------------------------------------------------


def test_entropic_solver_runs(nx):
    """In the entropic regime, the solver and ``c_transform`` produce finite values."""
    target_np, weights_np, _, max_cost, d, kind = regular_grid_problem()
    target, weights = lift(nx, target_np, weights_np)
    sampler = make_sampler(kind, d, nx, target)

    g = solve_semidiscrete(
        target,
        sampler,
        target_weights=weights,
        reg=0.05,
        n_iter=N_ITER,
        batch_size=BATCH_SIZE,
        proj_bound=max_cost,
    )
    samples = sampler(64)
    phi = c_transform(target, samples, g, target_weights=weights, reg=0.05)
    assert np.isfinite(nx.to_numpy(g)).all()
    assert np.isfinite(nx.to_numpy(phi)).all()


# ---------------------------------------------------------------------
# Custom cost
# ---------------------------------------------------------------------


def test_custom_quadratic_cost_matches_default(nx):
    """A user-supplied quadratic cost reaches the same optimum as the default."""
    target_np, weights_np, optimal, max_cost, d, kind = regular_grid_problem()
    target, weights = lift(nx, target_np, weights_np)
    sampler = make_sampler(kind, d, nx, target)

    def cost(x, y):
        diff = x[:, None, :] - y[None, :, :]
        return 0.5 * nx.sum(diff**2, axis=2)

    g = solve_semidiscrete(
        target,
        sampler,
        target_weights=weights,
        cost=cost,
        n_iter=N_ITER,
        batch_size=BATCH_SIZE,
        proj_bound=max_cost,
    )
    err = centered_l2_error(nx.to_numpy(g), optimal)
    assert err < TOLERANCE, f"err={err:.4f}"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


@pytest.mark.parametrize("reg", [0.0, 0.1])
def test_atom_weights_are_row_stochastic(nx, reg):
    """``atom_weights`` returns nonnegative weights that sum to 1 per row."""
    target_np, weights_np, _, _, d, kind = nonuniform_weights_problem()
    target, weights = lift(nx, target_np, weights_np)
    sampler = make_sampler(kind, d, nx, target)
    samples = sampler(32)
    g = nx.zeros((target_np.shape[0],), type_as=target)
    w = atom_weights(target, samples, g, target_weights=weights, reg=reg)
    w_np = nx.to_numpy(w)
    assert w_np.shape == (32, target_np.shape[0])
    assert (w_np >= 0).all()
    np.testing.assert_allclose(w_np.sum(axis=1), 1.0, atol=1e-10)


def test_ot_map_shape_and_finiteness(nx):
    """``ot_map`` returns finite values with the source-sample shape."""
    target_np, weights_np, _, _, d, kind = regular_grid_problem()
    target, weights = lift(nx, target_np, weights_np)
    sampler = make_sampler(kind, d, nx, target)
    samples = sampler(16)
    g = nx.zeros((target_np.shape[0],), type_as=target)
    transported = ot_map(target, samples, g, target_weights=weights)
    transported_np = nx.to_numpy(transported)
    samples_np = nx.to_numpy(samples)
    assert transported_np.shape == samples_np.shape
    assert np.isfinite(transported_np).all()


def test_c_transform_minimum_for_zero_potential(nx):
    """At ``g = 0``, ``phi_g(x) = -max_j(-c(x, y_j)) = min_j c(x, y_j)``."""
    target_np, weights_np, _, _, d, kind = regular_grid_problem()
    target, weights = lift(nx, target_np, weights_np)
    sampler = make_sampler(kind, d, nx, target)
    samples = sampler(8)
    g = nx.zeros((target_np.shape[0],), type_as=target)
    phi = c_transform(target, samples, g, target_weights=weights)
    samples_np = nx.to_numpy(samples)
    target_np = nx.to_numpy(target)
    diff = samples_np[:, None, :] - target_np[None, :, :]
    expected = 0.5 * (diff**2).sum(axis=2).min(axis=1)
    np.testing.assert_allclose(nx.to_numpy(phi), expected, atol=1e-10)


# ---------------------------------------------------------------------
# Solver options
# ---------------------------------------------------------------------


def test_warm_start_converges(nx):
    """Splitting one run into two warm-started halves still converges."""
    target_np, weights_np, optimal, max_cost, d, kind = regular_grid_problem()
    target, weights = lift(nx, target_np, weights_np)
    sampler = make_sampler(kind, d, nx, target)

    half = solve_semidiscrete(
        target,
        sampler,
        target_weights=weights,
        n_iter=N_ITER // 2,
        batch_size=BATCH_SIZE,
        proj_bound=max_cost,
    )
    g = solve_semidiscrete(
        target,
        sampler,
        target_weights=weights,
        n_iter=N_ITER // 2,
        batch_size=BATCH_SIZE,
        init_potential=half,
        proj_bound=max_cost,
    )
    err = centered_l2_error(nx.to_numpy(g), optimal)
    assert err < TOLERANCE, f"err={err:.4f}"


def test_init_potential_is_not_mutated(nx):
    """The ``init_potential`` array passed by the caller is left intact."""
    target_np, weights_np, _, max_cost, d, kind = regular_grid_problem()
    target, weights = lift(nx, target_np, weights_np)
    sampler = make_sampler(kind, d, nx, target)

    init_np = np.full(target_np.shape[0], 0.5)
    init = nx.from_numpy(init_np, type_as=target)
    snapshot = nx.to_numpy(init).copy()

    solve_semidiscrete(
        target,
        sampler,
        target_weights=weights,
        init_potential=init,
        n_iter=10,
        batch_size=4,
        proj_bound=max_cost,
    )
    np.testing.assert_array_equal(nx.to_numpy(init), snapshot)


def test_projection_clamps_last_iterate(nx):
    """With ``proj_bound=b``, every coordinate of the last iterate lies in ``[-b, b]``."""
    target_np, weights_np, _, _, d, kind = regular_grid_problem()
    target, weights = lift(nx, target_np, weights_np)
    sampler = make_sampler(kind, d, nx, target)

    bound = 0.05
    _, info = solve_semidiscrete(
        target,
        sampler,
        target_weights=weights,
        n_iter=300,
        batch_size=4,
        proj_bound=bound,
        log=True,
    )
    last = nx.to_numpy(info["last_potential"])
    assert np.abs(last).max() <= bound + 1e-10


def test_polyak_average_off_returns_last_iterate(nx):
    """With ``polyak_average=False`` the returned potential equals the last iterate."""
    target_np, weights_np, _, max_cost, d, kind = regular_grid_problem()
    target, weights = lift(nx, target_np, weights_np)
    sampler = make_sampler(kind, d, nx, target)

    final, info = solve_semidiscrete(
        target,
        sampler,
        target_weights=weights,
        n_iter=20,
        batch_size=4,
        polyak_average=False,
        proj_bound=max_cost,
        log=True,
    )
    np.testing.assert_array_equal(
        nx.to_numpy(final), nx.to_numpy(info["last_potential"])
    )


def test_log_returns_metadata(nx):
    """``log=True`` returns an info dict with the expected fields."""
    target_np, weights_np, _, max_cost, d, kind = regular_grid_problem()
    target, weights = lift(nx, target_np, weights_np)
    sampler = make_sampler(kind, d, nx, target)

    g, info = solve_semidiscrete(
        target,
        sampler,
        target_weights=weights,
        n_iter=50,
        batch_size=4,
        proj_bound=max_cost,
        log=True,
    )
    assert nx.to_numpy(g).shape == (target_np.shape[0],)
    assert info["n_iter"] == 50
    assert info["batch_size"] == 4
    assert info["proj_bound"] == max_cost
