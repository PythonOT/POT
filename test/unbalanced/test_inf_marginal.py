"""Contract tests for infinite marginal relaxation in :any:`ot.solve`.

``unbalanced=float("inf")`` is documented as the balanced limit, and
``unbalanced=(float("inf"), rho)`` / ``(rho, float("inf"))`` as the
semi-relaxed case.

A two-sided infinite relaxation is the hard-constraint limit of the *same*
objective. That limit is only equal to the problem solved by ``unbalanced=None``
when the balanced branch minimizes the same objective. These tests therefore
compare against an **independent analytic oracle** (a bounded scalar search on
the feasible set), not against another ``ot.solve`` call, and they require the
combinations whose limit is *not* representable to fail loudly instead of
returning a solution to a different optimization problem.

License: MIT License
"""

import itertools

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import ot

INF = float("inf")

# 2x2 symmetric fixtures: with a = b = (0.5, 0.5) the hard-constraint feasible
# set is one dimensional, P = [[x, .5-x], [.5-x, x]] with x in [0, .5], which
# makes an exact, solver-independent oracle possible.
A2 = np.array([0.5, 0.5])
B2 = np.array([0.5, 0.5])
M2 = np.array([[0.0, 1.0], [1.0, 0.0]])
M2_ZERO = np.zeros((2, 2))
C2_DEFAULT = np.outer(A2, B2)
C2_CUSTOM = np.array([[0.45, 0.05], [0.05, 0.45]])


def _data(n=9, m=7, seed=3):
    """Return a deterministic, non-uniform OT problem as Numpy arrays."""
    rng = np.random.RandomState(seed)
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(m)
    b /= b.sum()
    M = rng.rand(n, m)
    return a, b, M


def _gkl(x, y):
    """Generalized KL divergence, matching the ``mass=True`` convention."""
    return np.sum(x * np.log(np.maximum(x, 1e-300) / y) - x + y)


def _sym_plan(x):
    return np.array([[x, 0.5 - x], [0.5 - x, x]])


def _objective(x, reg, reg_type, c, M):
    """The documented objective restricted to the hard-constraint feasible set."""
    P = _sym_plan(x)
    value = float(np.sum(P * M))
    if reg > 0:
        if reg_type == "kl":
            value += reg * _gkl(P, c)
        elif reg_type == "l2":
            value += reg * 0.5 * np.sum((P - c) ** 2)
        elif reg_type == "entropy":
            value += reg * (np.sum(P * np.log(np.maximum(P, 1e-300))) - P.sum())
    return value


def _hard_limit_oracle(reg, reg_type, c, M=M2):
    """Minimize the original objective over the hard-constraint feasible set."""
    res = minimize_scalar(
        _objective,
        bounds=(0.0, 0.5),
        args=(reg, reg_type, c, M),
        method="bounded",
        options={"xatol": 1e-14},
    )
    return float(res.x), float(res.fun)


@pytest.mark.parametrize("unbalanced", [INF, (INF, INF), [INF, INF]])
def test_balanced_limit_exact(nx, unbalanced):
    """``unbalanced=inf`` with ``reg=0`` must reproduce exact balanced OT."""
    a, b, M = _data()
    a_nx, b_nx, M_nx = nx.from_numpy(a, b, M)

    ref = ot.solve(M_nx, a_nx, b_nx, reg=0)
    res = ot.solve(M_nx, a_nx, b_nx, reg=0, unbalanced=unbalanced)

    plan = nx.to_numpy(res.plan)
    assert np.isfinite(plan).all()
    assert np.isfinite(float(nx.to_numpy(res.value)))
    np.testing.assert_allclose(
        float(nx.to_numpy(res.value)), float(nx.to_numpy(ref.value)), rtol=1e-12
    )
    np.testing.assert_allclose(plan, nx.to_numpy(ref.plan), atol=1e-12)
    # both marginals are hard constraints in the balanced limit
    np.testing.assert_allclose(plan.sum(1), a, atol=1e-10)
    np.testing.assert_allclose(plan.sum(0), b, atol=1e-10)


def test_balanced_limit_exact_independent_reference(nx):
    """Check the balanced limit against a brute-force optimum, not against POT.

    With uniform weights on a square problem, the exact OT optimum is attained
    at a permutation, so the value can be computed independently of any OT
    solver. This keeps the regression test from re-deriving POT's own output.
    """
    # shifted so that the optimum is not trivially 0 (a zero optimum would make
    # the assertion vacuous against a solver that always returns 0)
    M = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 4.0, 1.0]])
    a = np.full(3, 1.0 / 3.0)
    b = np.full(3, 1.0 / 3.0)
    best = min(
        sum(M[i, perm[i]] for i in range(3)) / 3.0
        for perm in itertools.permutations(range(3))
    )

    a_nx, b_nx, M_nx = nx.from_numpy(a, b, M)
    res = ot.solve(M_nx, a_nx, b_nx, reg=0, unbalanced=INF)

    np.testing.assert_allclose(float(nx.to_numpy(res.value)), best, atol=1e-12)
    np.testing.assert_allclose(np.asarray(nx.to_numpy(res.plan)).sum(), 1.0, atol=1e-12)


@pytest.mark.parametrize("unbalanced_type", ["KL", "L2", "TV"])
def test_balanced_limit_kl_matches_hard_limit_oracle(nx, unbalanced_type):
    """KL regularization with the default reference: the limit is the oracle.

    ``reg_type="kl"`` with ``c = a b^T`` makes the balanced problem identical to
    the hard-constraint limit, so the dispatched result must match the analytic
    oracle exactly - including the value, which a value-only guard would miss.
    """
    x_star, v_star = _hard_limit_oracle(1.0, "kl", C2_DEFAULT)
    a_nx, b_nx, M_nx = nx.from_numpy(A2, B2, M2)

    res = ot.solve(
        M_nx, a_nx, b_nx, reg=1.0, reg_type="kl",
        unbalanced_type=unbalanced_type, unbalanced=INF,
    )
    plan = nx.to_numpy(res.plan)

    assert np.isfinite(plan).all()
    np.testing.assert_allclose(plan[0, 0], x_star, atol=1e-6)
    np.testing.assert_allclose(float(nx.to_numpy(res.value)), v_star, rtol=1e-8)
    np.testing.assert_allclose(plan.sum(1), A2, atol=1e-6)
    np.testing.assert_allclose(plan.sum(0), B2, atol=1e-6)


def test_kl_custom_reference_is_honoured(nx):
    """A custom ``c`` must survive the hard-constraint limit.

    With ``M = 0`` and ``c`` already feasible, the unique minimizer is ``c``
    itself (KL >= 0 with equality iff P = c). The balanced KL branch ignores
    ``c`` and would return ``a b^T`` instead, so this test would fail if the
    limit were dispatched there.
    """
    x_star, v_star = _hard_limit_oracle(1.0, "kl", C2_CUSTOM, M2_ZERO)
    a_nx, b_nx, M_nx = nx.from_numpy(A2, B2, M2_ZERO)
    c_nx = nx.from_numpy(C2_CUSTOM)

    res = ot.solve(
        M_nx, a_nx, b_nx, reg=1.0, reg_type="kl",
        unbalanced_type="KL", c=c_nx, unbalanced=INF,
    )
    plan = nx.to_numpy(res.plan)

    np.testing.assert_allclose(plan, C2_CUSTOM, atol=1e-6)
    np.testing.assert_allclose(plan[0, 0], x_star, atol=1e-6)
    np.testing.assert_allclose(float(nx.to_numpy(res.value)), v_star, atol=1e-8)


@pytest.mark.parametrize(
    "reg_type,unbalanced_type,custom_c",
    [
        # L2, reg=1 on this fixture: the balanced branch *solves*
        # reg/2 * ||P||^2 with no reference measure and *reports*
        # <P,M> + reg * ||P||^2, while the UOT hard limit is
        # <P,M> + reg/2 * ||P - c||^2. Oracle: x*=0.5, V*=0.125, but the
        # balanced call returns V=0.5. The plans coincide only because a and b
        # are uniform here; with a=(0.7,0.3), b=(0.6,0.4), M=0, reg=1 they
        # differ (0.42 vs 0.40). At reg=2 the same pair is x*=0.5, 0.25 vs 1.0.
        ("l2", "KL", False),
        ("l2", "L2", False),
        ("l2", "TV", False),
        ("l2", "KL", True),
        # entropy, reg=1: same argmin, but the balanced value omits -sum(P), so
        # the reported value is off by exactly `reg` (oracle V*=-2.0064, the
        # balanced call returns -1.0064).
        ("entropy", "KL", False),
        ("entropy", "L2", False),
        ("entropy", "KL", True),
        ("l2", "TV", True),
        # KL with a custom c: the limit is c, the balanced branch would drop it.
        ("kl", "L2", True),
        ("kl", "TV", True),
    ],
)
def test_hard_limit_rejects_different_objective(nx, reg_type, unbalanced_type, custom_c):
    """Combinations whose limit is not the balanced problem must raise.

    Silently returning the balanced solution here would solve a different
    optimization problem (wrong reference measure, coefficient or value
    convention), which is exactly the bug these tests lock down.
    """
    a_nx, b_nx, M_nx = nx.from_numpy(A2, B2, M2)
    kwargs = dict(
        reg=1.0, reg_type=reg_type, unbalanced_type=unbalanced_type, unbalanced=INF
    )
    if custom_c:
        kwargs["c"] = nx.from_numpy(C2_CUSTOM)

    with pytest.raises(NotImplementedError, match="hard-constraint limit"):
        ot.solve(M_nx, a_nx, b_nx, **kwargs)


def test_hard_limit_custom_regularizer_matches_balanced(nx):
    """A user supplied (f, df) pair is preserved by the balanced CG solver.

    This is a consistency check between the two paths (both call the same
    ``cg``), not an independent oracle: it guards against the limit being routed
    somewhere that drops ``f``.
    """
    if nx.__name__ != "numpy":
        pytest.skip(
            "ot.solve with a custom (f, df) reg_type is numpy-only: the CG "
            "solver passes backend arrays straight to the user callables."
        )

    def f(P):
        return np.sum(P * np.log(np.maximum(P, 1e-300)))

    def df(P):
        return np.log(np.maximum(P, 1e-300)) + 1

    res = ot.solve(
        nx.from_numpy(M2), nx.from_numpy(A2), nx.from_numpy(B2),
        reg=1.0, reg_type=(f, df), unbalanced=INF,
    )
    ref = ot.solve(
        nx.from_numpy(M2), nx.from_numpy(A2), nx.from_numpy(B2),
        reg=1.0, reg_type=(f, df),
    )
    np.testing.assert_allclose(
        float(nx.to_numpy(res.value)), float(nx.to_numpy(ref.value)), rtol=1e-10
    )
    np.testing.assert_allclose(
        nx.to_numpy(res.plan), nx.to_numpy(ref.plan), atol=1e-10
    )


@pytest.mark.parametrize("hard_side", ["left", "right"])
def test_semi_relaxed_sinkhorn(nx, hard_side):
    """The supported semi-relaxed path must honour the hard side only."""
    a, b, M = _data()
    a_nx, b_nx, M_nx = nx.from_numpy(a, b, M)
    reg = 0.5
    rho = 1.0
    unbalanced = (INF, rho) if hard_side == "left" else (rho, INF)

    res = ot.solve(M_nx, a_nx, b_nx, reg=reg, unbalanced=unbalanced)
    plan = nx.to_numpy(res.plan)
    value = float(nx.to_numpy(res.value))

    assert np.isfinite(plan).all()
    assert np.isfinite(value)

    # the hard constraint is enforced, the relaxed side is *not* forced.
    # 1e-5 is the order of the Sinkhorn stopping criterion used by ot.solve.
    if hard_side == "left":
        np.testing.assert_allclose(plan.sum(1), a, atol=1e-5)
        assert np.abs(plan.sum(0) - b).max() > 1e-4
        expected = np.sum(plan * M) + reg * _gkl(plan, np.outer(a, b))
        expected += rho * _gkl(plan.sum(0), b)
    else:
        np.testing.assert_allclose(plan.sum(0), b, atol=1e-5)
        assert np.abs(plan.sum(1) - a).max() > 1e-4
        expected = np.sum(plan * M) + reg * _gkl(plan, np.outer(a, b))
        expected += rho * _gkl(plan.sum(1), a)

    # independent recomputation of the entropic semi-relaxed objective
    np.testing.assert_allclose(value, expected, rtol=1e-8)


@pytest.mark.parametrize(
    "kwargs",
    [
        # exact semi-relaxed OT has no solver: reject instead of returning NaN
        dict(reg=0, unbalanced=(INF, 1.0)),
        dict(reg=0, unbalanced=(1.0, INF)),
        dict(reg=0, unbalanced=(INF, 1.0), unbalanced_type="L2"),
        dict(reg=0, unbalanced=(1.0, INF), unbalanced_type="L2"),
        dict(reg=0, unbalanced=(INF, 1.0), unbalanced_type="TV"),
        # the L-BFGSB path would silently solve the balanced problem
        dict(reg=0.5, reg_type="l2", unbalanced=(INF, 1.0)),
        dict(reg=0.5, reg_type="l2", unbalanced=(1.0, INF)),
        dict(reg=0.5, unbalanced=(INF, 1.0), unbalanced_type="L2"),
        dict(reg=0.5, unbalanced=(INF, 1.0), unbalanced_type="TV"),
        dict(reg=0.5, reg_type="entropy", unbalanced=(INF, 1.0)),
    ],
)
def test_unsupported_semi_relaxed_raises(nx, kwargs):
    """Unsupported semi-relaxed combinations must fail loudly, not return NaN."""
    a, b, M = _data()
    a_nx, b_nx, M_nx = nx.from_numpy(a, b, M)

    with pytest.raises(NotImplementedError, match="semi-relaxed|L-BFGSB|hard-constraint"):
        ot.solve(M_nx, a_nx, b_nx, **kwargs)


def test_lbfgsb_infinite_relaxation_not_silently_balanced(nx):
    """Regression: the L-BFGSB path used to return a balanced plan with NaN value."""
    a, b, M = _data()
    a_nx, b_nx, M_nx = nx.from_numpy(a, b, M)

    # a one-sided hard constraint must be rejected ...
    with pytest.raises(NotImplementedError):
        ot.solve(M_nx, a_nx, b_nx, reg=0.5, reg_type="l2", unbalanced=(INF, 1.0))

    # ... while the two-sided limit stays available for KL with the default c
    res = ot.solve(M_nx, a_nx, b_nx, reg=0.5, reg_type="kl", unbalanced=INF)
    plan = nx.to_numpy(res.plan)
    assert np.isfinite(float(nx.to_numpy(res.value)))
    # the balanced limit enforces both marginals
    np.testing.assert_allclose(plan.sum(1), a, atol=1e-6)
    np.testing.assert_allclose(plan.sum(0), b, atol=1e-6)


def test_finite_relaxation_unchanged(nx):
    """Finite relaxations must keep using the unbalanced solvers unchanged."""
    a, b, M = _data()
    a_nx, b_nx, M_nx = nx.from_numpy(a, b, M)

    res = ot.solve(M_nx, a_nx, b_nx, reg=0.5, unbalanced=1.0)
    plan = nx.to_numpy(res.plan)
    assert np.isfinite(plan).all()
    # unbalanced OT does not satisfy the marginals exactly
    assert np.abs(plan.sum(1) - a).max() > 1e-6
    np.testing.assert_allclose(
        float(nx.to_numpy(res.value)),
        np.sum(plan * M) + 0.5 * _gkl(plan, np.outer(a, b)) + _gkl(plan.sum(1), a)
        + _gkl(plan.sum(0), b),
        rtol=1e-8,
    )
