"""Regression coverage for opt-in full-plan Sinkhorn stopping."""

import numpy as np
import pytest

import ot


def planted(n, m):
    rng = np.random.default_rng(n * 1000 + m)
    M = np.ascontiguousarray(rng.uniform(0.0, 1.0, (n, m)))
    reg = 0.12
    kernel = np.exp(-M / reg)
    P = rng.uniform(0.5, 1, n)[:, None] * kernel * rng.uniform(0.5, 1, m)
    P /= P.sum()
    return P.sum(axis=1), P.sum(axis=0), M, reg, P


@pytest.mark.parametrize("shape", [(7, 8), (19, 100), (36, 192), (73, 67)])
def test_plan_tol_against_planted_plan(shape):
    a, b, M, reg, P = planted(*shape)
    target = 5e-9
    plan, info = ot.sinkhorn(a, b, M, reg, plan_tol=target, stopThr=1e-7, log=True)
    observed = 0.5 * np.abs(plan - P).sum()
    assert observed < target
    assert observed <= info["plan_tv_estimate"] + 2e-15
    assert info["plan_tv_estimate"] < target
    assert np.linalg.norm(plan.sum(axis=0) - b) < 1e-7
    assert np.linalg.norm(plan.sum(axis=1) - a) < 1e-7
    np.testing.assert_array_equal(
        plan, ot.sinkhorn(a, b, M, reg, plan_tol=target, stopThr=1e-7)
    )
    scaled, scaled_info = ot.sinkhorn(
        a, b, M * 64, reg * 64, plan_tol=target, stopThr=1e-7, log=True
    )
    np.testing.assert_array_equal(plan, scaled)
    assert info["niter"] == scaled_info["niter"]


def test_plan_tol_default_and_non_numpy64_unchanged():
    a, b, M, reg, _ = planted(12, 14)
    for dtype in (np.float32, np.float64):
        args = a.astype(dtype), b.astype(dtype), M.astype(dtype), reg
        baseline, history = ot.sinkhorn(*args, log=True)
        same, same_history = ot.sinkhorn(*args, log=True, plan_tol=None)
        np.testing.assert_array_equal(baseline, same)
        assert history.keys() == same_history.keys()
        assert history["niter"] == same_history["niter"]
    args = a, np.stack([b, b], axis=1), M, reg
    np.testing.assert_array_equal(ot.sinkhorn(*args), ot.sinkhorn(*args, plan_tol=None))


@pytest.mark.parametrize("portable_comparison", [False, True])
def test_plan_tol_mutated_cost_never_reuses_a_stale_kernel(
    monkeypatch, portable_comparison
):
    from ot.bregman import _sinkhorn_plan

    _sinkhorn_plan._cache.entry = None
    if portable_comparison:
        monkeypatch.setattr(_sinkhorn_plan, "_memcmp", None)
    a, b, M, reg, _ = planted(20, 45)
    old = ot.sinkhorn(a, b, M, reg, plan_tol=1e-9)
    again = ot.sinkhorn(a, b, M, reg, plan_tol=1e-9)
    np.testing.assert_array_equal(old, again)
    M[1, 1] += 0.5  # deliberately outside the three cheap cache sampling points
    fresh = ot.sinkhorn(a, b, M, reg, plan_tol=1e-9)
    exact = ot.sinkhorn(a, b, M, reg, method="sinkhorn_log", stopThr=1e-14)
    assert not np.array_equal(old, fresh)
    assert 0.5 * np.abs(fresh - exact).sum() < 1e-9
    np.testing.assert_array_equal(
        again, old
    )  # returning a plan never aliases the cached kernel


def test_plan_tol_rejects_unsupported_instead_of_implying_success():
    a, b, M, reg, _ = planted(7, 9)
    with pytest.raises(ValueError, match="only supported"):
        ot.sinkhorn(a, b, M, reg, plan_tol=1e-6, method="sinkhorn_log")
    with pytest.raises(ValueError, match="returns the transport matrix"):
        ot.sinkhorn2(a, b, M, reg, plan_tol=1e-6)
    with pytest.raises(ValueError, match="NumPy float64"):
        ot.sinkhorn(a.astype("f"), b.astype("f"), M.astype("f"), reg, plan_tol=1e-6)
    with pytest.raises(ValueError, match="finite tolerances"):
        ot.sinkhorn(a, b, M, reg, plan_tol=np.nan)
    with pytest.raises(ArithmeticError, match="numItermax"):
        ot.sinkhorn(a, b, M, reg, plan_tol=1e-12, numItermax=1)
    with pytest.raises(ValueError, match="one histogram pair"):
        ot.sinkhorn(a, np.stack([b, b], axis=1), M, reg, plan_tol=1e-6)
    a[1], a[0] = a[1] + a[0], 0.0
    with pytest.raises(ValueError, match="strictly positive probability marginals"):
        ot.sinkhorn(a, b, M, reg, plan_tol=1e-6)
    with np.errstate(all="ignore"):
        np.testing.assert_array_equal(
            ot.sinkhorn(a, b, M, reg), ot.sinkhorn(a, b, M, reg, plan_tol=None)
        )
    a, b, M, reg, _ = planted(7, 9)
    invalid = M.copy()
    invalid[1, 1] = np.nan
    with pytest.raises(ValueError, match="positive Gibbs kernel"):
        ot.sinkhorn(a, b, invalid, reg, plan_tol=1e-6)
    M[1, 1] = 1000.0
    with pytest.raises(ValueError, match="positive Gibbs kernel"):
        ot.sinkhorn(a, b, M, reg, plan_tol=1e-6)
    np.testing.assert_array_equal(
        ot.sinkhorn(a, b, M, reg), ot.sinkhorn(a, b, M, reg, plan_tol=None)
    )


def test_plan_tol_warmstart_uses_stock_log_scaling_convention():
    a, b, M, reg, P = planted(25, 100)
    _, stock = ot.sinkhorn(a, b, M, reg, stopThr=1e-12, log=True)
    warm = np.log(stock["u"]), np.log(stock["v"])
    plan, info = ot.sinkhorn(a, b, M, reg, plan_tol=1e-9, warmstart=warm, log=True)
    assert 0.5 * np.abs(plan - P).sum() < 1e-9
    assert info["niter"] == 0
    assert info["plan_tv_estimate"] < 1e-9


def test_plan_tol_does_not_take_ordinary_early_exit():
    a = np.array([2.0, 1.0]) / 3
    b = np.array([5.0, 4.0]) / 9
    M = np.array([[0.0, np.log(2)], [np.log(2), 0.0]])
    exact = np.array([[4.0, 2.0], [1.0, 2.0]]) / 9
    ordinary, ordinary_log = ot.sinkhorn(a, b, M, 1.0, stopThr=0.1, log=True)
    assert ordinary_log["niter"] == 0
    assert 0.5 * np.abs(ordinary - exact).sum() == pytest.approx(40 / 819)
    with pytest.raises(ArithmeticError, match="numItermax"):
        ot.sinkhorn(a, b, M, 1.0, stopThr=0.1, plan_tol=0.01, numItermax=1)
    plan, info = ot.sinkhorn(a, b, M, 1.0, stopThr=0.1, plan_tol=0.01, log=True)
    assert info["niter"] > 0
    assert 0.5 * np.abs(plan - exact).sum() < 0.01
    assert info["plan_tv_estimate"] == pytest.approx(0.5 * np.abs(plan - exact).sum())
    with pytest.raises(ValueError, match="only supported"):
        ot.sinkhorn(a, b, M, 1.0, method="greenkhorn", plan_tol=0.01)


def test_ordinary_public_call_uses_marginal_path_without_imposing_plan_tol(monkeypatch):
    from ot.bregman import _sinkhorn
    from ot.bregman import _sinkhorn_plan

    def unexpected(*args, **kwargs):
        pytest.fail("a supported ordinary solve dispatched to another solver")

    monkeypatch.setattr(_sinkhorn_plan, "_solve_full_plan", unexpected)
    monkeypatch.setattr(_sinkhorn, "_sinkhorn_knopp_standard", unexpected)
    a = np.array([2.0, 1.0]) / 3
    b = np.array([5.0, 4.0]) / 9
    M = np.array([[0.0, np.log(2)], [np.log(2), 0.0]])
    p = ot.sinkhorn(a, b, M, 1.0, stopThr=0.1)
    exact = np.array([[4.0, 2.0], [1.0, 2.0]]) / 9
    assert np.linalg.norm(p.sum(0) - b) < 0.1
    assert np.linalg.norm(p.sum(1) - a) < 0.1
    # An ordinary call can finish even though its TV exceeds stopThr / 4.
    assert np.abs(p - exact).sum() / 2 == pytest.approx(40 / 819)


@pytest.mark.parametrize("bad", ["zero_col", "tiny", "positive_inf", "negative", "nan"])
def test_ordinary_kernel_fallback_matches_standard_with_one_exponential(
    monkeypatch, bad
):
    import warnings
    from ot.bregman import _sinkhorn, _sinkhorn_plan

    a, b, M, reg, _ = planted(22, 31)
    if bad == "zero_col":
        M[:, 0] = 800 * reg
    else:
        M[0, 0] = {
            "tiny": 675 * reg,
            "negative": -800 * reg,
            "nan": np.nan,
            "positive_inf": np.inf,
        }[bad]
    with warnings.catch_warnings(record=True) as standard:
        warnings.simplefilter("always")
        expected = _sinkhorn._sinkhorn_knopp_standard(a, b, M, reg, numItermax=25)
    count = 0
    original = np.exp

    def counted(x, *args, **kwargs):
        nonlocal count
        if getattr(x, "shape", None) == M.shape:
            count += 1
        return original(x, *args, **kwargs)

    monkeypatch.setattr(np, "exp", counted)
    # NumPyBackend.exp delegates to np.exp on this path.
    _sinkhorn_plan._cache.entry = None
    for _ in range(3):
        before = count
        with warnings.catch_warnings(record=True) as actual:
            warnings.simplefilter("always")
            got = ot.sinkhorn(a, b, M, reg, numItermax=25)
        np.testing.assert_array_equal(got, expected)
        assert [(r.category, str(r.message)) for r in actual] == [
            (r.category, str(r.message)) for r in standard
        ]
        assert count - before == 1


def test_warm_strict_build_skips_uniform_product_and_cold_cache_reuses_it(monkeypatch):
    from ot.bregman import _sinkhorn_plan as sp

    a, b, M, reg, _ = planted(30, 77)
    sp._cache.entry = None
    fresh = []
    original = sp._Kernel.cold_columns

    def traced(self, u):
        fresh.append(self.uniform_columns is None)
        return original(self, u)

    monkeypatch.setattr(sp._Kernel, "cold_columns", traced)
    kwargs = {"warmstart": (np.zeros(len(a)), np.zeros(len(b))), "plan_tol": 1e-7}
    for _ in range(2):
        ot.sinkhorn(a, b, M, reg, **kwargs)
    assert fresh == []
    for _ in range(2):
        ot.sinkhorn(a, b, M, reg)
    assert fresh == [True, False]


def test_ordinary_unsupported_forms_and_cost_only_bypass_fast_solver(monkeypatch):
    from ot.bregman import _sinkhorn, _sinkhorn_plan

    a, b, M, reg, _ = planted(16, 21)

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported call touched the fast-path kernel cache")

    monkeypatch.setattr(_sinkhorn_plan, "_auto_kernel", unexpected)
    cases = [
        ((a.astype("f"), b.astype("f"), M.astype("f"), reg), {}),
        ((a, b, np.asfortranarray(M), reg), {}),
        ((a, np.column_stack([b, b]), M, reg), {}),
        ((a, b, M, reg), {"log": True}),
        ((a, b, M, reg), {"warmstart": (np.log(a).astype("f"), np.log(b))}),
        ((a, b, M, reg), {"numItermax": 3, "warn": False}),
    ]
    for args, kwargs in cases:
        got = ot.sinkhorn(*args, **kwargs)
        ref = _sinkhorn._sinkhorn_knopp_standard(*args, **kwargs)
        if kwargs.get("log"):
            assert got[1]["niter"] == ref[1]["niter"]
            got, ref = got[0], ref[0]
        np.testing.assert_array_equal(got, ref)
    assert ot.sinkhorn2(a, b, M, reg) == np.sum(
        _sinkhorn._sinkhorn_knopp_standard(a, b, M, reg) * M
    )


def test_default_mutated_cost_and_exhaustion_warning():
    import warnings

    from ot.bregman import _sinkhorn_plan as sp
    from ot.bregman._sinkhorn import _sinkhorn_knopp_standard

    a, b, M, reg, _ = planted(17, 61)
    sp._cache.entry = None
    one = ot.sinkhorn(a, b, M, reg)
    two = ot.sinkhorn(a, b, M.copy(), reg)
    M[2, 3] += 0.5
    changed = ot.sinkhorn(a, b, M, reg)
    np.testing.assert_array_equal(one, two)
    assert not np.array_equal(changed, two)
    sp._cache.entry = None
    np.testing.assert_array_equal(changed, ot.sinkhorn(a, b, M, reg))
    # A finite marginal solve can exhaust its budget. Match POT's last iterate
    # and warning policy, while the strict mode reports an unmet request.
    x = np.linspace(0, 1, len(a))
    y = np.linspace(0, 1, len(b))
    M = np.ascontiguousarray((x[:, None] - y) ** 2)
    args = a, b, M, 0.005
    for warn in [True, False]:
        kw = dict(numItermax=11, stopThr=1e-12, warn=warn)
        with warnings.catch_warnings(record=True) as standard:
            warnings.simplefilter("always")
            expected = _sinkhorn_knopp_standard(*args, **kw)
        with warnings.catch_warnings(record=True) as actual:
            warnings.simplefilter("always")
            result = ot.sinkhorn(*args, **kw)
        assert len(standard) == len(actual) == int(warn)
        if warn:
            assert str(actual[0].message) == str(standard[0].message)
        np.testing.assert_allclose(result, expected, rtol=3e-13, atol=1e-16)
    with pytest.raises(ArithmeticError, match="numItermax"):
        ot.sinkhorn(*args, **kw, plan_tol=1e-12)


@pytest.mark.parametrize("step", [2, 3, 4])
def test_ordinary_warm_checks_return_the_first_eligible_iterate(step):
    a, b, M, reg, _ = planted(19, 37)
    seed = (np.log(a), np.log(b))
    K = np.exp(-M / reg)
    u = np.exp(seed[0])
    errors = []
    for _ in range(step):
        v = b / (K.T @ u)
        u = a / (K @ v)
        errors.append(np.linalg.norm(v * (K.T @ u) - b))
    tol = float(np.sqrt(errors[-2] * errors[-1]))
    assert errors[-1] < tol < min(errors[:-1])
    kwargs = dict(warmstart=seed, stopThr=tol)
    P, info = ot.sinkhorn(a, b, M, reg, log=True, **kwargs)
    assert set(info) == {"err", "niter", "u", "v"}
    assert info["niter"] + 1 == step
    np.testing.assert_array_equal(P, ot.sinkhorn(a, b, M, reg, **kwargs))
    np.testing.assert_allclose(P, info["u"][:, None] * K * info["v"], rtol=1e-14)
    assert len(info["err"]) == 2
    assert info["err"][-1] == pytest.approx(np.linalg.norm(P.sum(0) - b))
    assert max(np.linalg.norm(P.sum(0) - b), np.linalg.norm(P.sum(1) - a)) < tol


@pytest.mark.parametrize(
    "kind", ["budget", "zero_kernel_column", "extreme_seed", "zero_mass"]
)
def test_ordinary_warm_warning_and_rollback(kind):
    import warnings
    from ot.bregman._sinkhorn import _sinkhorn_knopp_standard

    a, b, M, reg, _ = planted(24, 34)
    M = np.ascontiguousarray(
        (np.linspace(0.0, 1.0, len(a))[:, None] - np.linspace(0.0, 1.0, len(b))) ** 2
    )
    reg = 0.005
    seed = (np.zeros_like(a), np.zeros_like(b))
    if kind == "extreme_seed":
        seed[1][0] = 800.0
    elif kind == "zero_kernel_column":
        M[:, 0] = 1000.0
    elif kind == "zero_mass":
        a[1] += a[0]
        a[0] = 0.0
    for warn in (True, False):
        kw = dict(warmstart=seed, log=True, numItermax=12, stopThr=1e-13, warn=warn)
        with warnings.catch_warnings(record=True) as old:
            warnings.simplefilter("always")
            ref, prior = _sinkhorn_knopp_standard(a, b, M, reg, **kw)
        with warnings.catch_warnings(record=True) as new:
            warnings.simplefilter("always")
            result, history = ot.sinkhorn(a, b, M, reg, **kw)
        assert [(w.category, str(w.message)) for w in old] == [
            (w.category, str(w.message)) for w in new
        ]
        assert history["niter"] == prior["niter"]
        np.testing.assert_allclose(history["err"], prior["err"], rtol=5e-13, atol=1e-16)
        np.testing.assert_allclose(result, ref, rtol=5e-13, atol=1e-16)
