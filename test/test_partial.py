"""Tests for module partial"""

# Author:
#         Laetitia Chapel <laetitia.chapel@irisa.fr>
#
# License: MIT License

import numpy as np
import scipy as sp
import ot
from ot.backend import to_numpy, torch
import pytest


def test_raise_errors():
    n_samples = 20  # nb samples (gaussian)
    n_noise = 20  # nb of samples (noise)

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 2]])

    rng = np.random.RandomState(42)
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=rng)
    xs = np.append(xs, (rng.rand(n_noise, 2) + 1) * 4).reshape((-1, 2))
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=rng)
    xt = np.append(xt, (rng.rand(n_noise, 2) + 1) * -3).reshape((-1, 2))

    M = ot.dist(xs, xt)

    p = ot.unif(n_samples + n_noise)
    q = ot.unif(n_samples + n_noise)

    with pytest.raises(ValueError):
        ot.partial.partial_wasserstein_lagrange(p + 1, q, M, 1, log=True)

    with pytest.raises(ValueError):
        ot.partial.partial_wasserstein(p, q, M, m=2, log=True)

    with pytest.raises(ValueError):
        ot.partial.partial_wasserstein(p, q, M, m=-1, log=True)

    with pytest.raises(ValueError):
        ot.partial.entropic_partial_wasserstein(p, q, M, reg=1, m=2, log=True)

    with pytest.raises(ValueError):
        ot.partial.entropic_partial_wasserstein(p, q, M, reg=1, m=-1, log=True)

    with pytest.raises(ValueError):
        ot.partial.partial_gromov_wasserstein(M, M, p, q, m=2, log=True)

    with pytest.raises(ValueError):
        ot.partial.partial_gromov_wasserstein(M, M, p, q, m=-1, log=True)

    with pytest.raises(ValueError):
        ot.partial.entropic_partial_gromov_wasserstein(M, M, p, q, reg=1, m=2, log=True)

    with pytest.raises(ValueError):
        ot.partial.entropic_partial_gromov_wasserstein(
            M, M, p, q, reg=1, m=-1, log=True
        )

    with pytest.raises(AssertionError):
        xs_2d = rng.randn(n_samples, 2)
        xt_2d = rng.randn(n_samples, 2)
        ot.partial.partial_wasserstein_1d(xs_2d, xt_2d)


def test_partial_wasserstein_lagrange():
    n_samples = 20  # nb samples (gaussian)
    n_noise = 20  # nb of samples (noise)

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 2]])

    rng = np.random.RandomState(42)
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=rng)
    xs = np.append(xs, (rng.rand(n_noise, 2) + 1) * 4).reshape((-1, 2))
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=rng)
    xt = np.append(xt, (rng.rand(n_noise, 2) + 1) * -3).reshape((-1, 2))

    M = ot.dist(xs, xt)

    p = ot.unif(n_samples + n_noise)
    q = ot.unif(n_samples + n_noise)

    w0, log0 = ot.partial.partial_wasserstein_lagrange(p, q, M, 1, log=True)

    w0, log0 = ot.partial.partial_wasserstein_lagrange(p, q, M, 100, log=True)


def test_partial_wasserstein(nx):
    n_samples = 20  # nb samples (gaussian)
    n_noise = 20  # nb of samples (noise)

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 2]])

    rng = np.random.RandomState(42)
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=rng)
    xs = np.append(xs, (rng.rand(n_noise, 2) + 1) * 4).reshape((-1, 2))
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=rng)
    xt = np.append(xt, (rng.rand(n_noise, 2) + 1) * -3).reshape((-1, 2))

    M = ot.dist(xs, xt)

    p = ot.unif(n_samples + n_noise)
    q = ot.unif(n_samples + n_noise)

    m = 0.5

    p, q, M = nx.from_numpy(p, q, M)

    w0, log0 = ot.partial.partial_wasserstein(p, q, M, m=m, log=True)
    w, log = ot.partial.entropic_partial_wasserstein(
        p, q, M, reg=1, m=m, log=True, verbose=True
    )

    # check constraints
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=1) - p) <= 1e-5, [True] * len(p))
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=0) - q) <= 1e-5, [True] * len(q))
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=1) - p) <= 1e-5, [True] * len(p))
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=0) - q) <= 1e-5, [True] * len(q))

    # check transported mass
    np.testing.assert_allclose(np.sum(to_numpy(w0)), m, atol=1e-04)
    np.testing.assert_allclose(np.sum(to_numpy(w)), m, atol=1e-04)

    w0, log0 = ot.partial.partial_wasserstein2(p, q, M, m=m, log=True)
    w0_val = ot.partial.partial_wasserstein2(p, q, M, m=m, log=False)

    G = log0["T"]

    np.testing.assert_allclose(w0, w0_val, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_equal(to_numpy(nx.sum(G, axis=1) - p) <= 1e-5, [True] * len(p))
    np.testing.assert_equal(to_numpy(nx.sum(G, axis=0) - q) <= 1e-5, [True] * len(q))
    np.testing.assert_allclose(np.sum(to_numpy(G)), m, atol=1e-04)

    empty_array = nx.zeros(0, type_as=M)
    w = ot.partial.partial_wasserstein(empty_array, empty_array, M=M, m=None)

    # check constraints
    np.testing.assert_equal(to_numpy(nx.sum(w, axis=1) - p) <= 1e-5, [True] * len(p))
    np.testing.assert_equal(to_numpy(nx.sum(w, axis=0) - q) <= 1e-5, [True] * len(q))
    np.testing.assert_equal(to_numpy(nx.sum(w, axis=1) - p) <= 1e-5, [True] * len(p))
    np.testing.assert_equal(to_numpy(nx.sum(w, axis=0) - q) <= 1e-5, [True] * len(q))

    # check transported mass
    np.testing.assert_allclose(np.sum(to_numpy(w)), 1, atol=1e-04)

    w0 = ot.partial.entropic_partial_wasserstein(
        empty_array, empty_array, M=M, reg=10, m=None
    )

    # check constraints
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=1) - p) <= 1e-5, [True] * len(p))
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=0) - q) <= 1e-5, [True] * len(q))
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=1) - p) <= 1e-5, [True] * len(p))
    np.testing.assert_equal(to_numpy(nx.sum(w0, axis=0) - q) <= 1e-5, [True] * len(q))

    # check transported mass
    np.testing.assert_allclose(np.sum(to_numpy(w0)), 1, atol=1e-04)


def test_partial_wasserstein2_gradient():
    if torch:
        n_samples = 40

        mu = np.array([0, 0])
        cov = np.array([[1, 0], [0, 2]])

        xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov)
        xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov)

        M = torch.tensor(ot.dist(xs, xt), requires_grad=True, dtype=torch.float64)

        p = torch.tensor(ot.unif(n_samples), dtype=torch.float64)
        q = torch.tensor(ot.unif(n_samples), dtype=torch.float64)

        m = 0.5

        w, log = ot.partial.partial_wasserstein2(p, q, M, m=m, log=True)

        w.backward()

        assert M.grad is not None
        assert M.grad.shape == M.shape


def test_entropic_partial_wasserstein_gradient():
    if torch:
        n_samples = 40

        mu = np.array([0, 0])
        cov = np.array([[1, 0], [0, 2]])

        xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov)
        xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov)

        M = torch.tensor(ot.dist(xs, xt), requires_grad=True, dtype=torch.float64)

        p = torch.tensor(ot.unif(n_samples), requires_grad=True, dtype=torch.float64)
        q = torch.tensor(ot.unif(n_samples), requires_grad=True, dtype=torch.float64)

        m = 0.5
        reg = 1

        _, log = ot.partial.entropic_partial_wasserstein(
            p, q, M, m=m, reg=reg, log=True
        )

        log["partial_w_dist"].backward()

        assert M.grad is not None
        assert p.grad is not None
        assert q.grad is not None
        assert M.grad.shape == M.shape
        assert p.grad.shape == p.shape
        assert q.grad.shape == q.shape


def test_partial_gromov_wasserstein():
    rng = np.random.RandomState(42)
    n_samples = 20  # nb samples
    n_noise = 10  # nb of samples (noise)

    p = ot.unif(n_samples + n_noise)
    q = ot.unif(n_samples + n_noise)

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([0, 0, 0])
    cov_t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=rng)
    xs = np.concatenate((xs, ((rng.rand(n_noise, 2) + 1) * 4)), axis=0)
    P = sp.linalg.sqrtm(cov_t)
    xt = rng.randn(n_samples, 3).dot(P) + mu_t
    xt = np.concatenate((xt, ((rng.rand(n_noise, 3) + 1) * 10)), axis=0)
    xt2 = xs[::-1].copy()

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    C3 = ot.dist(xt2, xt2)

    m = 2 / 3
    res0, log0 = ot.partial.partial_gromov_wasserstein(
        C1, C3, p, q, m=m, log=True, verbose=True
    )
    np.testing.assert_allclose(res0, 0, atol=1e-1, rtol=1e-1)

    C1 = sp.spatial.distance.cdist(xs, xs)
    C2 = sp.spatial.distance.cdist(xt, xt)

    m = 1
    res0, log0 = ot.partial.partial_gromov_wasserstein(C1, C2, p, q, m=m, log=True)
    G = ot.gromov.gromov_wasserstein(C1, C2, p, q, "square_loss")
    np.testing.assert_allclose(G, res0, atol=1e-04)

    res, log = ot.partial.entropic_partial_gromov_wasserstein(
        C1, C2, p, q, 10, m=m, log=True
    )
    G = ot.gromov.entropic_gromov_wasserstein(C1, C2, p, q, "square_loss", epsilon=10)
    np.testing.assert_allclose(G, res, atol=1e-02)

    w0, log0 = ot.partial.partial_gromov_wasserstein2(C1, C2, p, q, m=m, log=True)
    w0_val = ot.partial.partial_gromov_wasserstein2(C1, C2, p, q, m=m, log=False)
    G = log0["T"]
    np.testing.assert_allclose(w0, w0_val, atol=1e-1, rtol=1e-1)

    m = 2 / 3
    res0, log0 = ot.partial.partial_gromov_wasserstein(C1, C2, p, q, m=m, log=True)
    res, log = ot.partial.entropic_partial_gromov_wasserstein(
        C1, C2, p, q, 100, m=m, log=True
    )

    # check constraints
    np.testing.assert_equal(
        res0.sum(1) <= p, [True] * len(p)
    )  # cf convergence wasserstein
    np.testing.assert_equal(
        res0.sum(0) <= q, [True] * len(q)
    )  # cf convergence wasserstein
    np.testing.assert_allclose(np.sum(res0), m, atol=1e-04)

    np.testing.assert_equal(
        res.sum(1) <= p, [True] * len(p)
    )  # cf convergence wasserstein
    np.testing.assert_equal(
        res.sum(0) <= q, [True] * len(q)
    )  # cf convergence wasserstein
    np.testing.assert_allclose(np.sum(res), m, atol=1e-04)


def test_partial_wasserstein_1d():
    n_samples = 20  # nb samples

    rng = np.random.RandomState(42)
    xs = rng.randn(n_samples, 1)
    xt = rng.randn(n_samples, 1)

    ind_xs_half, ind_xt_half, marginal_costs_half = ot.partial.partial_wasserstein_1d(
        xs, xt, n_transported_samples=n_samples // 2, p=1
    )

    ind_xs, ind_xt, marginal_costs = ot.partial.partial_wasserstein_1d(xs, xt, p=1)

    np.testing.assert_allclose(
        marginal_costs_half, marginal_costs[: n_samples // 2], atol=1e-04
    )
    np.testing.assert_allclose(
        np.sum(np.abs(np.sort(xs[ind_xs_half]) - np.sort(xt[ind_xt_half]))),
        np.sum(marginal_costs_half),
        atol=1e-04,
    )

    n = 20
    x = np.random.rand(n)
    y = np.random.rand(n)

    M = ot.dist(x[:, None], y[:, None], metric="minkowski", p=1)
    indices_x, indices_y, marginal_costs = ot.partial.partial_wasserstein_1d(
        x, y, n_transported_samples=n
    )
    costs = np.cumsum(marginal_costs)

    for i in [1, 5, 10]:
        np.testing.assert_allclose(
            costs[i - 1] / n,
            ot.partial.partial_wasserstein2([], [], M, m=i / n),
            atol=1e-8,
        )

        t = ot.partial.partial_wasserstein([], [], M, m=i / n)
        ind_x, ind_y = np.where(t > 1e-6)

        np.testing.assert_array_equal(np.sort(indices_x[:i]), np.sort(ind_x))
        np.testing.assert_array_equal(np.sort(indices_y[:i]), np.sort(ind_y))


# ---------------------------------------------------------------------------
# entropic_partial_wasserstein_logscale — new in this PR (rescue of #724)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("reg", [10.0, 1.0])
def test_entropic_partial_wasserstein_logscale_matches_old_at_large_reg(reg):
    """At large reg both solvers are stable; the plans must agree."""
    rng = np.random.RandomState(0)
    n = 20
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(n)
    b /= b.sum()
    M = ot.dist(rng.rand(n, 2), rng.rand(n, 2))
    m = 0.5

    G_old = ot.partial.entropic_partial_wasserstein(
        a, b, M, reg=reg, m=m, numItermax=2000
    )
    G_log = ot.partial.entropic_partial_wasserstein_logscale(
        a, b, M, reg=reg, m=m, numItermax=2000
    )

    # At reg >= 1.0 the two solvers agree to machine precision; if this
    # tightens it would indicate the logscale path silently diverged.
    np.testing.assert_allclose(G_old, G_log, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(G_log.sum(), m, atol=1e-10)


@pytest.mark.parametrize("reg", [0.1, 0.05, 0.01, 5e-3, 1e-3, 5e-4])
def test_entropic_partial_wasserstein_logscale_no_nan_at_small_reg(reg):
    """Issue #723: entropic_partial_wasserstein returns NaN at small reg.

    The logscale variant introduced by this PR is the fix; check that it
    stays finite and conserves mass across the regime that breaks the
    original solver.
    """
    rng = np.random.RandomState(1)
    n = 50
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(n)
    b /= b.sum()
    M = ot.dist(rng.rand(n, 2), rng.rand(n, 2)) * 50.0  # match issue cost scale
    m = 0.6

    G = ot.partial.entropic_partial_wasserstein_logscale(
        a, b, M, reg=reg, m=m, numItermax=2000
    )
    assert np.isfinite(G).all(), f"non-finite plan at reg={reg}"
    np.testing.assert_allclose(G.sum(), m, atol=5e-3)


def test_entropic_partial_wasserstein_logscale_approaches_exact_at_small_reg():
    """At small `reg` the entropic plan should approach the exact partial
    OT plan (modulo discretisation). Verifies the fix is mathematically
    meaningful, not just NaN-free."""
    rng = np.random.RandomState(3)
    n = 30
    a = np.ones(n) / n
    b = np.ones(n) / n
    M = ot.dist(rng.rand(n, 2), rng.rand(n, 2))
    m = 0.5

    G_exact = ot.partial.partial_wasserstein(a, b, M, m=m)
    G_log = ot.partial.entropic_partial_wasserstein_logscale(
        a, b, M, reg=1e-3, m=m, numItermax=5000
    )

    cost_exact = float((G_exact * M).sum())
    cost_log = float((G_log * M).sum())
    # The entropic objective is a relaxation of the exact one, so the
    # plan-cost gap should be small but non-negative at reg → 0.
    assert cost_log >= cost_exact - 1e-6
    assert (
        cost_log - cost_exact < 0.01
    ), f"logscale plan cost {cost_log:.4f} diverges from exact {cost_exact:.4f}"


def test_entropic_partial_wasserstein_logscale_log_dict():
    """`log=True` returns a dict with `err` and `partial_w_dist` keys."""
    rng = np.random.RandomState(2)
    n = 10
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(n)
    b /= b.sum()
    M = ot.dist(rng.rand(n, 2), rng.rand(n, 2))

    G, log = ot.partial.entropic_partial_wasserstein_logscale(
        a, b, M, reg=0.1, m=0.5, log=True
    )
    assert "err" in log
    assert "partial_w_dist" in log
    assert np.isfinite(G).all()


def test_entropic_partial_wasserstein_logscale_input_validation():
    """Out-of-range `m` should raise ValueError, matching the unstable solver."""
    n = 10
    a = np.ones(n) / n
    b = np.ones(n) / n
    M = np.ones((n, n))
    with pytest.raises(ValueError):
        ot.partial.entropic_partial_wasserstein_logscale(a, b, M, reg=0.1, m=-1.0)
    with pytest.raises(ValueError):
        ot.partial.entropic_partial_wasserstein_logscale(a, b, M, reg=0.1, m=2.0)


# ---------------------------------------------------------------------------
# entropic_partial_wasserstein method dispatch (sinkhorn / sinkhorn_log)
# ---------------------------------------------------------------------------
def _partial_problem(seed=7, n=20, m=0.5, scale=1.0):
    rng = np.random.RandomState(seed)
    a = rng.rand(n)
    a /= a.sum()
    b = rng.rand(n)
    b /= b.sum()
    M = ot.dist(rng.rand(n, 2), rng.rand(n, 2)) * scale
    return a, b, M, m


def test_entropic_partial_wasserstein_method_default_is_sinkhorn():
    """The default call and ``method='sinkhorn'`` must be identical."""
    a, b, M, m = _partial_problem()
    G_default = ot.partial.entropic_partial_wasserstein(a, b, M, reg=1.0, m=m)
    G_sinkhorn = ot.partial.entropic_partial_wasserstein(
        a, b, M, reg=1.0, m=m, method="sinkhorn"
    )
    np.testing.assert_array_equal(G_default, G_sinkhorn)


@pytest.mark.parametrize("reg", [1.0, 0.05])
def test_entropic_partial_wasserstein_method_sinkhorn_log_matches_logscale(reg):
    """``method='sinkhorn_log'`` must dispatch to the standalone logscale solver."""
    a, b, M, m = _partial_problem(scale=50.0)
    G_wrap = ot.partial.entropic_partial_wasserstein(
        a, b, M, reg=reg, m=m, method="sinkhorn_log", numItermax=2000
    )
    G_log = ot.partial.entropic_partial_wasserstein_logscale(
        a, b, M, reg=reg, m=m, numItermax=2000
    )
    np.testing.assert_array_equal(G_wrap, G_log)


def test_entropic_partial_wasserstein_method_is_case_insensitive():
    """Method matching follows ``ot.sinkhorn`` and is case-insensitive."""
    a, b, M, m = _partial_problem()
    G_lower = ot.partial.entropic_partial_wasserstein(
        a, b, M, reg=1.0, m=m, method="sinkhorn_log"
    )
    G_upper = ot.partial.entropic_partial_wasserstein(
        a, b, M, reg=1.0, m=m, method="Sinkhorn_Log"
    )
    np.testing.assert_array_equal(G_lower, G_upper)


def test_entropic_partial_wasserstein_method_log_dict():
    """``log=True`` is forwarded to the selected solver for both methods."""
    a, b, M, m = _partial_problem(n=10)
    for method in ("sinkhorn", "sinkhorn_log"):
        G, log = ot.partial.entropic_partial_wasserstein(
            a, b, M, reg=0.1, m=m, method=method, log=True
        )
        assert "err" in log
        assert "partial_w_dist" in log
        assert np.isfinite(G).all()


def test_entropic_partial_wasserstein_method_invalid():
    """An unknown method raises ValueError, matching ``ot.sinkhorn``."""
    a, b, M, m = _partial_problem(n=10)
    with pytest.raises(ValueError):
        ot.partial.entropic_partial_wasserstein(
            a, b, M, reg=0.1, m=m, method="not_a_solver"
        )
