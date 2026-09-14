"""Tests for module dr on Dimensionality Reduction"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Minhui Huang <mhhuang@ucdavis.edu>
#         Antoine Collas <antoine.collas@inria.fr>
#
# License: MIT License

import numpy as np
import ot
import pytest

try:  # test if autograd and pymanopt are installed
    import ot.dr

    nogo = not (ot.dr.HAS_AUTOGRAD and ot.dr.HAS_PYMANOPT)
except ImportError:
    nogo = True

try:
    import torch

    notorch = False
except ImportError:
    notorch = True


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_fda():
    n_samples = 90  # nb samples in source and target datasets
    rng = np.random.RandomState(0)

    # generate gaussian dataset
    xs, ys = ot.datasets.make_data_classif("gaussrot", n_samples, random_state=rng)

    n_features_noise = 8

    xs = np.hstack((xs, rng.randn(n_samples, n_features_noise)))

    p = 1

    Pfda, projfda = ot.dr.fda(xs, ys, p)

    projfda(xs)

    np.testing.assert_allclose(np.sum(Pfda**2, 0), np.ones(p))


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_fda_recovers_discriminant_direction():
    # classes are separated along the first feature only, all others are noise,
    # so FDA must return a direction aligned with e_0
    rng = np.random.RandomState(1)
    n_features = 5
    xs = np.concatenate(
        [
            rng.randn(60, n_features) * 0.3 + shift * np.eye(1, n_features, 0)
            for shift in [-6.0, 0.0, 6.0]
        ]
    )
    ys = np.repeat([0, 1, 2], 60)

    Pfda, _ = ot.dr.fda(xs, ys, p=1)

    direction = Pfda[:, 0] / np.linalg.norm(Pfda[:, 0])
    np.testing.assert_array_less(0.95, np.abs(direction[0]))


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_fda_projection_is_centered():
    rng = np.random.RandomState(0)
    xs, ys = ot.datasets.make_data_classif("gaussrot", 90, random_state=rng)
    xs = xs + 10.0  # off-centered data makes an absent centering visible

    _, projfda = ot.dr.fda(xs, ys, p=1)

    np.testing.assert_allclose(projfda(xs).mean(axis=0), 0.0, atol=1e-10)


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_fda_wda_do_not_modify_input():
    rng = np.random.RandomState(0)
    xs, ys = ot.datasets.make_data_classif("gaussrot", 90, random_state=rng)
    xs = xs + 10.0

    xs_copy = xs.copy()
    ot.dr.fda(xs, ys, p=1)
    np.testing.assert_allclose(xs, xs_copy)

    ot.dr.wda(xs, ys, p=1, reg=1.0, k=5, maxiter=5)
    np.testing.assert_allclose(xs, xs_copy)


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_wda():
    n_samples = 100  # nb samples in source and target datasets
    rng = np.random.RandomState(0)

    # generate gaussian dataset
    xs, ys = ot.datasets.make_data_classif("gaussrot", n_samples, random_state=rng)

    n_features_noise = 8

    xs = np.hstack((xs, rng.randn(n_samples, n_features_noise)))

    p = 2

    Pwda, projwda = ot.dr.wda(xs, ys, p, maxiter=10)

    projwda(xs)

    np.testing.assert_allclose(np.sum(Pwda**2, 0), np.ones(p))


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_wda_low_reg():
    n_samples = 100  # nb samples in source and target datasets
    rng = np.random.RandomState(0)

    # generate gaussian dataset
    xs, ys = ot.datasets.make_data_classif("gaussrot", n_samples, random_state=rng)

    n_features_noise = 8

    xs = np.hstack((xs, rng.randn(n_samples, n_features_noise)))

    p = 2

    Pwda, projwda = ot.dr.wda(
        xs, ys, p, reg=0.01, maxiter=10, sinkhorn_method="sinkhorn_log"
    )

    projwda(xs)

    np.testing.assert_allclose(np.sum(Pwda**2, 0), np.ones(p))


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_wda_normalized():
    n_samples = 100  # nb samples in source and target datasets
    rng = np.random.RandomState(0)

    # generate gaussian dataset
    xs, ys = ot.datasets.make_data_classif("gaussrot", n_samples, random_state=rng)

    n_features_noise = 8

    xs = np.hstack((xs, rng.randn(n_samples, n_features_noise)))

    p = 2

    P0 = rng.randn(10, p)
    P0 /= P0.sum(0, keepdims=True)

    Pwda, projwda = ot.dr.wda(xs, ys, p, maxiter=10, P0=P0, normalize=True)

    projwda(xs)

    np.testing.assert_allclose(np.sum(Pwda**2, 0), np.ones(p))


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_prw():
    d = 100  # Dimension
    n = 100  # Number samples
    k = 3  # Subspace dimension
    dim = 3

    def fragmented_hypercube(n, d, dim, rng):
        assert dim <= d
        assert dim >= 1
        assert dim == int(dim)

        a = (1.0 / n) * np.ones(n)
        b = (1.0 / n) * np.ones(n)

        # First measure : uniform on the hypercube
        X = rng.uniform(-1, 1, size=(n, d))

        # Second measure : fragmentation
        tmp_y = rng.uniform(-1, 1, size=(n, d))
        Y = tmp_y + 2 * np.sign(tmp_y) * np.array(dim * [1] + (d - dim) * [0])
        return a, b, X, Y

    rng = np.random.RandomState(42)
    a, b, X, Y = fragmented_hypercube(n, d, dim, rng)

    tau = 0.002
    reg = 0.2

    pi, U = ot.dr.projection_robust_wasserstein(
        X, Y, a, b, tau, reg=reg, k=k, maxiter=1000, verbose=1
    )

    U0 = rng.randn(d, k)
    U0, _ = np.linalg.qr(U0)

    pi, U = ot.dr.projection_robust_wasserstein(
        X, Y, a, b, tau, U0=U0, reg=reg, k=k, maxiter=1000, verbose=1
    )


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_ewca():
    d = 5
    n_samples = 50
    k = 3
    rng = np.random.RandomState(0)

    # generate gaussian dataset
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    D = rng.normal(size=d)
    D = (D / np.linalg.norm(D)) ** 4
    cov = Q @ np.diag(D) @ Q.T
    X = rng.multivariate_normal(np.zeros(d), cov, size=n_samples)
    X = X - X.mean(0, keepdims=True)
    assert X.shape == (n_samples, d)

    # compute first 3 components with BCD
    pi, U = ot.dr.ewca(
        X, reg=0.01, method="BCD", k=k, verbose=1, sinkhorn_method="sinkhorn_log"
    )
    assert pi.shape == (n_samples, n_samples)
    assert (pi >= 0).all()
    assert np.allclose(pi.sum(0), 1 / n_samples, atol=1e-3)
    assert np.allclose(pi.sum(1), 1 / n_samples, atol=1e-3)
    assert U.shape == (d, k)
    assert np.allclose(U.T @ U, np.eye(k), atol=1e-3)

    # test that U contains the principal components
    U_first_eigvec = np.linalg.svd(X.T, full_matrices=False)[0][:, :k]
    _, cos, _ = np.linalg.svd(U.T @ U_first_eigvec, full_matrices=False)
    assert np.allclose(cos, np.ones(k), atol=1e-3)

    # compute first 3 components with MM
    pi, U = ot.dr.ewca(
        X, reg=0.01, method="MM", k=k, verbose=1, sinkhorn_method="sinkhorn_log"
    )
    assert pi.shape == (n_samples, n_samples)
    assert (pi >= 0).all()
    assert np.allclose(pi.sum(0), 1 / n_samples, atol=1e-3)
    assert np.allclose(pi.sum(1), 1 / n_samples, atol=1e-3)
    assert U.shape == (d, k)
    assert np.allclose(U.T @ U, np.eye(k), atol=1e-3)

    # test that U contains the principal components
    U_first_eigvec = np.linalg.svd(X.T, full_matrices=False)[0][:, :k]
    _, cos, _ = np.linalg.svd(U.T @ U_first_eigvec, full_matrices=False)
    assert np.allclose(cos, np.ones(k), atol=1e-3)

    # compute last 3 components
    pi, U = ot.dr.ewca(
        X, reg=100000, method="MM", k=k, verbose=1, sinkhorn_method="sinkhorn_log"
    )

    # test that U contains the last principal components
    U_last_eigvec = np.linalg.svd(X.T, full_matrices=False)[0][:, -k:]
    _, cos, _ = np.linalg.svd(U.T @ U_last_eigvec, full_matrices=False)
    assert np.allclose(cos, np.ones(k), atol=1e-3)


@pytest.mark.skipif(notorch, reason="Missing module (torch)")
def test_wda_torch_solver():
    rng = np.random.RandomState(0)
    xs, ys = ot.datasets.make_data_classif("gaussrot", 90, random_state=rng)
    xs = np.hstack((xs, rng.randn(90, 4)))
    p = 2

    P, proj = ot.dr.wda(xs, ys, p, maxiter=10, solver="torch")

    np.testing.assert_allclose(np.sum(P**2, 0), np.ones(p), rtol=1e-6)
    assert proj(xs).shape == (90, p)


@pytest.mark.skipif(notorch, reason="Missing module (torch)")
def test_wda_torch_accepts_torch_tensors():
    rng = np.random.RandomState(0)
    xs, ys = ot.datasets.make_data_classif("gaussrot", 90, random_state=rng)
    xt = torch.tensor(xs, dtype=torch.float64)
    yt = torch.tensor(ys)

    P, proj = ot.dr.wda(xt, yt, 2, maxiter=5, solver="torch")

    assert torch.is_tensor(P)
    assert P.dtype == torch.float64
    assert proj(xt).shape == (90, 2)


@pytest.mark.skipif(notorch, reason="Missing module (torch)")
def test_wda_torch_does_not_modify_input():
    rng = np.random.RandomState(0)
    xs, ys = ot.datasets.make_data_classif("gaussrot", 90, random_state=rng)
    xs = xs + 10.0
    xs_copy = xs.copy()

    ot.dr.wda(xs, ys, 2, maxiter=5, solver="torch")

    np.testing.assert_allclose(xs, xs_copy)


@pytest.mark.skipif(notorch, reason="Missing module (torch)")
def test_wda_torch_sinkhorn_log():
    rng = np.random.RandomState(0)
    xs, ys = ot.datasets.make_data_classif("gaussrot", 90, random_state=rng)
    p = 2

    P, _ = ot.dr.wda(
        xs, ys, p, maxiter=10, solver="torch", sinkhorn_method="sinkhorn_log"
    )

    np.testing.assert_allclose(np.sum(P**2, 0), np.ones(p), rtol=1e-6)


@pytest.mark.skipif(nogo or notorch, reason="Missing modules")
def test_wda_backends_agree_on_cost_and_gradient():
    """The torch objective and its gradient must match the autograd ones."""
    import autograd
    import autograd.numpy as anp

    rng = np.random.RandomState(0)
    n, d, C, reg, k = 180, 6, 3, 1.0, 10
    X = np.vstack([rng.randn(n // C, d) + 3 * rng.randn(1, d) for _ in range(C)])
    y = np.repeat(np.arange(C), n // C)
    P0 = np.linalg.qr(rng.randn(d, 2))[0]
    Xc = X - X.mean(0)

    xc = [np.ascontiguousarray(Xc[y == c]) for c in range(C)]
    wc = [np.ones(x.shape[0]) / x.shape[0] for x in xc]

    def cost_autograd(P):
        loss_b, loss_w = 0.0, 0.0
        for i, xi in enumerate(xc):
            xi = anp.dot(xi, P)
            for j, xj in enumerate(xc[i:]):
                xj = anp.dot(xj, P)
                M = ot.dr.dist(xi, xj)
                G = ot.dr.sinkhorn(wc[i], wc[j + i], M, reg, k)
                term = anp.sum(G * M)
                if j == 0:
                    loss_w = loss_w + term
                else:
                    loss_b = loss_b + term
        return loss_w / loss_b

    xct = [torch.tensor(x, dtype=torch.float64) for x in xc]
    wct = [torch.tensor(w, dtype=torch.float64) for w in wc]
    rmt = torch.ones((C, C), dtype=torch.float64)
    Pt = torch.tensor(P0, dtype=torch.float64, requires_grad=True)
    v = ot.dr._wda_cost_torch(Pt, xct, wct, rmt, reg, k, ot.dr._sinkhorn_torch)
    (g,) = torch.autograd.grad(v, Pt)

    np.testing.assert_allclose(float(v.detach()), float(cost_autograd(P0)), rtol=1e-10)
    np.testing.assert_allclose(
        g.numpy(), autograd.grad(cost_autograd)(P0), rtol=1e-8, atol=1e-10
    )


@pytest.mark.skipif(nogo or notorch, reason="Missing modules")
def test_wda_solvers_reach_comparable_objective():
    """Both solvers minimise the same objective, so neither should be much worse."""
    rng = np.random.RandomState(0)
    xs, ys = ot.datasets.make_data_classif("gaussrot", 120, random_state=rng)
    xs = np.hstack((xs, rng.randn(120, 3)))
    P0 = np.linalg.qr(rng.randn(xs.shape[1], 2))[0]

    Pa, _ = ot.dr.wda(xs, ys, 2, maxiter=40, P0=P0)
    Pt, _ = ot.dr.wda(xs, ys, 2, maxiter=40, P0=P0, solver="torch")

    Xc = torch.tensor(xs - xs.mean(0), dtype=torch.float64)
    yt = torch.tensor(ys)
    xc = [Xc[yt == c] for c in torch.unique(yt)]
    wc = [torch.full((x.shape[0],), 1.0 / x.shape[0], dtype=torch.float64) for x in xc]
    rm = torch.ones((len(xc), len(xc)), dtype=torch.float64)

    def objective(P):
        return float(
            ot.dr._wda_cost_torch(
                torch.tensor(P, dtype=torch.float64),
                xc,
                wc,
                rm,
                1,
                10,
                ot.dr._sinkhorn_torch,
            )
        )

    assert objective(Pt) < 1.15 * objective(Pa)


@pytest.mark.skipif(notorch, reason="Missing module (torch)")
def test_wda_torch_unknown_sinkhorn_method():
    rng = np.random.RandomState(0)
    xs, ys = ot.datasets.make_data_classif("gaussrot", 60, random_state=rng)

    with pytest.raises(ValueError):
        ot.dr.wda(xs, ys, 2, maxiter=2, solver="torch", sinkhorn_method="nope")
