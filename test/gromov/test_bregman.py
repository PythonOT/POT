"""Tests for gromov._bregman.py"""

# Author: Rémi Flamary <remi.flamary@unice.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

import numpy as np
import pytest

import ot


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tf backend")
@pytest.mark.parametrize(
    "loss_fun",
    [
        "square_loss",
        "kl_loss",
        pytest.param("unknown_loss", marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_entropic_gromov(nx, loss_fun):
    n_samples = 10  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    G, log = ot.gromov.entropic_gromov_wasserstein(
        C1,
        C2,
        None,
        q,
        loss_fun,
        symmetric=None,
        G0=G0,
        epsilon=1e-2,
        max_iter=10,
        verbose=True,
        log=True,
    )
    Gb = nx.to_numpy(
        ot.gromov.entropic_gromov_wasserstein(
            C1b,
            C2b,
            pb,
            None,
            loss_fun,
            symmetric=True,
            G0=None,
            epsilon=1e-2,
            max_iter=10,
            verbose=True,
            log=False,
        )
    )

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-04)  # cf convergence gromov


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tf backend")
@pytest.mark.parametrize(
    "loss_fun",
    [
        "square_loss",
        "kl_loss",
        pytest.param("unknown_loss", marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_entropic_gromov2(nx, loss_fun):
    n_samples = 10  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    gw, log = ot.gromov.entropic_gromov_wasserstein2(
        C1,
        C2,
        p,
        None,
        loss_fun,
        symmetric=True,
        G0=None,
        max_iter=10,
        epsilon=1e-2,
        log=True,
    )
    gwb, logb = ot.gromov.entropic_gromov_wasserstein2(
        C1b,
        C2b,
        None,
        qb,
        loss_fun,
        symmetric=None,
        G0=G0b,
        max_iter=10,
        epsilon=1e-2,
        log=True,
    )
    gwb = nx.to_numpy(gwb)

    G = log["T"]
    Gb = nx.to_numpy(logb["T"])

    np.testing.assert_allclose(gw, gwb, atol=1e-06)
    np.testing.assert_allclose(gw, 0, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-04)  # cf convergence gromov


@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_proximal_gromov(nx):
    n_samples = 10  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    with pytest.raises(ValueError):
        loss_fun = "weird_loss_fun"
        G, log = ot.gromov.entropic_gromov_wasserstein(
            C1,
            C2,
            None,
            q,
            loss_fun,
            symmetric=None,
            G0=G0,
            epsilon=1e-1,
            max_iter=10,
            solver="PPA",
            verbose=True,
            log=True,
            numItermax=1,
        )

    G, log = ot.gromov.entropic_gromov_wasserstein(
        C1,
        C2,
        None,
        q,
        "square_loss",
        symmetric=None,
        G0=G0,
        epsilon=1e-1,
        max_iter=10,
        solver="PPA",
        verbose=True,
        log=True,
        numItermax=1,
    )
    Gb = nx.to_numpy(
        ot.gromov.entropic_gromov_wasserstein(
            C1b,
            C2b,
            pb,
            None,
            "square_loss",
            symmetric=True,
            G0=None,
            epsilon=1e-1,
            max_iter=10,
            solver="PPA",
            verbose=True,
            log=False,
            numItermax=1,
        )
    )

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-02)  # cf convergence gromov

    gw, log = ot.gromov.entropic_gromov_wasserstein2(
        C1,
        C2,
        p,
        q,
        "kl_loss",
        symmetric=True,
        G0=None,
        max_iter=10,
        epsilon=1e-1,
        solver="PPA",
        warmstart=True,
        log=True,
    )
    gwb, logb = ot.gromov.entropic_gromov_wasserstein2(
        C1b,
        C2b,
        pb,
        qb,
        "kl_loss",
        symmetric=None,
        G0=G0b,
        max_iter=10,
        epsilon=1e-1,
        solver="PPA",
        warmstart=True,
        log=True,
    )
    gwb = nx.to_numpy(gwb)

    G = log["T"]
    Gb = nx.to_numpy(logb["T"])

    np.testing.assert_allclose(gw, gwb, atol=1e-06)
    np.testing.assert_allclose(gw, 0, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-02)  # cf convergence gromov


@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_asymmetric_entropic_gromov(nx):
    n_samples = 10  # nb samples
    rng = np.random.RandomState(0)
    C1 = rng.uniform(low=0.0, high=10, size=(n_samples, n_samples))
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    C2 = C1[idx, :][:, idx]

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)
    G = ot.gromov.entropic_gromov_wasserstein(
        C1,
        C2,
        p,
        q,
        "square_loss",
        symmetric=None,
        G0=G0,
        epsilon=1e-1,
        max_iter=5,
        verbose=True,
        log=False,
    )
    Gb = nx.to_numpy(
        ot.gromov.entropic_gromov_wasserstein(
            C1b,
            C2b,
            pb,
            qb,
            "square_loss",
            symmetric=False,
            G0=None,
            epsilon=1e-1,
            max_iter=5,
            verbose=True,
            log=False,
        )
    )
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    gw = ot.gromov.entropic_gromov_wasserstein2(
        C1,
        C2,
        None,
        None,
        "kl_loss",
        symmetric=False,
        G0=None,
        max_iter=5,
        epsilon=1e-1,
        log=False,
    )
    gwb = ot.gromov.entropic_gromov_wasserstein2(
        C1b,
        C2b,
        pb,
        qb,
        "kl_loss",
        symmetric=None,
        G0=G0b,
        max_iter=5,
        epsilon=1e-1,
        log=False,
    )
    gwb = nx.to_numpy(gwb)

    np.testing.assert_allclose(gw, gwb, atol=1e-06)
    np.testing.assert_allclose(gw, 0, atol=1e-1, rtol=1e-1)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_gromov_dtype_device(nx):
    # setup
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        C1b, C2b, pb, qb = nx.from_numpy(C1, C2, p, q, type_as=tp)

        for solver in ["PGD", "PPA", "BAPG"]:
            if solver == "BAPG":
                Gb = ot.gromov.BAPG_gromov_wasserstein(
                    C1b, C2b, pb, qb, max_iter=2, verbose=True
                )
                gw_valb = ot.gromov.BAPG_gromov_wasserstein2(
                    C1b, C2b, pb, qb, max_iter=2, verbose=True
                )
            else:
                Gb = ot.gromov.entropic_gromov_wasserstein(
                    C1b, C2b, pb, qb, max_iter=2, solver=solver, verbose=True
                )
                gw_valb = ot.gromov.entropic_gromov_wasserstein2(
                    C1b, C2b, pb, qb, max_iter=2, solver=solver, verbose=True
                )

            nx.assert_same_dtype_device(C1b, Gb)
            nx.assert_same_dtype_device(C1b, gw_valb)


def test_BAPG_gromov(nx):
    n_samples = 10  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    # complete test with marginal loss = True
    marginal_loss = True
    with pytest.raises(ValueError):
        loss_fun = "weird_loss_fun"
        G, log = ot.gromov.BAPG_gromov_wasserstein(
            C1,
            C2,
            None,
            q,
            loss_fun,
            symmetric=None,
            G0=G0,
            epsilon=1e-1,
            max_iter=10,
            marginal_loss=marginal_loss,
            verbose=True,
            log=True,
        )

    G, log = ot.gromov.BAPG_gromov_wasserstein(
        C1,
        C2,
        None,
        q,
        "square_loss",
        symmetric=None,
        G0=G0,
        epsilon=1e-1,
        max_iter=10,
        marginal_loss=marginal_loss,
        verbose=True,
        log=True,
    )
    Gb = nx.to_numpy(
        ot.gromov.BAPG_gromov_wasserstein(
            C1b,
            C2b,
            pb,
            None,
            "square_loss",
            symmetric=True,
            G0=None,
            epsilon=1e-1,
            max_iter=10,
            marginal_loss=marginal_loss,
            verbose=True,
            log=False,
        )
    )

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-02)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-02)  # cf convergence gromov

    with pytest.warns(UserWarning):
        gw = ot.gromov.BAPG_gromov_wasserstein2(
            C1,
            C2,
            p,
            q,
            "kl_loss",
            symmetric=False,
            G0=None,
            max_iter=10,
            epsilon=1e-2,
            marginal_loss=marginal_loss,
            log=False,
        )

    gw, log = ot.gromov.BAPG_gromov_wasserstein2(
        C1,
        C2,
        p,
        q,
        "kl_loss",
        symmetric=False,
        G0=None,
        max_iter=10,
        epsilon=1.0,
        marginal_loss=marginal_loss,
        log=True,
    )
    gwb, logb = ot.gromov.BAPG_gromov_wasserstein2(
        C1b,
        C2b,
        pb,
        qb,
        "kl_loss",
        symmetric=None,
        G0=G0b,
        max_iter=10,
        epsilon=1.0,
        marginal_loss=marginal_loss,
        log=True,
    )
    gwb = nx.to_numpy(gwb)

    G = log["T"]
    Gb = nx.to_numpy(logb["T"])

    np.testing.assert_allclose(gw, gwb, atol=1e-06)
    np.testing.assert_allclose(gw, 0, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-02)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-02)  # cf convergence gromov

    marginal_loss = False
    G, log = ot.gromov.BAPG_gromov_wasserstein(
        C1,
        C2,
        None,
        q,
        "square_loss",
        symmetric=None,
        G0=G0,
        epsilon=1e-1,
        max_iter=10,
        marginal_loss=marginal_loss,
        verbose=True,
        log=True,
    )
    Gb = nx.to_numpy(
        ot.gromov.BAPG_gromov_wasserstein(
            C1b,
            C2b,
            pb,
            None,
            "square_loss",
            symmetric=False,
            G0=None,
            epsilon=1e-1,
            max_iter=10,
            marginal_loss=marginal_loss,
            verbose=True,
            log=False,
        )
    )


@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_fgw(nx):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    rng = np.random.RandomState(42)
    ys = rng.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    M = ot.dist(ys, yt)

    Mb, C1b, C2b, pb, qb, G0b = nx.from_numpy(M, C1, C2, p, q, G0)

    with pytest.raises(ValueError):
        loss_fun = "weird_loss_fun"
        G, log = ot.gromov.entropic_fused_gromov_wasserstein(
            M,
            C1,
            C2,
            None,
            None,
            loss_fun,
            symmetric=None,
            G0=G0,
            epsilon=1e-1,
            max_iter=10,
            verbose=True,
            log=True,
        )

    G, log = ot.gromov.entropic_fused_gromov_wasserstein(
        M,
        C1,
        C2,
        None,
        None,
        "square_loss",
        symmetric=None,
        G0=G0,
        epsilon=1e-1,
        max_iter=10,
        verbose=True,
        log=True,
    )
    Gb = nx.to_numpy(
        ot.gromov.entropic_fused_gromov_wasserstein(
            Mb,
            C1b,
            C2b,
            pb,
            qb,
            "square_loss",
            symmetric=True,
            G0=None,
            epsilon=1e-1,
            max_iter=10,
            verbose=True,
            log=False,
        )
    )

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    fgw, log = ot.gromov.entropic_fused_gromov_wasserstein2(
        M,
        C1,
        C2,
        p,
        q,
        "kl_loss",
        symmetric=True,
        G0=None,
        max_iter=10,
        epsilon=1e-1,
        log=True,
    )
    fgwb, logb = ot.gromov.entropic_fused_gromov_wasserstein2(
        Mb,
        C1b,
        C2b,
        pb,
        qb,
        "kl_loss",
        symmetric=None,
        G0=G0b,
        max_iter=10,
        epsilon=1e-1,
        log=True,
    )
    fgwb = nx.to_numpy(fgwb)

    G = log["T"]
    Gb = nx.to_numpy(logb["T"])

    np.testing.assert_allclose(fgw, fgwb, atol=1e-06)
    np.testing.assert_allclose(fgw, 0, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-04)  # cf convergence gromov


@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_proximal_fgw(nx):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    rng = np.random.RandomState(42)
    ys = rng.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    M = ot.dist(ys, yt)

    Mb, C1b, C2b, pb, qb, G0b = nx.from_numpy(M, C1, C2, p, q, G0)

    G, log = ot.gromov.entropic_fused_gromov_wasserstein(
        M,
        C1,
        C2,
        p,
        q,
        "square_loss",
        symmetric=None,
        G0=G0,
        epsilon=1e-1,
        max_iter=10,
        solver="PPA",
        verbose=True,
        log=True,
        numItermax=1,
    )
    Gb = nx.to_numpy(
        ot.gromov.entropic_fused_gromov_wasserstein(
            Mb,
            C1b,
            C2b,
            pb,
            qb,
            "square_loss",
            symmetric=True,
            G0=None,
            epsilon=1e-1,
            max_iter=10,
            solver="PPA",
            verbose=True,
            log=False,
            numItermax=1,
        )
    )

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    fgw, log = ot.gromov.entropic_fused_gromov_wasserstein2(
        M,
        C1,
        C2,
        p,
        None,
        "kl_loss",
        symmetric=True,
        G0=None,
        max_iter=5,
        epsilon=1e-1,
        solver="PPA",
        warmstart=True,
        log=True,
    )
    fgwb, logb = ot.gromov.entropic_fused_gromov_wasserstein2(
        Mb,
        C1b,
        C2b,
        None,
        qb,
        "kl_loss",
        symmetric=None,
        G0=G0b,
        max_iter=5,
        epsilon=1e-1,
        solver="PPA",
        warmstart=True,
        log=True,
    )
    fgwb = nx.to_numpy(fgwb)

    G = log["T"]
    Gb = nx.to_numpy(logb["T"])

    np.testing.assert_allclose(fgw, fgwb, atol=1e-06)
    np.testing.assert_allclose(fgw, 0, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-04)  # cf convergence gromov


def test_BAPG_fgw(nx):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    rng = np.random.RandomState(42)
    ys = rng.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    M = ot.dist(ys, yt)

    Mb, C1b, C2b, pb, qb, G0b = nx.from_numpy(M, C1, C2, p, q, G0)

    with pytest.raises(ValueError):
        loss_fun = "weird_loss_fun"
        G, log = ot.gromov.BAPG_fused_gromov_wasserstein(
            M, C1, C2, p, q, loss_fun=loss_fun, max_iter=1, log=True
        )

    # complete test with marginal loss = True
    marginal_loss = True

    G, log = ot.gromov.BAPG_fused_gromov_wasserstein(
        M,
        C1,
        C2,
        p,
        q,
        "square_loss",
        symmetric=None,
        G0=G0,
        epsilon=1e-1,
        max_iter=10,
        marginal_loss=marginal_loss,
        log=True,
    )
    Gb = nx.to_numpy(
        ot.gromov.BAPG_fused_gromov_wasserstein(
            Mb,
            C1b,
            C2b,
            pb,
            qb,
            "square_loss",
            symmetric=True,
            G0=None,
            epsilon=1e-1,
            max_iter=10,
            marginal_loss=marginal_loss,
            verbose=True,
        )
    )

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-02)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-02)  # cf convergence gromov

    with pytest.warns(UserWarning):
        fgw = ot.gromov.BAPG_fused_gromov_wasserstein2(
            M,
            C1,
            C2,
            p,
            q,
            "kl_loss",
            symmetric=False,
            G0=None,
            max_iter=10,
            epsilon=1e-3,
            marginal_loss=marginal_loss,
            log=False,
        )

    fgw, log = ot.gromov.BAPG_fused_gromov_wasserstein2(
        M,
        C1,
        C2,
        p,
        None,
        "kl_loss",
        symmetric=True,
        G0=None,
        max_iter=5,
        epsilon=1,
        marginal_loss=marginal_loss,
        log=True,
    )
    fgwb, logb = ot.gromov.BAPG_fused_gromov_wasserstein2(
        Mb,
        C1b,
        C2b,
        None,
        qb,
        "kl_loss",
        symmetric=None,
        G0=G0b,
        max_iter=5,
        epsilon=1,
        marginal_loss=marginal_loss,
        log=True,
    )
    fgwb = nx.to_numpy(fgwb)

    G = log["T"]
    Gb = nx.to_numpy(logb["T"])

    np.testing.assert_allclose(fgw, fgwb, atol=1e-06)
    np.testing.assert_allclose(fgw, 0, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-02)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-02)  # cf convergence gromov

    # Tests with marginal_loss = False
    marginal_loss = False
    G, log = ot.gromov.BAPG_fused_gromov_wasserstein(
        M,
        C1,
        C2,
        p,
        q,
        "square_loss",
        symmetric=False,
        G0=G0,
        epsilon=1e-1,
        max_iter=10,
        marginal_loss=marginal_loss,
        log=True,
    )
    Gb = nx.to_numpy(
        ot.gromov.BAPG_fused_gromov_wasserstein(
            Mb,
            C1b,
            C2b,
            pb,
            qb,
            "square_loss",
            symmetric=None,
            G0=None,
            epsilon=1e-1,
            max_iter=10,
            marginal_loss=marginal_loss,
            verbose=True,
        )
    )
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-02)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-02)  # cf convergence gromov


def test_asymmetric_entropic_fgw(nx):
    n_samples = 5  # nb samples
    rng = np.random.RandomState(0)
    C1 = rng.uniform(low=0.0, high=10, size=(n_samples, n_samples))
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    C2 = C1[idx, :][:, idx]

    ys = rng.randn(n_samples, 2)
    yt = ys[idx, :]
    M = ot.dist(ys, yt)

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    Mb, C1b, C2b, pb, qb, G0b = nx.from_numpy(M, C1, C2, p, q, G0)
    G = ot.gromov.entropic_fused_gromov_wasserstein(
        M,
        C1,
        C2,
        p,
        q,
        "square_loss",
        symmetric=None,
        G0=G0,
        max_iter=5,
        epsilon=1e-1,
        verbose=True,
        log=False,
    )
    Gb = nx.to_numpy(
        ot.gromov.entropic_fused_gromov_wasserstein(
            Mb,
            C1b,
            C2b,
            pb,
            qb,
            "square_loss",
            symmetric=False,
            G0=None,
            max_iter=5,
            epsilon=1e-1,
            verbose=True,
            log=False,
        )
    )
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    fgw = ot.gromov.entropic_fused_gromov_wasserstein2(
        M,
        C1,
        C2,
        p,
        q,
        "kl_loss",
        symmetric=False,
        G0=None,
        max_iter=5,
        epsilon=1e-1,
        log=False,
    )
    fgwb = ot.gromov.entropic_fused_gromov_wasserstein2(
        Mb,
        C1b,
        C2b,
        pb,
        qb,
        "kl_loss",
        symmetric=None,
        G0=G0b,
        max_iter=5,
        epsilon=1e-1,
        log=False,
    )
    fgwb = nx.to_numpy(fgwb)

    np.testing.assert_allclose(fgw, fgwb, atol=1e-06)
    np.testing.assert_allclose(fgw, 0, atol=1e-1, rtol=1e-1)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_fgw_dtype_device(nx):
    # setup
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    rng = np.random.RandomState(42)
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=rng)

    xt = xs[::-1].copy()

    ys = rng.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    M = ot.dist(ys, yt)
    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        Mb, C1b, C2b, pb, qb = nx.from_numpy(M, C1, C2, p, q, type_as=tp)

        for solver in ["PGD", "PPA", "BAPG"]:
            if solver == "BAPG":
                Gb = ot.gromov.BAPG_fused_gromov_wasserstein(
                    Mb, C1b, C2b, pb, qb, max_iter=2
                )
                fgw_valb = ot.gromov.BAPG_fused_gromov_wasserstein2(
                    Mb, C1b, C2b, pb, qb, max_iter=2
                )

            else:
                Gb = ot.gromov.entropic_fused_gromov_wasserstein(
                    Mb, C1b, C2b, pb, qb, max_iter=2, solver=solver
                )
                fgw_valb = ot.gromov.entropic_fused_gromov_wasserstein2(
                    Mb, C1b, C2b, pb, qb, max_iter=2, solver=solver
                )

            nx.assert_same_dtype_device(C1b, Gb)
            nx.assert_same_dtype_device(C1b, fgw_valb)


def test_entropic_fgw_barycenter(nx):
    ns = 5
    nt = 10

    rng = np.random.RandomState(42)
    Xs, ys = ot.datasets.make_data_classif("3gauss", ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif("3gauss2", nt, random_state=42)

    ys = rng.randn(Xs.shape[0], 2)
    yt = rng.randn(Xt.shape[0], 2)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)
    p1 = ot.unif(ns)
    p2 = ot.unif(nt)
    n_samples = 3
    p = ot.unif(n_samples)

    ysb, ytb, C1b, C2b, p1b, p2b, pb = nx.from_numpy(ys, yt, C1, C2, p1, p2, p)

    with pytest.raises(ValueError):
        loss_fun = "weird_loss_fun"
        X, C, log = ot.gromov.entropic_fused_gromov_barycenters(
            n_samples,
            [ys, yt],
            [C1, C2],
            None,
            p,
            [0.5, 0.5],
            loss_fun,
            0.1,
            max_iter=10,
            tol=1e-3,
            verbose=True,
            warmstartT=True,
            random_state=42,
            solver="PPA",
            numItermax=10,
            log=True,
            symmetric=True,
        )
    with pytest.raises(ValueError):
        stop_criterion = "unknown stop criterion"
        X, C, log = ot.gromov.entropic_fused_gromov_barycenters(
            n_samples,
            [ys, yt],
            [C1, C2],
            None,
            p,
            [0.5, 0.5],
            "square_loss",
            0.1,
            max_iter=10,
            tol=1e-3,
            stop_criterion=stop_criterion,
            verbose=True,
            warmstartT=True,
            random_state=42,
            solver="PPA",
            numItermax=10,
            log=True,
            symmetric=True,
        )

    for stop_criterion in ["barycenter", "loss"]:
        X, C, log = ot.gromov.entropic_fused_gromov_barycenters(
            n_samples,
            [ys, yt],
            [C1, C2],
            None,
            p,
            [0.5, 0.5],
            "square_loss",
            epsilon=0.1,
            max_iter=10,
            tol=1e-3,
            stop_criterion=stop_criterion,
            verbose=True,
            warmstartT=True,
            random_state=42,
            solver="PPA",
            numItermax=10,
            log=True,
            symmetric=True,
        )
        Xb, Cb = ot.gromov.entropic_fused_gromov_barycenters(
            n_samples,
            [ysb, ytb],
            [C1b, C2b],
            [p1b, p2b],
            None,
            [0.5, 0.5],
            "square_loss",
            epsilon=0.1,
            max_iter=10,
            tol=1e-3,
            stop_criterion=stop_criterion,
            verbose=False,
            warmstartT=True,
            random_state=42,
            solver="PPA",
            numItermax=10,
            log=False,
            symmetric=True,
        )
        Xb, Cb = nx.to_numpy(Xb, Cb)

        np.testing.assert_allclose(C, Cb, atol=1e-06)
        np.testing.assert_allclose(Cb.shape, (n_samples, n_samples))
        np.testing.assert_allclose(X, Xb, atol=1e-06)
        np.testing.assert_allclose(Xb.shape, (n_samples, ys.shape[1]))

    # test with 'kl_loss' and log=True
    # providing init_C, init_Y
    generator = ot.utils.check_random_state(42)
    xalea = generator.randn(n_samples, 2)
    init_C = ot.utils.dist(xalea, xalea)
    init_C /= init_C.max()
    init_Cb = nx.from_numpy(init_C)

    init_Y = np.zeros((n_samples, ys.shape[1]), dtype=ys.dtype)
    init_Yb = nx.from_numpy(init_Y)

    X, C, log = ot.gromov.entropic_fused_gromov_barycenters(
        n_samples,
        [ys, yt],
        [C1, C2],
        [p1, p2],
        p,
        None,
        "kl_loss",
        0.1,
        True,
        max_iter=10,
        tol=1e-3,
        verbose=False,
        warmstartT=False,
        random_state=42,
        solver="PPA",
        numItermax=1,
        init_C=init_C,
        init_Y=init_Y,
        log=True,
    )
    Xb, Cb, logb = ot.gromov.entropic_fused_gromov_barycenters(
        n_samples,
        [ysb, ytb],
        [C1b, C2b],
        [p1b, p2b],
        pb,
        [0.5, 0.5],
        "kl_loss",
        0.1,
        True,
        max_iter=10,
        tol=1e-3,
        verbose=False,
        warmstartT=False,
        random_state=42,
        solver="PPA",
        numItermax=1,
        init_C=init_Cb,
        init_Y=init_Yb,
        log=True,
    )
    Xb, Cb = nx.to_numpy(Xb, Cb)

    np.testing.assert_allclose(C, Cb, atol=1e-06)
    np.testing.assert_allclose(Cb.shape, (n_samples, n_samples))
    np.testing.assert_allclose(X, Xb, atol=1e-06)
    np.testing.assert_allclose(Xb.shape, (n_samples, ys.shape[1]))
    np.testing.assert_array_almost_equal(
        log["err_feature"], nx.to_numpy(*logb["err_feature"])
    )
    np.testing.assert_array_almost_equal(
        log["err_structure"], nx.to_numpy(*logb["err_structure"])
    )

    # add tests with fixed_structures or fixed_features
    init_C = ot.utils.dist(xalea, xalea)
    init_C /= init_C.max()
    init_Cb = nx.from_numpy(init_C)

    init_Y = np.zeros((n_samples, ys.shape[1]), dtype=ys.dtype)
    init_Yb = nx.from_numpy(init_Y)

    fixed_structure, fixed_features = True, False
    with pytest.raises(
        ot.utils.UndefinedParameter
    ):  # to raise an error when `fixed_structure=True`and `init_C=None`
        Xb, Cb = ot.gromov.entropic_fused_gromov_barycenters(
            n_samples,
            [ysb, ytb],
            [C1b, C2b],
            ps=[p1b, p2b],
            lambdas=None,
            fixed_structure=fixed_structure,
            init_C=None,
            fixed_features=fixed_features,
            p=None,
            max_iter=10,
            tol=1e-3,
        )

    Xb, Cb = ot.gromov.entropic_fused_gromov_barycenters(
        n_samples,
        [ysb, ytb],
        [C1b, C2b],
        ps=[p1b, p2b],
        lambdas=None,
        fixed_structure=fixed_structure,
        init_C=init_Cb,
        fixed_features=fixed_features,
        max_iter=10,
        tol=1e-3,
    )
    Xb, Cb = nx.to_numpy(Xb), nx.to_numpy(Cb)
    np.testing.assert_allclose(Cb, init_Cb)
    np.testing.assert_allclose(Xb.shape, (n_samples, ys.shape[1]))

    fixed_structure, fixed_features = False, True
    with pytest.raises(
        ot.utils.UndefinedParameter
    ):  # to raise an error when `fixed_features=True`and `init_X=None`
        Xb, Cb, logb = ot.gromov.entropic_fused_gromov_barycenters(
            n_samples,
            [ysb, ytb],
            [C1b, C2b],
            [p1b, p2b],
            lambdas=[0.5, 0.5],
            fixed_structure=fixed_structure,
            fixed_features=fixed_features,
            init_Y=None,
            p=pb,
            max_iter=10,
            tol=1e-3,
            warmstartT=True,
            log=True,
            random_state=98765,
            verbose=True,
        )
    Xb, Cb, logb = ot.gromov.entropic_fused_gromov_barycenters(
        n_samples,
        [ysb, ytb],
        [C1b, C2b],
        [p1b, p2b],
        lambdas=[0.5, 0.5],
        fixed_structure=fixed_structure,
        fixed_features=fixed_features,
        init_Y=init_Yb,
        p=pb,
        max_iter=10,
        tol=1e-3,
        warmstartT=True,
        log=True,
        random_state=98765,
        verbose=True,
    )

    X, C = nx.to_numpy(Xb), nx.to_numpy(Cb)
    np.testing.assert_allclose(C.shape, (n_samples, n_samples))
    np.testing.assert_allclose(Xb, init_Yb)

    # test edge cases for fgw barycenters:
    # C1 as list
    with pytest.raises(ValueError):
        C1_list = [list(c) for c in C1b]
        _, _, _ = ot.gromov.entropic_fused_gromov_barycenters(
            n_samples,
            [ysb],
            [C1_list],
            [p1b],
            lambdas=None,
            fixed_structure=False,
            fixed_features=False,
            init_Y=None,
            p=pb,
            max_iter=10,
            tol=1e-3,
            warmstartT=True,
            log=True,
            random_state=98765,
            verbose=True,
        )

    # p1, p2 as lists
    with pytest.raises(ValueError):
        p1_list = list(p1b)
        p2_list = list(p2b)
        _, _, _ = ot.gromov.entropic_fused_gromov_barycenters(
            n_samples,
            [ysb, ytb],
            [C1b, C2b],
            [p1_list, p2_list],
            lambdas=[0.5, 0.5],
            fixed_structure=False,
            fixed_features=False,
            init_Y=None,
            p=pb,
            max_iter=10,
            tol=1e-3,
            warmstartT=True,
            log=True,
            random_state=98765,
            verbose=True,
        )

    # unique input structure
    X, C = ot.gromov.entropic_fused_gromov_barycenters(
        n_samples,
        [ys],
        [C1],
        [p1],
        lambdas=None,
        fixed_structure=False,
        fixed_features=False,
        init_Y=init_Y,
        p=p,
        max_iter=10,
        tol=1e-3,
        warmstartT=True,
        log=False,
        random_state=98765,
        verbose=True,
    )

    Xb, Cb = ot.gromov.entropic_fused_gromov_barycenters(
        n_samples,
        [ysb],
        [C1b],
        [p1b],
        lambdas=None,
        fixed_structure=False,
        fixed_features=False,
        init_Y=init_Yb,
        p=pb,
        max_iter=10,
        tol=1e-3,
        warmstartT=True,
        log=False,
        random_state=98765,
        verbose=True,
    )

    np.testing.assert_allclose(C, Cb, atol=1e-06)
    np.testing.assert_allclose(X, Xb, atol=1e-06)


@pytest.mark.filterwarnings("ignore:divide")
def test_gromov_entropic_barycenter(nx):
    ns = 5
    nt = 10

    Xs, ys = ot.datasets.make_data_classif("3gauss", ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif("3gauss2", nt, random_state=42)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)
    p1 = ot.unif(ns)
    p2 = ot.unif(nt)
    n_samples = 2
    p = ot.unif(n_samples)

    C1b, C2b, p1b, p2b, pb = nx.from_numpy(C1, C2, p1, p2, p)

    with pytest.raises(ValueError):
        loss_fun = "weird_loss_fun"
        Cb = ot.gromov.entropic_gromov_barycenters(
            n_samples,
            [C1, C2],
            None,
            p,
            [0.5, 0.5],
            loss_fun,
            1e-3,
            max_iter=10,
            tol=1e-3,
            verbose=True,
            warmstartT=True,
            random_state=42,
        )
    with pytest.raises(ValueError):
        stop_criterion = "unknown stop criterion"
        Cb = ot.gromov.entropic_gromov_barycenters(
            n_samples,
            [C1, C2],
            None,
            p,
            [0.5, 0.5],
            "square_loss",
            1e-3,
            max_iter=10,
            tol=1e-3,
            stop_criterion=stop_criterion,
            verbose=True,
            warmstartT=True,
            random_state=42,
        )

    Cb = ot.gromov.entropic_gromov_barycenters(
        n_samples,
        [C1, C2],
        None,
        p,
        [0.5, 0.5],
        "square_loss",
        1e-3,
        max_iter=10,
        tol=1e-3,
        verbose=True,
        warmstartT=True,
        random_state=42,
    )
    Cbb = nx.to_numpy(
        ot.gromov.entropic_gromov_barycenters(
            n_samples,
            [C1b, C2b],
            [p1b, p2b],
            None,
            [0.5, 0.5],
            "square_loss",
            1e-3,
            max_iter=10,
            tol=1e-3,
            verbose=True,
            warmstartT=True,
            random_state=42,
        )
    )
    np.testing.assert_allclose(Cb, Cbb, atol=1e-06)
    np.testing.assert_allclose(Cbb.shape, (n_samples, n_samples))

    # test of entropic_gromov_barycenters with `log` on
    for stop_criterion in ["barycenter", "loss"]:
        Cb_, err_ = ot.gromov.entropic_gromov_barycenters(
            n_samples,
            [C1, C2],
            [p1, p2],
            p,
            None,
            "square_loss",
            1e-3,
            max_iter=10,
            tol=1e-3,
            stop_criterion=stop_criterion,
            verbose=True,
            random_state=42,
            log=True,
        )
        Cbb_, errb_ = ot.gromov.entropic_gromov_barycenters(
            n_samples,
            [C1b, C2b],
            [p1b, p2b],
            pb,
            [0.5, 0.5],
            "square_loss",
            1e-3,
            max_iter=10,
            tol=1e-3,
            stop_criterion=stop_criterion,
            verbose=True,
            random_state=42,
            log=True,
        )
        Cbb_ = nx.to_numpy(Cbb_)
        np.testing.assert_allclose(Cb_, Cbb_, atol=1e-06)
        np.testing.assert_array_almost_equal(err_["err"], nx.to_numpy(*errb_["err"]))
        np.testing.assert_allclose(Cbb_.shape, (n_samples, n_samples))

    Cb2 = ot.gromov.entropic_gromov_barycenters(
        n_samples,
        [C1, C2],
        [p1, p2],
        p,
        [0.5, 0.5],
        "kl_loss",
        1e-3,
        max_iter=10,
        tol=1e-3,
        random_state=42,
    )
    Cb2b = nx.to_numpy(
        ot.gromov.entropic_gromov_barycenters(
            n_samples,
            [C1b, C2b],
            [p1b, p2b],
            pb,
            [0.5, 0.5],
            "kl_loss",
            1e-3,
            max_iter=10,
            tol=1e-3,
            random_state=42,
        )
    )
    np.testing.assert_allclose(Cb2, Cb2b, atol=1e-06)
    np.testing.assert_allclose(Cb2b.shape, (n_samples, n_samples))

    # test of entropic_gromov_barycenters with `log` on
    # providing init_C
    generator = ot.utils.check_random_state(42)
    xalea = generator.randn(n_samples, 2)
    init_C = ot.utils.dist(xalea, xalea)
    init_C /= init_C.max()
    init_Cb = nx.from_numpy(init_C)

    Cb2_, err2_ = ot.gromov.entropic_gromov_barycenters(
        n_samples,
        [C1, C2],
        [p1, p2],
        p,
        [0.5, 0.5],
        "kl_loss",
        1e-3,
        max_iter=10,
        tol=1e-3,
        warmstartT=True,
        verbose=True,
        random_state=42,
        init_C=init_C,
        log=True,
    )
    Cb2b_, err2b_ = ot.gromov.entropic_gromov_barycenters(
        n_samples,
        [C1b, C2b],
        [p1b, p2b],
        pb,
        [0.5, 0.5],
        "kl_loss",
        1e-3,
        max_iter=10,
        tol=1e-3,
        warmstartT=True,
        verbose=True,
        random_state=42,
        init_Cb=init_Cb,
        log=True,
    )
    Cb2b_ = nx.to_numpy(Cb2b_)
    np.testing.assert_allclose(Cb2_, Cb2b_, atol=1e-06)
    np.testing.assert_array_almost_equal(err2_["err"], nx.to_numpy(*err2b_["err"]))
    np.testing.assert_allclose(Cb2b_.shape, (n_samples, n_samples))

    # test edge cases for gw barycenters:
    # C1 as list
    with pytest.raises(ValueError):
        C1_list = [list(c) for c in C1b]
        _, _ = ot.gromov.entropic_gromov_barycenters(
            n_samples,
            [C1_list],
            [p1b],
            pb,
            None,
            "square_loss",
            1e-3,
            max_iter=10,
            tol=1e-3,
            warmstartT=True,
            verbose=True,
            random_state=42,
            init_C=None,
            log=True,
        )

    # p1, p2 as lists
    with pytest.raises(ValueError):
        p1_list = list(p1b)
        p2_list = list(p2b)
        _, _ = ot.gromov.entropic_gromov_barycenters(
            n_samples,
            [C1b, C2b],
            [p1_list, p2_list],
            pb,
            None,
            "kl_loss",
            1e-3,
            max_iter=10,
            tol=1e-3,
            warmstartT=True,
            verbose=True,
            random_state=42,
            init_Cb=None,
            log=True,
        )

    # unique input structure
    Cb = ot.gromov.entropic_gromov_barycenters(
        n_samples,
        [C1],
        [p1],
        p,
        None,
        "square_loss",
        1e-3,
        max_iter=10,
        tol=1e-3,
        warmstartT=True,
        verbose=True,
        random_state=42,
        init_C=None,
        log=False,
    )

    Cbb = ot.gromov.entropic_gromov_barycenters(
        n_samples,
        [C1b],
        [p1b],
        pb,
        [1.0],
        "square_loss",
        1e-3,
        max_iter=10,
        tol=1e-3,
        warmstartT=True,
        verbose=True,
        random_state=42,
        init_Cb=None,
        log=False,
    )

    np.testing.assert_allclose(Cb, Cbb, atol=1e-06)
    np.testing.assert_allclose(Cbb.shape, (n_samples, n_samples))


def test_not_implemented_solver():
    # test sinkhorn
    n_samples = 5  # nb samples
    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    rng = np.random.RandomState(42)
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=rng)
    xt = xs[::-1].copy()
    ys = rng.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()
    M = ot.dist(ys, yt)

    solver = "not_implemented"
    # entropic gw and fgw
    with pytest.raises(ValueError):
        ot.gromov.entropic_gromov_wasserstein(
            C1, C2, p, q, "square_loss", epsilon=1e-1, solver=solver
        )
    with pytest.raises(ValueError):
        ot.gromov.entropic_fused_gromov_wasserstein(
            M, C1, C2, p, q, "square_loss", epsilon=1e-1, solver=solver
        )
