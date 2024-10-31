"""Tests for gromov._semirelaxed.py"""

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

import numpy as np
import pytest

import ot
from ot.backend import torch

from ot.gromov._utils import networkx_import, sklearn_import


def test_semirelaxed_gromov(nx):
    rng = np.random.RandomState(0)
    # unbalanced proportions
    list_n = [30, 15]
    nt = 2
    ns = np.sum(list_n)
    # create directed sbm with C2 as connectivity matrix
    C1 = np.zeros((ns, ns), dtype=np.float64)
    C2 = np.array([[0.8, 0.1], [0.1, 1.0]], dtype=np.float64)

    pos = [0, 30, 45]

    for i in range(nt):
        for j in range(nt):
            ni, nj = list_n[i], list_n[j]
            xij = rng.binomial(size=(ni, nj), n=1, p=C2[i, j])
            pos_i_min, pos_i_max = pos[i], pos[i + 1]
            pos_j_min, pos_j_max = pos[j], pos[j + 1]
            C1[pos_i_min:pos_i_max, pos_j_min:pos_j_max] = xij

    p = ot.unif(ns, type_as=C1)
    q0 = ot.unif(C2.shape[0], type_as=C1)
    G0 = p[:, None] * q0[None, :]
    # asymmetric
    C1b, C2b, pb, q0b, G0b = nx.from_numpy(C1, C2, p, q0, G0)

    for loss_fun in ["square_loss", "kl_loss"]:
        G, log = ot.gromov.semirelaxed_gromov_wasserstein(
            C1, C2, p, loss_fun="square_loss", symmetric=None, log=True, G0=G0
        )
        Gb, logb = ot.gromov.semirelaxed_gromov_wasserstein(
            C1b,
            C2b,
            None,
            loss_fun="square_loss",
            symmetric=False,
            log=True,
            G0=None,
            alpha_min=0.0,
            alpha_max=1.0,
        )

        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(p, nx.sum(Gb, axis=1), atol=1e-04)
        np.testing.assert_allclose(list_n / ns, np.sum(G, axis=0), atol=1e-01)
        np.testing.assert_allclose(list_n / ns, nx.sum(Gb, axis=0), atol=1e-01)

        srgw, log2 = ot.gromov.semirelaxed_gromov_wasserstein2(
            C1, C2, None, loss_fun="square_loss", symmetric=False, log=True, G0=G0
        )
        srgwb, logb2 = ot.gromov.semirelaxed_gromov_wasserstein2(
            C1b, C2b, pb, loss_fun="square_loss", symmetric=None, log=True, G0=None
        )

        G = log2["T"]
        Gb = nx.to_numpy(logb2["T"])
        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
        np.testing.assert_allclose(
            list_n / ns, Gb.sum(0), atol=1e-04
        )  # cf convergence gromov

        np.testing.assert_allclose(log2["srgw_dist"], logb["srgw_dist"], atol=1e-07)
        np.testing.assert_allclose(logb2["srgw_dist"], log["srgw_dist"], atol=1e-07)

    # symmetric - testing various initialization of the OT plan.
    C1 = 0.5 * (C1 + C1.T)

    C1b, C2b, pb, q0b, G0b = nx.from_numpy(C1, C2, p, q0, G0)

    init_plan_list = [
        (None, G0b),
        ("product", None),
        ("random_product", "random_product"),
    ]

    if networkx_import:
        init_plan_list += [("fluid", "fluid"), ("fluid_soft", "fluid_soft")]

    if sklearn_import:
        init_plan_list += [
            ("spectral", "spectral"),
            ("spectral_soft", "spectral_soft"),
            ("kmeans", "kmeans"),
            ("kmeans_soft", "kmeans_soft"),
        ]

    for init, init_b in init_plan_list:
        G, log = ot.gromov.semirelaxed_gromov_wasserstein(
            C1, C2, p, loss_fun="square_loss", symmetric=None, log=True, G0=init
        )
        Gb = ot.gromov.semirelaxed_gromov_wasserstein(
            C1b, C2b, pb, loss_fun="square_loss", symmetric=True, log=False, G0=init_b
        )

        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(
            p, nx.sum(Gb, axis=1), atol=1e-04
        )  # cf convergence gromov

        if not isinstance(init, str):
            np.testing.assert_allclose(
                list_n / ns, nx.sum(Gb, axis=0), atol=1e-02
            )  # cf convergence gromov
        else:
            if (
                "spectral" not in init
            ):  # issues with spectral clustering related to label switching
                np.testing.assert_allclose(list_n / ns, nx.sum(Gb, axis=0), atol=1e-02)

    srgw, log2 = ot.gromov.semirelaxed_gromov_wasserstein2(
        C1, C2, p, loss_fun="square_loss", symmetric=True, log=True, G0=G0
    )
    srgwb, logb2 = ot.gromov.semirelaxed_gromov_wasserstein2(
        C1b, C2b, pb, loss_fun="square_loss", symmetric=None, log=True, G0=None
    )

    srgw_ = ot.gromov.semirelaxed_gromov_wasserstein2(
        C1, C2, p, loss_fun="square_loss", symmetric=True, log=False, G0=G0
    )

    G = log2["T"]
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, nx.sum(Gb, 1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(list_n / ns, np.sum(G, axis=0), atol=1e-01)
    np.testing.assert_allclose(list_n / ns, nx.sum(Gb, axis=0), atol=1e-01)

    np.testing.assert_allclose(log2["srgw_dist"], log["srgw_dist"], atol=1e-07)
    np.testing.assert_allclose(logb2["srgw_dist"], log["srgw_dist"], atol=1e-07)
    np.testing.assert_allclose(srgw, srgw_, atol=1e-07)


def test_semirelaxed_gromov2_gradients():
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)

    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=5)

    p = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    if torch:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for loss_fun in ["square_loss", "kl_loss"]:
                # semirelaxed solvers do not support gradients over masses yet.
                p1 = torch.tensor(p, requires_grad=False, device=device)
                C11 = torch.tensor(C1, requires_grad=True, device=device)
                C12 = torch.tensor(C2, requires_grad=True, device=device)

                val = ot.gromov.semirelaxed_gromov_wasserstein2(
                    C11, C12, p1, loss_fun=loss_fun
                )

                val.backward()

                assert val.device == p1.device
                assert p1.grad is None
                assert C11.shape == C11.grad.shape
                assert C12.shape == C12.grad.shape


def test_srgw_helper_backend(nx):
    n_samples = 20  # nb samples

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=0)
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=1)

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    for loss_fun in ["square_loss", "kl_loss"]:
        C1b, C2b, pb, qb = nx.from_numpy(C1, C2, p, q)
        Gb, logb = ot.gromov.semirelaxed_gromov_wasserstein(
            C1b, C2b, pb, loss_fun, armijo=False, symmetric=True, G0=None, log=True
        )

        # calls with nx=None
        constCb, hC1b, hC2b, fC2tb = ot.gromov.init_matrix_semirelaxed(
            C1b, C2b, pb, loss_fun
        )
        ones_pb = nx.ones(pb.shape[0], type_as=pb)

        def f(G):
            qG = nx.sum(G, 0)
            marginal_product = nx.outer(ones_pb, nx.dot(qG, fC2tb))
            return ot.gromov.gwloss(constCb + marginal_product, hC1b, hC2b, G, nx=None)

        def df(G):
            qG = nx.sum(G, 0)
            marginal_product = nx.outer(ones_pb, nx.dot(qG, fC2tb))
            return ot.gromov.gwggrad(constCb + marginal_product, hC1b, hC2b, G, nx=None)

        def line_search(cost, G, deltaG, Mi, cost_G, df_G):
            return ot.gromov.solve_semirelaxed_gromov_linesearch(
                G, deltaG, cost_G, hC1b, hC2b, ones_pb, 0.0, 1.0, fC2t=fC2tb, nx=None
            )

        # feed the precomputed local optimum Gb to semirelaxed_cg
        res, log = ot.optim.semirelaxed_cg(
            pb,
            qb,
            0.0,
            1.0,
            f,
            df,
            Gb,
            line_search,
            log=True,
            numItermax=1e4,
            stopThr=1e-9,
            stopThr2=1e-9,
        )
        # check constraints
        np.testing.assert_allclose(res, Gb, atol=1e-06)


@pytest.mark.parametrize(
    "loss_fun",
    [
        "square_loss",
        "kl_loss",
        pytest.param("unknown_loss", marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_gw_semirelaxed_helper_validation(loss_fun):
    n_samples = 20  # nb samples
    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=0)
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=1)
    p = ot.unif(n_samples)
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    ot.gromov.init_matrix_semirelaxed(C1, C2, p, loss_fun=loss_fun)


def test_semirelaxed_fgw(nx):
    rng = np.random.RandomState(0)
    list_n = [16, 8]
    nt = 2
    ns = 24
    # create directed sbm with C2 as connectivity matrix
    C1 = np.zeros((ns, ns))
    C2 = np.array([[0.7, 0.05], [0.05, 0.9]])

    pos = [0, 16, 24]

    for i in range(nt):
        for j in range(nt):
            ni, nj = list_n[i], list_n[j]
            xij = rng.binomial(size=(ni, nj), n=1, p=C2[i, j])
            pos_i_min, pos_i_max = pos[i], pos[i + 1]
            pos_j_min, pos_j_max = pos[j], pos[j + 1]
            C1[pos_i_min:pos_i_max, pos_j_min:pos_j_max] = xij

    F1 = np.zeros((ns, 1))
    F1[:16] = rng.normal(loc=0.0, scale=0.01, size=(16, 1))
    F1[16:] = rng.normal(loc=1.0, scale=0.01, size=(8, 1))
    F2 = np.zeros((2, 1))
    F2[1, :] = 1.0
    M = (
        (F1**2).dot(np.ones((1, nt)))
        + np.ones((ns, 1)).dot((F2**2).T)
        - 2 * F1.dot(F2.T)
    )

    p = ot.unif(ns)
    q0 = ot.unif(C2.shape[0])
    G0 = p[:, None] * q0[None, :]

    # asymmetric structure - checking constraints and values
    Mb, C1b, C2b, pb, q0b, G0b = nx.from_numpy(M, C1, C2, p, q0, G0)
    G, log = ot.gromov.semirelaxed_fused_gromov_wasserstein(
        M,
        C1,
        C2,
        None,
        loss_fun="square_loss",
        alpha=0.5,
        symmetric=None,
        log=True,
        G0=None,
    )
    Gb, logb = ot.gromov.semirelaxed_fused_gromov_wasserstein(
        Mb,
        C1b,
        C2b,
        pb,
        loss_fun="square_loss",
        alpha=0.5,
        symmetric=False,
        log=True,
        G0=G0b,
    )

    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, nx.sum(Gb, axis=1), atol=1e-04
    )  # cf convergence gromov
    np.testing.assert_allclose(
        [2 / 3, 1 / 3], nx.sum(Gb, axis=0), atol=1e-02
    )  # cf convergence gromov

    # asymmetric - check consistency between srFGW and srFGW2
    srgw, log2 = ot.gromov.semirelaxed_fused_gromov_wasserstein2(
        M,
        C1,
        C2,
        p,
        loss_fun="square_loss",
        alpha=0.5,
        symmetric=False,
        log=True,
        G0=G0,
    )
    srgwb, logb2 = ot.gromov.semirelaxed_fused_gromov_wasserstein2(
        Mb,
        C1b,
        C2b,
        None,
        loss_fun="square_loss",
        alpha=0.5,
        symmetric=None,
        log=True,
        G0=None,
    )

    G = log2["T"]
    Gb = nx.to_numpy(logb2["T"])

    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        [2 / 3, 1 / 3], G.sum(0), atol=1e-04
    )  # cf convergence gromov

    np.testing.assert_allclose(log2["srfgw_dist"], logb["srfgw_dist"], atol=1e-07)
    np.testing.assert_allclose(logb2["srfgw_dist"], log["srfgw_dist"], atol=1e-07)

    # symmetric structures + checking losses + inits
    C1 = 0.5 * (C1 + C1.T)
    Mb, C1b, C2b, pb, q0b, G0b = nx.from_numpy(M, C1, C2, p, q0, G0)

    init_plan_list = [
        (None, G0b),
        ("product", None),
        ("random_product", "random_product"),
    ]

    if networkx_import:
        init_plan_list += [("fluid", "fluid")]

    if sklearn_import:
        init_plan_list += [("kmeans", "kmeans")]

    for loss_fun in ["square_loss", "kl_loss"]:
        for init, init_b in init_plan_list:
            G, log = ot.gromov.semirelaxed_fused_gromov_wasserstein(
                M,
                C1,
                C2,
                p,
                loss_fun=loss_fun,
                alpha=0.5,
                symmetric=None,
                log=True,
                G0=init,
            )
            Gb = ot.gromov.semirelaxed_fused_gromov_wasserstein(
                Mb,
                C1b,
                C2b,
                pb,
                loss_fun=loss_fun,
                alpha=0.5,
                symmetric=True,
                log=False,
                G0=init_b,
            )

            np.testing.assert_allclose(G, Gb, atol=1e-06)
            np.testing.assert_allclose(
                p, nx.sum(Gb, axis=1), atol=1e-04
            )  # cf convergence gromov
            np.testing.assert_allclose(
                [2 / 3, 1 / 3], nx.sum(Gb, axis=0), atol=1e-02
            )  # cf convergence gromov

        # checking consistency with srFGW and srFGW2 solvers

        srgw, log2 = ot.gromov.semirelaxed_fused_gromov_wasserstein2(
            M, C1, C2, p, loss_fun=loss_fun, alpha=0.5, symmetric=True, log=True, G0=G0
        )
        srgwb, logb2 = ot.gromov.semirelaxed_fused_gromov_wasserstein2(
            Mb,
            C1b,
            C2b,
            pb,
            loss_fun=loss_fun,
            alpha=0.5,
            symmetric=None,
            log=True,
            G0=None,
        )

        G2 = log2["T"]
        Gb2 = nx.to_numpy(logb2["T"])
        # check constraints
        np.testing.assert_allclose(G2, Gb2, atol=1e-06)
        np.testing.assert_allclose(G2, G, atol=1e-06)

        np.testing.assert_allclose(log2["srfgw_dist"], log["srfgw_dist"], atol=1e-07)
        np.testing.assert_allclose(logb2["srfgw_dist"], log["srfgw_dist"], atol=1e-07)
        np.testing.assert_allclose(srgw, srgwb, atol=1e-07)


def test_semirelaxed_fgw2_gradients():
    n_samples = 20  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)

    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=5)

    p = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)
    M = ot.dist(xs, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    if torch:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            # semirelaxed solvers do not support gradients over masses yet.
            for loss_fun in ["square_loss", "kl_loss"]:
                p1 = torch.tensor(p, requires_grad=False, device=device)
                C11 = torch.tensor(C1, requires_grad=True, device=device)
                C12 = torch.tensor(C2, requires_grad=True, device=device)
                M1 = torch.tensor(M, requires_grad=True, device=device)

                val = ot.gromov.semirelaxed_fused_gromov_wasserstein2(
                    M1, C11, C12, p1, loss_fun=loss_fun
                )

                val.backward()

                assert val.device == p1.device
                assert p1.grad is None
                assert C11.shape == C11.grad.shape
                assert C12.shape == C12.grad.shape
                assert M1.shape == M1.grad.shape

                # full gradients with alpha
                p1 = torch.tensor(p, requires_grad=False, device=device)
                C11 = torch.tensor(C1, requires_grad=True, device=device)
                C12 = torch.tensor(C2, requires_grad=True, device=device)
                M1 = torch.tensor(M, requires_grad=True, device=device)
                alpha = torch.tensor(0.5, requires_grad=True, device=device)

                val = ot.gromov.semirelaxed_fused_gromov_wasserstein2(
                    M1, C11, C12, p1, loss_fun=loss_fun, alpha=alpha
                )

                val.backward()

                assert val.device == p1.device
                assert p1.grad is None
                assert C11.shape == C11.grad.shape
                assert C12.shape == C12.grad.shape
                assert alpha.shape == alpha.grad.shape


def test_srfgw_helper_backend(nx):
    n_samples = 20  # nb samples

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])

    rng = np.random.RandomState(42)
    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=0)
    ys = rng.randn(xs.shape[0], 2)
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=1)
    yt = rng.randn(xt.shape[0], 2)

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    M = ot.dist(ys, yt)
    M /= M.max()

    Mb, C1b, C2b, pb, qb, G0b = nx.from_numpy(M, C1, C2, p, q, G0)
    alpha = 0.5
    Gb, logb = ot.gromov.semirelaxed_fused_gromov_wasserstein(
        Mb,
        C1b,
        C2b,
        pb,
        "square_loss",
        alpha=0.5,
        armijo=False,
        symmetric=True,
        G0=G0b,
        log=True,
    )

    # calls with nx=None
    constCb, hC1b, hC2b, fC2tb = ot.gromov.init_matrix_semirelaxed(
        C1b, C2b, pb, loss_fun="square_loss"
    )
    ones_pb = nx.ones(pb.shape[0], type_as=pb)

    def f(G):
        qG = nx.sum(G, 0)
        marginal_product = nx.outer(ones_pb, nx.dot(qG, fC2tb))
        return ot.gromov.gwloss(constCb + marginal_product, hC1b, hC2b, G, nx=None)

    def df(G):
        qG = nx.sum(G, 0)
        marginal_product = nx.outer(ones_pb, nx.dot(qG, fC2tb))
        return ot.gromov.gwggrad(constCb + marginal_product, hC1b, hC2b, G, nx=None)

    def line_search(cost, G, deltaG, Mi, cost_G, df_G):
        return ot.gromov.solve_semirelaxed_gromov_linesearch(
            G, deltaG, cost_G, C1b, C2b, ones_pb, M=(1 - alpha) * Mb, reg=alpha, nx=None
        )

    # feed the precomputed local optimum Gb to semirelaxed_cg
    res, log = ot.optim.semirelaxed_cg(
        pb,
        qb,
        (1 - alpha) * Mb,
        alpha,
        f,
        df,
        Gb,
        line_search,
        log=True,
        numItermax=1e4,
        stopThr=1e-9,
        stopThr2=1e-9,
    )
    # check constraints
    np.testing.assert_allclose(res, Gb, atol=1e-06)


def test_entropic_semirelaxed_gromov(nx):
    # unbalanced proportions
    list_n = [30, 15]
    nt = 2
    ns = np.sum(list_n)
    # create directed sbm with C2 as connectivity matrix
    C1 = np.zeros((ns, ns), dtype=np.float64)
    C2 = np.array([[0.8, 0.1], [0.1, 0.9]], dtype=np.float64)

    rng = np.random.RandomState(0)

    pos = [0, 30, 45]

    for i in range(nt):
        for j in range(nt):
            ni, nj = list_n[i], list_n[j]
            xij = rng.binomial(size=(ni, nj), n=1, p=C2[i, j])
            pos_i_min, pos_i_max = pos[i], pos[i + 1]
            pos_j_min, pos_j_max = pos[j], pos[j + 1]
            C1[pos_i_min:pos_i_max, pos_j_min:pos_j_max] = xij

    p = ot.unif(ns, type_as=C1)
    q0 = ot.unif(C2.shape[0], type_as=C1)
    G0 = p[:, None] * q0[None, :]
    # asymmetric
    C1b, C2b, pb, q0b, G0b = nx.from_numpy(C1, C2, p, q0, G0)
    epsilon = 0.1
    for loss_fun in ["square_loss", "kl_loss"]:
        G, log = ot.gromov.entropic_semirelaxed_gromov_wasserstein(
            C1,
            C2,
            p,
            loss_fun=loss_fun,
            epsilon=epsilon,
            symmetric=None,
            log=True,
            G0=G0,
        )
        Gb, logb = ot.gromov.entropic_semirelaxed_gromov_wasserstein(
            C1b,
            C2b,
            None,
            loss_fun=loss_fun,
            epsilon=epsilon,
            symmetric=False,
            log=True,
            G0=None,
        )

        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(p, nx.sum(Gb, axis=1), atol=1e-04)
        np.testing.assert_allclose(list_n / ns, np.sum(G, axis=0), atol=1e-01)
        np.testing.assert_allclose(list_n / ns, nx.sum(Gb, axis=0), atol=1e-01)

        srgw, log2 = ot.gromov.entropic_semirelaxed_gromov_wasserstein2(
            C1,
            C2,
            None,
            loss_fun=loss_fun,
            epsilon=epsilon,
            symmetric=False,
            log=True,
            G0=G0,
        )
        srgwb, logb2 = ot.gromov.entropic_semirelaxed_gromov_wasserstein2(
            C1b,
            C2b,
            pb,
            loss_fun=loss_fun,
            epsilon=epsilon,
            symmetric=None,
            log=True,
            G0=None,
        )

        G = log2["T"]
        Gb = nx.to_numpy(logb2["T"])
        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
        np.testing.assert_allclose(
            list_n / ns, Gb.sum(0), atol=1e-04
        )  # cf convergence gromov

        np.testing.assert_allclose(log2["srgw_dist"], logb["srgw_dist"], atol=1e-07)
        np.testing.assert_allclose(logb2["srgw_dist"], log["srgw_dist"], atol=1e-07)

    # symmetric - testing various initialization of the OT plan.

    C1 = 0.5 * (C1 + C1.T)
    C1b, C2b, pb, q0b, G0b = nx.from_numpy(C1, C2, p, q0, G0)

    init_plan_list = []  # tests longer than with CG so we do not test all inits.

    if networkx_import:
        init_plan_list += [("fluid", "fluid")]

    if sklearn_import:
        init_plan_list += [("kmeans", "kmeans")]

    init_plan_list += [("product", None), (None, G0b)]

    for init, init_b in init_plan_list:
        print(f"---- init : {init} / init_b : {init_b}")
        G, log = ot.gromov.entropic_semirelaxed_gromov_wasserstein(
            C1,
            C2,
            p,
            loss_fun="square_loss",
            epsilon=epsilon,
            symmetric=None,
            log=True,
            G0=init,
        )
        Gb, logb = ot.gromov.entropic_semirelaxed_gromov_wasserstein(
            C1b,
            C2b,
            pb,
            loss_fun="square_loss",
            epsilon=epsilon,
            symmetric=True,
            log=True,
            G0=init_b,
        )

        # check constraints
        np.testing.assert_allclose(G, Gb, atol=1e-06)
        np.testing.assert_allclose(
            p, nx.sum(Gb, axis=1), atol=1e-04
        )  # cf convergence gromov

        if not isinstance(init, str):
            np.testing.assert_allclose(
                list_n / ns, nx.sum(Gb, axis=0), atol=1e-02
            )  # cf convergence gromov
        else:
            np.testing.assert_allclose(list_n / ns, nx.sum(Gb, axis=0), atol=1e-02)

    # comparison between srGW and srGW2 solvers

    srgw, log2 = ot.gromov.entropic_semirelaxed_gromov_wasserstein2(
        C1,
        C2,
        p,
        loss_fun="square_loss",
        epsilon=epsilon,
        symmetric=True,
        log=True,
        G0=init,
    )
    srgwb, logb2 = ot.gromov.entropic_semirelaxed_gromov_wasserstein2(
        C1b,
        C2b,
        pb,
        loss_fun="square_loss",
        epsilon=epsilon,
        symmetric=None,
        log=True,
        G0=init_b,
    )

    srgw_ = ot.gromov.entropic_semirelaxed_gromov_wasserstein2(
        C1,
        C2,
        p,
        loss_fun="square_loss",
        epsilon=epsilon,
        symmetric=True,
        log=False,
        G0=G0,
    )

    G2 = log2["T"]
    G2b = logb2["T"]
    # check constraints
    np.testing.assert_allclose(G2, G2b, atol=1e-06)
    np.testing.assert_allclose(G2, G, atol=1e-06)

    np.testing.assert_allclose(log2["srgw_dist"], log["srgw_dist"], atol=1e-07)
    np.testing.assert_allclose(logb2["srgw_dist"], log["srgw_dist"], atol=1e-07)
    np.testing.assert_allclose(srgw, srgw_, atol=1e-07)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_semirelaxed_gromov_dtype_device(nx):
    # setup
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))
        for loss_fun in ["square_loss", "kl_loss"]:
            C1b, C2b, pb = nx.from_numpy(C1, C2, p, type_as=tp)

            Gb = ot.gromov.entropic_semirelaxed_gromov_wasserstein(
                C1b, C2b, pb, loss_fun, epsilon=0.1, verbose=True
            )
            gw_valb = ot.gromov.entropic_semirelaxed_gromov_wasserstein2(
                C1b, C2b, pb, loss_fun, epsilon=0.1, verbose=True
            )

            nx.assert_same_dtype_device(C1b, Gb)
            nx.assert_same_dtype_device(C1b, gw_valb)


def test_entropic_semirelaxed_fgw(nx):
    rng = np.random.RandomState(0)
    list_n = [16, 8]
    nt = 2
    ns = 24
    # create directed sbm with C2 as connectivity matrix
    C1 = np.zeros((ns, ns))
    C2 = np.array([[0.7, 0.05], [0.05, 0.9]])

    pos = [0, 16, 24]

    for i in range(nt):
        for j in range(nt):
            ni, nj = list_n[i], list_n[j]
            xij = rng.binomial(size=(ni, nj), n=1, p=C2[i, j])
            pos_i_min, pos_i_max = pos[i], pos[i + 1]
            pos_j_min, pos_j_max = pos[j], pos[j + 1]
            C1[pos_i_min:pos_i_max, pos_j_min:pos_j_max] = xij

    F1 = np.zeros((ns, 1))
    F1[:16] = rng.normal(loc=0.0, scale=0.01, size=(16, 1))
    F1[16:] = rng.normal(loc=1.0, scale=0.01, size=(8, 1))
    F2 = np.zeros((2, 1))
    F2[1, :] = 1.0
    M = (
        (F1**2).dot(np.ones((1, nt)))
        + np.ones((ns, 1)).dot((F2**2).T)
        - 2 * F1.dot(F2.T)
    )

    p = ot.unif(ns)
    q0 = ot.unif(C2.shape[0])
    G0 = p[:, None] * q0[None, :]

    # asymmetric structure - checking constraints and values
    Mb, C1b, C2b, pb, q0b, G0b = nx.from_numpy(M, C1, C2, p, q0, G0)

    G, log = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
        M,
        C1,
        C2,
        None,
        loss_fun="square_loss",
        epsilon=0.1,
        alpha=0.5,
        symmetric=None,
        log=True,
        G0=None,
    )
    Gb, logb = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
        Mb,
        C1b,
        C2b,
        pb,
        loss_fun="square_loss",
        epsilon=0.1,
        alpha=0.5,
        symmetric=False,
        log=True,
        G0=G0b,
    )

    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, nx.sum(Gb, axis=1), atol=1e-04
    )  # cf convergence gromov
    np.testing.assert_allclose(
        [2 / 3, 1 / 3], nx.sum(Gb, axis=0), atol=1e-02
    )  # cf convergence gromov

    srgw, log2 = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein2(
        M,
        C1,
        C2,
        p,
        loss_fun="square_loss",
        epsilon=0.1,
        alpha=0.5,
        symmetric=False,
        log=True,
        G0=G0,
    )
    srgwb, logb2 = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein2(
        Mb,
        C1b,
        C2b,
        None,
        loss_fun="square_loss",
        epsilon=0.1,
        alpha=0.5,
        symmetric=None,
        log=True,
        G0=None,
    )

    G = log2["T"]
    Gb = nx.to_numpy(logb2["T"])
    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        [2 / 3, 1 / 3], Gb.sum(0), atol=1e-04
    )  # cf convergence gromov

    np.testing.assert_allclose(log2["srfgw_dist"], logb["srfgw_dist"], atol=1e-07)
    np.testing.assert_allclose(logb2["srfgw_dist"], log["srfgw_dist"], atol=1e-07)

    # symmetric structures + checking losses + inits
    C1 = 0.5 * (C1 + C1.T)
    Mb, C1b, C2b, pb, q0b, G0b = nx.from_numpy(M, C1, C2, p, q0, G0)

    init_plan_list = [
        (None, G0b),
        ("product", None),
        ("random_product", "random_product"),
    ]

    if networkx_import:
        init_plan_list += [("fluid", "fluid")]

    if sklearn_import:
        init_plan_list += [("kmeans", "kmeans")]

    for loss_fun in ["square_loss", "kl_loss"]:
        for init, init_b in init_plan_list:
            G, log = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
                M,
                C1,
                C2,
                p,
                loss_fun=loss_fun,
                epsilon=0.1,
                alpha=0.5,
                symmetric=None,
                log=True,
                G0=init,
            )
            Gb = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
                Mb,
                C1b,
                C2b,
                pb,
                loss_fun=loss_fun,
                epsilon=0.1,
                alpha=0.5,
                symmetric=True,
                log=False,
                G0=init_b,
            )

            np.testing.assert_allclose(G, Gb, atol=1e-06)
            np.testing.assert_allclose(
                p, nx.sum(Gb, axis=1), atol=1e-04
            )  # cf convergence gromov
            np.testing.assert_allclose(
                [2 / 3, 1 / 3], nx.sum(Gb, axis=0), atol=1e-02
            )  # cf convergence gromov

        # checking consistency with srFGW and srFGW2 solvers
        srgw, log2 = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein2(
            M,
            C1,
            C2,
            p,
            loss_fun=loss_fun,
            epsilon=0.1,
            alpha=0.5,
            symmetric=True,
            log=True,
            G0=init,
        )
        srgwb, logb2 = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein2(
            Mb,
            C1b,
            C2b,
            pb,
            loss_fun=loss_fun,
            epsilon=0.1,
            alpha=0.5,
            symmetric=None,
            log=True,
            G0=init_b,
        )

        G2 = log2["T"]
        Gb2 = nx.to_numpy(logb2["T"])
        np.testing.assert_allclose(G2, Gb2, atol=1e-06)
        np.testing.assert_allclose(G2, G, atol=1e-06)
        np.testing.assert_allclose(p, Gb2.sum(1), atol=1e-04)  # cf convergence gromov
        np.testing.assert_allclose(
            [2 / 3, 1 / 3], Gb2.sum(0), atol=1e-04
        )  # cf convergence gromov

        np.testing.assert_allclose(log2["srfgw_dist"], log["srfgw_dist"], atol=1e-07)
        np.testing.assert_allclose(logb2["srfgw_dist"], log["srfgw_dist"], atol=1e-07)
        np.testing.assert_allclose(srgw, srgwb, atol=1e-07)


@pytest.skip_backend("tf", reason="test very slow with tf backend")
def test_entropic_semirelaxed_fgw_dtype_device(nx):
    # setup
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    rng = np.random.RandomState(42)
    ys = rng.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    M = ot.dist(ys, yt)
    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        Mb, C1b, C2b, pb = nx.from_numpy(M, C1, C2, p, type_as=tp)

        for loss_fun in ["square_loss", "kl_loss"]:
            Gb = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
                Mb, C1b, C2b, pb, loss_fun, epsilon=0.1, verbose=True
            )
            fgw_valb = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein2(
                Mb, C1b, C2b, pb, loss_fun, epsilon=0.1, verbose=True
            )

            nx.assert_same_dtype_device(C1b, Gb)
            nx.assert_same_dtype_device(C1b, fgw_valb)


@pytest.skip_backend("tf", reason="test very slow with tf backend")
@pytest.skip_backend("jax", reason="test very slow with tf backend")
def test_semirelaxed_gromov_barycenter(nx):
    ns = 5
    nt = 8

    Xs, ys = ot.datasets.make_data_classif("3gauss", ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif("3gauss2", nt, random_state=42)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)
    p1 = ot.unif(ns)
    p2 = ot.unif(nt)
    n_samples = 3

    C1b, C2b, p1b, p2b = nx.from_numpy(C1, C2, p1, p2)

    # test on admissible stopping criterion
    with pytest.raises(ValueError):
        stop_criterion = "unknown stop criterion"
        _ = ot.gromov.semirelaxed_gromov_barycenters(
            n_samples,
            [C1, C2],
            None,
            [0.5, 0.5],
            "square_loss",
            max_iter=10,
            tol=1e-3,
            stop_criterion=stop_criterion,
            verbose=False,
            random_state=42,
        )

    # test consistency of outputs across backends with 'square_loss'
    # using different losses
    # + tests on different inits
    init_plan_list = [("fluid", "fluid"), ("kmeans", "kmeans"), ("random", "random")]

    for init, init_b in init_plan_list:
        for stop_criterion in ["barycenter", "loss"]:
            print("--- stop_criterion:", stop_criterion)

            if (init == "fluid") and (not networkx_import):
                with pytest.raises(ValueError):
                    Cb = ot.gromov.semirelaxed_gromov_barycenters(
                        n_samples,
                        [C1, C2],
                        None,
                        [0.5, 0.5],
                        "square_loss",
                        max_iter=5,
                        tol=1e-3,
                        stop_criterion=stop_criterion,
                        verbose=False,
                        random_state=42,
                        G0=init,
                    )

            elif (init == "kmeans") and (not sklearn_import):
                with pytest.raises(ValueError):
                    Cb = ot.gromov.semirelaxed_gromov_barycenters(
                        n_samples,
                        [C1, C2],
                        None,
                        [0.5, 0.5],
                        "square_loss",
                        max_iter=5,
                        tol=1e-3,
                        stop_criterion=stop_criterion,
                        verbose=False,
                        random_state=42,
                        G0=init,
                    )
            else:
                Cb = ot.gromov.semirelaxed_gromov_barycenters(
                    n_samples,
                    [C1, C2],
                    None,
                    [0.5, 0.5],
                    "square_loss",
                    max_iter=5,
                    tol=1e-3,
                    stop_criterion=stop_criterion,
                    verbose=False,
                    random_state=42,
                    G0=init,
                )

                Cbb = nx.to_numpy(
                    ot.gromov.semirelaxed_gromov_barycenters(
                        n_samples,
                        [C1b, C2b],
                        [p1b, p2b],
                        [0.5, 0.5],
                        "square_loss",
                        max_iter=5,
                        tol=1e-3,
                        stop_criterion=stop_criterion,
                        verbose=False,
                        random_state=42,
                        G0=init_b,
                    )
                )
                np.testing.assert_allclose(Cb, Cbb, atol=1e-06)
                np.testing.assert_allclose(Cbb.shape, (n_samples, n_samples))

                # test of gromov_barycenters with `log` on
                Cb_, err_ = ot.gromov.semirelaxed_gromov_barycenters(
                    n_samples,
                    [C1, C2],
                    [p1, p2],
                    None,
                    "square_loss",
                    max_iter=5,
                    tol=1e-3,
                    stop_criterion=stop_criterion,
                    verbose=False,
                    warmstartT=True,
                    random_state=42,
                    log=True,
                    G0=init,
                )
                Cbb_, errb_ = ot.gromov.semirelaxed_gromov_barycenters(
                    n_samples,
                    [C1b, C2b],
                    [p1b, p2b],
                    [0.5, 0.5],
                    "square_loss",
                    max_iter=5,
                    tol=1e-3,
                    stop_criterion=stop_criterion,
                    verbose=False,
                    warmstartT=True,
                    random_state=42,
                    log=True,
                    G0=init_b,
                )

                Cbb_ = nx.to_numpy(Cbb_)

                np.testing.assert_allclose(Cb_, Cbb_, atol=1e-06)
                np.testing.assert_array_almost_equal(
                    err_["err"], nx.to_numpy(*errb_["err"])
                )
                np.testing.assert_allclose(Cbb_.shape, (n_samples, n_samples))

    # test consistency across backends with larger barycenter than inputs
    if sklearn_import:
        C = ot.gromov.semirelaxed_gromov_barycenters(
            ns,
            [C1, C2],
            None,
            [0.5, 0.5],
            "square_loss",
            max_iter=5,
            tol=1e-3,
            stop_criterion="loss",
            verbose=False,
            random_state=42,
            G0="kmeans",
        )
        Cb = ot.gromov.semirelaxed_gromov_barycenters(
            ns,
            [C1b, C2b],
            [p1b, p2b],
            [0.5, 0.5],
            "square_loss",
            max_iter=5,
            tol=1e-3,
            stop_criterion=stop_criterion,
            verbose=False,
            random_state=42,
            G0="kmeans",
        )

        np.testing.assert_allclose(C, nx.to_numpy(Cb), atol=1e-06)

    # test providing init_C
    C_ = ot.gromov.semirelaxed_gromov_barycenters(
        ns,
        [C1, C2],
        None,
        [0.5, 0.5],
        "square_loss",
        max_iter=5,
        tol=1e-3,
        stop_criterion="loss",
        verbose=False,
        random_state=42,
        G0=init,
        init_C=C1,
    )

    Cb_ = ot.gromov.semirelaxed_gromov_barycenters(
        ns,
        [C1b, C2b],
        [p1b, p2b],
        [0.5, 0.5],
        "square_loss",
        max_iter=5,
        tol=1e-3,
        stop_criterion=stop_criterion,
        verbose=False,
        random_state=42,
        G0=init_b,
        init_C=C1b,
    )

    np.testing.assert_allclose(C_, Cb_, atol=1e-06)

    # test consistency across backends with 'kl_loss'
    Cb2, err = ot.gromov.semirelaxed_gromov_barycenters(
        n_samples,
        [C1, C2],
        [p1, p2],
        [0.5, 0.5],
        "kl_loss",
        max_iter=5,
        tol=1e-3,
        warmstartT=False,
        stop_criterion="loss",
        log=True,
        G0=init_b,
        random_state=42,
    )
    Cb2b, errb = ot.gromov.semirelaxed_gromov_barycenters(
        n_samples,
        [C1b, C2b],
        [p1b, p2b],
        [0.5, 0.5],
        "kl_loss",
        max_iter=5,
        tol=1e-3,
        warmstartT=False,
        stop_criterion="loss",
        log=True,
        G0=init_b,
        random_state=42,
    )
    Cb2b = nx.to_numpy(Cb2b)

    try:
        np.testing.assert_allclose(Cb2, Cb2b, atol=1e-06)  # may differ from permutation
    except AssertionError:
        np.testing.assert_allclose(err["loss"][-1], errb["loss"][-1], atol=1e-06)

    np.testing.assert_allclose(Cb2b.shape, (n_samples, n_samples))

    # test of gromov_barycenters with `log` on
    # providing init_C similarly than in the function.
    rng = ot.utils.check_random_state(42)
    xalea = rng.randn(n_samples, 2)
    init_C = ot.utils.dist(xalea, xalea)
    init_C /= init_C.max()
    init_Cb = nx.from_numpy(init_C)

    Cb2_, err2_ = ot.gromov.semirelaxed_gromov_barycenters(
        n_samples,
        [C1, C2],
        [p1, p2],
        [0.5, 0.5],
        "square_loss",
        max_iter=10,
        tol=1e-3,
        verbose=False,
        random_state=42,
        log=True,
        init_C=init_C,
    )
    Cb2b_, err2b_ = ot.gromov.semirelaxed_gromov_barycenters(
        n_samples,
        [C1b, C2b],
        [p1b, p2b],
        [0.5, 0.5],
        "square_loss",
        max_iter=10,
        tol=1e-3,
        verbose=True,
        random_state=42,
        init_C=init_Cb,
        log=True,
    )
    Cb2b_ = nx.to_numpy(Cb2b_)
    np.testing.assert_allclose(Cb2_, Cb2b_, atol=1e-06)
    np.testing.assert_array_almost_equal(err2_["err"], nx.to_numpy(*err2b_["err"]))
    np.testing.assert_allclose(Cb2b_.shape, (n_samples, n_samples))

    # test edge cases for gw barycenters:
    # unique input structure
    Cb = ot.gromov.semirelaxed_gromov_barycenters(
        n_samples,
        [C1],
        None,
        None,
        "square_loss",
        max_iter=1,
        tol=1e-3,
        stop_criterion=stop_criterion,
        verbose=False,
        random_state=42,
    )
    Cbb = nx.to_numpy(
        ot.gromov.semirelaxed_gromov_barycenters(
            n_samples,
            [C1b],
            None,
            [1.0],
            "square_loss",
            max_iter=1,
            tol=1e-3,
            stop_criterion=stop_criterion,
            verbose=False,
            random_state=42,
        )
    )
    np.testing.assert_allclose(Cb, Cbb, atol=1e-06)
    np.testing.assert_allclose(Cbb.shape, (n_samples, n_samples))


def test_semirelaxed_fgw_barycenter(nx):
    ns = 10
    nt = 20

    Xs, ys = ot.datasets.make_data_classif("3gauss", ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif("3gauss2", nt, random_state=42)

    rng = np.random.RandomState(42)
    ys = rng.randn(Xs.shape[0], 2)
    yt = rng.randn(Xt.shape[0], 2)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)
    C1 /= C1.max()
    C2 /= C2.max()

    p1, p2 = ot.unif(ns), ot.unif(nt)
    n_samples = 3

    ysb, ytb, C1b, C2b, p1b, p2b = nx.from_numpy(ys, yt, C1, C2, p1, p2)

    lambdas = [0.5, 0.5]
    Csb = [C1b, C2b]
    Ysb = [ysb, ytb]
    Xb, Cb, logb = ot.gromov.semirelaxed_fgw_barycenters(
        n_samples,
        Ysb,
        Csb,
        None,
        lambdas,
        0.5,
        fixed_structure=False,
        fixed_features=False,
        loss_fun="square_loss",
        max_iter=10,
        tol=1e-3,
        random_state=12345,
        log=True,
    )
    # test correspondance with utils function
    recovered_Cb = ot.gromov.update_barycenter_structure(logb["T"], Csb, lambdas)
    recovered_Xb = ot.gromov.update_barycenter_feature(logb["T"], Ysb, lambdas)

    np.testing.assert_allclose(Cb, recovered_Cb)
    np.testing.assert_allclose(Xb, recovered_Xb)

    xalea = rng.randn(n_samples, 2)
    init_C = ot.dist(xalea, xalea)
    init_C /= init_C.max()
    init_Cb = nx.from_numpy(init_C)

    with pytest.raises(
        ot.utils.UndefinedParameter
    ):  # to raise an error when `fixed_structure=True`and `init_C=None`
        Xb, Cb, logb = ot.gromov.semirelaxed_fgw_barycenters(
            n_samples,
            Ysb,
            Csb,
            ps=[p1b, p2b],
            lambdas=None,
            alpha=0.5,
            fixed_structure=True,
            init_C=None,
            fixed_features=False,
            loss_fun="square_loss",
            max_iter=10,
            tol=1e-3,
        )

    Xb, Cb = ot.gromov.semirelaxed_fgw_barycenters(
        n_samples,
        [ysb, ytb],
        [C1b, C2b],
        ps=[p1b, p2b],
        lambdas=None,
        alpha=0.5,
        fixed_structure=True,
        init_C=init_Cb,
        fixed_features=False,
        loss_fun="square_loss",
        max_iter=10,
        tol=1e-3,
    )
    Xb, Cb = nx.to_numpy(Xb), nx.to_numpy(Cb)
    np.testing.assert_allclose(Cb.shape, (n_samples, n_samples))
    np.testing.assert_allclose(Xb.shape, (n_samples, ys.shape[1]))

    init_X = rng.randn(n_samples, ys.shape[1])
    init_Xb = nx.from_numpy(init_X)

    # Tests with `fixed_structure=False` and `fixed_features=True`
    with pytest.raises(
        ot.utils.UndefinedParameter
    ):  # to raise an error when `fixed_features=True`and `init_X=None`
        Xb, Cb, logb = ot.gromov.semirelaxed_fgw_barycenters(
            n_samples,
            [ysb, ytb],
            [C1b, C2b],
            [p1b, p2b],
            [0.5, 0.5],
            0.5,
            fixed_structure=False,
            fixed_features=True,
            init_X=None,
            loss_fun="square_loss",
            max_iter=10,
            tol=1e-3,
            warmstartT=True,
            log=True,
            random_state=98765,
            verbose=True,
        )
    Xb, Cb, logb = ot.gromov.semirelaxed_fgw_barycenters(
        n_samples,
        [ysb, ytb],
        [C1b, C2b],
        [p1b, p2b],
        [0.5, 0.5],
        0.5,
        fixed_structure=False,
        fixed_features=True,
        init_X=init_Xb,
        loss_fun="square_loss",
        max_iter=10,
        tol=1e-3,
        warmstartT=True,
        log=True,
        random_state=98765,
        verbose=True,
    )

    X, C = nx.to_numpy(Xb), nx.to_numpy(Cb)
    np.testing.assert_allclose(C.shape, (n_samples, n_samples))
    np.testing.assert_allclose(X.shape, (n_samples, ys.shape[1]))

    # add test with 'kl_loss'
    with pytest.raises(ValueError):
        stop_criterion = "unknown stop criterion"
        X, C, log = ot.gromov.semirelaxed_fgw_barycenters(
            n_samples,
            [ys, yt],
            [C1, C2],
            [p1, p2],
            [0.5, 0.5],
            0.5,
            fixed_structure=False,
            fixed_features=False,
            loss_fun="kl_loss",
            max_iter=10,
            tol=1e-3,
            stop_criterion=stop_criterion,
            init_C=C,
            init_X=X,
            warmstartT=True,
            random_state=12345,
            log=True,
        )

    for stop_criterion in ["barycenter", "loss"]:
        X, C, log = ot.gromov.semirelaxed_fgw_barycenters(
            n_samples,
            [ys, yt],
            [C1, C2],
            [p1, p2],
            [0.5, 0.5],
            0.5,
            fixed_structure=False,
            fixed_features=False,
            loss_fun="kl_loss",
            max_iter=10,
            tol=1e-3,
            stop_criterion=stop_criterion,
            init_C=C,
            init_X=X,
            warmstartT=True,
            random_state=12345,
            log=True,
            verbose=True,
        )
        np.testing.assert_allclose(C.shape, (n_samples, n_samples))
        np.testing.assert_allclose(X.shape, (n_samples, ys.shape[1]))

    # test correspondance with utils function

    recovered_C = ot.gromov.update_barycenter_structure(
        log["T"], [C1, C2], lambdas, None, "kl_loss", True
    )

    np.testing.assert_allclose(C, recovered_C)

    # test consistency of outputs across backends with 'square_loss'
    # with various initialization of G0
    init_plan_list = [
        ("fluid", "fluid"),
        ("kmeans", "kmeans"),
        ("product", "product"),
        ("random", "random"),
    ]

    for init, init_b in init_plan_list:
        print(f"---- init : {init} / init_b : {init_b}")

        if (init == "fluid") and (not networkx_import):
            with pytest.raises(ValueError):
                X, C, log = ot.gromov.semirelaxed_fgw_barycenters(
                    n_samples,
                    [ys, yt],
                    [C1, C2],
                    [p1, p2],
                    [0.5, 0.5],
                    0.5,
                    fixed_structure=False,
                    fixed_features=False,
                    loss_fun="square_loss",
                    max_iter=10,
                    tol=1e-3,
                    stop_criterion="loss",
                    G0=init,
                    warmstartT=True,
                    random_state=12345,
                    log=True,
                    verbose=True,
                )
        elif (init == "kmeans") and (not sklearn_import):
            with pytest.raises(ValueError):
                X, C, log = ot.gromov.semirelaxed_fgw_barycenters(
                    n_samples,
                    [ys, yt],
                    [C1, C2],
                    [p1, p2],
                    [0.5, 0.5],
                    0.5,
                    fixed_structure=False,
                    fixed_features=False,
                    loss_fun="square_loss",
                    max_iter=10,
                    tol=1e-3,
                    stop_criterion="loss",
                    G0=init,
                    warmstartT=True,
                    random_state=12345,
                    log=True,
                    verbose=True,
                )
        else:
            X, C, log = ot.gromov.semirelaxed_fgw_barycenters(
                n_samples,
                [ys, yt],
                [C1, C2],
                [p1, p2],
                [0.5, 0.5],
                0.5,
                fixed_structure=False,
                fixed_features=False,
                loss_fun="square_loss",
                max_iter=10,
                tol=1e-3,
                stop_criterion="loss",
                G0=init,
                warmstartT=True,
                random_state=12345,
                log=True,
                verbose=True,
            )
            Xb, Cb, logb = ot.gromov.semirelaxed_fgw_barycenters(
                n_samples,
                [ysb, ytb],
                [C1b, C2b],
                [p1b, p2b],
                [0.5, 0.5],
                0.5,
                fixed_structure=False,
                fixed_features=False,
                loss_fun="square_loss",
                max_iter=10,
                tol=1e-3,
                stop_criterion="loss",
                G0=init_b,
                warmstartT=True,
                random_state=12345,
                log=True,
                verbose=True,
            )
            np.testing.assert_allclose(X, nx.to_numpy(Xb))
            np.testing.assert_allclose(C, nx.to_numpy(Cb))

    # test while providing advanced T inits and init_X != None, and init_C !=None
    Xb_, Cb_, logb_ = ot.gromov.semirelaxed_fgw_barycenters(
        n_samples,
        [ysb, ytb],
        [C1b, C2b],
        [p1b, p2b],
        [0.5, 0.5],
        0.5,
        fixed_structure=False,
        fixed_features=False,
        loss_fun="square_loss",
        max_iter=10,
        tol=1e-3,
        stop_criterion="loss",
        G0="random",
        warmstartT=True,
        random_state=12345,
        log=True,
        verbose=True,
        init_C=Cb,
        init_X=Xb,
    )
    np.testing.assert_allclose(Xb, Xb_)
    np.testing.assert_allclose(Cb, Cb_)

    # test consistency of backends while barycenter size not strictly inferior to sizes
    if sklearn_import:
        Xb_, Cb_, logb_ = ot.gromov.semirelaxed_fgw_barycenters(
            n_samples,
            [ysb, ytb],
            [C1b, C2b],
            [p1b, p2b],
            [0.5, 0.5],
            0.5,
            fixed_structure=False,
            fixed_features=False,
            loss_fun="square_loss",
            max_iter=10,
            tol=1e-3,
            stop_criterion="loss",
            G0="kmeans",
            warmstartT=True,
            random_state=12345,
            log=True,
            verbose=True,
            init_C=Cb,
            init_X=Xb,
        )

        X, C, log = ot.gromov.semirelaxed_fgw_barycenters(
            ns,
            [ys, yt],
            [C1, C2],
            [p1, p2],
            [0.5, 0.5],
            0.5,
            fixed_structure=False,
            fixed_features=False,
            loss_fun="square_loss",
            max_iter=10,
            tol=1e-3,
            stop_criterion="loss",
            G0="kmeans",
            warmstartT=True,
            random_state=12345,
            log=True,
            verbose=True,
        )
        Xb, Cb, logb = ot.gromov.semirelaxed_fgw_barycenters(
            ns,
            [ysb, ytb],
            [C1b, C2b],
            [p1b, p2b],
            [0.5, 0.5],
            0.5,
            fixed_structure=False,
            fixed_features=False,
            loss_fun="square_loss",
            max_iter=10,
            tol=1e-3,
            stop_criterion="loss",
            G0="kmeans",
            warmstartT=True,
            random_state=12345,
            log=True,
            verbose=True,
        )
        np.testing.assert_allclose(X, nx.to_numpy(Xb))
        np.testing.assert_allclose(C, nx.to_numpy(Cb))

    # test edge cases for semirelaxed fgw barycenters:
    # unique input structure
    X, C = ot.gromov.semirelaxed_fgw_barycenters(
        n_samples,
        [ys],
        [C1],
        [p1],
        None,
        0.5,
        fixed_structure=False,
        fixed_features=False,
        loss_fun="square_loss",
        max_iter=2,
        tol=1e-3,
        stop_criterion=stop_criterion,
        warmstartT=True,
        random_state=12345,
        log=False,
        verbose=False,
    )
    Xb, Cb = ot.gromov.semirelaxed_fgw_barycenters(
        n_samples,
        [ysb],
        [C1b],
        [p1b],
        [1.0],
        0.5,
        fixed_structure=False,
        fixed_features=False,
        loss_fun="square_loss",
        max_iter=2,
        tol=1e-3,
        stop_criterion=stop_criterion,
        warmstartT=True,
        random_state=12345,
        log=False,
        verbose=False,
    )

    np.testing.assert_allclose(C, Cb, atol=1e-06)
    np.testing.assert_allclose(X, Xb, atol=1e-06)
