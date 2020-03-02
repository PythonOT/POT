"""Tests for module gromov  """

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#
# License: MIT License

import numpy as np
import ot


def test_gromov():
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    G = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', verbose=True)

    # check constratints
    np.testing.assert_allclose(
        p, G.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, G.sum(0), atol=1e-04)  # cf convergence gromov

    Id = (1 / (1.0 * n_samples)) * np.eye(n_samples, n_samples)

    np.testing.assert_allclose(
        G, np.flipud(Id), atol=1e-04)

    gw, log = ot.gromov.gromov_wasserstein2(C1, C2, p, q, 'kl_loss', log=True)

    gw_val = ot.gromov.gromov_wasserstein2(C1, C2, p, q, 'kl_loss', log=False)

    G = log['T']

    np.testing.assert_allclose(gw, 0, atol=1e-1, rtol=1e-1)

    np.testing.assert_allclose(gw, gw_val, atol=1e-1, rtol=1e-1)  # cf log=False

    # check constratints
    np.testing.assert_allclose(
        p, G.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, G.sum(0), atol=1e-04)  # cf convergence gromov


def test_entropic_gromov():
    n_samples = 50  # nb samples

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

    G = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, p, q, 'square_loss', epsilon=5e-4, verbose=True)

    # check constratints
    np.testing.assert_allclose(
        p, G.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, G.sum(0), atol=1e-04)  # cf convergence gromov

    gw, log = ot.gromov.entropic_gromov_wasserstein2(
        C1, C2, p, q, 'kl_loss', epsilon=1e-2, log=True)

    G = log['T']

    np.testing.assert_allclose(gw, 0, atol=1e-1, rtol=1e-1)

    # check constratints
    np.testing.assert_allclose(
        p, G.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, G.sum(0), atol=1e-04)  # cf convergence gromov


def test_gromov_barycenter():
    ns = 50
    nt = 60

    Xs, ys = ot.datasets.make_data_classif('3gauss', ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', nt, random_state=42)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)

    n_samples = 3
    Cb = ot.gromov.gromov_barycenters(n_samples, [C1, C2],
                                      [ot.unif(ns), ot.unif(nt)
                                       ], ot.unif(n_samples), [.5, .5],
                                      'square_loss',  # 5e-4,
                                      max_iter=100, tol=1e-3,
                                      verbose=True)
    np.testing.assert_allclose(Cb.shape, (n_samples, n_samples))

    Cb2 = ot.gromov.gromov_barycenters(n_samples, [C1, C2],
                                       [ot.unif(ns), ot.unif(nt)
                                        ], ot.unif(n_samples), [.5, .5],
                                       'kl_loss',  # 5e-4,
                                       max_iter=100, tol=1e-3)
    np.testing.assert_allclose(Cb2.shape, (n_samples, n_samples))


def test_gromov_entropic_barycenter():
    ns = 50
    nt = 60

    Xs, ys = ot.datasets.make_data_classif('3gauss', ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', nt, random_state=42)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)

    n_samples = 3
    Cb = ot.gromov.entropic_gromov_barycenters(n_samples, [C1, C2],
                                               [ot.unif(ns), ot.unif(nt)
                                                ], ot.unif(n_samples), [.5, .5],
                                               'square_loss', 2e-3,
                                               max_iter=100, tol=1e-3,
                                               verbose=True)
    np.testing.assert_allclose(Cb.shape, (n_samples, n_samples))

    Cb2 = ot.gromov.entropic_gromov_barycenters(n_samples, [C1, C2],
                                                [ot.unif(ns), ot.unif(nt)
                                                 ], ot.unif(n_samples), [.5, .5],
                                                'kl_loss', 2e-3,
                                                max_iter=100, tol=1e-3)
    np.testing.assert_allclose(Cb2.shape, (n_samples, n_samples))


def test_fgw():

    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    ys = np.random.randn(xs.shape[0], 2)
    yt = ys[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    M = ot.dist(ys, yt)
    M /= M.max()

    G = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, 'square_loss', alpha=0.5)

    # check constratints
    np.testing.assert_allclose(
        p, G.sum(1), atol=1e-04)  # cf convergence fgw
    np.testing.assert_allclose(
        q, G.sum(0), atol=1e-04)  # cf convergence fgw

    Id = (1 / (1.0 * n_samples)) * np.eye(n_samples, n_samples)

    np.testing.assert_allclose(
        G, np.flipud(Id), atol=1e-04)  # cf convergence gromov

    fgw, log = ot.gromov.fused_gromov_wasserstein2(M, C1, C2, p, q, 'square_loss', alpha=0.5, log=True)

    G = log['T']

    np.testing.assert_allclose(fgw, 0, atol=1e-1, rtol=1e-1)

    # check constratints
    np.testing.assert_allclose(
        p, G.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, G.sum(0), atol=1e-04)  # cf convergence gromov


def test_fgw_barycenter():
    np.random.seed(42)

    ns = 50
    nt = 60

    Xs, ys = ot.datasets.make_data_classif('3gauss', ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', nt, random_state=42)

    ys = np.random.randn(Xs.shape[0], 2)
    yt = np.random.randn(Xt.shape[0], 2)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)

    n_samples = 3
    X, C = ot.gromov.fgw_barycenters(n_samples, [ys, yt], [C1, C2], [ot.unif(ns), ot.unif(nt)], [.5, .5], 0.5,
                                     fixed_structure=False, fixed_features=False,
                                     p=ot.unif(n_samples), loss_fun='square_loss',
                                     max_iter=100, tol=1e-3)
    np.testing.assert_allclose(C.shape, (n_samples, n_samples))
    np.testing.assert_allclose(X.shape, (n_samples, ys.shape[1]))

    xalea = np.random.randn(n_samples, 2)
    init_C = ot.dist(xalea, xalea)

    X, C = ot.gromov.fgw_barycenters(n_samples, [ys, yt], [C1, C2], ps=[ot.unif(ns), ot.unif(nt)], lambdas=[.5, .5], alpha=0.5,
                                     fixed_structure=True, init_C=init_C, fixed_features=False,
                                     p=ot.unif(n_samples), loss_fun='square_loss',
                                     max_iter=100, tol=1e-3)
    np.testing.assert_allclose(C.shape, (n_samples, n_samples))
    np.testing.assert_allclose(X.shape, (n_samples, ys.shape[1]))

    init_X = np.random.randn(n_samples, ys.shape[1])

    X, C = ot.gromov.fgw_barycenters(n_samples, [ys, yt], [C1, C2], [ot.unif(ns), ot.unif(nt)], [.5, .5], 0.5,
                                     fixed_structure=False, fixed_features=True, init_X=init_X,
                                     p=ot.unif(n_samples), loss_fun='square_loss',
                                     max_iter=100, tol=1e-3)
    np.testing.assert_allclose(C.shape, (n_samples, n_samples))
    np.testing.assert_allclose(X.shape, (n_samples, ys.shape[1]))


def test_gromov_1d():
    np.random.seed(42)
    # Test cost for diag
    u = np.array([1, 0, 4])
    v = np.array([1, 4, 0])
    cost_gw1D = ot.gromov.gromov_1d2(u, v)
    T = ot.gromov.gromov_1d(u, v, dense=False)

    assert cost_gw1D == 0
    assert ot.gromov.gromov_loss_sorted_1d(np.dot(u, 3 * T), v) == 0

    # Test for anti diag
    u = np.array([1, 0, 4])
    v = np.array([-1, 2, 3])
    cost_gw1D = ot.gromov.gromov_1d2(u, v)
    T = ot.gromov.gromov_1d(u, v, dense=False)

    assert cost_gw1D == 0
    assert ot.gromov.gromov_loss_sorted_1d(np.dot(u, 3 * T), v) == 0

    # Test GW 1d better than GW POT
    all_good = []
    its_all_good_man = False
    for n in range(3, 100):
        ns = n
        nt = n
        xs_alea = np.random.randn(ns, 1)
        xt_alea = np.random.randn(nt, 1)
        T_1d, log_1d = ot.gromov.gromov_1d(xs_alea.ravel(), xt_alea.ravel(), log=True, dense=False)

        C1 = ot.dist(xs_alea, metric='sqeuclidean')
        C2 = ot.dist(xt_alea, metric='sqeuclidean')
        p = np.ones(C1.shape[0]) / C1.shape[0]
        q = np.ones(C2.shape[0]) / C2.shape[0]
        T_GW, log_GW = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', log=True)

        all_good.append(log_1d['gw_dist'] - log_GW['gw_dist'])

    all_good = np.array(all_good)

    if len(all_good) == 0:
        its_all_good_man = True
    elif np.max(all_good[all_good >= 0]) <= 1e-14:
        its_all_good_man = True

    assert its_all_good_man

    all_good = []
    its_all_good_man = False
    for repeat in range(100):
        ns = 5
        nt = 5
        xs_alea = np.random.randn(ns, 1)
        xt_alea = np.random.randn(nt, 1)
        T_1d, log_1d = ot.gromov.gromov_1d(xs_alea.ravel(), xt_alea.ravel(), log=True, dense=False)

        C1 = ot.dist(xs_alea, metric='sqeuclidean')
        C2 = ot.dist(xt_alea, metric='sqeuclidean')
        p = np.ones(C1.shape[0]) / C1.shape[0]
        q = np.ones(C2.shape[0]) / C2.shape[0]

        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, 'square_loss')
        d_1D = ot.gromov.gwloss(constC, hC1, hC2, T_1d)
        d_true_1D = log_1d['gw_dist']

        all_good.append(np.abs(d_1D - d_true_1D))

    all_good = np.array(all_good)
    assert np.all(all_good <= 1e-13)
