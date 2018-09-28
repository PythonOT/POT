"""Tests for module gromov  """

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
import ot


def test_gromov():
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s)

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

    gw, log = ot.gromov.gromov_wasserstein2(C1, C2, p, q, 'kl_loss', log=True)

    G = log['T']

    np.testing.assert_allclose(gw, 0, atol=1e-1, rtol=1e-1)

    # check constratints
    np.testing.assert_allclose(
        p, G.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, G.sum(0), atol=1e-04)  # cf convergence gromov


def test_entropic_gromov():
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s)

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

    Xs, ys = ot.datasets.make_data_classif('3gauss', ns)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', nt)

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

    Xs, ys = ot.datasets.make_data_classif('3gauss', ns)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', nt)

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
