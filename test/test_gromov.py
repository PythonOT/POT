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

    xs = ot.datasets.get_2D_samples_gauss(n_samples, mu_s, cov_s)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    G = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss')

    # check constratints
    np.testing.assert_allclose(
        p, G.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, G.sum(0), atol=1e-04)  # cf convergence gromov


def test_entropic_gromov():
    n_samples = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.get_2D_samples_gauss(n_samples, mu_s, cov_s)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    G = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, p, q, 'square_loss', epsilon=5e-4)

    # check constratints
    np.testing.assert_allclose(
        p, G.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, G.sum(0), atol=1e-04)  # cf convergence gromov
