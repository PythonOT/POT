"""Tests for module regularization path"""

# Author: Haoran Wu <haoran.wu@univ-ubs.fr>
#
# License: MIT License

import numpy as np
import ot


def test_fully_relaxed_path():

    n_source = 50   # nb source samples (gaussian)
    n_target = 40   # nb target samples (gaussian)

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 2]])

    np.random.seed(0)
    xs = ot.datasets.make_2D_samples_gauss(n_source, mu, cov)
    xt = ot.datasets.make_2D_samples_gauss(n_target, mu, cov)

    # source and target distributions
    a = ot.utils.unif(n_source)
    b = ot.utils.unif(n_target)

    # loss matrix
    M = ot.dist(xs, xt)
    M /= M.max()

    t, _, _ = ot.regpath.regularization_path(a, b, M, reg=1e-8,
                                             semi_relaxed=False)

    G = t.reshape((n_source, n_target))
    np.testing.assert_allclose(a, G.sum(1), atol=1e-05)
    np.testing.assert_allclose(b, G.sum(0), atol=1e-05)


def test_semi_relaxed_path():

    n_source = 50   # nb source samples (gaussian)
    n_target = 40   # nb target samples (gaussian)

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 2]])

    np.random.seed(0)
    xs = ot.datasets.make_2D_samples_gauss(n_source, mu, cov)
    xt = ot.datasets.make_2D_samples_gauss(n_target, mu, cov)

    # source and target distributions
    a = ot.utils.unif(n_source)
    b = ot.utils.unif(n_target)

    # loss matrix
    M = ot.dist(xs, xt)
    M /= M.max()

    t, _, _ = ot.regpath.regularization_path(a, b, M, reg=1e-8,
                                             semi_relaxed=True)

    G = t.reshape((n_source, n_target))
    np.testing.assert_allclose(a, G.sum(1), atol=1e-05)
    np.testing.assert_allclose(b, G.sum(0), atol=1e-10)
