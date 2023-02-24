"""Tests for module COOT on OT """

# Author: Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

import numpy as np
import ot
from ot.coot import co_optimal_transport as coot
from ot.coot import co_optimal_transport2 as coot2


def test_coot(nx):
    n_samples = 60  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(
        n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()
    xs_nx = nx.from_numpy(xs)
    xt_nx = nx.from_numpy(xt)

    # test couplings
    pi_sample, pi_feature = coot(X=xs, Y=xt)
    pi_sample_nx, pi_feature_nx = coot(X=xs_nx, Y=xt_nx)

    id_sample = np.eye(n_samples, n_samples) / n_samples
    id_feature = np.eye(2, 2) / 2

    np.testing.assert_allclose(pi_sample, np.flipud(id_sample), atol=1e-04)
    np.testing.assert_allclose(pi_feature, id_feature, atol=1e-04)
    np.testing.assert_allclose(pi_sample_nx, nx.from_numpy(
        np.flipud(id_sample).copy()), atol=1e-04)
    np.testing.assert_allclose(
        pi_feature_nx, nx.from_numpy(id_feature.copy()), atol=1e-04)

    # test marginal distributions
    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    px_s_nx, px_f_nx = nx.from_numpy(px_s), nx.from_numpy(px_f)
    py_s_nx, py_f_nx = nx.from_numpy(py_s), nx.from_numpy(py_f)

    np.testing.assert_allclose(px_s_nx, nx.sum(pi_sample_nx, axis=0), atol=1e-04)
    np.testing.assert_allclose(py_s_nx, nx.sum(pi_sample_nx, axis=1), atol=1e-04)
    np.testing.assert_allclose(px_f_nx, nx.sum(pi_feature_nx, axis=0), atol=1e-04)
    np.testing.assert_allclose(py_f_nx, nx.sum(pi_feature_nx, axis=1), atol=1e-04)

    np.testing.assert_allclose(px_s, nx.sum(pi_sample, axis=0), atol=1e-04)
    np.testing.assert_allclose(py_s, nx.sum(pi_sample, axis=1), atol=1e-04)
    np.testing.assert_allclose(px_f, nx.sum(pi_feature, axis=0), atol=1e-04)
    np.testing.assert_allclose(py_f, nx.sum(pi_feature, axis=1), atol=1e-04)

    # test COOT distance

    coot_np = coot2(X=xs, Y=xt)
    coot_nx = coot2(X=xs_nx, Y=xt_nx).item()
    np.testing.assert_allclose(coot_np, 0, atol=1e-08)
    np.testing.assert_allclose(coot_nx, 0, atol=1e-08)


def test_entropic_coot(nx):
    n_samples = 60  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(
        n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()
    xs_nx = nx.from_numpy(xs)
    xt_nx = nx.from_numpy(xt)

    eps = (1, 1e-1)
    nits_ot = 2000

    # test couplings
    pi_sample, pi_feature = coot(X=xs, Y=xt, eps=eps, nits_ot=nits_ot)
    pi_sample_nx, pi_feature_nx = coot(
        X=xs_nx, Y=xt_nx, eps=eps, nits_ot=nits_ot)

    np.testing.assert_allclose(
        pi_sample_nx, nx.from_numpy(pi_sample.copy()), atol=1e-04)
    np.testing.assert_allclose(
        pi_feature_nx, nx.from_numpy(pi_feature.copy()), atol=1e-04)

    # test marginal distributions
    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    px_s_nx, px_f_nx = nx.from_numpy(px_s), nx.from_numpy(px_f)
    py_s_nx, py_f_nx = nx.from_numpy(py_s), nx.from_numpy(py_f)

    np.testing.assert_allclose(px_s_nx, nx.sum(pi_sample_nx, axis=0), atol=1e-04)
    np.testing.assert_allclose(py_s_nx, nx.sum(pi_sample_nx, axis=1), atol=1e-04)
    np.testing.assert_allclose(px_f_nx, nx.sum(pi_feature_nx, axis=0), atol=1e-04)
    np.testing.assert_allclose(py_f_nx, nx.sum(pi_feature_nx, axis=1), atol=1e-04)

    np.testing.assert_allclose(px_s, nx.sum(pi_sample, axis=0), atol=1e-04)
    np.testing.assert_allclose(py_s, nx.sum(pi_sample, axis=1), atol=1e-04)
    np.testing.assert_allclose(px_f, nx.sum(pi_feature, axis=0), atol=1e-04)
    np.testing.assert_allclose(py_f, nx.sum(pi_feature, axis=1), atol=1e-04)

    # test entropic COOT distance

    coot_np = coot2(X=xs, Y=xt, eps=eps, nits_ot=nits_ot)
    coot_nx = coot2(X=xs_nx, Y=xt_nx, eps=eps, nits_ot=nits_ot).item()

    np.testing.assert_allclose(coot_np, coot_nx, atol=1e-08)


def test_coot_with_linear_terms(nx):
    n_samples = 60  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(
        n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()
    xs_nx = nx.from_numpy(xs)
    xt_nx = nx.from_numpy(xt)

    D_sample = np.ones((n_samples, n_samples))
    np.fill_diagonal(np.fliplr(D_sample), 0)
    D_feature = np.ones((2, 2))
    np.fill_diagonal(D_feature, 0)
    D = (D_sample, D_feature)
    D_nx = (nx.from_numpy(D_sample), nx.from_numpy(D_feature))

    alpha = (1, 2)

    # test couplings
    id_sample = np.flipud(np.eye(n_samples, n_samples)) / n_samples
    id_feature = np.eye(2, 2) / 2

    pi_sample, pi_feature = coot(X=xs, Y=xt, alpha=alpha, D=D)
    pi_sample_nx, pi_feature_nx = coot(
        X=xs_nx, Y=xt_nx, alpha=alpha, D=D_nx)

    np.testing.assert_allclose(pi_sample, id_sample, atol=1e-04)
    np.testing.assert_allclose(pi_feature, id_feature, atol=1e-04)
    np.testing.assert_allclose(
        pi_sample_nx, nx.from_numpy(id_sample), atol=1e-04)
    np.testing.assert_allclose(
        pi_feature_nx, nx.from_numpy(id_feature), atol=1e-04)

    # test marginal distributions
    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    px_s_nx, px_f_nx = nx.from_numpy(px_s), nx.from_numpy(px_f)
    py_s_nx, py_f_nx = nx.from_numpy(py_s), nx.from_numpy(py_f)

    np.testing.assert_allclose(px_s_nx, nx.sum(pi_sample_nx, axis=0), atol=1e-04)
    np.testing.assert_allclose(py_s_nx, nx.sum(pi_sample_nx, axis=1), atol=1e-04)
    np.testing.assert_allclose(px_f_nx, nx.sum(pi_feature_nx, axis=0), atol=1e-04)
    np.testing.assert_allclose(py_f_nx, nx.sum(pi_feature_nx, axis=1), atol=1e-04)

    np.testing.assert_allclose(px_s, nx.sum(pi_sample, axis=0), atol=1e-04)
    np.testing.assert_allclose(py_s, nx.sum(pi_sample, axis=1), atol=1e-04)
    np.testing.assert_allclose(px_f, nx.sum(pi_feature, axis=0), atol=1e-04)
    np.testing.assert_allclose(py_f, nx.sum(pi_feature, axis=1), atol=1e-04)

    # test COOT distance

    coot_np = coot2(X=xs, Y=xt, alpha=alpha, D=D)
    coot_nx = coot2(X=xs_nx, Y=xt_nx, alpha=alpha, D=D_nx).item()
    np.testing.assert_allclose(coot_np, 0, atol=1e-08)
    np.testing.assert_allclose(coot_nx, 0, atol=1e-08)
