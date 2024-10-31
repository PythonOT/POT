"""Tests for module COOT on OT"""

# Author: Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

import numpy as np
import ot
from ot.coot import co_optimal_transport as coot
from ot.coot import co_optimal_transport2 as coot2
import pytest


@pytest.mark.parametrize("verbose", [False, True, 1, 0])
def test_coot(nx, verbose):
    n_samples = 60  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()
    xs_nx = nx.from_numpy(xs)
    xt_nx = nx.from_numpy(xt)

    # test couplings
    pi_sample, pi_feature = coot(X=xs, Y=xt, verbose=verbose)
    pi_sample_nx, pi_feature_nx = coot(X=xs_nx, Y=xt_nx, verbose=verbose)
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    anti_id_sample = np.flipud(np.eye(n_samples, n_samples)) / n_samples
    id_feature = np.eye(2, 2) / 2

    np.testing.assert_allclose(pi_sample, anti_id_sample, atol=1e-04)
    np.testing.assert_allclose(pi_sample_nx, anti_id_sample, atol=1e-04)
    np.testing.assert_allclose(pi_feature, id_feature, atol=1e-04)
    np.testing.assert_allclose(pi_feature_nx, id_feature, atol=1e-04)

    # test marginal distributions
    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    np.testing.assert_allclose(px_s, pi_sample_nx.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_s, pi_sample_nx.sum(1), atol=1e-04)
    np.testing.assert_allclose(px_f, pi_feature_nx.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_f, pi_feature_nx.sum(1), atol=1e-04)

    np.testing.assert_allclose(px_s, pi_sample.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_s, pi_sample.sum(1), atol=1e-04)
    np.testing.assert_allclose(px_f, pi_feature.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_f, pi_feature.sum(1), atol=1e-04)

    # test COOT distance

    coot_np = coot2(X=xs, Y=xt, verbose=verbose)
    coot_nx = nx.to_numpy(coot2(X=xs_nx, Y=xt_nx, verbose=verbose))
    np.testing.assert_allclose(coot_np, 0, atol=1e-08)
    np.testing.assert_allclose(coot_nx, 0, atol=1e-08)


def test_entropic_coot(nx):
    n_samples = 60  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()
    xs_nx = nx.from_numpy(xs)
    xt_nx = nx.from_numpy(xt)

    epsilon = (1, 1e-1)
    nits_ot = 2000

    # test couplings
    pi_sample, pi_feature = coot(X=xs, Y=xt, epsilon=epsilon, nits_ot=nits_ot)
    pi_sample_nx, pi_feature_nx = coot(
        X=xs_nx, Y=xt_nx, epsilon=epsilon, nits_ot=nits_ot
    )
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    np.testing.assert_allclose(pi_sample, pi_sample_nx, atol=1e-04)
    np.testing.assert_allclose(pi_feature, pi_feature_nx, atol=1e-04)

    # test marginal distributions
    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    np.testing.assert_allclose(px_s, pi_sample_nx.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_s, pi_sample_nx.sum(1), atol=1e-04)
    np.testing.assert_allclose(px_f, pi_feature_nx.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_f, pi_feature_nx.sum(1), atol=1e-04)

    np.testing.assert_allclose(px_s, pi_sample.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_s, pi_sample.sum(1), atol=1e-04)
    np.testing.assert_allclose(px_f, pi_feature.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_f, pi_feature.sum(1), atol=1e-04)

    # test entropic COOT distance

    coot_np = coot2(X=xs, Y=xt, epsilon=epsilon, nits_ot=nits_ot)
    coot_nx = nx.to_numpy(coot2(X=xs_nx, Y=xt_nx, epsilon=epsilon, nits_ot=nits_ot))

    np.testing.assert_allclose(coot_np, coot_nx, atol=1e-08)


def test_coot_with_linear_terms(nx):
    n_samples = 60  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=4)
    xt = xs[::-1].copy()
    xs_nx = nx.from_numpy(xs)
    xt_nx = nx.from_numpy(xt)

    M_samp = np.ones((n_samples, n_samples))
    np.fill_diagonal(np.fliplr(M_samp), 0)
    M_feat = np.ones((2, 2))
    np.fill_diagonal(M_feat, 0)
    M_samp_nx, M_feat_nx = nx.from_numpy(M_samp), nx.from_numpy(M_feat)

    alpha = (1, 2)

    # test couplings
    anti_id_sample = np.flipud(np.eye(n_samples, n_samples)) / n_samples
    id_feature = np.eye(2, 2) / 2

    pi_sample, pi_feature = coot(X=xs, Y=xt, alpha=alpha, M_samp=M_samp, M_feat=M_feat)
    pi_sample_nx, pi_feature_nx = coot(
        X=xs_nx, Y=xt_nx, alpha=alpha, M_samp=M_samp_nx, M_feat=M_feat_nx
    )
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    np.testing.assert_allclose(pi_sample, anti_id_sample, atol=1e-04)
    np.testing.assert_allclose(pi_sample_nx, anti_id_sample, atol=1e-04)
    np.testing.assert_allclose(pi_feature, id_feature, atol=1e-04)
    np.testing.assert_allclose(pi_feature_nx, id_feature, atol=1e-04)

    # test marginal distributions
    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    np.testing.assert_allclose(px_s, pi_sample_nx.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_s, pi_sample_nx.sum(1), atol=1e-04)
    np.testing.assert_allclose(px_f, pi_feature_nx.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_f, pi_feature_nx.sum(1), atol=1e-04)

    np.testing.assert_allclose(px_s, pi_sample.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_s, pi_sample.sum(1), atol=1e-04)
    np.testing.assert_allclose(px_f, pi_feature.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_f, pi_feature.sum(1), atol=1e-04)

    # test COOT distance

    coot_np = coot2(X=xs, Y=xt, alpha=alpha, M_samp=M_samp, M_feat=M_feat)
    coot_nx = nx.to_numpy(
        coot2(X=xs_nx, Y=xt_nx, alpha=alpha, M_samp=M_samp_nx, M_feat=M_feat_nx)
    )
    np.testing.assert_allclose(coot_np, 0, atol=1e-08)
    np.testing.assert_allclose(coot_nx, 0, atol=1e-08)


def test_coot_raise_value_error(nx):
    n_samples = 80  # nb samples

    mu_s = np.array([2, 4])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=43)
    xt = xs[::-1].copy()
    xs_nx = nx.from_numpy(xs)
    xt_nx = nx.from_numpy(xt)

    # raise value error of method sinkhorn
    def coot_sh(method_sinkhorn):
        return coot(X=xs, Y=xt, method_sinkhorn=method_sinkhorn)

    def coot_sh_nx(method_sinkhorn):
        return coot(X=xs_nx, Y=xt_nx, method_sinkhorn=method_sinkhorn)

    np.testing.assert_raises(ValueError, coot_sh, "not_sinkhorn")
    np.testing.assert_raises(ValueError, coot_sh_nx, "not_sinkhorn")

    # raise value error for epsilon
    def coot_eps(epsilon):
        return coot(X=xs, Y=xt, epsilon=epsilon)

    def coot_eps_nx(epsilon):
        return coot(X=xs_nx, Y=xt_nx, epsilon=epsilon)

    np.testing.assert_raises(ValueError, coot_eps, (1, 2, 3))
    np.testing.assert_raises(ValueError, coot_eps_nx, [1, 2, 3, 4])

    # raise value error for alpha
    def coot_alpha(alpha):
        return coot(X=xs, Y=xt, alpha=alpha)

    def coot_alpha_nx(alpha):
        return coot(X=xs_nx, Y=xt_nx, alpha=alpha)

    np.testing.assert_raises(ValueError, coot_alpha, [1])
    np.testing.assert_raises(ValueError, coot_alpha_nx, np.arange(4))


def test_coot_warmstart(nx):
    n_samples = 80  # nb samples

    mu_s = np.array([2, 3])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=125)
    xt = xs[::-1].copy()
    xs_nx = nx.from_numpy(xs)
    xt_nx = nx.from_numpy(xt)

    # initialize warmstart
    rng = np.random.RandomState(42)
    init_pi_sample = rng.rand(n_samples, n_samples)
    init_pi_sample = init_pi_sample / np.sum(init_pi_sample)
    init_pi_sample_nx = nx.from_numpy(init_pi_sample)

    init_pi_feature = rng.rand(2, 2)
    init_pi_feature /= init_pi_feature / np.sum(init_pi_feature)
    init_pi_feature_nx = nx.from_numpy(init_pi_feature)

    init_duals_sample = (rng.random(n_samples) * 2 - 1, rng.random(n_samples) * 2 - 1)
    init_duals_sample_nx = (
        nx.from_numpy(init_duals_sample[0]),
        nx.from_numpy(init_duals_sample[1]),
    )

    init_duals_feature = (rng.random(2) * 2 - 1, rng.random(2) * 2 - 1)
    init_duals_feature_nx = (
        nx.from_numpy(init_duals_feature[0]),
        nx.from_numpy(init_duals_feature[1]),
    )

    warmstart = {
        "pi_sample": init_pi_sample,
        "pi_feature": init_pi_feature,
        "duals_sample": init_duals_sample,
        "duals_feature": init_duals_feature,
    }

    warmstart_nx = {
        "pi_sample": init_pi_sample_nx,
        "pi_feature": init_pi_feature_nx,
        "duals_sample": init_duals_sample_nx,
        "duals_feature": init_duals_feature_nx,
    }

    # test couplings
    pi_sample, pi_feature = coot(X=xs, Y=xt, warmstart=warmstart)
    pi_sample_nx, pi_feature_nx = coot(X=xs_nx, Y=xt_nx, warmstart=warmstart_nx)
    pi_sample_nx = nx.to_numpy(pi_sample_nx)
    pi_feature_nx = nx.to_numpy(pi_feature_nx)

    anti_id_sample = np.flipud(np.eye(n_samples, n_samples)) / n_samples
    id_feature = np.eye(2, 2) / 2

    np.testing.assert_allclose(pi_sample, anti_id_sample, atol=1e-04)
    np.testing.assert_allclose(pi_sample_nx, anti_id_sample, atol=1e-04)
    np.testing.assert_allclose(pi_feature, id_feature, atol=1e-04)
    np.testing.assert_allclose(pi_feature_nx, id_feature, atol=1e-04)

    # test marginal distributions
    px_s, px_f = ot.unif(n_samples), ot.unif(2)
    py_s, py_f = ot.unif(n_samples), ot.unif(2)

    np.testing.assert_allclose(px_s, pi_sample_nx.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_s, pi_sample_nx.sum(1), atol=1e-04)
    np.testing.assert_allclose(px_f, pi_feature_nx.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_f, pi_feature_nx.sum(1), atol=1e-04)

    np.testing.assert_allclose(px_s, pi_sample.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_s, pi_sample.sum(1), atol=1e-04)
    np.testing.assert_allclose(px_f, pi_feature.sum(0), atol=1e-04)
    np.testing.assert_allclose(py_f, pi_feature.sum(1), atol=1e-04)

    # test COOT distance
    coot_np = coot2(X=xs, Y=xt, warmstart=warmstart)
    coot_nx = nx.to_numpy(coot2(X=xs_nx, Y=xt_nx, warmstart=warmstart_nx))
    np.testing.assert_allclose(coot_np, 0, atol=1e-08)
    np.testing.assert_allclose(coot_nx, 0, atol=1e-08)


def test_coot_log(nx):
    n_samples = 90  # nb samples

    mu_s = np.array([-2, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=43)
    xt = xs[::-1].copy()
    xs_nx = nx.from_numpy(xs)
    xt_nx = nx.from_numpy(xt)

    pi_sample, pi_feature, log = coot(X=xs, Y=xt, log=True)
    pi_sample_nx, pi_feature_nx, log_nx = coot(X=xs_nx, Y=xt_nx, log=True)

    duals_sample, duals_feature = log["duals_sample"], log["duals_feature"]
    assert len(duals_sample) == 2
    assert len(duals_feature) == 2
    assert len(duals_sample[0]) == n_samples
    assert len(duals_sample[1]) == n_samples
    assert len(duals_feature[0]) == 2
    assert len(duals_feature[1]) == 2

    duals_sample_nx = log_nx["duals_sample"]
    assert len(duals_sample_nx) == 2
    assert len(duals_sample_nx[0]) == n_samples
    assert len(duals_sample_nx[1]) == n_samples

    duals_feature_nx = log_nx["duals_feature"]
    assert len(duals_feature_nx) == 2
    assert len(duals_feature_nx[0]) == 2
    assert len(duals_feature_nx[1]) == 2

    list_coot = log["distances"]
    assert len(list_coot) >= 1

    list_coot_nx = log_nx["distances"]
    assert len(list_coot_nx) >= 1

    # test with coot distance
    coot_np, log = coot2(X=xs, Y=xt, log=True)
    coot_nx, log_nx = coot2(X=xs_nx, Y=xt_nx, log=True)

    duals_sample, duals_feature = log["duals_sample"], log["duals_feature"]
    assert len(duals_sample) == 2
    assert len(duals_feature) == 2
    assert len(duals_sample[0]) == n_samples
    assert len(duals_sample[1]) == n_samples
    assert len(duals_feature[0]) == 2
    assert len(duals_feature[1]) == 2

    duals_sample_nx = log_nx["duals_sample"]
    assert len(duals_sample_nx) == 2
    assert len(duals_sample_nx[0]) == n_samples
    assert len(duals_sample_nx[1]) == n_samples

    duals_feature_nx = log_nx["duals_feature"]
    assert len(duals_feature_nx) == 2
    assert len(duals_feature_nx[0]) == 2
    assert len(duals_feature_nx[1]) == 2

    list_coot = log["distances"]
    assert len(list_coot) >= 1

    list_coot_nx = log_nx["distances"]
    assert len(list_coot_nx) >= 1
