"""Tests for module gromov  """

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

import numpy as np
import pytest
import warnings

import ot
from ot.backend import NumpyBackend
from ot.backend import torch, tf


def test_pointwise_gromov(nx):
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

    C1b, C2b, pb, qb = nx.from_numpy(C1, C2, p, q)

    def loss(x, y):
        return np.abs(x - y)

    def lossb(x, y):
        return nx.abs(x - y)

    G, log = ot.gromov.pointwise_gromov_wasserstein(
        C1, C2, p, q, loss, max_iter=100, log=True, verbose=True, random_state=42)
    G = NumpyBackend().todense(G)
    Gb, logb = ot.gromov.pointwise_gromov_wasserstein(
        C1b, C2b, pb, qb, lossb, max_iter=100, log=True, verbose=True, random_state=42)
    Gb = nx.to_numpy(nx.todense(Gb))

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov

    np.testing.assert_allclose(float(logb['gw_dist_estimated']), 0.0, atol=1e-08)
    np.testing.assert_allclose(float(logb['gw_dist_std']), 0.0, atol=1e-08)

    G, log = ot.gromov.pointwise_gromov_wasserstein(
        C1, C2, p, q, loss, max_iter=100, alpha=0.1, log=True, verbose=True, random_state=42)
    G = NumpyBackend().todense(G)
    Gb, logb = ot.gromov.pointwise_gromov_wasserstein(
        C1b, C2b, pb, qb, lossb, max_iter=100, alpha=0.1, log=True, verbose=True, random_state=42)
    Gb = nx.to_numpy(nx.todense(Gb))

    np.testing.assert_allclose(G, Gb, atol=1e-06)


@pytest.skip_backend("tf", reason="test very slow with tf backend")
@pytest.skip_backend("jax", reason="test very slow with jax backend")
def test_sampled_gromov(nx):
    n_samples = 5  # nb samples

    mu_s = np.array([0, 0], dtype=np.float64)
    cov_s = np.array([[1, 0], [0, 1]], dtype=np.float64)

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb = nx.from_numpy(C1, C2, p, q)

    def loss(x, y):
        return np.abs(x - y)

    def lossb(x, y):
        return nx.abs(x - y)

    G, log = ot.gromov.sampled_gromov_wasserstein(
        C1, C2, p, q, loss, max_iter=20, nb_samples_grad=2, epsilon=1, log=True, verbose=True, random_state=42)
    Gb, logb = ot.gromov.sampled_gromov_wasserstein(
        C1b, C2b, pb, qb, lossb, max_iter=20, nb_samples_grad=2, epsilon=1, log=True, verbose=True, random_state=42)
    Gb = nx.to_numpy(Gb)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(
        p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, Gb.sum(0), atol=1e-04)  # cf convergence gromov
