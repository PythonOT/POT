"""Tests for main module ot.weak """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import ot
import numpy as np


def test_factored_ot():
    # test weak ot solver and identity stationary point
    n = 50
    rng = np.random.RandomState(0)

    xs = rng.randn(n, 2)
    xt = rng.randn(n, 2)
    u = ot.utils.unif(n)

    Ga, Gb, X, log = ot.factored_optimal_transport(xs, xt, u, u, r=10, log=True)

    # check constraints
    np.testing.assert_allclose(u, Ga.sum(1))
    np.testing.assert_allclose(u, Gb.sum(0))

    Ga, Gb, X, log = ot.factored_optimal_transport(xs, xt, u, u, reg=1, r=10, log=True)

    # check constraints
    np.testing.assert_allclose(u, Ga.sum(1))
    np.testing.assert_allclose(u, Gb.sum(0))


def test_factored_ot_backends(nx):
    # test weak ot solver for different backends
    n = 50
    rng = np.random.RandomState(0)

    xs = rng.randn(n, 2)
    xt = rng.randn(n, 2)
    u = ot.utils.unif(n)

    xs2 = nx.from_numpy(xs)
    xt2 = nx.from_numpy(xt)
    u2 = nx.from_numpy(u)

    Ga2, Gb2, X2 = ot.factored_optimal_transport(xs2, xt2, u2, u2, r=10)

    # check constraints
    np.testing.assert_allclose(u, nx.to_numpy(Ga2).sum(1))
    np.testing.assert_allclose(u, nx.to_numpy(Gb2).sum(0))

    Ga2, Gb2, X2 = ot.factored_optimal_transport(xs2, xt2, reg=1, r=10, X0=X2)

    # check constraints
    np.testing.assert_allclose(u, nx.to_numpy(Ga2).sum(1))
    np.testing.assert_allclose(u, nx.to_numpy(Gb2).sum(0))
