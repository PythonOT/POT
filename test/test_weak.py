"""Tests for main module ot.weak """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import ot
import numpy as np


def test_weak_ot():
    # test weak ot solver and identity stationary point
    n = 50
    rng = np.random.RandomState(0)

    xs = rng.randn(n, 2)
    xt = rng.randn(n, 2)
    u = ot.utils.unif(n)

    G, log = ot.weak_optimal_transport(xs, xt, u, u, log=True)

    # check constraints
    np.testing.assert_allclose(u, G.sum(1))
    np.testing.assert_allclose(u, G.sum(0))

    # chaeck that identity is recovered
    G = ot.weak_optimal_transport(xs, xs, G0=np.eye(n) / n)

    # check G is identity
    np.testing.assert_allclose(G, np.eye(n) / n)

    # check constraints
    np.testing.assert_allclose(u, G.sum(1))
    np.testing.assert_allclose(u, G.sum(0))


def test_weak_ot_bakends(nx):
    # test weak ot solver for different backends
    n = 50
    rng = np.random.RandomState(0)

    xs = rng.randn(n, 2)
    xt = rng.randn(n, 2)
    u = ot.utils.unif(n)

    G = ot.weak_optimal_transport(xs, xt, u, u)

    xs2, xt2, u2 = nx.from_numpy(xs, xt, u)

    G2 = ot.weak_optimal_transport(xs2, xt2, u2, u2)

    np.testing.assert_allclose(nx.to_numpy(G2), G)
