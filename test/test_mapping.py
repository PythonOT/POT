"""Tests for module mapping"""
# Author: Eloi Tanguy <eloi.tanguy@u-paris.fr>
#
# License: MIT License

import numpy as np
import pytest
import ot


def test_ssnb_qcqp_constants():
    c1, c2, c3 = ot.mapping.ssnb_qcqp_constants(.5, 1)
    np.testing.assert_almost_equal(c1, 1)
    np.testing.assert_almost_equal(c2, .5)
    np.testing.assert_almost_equal(c3, 1)


def test_nearest_brenier_potential_fit(nx):
    X = nx.ones((2, 2))
    phi, G, log = ot.nearest_brenier_potential_fit(X, X, its=3, log=True)
    np.testing.assert_almost_equal(G, X)  # image of source should be close to target
    # test without log but with X_classes and seed
    ot.nearest_brenier_potential_fit(X, X, X_classes=nx.ones(2), its=1, seed=0)
    # test with seed being a np.random.RandomState
    ot.nearest_brenier_potential_fit(X, X, its=1, seed=np.random.RandomState(seed=0))


def test_brenier_potential_predict_bounds(nx):
    X = nx.ones((2, 2))
    phi, G = ot.nearest_brenier_potential_fit(X, X, its=3)
    phi_lu, G_lu, log = ot.nearest_brenier_potential_predict_bounds(X, phi, G, X, log=True)
    np.testing.assert_almost_equal(G_lu[0], X)  # 'new' input isn't new, so should be equal to target
    np.testing.assert_almost_equal(G_lu[1], X)
    # test with no log but classes
    ot.nearest_brenier_potential_predict_bounds(X, phi, G, X, X_classes=nx.ones(2), Y_classes=nx.ones(2))
