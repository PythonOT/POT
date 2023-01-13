"""Tests for module gaussian"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import pytest

import ot
from ot.datasets import make_data_classif


def test_linear_mapping(nx):
    ns = 50
    nt = 50

    Xs, ys = make_data_classif('3gauss', ns)
    Xt, yt = make_data_classif('3gauss2', nt)

    Xsb, Xtb = nx.from_numpy(Xs, Xt)

    A, b = ot.gaussian.OT_mapping_linear(Xsb, Xtb)

    Xst = nx.to_numpy(nx.dot(Xsb, A) + b)

    Ct = np.cov(Xt.T)
    Cst = np.cov(Xst.T)

    np.testing.assert_allclose(Ct, Cst, rtol=1e-2, atol=1e-2)


def test_bures_wasserstein_distance(nx):
    ms, mt, Cs, Ct = [0], [10], [[1]], [[1]]
    msb, mtb, Csb, Ctb = nx.from_numpy(ms, mt, Cs, Ct)
    Wb = ot.gaussian.bures_wasserstein_distance(msb, mtb, Csb, Ctb)

    np.testing.assert_allclose(10, nx.to_numpy(Wb), rtol=1e-2, atol=1e-2)


def test_empirical_bures_wasserstein_distance(nx):
    ns = 200
    nt = 200

    rng = np.random.RandomState(1)
    Xs = rng.normal(0, 1, ns)[:, np.newaxis]
    Xt = rng.normal(10, 1, nt)[:, np.newaxis]
    Xsb, Xtb = nx.from_numpy(Xs, Xt)
    Wb = ot.gaussian.empirical_bures_wasserstein_distance(Xsb, Xtb, bias=True)

    np.testing.assert_allclose(10, nx.to_numpy(Wb), rtol=1e-2, atol=1e-2)
