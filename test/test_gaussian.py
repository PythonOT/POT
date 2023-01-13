"""Tests for module gaussian"""

# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: MIT License

import numpy as np

import pytest

import ot
from ot.datasets import make_data_classif


def test_bures_wasserstein_mapping(nx):
    ns = 50
    nt = 50

    Xs, ys = make_data_classif('3gauss', ns)
    Xt, yt = make_data_classif('3gauss2', nt)
    ms = np.mean(Xs, axis=0)[None, :]
    mt = np.mean(Xt, axis=0)[None, :]
    Cs = np.cov(Xs.T)
    Ct = np.cov(Xt.T)

    Xsb, msb, mtb, Csb, Ctb = nx.from_numpy(Xs, ms, mt, Cs, Ct)

    A, b, log = ot.gaussian.bures_wasserstein_mapping(msb, mtb, Csb, Ctb, log=True)

    Xst = nx.to_numpy(nx.dot(Xsb, A) + b)

    Cst = np.cov(Xst.T)

    np.testing.assert_allclose(Ct, Cst, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("bias", [True, False])
def test_empirical_bures_wasserstein_mapping(nx, bias):
    ns = 50
    nt = 50

    Xs, ys = make_data_classif('3gauss', ns)
    Xt, yt = make_data_classif('3gauss2', nt)

    if not bias:
        ms = np.mean(Xs, axis=0)[None, :]
        mt = np.mean(Xt, axis=0)[None, :]

        Xs = Xs - ms
        Xt = Xt - mt

    Xsb, Xtb = nx.from_numpy(Xs, Xt)

    A, b, log = ot.gaussian.empirical_bures_wasserstein_mapping(Xsb, Xtb, log=True, bias=bias)

    Xst = nx.to_numpy(nx.dot(Xsb, A) + b)

    Ct = np.cov(Xt.T)
    Cst = np.cov(Xst.T)

    np.testing.assert_allclose(Ct, Cst, rtol=1e-2, atol=1e-2)


def test_bures_wasserstein_distance(nx):
    ms, mt = np.array([0]), np.array([10])
    Cs, Ct = np.array([[1]]).astype(np.float32), np.array([[1]]).astype(np.float32)
    msb, mtb, Csb, Ctb = nx.from_numpy(ms, mt, Cs, Ct)
    Wb, log = ot.gaussian.bures_wasserstein_distance(msb, mtb, Csb, Ctb, log=True)

    np.testing.assert_allclose(10, nx.to_numpy(Wb), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("bias", [True, False])
def test_empirical_bures_wasserstein_distance(nx, bias):
    ns = 200
    nt = 200

    rng = np.random.RandomState(2)
    Xs = rng.normal(0, 1, ns)[:, np.newaxis]
    Xt = rng.normal(10 * bias, 1, nt)[:, np.newaxis]

    Xsb, Xtb = nx.from_numpy(Xs, Xt)
    Wb, log = ot.gaussian.empirical_bures_wasserstein_distance(Xsb, Xtb, log=True, bias=bias)

    np.testing.assert_allclose(10 * bias, nx.to_numpy(Wb), rtol=1e-2, atol=1e-2)
