"""Tests for module da on Domain Adaptation """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import pytest

import ot
from ot.datasets import make_data_classif, make_1D_samples_gauss


@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
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


@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
def test_bures_wasserstein_distance(nx):
    ns = 200
    nt = 200

    Xs = make_1D_samples_gauss(ns, 0, 1, random_state=42)
    Xt = make_1D_samples_gauss(nt, 10, 1, random_state=42)

    Xsb, Xtb = nx.from_numpy(Xs, Xt)
    Wb = ot.gaussian.bures_wasserstein_distance(Xsb, Xtb, bias=True)

    np.testing.assert_allclose(10, nx.to_numpy(Wb), rtol=1e-2, atol=1e-2)
