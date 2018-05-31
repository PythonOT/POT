"""Tests for ot.smooth model """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import warnings

import numpy as np

import ot
from ot.datasets import get_1D_gauss as gauss
import pytest


def test_smooth_ot_dual():
    # test sinkhorn
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = ot.smooth.smooth_ot_dual(u, u, M, 1, stopThr=1e-10)

    # check constratints
    np.testing.assert_allclose(
        u, G.sum(1), atol=1e-05)  # cf convergence sinkhorn
    np.testing.assert_allclose(
        u, G.sum(0), atol=1e-05)  # cf convergence sinkhorn    
    