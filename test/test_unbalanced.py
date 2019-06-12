"""Tests for module Unbalanced OT with entropy regularization"""

# Author: Hicham Janati <hicham.janati@inria.fr>
#
# License: MIT License

import numpy as np
import ot
import pytest


@pytest.mark.parametrize("method", ["sinkhorn"])
def test_unbalanced_convergence(method):
    # test generalized sinkhorn for unbalanced OT
    n = 100
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = ot.utils.unif(n) * 1.5

    M = ot.dist(x, x)
    epsilon = 1.
    alpha = 1.
    K = np.exp(- M / epsilon)

    G, log = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=epsilon, alpha=alpha,
                                               stopThr=1e-10, method=method,
                                               log=True)

    # check fixed point equations
    fi = alpha / (alpha + epsilon)
    v_final = (b / K.T.dot(log["u"])) ** fi
    u_final = (a / K.dot(log["v"])) ** fi

    np.testing.assert_allclose(
        u_final, log["u"], atol=1e-05)
    np.testing.assert_allclose(
        v_final, log["v"], atol=1e-05)


def test_unbalanced_barycenter():
    # test generalized sinkhorn for unbalanced OT barycenter
    n = 100
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    A = rng.rand(n, 2)

    # make dists unbalanced
    A = A * np.array([1, 2])[None, :]
    M = ot.dist(x, x)
    epsilon = 1.
    alpha = 1.
    K = np.exp(- M / epsilon)

    q, log = ot.unbalanced.barycenter_unbalanced(A, M, reg=epsilon, alpha=alpha,
                                                 stopThr=1e-10,
                                                 log=True)

    # check fixed point equations
    fi = alpha / (alpha + epsilon)
    v_final = (q[:, None] / K.T.dot(log["u"])) ** fi
    u_final = (A / K.dot(log["v"])) ** fi

    np.testing.assert_allclose(
        u_final, log["u"], atol=1e-05)
    np.testing.assert_allclose(
        v_final, log["v"], atol=1e-05)
