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
    loss = ot.unbalanced.sinkhorn_unbalanced2(a, b, M, epsilon, alpha,
                                              method=method)
    # check fixed point equations
    fi = alpha / (alpha + epsilon)
    v_final = (b / K.T.dot(log["u"])) ** fi
    u_final = (a / K.dot(log["v"])) ** fi

    np.testing.assert_allclose(
        u_final, log["u"], atol=1e-05)
    np.testing.assert_allclose(
        v_final, log["v"], atol=1e-05)

    # check if sinkhorn_unbalanced2 returns the correct loss
    np.testing.assert_allclose((G * M).sum(), loss, atol=1e-5)


@pytest.mark.parametrize("method", ["sinkhorn"])
def test_unbalanced_multiple_inputs(method):
    # test generalized sinkhorn for unbalanced OT
    n = 100
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = rng.rand(n, 2)

    M = ot.dist(x, x)
    epsilon = 1.
    alpha = 1.
    K = np.exp(- M / epsilon)

    loss, log = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=epsilon,
                                                  alpha=alpha,
                                                  stopThr=1e-10, method=method,
                                                  log=True)
    # check fixed point equations
    fi = alpha / (alpha + epsilon)
    v_final = (b / K.T.dot(log["u"])) ** fi

    u_final = (a[:, None] / K.dot(log["v"])) ** fi

    np.testing.assert_allclose(
        u_final, log["u"], atol=1e-05)
    np.testing.assert_allclose(
        v_final, log["v"], atol=1e-05)

    assert len(loss) == b.shape[1]


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


def test_implemented_methods():
    IMPLEMENTED_METHODS = ['sinkhorn']
    TO_BE_IMPLEMENTED_METHODS = ['sinkhorn_stabilized',
                                 'sinkhorn_epsilon_scaling']
    NOT_VALID_TOKENS = ['foo']
    # test generalized sinkhorn for unbalanced OT barycenter
    n = 3
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = ot.utils.unif(n) * 1.5

    M = ot.dist(x, x)
    epsilon = 1.
    alpha = 1.
    for method in IMPLEMENTED_METHODS:
        ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, alpha,
                                          method=method)
        ot.unbalanced.sinkhorn_unbalanced2(a, b, M, epsilon, alpha,
                                           method=method)
    with pytest.warns(UserWarning, match='not implemented'):
        for method in set(TO_BE_IMPLEMENTED_METHODS):
            ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, alpha,
                                              method=method)
            ot.unbalanced.sinkhorn_unbalanced2(a, b, M, epsilon, alpha,
                                               method=method)
    with pytest.raises(ValueError):
        for method in set(NOT_VALID_TOKENS):
            ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, alpha,
                                              method=method)
            ot.unbalanced.sinkhorn_unbalanced2(a, b, M, epsilon, alpha,
                                               method=method)
