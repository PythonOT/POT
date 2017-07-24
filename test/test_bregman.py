

import ot
import numpy as np

# import pytest


def test_sinkhorn():
    # test sinkhorn
    n = 100
    np.random.seed(0)

    x = np.random.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = ot.sinkhorn(u, u, M, 1, stopThr=1e-10)

    # check constratints
    assert np.allclose(u, G.sum(1), atol=1e-05)  # cf convergence sinkhorn
    assert np.allclose(u, G.sum(0), atol=1e-05)  # cf convergence sinkhorn


def test_sinkhorn_empty():
    # test sinkhorn
    n = 100
    np.random.seed(0)

    x = np.random.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = ot.sinkhorn([], [], M, 1, stopThr=1e-10)
    # check constratints
    assert np.allclose(u, G.sum(1), atol=1e-05)  # cf convergence sinkhorn
    assert np.allclose(u, G.sum(0), atol=1e-05)  # cf convergence sinkhorn

    G = ot.sinkhorn([], [], M, 1, stopThr=1e-10, method='sinkhorn_stabilized')
    # check constratints
    assert np.allclose(u, G.sum(1), atol=1e-05)  # cf convergence sinkhorn
    assert np.allclose(u, G.sum(0), atol=1e-05)  # cf convergence sinkhorn

    G = ot.sinkhorn(
        [], [], M, 1, stopThr=1e-10, method='sinkhorn_epsilon_scaling')
    # check constratints
    assert np.allclose(u, G.sum(1), atol=1e-05)  # cf convergence sinkhorn
    assert np.allclose(u, G.sum(0), atol=1e-05)  # cf convergence sinkhorn


def test_sinkhorn_variants():
    # test sinkhorn
    n = 100
    np.random.seed(0)

    x = np.random.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G0 = ot.sinkhorn(u, u, M, 1, method='sinkhorn', stopThr=1e-10)
    Gs = ot.sinkhorn(u, u, M, 1, method='sinkhorn_stabilized', stopThr=1e-10)
    Ges = ot.sinkhorn(
        u, u, M, 1, method='sinkhorn_epsilon_scaling', stopThr=1e-10)
    Gerr = ot.sinkhorn(u, u, M, 1, method='do_not_exists', stopThr=1e-10)

    # check values
    assert np.allclose(G0, Gs, atol=1e-05)
    assert np.allclose(G0, Ges, atol=1e-05)
    assert np.allclose(G0, Gerr)
