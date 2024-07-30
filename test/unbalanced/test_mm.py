"""Tests for module Unbalanced OT with entropy regularization"""

# Author: Hicham Janati <hicham.janati@inria.fr>
#         Laetitia Chapel <laetitia.chapel@univ-ubs.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License


import numpy as np
import ot
import pytest


@pytest.mark.parametrize("div", ["kl", "l2"])
def test_mm_convergence(nx, div):
    n = 100
    rng = np.random.RandomState(42)
    x = rng.randn(n, 2)
    rng = np.random.RandomState(75)
    y = rng.randn(n, 2)
    a_np = ot.utils.unif(n)
    b_np = ot.utils.unif(n)

    M = ot.dist(x, y)
    M = M / M.max()
    reg_m = 100
    a, b, M = nx.from_numpy(a_np, b_np, M)

    G, _ = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, div=div,
                                       verbose=False, log=True)
    _, log = ot.unbalanced.mm_unbalanced2(a, b, M, reg_m, div=div, verbose=True, log=True)
    linear_cost = nx.to_numpy(log["cost"])

    # check if the marginals come close to the true ones when large reg
    np.testing.assert_allclose(np.sum(nx.to_numpy(G), 1), a_np, atol=1e-03)
    np.testing.assert_allclose(np.sum(nx.to_numpy(G), 0), b_np, atol=1e-03)

    # check if mm_unbalanced2 returns the correct loss
    np.testing.assert_allclose(nx.to_numpy(nx.sum(G * M)), linear_cost, atol=1e-5)

    # check in case no histogram is provided
    a_np, b_np = np.array([]), np.array([])
    a, b = nx.from_numpy(a_np, b_np)

    G_null = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, div=div, verbose=False)
    np.testing.assert_allclose(nx.to_numpy(G_null), nx.to_numpy(G))

    # test when G0 is given
    G0 = ot.emd(a, b, M)
    G0_np = nx.to_numpy(G0)
    reg_m = 10000
    G = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, div=div, G0=G0, verbose=False)
    np.testing.assert_allclose(G0_np, nx.to_numpy(G), atol=1e-05)


@pytest.mark.parametrize("div", ["kl", "l2"])
def test_mm_relaxation_parameters(nx, div):
    n = 100
    rng = np.random.RandomState(42)
    x = rng.randn(n, 2)
    rng = np.random.RandomState(75)
    y = rng.randn(n, 2)
    a_np = ot.utils.unif(n)
    b_np = ot.utils.unif(n)

    M = ot.dist(x, y)
    M = M / M.max()
    a, b, M = nx.from_numpy(a_np, b_np, M)

    reg = 1e-2

    reg_m = 100
    full_list_reg_m = [reg_m, reg_m]
    full_tuple_reg_m = (reg_m, reg_m)
    tuple_reg_m, list_reg_m = (reg_m), [reg_m]
    nx1_reg_m = reg_m * nx.ones(1)
    nx2_reg_m = reg_m * nx.ones(2)

    list_options = [nx1_reg_m, nx2_reg_m, full_tuple_reg_m,
                    tuple_reg_m, full_list_reg_m, list_reg_m]

    G0, _ = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, reg=reg,
                                        div=div, verbose=False, log=True)
    loss_0 = nx.to_numpy(
        ot.unbalanced.mm_unbalanced2(a, b, M, reg_m=reg_m, reg=reg,
                                     div=div, verbose=True)
    )

    for opt in list_options:
        G1, _ = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=opt,
                                            reg=reg, div=div,
                                            verbose=False, log=True)
        loss_1 = nx.to_numpy(
            ot.unbalanced.mm_unbalanced2(a, b, M, reg_m=opt,
                                         reg=reg, div=div, verbose=True)
        )

        np.testing.assert_allclose(nx.to_numpy(G0), nx.to_numpy(G1), atol=1e-05)
        np.testing.assert_allclose(loss_0, loss_1, atol=1e-5)


@pytest.mark.parametrize("div", ["kl", "l2"])
def test_mm_reference_measure(nx, div):
    n = 100
    rng = np.random.RandomState(42)
    x = rng.randn(n, 2)
    rng = np.random.RandomState(75)
    y = rng.randn(n, 2)
    a_np = ot.utils.unif(n)
    b_np = ot.utils.unif(n)

    M = ot.dist(x, y)
    M = M / M.max()
    a, b, M = nx.from_numpy(a_np, b_np, M)
    c = a[:, None] * b[None, :]

    reg = 1e-2
    reg_m = 100

    G0, _ = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, c=None, reg=reg,
                                        div=div, verbose=False, log=True)
    loss_0 = ot.unbalanced.mm_unbalanced2(a, b, M, reg_m=reg_m, c=None, reg=reg,
                                          div=div, verbose=True)
    loss_0 = nx.to_numpy(loss_0)

    G1, _ = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, c=c,
                                        reg=reg, div=div,
                                        verbose=False, log=True)
    loss_1 = ot.unbalanced.mm_unbalanced2(a, b, M, reg_m=reg_m, c=c,
                                          reg=reg, div=div, verbose=True)
    loss_1 = nx.to_numpy(loss_1)

    np.testing.assert_allclose(nx.to_numpy(G0), nx.to_numpy(G1), atol=1e-05)
    np.testing.assert_allclose(loss_0, loss_1, atol=1e-5)


def test_mm_wrong_divergence(nx):

    n = 100
    rng = np.random.RandomState(42)
    x = rng.randn(n, 2)
    rng = np.random.RandomState(75)
    y = rng.randn(n, 2)
    a_np = ot.utils.unif(n)
    b_np = ot.utils.unif(n)

    M = ot.dist(x, y)
    M = M / M.max()
    a, b, M = nx.from_numpy(a_np, b_np, M)

    reg = 1e-2
    reg_m = 100

    def mm_div(div):
        return ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, reg=reg,
                                           div=div, verbose=False, log=True)

    def mm2_div(div):
        return ot.unbalanced.mm_unbalanced2(a, b, M, reg_m=reg_m, reg=reg,
                                            div=div, verbose=True)

    np.testing.assert_raises(ValueError, mm_div, "div_not_existed")
    np.testing.assert_raises(ValueError, mm2_div, "div_not_existed")
