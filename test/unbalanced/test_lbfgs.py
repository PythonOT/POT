"""Tests for module Unbalanced OT with entropy regularization"""

# Author: Hicham Janati <hicham.janati@inria.fr>
#         Laetitia Chapel <laetitia.chapel@univ-ubs.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License


import itertools
import numpy as np
import ot
import pytest


@pytest.mark.parametrize("reg_div,regm_div,returnCost", itertools.product(['kl', 'l2', 'entropy'], ['kl', 'l2', 'tv'], ['linear', 'total']))
def test_lbfgsb_unbalanced(nx, reg_div, regm_div, returnCost):

    np.random.seed(42)

    xs = np.random.randn(5, 2)
    xt = np.random.randn(6, 2)

    M = ot.dist(xs, xt)

    a = ot.unif(5)
    b = ot.unif(6)

    G, log = ot.unbalanced.lbfgsb_unbalanced(a, b, M, 1, 10,
                                             reg_div=reg_div, regm_div=regm_div,
                                             log=True, verbose=False)
    loss, _ = ot.unbalanced.lbfgsb_unbalanced2(a, b, M, 1, 10,
                                               reg_div=reg_div, regm_div=regm_div,
                                               returnCost=returnCost, log=True, verbose=False)

    ab, bb, Mb = nx.from_numpy(a, b, M)

    Gb, log = ot.unbalanced.lbfgsb_unbalanced(ab, bb, Mb, 1, 10,
                                              reg_div=reg_div, regm_div=regm_div,
                                              log=True, verbose=False)
    loss0, log = ot.unbalanced.lbfgsb_unbalanced2(ab, bb, Mb, 1, 10,
                                                  reg_div=reg_div, regm_div=regm_div,
                                                  returnCost=returnCost, log=True, verbose=False)

    np.testing.assert_allclose(G, nx.to_numpy(Gb))
    np.testing.assert_allclose(loss, nx.to_numpy(loss0), atol=1e-06)


@pytest.mark.parametrize("reg_div,regm_div,returnCost", itertools.product(['kl', 'l2', 'entropy'], ['kl', 'l2', 'tv'], ['linear', 'total']))
def test_lbfgsb_unbalanced_relaxation_parameters(nx, reg_div, regm_div, returnCost):

    np.random.seed(42)

    xs = np.random.randn(5, 2)
    xt = np.random.randn(6, 2)

    M = ot.dist(xs, xt)

    a = ot.unif(5)
    b = ot.unif(6)

    a, b, M = nx.from_numpy(a, b, M)

    reg_m = 10
    full_list_reg_m = [reg_m, reg_m]
    full_tuple_reg_m = (reg_m, reg_m)
    tuple_reg_m, list_reg_m = (reg_m), [reg_m]
    np1_reg_m = reg_m * np.ones(1)
    np2_reg_m = reg_m * np.ones(2)

    list_options = [np1_reg_m, np2_reg_m, full_tuple_reg_m,
                    tuple_reg_m, full_list_reg_m, list_reg_m]

    G = ot.unbalanced.lbfgsb_unbalanced(a, b, M, 1, reg_m=reg_m,
                                        reg_div=reg_div, regm_div=regm_div,
                                        log=False, verbose=False)
    loss = ot.unbalanced.lbfgsb_unbalanced2(a, b, M, 1,
                                            reg_m=reg_m, reg_div=reg_div, regm_div=regm_div,
                                            returnCost=returnCost, log=False, verbose=False)

    for opt in list_options:
        G0 = ot.unbalanced.lbfgsb_unbalanced(
            a, b, M, 1, reg_m=opt, reg_div=reg_div,
            regm_div=regm_div, log=False, verbose=False
        )
        loss0 = ot.unbalanced.lbfgsb_unbalanced2(
            a, b, M, 1, reg_m=opt, reg_div=reg_div,
            regm_div=regm_div, returnCost=returnCost,
            log=False, verbose=False
        )

        np.testing.assert_allclose(nx.to_numpy(G), nx.to_numpy(G0), atol=1e-06)
        np.testing.assert_allclose(nx.to_numpy(loss), nx.to_numpy(loss0), atol=1e-06)


@pytest.mark.parametrize("reg_div,regm_div,returnCost", itertools.product(['kl', 'l2', 'entropy'], ['kl', 'l2', 'tv'], ['linear', 'total']))
def test_lbfgsb_reference_measure(nx, reg_div, regm_div, returnCost):

    np.random.seed(42)

    xs = np.random.randn(5, 2)
    xt = np.random.randn(6, 2)
    M = ot.dist(xs, xt)
    a = ot.unif(5)
    b = ot.unif(6)

    a, b, M = nx.from_numpy(a, b, M)
    c = a[:, None] * b[None, :]

    G, _ = ot.unbalanced.lbfgsb_unbalanced(a, b, M, reg=1, reg_m=10, c=None,
                                           reg_div=reg_div, regm_div=regm_div,
                                           log=True, verbose=False)
    loss, _ = ot.unbalanced.lbfgsb_unbalanced2(a, b, M, reg=1, reg_m=10, c=None,
                                               reg_div=reg_div, regm_div=regm_div,
                                               returnCost=returnCost, log=True, verbose=False)

    G0, _ = ot.unbalanced.lbfgsb_unbalanced(a, b, M, reg=1, reg_m=10, c=c,
                                            reg_div=reg_div, regm_div=regm_div,
                                            log=True, verbose=False)

    loss0, _ = ot.unbalanced.lbfgsb_unbalanced2(a, b, M, reg=1, reg_m=10, c=c,
                                                reg_div=reg_div, regm_div=regm_div,
                                                returnCost=returnCost, log=True, verbose=False)

    np.testing.assert_allclose(nx.to_numpy(G), nx.to_numpy(G0), atol=1e-06)
    np.testing.assert_allclose(nx.to_numpy(loss), nx.to_numpy(loss0), atol=1e-06)


def test_lbfgsb_wrong_divergence(nx):

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

    def lbfgsb_div(div):
        return ot.unbalanced.lbfgsb_unbalanced(a, b, M, reg=1, reg_m=10, reg_div=div)

    def lbfgsb2_div(div):
        return ot.unbalanced.lbfgsb_unbalanced2(a, b, M, reg=1, reg_m=10, reg_div=div)

    np.testing.assert_raises(ValueError, lbfgsb_div, "div_not_existed")
    np.testing.assert_raises(ValueError, lbfgsb2_div, "div_not_existed")


def test_lbfgsb_wrong_marginal_divergence(nx):

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

    def lbfgsb_div(div):
        return ot.unbalanced.lbfgsb_unbalanced(a, b, M, reg=1, reg_m=10, regm_div=div)

    def lbfgsb2_div(div):
        return ot.unbalanced.lbfgsb_unbalanced2(a, b, M, reg=1, reg_m=10, regm_div=div)

    np.testing.assert_raises(ValueError, lbfgsb_div, "div_not_existed")
    np.testing.assert_raises(ValueError, lbfgsb2_div, "div_not_existed")


def test_lbfgsb_wrong_returnCost(nx):

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

    def lbfgsb2(returnCost):
        return ot.unbalanced.lbfgsb_unbalanced2(a, b, M, reg=1, reg_m=10,
                                                returnCost=returnCost, verbose=True)

    np.testing.assert_raises(ValueError, lbfgsb2, "invalid_returnCost")
