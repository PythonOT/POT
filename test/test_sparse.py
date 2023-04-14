"""Tests for ot.smooth model """

# Author: Tianlin Liu <t.liu@unibas.ch>
#
# License: MIT License

import numpy as np
import ot


def test_sparsity_constrained_ot_dual():

    # get data
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    max_nz = 2

    plan = ot.sparse.sparsity_constrained_ot_dual(
        u, u, M, 1, max_nz=max_nz, stopThr=1e-10)

    # check marginal constraints
    np.testing.assert_allclose(u, plan.sum(1), atol=1e-03)
    np.testing.assert_allclose(u, plan.sum(0), atol=1e-03)

    # check sparsity constraint
    np.testing.assert_array_less(
        np.sum(plan > 0, axis=0),
        np.ones(n) * max_nz + 1)


def test_sparsity_constrained_ot_semi_dual():

    # get data
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    max_nz = 2
    plan, log = ot.sparse.sparsity_constrained_ot_semi_dual(
        u, u, M, 1,
        max_nz=max_nz,
        log=True,
        stopThr=1e-10)

    # check marginal constraints
    np.testing.assert_allclose(u, plan.sum(1), atol=1e-03)
    np.testing.assert_allclose(u, plan.sum(0), atol=1e-03)

    # check sparsity constraint
    np.testing.assert_array_less(
        np.sum(plan > 0, axis=0),
        np.ones(n) * max_nz + 1)
