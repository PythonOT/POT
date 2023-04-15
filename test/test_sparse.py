"""Tests for ot.sparse model """

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


def test_projection_sparse_simplex():

    def double_sort_projection_sparse_simplex(X, max_nz, z=1, axis=None):
        r"""This is an equivalent but less efficient version
        of ot.sparse.projection_sparse_simplex, as it uses two
        sorts instead of one.
        """

        if axis == 0:
            # For each column of X, find top max_nz values and
            # their corresponding indices. This incurs a sort.
            max_nz_indices = np.argpartition(
                X,
                kth=-max_nz,
                axis=0)[-max_nz:]

            max_nz_values = X[max_nz_indices, np.arange(X.shape[1])]

            # Project the top max_nz values onto the simplex.
            # This incurs a second sort.
            G_nz_values = ot.smooth.projection_simplex(
                max_nz_values, z=z, axis=0)

            # Put the projection of max_nz_values to their original indices
            # and set all other values zero.
            G = np.zeros_like(X)
            G[max_nz_indices, np.arange(X.shape[1])] = G_nz_values
            return G
        elif axis == 1:
            return double_sort_projection_sparse_simplex(
                X.T, max_nz, z, axis=0).T

        else:
            X = X.ravel().reshape(-1, 1)
            return double_sort_projection_sparse_simplex(
                X, max_nz, z, axis=0).ravel()

    m, n = 5, 10
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(m, n))
    max_nz = 3

    for axis in [0, 1, None]:
        slow_sparse_proj = double_sort_projection_sparse_simplex(
            X, max_nz, axis=axis)
        fast_sparse_proj = ot.sparse.projection_sparse_simplex(
            X, max_nz, axis=axis)

        # check that two versions produce the same result
        np.testing.assert_allclose(
            slow_sparse_proj, fast_sparse_proj)
