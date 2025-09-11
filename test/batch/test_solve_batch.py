"""Tests for module bregman on OT with bregman projections"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Kilian Fatras <kilian.fatras@irisa.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#         Eduardo Fernandes Montesuma <eduardo.fernandes-montesuma@universite-paris-saclay.fr>
#
# License: MIT License

import numpy as np
from ot.batch import (
    solve_batch,
    solve_sample_batch,
    dist_batch,
    loss_linear_samples_batch,
    loss_linear_batch,
)

from ot import solve
import pytest


def test_solve_batch():
    """Check that solve_batch gives the same results as solve for each instance in the batch."""
    batchsize = 4
    n = 16
    rng = np.random.RandomState(0)

    M = rng.rand(batchsize, n, n)

    reg = 0.1
    max_iter = 10000
    tol = 1e-5

    res = solve_batch(
        M,
        a=None,
        b=None,
        reg=reg,
        max_iter=max_iter,
        tol=tol,
        solver="log_sinkhorn",
        grad="detach",
    )
    plan_batch = res.plan
    values_batch = res.value_linear

    for i in range(batchsize):
        M_i = M[i]
        res_i = solve(M_i, a=None, b=None, reg=reg, max_iter=max_iter, tol=tol)
        plan_i = res_i.plan
        value_i = res_i.value_linear
        np.testing.assert_allclose(plan_i, plan_batch[i], atol=1e-05)
        np.testing.assert_allclose(value_i, values_batch[i], atol=1e-4)


@pytest.mark.parametrize("metric", ["sqeuclidean", "euclidean", "minkowski", "kl"])
def test_all(metric):
    """Check that all functions run without error."""

    batchsize = 2
    n = 4
    d = 2
    rng = np.random.RandomState(0)
    X = rng.rand(batchsize, n, d)
    if metric == "kl":
        X = np.abs(X) + 1e-6
        X = X / np.sum(X, axis=-1, keepdims=True)
    M = dist_batch(X, X, metric=metric)
    is_positive = M >= 0
    np.testing.assert_equal(is_positive.all(), True)

    # Solve batch
    res = solve_batch(M, reg=0.1, max_iter=10, tol=1e-5)

    # Solve sample batch
    res = solve_sample_batch(X, X, reg=0.1, max_iter=10, tol=1e-5, metric=metric)

    # Compute loss
    loss = res.value_linear  # loss given by solver
    loss2 = loss_linear_batch(M, res.plan)  # recompute loss from plan
    loss3 = loss_linear_samples_batch(
        X, X, res.plan, metric=metric
    )  # recompute loss from plan and samples
    np.testing.assert_allclose(loss, loss2, atol=1e-5)
    np.testing.assert_allclose(loss, loss3, atol=1e-5)
