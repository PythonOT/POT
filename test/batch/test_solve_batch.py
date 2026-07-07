"""Tests for module batch"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Paul Krzakala <paul.krzakala@gmail.com>
#         Sonia Mazelet <sonia.mazelet@polytechnique.edu>
#         Thibaut Germain <thibaut.germain.pro@gmail.com>
#
# License: MIT License

import numpy as np
from ot.batch import (
    solve_batch,
    solve_sample_batch,
    dist_batch,
    loss_linear_samples_batch,
    loss_linear_batch,
    bregman_projection_batch,
    bregman_log_projection_batch,
    proximal_bregman_log_plan_batch,
)

from ot import solve
from contextlib import nullcontext
import pytest
from ot.backend import torch


@pytest.mark.parametrize("reg", [None, 0, 1e-0])
@pytest.mark.parametrize("method", ["auto", "proximal", "sinkhorn", "log_sinkhorn"])
@pytest.mark.parametrize("reg_type", ["kl", "entropy"])
def test_solve_batch_vs_solve(reg, method, reg_type):
    """Check that solve_batch gives the same results as solve for each instance in the batch."""

    should_fail = method in ["sinkhorn", "log_sinkhorn"] and (reg is None or reg <= 0)

    ctx = pytest.raises(Exception) if should_fail else nullcontext()

    with ctx:
        tol = 1e-4
        batchsize = 3
        n = 5
        d = 7
        rng = np.random.RandomState(0)
        C = rng.rand(batchsize, n, d)

        base_plan = np.zeros((batchsize, n, d))
        base_value = np.zeros(batchsize)
        for i in range(batchsize):
            C_i = C[i]
            res_i = solve(C_i, reg=reg, tol=tol, reg_type=reg_type)
            base_plan[i] = res_i.plan
            base_value[i] = res_i.value_linear

        res = solve_batch(
            C,
            max_iter=10000,
            tol=tol,
            grad="detach",
            reg=reg,
            method=method,
            reg_type=reg_type,
            inner_reg=1e-3,
        )
        plan = res.plan
        value = res.value_linear
        np.testing.assert_allclose(plan, base_plan, atol=tol * 10)
        np.testing.assert_allclose(value, base_value, atol=tol * 10)


@pytest.mark.parametrize("reg", [None, 0, 1e-0])
@pytest.mark.parametrize("inner_iter", [1, 5, 10])
def test_backend_proximal_bregman_log_plan_batch(nx, reg, inner_iter):
    tol = 1e-4
    batchsize = 3
    n = 5
    d = 7
    rng = np.random.RandomState(0)
    C = rng.rand(batchsize, n, d)
    res = proximal_bregman_log_plan_batch(
        nx.from_numpy(C),
        reg=reg,
        inner_reg=1e-3,
        max_iter=10000,
        tol=tol,
        inner_iter=inner_iter,
        grad="detach",
    )
    plan = nx.to_numpy(res["T"])
    for i in range(batchsize):
        C_i = C[i]
        res_i = solve(
            C_i,
            reg=reg,
            tol=tol,
        )
        plan_i = res_i.plan
        np.testing.assert_allclose(plan_i, plan[i], atol=tol * 10)


def test_bregman_batch():
    batchsize = 4
    d = 2
    n = 4
    rng = np.random.RandomState(0)
    X = rng.rand(batchsize, n, d)
    M = dist_batch(X, X)
    K = np.exp(-M / 0.01)
    log_K = -M / 0.01
    res = bregman_projection_batch(K, max_iter=50, tol=1e-10)
    plan = res["T"]
    res_log = bregman_log_projection_batch(log_K, max_iter=50, tol=1e-10)
    plan_log = res_log["T"]
    np.testing.assert_allclose(plan, plan_log, atol=1e-3)


@pytest.mark.parametrize("metric", ["sqeuclidean", "euclidean", "minkowski", "kl"])
@pytest.mark.parametrize("method", ["proximal", "sinkhorn", "log_sinkhorn"])
def test_sample_solve_batch_vs_solve_batch(metric, method):
    """Check that all functions run without error."""
    tol = 1e-5
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

    # Solve sample batch
    res = solve_sample_batch(
        X, X, reg=0.1, max_iter=10, tol=tol, metric=metric, method=method
    )

    # Compute loss
    loss = res.value_linear  # loss given by solver
    loss2 = loss_linear_batch(M, res.plan)  # recompute loss from plan
    loss3 = loss_linear_samples_batch(
        X, X, res.plan, metric=metric
    )  # recompute loss from plan and samples
    np.testing.assert_allclose(loss, loss2, atol=tol * 10)
    np.testing.assert_allclose(loss, loss3, atol=tol * 10)


@pytest.mark.skipif(not torch, reason="torch not installed")
@pytest.mark.parametrize("grad", ["detach", "envelope", "autodiff", "last_step"])
@pytest.mark.parametrize("method", ["proximal", "sinkhorn", "log_sinkhorn"])
def test_gradients_torch(grad, method):
    """Check that all gradient methods run without error."""
    batchsize = 2
    n = 4
    d = 2
    X = torch.randn((batchsize, n, d), requires_grad=True)
    M = dist_batch(X, X)
    res = solve_batch(M, reg=0.1, max_iter=10, tol=1e-5, grad=grad, method=method)
    loss = res.value_linear.sum()
    loss_plan = res.plan.sum()
    if grad == "detach":
        assert loss.grad == None
    elif grad == "envelope":
        loss.backward()
        assert X.grad is not None
    elif grad in ["autodiff", "last_step"]:
        loss_plan.backward()
        assert X.grad is not None


@pytest.mark.parametrize("method", ["proximal", "sinkhorn", "log_sinkhorn"])
def test_backend(nx, method):
    """Check that all gradient methods run without error."""
    batchsize = 2
    n = 4
    d = 2
    X = np.random.randn(batchsize, n, d)
    X = nx.from_numpy(X)
    M = dist_batch(X, X)
    solve_batch(M, reg=0.1, max_iter=10, tol=1e-5, method=method)
    solve_sample_batch(X, X, reg=0.1, max_iter=10, tol=1e-5, method=method)
