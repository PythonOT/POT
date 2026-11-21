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

        inner_reg = 1e-1 if reg is None or reg == 0 else 1e-3
        res = solve_batch(
            C,
            max_iter=10000,
            tol=tol,
            grad="detach",
            reg=reg,
            method=method,
            reg_type=reg_type,
            inner_reg=inner_reg,
        )
        plan = res.plan
        value = res.value_linear
        if reg is None or reg == 0:
            np.testing.assert_allclose(plan.sum(axis=2), 1 / n, atol=tol * 10)
            np.testing.assert_allclose(plan.sum(axis=1), 1 / d, atol=tol * 10)
        else:
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
    inner_reg = 1e-1 if reg is None or reg == 0 else 1e-3
    res = proximal_bregman_log_plan_batch(
        nx.from_numpy(C),
        reg=reg,
        inner_reg=inner_reg,
        max_iter=10000,
        tol=tol,
        inner_iter=inner_iter,
        grad="detach",
    )
    plan = nx.to_numpy(res["T"])
    np.testing.assert_allclose(plan.sum(axis=2), 1 / n, atol=tol * 10)
    np.testing.assert_allclose(plan.sum(axis=1), 1 / d, atol=tol * 10)
    for i in range(batchsize):
        C_i = C[i]
        res_i = solve(
            C_i,
            reg=reg,
            tol=tol,
        )
        if reg is None or reg == 0:
            value = np.sum(C_i * plan[i])
            np.testing.assert_allclose(res_i.value_linear, value, atol=tol * 10)
        else:
            np.testing.assert_allclose(res_i.plan, plan[i], atol=tol * 10)


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
        assert loss.grad is None
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


@pytest.mark.skipif(not torch, reason="torch not installed")
@pytest.mark.parametrize("method", ["proximal", "sinkhorn", "log_sinkhorn"])
def test_solve_batch_device(method):
    """Solve on each available device without internal CPU/GPU mismatch."""
    batchsize = 2
    n = 4
    d = 2
    X = np.random.randn(batchsize, n, d)

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        Xd = torch.tensor(X, device=device)
        M = dist_batch(Xd, Xd)
        res = solve_batch(M, reg=0.1, max_iter=10, tol=1e-5, method=method)
        assert res.plan.device == Xd.device


def test_dist_batch_large_common_offset_counterexample():
    """Gram cancellation returned -8 where the exact squared distance is 6."""
    X = np.array(
        [
            [
                [100000001.0, 100000004.0, 100000001.0],
                [100000003.0, 100000003.0, 100000002.0],
            ]
        ]
    )
    M = dist_batch(X, X, metric="sqeuclidean")
    np.testing.assert_allclose(M, [[[0.0, 6.0], [6.0, 0.0]]], rtol=0, atol=0)
    M1 = dist_batch(X, X, metric="euclidean")
    np.testing.assert_allclose(
        M1, [[[0.0, np.sqrt(6.0)], [np.sqrt(6.0), 0.0]]], rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("dtype,offset", [(np.float32, 1e4), (np.float64, 1e8)])
def test_dist_batch_large_offset_properties(dtype, offset):
    """Non-negativity, exact self-distance and agreement with coordinate differences."""
    rng = np.random.RandomState(0)
    X = (offset + rng.randn(1, 6, 13)).astype(dtype)
    M = dist_batch(X, X, metric="sqeuclidean")
    assert np.all(M >= 0)
    atol = 1e-3 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(np.diagonal(M, axis1=1, axis2=2), 0.0, rtol=0, atol=atol)
    D = X[:, :, None, :].astype(np.float64) - X[:, None, :, :].astype(np.float64)
    expected = np.sum(D**2, axis=-1)
    rtol = 1e-3 if dtype == np.float32 else 1e-10
    atol = 1e-2 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(M, expected, rtol=rtol, atol=atol)
    M1 = dist_batch(X, X, metric="euclidean")
    assert not np.any(np.isnan(M1))


def test_dist_batch_translation_invariance_exact():
    """Distances of integer-valued points are unchanged by an exact large shift."""
    rng = np.random.RandomState(42)
    X = rng.randint(-4, 5, size=(1, 5, 4)).astype(np.float64)
    Y = rng.randint(-4, 5, size=(1, 5, 4)).astype(np.float64)
    c = 2.0**27  # X + c remains exactly representable
    np.testing.assert_allclose(
        dist_batch(X + c, Y + c), dist_batch(X, Y), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch, reason="torch not installed")
def test_dist_batch_torch_large_offset_gradient():
    """The shifted implementation keeps correct, translation-consistent gradients."""
    rng = np.random.RandomState(11)
    base = rng.randn(1, 4, 5)
    c = 1e8

    def value_and_grad(offset):
        X = torch.tensor(base + offset, dtype=torch.float64, requires_grad=True)
        M = dist_batch(X, X, metric="sqeuclidean")
        M.sum().backward()
        return M.detach(), X.grad.clone()

    M_c, g_c = value_and_grad(c)
    _, g_0 = value_and_grad(0.0)
    D = torch.tensor(base + c, dtype=torch.float64)
    expected = ((D[:, :, None, :] - D[:, None, :, :]) ** 2).sum(-1)
    torch.testing.assert_close(M_c, expected, rtol=1e-10, atol=1e-10)
    assert torch.isfinite(g_c).all()
    torch.testing.assert_close(g_c, g_0, rtol=1e-8, atol=1e-7)


def test_solve_batch_large_common_offset():
    """A corrupted cost matrix produces a negative OT value; the fix must not."""
    rng = np.random.RandomState(7)
    X = 1e8 + rng.randn(2, 5, 13)
    Y = 1e8 + rng.randn(2, 5, 13)
    M = dist_batch(X, Y, metric="sqeuclidean")
    D = X[:, :, None, :] - Y[:, None, :, :]
    M_ref = np.sum(D**2, axis=-1)
    kw = {
        "reg": 0.1,
        "method": "proximal",
        "max_iter": 1000,
        "tol": 1e-9,
        "grad": "detach",
    }
    res = solve_batch(M, **kw)
    res_ref = solve_batch(M_ref, **kw)
    assert np.all(np.asarray(res.value_linear) > 0)
    np.testing.assert_allclose(
        res.value_linear, res_ref.value_linear, rtol=1e-6, atol=1e-8
    )


def test_dist_batch_empty_source_samples():
    """Zero-sample batches must keep working."""
    X = np.zeros((2, 0, 3))
    Y = np.zeros((2, 4, 3))
    assert dist_batch(X, Y).shape == (2, 0, 4)
    assert dist_batch(X, X).shape == (2, 0, 0)
