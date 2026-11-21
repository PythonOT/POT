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


@pytest.mark.skipif(not torch, reason="torch not installed")
@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_solve_batch_envelope_entropy_gradients_torch(batch_size):
    """Envelope mode must propagate M and marginal gradients batch-wise."""
    rng = np.random.RandomState(0)
    p_np = 0.5 + 0.4 * rng.rand(batch_size)
    w_np = rng.randn(batch_size)

    p = torch.tensor(p_np, dtype=torch.float64, requires_grad=True)
    a = torch.stack([p, 1 - p], dim=1)
    b = torch.full((batch_size, 2), 0.5, dtype=torch.float64)
    M = torch.zeros((batch_size, 2, 2), dtype=torch.float64, requires_grad=True)
    w = torch.tensor(w_np, dtype=torch.float64)

    res = solve_batch(
        M, reg=1.0, a=a, b=b, method="log_sinkhorn", reg_type="entropy",
        grad="envelope", max_iter=2000, tol=1e-14,
    )
    grad_M, grad_p = torch.autograd.grad(
        (res.value * w).sum(), (M, p), allow_unused=True
    )

    # M = 0 gives F(p) = reg*(p log p + (1-p) log(1-p) - log 2)
    expected_p = w_np * (np.log(p_np) - np.log(1 - p_np))
    expected_M = w_np[:, None, None] * res.plan.detach().numpy()
    assert grad_p is not None
    np.testing.assert_allclose(grad_p.detach().numpy(), expected_p, atol=1e-10)
    np.testing.assert_allclose(grad_M.numpy(), expected_M, atol=1e-12)
    assert grad_p.shape == p.shape and grad_p.dtype == p.dtype
    assert grad_M.shape == M.shape and grad_M.device == M.device


@pytest.mark.skipif(not torch, reason="torch not installed")
def test_solve_batch_envelope_two_batch_exact_cotangent():
    """p1=3/4, p2=4/5 with w=(1,-2) must give (log 3, -2 log 4)."""
    p = torch.tensor([0.75, 0.8], dtype=torch.float64, requires_grad=True)
    a = torch.stack([p, 1 - p], dim=1)
    b = torch.full((2, 2), 0.5, dtype=torch.float64)
    M = torch.zeros((2, 2, 2), dtype=torch.float64, requires_grad=True)
    res = solve_batch(
        M, reg=1.0, a=a, b=b, method="log_sinkhorn", reg_type="entropy",
        grad="envelope", max_iter=2000, tol=1e-14,
    )
    _, grad_p = torch.autograd.grad(
        (res.value * torch.tensor([1.0, -2.0], dtype=torch.float64)).sum(),
        (M, p), allow_unused=True,
    )
    np.testing.assert_allclose(
        grad_p.numpy(), [np.log(3.0), -2 * np.log(4.0)], atol=1e-12
    )


@pytest.mark.skipif(not torch, reason="torch not installed")
@pytest.mark.parametrize("reg_type", ["entropy", "kl"])
def test_solve_batch_envelope_marginal_vs_finite_difference(reg_type):
    """Independent mass-conserving finite difference, including KL reference term."""
    M_np = np.array([[0.1, 0.9], [1.3, 0.2]], dtype=np.float64)
    p0 = 0.75
    h = 1e-6

    def value_at(p):
        a = np.array([[p, 1 - p]], dtype=np.float64)
        b = np.full((1, 2), 0.5, dtype=np.float64)
        return float(solve_batch(
            M_np[None], reg=1.0, a=a, b=b, method="log_sinkhorn",
            reg_type=reg_type, grad="detach", max_iter=5000, tol=1e-14,
        ).value[0])

    oracle = (value_at(p0 + h) - value_at(p0 - h)) / (2 * h)

    p = torch.tensor([p0], dtype=torch.float64, requires_grad=True)
    a = torch.cat([p, 1 - p]).unsqueeze(0)
    b = torch.full((1, 2), 0.5, dtype=torch.float64)
    M = torch.tensor(M_np[None], dtype=torch.float64, requires_grad=True)
    res = solve_batch(
        M, reg=1.0, a=a, b=b, method="log_sinkhorn", reg_type=reg_type,
        grad="envelope", max_iter=5000, tol=1e-14,
    )
    _, grad_p = torch.autograd.grad(res.value.sum(), (M, p), allow_unused=True)
    assert grad_p is not None
    np.testing.assert_allclose(grad_p.item(), oracle, atol=1e-7)


@pytest.mark.skipif(not torch, reason="torch not installed")
@pytest.mark.parametrize("reg_type", ["entropy", "kl"])
def test_solve_batch_envelope_forward_unchanged(reg_type):
    """Envelope VJP registration must not alter value or plan."""
    rng = np.random.RandomState(1)
    M_np = rng.rand(2, 3, 4)
    a_np = rng.rand(2, 3)
    a_np /= a_np.sum(axis=1, keepdims=True)
    b_np = rng.rand(2, 4)
    b_np /= b_np.sum(axis=1, keepdims=True)

    def solve(grad):
        return solve_batch(
            torch.tensor(M_np, requires_grad=True),
            reg=0.5, a=torch.tensor(a_np), b=torch.tensor(b_np),
            method="log_sinkhorn", reg_type=reg_type, grad=grad,
            max_iter=1000, tol=1e-12,
        )

    ref, env = solve("detach"), solve("envelope")
    np.testing.assert_array_equal(env.value.detach().numpy(), ref.value.numpy())
    np.testing.assert_array_equal(env.plan.detach().numpy(), ref.plan.numpy())


@pytest.mark.skipif(not torch, reason="torch not installed")
def test_solve_batch_envelope_both_marginals_analytic_torch():
    """Both a and b gradients in one autograd call, against closed-form values.

    For M=0 and reg=1 the envelope objective separates as
    F = a log a + b log b, hence with a=(p,1-p), p=3/4 and
    b=(q, 1/3, 2/3-q), q=1/2: dF/dp = dF/dq = log 3.
    """
    p = torch.tensor([0.75], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    a = torch.cat([p, 1 - p]).unsqueeze(0)
    b = torch.cat([q, torch.tensor([1.0 / 3.0], dtype=torch.float64), 2.0 / 3.0 - q])
    b = b.unsqueeze(0)
    M = torch.zeros((1, 2, 3), dtype=torch.float64, requires_grad=True)
    w = torch.tensor([-2.0], dtype=torch.float64)

    res = solve_batch(
        M, reg=1.0, a=a, b=b, method="log_sinkhorn", reg_type="entropy",
        grad="envelope", max_iter=2000, tol=1e-14,
    )
    grad_M, grad_p, grad_q = torch.autograd.grad(
        (res.value * w).sum(), (M, p, q), allow_unused=True
    )

    expected = float(np.log(3.0))
    assert grad_p is not None and grad_q is not None
    # Upstream cotangent scaling w=-2 is applied to both marginals.
    np.testing.assert_allclose(grad_p.numpy(), [w.item() * expected], atol=1e-12)
    np.testing.assert_allclose(grad_q.numpy(), [w.item() * expected], atol=1e-12)
    # grad_M = w * P* and only the M path is scaled by the plan.
    np.testing.assert_allclose(
        grad_M.numpy(), w.item() * res.plan.detach().numpy(), atol=1e-12
    )
    assert grad_p.shape == p.shape and grad_q.shape == q.shape
    assert grad_p.dtype == p.dtype and grad_q.dtype == q.dtype
    assert grad_p.device == p.device and grad_q.device == q.device
