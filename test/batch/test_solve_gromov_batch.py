"""Tests for module batch"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Paul Krzakala <paul.krzakala@gmail.com>
#         Sonia Mazelet <sonia.mazelet@polytechnique.edu>


#
# License: MIT License

import numpy as np
from ot.batch import (
    solve_gromov_batch,
    loss_quadratic_batch,
    loss_linear_batch,
)
from ot import solve_gromov
from ot.batch._linear import dist_batch
import pytest
from itertools import product
from ot.backend import torch, get_backend
from ot.batch._quadratic import (
    div_between_product_batch,
    loss_quadratic_samples_batch,
)
from ot.gromov._utils import div_between_product


def test_solve_gromov_batch():
    """Check that solve_gromov_batch gives the same results as solve for each instance in the batch."""
    b = 2
    n = 8
    d = 2
    reg = 0.01
    max_iter = 1000
    max_iter_inner = 10000
    tol = 1e-5
    tol_inner = 1e-5
    alpha = 0.5

    rng = np.random.RandomState(0)

    X1 = rng.randn(b, n, d).astype("float32")
    C1 = rng.randn(b, n, n).astype("float32")

    permutation = np.random.permutation(n)
    X2 = X1[:, permutation, :] + 0.01 * rng.randn(b, n, d).astype("float32")
    C2 = C1[:, permutation, :][:, :, permutation] + 0.01 * rng.randn(b, n, n).astype(
        "float32"
    )

    M = dist_batch(X1, X2)

    res = solve_gromov_batch(
        alpha=alpha,
        reg=reg,
        M=M,
        Ca=C1,
        Cb=C2,
        max_iter=max_iter,
        tol=tol,
        max_iter_inner=max_iter_inner,
        tol_inner=tol_inner,
        symmetric=False,
    )

    plan_batch = res.plan
    values_quadratic_batch = res.value_quad
    values_linear_batch = res.value_linear

    for i in range(b):
        M_i = M[i]
        C1_i = C1[i]
        C2_i = C2[i]
        res_i = solve_gromov(C1_i, C2_i, M=M_i, alpha=alpha, symmetric=False)
        plan_i = res_i.plan
        values_quadratic_i = res_i.value_quad
        values_linear_i = res_i.value_linear
        np.testing.assert_allclose(values_linear_i, values_linear_batch[i], atol=1e-4)
        np.testing.assert_allclose(
            values_quadratic_i, values_quadratic_batch[i], atol=1e-4
        )
        np.testing.assert_allclose(plan_i, plan_batch[i], atol=1e-05)


@pytest.mark.parametrize(
    "loss, logits",
    product(
        ["sqeuclidean", "kl"],
        [True, False],
    ),
)
def test_all(loss, logits):
    """Check that all functions run without error."""

    batchsize = 2
    n = 4
    d = 2
    rng = np.random.RandomState(0)
    C = rng.rand(batchsize, n, n, d)
    a = np.ones((batchsize, n))
    if loss == "kl":
        C = np.abs(C) + 1e-6
        C = C / np.sum(C, axis=-1, keepdims=True)

    res = solve_gromov_batch(Ca=C, Cb=C, a=a, b=a, loss=loss, logits=logits)

    loss1 = res.value_quad
    loss2 = loss_quadratic_batch(
        a=a, b=a, Ca=C, Cb=C, T=res.plan, loss=loss, logits=logits
    )
    np.testing.assert_allclose(loss1, loss2, atol=1e-5)


@pytest.mark.skipif(not torch, reason="torch not installed")
@pytest.mark.parametrize("grad", ["detach", "envelope", "autodiff"])
def test_gradients_torch(grad):
    """Check that all gradient methods run without error."""
    batchsize = 2
    n = 4
    d = 2
    C = torch.randn((batchsize, n, n, d), requires_grad=True)
    res = solve_gromov_batch(
        Ca=C, Cb=C, a=None, b=None, loss="sqeuclidean", logits=False, grad=grad
    )
    loss = res.value.sum()
    loss_plan = res.plan.sum()
    if grad == "detach":
        assert loss.grad is None
    elif grad == "envelope":
        loss.backward()
        assert C.grad is not None
    elif grad == "autodiff":
        loss_plan.backward()
        assert C.grad is not None


def test_backend(nx):
    """Check that all gradient methods run without error."""
    batchsize = 2
    n = 4
    d = 2
    C = np.random.randn(batchsize, n, n, d)
    C = nx.from_numpy(C)
    solve_gromov_batch(Ca=C, Cb=C, a=None, b=None, loss="sqeuclidean", logits=False)


@pytest.mark.parametrize(
    "loss, logits, unbalanced_type",
    product(["sqeuclidean", "kl"], [True, False], ["kl", "l2"]),
)
def test_fugw_loss(unbalanced_type, loss, logits):
    """Check that loss_quadratic_batch runs without error."""
    batchsize = 2
    n = 4
    d = 2
    rng = np.random.RandomState(0)
    C1 = rng.rand(batchsize, n, n, d)
    C2 = rng.rand(batchsize, n, n, d)
    M = rng.rand(batchsize, n, n)
    a = np.ones((batchsize, n))
    T = rng.rand(batchsize, n, n)
    alpha = rng.rand()
    reg_marginals = rng.rand()

    loss_fugw = loss_quadratic_batch(
        a,
        a,
        C1,
        C2,
        T,
        M,
        alpha=alpha,
        unbalanced=reg_marginals,
        unbalanced_type=unbalanced_type,
        loss=loss,
        logits=logits,
    )

    # unbalanced quadratic
    loss_fugw_unbalanced_only = loss_quadratic_batch(
        a,
        a,
        C1,
        C2,
        T,
        M=None,
        alpha=alpha,
        unbalanced=reg_marginals,
        unbalanced_type=unbalanced_type,
        loss=loss,
        logits=logits,
    )

    # alpha is None
    loss_fugw_no_alpha = loss_quadratic_batch(
        a,
        a,
        C1,
        C2,
        T,
        M=None,
        alpha=None,
        unbalanced=reg_marginals,
        unbalanced_type=unbalanced_type,
        loss=loss,
        logits=logits,
    )

    # balanced
    loss_fugw_no_unbalanced = loss_quadratic_batch(
        a,
        a,
        C1,
        C2,
        T,
        M=M,
        alpha=alpha,
        loss=loss,
        logits=logits,
    )
    assert np.isfinite(loss_fugw).all()
    assert np.isfinite(loss_fugw_unbalanced_only).all()
    assert np.isfinite(loss_fugw_no_alpha).all()
    assert np.isfinite(loss_fugw_no_unbalanced).all()


def test_fugw_backend(nx):
    """Check that loss_quadratic_batch runs without error for all backends."""
    batchsize = 2
    n = 4
    d = 2
    rng = np.random.RandomState(0)
    C1_np = rng.rand(batchsize, n, n, d)
    C1 = nx.from_numpy(C1_np)
    C2_np = rng.rand(batchsize, n, n, d)
    C2 = nx.from_numpy(C2_np)
    M_np = rng.rand(batchsize, n, n)
    M = nx.from_numpy(M_np)
    a_np = np.ones((batchsize, n))
    a = nx.from_numpy(a_np)
    T_np = rng.rand(batchsize, n, n)
    T = nx.from_numpy(T_np)
    alpha_np = rng.rand()
    alpha = nx.from_numpy(np.array(alpha_np))
    unbalanced_np = rng.rand()
    unbalanced = nx.from_numpy(np.array(unbalanced_np))

    loss_fugw_np = loss_quadratic_batch(
        a_np,
        a_np,
        C1_np,
        C2_np,
        T_np,
        M_np,
        alpha=alpha_np,
        unbalanced=unbalanced_np,
        unbalanced_type="kl",
        loss="sqeuclidean",
        logits=False,
    )
    loss_fugw = loss_quadratic_batch(
        a,
        a,
        C1,
        C2,
        T,
        M,
        alpha=alpha,
        unbalanced=unbalanced,
        unbalanced_type="kl",
        loss="sqeuclidean",
        logits=False,
    )

    assert np.allclose(loss_fugw_np, loss_fugw, atol=1e-5)


def test_fugw_paramters_arrays():
    batchsize = 2
    n = 4
    d = 2
    rng = np.random.RandomState(0)
    C1 = rng.rand(batchsize, n, n, d)
    C2 = rng.rand(batchsize, n, n, d)
    M = rng.rand(batchsize, n, n)
    a = np.ones((batchsize, n))
    T = rng.rand(batchsize, n, n)
    alpha = rng.rand()
    # check that alpha and reg_marginals can be passed as lists or arrays of shape (batchsize,)
    alpha = rng.rand(batchsize)
    unbalanced = rng.rand(batchsize)
    alpha_list = alpha.tolist()
    unbalanced_list = unbalanced.tolist()

    loss_fugw = loss_quadratic_batch(
        a,
        a,
        C1,
        C2,
        T,
        M,
        alpha=alpha,
        unbalanced=unbalanced,
        unbalanced_type="kl",
        loss="l2",
        logits=False,
    )
    loss_fugw_list = loss_quadratic_batch(
        a,
        a,
        C1,
        C2,
        T,
        M,
        alpha=alpha_list,
        unbalanced=unbalanced_list,
        unbalanced_type="kl",
        loss="l2",
        logits=False,
    )

    assert np.isfinite(loss_fugw).all()
    assert np.isfinite(loss_fugw_list).all()
    np.testing.assert_allclose(loss_fugw, loss_fugw_list)


def test_fugw_invalid_loss_values():
    batchsize = 2
    n = 4
    d = 2
    rng = np.random.RandomState(0)
    C1 = rng.rand(batchsize, n, n, d)
    C2 = rng.rand(batchsize, n, n, d)
    M = rng.rand(batchsize, n, n)
    a = np.ones((batchsize, n))
    T = rng.rand(batchsize, n, n)
    alpha = rng.rand()
    unbalanced = rng.rand()

    # check that invalid loss raise an error
    with pytest.raises(ValueError):
        loss_quadratic_batch(
            a,
            a,
            C1,
            C2,
            T,
            M,
            alpha=alpha,
            unbalanced=unbalanced,
            unbalanced_type="kl",
            loss="test",
            logits=False,
        )

    # check that invalid unbalanced_type raise an error
    with pytest.raises(ValueError):
        loss_quadratic_batch(
            a,
            a,
            C1,
            C2,
            T,
            M,
            alpha=alpha,
            unbalanced=unbalanced,
            unbalanced_type="test",
            loss="l2",
            logits=False,
        )


def test_fugw_invalid_shapes():
    batchsize = 2
    n = 4
    d = 2
    rng = np.random.RandomState(0)
    C1 = rng.rand(batchsize, n, n, d)
    C2 = rng.rand(batchsize, n, n, d)
    M = rng.rand(batchsize, n, n)
    a = np.ones((batchsize, n))
    T = rng.rand(batchsize, n, n)
    alpha = rng.rand()
    unbalanced = rng.rand()

    # check that invalid alpha shape raise an error
    alpha = rng.rand(batchsize + 1)
    with pytest.raises(ValueError):
        loss_quadratic_batch(
            a,
            a,
            C1,
            C2,
            T,
            M,
            alpha=alpha,
            unbalanced=unbalanced,
            unbalanced_type="kl",
            loss="l2",
            logits=False,
        )

    # check that invalid rho shape raise an error
    alpha = rng.rand(batchsize)
    unbalanced = rng.rand(batchsize + 1)
    with pytest.raises(ValueError):
        loss_quadratic_batch(
            a,
            a,
            C1,
            C2,
            T,
            M,
            alpha=alpha,
            unbalanced=unbalanced,
            unbalanced_type="kl",
            loss="l2",
            logits=False,
        )


@pytest.mark.parametrize("unbalanced_type", ["kl", "l2"])
@pytest.mark.parametrize("loss", ["sqeuclidean", "kl"])
def test_valid_fugw_loss_endpoints(unbalanced_type, loss):
    """Check that loss_fugw_batch gives the same results as solve_gromov_batch and solve_linear_batch for alpha=0 and alpha=1."""
    batchsize = 2
    n = 4
    d = 2
    rng = np.random.RandomState(0)
    C1 = rng.rand(batchsize, n, n, d)
    C2 = rng.rand(batchsize, n, n, d)
    M = rng.rand(batchsize, n, n)
    a = np.ones((batchsize, n))
    reg_marginals = 0
    T = rng.rand(batchsize, n, n)

    loss_fugw = loss_quadratic_batch(
        a,
        a,
        C1,
        C2,
        T,
        M,
        alpha=0.0,
        unbalanced=reg_marginals,
        unbalanced_type=unbalanced_type,
        loss=loss,
        logits=False,
    )
    loss_linear = loss_linear_batch(M, T)
    np.testing.assert_allclose(loss_fugw, loss_linear, atol=1e-5)

    loss_fugw = loss_quadratic_batch(
        a,
        a,
        C1,
        C2,
        T,
        M,
        alpha=1.0,
        unbalanced=reg_marginals,
        unbalanced_type=unbalanced_type,
        loss=loss,
        logits=False,
    )
    loss_gromov = loss_quadratic_batch(
        a,
        a,
        C1,
        C2,
        T,
        unbalanced=reg_marginals,
        unbalanced_type=unbalanced_type,
        loss=loss,
        logits=False,
    )
    np.testing.assert_allclose(loss_fugw, loss_gromov, atol=1e-5)


@pytest.mark.parametrize("divergence", ["kl", "l2"])
def test_div_between_product(divergence):
    batchsize = 2
    n = 4
    m = 3
    rng = np.random.RandomState(0)
    mu = rng.rand(batchsize, n)
    nu = rng.rand(batchsize, m)
    alpha = rng.rand(batchsize, n)
    beta = rng.rand(batchsize, m)

    res_batch = div_between_product_batch(
        mu, nu, alpha, beta, divergence=divergence, nx=None
    )
    res = np.array(
        [
            div_between_product(mu[i], nu[i], alpha[i], beta[i], divergence)
            for i in range(batchsize)
        ]
    )
    np.testing.assert_allclose(res_batch, res, atol=1e-5)


def test_loss_quadratic_samples_batch_deprecated():
    rng = np.random.RandomState(0)
    batchsize = 2
    n = 4
    d = 2
    C1 = rng.rand(batchsize, n, n, d)
    C2 = rng.rand(batchsize, n, n, d)
    a = np.ones((batchsize, n))
    T = rng.rand(batchsize, n, n)

    with pytest.warns(DeprecationWarning, match="loss_quadratic_batch"):
        loss_quadratic_samples_batch(a, a, C1, C2, T, loss="sqeuclidean")


def test_loss_quadratic_batch_log_balanced():
    batchsize = 2
    n = 4
    d = 2
    rng = np.random.RandomState(0)
    C1 = rng.rand(batchsize, n, n, d)
    C2 = rng.rand(batchsize, n, n, d)
    a = np.ones((batchsize, n))
    T = rng.rand(batchsize, n, n)

    value, log = loss_quadratic_batch(a, a, C1, C2, T, loss="sqeuclidean", log=True)
    expected = loss_quadratic_batch(a, a, C1, C2, T, loss="sqeuclidean")

    np.testing.assert_allclose(value, expected)
    np.testing.assert_allclose(log["value"], expected)
    np.testing.assert_allclose(log["value_quadratic"], expected)
    assert log["value_linear"] is None
    assert log["value_unbalanced"] is None


def test_loss_quadratic_batch_log_fugw():
    batchsize = 2
    n = 4
    d = 2
    rng = np.random.RandomState(0)
    C1 = rng.rand(batchsize, n, n, d)
    C2 = rng.rand(batchsize, n, n, d)
    M = rng.rand(batchsize, n, n)
    a = np.ones((batchsize, n))
    T = rng.rand(batchsize, n, n)
    alpha = rng.rand()
    unbalanced = rng.rand()

    value, log = loss_quadratic_batch(
        a,
        a,
        C1,
        C2,
        T,
        M=M,
        alpha=alpha,
        unbalanced=unbalanced,
        unbalanced_type="kl",
        loss="sqeuclidean",
        log=True,
    )

    expected = (1 - alpha) * log["value_linear"] + alpha * log["value_quadratic"]
    expected = expected + unbalanced * log["value_unbalanced"]
    np.testing.assert_allclose(value, expected)
    np.testing.assert_allclose(log["value"], expected)


@pytest.mark.skipif(not torch, reason="torch not installed")
@pytest.mark.parametrize("fused", [False, True])
def test_solve_gromov_batch_envelope_marginal_gradients_torch(fused):
    """Envelope marginal gradients match the exact 2x2 GW/FGW Oracle.

    Ca=[[0,1],[1,0]], Cb=[[0,2],[2,0]], a=(p,1-p), b=(q,1-q), p=3/5, q=3/10.
    The optimum is the boundary t*=0 with G*=18/25 (pure) and F*=51/100
    (fused, M=diag(3,3), alpha=1/2).  Sensitivities are dG/dp=-14/5,
    dG/dq=-8/5 and dF/dp=-29/10, dF/dq=-23/10 (verified with Wolfram).
    """
    Ca = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    Cb = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float64)
    p = torch.tensor([0.6], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    a = torch.cat([p, 1 - p]).unsqueeze(0)
    b = torch.cat([q, 1 - q]).unsqueeze(0)
    kw = {}
    if fused:
        M = torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
        kw = dict(M=M.unsqueeze(0), alpha=0.5)

    res = solve_gromov_batch(
        Ca.unsqueeze(0),
        Cb.unsqueeze(0),
        reg=0.1,
        a=a,
        b=b,
        loss="sqeuclidean",
        symmetric=True,
        grad="envelope",
        max_iter=200,
        tol=1e-12,
        max_iter_inner=200,
        tol_inner=1e-12,
        **kw,
    )
    inputs = [p, q]
    gp, gq = torch.autograd.grad(res.value.sum(), inputs, allow_unused=True)

    assert gp is not None and gq is not None
    exp_p, exp_q = (-2.9, -2.3) if fused else (-2.8, -1.6)
    np.testing.assert_allclose(gp.numpy(), [exp_p], atol=1e-8)
    np.testing.assert_allclose(gq.numpy(), [exp_q], atol=1e-8)
    exp_val = 0.51 if fused else 0.72
    np.testing.assert_allclose(res.value.item(), exp_val, atol=1e-8)
    np.testing.assert_allclose(
        a.detach().numpy(), res.plan.sum(2).detach().numpy(), atol=1e-10
    )
    assert res.log["converged"] is True
    assert gp.shape == p.shape and gp.dtype == p.dtype and gp.device == p.device


@pytest.mark.skipif(not torch, reason="torch not installed")
@pytest.mark.parametrize("batch_size", [2, 3, 8])
def test_solve_gromov_batch_envelope_batch_cotangent_torch(batch_size):
    """Upstream cotangent is applied per batch with no cross-batch leakage."""
    Ca = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    Cb = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float64)
    w_np = np.asarray([(-1) ** i * (i + 1) / 4 for i in range(batch_size)])
    p = torch.full((batch_size,), 0.6, dtype=torch.float64, requires_grad=True)
    q = torch.full((batch_size,), 0.3, dtype=torch.float64, requires_grad=True)
    a = torch.stack([p, 1 - p], dim=1)
    b = torch.stack([q, 1 - q], dim=1)
    res = solve_gromov_batch(
        Ca.unsqueeze(0).expand(batch_size, -1, -1),
        Cb.unsqueeze(0).expand(batch_size, -1, -1),
        reg=0.1,
        a=a,
        b=b,
        loss="sqeuclidean",
        symmetric=True,
        grad="envelope",
        max_iter=200,
        tol=1e-12,
        max_iter_inner=200,
        tol_inner=1e-12,
    )
    w = torch.tensor(w_np, dtype=torch.float64)
    gp, gq = torch.autograd.grad((res.value * w).sum(), (p, q), allow_unused=True)
    np.testing.assert_allclose(gp.numpy(), w_np * -2.8, atol=1e-9)
    np.testing.assert_allclose(gq.numpy(), w_np * -1.6, atol=1e-9)


@pytest.mark.skipif(not torch, reason="torch not installed")
def test_solve_gromov_batch_envelope_gate_keeps_cost_gradients_only():
    """When the marginal gate fails, cost gradients remain but p/q get none.

    Two gate failures are covered:
      * non-converged outer loop (converged=False);
      * converged=True but insufficient inner iterations, which violates
        marginal feasibility.
    In both cases ``torch.autograd.grad(..., allow_unused=True)`` must return
    a gradient for Ca and None for p and q.
    """
    for name, kwargs in [
        (
            "outer-not-converged",
            dict(max_iter=3, tol=1e-14, max_iter_inner=5, tol_inner=1e-14),
        ),
        (
            "marginal-infeasible",
            dict(max_iter=200, tol=1e-12, max_iter_inner=2, tol_inner=1e-12),
        ),
    ]:
        Ca = torch.tensor(
            [[0.26, 0.42], [0.42, 0.54]], dtype=torch.float64, requires_grad=True
        )
        Cb = torch.tensor(
            [[0.22, 0.39], [0.39, 0.89]], dtype=torch.float64, requires_grad=True
        )
        p = torch.tensor([0.6], dtype=torch.float64, requires_grad=True)
        q = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
        a = torch.cat([p, 1 - p]).unsqueeze(0)
        b = torch.cat([q, 1 - q]).unsqueeze(0)
        M = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64, requires_grad=True
        )
        res = solve_gromov_batch(
            Ca.unsqueeze(0),
            Cb.unsqueeze(0),
            reg=0.1,
            a=a,
            b=b,
            M=M.unsqueeze(0),
            alpha=0.5,
            loss="sqeuclidean",
            symmetric=True,
            grad="envelope",
            **kwargs,
        )
        gp, gq, gCa, gCb, gM = torch.autograd.grad(
            res.value.sum(), (p, q, Ca, Cb, M), allow_unused=True
        )
        assert gp is None and gq is None, f"{name}: unexpected marginal gradient"
        assert gCa is not None and gCb is not None and gM is not None
        assert torch.isfinite(gCa).all() and torch.isfinite(gCb).all()
        # Exposed flags must reflect the actual decision.
        trusted = (
            res.log["converged"]
            and res.log["marginal_feasible"]
            and res.log["potentials_stable"]
        )
        assert trusted is False, f"{name}: gate flags inconsistent"


@pytest.mark.skipif(not torch, reason="torch not installed")
def test_solve_gromov_batch_envelope_converged_flag_is_not_sufficient():
    """A loose outer tol can stop with the right value but stale potentials.

    This is a negative control for the converged-only gate: value matches the
    exact oracle, but the returned marginal gradient is wrong and must be
    rejected by the potential-stability check.
    """
    Ca = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    Cb = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([0.6], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    a = torch.cat([p, 1 - p]).unsqueeze(0)
    b = torch.cat([q, 1 - q]).unsqueeze(0)
    res = solve_gromov_batch(
        Ca.unsqueeze(0),
        Cb.unsqueeze(0),
        reg=0.1,
        a=a,
        b=b,
        loss="sqeuclidean",
        symmetric=True,
        grad="envelope",
        max_iter=200,
        tol=1e-2,
        max_iter_inner=200,
        tol_inner=1e-12,
    )
    np.testing.assert_allclose(res.value.item(), 0.72, atol=1e-8)
    assert res.log["converged"] is True
    assert res.log["potentials_stable"] is False
    # A valid cost graph guarantees autograd.grad returns rather than raises,
    # so allow_unused=True gives a precise assertion: cost gradients present,
    # marginal gradients exactly None.  No blanket RuntimeError catch is used.
    gp, gq, gCa, gCb = torch.autograd.grad(
        res.value.sum(), (p, q, Ca, Cb), allow_unused=True
    )
    assert gp is None and gq is None
    assert gCa is not None and gCb is not None
    assert torch.isfinite(gCa).all() and torch.isfinite(gCb).all()
    # value_linear = <M, T> keeps its path to M in envelope mode (it does not
    # depend on Ca/Cb, which only enter the quadratic term).
    M = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    res_m = solve_gromov_batch(
        Ca.unsqueeze(0),
        Cb.unsqueeze(0),
        reg=0.1,
        a=a,
        b=b,
        M=M.unsqueeze(0),
        alpha=0.0,
        loss="sqeuclidean",
        symmetric=True,
        grad="envelope",
        max_iter=200,
        tol=1e-2,
        max_iter_inner=200,
        tol_inner=1e-12,
    )
    gl = torch.autograd.grad(res_m.value_linear.sum(), M, allow_unused=True)[0]
    assert gl is not None and torch.isfinite(gl).all()
    # value_linear = (1-alpha)<M, T> with T detached, so d(value_linear)/dM
    # must equal (1-alpha) * T.  alpha=0 here gives exactly T.
    np.testing.assert_allclose(
        gl.detach().numpy().reshape(M.shape),
        res_m.plan.detach().numpy().reshape(M.shape),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        res_m.value_linear.item(),
        float((M.detach().unsqueeze(0) * res_m.plan.detach()).sum()),
        atol=1e-10,
    )


@pytest.mark.skipif(not torch, reason="torch not installed")
def test_solve_gromov_batch_envelope_centered_sensitivities_gauge_invariant():
    """The registered marginal sensitivities are invariant to u+c, v-c."""
    from ot.batch._quadratic import _centered_marginal_sensitivities

    u = torch.tensor([[2.46796, 30.46796], [1.0, 5.0]], dtype=torch.float64)
    v = torch.tensor([[-10.46796, 5.53204], [3.0, -2.0]], dtype=torch.float64)
    nx = get_backend(u, v)
    ga, gb = _centered_marginal_sensitivities(u, v, 0.1, nx)
    ga_c, gb_c = _centered_marginal_sensitivities(u + 137.0, v - 137.0, 0.1, nx)
    torch.testing.assert_close(ga, ga_c, atol=1e-12, rtol=0)
    torch.testing.assert_close(gb, gb_c, atol=1e-12, rtol=0)


@pytest.mark.skipif(not torch, reason="torch not installed")
def test_solve_gromov_batch_envelope_gauge_shift_keeps_gate_and_gradient(monkeypatch):
    """A large constant gauge shift must not change the gate or the gradient.

    The inner Bregman projection is wrapped to add +250/-250 to its returned
    potentials.  Because the gate compares centred sensitivities, all log
    flags and the registered marginal gradients must remain unchanged.
    """
    import ot.batch._quadratic as qm

    original = qm.bregman_log_projection_batch

    def shifted(*args, **kwargs):
        out = original(*args, **kwargs)
        u, v = out["potentials"]
        return dict(out, potentials=(u + 250.0, v - 250.0))

    def solve(tol):
        Ca = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )[None]
        Cb = torch.tensor(
            [[0.0, 2.0], [2.0, 0.0]], dtype=torch.float64, requires_grad=True
        )[None]
        p = torch.tensor([0.6], dtype=torch.float64, requires_grad=True)
        q = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
        r = solve_gromov_batch(
            Ca,
            Cb,
            reg=0.1,
            a=torch.cat([p, 1 - p])[None],
            b=torch.cat([q, 1 - q])[None],
            loss="sqeuclidean",
            symmetric=True,
            grad="envelope",
            max_iter=200,
            tol=tol,
            max_iter_inner=200,
            tol_inner=1e-12,
        )
        # Keep a valid cost graph so autograd returns rather than raises;
        # marginal gradients are then asserted precisely as None/absent.
        gp, gq, gCa, gCb = torch.autograd.grad(
            r.value.sum(), (p, q, Ca, Cb), allow_unused=True
        )
        assert gCa is not None and gCb is not None
        grads = (
            gp.item() if gp is not None else None,
            gq.item() if gq is not None else None,
        )
        return r.log, grads

    base = solve(1e-12)
    monkeypatch.setattr(qm, "bregman_log_projection_batch", shifted)
    shifted_result = solve(1e-12)
    assert base[0] == shifted_result[0]
    np.testing.assert_allclose(base[1], shifted_result[1], atol=1e-9)
    # The loose-tol counterexample must still be rejected under a gauge shift.
    assert base[0]["potentials_stable"] is True
    assert solve(1e-2)[0]["potentials_stable"] is False
    monkeypatch.undo()


def test_solve_gromov_batch_extra_projection_scope_numpy():
    """Non-Torch backends must not pay for the extra stability projection.

    This test runs unconditionally (NumPy is always available).  It asserts
    that envelope/detach/autodiff all perform the original number of inner
    projections, i.e. no extra projection is inserted outside the Torch
    envelope marginal path.
    """
    import ot.batch._quadratic as qm

    original = qm.bregman_log_projection_batch
    counter = {"n": 0}

    def counted(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    Ca = np.array([[[0.0, 1.0], [1.0, 0.0]]])
    Cb = np.array([[[0.0, 2.0], [2.0, 0.0]]])
    a = np.array([[0.6, 0.4]])
    b = np.array([[0.3, 0.7]])

    counts = {}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(qm, "bregman_log_projection_batch", counted)
    try:
        for grad in ["envelope", "detach", "autodiff"]:
            counter["n"] = 0
            solve_gromov_batch(
                Ca,
                Cb,
                reg=0.1,
                a=a,
                b=b,
                loss="sqeuclidean",
                symmetric=True,
                grad=grad,
                max_iter=200,
                tol=1e-12,
                max_iter_inner=200,
                tol_inner=1e-12,
            )
            counts[grad] = counter["n"]
    finally:
        monkeypatch.undo()
    # No extra projection for any NumPy gradient mode.
    assert counts["envelope"] == counts["detach"] == counts["autodiff"]
    assert counts["envelope"] > 0


@pytest.mark.skipif(not torch, reason="torch not installed")
def test_solve_gromov_batch_extra_projection_scope_torch():
    """Only the Torch envelope marginal path pays for one extra projection."""
    import ot.batch._quadratic as qm

    original = qm.bregman_log_projection_batch
    counter = {"n": 0}

    def counted(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    Ca = np.array([[[0.0, 1.0], [1.0, 0.0]]])
    Cb = np.array([[[0.0, 2.0], [2.0, 0.0]]])
    a = np.array([[0.6, 0.4]])
    b = np.array([[0.3, 0.7]])

    def run(use_torch, grad):
        counter["n"] = 0
        if use_torch:
            c_a, c_b, aa, bb = (
                torch.tensor(Ca),
                torch.tensor(Cb),
                torch.tensor(a),
                torch.tensor(b),
            )
        else:
            c_a, c_b, aa, bb = Ca, Cb, a, b
        solve_gromov_batch(
            c_a,
            c_b,
            reg=0.1,
            a=aa,
            b=bb,
            loss="sqeuclidean",
            symmetric=True,
            grad=grad,
            max_iter=200,
            tol=1e-12,
            max_iter_inner=200,
            tol_inner=1e-12,
        )
        return counter["n"]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(qm, "bregman_log_projection_batch", counted)
    try:
        base = run(False, "envelope")
        for use_torch, grad in [
            (False, "detach"),
            (False, "autodiff"),
            (True, "detach"),
            (True, "autodiff"),
        ]:
            assert run(use_torch, grad) == base
        # Torch envelope performs exactly one additional projection.
        assert run(True, "envelope") == base + 1
    finally:
        monkeypatch.undo()


@pytest.mark.skipif(not torch, reason="torch not installed")
def test_solve_gromov_batch_envelope_mixed_batch_rejects_all():
    """One failing sample suppresses marginal gradients for the whole batch."""
    Ca = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)[None]
    Cb = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float64)[None]

    def solve(ps, qs):
        ca = Ca.expand(len(ps), -1, -1).clone().requires_grad_(True)
        cb = Cb.expand(len(ps), -1, -1).clone().requires_grad_(True)
        p = torch.tensor(ps, dtype=torch.float64, requires_grad=True)
        q = torch.tensor(qs, dtype=torch.float64, requires_grad=True)
        r = solve_gromov_batch(
            ca,
            cb,
            reg=0.1,
            a=torch.stack([p, 1 - p], 1),
            b=torch.stack([q, 1 - q], 1),
            loss="sqeuclidean",
            symmetric=True,
            grad="envelope",
            max_iter=200,
            tol=1e-3,
            max_iter_inner=200,
            tol_inner=1e-12,
        )
        # Valid cost graph: autograd must return, not raise. Assert that the
        # cost gradients exist while the marginal gradients are exactly None.
        gp, gq, gca, gcb = torch.autograd.grad(
            r.value.sum(), (p, q, ca, cb), allow_unused=True
        )
        assert gca is not None and gcb is not None
        has = gp is not None
        if not has:
            assert gq is None
        return r.log, has

    single_ok, has_ok = solve([0.6], [0.3])
    single_bad, has_bad = solve([0.999], [0.999])
    mixed, has_mixed = solve([0.6, 0.999], [0.3, 0.999])
    assert has_ok is True and single_ok["potentials_stable"] is True
    assert has_bad is False and single_bad["potentials_stable"] is False
    # Conservative whole-batch rejection: no zero-gradient substitute.
    assert has_mixed is False and mixed["potentials_stable"] is False
