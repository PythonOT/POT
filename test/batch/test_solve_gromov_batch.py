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
    loss_quadratic_samples_batch,
)
from ot import solve_gromov
from ot.batch._linear import dist_batch
import pytest
from itertools import product
from ot.backend import torch
from ot.batch._quadratic import (
    tensor_batch,
    div_between_product_batch,
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
        C1=C1,
        C2=C2,
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

    res = solve_gromov_batch(C1=C, C2=C, a=a, b=a, loss=loss, logits=logits)

    loss1 = res.value_quad
    loss2 = loss_quadratic_samples_batch(
        a=a, b=a, C1=C, C2=C, T=res.plan, loss=loss, logits=logits
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
        C1=C, C2=C, a=None, b=None, loss="sqeuclidean", logits=False, grad=grad
    )
    loss = res.value.sum()
    loss_plan = res.plan.sum()
    if grad == "detach":
        assert loss.grad == None
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
    solve_gromov_batch(C1=C, C2=C, a=None, b=None, loss="sqeuclidean", logits=False)


@pytest.mark.parametrize("unbalanced_type", ["kl", "l2"])
@pytest.mark.parametrize("loss", ["sqeuclidean", "kl"])
def test_fugw_loss(unbalanced_type, loss):
    """Check that loss_fugw_batch and loss_fugw_samples_batch run without error."""
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

    loss_fugw = loss_quadratic_samples_batch(
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
        logits=False,
    )
    loss_fugw_unbalanced_only = loss_quadratic_samples_batch(
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
        logits=False,
    )
    loss_fugw_no_alpha = loss_quadratic_samples_batch(
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
        logits=False,
    )
    loss_fugw_no_unbalanced = loss_quadratic_samples_batch(
        a,
        a,
        C1,
        C2,
        T,
        M=M,
        alpha=alpha,
        loss=loss,
        logits=False,
    )
    assert np.isfinite(loss_fugw_unbalanced_only).all()
    assert np.isfinite(loss_fugw).all()
    assert np.isfinite(loss_fugw_no_alpha).all()
    assert np.isfinite(loss_fugw_no_unbalanced).all()

    # check that alpha and reg_marginals can be passed as lists or arrays of shape (batchsize,)
    alpha = rng.rand(batchsize)
    reg_marginals = rng.rand(batchsize)
    alpha_list = alpha.tolist()
    reg_marginals_list = reg_marginals.tolist()

    loss_fugw = loss_quadratic_samples_batch(
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
        logits=False,
    )
    loss_fugw_list = loss_quadratic_samples_batch(
        a,
        a,
        C1,
        C2,
        T,
        M,
        alpha=alpha_list,
        unbalanced=reg_marginals_list,
        unbalanced_type=unbalanced_type,
        loss=loss,
        logits=False,
    )

    assert np.isfinite(loss_fugw).all()
    assert np.isfinite(loss_fugw_list).all()
    np.testing.assert_allclose(loss_fugw, loss_fugw_list)

    # check that invalid loss raise an error
    with pytest.raises(ValueError):
        loss_fugw = loss_quadratic_samples_batch(
            a,
            a,
            C1,
            C2,
            T,
            M,
            alpha=alpha,
            unbalanced=reg_marginals,
            unbalanced_type=unbalanced_type,
            loss="test",
            logits=False,
        )

    # check that invalid loss raise an error
    with pytest.raises(ValueError):
        loss_fugw = loss_quadratic_samples_batch(
            a,
            a,
            C1,
            C2,
            T,
            M,
            alpha=alpha,
            unbalanced=reg_marginals,
            unbalanced_type="test",
            loss=loss,
            logits=False,
        )

    # check that invalid alpha shape raise an error
    alpha = rng.rand(batchsize + 1)
    with pytest.raises(ValueError):
        loss_quadratic_samples_batch(
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
            logits=False,
        )

    # check that invalid rho shape raise an error
    alpha = rng.rand(batchsize)
    reg_marginals = rng.rand(batchsize + 1)
    with pytest.raises(ValueError):
        loss_quadratic_samples_batch(
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

    loss_fugw = loss_quadratic_samples_batch(
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

    loss_fugw = loss_quadratic_samples_batch(
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
    loss_gromov = loss_quadratic_samples_batch(
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
