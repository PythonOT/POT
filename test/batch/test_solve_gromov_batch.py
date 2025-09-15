"""Tests for module bregman on OT with bregman projections"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Kilian Fatras <kilian.fatras@irisa.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#         Eduardo Fernandes Montesuma <eduardo.fernandes-montesuma@universite-paris-saclay.fr>
#
# License: MIT License

import numpy as np
from ot.batch import solve_gromov_batch, loss_quadratic_samples_batch
from ot import solve_gromov
from ot.batch._linear import dist_batch
import pytest
from itertools import product
from ot.backend import torch


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


# @pytest.mark.skipif(not torch, reason="torch not installed")
# @pytest.mark.parametrize("grad", ["detach", "envelope", "autodiff", "last_step"])
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
    elif grad in ["autodiff", "last_step"]:
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
