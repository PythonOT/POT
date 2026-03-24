"""Tests for module ot.bsp"""

# Author: Baptiste Genest <baptistegenest@gmail.com>
#
# License: MIT License

import ot
import ot.bsp
import numpy as np


def test_bsp_ot_exact_identity():
    # test that bsp-ot is exact for similarity transform
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    # create B as a similarity transform of A, scale by 2 and translate x coord by 1
    B = 2 * A + np.array([1, 0])

    _, perm, _ = ot.bsp.bsp_solve(A, B, 1)

    # check that the permutation is the identity
    np.testing.assert_allclose(perm, np.arange(n))


def test_bsp_ot_identity_null_cost():
    # test that the cost returned by bsp-ot is zero for the identity transform
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)

    cost, _, _ = ot.bsp.bsp_solve(A, A, 1)

    # check that the cost is zero
    np.testing.assert_allclose(cost, 0)


def test_bsp_ot_bijective():
    # test that the output of bsp-ot is indeed a bijection
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    # create B as a similarity transform of A, scale by 2 and translate x coord by 1
    B = 2 * A + np.array([1, 0])

    _, perm, _ = ot.bsp.bsp_solve(A, B, 1)

    # check that the permutation is a bijection
    assert len(set(perm)) == n, "Permutation is not a bijection"

    # check that the permutation is a valid permutation
    assert set(perm) == set(range(n)), "Permutation is not valid"


def test_bsp_ot_plan_merge_decrease():
    # test that merging two plans gives a smaller cost than the plans separately
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)

    cost, plan, plans = ot.bsp.bsp_solve(A, B, 2)

    # evaluate mean squared cost lambda
    def cost_lambda(A, B, T):
        return np.mean(np.sum((A - B[T]) ** 2, axis=1))

    cost_plan_1 = cost_lambda(A, B, plans[0])
    cost_plan_2 = cost_lambda(A, B, plans[1])

    # check that the cost of the merged plan is smaller than the cost of the separate plans
    assert (
        cost <= cost_plan_1
    ), "Cost of merged plan is not smaller than one of the input plans"
    assert (
        cost <= cost_plan_2
    ), "Cost of merged plan is not smaller than one of the input plans"


def test_bsp_ot_relative_error():
    # test that the cost returned by bsp-ot is close to the cost of the optimal transport plan
    n = 100
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)

    cost, perm, _ = ot.bsp.bsp_solve(A, B, 1000)

    w = ot.utils.unif(n)

    # solve exact with ot solver
    M = ot.dist(A, B, metric="sqeuclidean")
    G = ot.emd(w, w, M)

    cost_ot = np.sum(G * M)

    # check that the relative error is small
    relative_error = np.abs(cost - cost_ot) / (cost_ot + 1e-8)
    assert relative_error < 0.1, "Relative error is too large: {}".format(
        relative_error
    )


def test_bsp_ot_torch_backend():
    import torch

    # test that bsp-ot works with torch tensors
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)

    A_torch = torch.from_numpy(A)
    B_torch = torch.from_numpy(B)
    A_torch.requires_grad_()
    B_torch.requires_grad_()

    cost, perm, _ = ot.bsp.bsp_solve(A_torch, B_torch, 1)

    # compute gradients w.r.t. A and B
    cost.backward()

    # advected points (should match B)
    A_new = A - A_torch.grad.detach().numpy() * n / 2

    # compute new cost with advected points
    cost_new, _, _ = ot.bsp.bsp_solve(A_new, B, 1)

    # cost should be zero
    np.testing.assert_allclose(cost_new, 0, atol=1e-5)


test_bsp_ot_exact_identity()
test_bsp_ot_bijective()
test_bsp_ot_identity_null_cost()
test_bsp_ot_plan_merge_decrease()
test_bsp_ot_relative_error()
test_bsp_ot_torch_backend()
