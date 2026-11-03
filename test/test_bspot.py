"""Tests for module ot.bsp"""

# Author: Baptiste Genest <baptistegenest@gmail.com>
#
# License: MIT License

import warnings

import ot
import ot.bsp
import numpy as np
import pytest
from ot.backend import torch


def test_bsp_ot_exact_identity():
    # test that bsp-ot is exact for similarity transform
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    # create B as a similarity transform of A, scale by 2 and translate x coord by 1
    B = 2 * A + np.array([1, 0])

    _, perm, _ = ot.bsp.compute_bspot_bijection(A, B, 1)

    # check that the permutation is the identity
    np.testing.assert_allclose(perm, np.arange(n))


def test_bsp_ot_identity_null_cost():
    # test that the cost returned by bsp-ot is zero for the identity transform
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)

    cost, _, _ = ot.bsp.compute_bspot_bijection(A, A, 1)

    # check that the cost is zero
    np.testing.assert_allclose(cost, 0)


def test_bsp_ot_bijective():
    # test that the output of bsp-ot is indeed a bijection
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    # create B as a similarity transform of A, scale by 2 and translate x coord by 1
    B = 2 * A + np.array([1, 0])

    _, perm, _ = ot.bsp.compute_bspot_bijection(A, B, 1)

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

    cost, plan, plans = ot.bsp.compute_bspot_bijection(A, B, 2)

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

    cost, perm, _ = ot.bsp.compute_bspot_bijection(A, B, 1000)

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


@pytest.mark.skipif(not torch, reason="PyTorch not installed")
def test_bsp_ot_torch_backend():
    # test that bsp-ot works with torch tensors
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)

    A_torch = torch.from_numpy(A)
    B_torch = torch.from_numpy(B)
    A_torch.requires_grad_()
    B_torch.requires_grad_()

    cost, perm, _ = ot.bsp.compute_bspot_bijection(A_torch, B_torch, 1)

    # compute gradients w.r.t. A and B
    cost.backward()

    # advected points (should match B)
    A_new = A - A_torch.grad.detach().numpy() * n / 2

    # compute new cost with advected points
    cost_new, _, _ = ot.bsp.compute_bspot_bijection(A_new, B, 1)

    # cost should be zero
    np.testing.assert_allclose(cost_new, 0, atol=1e-5)


def _mean_squared_cost(A, B, perm):
    # cost of ``perm`` computed here, independently of what the solver reports
    return float(np.mean(np.sum((A - B[np.asarray(perm)]) ** 2, axis=1)))


def test_bsp_merge_fewer_plans_than_samples():
    # perms is a stack of (n_plans, n_samples) permutations, so the merge has n_plans of
    # them to read. With one plan over 32 samples the old code read one plan per sample
    # and walked off the end of the array (issue #910).
    n = 32
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)
    perms = np.arange(n, dtype=np.int32).reshape(1, n)

    with warnings.catch_warnings():
        # merge_bijections warns when its reported cost does not match the permutation
        warnings.simplefilter("error")
        cost, perm = ot.bsp.merge_bijections(A, B, perms, 2)

    perm = np.asarray(perm)
    np.testing.assert_array_equal(np.sort(perm), np.arange(n))
    np.testing.assert_allclose(cost, _mean_squared_cost(A, B, perm), rtol=1e-9)
    assert cost <= _mean_squared_cost(A, B, np.arange(n)) + 1e-12


def test_bsp_merge_uses_every_supplied_plan():
    # The plans past the n_samples-th one must reach the merge. Both clouds are the same,
    # so the identity is the only plan with zero cost and the only one that can produce
    # it: if it is dropped the merged cost stays on the mediocre plan.
    n = 8
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = A.copy()

    mediocre = np.roll(np.arange(n), 1).astype(np.int32)
    without = np.tile(mediocre, (n, 1))
    with_last = np.vstack([without, np.arange(n)]).astype(np.int32)

    cost_without, _ = ot.bsp.merge_bijections(A, B, without, 2)
    cost_with, perm_with = ot.bsp.merge_bijections(A, B, with_last, 2)

    assert cost_with < cost_without, "the plan past n_samples was ignored"
    np.testing.assert_array_equal(np.asarray(perm_with), np.arange(n))
    np.testing.assert_allclose(cost_with, 0.0, atol=1e-12)


def test_bsp_merge_rejects_plans_of_wrong_length():
    # A stack whose rows are not one entry per point cannot be merged; it must fail here
    # instead of being read as a number of plans by C++.
    n = 8
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)

    with pytest.raises(ValueError, match="n_samples is the number of points"):
        ot.bsp.merge_bijections(A, B, np.tile(np.roll(np.arange(7), 1), (3, 1)), 2)

    with pytest.raises(ValueError, match="2D"):
        ot.bsp.merge_bijections(A, B, np.arange(n, dtype=np.int32), 2)

    with pytest.raises(ValueError, match="2D"):
        ot.bsp.merge_bijections(A, B, np.zeros((n, n, 1), dtype=np.int32), 2)


def test_bsp_merge_on_a_standard_stack():
    # The ordinary case must keep its documented behaviour: a bijection no worse than any
    # plan it merged, and a cost that is the cost of that bijection.
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)
    perms = np.stack([np.roll(np.arange(n), s) for s in (1, 7, 23, 41)]).astype(np.int32)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cost, perm = ot.bsp.merge_bijections(A, B, perms, 2)

    perm = np.asarray(perm)
    np.testing.assert_array_equal(np.sort(perm), np.arange(n))
    np.testing.assert_allclose(cost, _mean_squared_cost(A, B, perm), rtol=1e-9)
    for plan in perms:
        assert cost <= _mean_squared_cost(A, B, plan) + 1e-12


def test_bsp_merge_converts_dtypes_and_refuses_views():
    # np.asarray(perms, dtype=np.int32) already accepted these, and the fix must not change
    # that; a non-contiguous view was refused before the fix too, rather than read loosely.
    n = 8
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)
    plans = np.tile(np.roll(np.arange(n), 1), (4, 1))

    ref_cost, ref_perm = ot.bsp.merge_bijections(A, B, plans.astype(np.int32), 2)
    for other in (plans.astype(np.int64), plans.astype(np.float64), plans.tolist()):
        cost, perm = ot.bsp.merge_bijections(A, B, other, 2)
        np.testing.assert_allclose(cost, ref_cost, rtol=0, atol=0)
        np.testing.assert_array_equal(np.asarray(perm), np.asarray(ref_perm))

    view = np.asfortranarray(plans.astype(np.int32))
    assert not view.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        ot.bsp.merge_bijections(A, B, view, 2)
