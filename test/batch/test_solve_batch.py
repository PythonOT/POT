"""Tests for module bregman on OT with bregman projections"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Kilian Fatras <kilian.fatras@irisa.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#         Eduardo Fernandes Montesuma <eduardo.fernandes-montesuma@universite-paris-saclay.fr>
#
# License: MIT License

import numpy as np
from ot.batch import solve_batch
from ot import solve


def test_solve_batch():
    """Check that solve_batch gives the same results as solve for each instance in the batch."""
    batchsize = 4
    n = 16
    rng = np.random.RandomState(0)

    M = rng.rand(batchsize, n, n)

    epsilon = 0.1
    max_iter = 10000
    tol = 1e-5

    res = solve_batch(
        M,
        a=None,
        b=None,
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
        log_dual=True,
        grad="detach",
    )
    plan_batch = res.plan
    values_batch = res.value_linear

    for i in range(batchsize):
        M_i = M[i]
        res_i = solve(M_i, a=None, b=None, reg=epsilon, max_iter=max_iter, tol=tol)
        plan_i = res_i.plan
        value_i = res_i.value_linear
        np.testing.assert_allclose(plan_i, plan_batch[i], atol=1e-05)
        np.testing.assert_allclose(value_i, values_batch[i], atol=1e-4)


test_solve_batch()
