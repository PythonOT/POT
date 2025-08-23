"""Tests for module bregman on OT with bregman projections"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Kilian Fatras <kilian.fatras@irisa.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#         Eduardo Fernandes Montesuma <eduardo.fernandes-montesuma@universite-paris-saclay.fr>
#
# License: MIT License

import numpy as np
from ot.batch import quadratic_solver_batch
from ot import solve_gromov
from ot.batch._linear import cost_matrix_l2_batch


def test_quadratic_solver_batch():
    """Check that quadratic_solver_batch gives the same results as solve for each instance in the batch."""
    b = 2
    n = 8
    d = 2
    epsilon = 0.01
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

    M = cost_matrix_l2_batch(X1, X2)

    res = quadratic_solver_batch(
        alpha=alpha,
        epsilon=epsilon,
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


test_quadratic_solver_batch()
