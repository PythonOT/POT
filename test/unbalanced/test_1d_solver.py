"""Tests for module 1D Unbalanced OT"""

# Author: Clément Bonet <clement.bonet.mapp@polytechnique.edu>
#
# License: MIT License

import itertools
import numpy as np
import ot
import pytest
import cvxpy as cp


@pytest.skip_backend("tf")
def test_uot_1d(nx):
    n_samples = 20  # nb samples

    rng = np.random.RandomState(42)
    xs = rng.randn(n_samples, 1)
    xt = rng.randn(n_samples, 1)

    a_np = ot.utils.unif(n_samples)
    b_np = ot.utils.unif(n_samples)

    reg_m = 1.0

    M = ot.dist(xs, xt)
    # M = M / M.max()
    a, b, M = nx.from_numpy(a_np, b_np, M)
    xs, xt = nx.from_numpy(xs, xt)

    loss_mm = ot.unbalanced.mm_unbalanced2(a, b, M, reg_m, div="kl")
    G = ot.unbalanced.mm_unbalanced(a, b, M, reg_m, div="kl")

    P = cp.Variable((n_samples, n_samples))

    u = np.ones((n_samples, 1))
    v = np.ones((n_samples, 1))
    q = cp.sum(cp.kl_div(cp.matmul(P, v), a[:, None]))
    r = cp.sum(cp.kl_div(cp.matmul(P.T, u), b[:, None]))

    constr = [0 <= P]
    objective = cp.Minimize(cp.sum(cp.multiply(P, M)) + reg_m * q + reg_m * r)

    prob = cp.Problem(objective, constr)
    result = prob.solve()
    G_cvxpy = P.value
    loss_cvxpy = np.sum(G_cvxpy * M)

    print("?", nx.__name__)
    print("??", loss_mm.item(), G.sum(), loss_cvxpy, G_cvxpy.sum())

    if nx.__name__ != "jax":
        f, g, loss_1d = ot.unbalanced.uot_1d(xs, xt, reg_m, mode="icdf", p=2)
        print("!! ", loss_1d.item())
        np.testing.assert_allclose(loss_1d, loss_mm)

    if nx.__name__ in ["jax", "torch"]:
        f, g, loss_1d = ot.unbalanced.uot_1d(xs, xt, reg_m, mode="backprop", p=2)

        print("???", loss_1d.item(), f.sum())

        np.testing.assert_allclose(loss_1d, loss_mm)
