"""Tests for module 1D Unbalanced OT"""

# Author: Clément Bonet <clement.bonet.mapp@polytechnique.edu>
#
# License: MIT License

import itertools
import numpy as np
import ot
import pytest


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

    print("?", nx.__name__)

    if nx.__name__ != "jax":
        f, g, loss_1d = ot.unbalanced.uot_1d(xs, xt, reg_m, mode="icdf", numItermax=100)
        print("!! ", loss_1d.item())
        np.testing.assert_allclose(loss_1d, loss_mm)

    if nx.__name__ in ["jax", "torch"]:
        print("??", loss_mm.item())

        f, g, loss_1d = ot.unbalanced.uot_1d(
            xs, xt, reg_m, mode="backprop", numItermax=100
        )

        print("???", loss_1d.item())

        np.testing.assert_allclose(loss_1d, loss_mm)
