"""Tests for module 1D Unbalanced OT"""

# Author:
#
# License: MIT License

import itertools
import numpy as np
import ot
import pytest


def test_uot_1d(nx):
    pass

    n_samples = 20  # nb samples

    rng = np.random.RandomState(42)
    xs = rng.randn(n_samples, 1)
    xt = rng.randn(n_samples, 1)

    a_np = ot.utils.unif(n_samples)
    b_np = ot.utils.unif(n_samples)

    reg_m = 1.0

    M = ot.dist(xs, xt)
    M = M / M.max()
    a, b, M = nx.from_numpy(a_np, b_np, M)

    loss_mm = ot.unbalanced.mm_unbalanced2(a, b, M, reg_m, div="kl")

    print("??", loss_mm)

    if nx.__name__ in ["jax", "torch"]:
        f, g, loss_1d = ot.unbalanced.uot_1d(xs, xt, reg_m, mode="backprop")

        print("???", loss_1d[0])

        np.testing.assert_allclose(loss_1d, loss_mm)
