"""Tests for module sliced Unbalanced OT"""

# Author: Clément Bonet <clement.bonet.mapp@polytechnique.edu>
#
# License: MIT License

import itertools
import numpy as np
import ot
import pytest


# Classical sliced tests
# Check inf <-> SW
# Checks regs, semi-unbalanced etc


def test_sliced_uot_same_dist(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    x, u = nx.from_numpy(x, u)

    if nx.__name__ in ["torch", "jax"]:
        res = ot.sliced_unbalanced_ot(x, x, 1, u, u, 10, seed=42)
        np.testing.assert_almost_equal(res, 0.0)

        _, _, res = ot.unbalanced_sliced_ot(x, x, 1, u, u, 10, seed=42)
        np.testing.assert_almost_equal(res, 0.0)


def test_sliced_uot_bad_shapes(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(n, 4)
    u = ot.utils.unif(n)

    if nx.__name__ in ["torch", "jax"]:
        x, y, u = nx.from_numpy(x, y, u)

        with pytest.raises(ValueError):
            _ = ot.sliced_unbalanced_ot(x, y, 1, u, u, 10, seed=42)

        with pytest.raises(ValueError):
            _ = ot.unbalanced_sliced_ot(x, y, 1, u, u, 10, seed=42)


def test_sliced_uot_log(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 4)
    y = rng.randn(n, 4)
    u = ot.utils.unif(n)

    if nx.__name__ in ["torch", "jax"]:
        x, y, u = nx.from_numpy(x, y, u)

        res, log = ot.sliced_unbalanced_ot(x, y, 1, u, u, 10, p=1, seed=42, log=True)
        assert len(log) == 4
        projections = log["projections"]
        projected_uots = log["projected_uots"]
        a_reweighted = log["a_reweighted"]
        b_reweighted = log["b_reweighted"]

        assert projections.shape[1] == len(projected_uots) == 10

        for emd in projected_uots:
            assert emd > 0

        assert res > 0
        assert a_reweighted.shape == b_reweighted.shape == (n, 10)


def test_unbalanced_sot_log(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 4)
    y = rng.randn(n, 4)
    u = ot.utils.unif(n)

    if nx.__name__ in ["torch", "jax"]:
        x, y, u = nx.from_numpy(x, y, u)

        f, g, res, log = ot.unbalanced_sliced_ot(
            x, y, 1, u, u, 10, p=1, seed=42, log=True
        )
        assert len(log) == 4

        projections = log["projections"]
        sot_loss = log["sot_loss"]
        ot_loss = log["1d_losses"]
        full_mass = log["full_mass"]

        assert projections.shape[1] == 10
        assert res > 0

        assert f.shape == g.shape == u.shape
        np.testing.assert_almost_equal(f.sum(), g.sum())
        np.testing.assert_equal(sot_loss, nx.mean(ot_loss * full_mass))


def test_1d_sliced_equals_uot(nx):
    n = 100
    m = 120
    rng = np.random.RandomState(42)

    x = rng.randn(n, 1)
    y = rng.randn(m, 1)

    a = rng.uniform(0, 1, n) / 10  # unbalanced
    u = ot.utils.unif(m)

    reg_m = 1

    if nx.__name__ in ["torch", "jax"]:
        x, y, a, u = nx.from_numpy(x, y, a, u)

        res, log = ot.sliced_unbalanced_ot(
            x, y, reg_m, a, u, 10, seed=42, p=2, log=True
        )
        _, _, expected = ot.uot_1d(
            x.squeeze(), y.squeeze(), reg_m, a, u, returnCost="total", p=2
        )
        np.testing.assert_almost_equal(res, expected)

        f, g, res, log = ot.unbalanced_sliced_ot(
            x, y, reg_m, a, u, 10, seed=42, p=2, log=True
        )
        np.testing.assert_almost_equal(res, expected)
