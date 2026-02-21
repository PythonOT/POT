"""Tests for module sliced Unbalanced OT"""

# Author: Clément Bonet <clement.bonet.mapp@polytechnique.edu>
#
# License: MIT License

import itertools
import numpy as np
import ot
import pytest


@pytest.skip_backend("numpy")
@pytest.skip_backend("tf")
@pytest.skip_backend("cupy")
def test_sliced_uot_same_dist(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    x, u = nx.from_numpy(x, u)

    res = ot.sliced_unbalanced_ot(x, x, 1, u, u, 10, seed=42)
    np.testing.assert_almost_equal(res, 0.0)

    _, _, res = ot.unbalanced_sliced_ot(x, x, 1, u, u, 10, seed=42)
    np.testing.assert_almost_equal(res, 0.0)


@pytest.skip_backend("numpy")
@pytest.skip_backend("tf")
@pytest.skip_backend("cupy")
def test_sliced_uot_bad_shapes(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(n, 4)
    u = ot.utils.unif(n)

    x, y, u = nx.from_numpy(x, y, u)

    with pytest.raises(ValueError):
        _ = ot.sliced_unbalanced_ot(x, y, 1, u, u, 10, seed=42)

    with pytest.raises(ValueError):
        _ = ot.unbalanced_sliced_ot(x, y, 1, u, u, 10, seed=42)


@pytest.skip_backend("numpy")
@pytest.skip_backend("tf")
@pytest.skip_backend("cupy")
def test_sliced_uot_log(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 4)
    y = rng.randn(n, 4)
    u = ot.utils.unif(n)

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


@pytest.skip_backend("numpy")
@pytest.skip_backend("tf")
@pytest.skip_backend("cupy")
def test_usot_log(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 4)
    y = rng.randn(n, 4)
    u = ot.utils.unif(n)

    x, y, u = nx.from_numpy(x, y, u)

    f, g, res, log = ot.unbalanced_sliced_ot(x, y, 1, u, u, 10, p=1, seed=42, log=True)
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


@pytest.skip_backend("numpy")
@pytest.skip_backend("tf")
@pytest.skip_backend("cupy")
def test_1d_sliced_equals_uot(nx):
    n = 100
    m = 120
    rng = np.random.RandomState(42)

    x = rng.randn(n, 1)
    y = rng.randn(m, 1)

    a = rng.uniform(0, 1, n) / 10  # unbalanced
    u = ot.utils.unif(m)

    reg_m = 1

    x, y, a, u = nx.from_numpy(x, y, a, u)

    res, log = ot.sliced_unbalanced_ot(x, y, reg_m, a, u, 10, seed=42, p=2, log=True)
    a_exp, u_exp, expected = ot.uot_1d(
        x.squeeze(), y.squeeze(), reg_m, a, u, returnCost="total", p=2
    )
    np.testing.assert_almost_equal(res, expected)
    np.testing.assert_allclose(log["a_reweighted"][:, 0], a_exp)
    np.testing.assert_allclose(log["b_reweighted"][:, 0], u_exp)

    f, g, res, log = ot.unbalanced_sliced_ot(
        x, y, reg_m, a, u, 10, seed=42, p=2, log=True
    )
    np.testing.assert_almost_equal(res, expected)
    np.testing.assert_allclose(f, a_exp)
    np.testing.assert_allclose(g, u_exp)


@pytest.skip_backend("numpy")
@pytest.skip_backend("tf")
@pytest.skip_backend("cupy")
def test_sliced_projections(nx):
    n = 100
    m = 120
    rng = np.random.RandomState(42)

    x = rng.randn(n, 4)
    y = rng.randn(m, 4)

    a = rng.uniform(0, 1, n) / 10  # unbalanced
    u = ot.utils.unif(m)

    reg_m = 1

    x, y, a, u = nx.from_numpy(x, y, a, u)

    res, log = ot.sliced_unbalanced_ot(x, y, reg_m, a, u, 10, seed=42, p=2, log=True)

    projections = log["projections"]

    res2 = ot.sliced_unbalanced_ot(x, y, reg_m, a, u, 10, seed=42, p=2)
    np.testing.assert_almost_equal(res, res2)

    res3 = ot.sliced_unbalanced_ot(x, y, reg_m, a, u, 10, projections=projections, p=2)
    np.testing.assert_almost_equal(res, res3)

    _, _, res = ot.unbalanced_sliced_ot(x, y, reg_m, a, u, 10, seed=42, p=2)

    _, _, res2 = ot.unbalanced_sliced_ot(
        x, y, reg_m, a, u, 10, projections=projections, p=2
    )
    np.testing.assert_almost_equal(res, res2)


@pytest.skip_backend("numpy")
@pytest.skip_backend("tf")
@pytest.skip_backend("cupy")
def test_sliced_inf_reg_m(nx):
    n_samples = 20  # nb samples

    rng = np.random.RandomState(42)
    xs = rng.randn(n_samples, 4)
    xt = rng.randn(n_samples, 4)

    a_np = ot.utils.unif(n_samples)
    b_np = ot.utils.unif(n_samples)

    reg_m = float("inf")

    a, b = nx.from_numpy(a_np, b_np)
    xs, xt = nx.from_numpy(xs, xt)

    suot = ot.sliced_unbalanced_ot(xs, xt, reg_m, a, b, 10, seed=42, p=2)

    a_reweighted, b_reweighted, usot = ot.unbalanced_sliced_ot(
        xs, xt, reg_m, a, b, 10, seed=42, p=2
    )

    sw = ot.sliced_wasserstein_distance(xs, xt, n_projections=10, seed=42, p=2)

    # Check right loss
    np.testing.assert_almost_equal(suot, sw**2)
    np.testing.assert_almost_equal(usot, sw**2)
    np.testing.assert_allclose(a_reweighted, a)
    np.testing.assert_allclose(b_reweighted, b)


@pytest.skip_backend("numpy")
@pytest.skip_backend("tf")
@pytest.skip_backend("cupy")
def test_semi_usot_1d(nx):
    n_samples = 20  # nb samples

    rng = np.random.RandomState(42)
    xs = rng.randn(n_samples, 1)
    xt = rng.randn(n_samples, 1)

    a_np = ot.utils.unif(n_samples)
    b_np = ot.utils.unif(n_samples)

    a, b = nx.from_numpy(a_np, b_np)
    xs, xt = nx.from_numpy(xs, xt)

    reg_m = (float("inf"), 1.0)

    a_reweighted, b_reweighted, usot = ot.unbalanced_sliced_ot(
        xs, xt, reg_m, a, b, 10, seed=42, p=2
    )
    # Check right marginals
    np.testing.assert_allclose(a, a_reweighted)
    np.testing.assert_allclose(b_reweighted.sum(), 1)

    reg_m = (1.0, float("inf"))

    a_reweighted, b_reweighted, usot = ot.unbalanced_sliced_ot(
        xs, xt, reg_m, a, b, 10, seed=42, p=2
    )
    # Check right marginals
    np.testing.assert_allclose(b, b_reweighted)
    np.testing.assert_allclose(a_reweighted.sum(), 1)


@pytest.skip_backend("numpy")
@pytest.skip_backend("tf")
@pytest.skip_backend("cupy")
@pytest.mark.parametrize(
    "reg_m",
    itertools.product(
        [1, float("inf")],
    ),
)
def test_sliced_unbalanced_relaxation_parameters(nx, reg_m):
    n = 100
    rng = np.random.RandomState(50)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = rng.rand(n)

    a, b, x = nx.from_numpy(a, b, x)

    reg_m = reg_m[0]

    # options for reg_m
    full_list_reg_m = [reg_m, reg_m]
    full_tuple_reg_m = (reg_m, reg_m)
    tuple_reg_m, list_reg_m = (reg_m), [reg_m]
    nx_reg_m = reg_m * nx.ones(1)

    list_options = [
        nx_reg_m,
        full_tuple_reg_m,
        tuple_reg_m,
        full_list_reg_m,
        list_reg_m,
    ]

    _, _, usot = ot.unbalanced_sliced_ot(x, x, reg_m, a, b, 10, seed=42, p=2)

    suot = ot.sliced_unbalanced_ot(x, x, reg_m, a, b, 10, seed=42, p=2)

    for opt in list_options:
        _, _, usot_opt = ot.unbalanced_sliced_ot(x, x, opt, a, b, 10, seed=42, p=2)
        np.testing.assert_allclose(nx.to_numpy(usot), nx.to_numpy(usot_opt), atol=1e-05)

        suot_opt = ot.sliced_unbalanced_ot(x, x, opt, a, b, 10, seed=42, p=2)
        np.testing.assert_allclose(nx.to_numpy(suot), nx.to_numpy(suot_opt), atol=1e-05)


@pytest.skip_backend("numpy")
@pytest.skip_backend("tf")
@pytest.skip_backend("cupy")
@pytest.mark.parametrize(
    "reg_m1, reg_m2",
    itertools.product(
        [1, float("inf")],
        [1, float("inf")],
    ),
)
def test_sliced_unbalanced_relaxation_parameters_pair(nx, reg_m1, reg_m2):
    n = 100
    rng = np.random.RandomState(50)

    x = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = rng.rand(n)

    a, b, x = nx.from_numpy(a, b, x)

    # options for reg_m
    full_list_reg_m = [reg_m1, reg_m2]
    full_tuple_reg_m = (reg_m1, reg_m2)
    list_options = [full_tuple_reg_m, full_list_reg_m]

    _, _, usot = ot.unbalanced_sliced_ot(x, x, (reg_m1, reg_m2), a, b, 10, seed=42, p=2)

    suot = ot.sliced_unbalanced_ot(x, x, (reg_m1, reg_m2), a, b, 10, seed=42, p=2)

    for opt in list_options:
        _, _, usot_opt = ot.unbalanced_sliced_ot(x, x, opt, a, b, 10, seed=42, p=2)
        np.testing.assert_allclose(nx.to_numpy(usot), nx.to_numpy(usot_opt), atol=1e-05)

        suot_opt = ot.sliced_unbalanced_ot(x, x, opt, a, b, 10, seed=42, p=2)
        np.testing.assert_allclose(nx.to_numpy(suot), nx.to_numpy(suot_opt), atol=1e-05)


def test_sliced_uot_type_devices(nx):
    rng = np.random.RandomState(0)

    n = 10
    x = rng.randn(n, 2)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    reg_m = 1.0

    xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v)

    if nx.__name__ in ["torch", "jax"]:
        f, g, usot = ot.unbalanced_sliced_ot(
            xb, xb, reg_m, rho_ub, rho_vb, 10, seed=42, p=2
        )

        nx.assert_same_dtype_device(xb, f)
        nx.assert_same_dtype_device(xb, g)
        nx.assert_same_dtype_device(xb, usot)
    else:
        np.testing.assert_raises(
            AssertionError,
            ot.unbalanced_sliced_ot,
            xb,
            xb,
            reg_m,
            rho_ub,
            rho_vb,
            10,
            seed=42,
            p=2,
        )

    if nx.__name__ in ["torch", "jax"]:
        suot = ot.sliced_unbalanced_ot(xb, xb, reg_m, rho_ub, rho_vb, 10, seed=42, p=2)

        nx.assert_same_dtype_device(xb, suot)
    else:
        np.testing.assert_raises(
            AssertionError,
            ot.sliced_unbalanced_ot,
            xb,
            xb,
            reg_m,
            rho_ub,
            rho_vb,
            10,
            seed=42,
            p=2,
        )
