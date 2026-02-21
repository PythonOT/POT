"""Tests for module 1D Unbalanced OT"""

# Author: Clément Bonet <clement.bonet.mapp@polytechnique.edu>
#
# License: MIT License

import itertools
import numpy as np
import ot
import pytest


@pytest.skip_backend("numpy")
@pytest.skip_backend("tensorflow")
@pytest.skip_backend("cupy")
def test_uot_1d(nx):
    n_samples = 20  # nb samples

    rng = np.random.RandomState(42)
    xs = rng.randn(n_samples, 1)
    xt = rng.randn(n_samples, 1)

    a_np = ot.utils.unif(n_samples)
    b_np = ot.utils.unif(n_samples)

    reg_m = 1.0

    M = ot.dist(xs, xt)
    a, b, M = nx.from_numpy(a_np, b_np, M)
    xs, xt = nx.from_numpy(xs, xt)

    G, log = ot.unbalanced.mm_unbalanced(a, b, M, reg_m, div="kl", log=True)
    loss_mm = log["cost"]

    f, g, loss_1d = ot.unbalanced.uot_1d(xs, xt, reg_m, p=2)
    np.testing.assert_allclose(loss_1d, loss_mm, atol=1e-2)
    np.testing.assert_allclose(G.sum(0), g[:, 0], atol=1e-2)
    np.testing.assert_allclose(G.sum(1), f[:, 0], atol=1e-2)


@pytest.skip_backend("numpy")
@pytest.skip_backend("tensorflow")
@pytest.skip_backend("cupy")
def test_uot_1d_convergence(nx):
    n_samples = 20  # nb samples

    rng = np.random.RandomState(42)
    xs = rng.randn(n_samples, 1)
    xt = rng.randn(n_samples, 1)
    xs, xt = nx.from_numpy(xs, xt)

    reg_m = 1000

    # wass1d = ot.wasserstein_1d(xs, xt, p=2)
    G_1d, log = ot.emd_1d(xs, xt, metric="sqeuclidean", log=True)
    wass1d = log["cost"]
    u_w1d, v_w1d = nx.sum(G_1d, 1), nx.sum(G_1d, 0)

    u, v, loss_1d = ot.unbalanced.uot_1d(xs, xt, reg_m, p=2)
    np.testing.assert_allclose(loss_1d, wass1d, atol=1e-2)
    np.testing.assert_allclose(v_w1d, v[:, 0], atol=1e-2)
    np.testing.assert_allclose(u_w1d, u[:, 0], atol=1e-2)


@pytest.skip_backend("numpy")
@pytest.skip_backend("tensorflow")
@pytest.skip_backend("cupy")
def test_uot_1d_batch(nx):
    n_samples = 20  # nb samples
    m_samples = 30

    rng = np.random.RandomState(42)
    xs = rng.randn(n_samples, 1)
    xt = rng.randn(m_samples, 1)
    xs = np.concatenate([xs, xs], axis=1)
    xt = np.concatenate([xt, xt], axis=1)

    a_np = rng.uniform(0, 1, n_samples)  # unbalanced
    b_np = ot.utils.unif(m_samples)

    xs, xt, a, b = nx.from_numpy(xs, xt, a_np, b_np)

    reg_m = 1

    u1, v1, uot_1d = ot.unbalanced.uot_1d(xs[:, 0], xt[:, 0], reg_m, a, b, p=2)
    u, v, loss_1d = ot.unbalanced.uot_1d(xs, xt, reg_m, a, b, p=2)

    np.testing.assert_allclose(loss_1d[0], loss_1d[1], atol=1e-5)
    np.testing.assert_allclose(loss_1d[0], uot_1d, atol=1e-5)

    u1, v1, uot_1d = ot.unbalanced.uot_1d(
        xs[:, 0], xt[:, 0], reg_m, a, b, p=2, returnCost="total"
    )
    u, v, loss_1d = ot.unbalanced.uot_1d(xs, xt, reg_m, a, b, p=2, returnCost="total")

    np.testing.assert_allclose(loss_1d[0], loss_1d[1], atol=1e-5)
    np.testing.assert_allclose(loss_1d[0], uot_1d, atol=1e-5)


@pytest.skip_backend("numpy")
@pytest.skip_backend("tensorflow")
@pytest.skip_backend("cupy")
def test_uot_1d_inf_reg_m_backprop(nx):
    n_samples = 20  # nb samples

    rng = np.random.RandomState(42)
    xs = rng.randn(n_samples, 1)
    xt = rng.randn(n_samples, 1)

    a_np = ot.utils.unif(n_samples)
    b_np = ot.utils.unif(n_samples)

    reg_m = float("inf")

    a, b = nx.from_numpy(a_np, b_np)
    xs, xt = nx.from_numpy(xs, xt)

    f_w1d, g_w1d, wass1d = ot.emd_1d_dual_backprop(xs, xt, a, b, p=2)
    u, v, loss_1d, log = ot.unbalanced.uot_1d(xs, xt, reg_m, a, b, p=2, log=True)

    # Check right loss
    np.testing.assert_allclose(loss_1d, wass1d)

    # Check right marginals
    np.testing.assert_allclose(a, u[:, 0])
    np.testing.assert_allclose(b, v[:, 0])

    # Check potentials
    np.testing.assert_allclose(f_w1d, log["f"])
    np.testing.assert_allclose(g_w1d, log["g"])


@pytest.skip_backend("numpy")
@pytest.skip_backend("tensorflow")
@pytest.skip_backend("cupy")
def test_semi_uot_1d_backprop(nx):
    n_samples = 20  # nb samples

    rng = np.random.RandomState(42)
    xs = rng.randn(n_samples, 1)
    xt = rng.randn(n_samples, 1)

    a_np = ot.utils.unif(n_samples)
    b_np = ot.utils.unif(n_samples)

    a, b = nx.from_numpy(a_np, b_np)
    xs, xt = nx.from_numpy(xs, xt)

    reg_m = (float("inf"), 1.0)

    u, v, loss_1d = ot.unbalanced.uot_1d(xs, xt, reg_m, p=2)

    # Check right marginals
    np.testing.assert_allclose(a, u[:, 0])
    np.testing.assert_allclose(v[:, 0].sum(), 1)

    reg_m = (1.0, float("inf"))

    u, v, loss_1d = ot.unbalanced.uot_1d(xs, xt, reg_m, p=2)

    # Check right marginals
    np.testing.assert_allclose(b, v[:, 0])
    np.testing.assert_allclose(u[:, 0].sum(), 1)


@pytest.skip_backend("jax")  # problem with jax on macOS
@pytest.skip_backend("numpy")
@pytest.skip_backend("tensorflow")
@pytest.skip_backend("cupy")
@pytest.mark.parametrize(
    "reg_m",
    itertools.product(
        [1, float("inf")],
    ),
)
def test_unbalanced_relaxation_parameters_backprop(nx, reg_m):
    n = 100
    rng = np.random.RandomState(50)

    x = rng.randn(n, 2)
    y = rng.randn(n, 2)
    a = ot.utils.unif(n)

    # make dists unbalanced
    b = rng.rand(n, 2)

    a, b, x, y = nx.from_numpy(a, b, x, y)

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

    u, v, loss = ot.unbalanced.uot_1d(x, y, reg_m, u_weights=a, v_weights=b, p=2)

    for opt in list_options:
        u, v, loss_opt = ot.unbalanced.uot_1d(x, y, opt, u_weights=a, v_weights=b, p=2)

        np.testing.assert_allclose(nx.to_numpy(loss), nx.to_numpy(loss_opt), atol=1e-05)


@pytest.skip_backend("jax")  # problem with jax on macOS
@pytest.skip_backend("numpy")
@pytest.skip_backend("tensorflow")
@pytest.skip_backend("cupy")
@pytest.mark.parametrize(
    "reg_m1, reg_m2",
    itertools.product(
        [1, float("inf")],
        [1, float("inf")],
    ),
)
def test_unbalanced_relaxation_parameters_pair_backprop(nx, reg_m1, reg_m2):
    # test generalized sinkhorn for unbalanced OT
    n = 100
    rng = np.random.RandomState(50)

    x = rng.randn(n, 2)
    y = rng.randn(n, 2)
    a = ot.utils.unif(n)
    b = ot.utils.unif(n)

    a, b, x, y = nx.from_numpy(a, b, x, y)

    # options for reg_m
    full_list_reg_m = [reg_m1, reg_m2]
    full_tuple_reg_m = (reg_m1, reg_m2)
    list_options = [full_tuple_reg_m, full_list_reg_m]

    _, _, loss = ot.unbalanced.uot_1d(
        x, y, (reg_m1, reg_m2), u_weights=a, v_weights=b, p=2
    )

    for opt in list_options:
        _, _, loss_opt = ot.unbalanced.uot_1d(x, y, opt, u_weights=a, v_weights=b, p=2)

        np.testing.assert_allclose(nx.to_numpy(loss), nx.to_numpy(loss_opt), atol=1e-05)


def test_uot_1d_type_devices_backprop(nx):
    rng = np.random.RandomState(0)

    n = 10
    x = np.linspace(0, 5, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    reg_m = 1.0

    xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v)

    if nx.__name__ in ["torch", "jax"]:
        f, g, _ = ot.unbalanced.uot_1d(xb, xb, reg_m, rho_ub, rho_vb, p=2)

        nx.assert_same_dtype_device(xb, f)
        nx.assert_same_dtype_device(xb, g)
    else:
        np.testing.assert_raises(
            AssertionError, ot.unbalanced.uot_1d, xb, xb, reg_m, rho_ub, rho_vb, p=2
        )
