"""Tests for module 1d Wasserstein solver"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty <ncourty@irisa.fr>
#         Clément Bonet <clement.bonet.mapp@polytechnique.edu>
#         Laetitia Chapel <laetitia.chapel@irisa.fr>
#
# License: MIT License

import numpy as np
import pytest

import ot
from ot.backend import tf
from ot.lp import wasserstein_1d
from scipy.stats import wasserstein_distance
from scipy.sparse import coo_array


def test_emd_1d_emd2_1d_with_weights():
    # test emd1d gives similar results as emd
    n = 20
    m = 30
    rng = np.random.RandomState(0)
    u = rng.randn(n, 1)
    v = rng.randn(m, 1)

    w_u = rng.uniform(0.0, 1.0, n)
    w_u = w_u / w_u.sum()

    w_v = rng.uniform(0.0, 1.0, m)
    w_v = w_v / w_v.sum()

    M = ot.dist(u, v, metric="sqeuclidean")

    G, log = ot.emd(w_u, w_v, M, log=True)
    wass = log["cost"]
    G_1d, log = ot.emd_1d(u, v, w_u, w_v, metric="sqeuclidean", log=True)
    wass1d = log["cost"]
    wass1d_emd2 = ot.emd2_1d(u, v, w_u, w_v, metric="sqeuclidean", log=False)
    wass1d_euc = ot.emd2_1d(u, v, w_u, w_v, metric="euclidean", log=False)

    # check loss is similar
    np.testing.assert_allclose(wass, wass1d)
    np.testing.assert_allclose(wass, wass1d_emd2)

    # check loss is similar to scipy's implementation for Euclidean metric
    wass_sp = wasserstein_distance(u.reshape((-1,)), v.reshape((-1,)), w_u, w_v)
    np.testing.assert_allclose(wass_sp, wass1d_euc)

    # check constraints
    np.testing.assert_allclose(w_u, G.sum(1))
    np.testing.assert_allclose(w_v, G.sum(0))

    # check that an error is raised if the metric is not a Minkowski one
    np.testing.assert_raises(ValueError, ot.emd_1d, u, v, w_u, w_v, metric="cosine")
    np.testing.assert_raises(ValueError, ot.emd2_1d, u, v, w_u, w_v, metric="cosine")


def test_wasserstein_1d(nx):
    rng = np.random.RandomState(0)

    n = 100
    x = np.linspace(0, 5, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v)

    # test 1 : wasserstein_1d should be close to scipy W_1 implementation
    np.testing.assert_almost_equal(
        wasserstein_1d(xb, xb, rho_ub, rho_vb, p=1),
        wasserstein_distance(x, x, rho_u, rho_v),
    )

    # test 2 : wasserstein_1d should be close to one when only translating the support
    np.testing.assert_almost_equal(wasserstein_1d(xb, xb + 1, p=2), 1.0)

    # test 3 : arrays test
    X = np.stack((np.linspace(0, 5, n), np.linspace(0, 5, n) * 10), -1)
    Xb = nx.from_numpy(X)
    res = wasserstein_1d(Xb, Xb, rho_ub, rho_vb, p=2)
    np.testing.assert_almost_equal(100 * res[0], res[1], decimal=4)


def test_wasserstein_1d_type_devices(nx):
    rng = np.random.RandomState(0)

    n = 10
    x = np.linspace(0, 5, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    for tp in nx.__type_list__:
        # print(nx.dtype_device(tp))

        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v, type_as=tp)

        res = wasserstein_1d(xb, xb, rho_ub, rho_vb, p=1)

        nx.assert_same_dtype_device(xb, res)


@pytest.mark.skipif(not tf, reason="tf not installed")
def test_wasserstein_1d_device_tf():
    nx = ot.backend.TensorflowBackend()
    rng = np.random.RandomState(0)
    n = 10
    x = np.linspace(0, 5, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    # Check that everything stays on the CPU
    with tf.device("/CPU:0"):
        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v)
        res = wasserstein_1d(xb, xb, rho_ub, rho_vb, p=1)
        nx.assert_same_dtype_device(xb, res)

    if len(tf.config.list_physical_devices("GPU")) > 0:
        # Check that everything happens on the GPU
        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v)
        res = wasserstein_1d(xb, xb, rho_ub, rho_vb, p=1)
        nx.assert_same_dtype_device(xb, res)
        assert nx.dtype_device(res)[1].startswith("GPU")


def test_emd_1d_emd2_1d():
    # test emd1d gives similar results as emd
    n = 20
    m = 30
    rng = np.random.RandomState(0)
    u = rng.randn(n, 1)
    v = rng.randn(m, 1)

    M = ot.dist(u, v, metric="sqeuclidean")

    G, log = ot.emd([], [], M, log=True)
    wass = log["cost"]
    G_1d, log = ot.emd_1d(u, v, [], [], metric="sqeuclidean", log=True)
    wass1d = log["cost"]
    wass1d_emd2 = ot.emd2_1d(u, v, [], [], metric="sqeuclidean", log=False)
    wass1d_euc = ot.emd2_1d(u, v, [], [], metric="euclidean", log=False)

    # check loss is similar
    np.testing.assert_allclose(wass, wass1d)
    np.testing.assert_allclose(wass, wass1d_emd2)

    # check loss is similar to scipy's implementation for Euclidean metric
    wass_sp = wasserstein_distance(u.reshape((-1,)), v.reshape((-1,)))
    np.testing.assert_allclose(wass_sp, wass1d_euc)

    # check constraints
    np.testing.assert_allclose(np.ones((n,)) / n, G.sum(1))
    np.testing.assert_allclose(np.ones((m,)) / m, G.sum(0))

    # check G is similar
    np.testing.assert_allclose(G, G_1d, atol=1e-15)

    # check AssertionError is raised if called on non 1d arrays
    u = rng.randn(n, 2)
    v = rng.randn(m, 2)
    with pytest.raises(AssertionError):
        ot.emd_1d(u, v, [], [])


def test_emd1d_type_devices(nx):
    rng = np.random.RandomState(0)

    n = 10
    x = np.linspace(0, 5, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    for tp in nx.__type_list__:
        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v, type_as=tp)

        emd = ot.emd_1d(xb, xb, rho_ub, rho_vb)
        emd2 = ot.emd2_1d(xb, xb, rho_ub, rho_vb)

        nx.assert_same_dtype_device(xb, emd)
        nx.assert_same_dtype_device(xb, emd2)


@pytest.mark.skipif(not tf, reason="tf not installed")
def test_emd1d_device_tf():
    nx = ot.backend.TensorflowBackend()
    rng = np.random.RandomState(0)
    n = 10
    x = np.linspace(0, 5, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    # Check that everything stays on the CPU
    with tf.device("/CPU:0"):
        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v)
        emd = ot.emd_1d(xb, xb, rho_ub, rho_vb)
        emd2 = ot.emd2_1d(xb, xb, rho_ub, rho_vb)
        nx.assert_same_dtype_device(xb, emd)
        nx.assert_same_dtype_device(xb, emd2)

    if len(tf.config.list_physical_devices("GPU")) > 0:
        # Check that everything happens on the GPU
        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v)
        emd = ot.emd_1d(xb, xb, rho_ub, rho_vb)
        emd2 = ot.emd2_1d(xb, xb, rho_ub, rho_vb)
        nx.assert_same_dtype_device(xb, emd)
        nx.assert_same_dtype_device(xb, emd2)
        assert nx.dtype_device(emd)[1].startswith("GPU")


@pytest.skip_backend("numpy")
@pytest.skip_backend("tf")
@pytest.skip_backend("cupy")
def test_emd1d_dual_with_weights(nx):
    # test emd1d_dual gives similar results as emd
    n = 20
    m = 30
    rng = np.random.RandomState(0)
    u = rng.randn(n, 1)
    v = rng.randn(m, 1)

    w_u = rng.uniform(0.0, 1.0, n)
    w_u = w_u / w_u.sum()

    w_v = rng.uniform(0.0, 1.0, m)
    w_v = w_v / w_v.sum()

    u, v, w_u, w_v = nx.from_numpy(u, v, w_u, w_v)

    M = ot.dist(u, v, metric="sqeuclidean")
    G, log = ot.emd(w_u, w_v, M, log=True)
    wass = log["cost"]

    f, g, wass1d = ot.emd_1d_dual_backprop(u, v, w_u, w_v, p=2)

    # check loss is similar
    np.testing.assert_allclose(wass, wass1d)
    np.testing.assert_allclose(wass, nx.sum(f[:, 0] * w_u) + nx.sum(g[:, 0] * w_v))


@pytest.skip_backend("jax")  # problem with jax on macOS
def test_emd1d_dual_backprop_batch(nx):
    rng = np.random.RandomState(0)

    n = 20

    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    rho_ub, rho_vb = nx.from_numpy(rho_u, rho_v)

    X = np.stack((np.linspace(0, 5, n), np.linspace(0, 5, n) * 10), -1)

    Xb = nx.from_numpy(X)

    if nx.__name__ in ["torch", "jax"]:
        f, g, res = ot.emd_1d_dual_backprop(Xb, Xb, rho_ub, rho_vb, p=2)

        cost_dual = nx.sum(f * rho_ub[:, None], axis=0) + nx.sum(
            g * rho_vb[:, None], axis=0
        )

        np.testing.assert_almost_equal(100 * res[0], res[1], decimal=4)
        np.testing.assert_allclose(cost_dual, res)
    else:
        np.testing.assert_raises(
            AssertionError, ot.emd_1d_dual_backprop, Xb, Xb, rho_ub, rho_vb, p=2
        )


def test_emd1d_dual_type_devices(nx):
    rng = np.random.RandomState(0)

    n = 10
    x = np.linspace(0, 5, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    for tp in nx.__type_list__:
        xb, rho_ub, rho_vb = nx.from_numpy(x, rho_u, rho_v, type_as=tp)
        if nx.__name__ == "torch" or nx.__name__ == "jax":
            f, g, res = ot.emd_1d_dual_backprop(xb, xb, rho_ub, rho_vb, p=1)
            nx.assert_same_dtype_device(xb, res)
            nx.assert_same_dtype_device(xb, f)
            nx.assert_same_dtype_device(xb, g)


def test_index_quantile_function(nx):
    n = 10
    rng = np.random.RandomState(0)
    u = rng.randn(
        n,
    )
    a = ot.unif(n)

    u, a = nx.from_numpy(u, a)
    u = nx.sort(u)

    quantiles = nx.cumsum(a)

    q = ot.lp.solver_1d.quantile_function(quantiles, quantiles, u)
    q2, idx = ot.lp.solver_1d.quantile_function(
        quantiles, quantiles, u, return_index=True
    )

    np.testing.assert_allclose(q, q2)
    np.testing.assert_allclose(nx.to_numpy(idx), np.arange(n))


def test_wasserstein_1d_plan():
    rng = np.random.RandomState(0)

    n = 10
    m = 4
    d = 1
    rng = np.random.RandomState(0)

    x = rng.randn(n, d)
    y = rng.randn(m, d)
    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, m)
    b /= b.sum()

    _, plan_1d = wasserstein_1d(x, y, a, b, p=2, return_plans=True)
    plan_1d = plan_1d[0].toarray()
    plan_emd = ot.emd(a, b, ot.dist(x, y))
    np.testing.assert_allclose(plan_1d, plan_emd, atol=1e-05)

    _, plan_1d_tuple = wasserstein_1d(x, y, a, b, p=2, return_plans="coo_tuple")
    plan_1d_tuple = coo_array(
        (plan_1d_tuple[0].data, (plan_1d_tuple[0].rows, plan_1d_tuple[0].cols)),
        shape=(x.shape[0], y.shape[0]),
    ).toarray()
    np.testing.assert_allclose(plan_1d_tuple, plan_emd, atol=1e-05)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("weights", ["uniform", "shared", "batched"])
@pytest.mark.parametrize("require_sort", [False, True])
@pytest.mark.parametrize("return_plans", [True, "coo_tuple"])
def test_wasserstein_1d_batched_plans(
    nx, batch_shape, weights, require_sort, return_plans
):
    rng = np.random.RandomState(42)
    n, m = 7, 5
    u = rng.randn(n, *batch_shape)
    v = rng.randn(m, *batch_shape)
    if not require_sort:
        u = np.sort(u, axis=0)
        v = np.sort(v, axis=0)

    if weights == "uniform":
        a, b = np.full(u.shape, 1 / n), np.full(v.shape, 1 / m)
        ab = bb = None
    else:
        weight_shape = batch_shape if weights == "batched" else ()
        a = rng.uniform(0.1, 1, (n, *weight_shape))
        b = rng.uniform(0.1, 1, (m, *weight_shape))
        a[0] = b[-1] = 0
        a /= a.sum(axis=0)
        b /= b.sum(axis=0)
        ab, bb = nx.from_numpy(a, b)
        if weights == "shared":
            a = np.broadcast_to(a.reshape((n,) + (1,) * len(batch_shape)), u.shape)
            b = np.broadcast_to(b.reshape((m,) + (1,) * len(batch_shape)), v.shape)

    ub, vb = nx.from_numpy(u, v)
    cost, plans = wasserstein_1d(
        ub, vb, ab, bb, p=2, require_sort=require_sort, return_plans=return_plans
    )
    cost_only = wasserstein_1d(ub, vb, ab, bb, p=2, require_sort=require_sort)
    assert cost.shape == batch_shape
    nx.assert_same_dtype_device(ub, cost)
    np.testing.assert_allclose(nx.to_numpy(cost), nx.to_numpy(cost_only))
    batches = list(np.ndindex(batch_shape))
    assert len(plans) == len(batches)
    for plan, batch in zip(plans, batches):
        if return_plans == "coo_tuple":
            data, rows, cols = nx.to_numpy(plan.data, plan.rows, plan.cols)
            plan = coo_array((data, (rows, cols)), shape=(n, m)).toarray()
        else:
            plan = nx.to_numpy(nx.todense(plan))
        index = (slice(None),) + batch
        ai, bi = np.ascontiguousarray(a[index]), np.ascontiguousarray(b[index])
        matrix = (u[index][:, None] - v[index][None, :]) ** 2
        reference = ot.emd(ai, bi, matrix)
        np.testing.assert_allclose(plan, reference, atol=1e-12)
        np.testing.assert_allclose(plan.sum(axis=1), ai, atol=1e-12)
        np.testing.assert_allclose(plan.sum(axis=0), bi, atol=1e-12)
        np.testing.assert_allclose(nx.to_numpy(cost)[batch], (plan * matrix).sum())


@pytest.mark.parametrize("p", [1, 2])
def test_wasserstein_1d_shared_batch_weights(nx, p):
    rng = np.random.RandomState(4)
    u, v = [rng.randn(n, 2, 3) for n in (7, 5)]
    a, b = [rng.uniform(0.1, 1, (n, 2)) for n in (7, 5)]
    a /= a.sum(axis=0)
    b /= b.sum(axis=0)
    cost = nx.to_numpy(wasserstein_1d(*nx.from_numpy(u, v, a, b), p=p))
    assert cost.shape == (2, 3)
    for i, j in np.ndindex(2, 3):
        matrix = np.abs(u[:, i, j, None] - v[None, :, i, j]) ** p
        reference = ot.emd2(
            np.ascontiguousarray(a[:, i]), np.ascontiguousarray(b[:, i]), matrix
        )
        np.testing.assert_allclose(cost[i, j], reference)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3), (2, 1, 3)])
def test_quantile_function_batch_axes(nx, batch_shape):
    rng = np.random.RandomState(0)
    xs = np.sort(rng.randn(7, *batch_shape), axis=0)
    weights = rng.uniform(0.1, 1, xs.shape)
    cws = np.cumsum(weights / weights.sum(axis=0), axis=0)
    qs = np.sort(rng.uniform(0, 1, (5, *batch_shape)), axis=0)
    qs[0], qs[-1] = 0, 1
    expected_idx = np.empty(qs.shape, dtype=int)
    for batch in np.ndindex(batch_shape):
        index = (slice(None),) + batch
        expected_idx[index] = np.clip(np.searchsorted(cws[index], qs[index]), 0, 6)
    q, idx = ot.lp.solver_1d.quantile_function(
        *nx.from_numpy(qs, cws, xs), return_index=True
    )
    np.testing.assert_array_equal(nx.to_numpy(idx), expected_idx)
    np.testing.assert_array_equal(
        nx.to_numpy(q), np.take_along_axis(xs, expected_idx, axis=0)
    )


def test_wasserstein_1d_multibatch_gradients_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.RandomState(5)
    u, v = [
        torch.tensor(rng.randn(n, 2, 3), requires_grad=True).transpose(1, 2)
        for n in (4, 5)
    ]
    a, b = [torch.tensor(rng.randn(n), requires_grad=True) for n in (4, 5)]

    def loss(u, v, a, b):
        return wasserstein_1d(u, v, a.softmax(0), b.softmax(0), p=2).sum()

    result = loss(u, v, a, b)
    reference = sum(
        wasserstein_1d(u[:, i, j], v[:, i, j], a.softmax(0), b.softmax(0), p=2)
        for i in range(3)
        for j in range(2)
    )
    np.testing.assert_allclose(result.detach(), reference.detach())
    actual_grads = torch.autograd.grad(result, (u, v, a, b))
    reference_grads = torch.autograd.grad(reference, (u, v, a, b))
    for actual, expected in zip(actual_grads, reference_grads):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert torch.autograd.gradcheck(loss, (u, v, a, b))


def test_wasserstein_1d_multibatch_gradients_jax():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    rng = np.random.RandomState(5)
    u, v, a, b = [jnp.array(rng.randn(n, 2, 3)) for n in (4, 5, 4, 5)]

    def loss(u, v, a, b):
        return wasserstein_1d(
            u, v, jax.nn.softmax(a, axis=0), jax.nn.softmax(b, axis=0), p=2
        ).sum()

    def reference(u, v, a, b):
        return sum(
            loss(u[:, i, j], v[:, i, j], a[:, i, j], b[:, i, j])
            for i in range(2)
            for j in range(3)
        )

    np.testing.assert_allclose(jax.jit(loss)(u, v, a, b), reference(u, v, a, b))
    actual_grads = jax.grad(loss, argnums=(0, 1, 2, 3))(u, v, a, b)
    reference_grads = jax.grad(reference, argnums=(0, 1, 2, 3))(u, v, a, b)
    for actual, expected in zip(actual_grads, reference_grads):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
