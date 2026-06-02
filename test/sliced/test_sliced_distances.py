"""Tests for module sliced_distances"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty <ncourty@irisa.fr>
#         Eloi Tanguy <eloi.tanguy@math.cnrs.fr>
#         Laetitia Chapel <laetitia.chapel@irisa.fr>
#
# License: MIT License

import numpy as np
import pytest

import ot
from ot.sliced import get_random_projections
from ot.backend import tf, torch


def test_env():
    import sys

    print(sys.executable)


def test_get_random_projections():
    rng = np.random.RandomState(0)
    projections = get_random_projections(1000, 50, rng)
    np.testing.assert_almost_equal(np.sum(projections**2, 0), 1.0)


def test_sliced_same_dist():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    res = ot.sliced_wasserstein_distance(x, x, u, u, 10, seed=rng)
    np.testing.assert_almost_equal(res, 0.0)


def test_sliced_bad_shapes():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(n, 4)
    u = ot.utils.unif(n)

    with pytest.raises(ValueError):
        _ = ot.sliced_wasserstein_distance(x, y, u, u, 10, seed=rng)


def test_sliced_log():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 4)
    y = rng.randn(n, 4)
    u = ot.utils.unif(n)

    _, log = ot.sliced_wasserstein_distance(x, y, u, u, 10, p=1, seed=rng, log=True)
    assert len(log) == 2
    projections = log["projections"]
    projected_emds = log["projected_emds"]

    assert projections.shape[1] == len(projected_emds) == 10
    for emd in projected_emds:
        assert emd > 0


def test_sliced_different_dists():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)
    y = rng.randn(n, 2)

    res = ot.sliced_wasserstein_distance(x, y, u, u, 10, seed=rng)
    assert res > 0.0


def test_1d_sliced_equals_emd():
    n = 100
    m = 120
    rng = np.random.RandomState(0)

    x = rng.randn(n, 1)
    a = rng.uniform(0, 1, n)
    a /= a.sum()
    y = rng.randn(m, 1)
    u = ot.utils.unif(m)
    res = ot.sliced_wasserstein_distance(x, y, a, u, 10, seed=42)
    expected = ot.emd2_1d(x.squeeze(), y.squeeze(), a, u)
    np.testing.assert_almost_equal(res**2, expected)


def test_max_sliced_same_dist():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    res = ot.max_sliced_wasserstein_distance(x, x, u, u, 10, seed=rng)
    np.testing.assert_almost_equal(res, 0.0)


def test_max_sliced_different_dists():
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)
    y = rng.randn(n, 2)

    res, _ = ot.max_sliced_wasserstein_distance(x, y, u, u, 10, seed=rng, log=True)
    assert res > 0.0


def test_max_sliced_dim_check():
    n = 3
    x = np.zeros((n, 2))
    y = np.zeros((n + 1, 3))
    with pytest.raises(ValueError):
        _ = ot.max_sliced_wasserstein_distance(x, y, n_projections=10)


def test_sliced_same_proj():
    n_projections = 10
    seed = 12
    rng = np.random.RandomState(0)
    X = rng.randn(8, 2)
    Y = rng.randn(8, 2)
    cost1, log1 = ot.sliced_wasserstein_distance(
        X, Y, seed=seed, n_projections=n_projections, log=True
    )
    P = get_random_projections(X.shape[1], n_projections=10, seed=seed)
    cost2, log2 = ot.sliced_wasserstein_distance(X, Y, projections=P, log=True)

    assert np.allclose(log1["projections"], log2["projections"])
    assert np.isclose(cost1, cost2)


def test_sliced_backend(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(2 * n, 2)

    P = rng.randn(2, 20)
    P = P / np.sqrt((P**2).sum(0, keepdims=True))

    n_projections = 20

    xb, yb, Pb = nx.from_numpy(x, y, P)

    val0 = ot.sliced_wasserstein_distance(x, y, projections=P)

    val = ot.sliced_wasserstein_distance(xb, yb, n_projections=n_projections, seed=0)
    val2 = ot.sliced_wasserstein_distance(xb, yb, n_projections=n_projections, seed=0)

    assert val > 0
    assert val == val2

    valb = nx.to_numpy(ot.sliced_wasserstein_distance(xb, yb, projections=Pb))

    assert np.allclose(val0, valb)

    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, 2 * n)
    b /= b.sum()
    a_b = nx.from_numpy(a)
    b_b = nx.from_numpy(b)
    val = ot.sliced_wasserstein_distance(x, y, a=a, b=b, projections=P)
    val_b = ot.sliced_wasserstein_distance(xb, yb, a=a_b, b=b_b, projections=Pb)
    np.testing.assert_almost_equal(val, nx.to_numpy(val_b))


def test_sliced_backend_type_devices(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(2 * n, 2)

    P = rng.randn(2, 20)
    P = P / np.sqrt((P**2).sum(0, keepdims=True))

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        xb, yb, Pb = nx.from_numpy(x, y, P, type_as=tp)

        valb = ot.sliced_wasserstein_distance(xb, yb, projections=Pb)

        nx.assert_same_dtype_device(xb, valb)


@pytest.mark.skipif(not tf, reason="tf not installed")
def test_sliced_backend_device_tf():
    nx = ot.backend.TensorflowBackend()
    n = 100
    rng = np.random.RandomState(0)
    x = rng.randn(n, 2)
    y = rng.randn(2 * n, 2)
    P = rng.randn(2, 20)
    P = P / np.sqrt((P**2).sum(0, keepdims=True))

    # Check that everything stays on the CPU
    with tf.device("/CPU:0"):
        xb, yb, Pb = nx.from_numpy(x, y, P)
        valb = ot.sliced_wasserstein_distance(xb, yb, projections=Pb)
        nx.assert_same_dtype_device(xb, valb)

    if len(tf.config.list_physical_devices("GPU")) > 0:
        # Check that everything happens on the GPU
        xb, yb, Pb = nx.from_numpy(x, y, P)
        valb = ot.sliced_wasserstein_distance(xb, yb, projections=Pb)
        nx.assert_same_dtype_device(xb, valb)
        assert nx.dtype_device(valb)[1].startswith("GPU")


def test_max_sliced_backend(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(2 * n, 2)

    P = rng.randn(2, 20)
    P = P / np.sqrt((P**2).sum(0, keepdims=True))

    n_projections = 20

    xb, yb, Pb = nx.from_numpy(x, y, P)

    val0 = ot.max_sliced_wasserstein_distance(x, y, projections=P)

    val = ot.max_sliced_wasserstein_distance(
        xb, yb, n_projections=n_projections, seed=0
    )
    val2 = ot.max_sliced_wasserstein_distance(
        xb, yb, n_projections=n_projections, seed=0
    )

    assert val > 0
    assert val == val2

    valb = nx.to_numpy(ot.max_sliced_wasserstein_distance(xb, yb, projections=Pb))

    assert np.allclose(val0, valb)

    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, 2 * n)
    b /= b.sum()
    a_b = nx.from_numpy(a)
    b_b = nx.from_numpy(b)
    val = ot.max_sliced_wasserstein_distance(x, y, a=a, b=b, projections=P)
    val_b = ot.max_sliced_wasserstein_distance(xb, yb, a=a_b, b=b_b, projections=Pb)
    np.testing.assert_almost_equal(val, nx.to_numpy(val_b))


def test_max_sliced_backend_type_devices(nx):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(2 * n, 2)

    P = rng.randn(2, 20)
    P = P / np.sqrt((P**2).sum(0, keepdims=True))

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        xb, yb, Pb = nx.from_numpy(x, y, P, type_as=tp)

        valb = ot.max_sliced_wasserstein_distance(xb, yb, projections=Pb)

        nx.assert_same_dtype_device(xb, valb)


@pytest.mark.skipif(not tf, reason="tf not installed")
def test_max_sliced_backend_device_tf():
    nx = ot.backend.TensorflowBackend()
    n = 100
    rng = np.random.RandomState(0)
    x = rng.randn(n, 2)
    y = rng.randn(2 * n, 2)
    P = rng.randn(2, 20)
    P = P / np.sqrt((P**2).sum(0, keepdims=True))

    # Check that everything stays on the CPU
    with tf.device("/CPU:0"):
        xb, yb, Pb = nx.from_numpy(x, y, P)
        valb = ot.max_sliced_wasserstein_distance(xb, yb, projections=Pb)
        nx.assert_same_dtype_device(xb, valb)

    if len(tf.config.list_physical_devices("GPU")) > 0:
        # Check that everything happens on the GPU
        xb, yb, Pb = nx.from_numpy(x, y, P)
        valb = ot.max_sliced_wasserstein_distance(xb, yb, projections=Pb)
        nx.assert_same_dtype_device(xb, valb)
        assert nx.dtype_device(valb)[1].startswith("GPU")


class TestSlicedWassersteinScaler:
    """Integration tests for the scaler parameter in sliced_wasserstein_distance."""

    def test_scaler_none_matches_no_scaler(self):
        rng = np.random.RandomState(0)
        X_s = rng.normal(0, 1, (50, 3))
        X_t = rng.normal(1, 1, (50, 3))
        result_default = ot.sliced_wasserstein_distance(X_s, X_t, seed=0)
        result_none = ot.sliced_wasserstein_distance(X_s, X_t, seed=0, scaler=None)
        np.testing.assert_allclose(result_default, result_none)

    def test_scaler_with_datascaler_runs(self):
        rng = np.random.RandomState(0)
        X_s = rng.normal(0, 1, (50, 3))
        X_t = rng.normal(1, 1, (50, 3))
        scaler = ot.utils.DataScaler(norm="standard").fit([X_s, X_t])
        result = ot.sliced_wasserstein_distance(X_s, X_t, seed=0, scaler=scaler)
        assert np.isfinite(result)
        assert result >= 0

    def test_scaler_surfaces_small_scale_signal(self):
        """Scaled SWD detects a shift in a small-magnitude feature that unscaled SWD misses."""
        rng = np.random.RandomState(0)
        n = 500
        X_s = np.column_stack(
            [
                rng.normal(1000, 100, n),
                rng.normal(0, 1, n),
            ]
        )
        X_t = np.column_stack(
            [
                rng.normal(1000, 100, n),
                rng.normal(5, 1, n),
            ]
        )
        scaler = ot.utils.DataScaler(norm="standard").fit([X_s, X_t])
        swd_scaled = ot.sliced_wasserstein_distance(
            X_s, X_t, seed=0, n_projections=200, scaler=scaler
        )
        assert swd_scaled > 1.0

    def test_scaler_with_lambda(self):
        X_s = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_t = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = ot.sliced_wasserstein_distance(
            X_s, X_t, seed=0, scaler=lambda x: x / 10
        )
        assert np.isfinite(result)

    def test_invalid_scaler_raises(self):
        X_s = np.array([[1.0, 2.0]])
        X_t = np.array([[2.0, 3.0]])
        with pytest.raises(ValueError, match="scaler must be"):
            ot.sliced_wasserstein_distance(X_s, X_t, scaler=42)

    def test_max_sliced_scaler_integration(self):
        rng = np.random.RandomState(0)
        X_s = rng.normal(0, 1, (50, 3))
        X_t = rng.normal(1, 1, (50, 3))
        scaler = ot.utils.DataScaler(norm="standard").fit([X_s, X_t])
        result = ot.max_sliced_wasserstein_distance(X_s, X_t, seed=0, scaler=scaler)
        assert np.isfinite(result)


def test_sliced_wasserstein_scaler_backend(nx):
    rng = np.random.RandomState(0)
    X_s_np = rng.normal(0, 1, (60, 3))
    X_t_np = rng.normal(2, 1, (60, 3))

    n_projections = 50
    P = get_random_projections(X_s_np.shape[1], n_projections, seed=0)

    X_s_b, X_t_b, Pb = nx.from_numpy(X_s_np, X_t_np, P)

    scaler_np = ot.utils.DataScaler(norm="standard").fit([X_s_np, X_t_np])
    scaler_b = ot.utils.DataScaler(norm="standard").fit([X_s_b, X_t_b])

    val_np = ot.sliced_wasserstein_distance(
        X_s_np, X_t_np, projections=P, scaler=scaler_np
    )
    val_b = ot.sliced_wasserstein_distance(
        X_s_b, X_t_b, projections=Pb, scaler=scaler_b
    )

    np.testing.assert_allclose(nx.to_numpy(val_b), val_np, atol=1e-5)

    val_np_max = ot.max_sliced_wasserstein_distance(
        X_s_np, X_t_np, projections=P, scaler=scaler_np
    )
    val_b_max = ot.max_sliced_wasserstein_distance(
        X_s_b, X_t_b, projections=Pb, scaler=scaler_b
    )

    np.testing.assert_allclose(nx.to_numpy(val_b_max), val_np_max, atol=1e-5)
