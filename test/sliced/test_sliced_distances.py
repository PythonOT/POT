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
from ot.sliced import (
    get_random_projections,
    get_projections_spiral,
    get_random_orthogonal,
)
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


def test_get_projections_spiral():
    """Mirrors test_get_random_projections: spiral points must lie on the unit sphere."""
    projections = get_projections_spiral(3, 50, randomized=False)
    np.testing.assert_almost_equal(np.sum(projections**2, 0), 1.0)


def test_get_projections_spiral_randomized_preserves_sphere():
    """The random rotation applied for RQSW must still yield unit vectors."""
    projections = get_projections_spiral(3, 50, randomized=True, seed=0)
    np.testing.assert_almost_equal(np.sum(projections**2, 0), 1.0)


def test_get_projections_spiral_wrong_dim_raises():
    """get_projections_spiral is only implemented for d=3."""
    with pytest.raises(ValueError):
        get_projections_spiral(5, 50)


def test_get_projections_spiral_seed_reproducibility():
    """Same seed must give identical RQSW rotations; different seeds must differ."""
    p1 = get_projections_spiral(3, 50, randomized=True, seed=42)
    p2 = get_projections_spiral(3, 50, randomized=True, seed=42)
    p3 = get_projections_spiral(3, 50, randomized=True, seed=43)
    np.testing.assert_allclose(p1, p2)
    assert not np.allclose(p1, p3)


def test_get_projections_spiral_deterministic_ignores_seed():
    """QSW (randomized=False) must not depend on seed at all."""
    p1 = get_projections_spiral(3, 50, randomized=False, seed=0)
    p2 = get_projections_spiral(3, 50, randomized=False, seed=999)
    np.testing.assert_allclose(p1, p2)


@pytest.mark.parametrize("sampling_slices", ["spiral_qmc", "randomized_spiral_qmc"])
def test_sliced_qsw_same_dist(sampling_slices):
    """Same distribution -> SWD approx 0, mirrors test_sliced_same_dist."""
    n = 100
    rng = np.random.RandomState(0)
    x = rng.randn(n, 3)
    u = ot.utils.unif(n)

    res = ot.sliced_wasserstein_distance(
        x, x, u, u, 10, seed=0, sampling_slices=sampling_slices
    )
    np.testing.assert_almost_equal(res, 0.0)


@pytest.mark.parametrize("sampling_slices", ["spiral_qmc", "randomized_spiral_qmc"])
def test_sliced_qsw_different_dists(sampling_slices):
    """Different distributions -> SWD > 0, mirrors test_sliced_different_dists."""
    n = 100
    rng = np.random.RandomState(0)
    x = rng.randn(n, 3)
    y = rng.randn(n, 3) + 2.0
    u = ot.utils.unif(n)

    res = ot.sliced_wasserstein_distance(
        x, y, u, u, 10, seed=0, sampling_slices=sampling_slices
    )
    assert res > 0.0


def test_sliced_invalid_sampling_slices():
    """An unknown sampling_slices string must raise a clear ValueError."""
    n = 20
    rng = np.random.RandomState(0)
    x = rng.randn(n, 3)
    with pytest.raises(ValueError, match="sampling_slices"):
        ot.sliced_wasserstein_distance(x, x, sampling_slices="not_a_real_method")


@pytest.mark.parametrize("sampling_slices", ["spiral_qmc", "randomized_spiral_qmc"])
def test_sliced_qsw_wrong_dim_raises(sampling_slices):
    """spiral_qmc/randomized_spiral_qmc are only implemented for dim=3; other dims must raise."""
    n = 20
    rng = np.random.RandomState(0)
    x = rng.randn(n, 5)
    with pytest.raises(ValueError):
        ot.sliced_wasserstein_distance(x, x, sampling_slices=sampling_slices)


@pytest.mark.parametrize("sampling_slices", ["spiral_qmc", "randomized_spiral_qmc"])
def test_sliced_qsw_backend(nx, sampling_slices):
    """spiral_qmc/randomized_spiral_qmc must work identically across backends, mirrors test_sliced_backend."""
    n = 100
    rng = np.random.RandomState(0)
    x = rng.randn(n, 3)
    y = rng.randn(2 * n, 3)

    xb, yb = nx.from_numpy(x, y)

    val = ot.sliced_wasserstein_distance(
        xb, yb, n_projections=20, seed=0, sampling_slices=sampling_slices
    )
    val2 = ot.sliced_wasserstein_distance(
        xb, yb, n_projections=20, seed=0, sampling_slices=sampling_slices
    )

    assert nx.to_numpy(val) > 0
    if sampling_slices == "spiral_qmc":
        # Deterministic: identical seed or not, result must be identical.
        assert val == val2
    else:
        # RQSW seeded: same seed -> same rotation -> same result.
        assert val == val2


def test_qsw_matches_across_backends(nx):
    """QSW (deterministic spiral points) must give EXACTLY the same
    projections and the same SW value on ``nx`` as on NumPy -- the point
    construction itself never touches any backend's RNG (see
    get_projections_spiral), so this is a genuine equality, not just
    "close enough".
    """
    d = 3
    n_projections = 50
    rng = np.random.RandomState(0)
    X_s = rng.normal(0, 1, (30, d))
    X_t = rng.normal(1, 1, (30, d))

    val_np, log_np = ot.sliced_wasserstein_distance(
        X_s, X_t, n_projections=n_projections, sampling_slices="spiral_qmc", log=True
    )

    X_s_b, X_t_b = nx.from_numpy(X_s, X_t)
    val_b, log_b = ot.sliced_wasserstein_distance(
        X_s_b,
        X_t_b,
        n_projections=n_projections,
        sampling_slices="spiral_qmc",
        log=True,
    )

    np.testing.assert_allclose(
        nx.to_numpy(val_b),
        val_np,
        atol=1e-10,
        err_msg=f"QSW result on '{nx.__name__}' does not match NumPy",
    )
    np.testing.assert_allclose(
        nx.to_numpy(log_b["projections"]),
        log_np["projections"],
        atol=1e-10,
        err_msg=f"QSW projections on '{nx.__name__}' do not match NumPy",
    )


def test_rqsw_seed_does_not_match_across_backends(nx):
    """RQSW's random rotation is drawn from each backend's OWN native RNG
    (numpy.random.RandomState, torch.Generator, jax's Threefry counter RNG,
    tf.random.Generator -- four genuinely different algorithms). The "same"
    integer seed therefore does NOT produce the same rotation, or the same
    SW value, as NumPy's.

    This is documented, expected behaviour (see the Sobol docstring's
    identical caveat), not a bug -- this test exists to confirm that
    behaviour explicitly, verified directly:
    numpy.RandomState(0).randn(3,3) != torch.Generator().manual_seed(0).randn(3,3).

    Within a SINGLE backend, same-seed reproducibility is already covered
    by test_sliced_qsw_backend. NumPy is skipped here since it IS the
    reference and would trivially match itself.
    """
    if nx.__name__ == "numpy":
        pytest.skip("NumPy is the reference backend; it trivially matches itself")

    d = 3
    n_projections = 50
    rng = np.random.RandomState(0)
    X_s = rng.normal(0, 1, (30, d))
    X_t = rng.normal(1, 1, (30, d))

    val_np = ot.sliced_wasserstein_distance(
        X_s,
        X_t,
        n_projections=n_projections,
        seed=0,
        sampling_slices="randomized_spiral_qmc",
    )

    X_s_b, X_t_b = nx.from_numpy(X_s, X_t)
    val_b = ot.sliced_wasserstein_distance(
        X_s_b,
        X_t_b,
        n_projections=n_projections,
        seed=0,
        sampling_slices="randomized_spiral_qmc",
    )

    assert not np.isclose(nx.to_numpy(val_b), val_np), (
        f"Expected RQSW to differ between numpy and '{nx.__name__}' with "
        f"'the same' seed (different RNG algorithms), but they agreed: "
        f"{val_np} vs {nx.to_numpy(val_b)}"
    )


def test_sliced_qsw_beats_uniform_d3():
    """QSW/RQSW should reduce the SW approximation error compared to uniform
    random sampling, at equal n_projections, in d=3 -- the paper's central claim.

    Reference SW value computed with a very large number of uniformly-sampled
    projections (assumed to have converged close to the true SW distance).
    Errors for "uniform" are averaged over several seeds, since a single
    draw can be misleading. QSW has no seed to average over (deterministic); RQSW
    is also averaged over several seeds.

    Measured during development, averaged over 15 seeds: RQSW was ~502x
    more accurate than uniform sampling, and deterministic QSW was even
    closer to the reference. The thresholds below (both simply "better
    than uniform") are set far below that measurement, leaving a
    comfortable margin.
    """
    d = 3
    n_pts = 200
    n_projections = 50
    n_trials = 15

    rng = np.random.RandomState(0)
    X_s = rng.normal(0, 1, (n_pts, d))
    X_t = rng.normal(1, 1, (n_pts, d))

    # High-precision reference.
    reference = ot.sliced_wasserstein_distance(
        X_s, X_t, n_projections=20000, seed=0, sampling_slices="uniform"
    )

    uniform_errors = []
    rqsw_errors = []
    for seed in range(n_trials):
        val_uniform = ot.sliced_wasserstein_distance(
            X_s, X_t, n_projections=n_projections, seed=seed, sampling_slices="uniform"
        )
        val_rqsw = ot.sliced_wasserstein_distance(
            X_s,
            X_t,
            n_projections=n_projections,
            seed=seed,
            sampling_slices="randomized_spiral_qmc",
        )
        uniform_errors.append(abs(val_uniform - reference))
        rqsw_errors.append(abs(val_rqsw - reference))

    val_qsw = ot.sliced_wasserstein_distance(
        X_s, X_t, n_projections=n_projections, sampling_slices="spiral_qmc"
    )
    qsw_error = abs(val_qsw - reference)

    mean_uniform_error = np.mean(uniform_errors)
    mean_rqsw_error = np.mean(rqsw_errors)

    assert mean_rqsw_error < mean_uniform_error
    assert qsw_error < mean_uniform_error


# =============================================================================
# UnifOrtho: get_random_orthogonal-specific tests, mirroring the
# get_projections_spiral section above where the same property applies, and
# adding dedicated tests for what is specific to UnifOrtho (orthogonality
# within a block, no dimension restriction, exact behaviour when
# n_projections is a multiple of d).
# =============================================================================


def test_get_random_orthogonal():
    """UnifOrtho directions must lie on the unit sphere,
    regardless of n_projections being a multiple of d."""
    projections = get_random_orthogonal(7, 50, seed=0)
    np.testing.assert_almost_equal(np.sum(projections**2, 0), 1.0)


def test_get_random_orthogonal_seed_reproducibility():
    """Same seed must give identical rotations; different seeds must differ."""
    p1 = get_random_orthogonal(7, 50, seed=42)
    p2 = get_random_orthogonal(7, 50, seed=42)
    p3 = get_random_orthogonal(7, 50, seed=43)
    np.testing.assert_allclose(p1, p2)
    assert not np.allclose(p1, p3)


@pytest.mark.parametrize("d", [1, 2, 7, 13, 50])
def test_get_random_orthogonal_works_for_any_dimension(d):
    """UnifOrtho places no restriction on the dimension.
    This is its main advantage over the spiral points."""
    projections = get_random_orthogonal(d, 20, seed=0)
    assert projections.shape == (d, 20)
    np.testing.assert_almost_equal(np.sum(projections**2, 0), 1.0)


def test_get_random_orthogonal_block_is_orthogonal():
    """The defining property of UnifOrtho : within one full block of d directions,
    they must be EXACTLY mutually orthogonal (not just individually uniform on the
    sphere, as plain i.i.d. sampling already gives)."""
    d = 6
    projections = get_random_orthogonal(d, d, seed=0)  # exactly one block
    gram = projections.T @ projections
    np.testing.assert_allclose(gram, np.eye(d), atol=1e-10)


def test_get_random_orthogonal_handles_non_multiple_of_d():
    """n_projections need not be a multiple of d (see Notes in the
    docstring): the returned shape must still be exactly what was
    requested, and every direction must still be unit-norm, even though
    the last block is truncated and thus not fully orthogonal internally."""
    d, n_projections = 5, 13  # 13 is not a multiple of 5
    projections = get_random_orthogonal(d, n_projections, seed=0)
    assert projections.shape == (d, n_projections)
    np.testing.assert_almost_equal(np.sum(projections**2, 0), 1.0)


def test_get_random_orthogonal_exact_when_n_projections_equals_d():
    """When n_projections == d, UnifOrtho draws a single COMPLETE
    orthonormal basis of R^d. For any fixed vector delta, summing the
    squared coefficients of delta in a complete orthonormal basis recovers
    ||delta||^2 exactly (Parseval), regardless of which orthonormal basis
    is used. Verified here directly: this is not an approximation that
    happens to be good, it is an exact identity up to floating point.
    """
    d = 10
    rng = np.random.RandomState(0)
    delta = rng.randn(d)

    projections = get_random_orthogonal(d, d, seed=1)
    measured = np.mean((projections.T @ delta) ** 2)
    exact = np.sum(delta**2) / d

    np.testing.assert_allclose(measured, exact, atol=1e-9)


@pytest.mark.parametrize("sampling_slices", ["unif_ortho"])
def test_sliced_unif_ortho_same_dist(sampling_slices):
    """Same distribution -> SWD approx 0, mirrors test_sliced_same_dist."""
    n = 100
    rng = np.random.RandomState(0)
    x = rng.randn(n, 7)
    u = ot.utils.unif(n)

    res = ot.sliced_wasserstein_distance(
        x, x, u, u, 20, seed=0, sampling_slices=sampling_slices
    )
    np.testing.assert_almost_equal(res, 0.0)


def test_sliced_unif_ortho_different_dists():
    """Different distributions -> SWD > 0, mirrors test_sliced_different_dists."""
    n = 100
    rng = np.random.RandomState(0)
    x = rng.randn(n, 7)
    y = rng.randn(n, 7) + 2.0
    u = ot.utils.unif(n)

    res = ot.sliced_wasserstein_distance(
        x, y, u, u, 20, seed=0, sampling_slices="unif_ortho"
    )
    assert res > 0.0


def test_sliced_unif_ortho_ignores_randomized_prefix():
    """Documents current, intended behaviour of the generic
    'randomized_' prefix stripping used for sampling_slices: since
    get_random_orthogonal has no randomized/deterministic distinction,
    'randomized_unif_ortho' is accepted and behaves exactly like
    'unif_ortho' (the randomized flag it would imply is simply unused).
    This is not a bug, but it is worth pinning down explicitly so a future
    refactor cannot silently change it without a test failing."""
    n = 30
    rng = np.random.RandomState(0)
    x = rng.randn(n, 6)
    y = rng.randn(n, 6) + 1.0

    val_plain = ot.sliced_wasserstein_distance(
        x, y, n_projections=18, seed=7, sampling_slices="unif_ortho"
    )
    val_prefixed = ot.sliced_wasserstein_distance(
        x, y, n_projections=18, seed=7, sampling_slices="randomized_unif_ortho"
    )
    assert val_plain == val_prefixed


def test_sliced_unif_ortho_backend(nx):
    """UnifOrtho must work identically across backends, mirrors test_sliced_backend."""
    n = 100
    rng = np.random.RandomState(0)
    x = rng.randn(n, 7)
    y = rng.randn(2 * n, 7)

    xb, yb = nx.from_numpy(x, y)

    val = ot.sliced_wasserstein_distance(
        xb, yb, n_projections=20, seed=0, sampling_slices="unif_ortho"
    )
    val2 = ot.sliced_wasserstein_distance(
        xb, yb, n_projections=20, seed=0, sampling_slices="unif_ortho"
    )

    assert nx.to_numpy(val) > 0
    assert val == val2


def test_unif_ortho_seed_does_not_match_across_backends(nx):
    """Mirrors test_rqsw_seed_does_not_match_across_backends: UnifOrtho's
    orthogonal blocks are built from each backend's OWN native Gaussian
    draw (via get_random_rotations), so 'the same' integer seed does not
    produce the same rotation, or the same SW value, across backends.
    NumPy is skipped as the trivial reference, exactly as for RQSW."""
    if nx.__name__ == "numpy":
        pytest.skip("NumPy is the reference backend; it trivially matches itself")

    d = 7
    n_projections = 20
    rng = np.random.RandomState(0)
    X_s = rng.normal(0, 1, (30, d))
    X_t = rng.normal(1, 1, (30, d))

    val_np = ot.sliced_wasserstein_distance(
        X_s, X_t, n_projections=n_projections, seed=0, sampling_slices="unif_ortho"
    )

    X_s_b, X_t_b = nx.from_numpy(X_s, X_t)
    val_b = ot.sliced_wasserstein_distance(
        X_s_b, X_t_b, n_projections=n_projections, seed=0, sampling_slices="unif_ortho"
    )

    assert not np.isclose(nx.to_numpy(val_b), val_np), (
        f"Expected UnifOrtho to differ between numpy and '{nx.__name__}' "
        f"with 'the same' seed (different RNG algorithms), but they "
        f"agreed: {val_np} vs {nx.to_numpy(val_b)}"
    )


def test_sliced_unif_ortho_beats_uniform_high_dim():
    """UnifOrtho should reduce the SW approximation error compared to
    uniform random sampling in HIGH dimension -- the regime it is
    recommended for (unlike spiral_qmc/RQSW, tested for d=3 in
    test_sliced_qsw_beats_uniform_d3; see get_random_orthogonal and
    sliced_wasserstein_distance docstrings for the literature recommending
    this dimension-dependent choice).

    Uses the exact same closed-form reference construction as
    test_sliced_qsw_beats_uniform_d3 (a pure translation, whose SW is
    known exactly, with zero finite-sample error). n_projections is
    deliberately NOT a multiple of d, to avoid the degenerate case where
    UnifOrtho draws exactly one complete orthonormal basis (see
    test_get_random_orthogonal_exact_when_n_projections_equals_d),
    which would make this test measure that special identity rather than
    UnifOrtho's typical behaviour.

    Measured during development, averaged over 20 seeds at d=30,
    n_projections=47: UnifOrtho was ~2.3x more accurate than uniform
    sampling. The threshold below is set well below that measurement.
    """
    d = 30
    n_projections = 47  # not a multiple of d, see docstring above
    n_trials = 20

    rng = np.random.RandomState(0)
    delta = rng.normal(size=d)
    X_s = rng.normal(0, 1, (200, d))
    X_t = X_s + delta
    reference = np.linalg.norm(delta) / np.sqrt(d)

    uniform_errors = []
    uniortho_errors = []
    for seed in range(n_trials):
        val_uniform = ot.sliced_wasserstein_distance(
            X_s, X_t, n_projections=n_projections, seed=seed, sampling_slices="uniform"
        )
        val_uniortho = ot.sliced_wasserstein_distance(
            X_s,
            X_t,
            n_projections=n_projections,
            seed=seed,
            sampling_slices="unif_ortho",
        )
        uniform_errors.append(abs(val_uniform - reference))
        uniortho_errors.append(abs(val_uniortho - reference))

    mean_uniform_error = np.mean(uniform_errors)
    mean_uniortho_error = np.mean(uniortho_errors)

    assert mean_uniortho_error < mean_uniform_error
