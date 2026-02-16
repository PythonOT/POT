"""Tests for main module ot"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import warnings

import numpy as np
import pytest

import ot
from ot.datasets import make_1D_gauss as gauss
from ot.backend import torch, tf, get_backend


def test_emd_dimension_and_mass_mismatch():
    # test emd and emd2 for dimension mismatch
    n_samples = 100
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples, n_features)
    a = ot.utils.unif(n_samples + 1)

    M = ot.dist(x, x)

    np.testing.assert_raises(AssertionError, ot.emd, a, a, M)

    np.testing.assert_raises(AssertionError, ot.emd2, a, a, M)

    # test emd and emd2 for mass mismatch
    a = ot.utils.unif(n_samples)
    b = a.copy()
    a[0] = 100
    np.testing.assert_raises(AssertionError, ot.emd, a, b, M)
    np.testing.assert_raises(AssertionError, ot.emd2, a, b, M)


def test_emd_backends(nx):
    n_samples = 100
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples, n_features)
    a = ot.utils.unif(n_samples)

    M = ot.dist(x, y)

    G = ot.emd(a, a, M)

    ab, Mb = nx.from_numpy(a, M)

    Gb = ot.emd(ab, ab, Mb)

    np.allclose(G, nx.to_numpy(Gb))


def test_emd2_backends(nx):
    n_samples = 100
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples, n_features)
    a = ot.utils.unif(n_samples)

    M = ot.dist(x, y)

    val = ot.emd2(a, a, M)

    ab, Mb = nx.from_numpy(a, M)

    valb = ot.emd2(ab, ab, Mb)

    # check with empty inputs
    valb2 = ot.emd2([], [], Mb)

    np.allclose(val, nx.to_numpy(valb))
    np.allclose(val, nx.to_numpy(valb2))


def test_emd_emd2_types_devices(nx):
    n_samples = 100
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples, n_features)
    a = ot.utils.unif(n_samples)

    M = ot.dist(x, y)

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        ab, Mb = nx.from_numpy(a, M, type_as=tp)

        Gb = ot.emd(ab, ab, Mb)

        w = ot.emd2(ab, ab, Mb)

        nx.assert_same_dtype_device(Mb, Gb)
        nx.assert_same_dtype_device(Mb, w)


@pytest.mark.skipif(not tf, reason="tf not installed")
def test_emd_emd2_devices_tf():
    nx = ot.backend.TensorflowBackend()

    n_samples = 100
    n_features = 2
    rng = np.random.RandomState(0)
    x = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples, n_features)
    a = ot.utils.unif(n_samples)
    M = ot.dist(x, y)

    # Check that everything stays on the CPU
    with tf.device("/CPU:0"):
        ab, Mb = nx.from_numpy(a, M)
        Gb = ot.emd(ab, ab, Mb)
        w = ot.emd2(ab, ab, Mb)
        nx.assert_same_dtype_device(Mb, Gb)
        nx.assert_same_dtype_device(Mb, w)

    if len(tf.config.list_physical_devices("GPU")) > 0:
        # Check that everything happens on the GPU
        ab, Mb = nx.from_numpy(a, M)
        Gb = ot.emd(ab, ab, Mb)
        w = ot.emd2(ab, ab, Mb)
        nx.assert_same_dtype_device(Mb, Gb)
        nx.assert_same_dtype_device(Mb, w)
        assert nx.dtype_device(Gb)[1].startswith("GPU")


def test_emd2_gradients():
    n_samples = 100
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples, n_features)
    a = ot.utils.unif(n_samples)

    M = ot.dist(x, y)

    if torch:
        a1 = torch.tensor(a, requires_grad=True)
        b1 = torch.tensor(a, requires_grad=True)
        M1 = torch.tensor(M, requires_grad=True)

        val, log = ot.emd2(a1, b1, M1, log=True)

        val.backward()

        assert a1.shape == a1.grad.shape
        assert b1.shape == b1.grad.shape
        assert M1.shape == M1.grad.shape

        assert np.allclose(
            a1.grad.cpu().detach().numpy(),
            log["u"].cpu().detach().numpy() - log["u"].cpu().detach().numpy().mean(),
        )

        assert np.allclose(
            b1.grad.cpu().detach().numpy(),
            log["v"].cpu().detach().numpy() - log["v"].cpu().detach().numpy().mean(),
        )

        # Testing for bug #309, checking for scaling of gradient
        a2 = torch.tensor(a, requires_grad=True)
        b2 = torch.tensor(a, requires_grad=True)
        M2 = torch.tensor(M, requires_grad=True)

        val = 10.0 * ot.emd2(a2, b2, M2)

        val.backward()

        assert np.allclose(
            10.0 * a1.grad.cpu().detach().numpy(), a2.grad.cpu().detach().numpy()
        )
        assert np.allclose(
            10.0 * b1.grad.cpu().detach().numpy(), b2.grad.cpu().detach().numpy()
        )
        assert np.allclose(
            10.0 * M1.grad.cpu().detach().numpy(), M2.grad.cpu().detach().numpy()
        )


def test_emd_emd2():
    # test emd and emd2 for simple identity
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = ot.emd(u, u, M)

    # check G is identity
    np.testing.assert_allclose(G, np.eye(n) / n)
    # check constraints
    np.testing.assert_allclose(u, G.sum(1))  # cf convergence sinkhorn
    np.testing.assert_allclose(u, G.sum(0))  # cf convergence sinkhorn

    w = ot.emd2(u, u, M)
    # check loss=0
    np.testing.assert_allclose(w, 0)


def test_omp_emd2():
    # test emd2 and emd2 with openmp for simple identity
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    w = ot.emd2(u, u, M)
    w2 = ot.emd2(u, u, M, numThreads=2)

    np.testing.assert_allclose(w, w2)


def test_emd_empty():
    # test emd and emd2 for simple identity
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = ot.emd([], [], M)

    # check G is identity
    np.testing.assert_allclose(G, np.eye(n) / n)
    # check constraints
    np.testing.assert_allclose(u, G.sum(1))  # cf convergence sinkhorn
    np.testing.assert_allclose(u, G.sum(0))  # cf convergence sinkhorn

    w = ot.emd2([], [], M)
    # check loss=0
    np.testing.assert_allclose(w, 0)


def test_emd2_multi():
    n = 500  # nb bins

    # bin positions
    x = np.arange(n, dtype=np.float64)

    # Gaussian distributions
    a = gauss(n, m=20, s=5)  # m= mean, s= std

    ls = np.arange(20, 500, 100)
    nb = len(ls)
    b = np.zeros((n, nb))
    for i in range(nb):
        b[:, i] = gauss(n, m=ls[i], s=10)

    # loss matrix
    M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
    # M/=M.max()

    print("Computing {} EMD ".format(nb))

    # emd loss 1 proc
    ot.tic()
    emd1 = ot.emd2(a, b, M, 1)
    ot.toc("1 proc : {} s")

    # emd loss multipro proc
    ot.tic()
    emdn = ot.emd2(a, b, M)
    ot.toc("multi proc : {} s")

    np.testing.assert_allclose(emd1, emdn)

    # emd loss multipro proc with log
    ot.tic()
    emdn = ot.emd2(a, b, M, log=True, return_matrix=True)
    ot.toc("multi proc : {} s")

    for i in range(len(emdn)):
        emd = emdn[i]
        log = emd[1]
        cost = emd[0]
        check_duality_gap(a, b[:, i], M, log["G"], log["u"], log["v"], cost)
        emdn[i] = cost

    emdn = np.array(emdn)
    np.testing.assert_allclose(emd1, emdn)


def test_lp_barycenter():
    a1 = np.array([1.0, 0, 0])[:, None]
    a2 = np.array([0, 0, 1.0])[:, None]

    A = np.hstack((a1, a2))
    M = np.array([[0, 1.0, 4.0], [1.0, 0, 1.0], [4.0, 1.0, 0]])

    # obvious barycenter between two Diracs
    bary0 = np.array([0, 1.0, 0])

    bary = ot.lp.barycenter(A, M, [0.5, 0.5])

    np.testing.assert_allclose(bary, bary0, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(bary.sum(), 1)


def test_free_support_barycenter():
    measures_locations = [
        np.array([-1.0]).reshape((1, 1)),
        np.array([1.0]).reshape((1, 1)),
    ]
    measures_weights = [np.array([1.0]), np.array([1.0])]

    X_init = np.array([-12.0]).reshape((1, 1))

    # obvious barycenter location between two Diracs
    bar_locations = np.array([0.0]).reshape((1, 1))

    X = ot.lp.free_support_barycenter(measures_locations, measures_weights, X_init)

    np.testing.assert_allclose(X, bar_locations, rtol=1e-5, atol=1e-7)


def test_free_support_barycenter_backends(nx):
    measures_locations = [
        np.array([-1.0]).reshape((1, 1)),
        np.array([1.0]).reshape((1, 1)),
    ]
    measures_weights = [np.array([1.0]), np.array([1.0])]
    X_init = np.array([-12.0]).reshape((1, 1))

    X = ot.lp.free_support_barycenter(measures_locations, measures_weights, X_init)

    measures_locations2 = nx.from_numpy(*measures_locations)
    measures_weights2 = nx.from_numpy(*measures_weights)
    X_init2 = nx.from_numpy(X_init)

    X2 = ot.lp.free_support_barycenter(measures_locations2, measures_weights2, X_init2)

    np.testing.assert_allclose(X, nx.to_numpy(X2))


def test_generalised_free_support_barycenter():
    X = [
        np.array([-1.0, -1.0]).reshape((1, 2)),
        np.array([1.0, 1.0]).reshape((1, 2)),
    ]  # two 2D points bar is obviously 0
    a = [np.array([1.0]), np.array([1.0])]

    P = [np.eye(2), np.eye(2)]

    Y_init = np.array([-12.0, 7.0]).reshape((1, 2))

    # obvious barycenter location between two 2D Diracs
    Y_true = np.array([0.0, 0.0]).reshape((1, 2))

    # test without log and no init
    Y = ot.lp.generalized_free_support_barycenter(X, a, P, 1)
    np.testing.assert_allclose(Y, Y_true, rtol=1e-5, atol=1e-7)

    # test with log and init
    Y, _ = ot.lp.generalized_free_support_barycenter(
        X, a, P, 1, Y_init=Y_init, b=np.array([1.0]), log=True
    )
    np.testing.assert_allclose(Y, Y_true, rtol=1e-5, atol=1e-7)


def test_generalised_free_support_barycenter_backends(nx):
    X = [np.array([-1.0]).reshape((1, 1)), np.array([1.0]).reshape((1, 1))]
    a = [np.array([1.0]), np.array([1.0])]
    P = [np.array([1.0]).reshape((1, 1)), np.array([1.0]).reshape((1, 1))]
    Y_init = np.array([-12.0]).reshape((1, 1))

    Y = ot.lp.generalized_free_support_barycenter(X, a, P, 1, Y_init=Y_init)

    X2 = nx.from_numpy(*X)
    a2 = nx.from_numpy(*a)
    P2 = nx.from_numpy(*P)
    Y_init2 = nx.from_numpy(Y_init)

    Y2 = ot.lp.generalized_free_support_barycenter(X2, a2, P2, 1, Y_init=Y_init2)

    np.testing.assert_allclose(Y, nx.to_numpy(Y2))


def test_free_support_barycenter_generic_costs():
    measures_locations = [
        np.array([-1.0]).reshape((1, 1)),
        np.array([1.0]).reshape((1, 1)),
    ]
    measures_weights = [np.array([1.0]), np.array([1.0])]

    X_init = np.array([-12.0]).reshape((1, 1))

    # obvious barycenter location between two Diracs
    bar_locations = np.array([0.0]).reshape((1, 1))

    def cost(x, y):
        return ot.dist(x, y)

    cost_list = [cost, cost]

    def ground_bary(y):
        out = 0
        for yk in y:
            out += yk / len(y)
        return out

    X = ot.lp.free_support_barycenter_generic_costs(
        measures_locations, measures_weights, X_init, cost_list, ground_bary
    )

    np.testing.assert_allclose(X, bar_locations, rtol=1e-5, atol=1e-7)

    # test with log and specific weights
    X2, log = ot.lp.free_support_barycenter_generic_costs(
        measures_locations,
        measures_weights,
        X_init,
        cost_list,
        ground_bary,
        a=ot.unif(1),
        log=True,
    )

    assert "X_list" in log
    assert "exit_status" in log
    assert "diff_list" in log

    np.testing.assert_allclose(X, X2, rtol=1e-5, atol=1e-7)

    # test with one iteration for Max Iterations Reached
    X3, log2 = ot.lp.free_support_barycenter_generic_costs(
        measures_locations,
        measures_weights,
        X_init,
        cost_list,
        ground_bary,
        numItermax=1,
        log=True,
    )
    assert log2["exit_status"] == "Max iterations reached"

    # test with a single callable cost
    X3, log3 = ot.lp.free_support_barycenter_generic_costs(
        measures_locations,
        measures_weights,
        X_init,
        cost,
        ground_bary,
        numItermax=1,
        log=True,
    )

    # test with no ground_bary but in numpy: requires pytorch backend
    with pytest.raises(AssertionError):
        ot.lp.free_support_barycenter_generic_costs(
            measures_locations,
            measures_weights,
            X_init,
            cost_list,
            ground_bary=None,
            numItermax=1,
        )

    # test with unknown method
    with pytest.raises(AssertionError):
        ot.lp.free_support_barycenter_generic_costs(
            measures_locations,
            measures_weights,
            X_init,
            cost_list,
            ground_bary,
            numItermax=1,
            method="unknown_method",
        )

    # test true fixed-point method
    X4, a4, log4 = ot.lp.free_support_barycenter_generic_costs(
        measures_locations,
        measures_weights,
        X_init,
        cost_list,
        ground_bary,
        numItermax=3,
        method="true_fixed_point",
        log=True,
    )

    assert "a_list" in log4
    assert X4.shape[0] == a4.shape[0] == 1
    np.testing.assert_allclose(a4, ot.unif(1), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(X, X4, rtol=1e-5, atol=1e-7)

    # test with measure cleaning and no log
    X5, a5 = ot.lp.free_support_barycenter_generic_costs(
        measures_locations,
        measures_weights,
        X_init,
        cost_list,
        ground_bary,
        numItermax=3,
        method="true_fixed_point",
        clean_measure=True,
    )
    np.testing.assert_allclose(a5, ot.unif(1), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(X, X5, rtol=1e-5, atol=1e-7)

    # test with (too) lax convergence criterion
    # for Stationary Point exit status
    X6, log6 = ot.lp.free_support_barycenter_generic_costs(
        [np.array([-1.0]).reshape((1, 1))],
        measures_weights,
        X_init,
        cost_list,
        ground_bary,
        numItermax=3,
        stopThr=1e20,
        log=True,
    )
    assert log6["exit_status"] == "Stationary Point"


@pytest.mark.skipif(not torch, reason="No torch available")
def test_free_support_barycenter_generic_costs_auto_ground_bary():
    measures_locations = [
        torch.tensor([1.0]).reshape((1, 1)),
        torch.tensor([2.0]).reshape((1, 1)),
    ]
    measures_weights = [torch.tensor([1.0]), torch.tensor([1.0])]

    X_init = torch.tensor([1.2]).reshape((1, 1))

    def cost(x, y):
        return ot.dist(x, y)

    cost_list = [cost, cost]

    def ground_bary(y):
        out = 0
        for yk in y:
            out += yk / len(y)
        return out

    X = ot.lp.free_support_barycenter_generic_costs(
        measures_locations,
        measures_weights,
        X_init,
        cost_list,
        ground_bary,
        numItermax=1,
        stopThr=-1,
    )

    X2, log2 = ot.lp.free_support_barycenter_generic_costs(
        measures_locations,
        measures_weights,
        X_init,
        cost_list,
        ground_bary=None,
        ground_bary_lr=2e-2,
        ground_bary_stopThr=1e-20,
        ground_bary_numItermax=100,
        numItermax=10,
        stopThr=-1,
        log=True,
    )

    np.testing.assert_allclose(X2.numpy(), X.numpy(), rtol=1e-4, atol=1e-4)

    X3 = ot.lp.free_support_barycenter_generic_costs(
        measures_locations,
        measures_weights,
        X_init,
        cost_list,
        ground_bary=None,
        ground_bary_lr=1e-2,
        ground_bary_stopThr=1e-20,
        ground_bary_numItermax=100,
        numItermax=10,
        ground_bary_solver="Adam",
        stopThr=-1,
    )

    np.testing.assert_allclose(X2.numpy(), X3.numpy(), rtol=1e-3, atol=1e-3)

    # test with (too) lax convergence criterion for ground barycenter
    ot.lp.free_support_barycenter_generic_costs(
        measures_locations,
        measures_weights,
        X_init,
        cost_list,
        ground_bary=None,
        numItermax=1,
        ground_bary_stopThr=100,
    )


@pytest.skip_backend("tf")  # skips because of array assignment
@pytest.skip_backend("jax")
def test_free_support_barycenter_generic_costs_backends(nx):
    measures_locations = [
        np.array([-1.0]).reshape((1, 1)),
        np.array([1.0]).reshape((1, 1)),
    ]
    measures_weights = [np.array([1.0]), np.array([1.0])]
    X_init = np.array([-12.0]).reshape((1, 1))

    def cost(x, y):
        return ot.dist(x, y)

    cost_list = [cost, cost]

    def ground_bary(y):
        out = 0
        for yk in y:
            out += yk / len(y)
        return out

    X = ot.lp.free_support_barycenter_generic_costs(
        measures_locations,
        measures_weights,
        X_init,
        cost_list,
        ground_bary,
        method="L2_barycentric_proj",
    )

    measures_locations2 = nx.from_numpy(*measures_locations)
    measures_weights2 = nx.from_numpy(*measures_weights)
    X_init2 = nx.from_numpy(X_init)

    X2 = ot.lp.free_support_barycenter_generic_costs(
        measures_locations2,
        measures_weights2,
        X_init2,
        cost_list,
        ground_bary,
        method="L2_barycentric_proj",
    )

    np.testing.assert_allclose(X, nx.to_numpy(X2))

    X, a = ot.lp.free_support_barycenter_generic_costs(
        measures_locations,
        measures_weights,
        X_init,
        cost_list,
        ground_bary,
        method="true_fixed_point",
    )

    measures_locations2 = nx.from_numpy(*measures_locations)
    measures_weights2 = nx.from_numpy(*measures_weights)
    X_init2 = nx.from_numpy(X_init)

    X2, a2 = ot.lp.free_support_barycenter_generic_costs(
        measures_locations2,
        measures_weights2,
        X_init2,
        cost_list,
        ground_bary,
        method="true_fixed_point",
    )

    np.testing.assert_allclose(a, nx.to_numpy(a2))
    np.testing.assert_allclose(X, nx.to_numpy(X2))

    ot.lp.ot_barycenter_energy(  # test without backend and callable cost
        measures_locations, measures_weights, X, a, cost, nx=None
    )


def verify_gluing_validity(gamma, J, w, pi_list):
    """
    Test the validity of the North-West gluing.
    """
    nx = get_backend(gamma)
    K = len(pi_list)
    n = pi_list[0].shape[0]
    nk_list = [pi.shape[1] for pi in pi_list]

    # Check first marginal
    a = nx.sum(gamma, axis=tuple(range(1, K + 1)))
    assert nx.allclose(a, nx.sum(pi_list[0], axis=1))

    # Check other marginals
    for k in range(K):
        b_k = nx.sum(gamma, axis=tuple(i for i in range(K + 1) if i != k + 1))
        assert nx.allclose(b_k, nx.sum(pi_list[k], axis=0))

    # Check bi-marginals
    for k in range(K):
        gamma_0k = nx.sum(gamma, axis=tuple(i for i in range(1, K + 1) if i != k + 1))
        assert nx.allclose(gamma_0k, pi_list[k])

    # Check that N <= n + sum_k n_k - K
    N = J.shape[0]
    n_k_sum = sum(nk_list)
    assert N <= n + n_k_sum - K, f"N={N}, n={n}, sum(n_k)={n_k_sum}, K={K}"

    # Check that w is on the simplex
    w_sum = nx.sum(w)
    assert nx.allclose(w_sum, 1), f"Sum of weights w is not 1: {w_sum}"

    # Check that gamma_1...K and (J, w) are consistent
    rho = nx.zeros(nk_list, type_as=gamma)
    for i in range(N):
        jj = J[i]
        rho[tuple(jj)] += w[i]

    gamma_1toK = nx.sum(gamma, axis=0)
    assert nx.allclose(rho, gamma_1toK), "rho and gamma_1...K are not consistent"


def test_north_west_mm_gluing():
    rng = np.random.RandomState(0)
    n = 7
    nk_list = [5, 6, 4]
    a = rng.rand(n)
    a = a / np.sum(a)
    b_list = [rng.rand(nk) for nk in nk_list]
    b_list = [b / np.sum(b) for b in b_list]
    M_list = [rng.rand(n, nk) for nk in nk_list]
    pi_list = [ot.emd(a, b, M) for b, M in zip(b_list, M_list)]
    J, w, log_dict = ot.lp.NorthWestMMGluing(pi_list, log=True)
    # Test the validity of the gluing
    gamma = log_dict["gamma"]
    verify_gluing_validity(gamma, J, w, pi_list)

    # test without log
    J2, w2 = ot.lp.NorthWestMMGluing(pi_list, log=False)
    np.testing.assert_allclose(J, J2)
    np.testing.assert_allclose(w, w2)

    # test setting with highly non-injective plans
    n = 6
    a = ot.unif(n)
    b_list = [a] * 3
    pi_list = [a[:, None] @ a[None, :]] * 3
    J, w, log_dict = ot.lp.NorthWestMMGluing(pi_list, log=True)
    # Test the validity of the gluing
    gamma = log_dict["gamma"]
    verify_gluing_validity(gamma, J, w, pi_list)


@pytest.skip_backend("tf")  # skips because of array assignment
@pytest.skip_backend("jax")
def test_north_west_mm_gluing_backends(nx):
    rng = np.random.RandomState(0)
    n = 7
    nk_list = [5, 6, 4]
    a = rng.rand(n)
    a = a / np.sum(a)
    b_list = [rng.rand(nk) for nk in nk_list]
    b_list = [b / np.sum(b) for b in b_list]
    M_list = [rng.rand(n, nk) for nk in nk_list]
    pi_list = [ot.emd(a, b, M) for b, M in zip(b_list, M_list)]

    pi_list2 = [nx.from_numpy(pi) for pi in pi_list]
    J, w, log_dict = ot.lp.NorthWestMMGluing(pi_list2, log=True, nx=nx)
    gamma = log_dict["gamma"]

    # Test equality with numpy solution
    J_np, w_np, log_dict_np = ot.lp.NorthWestMMGluing(pi_list, log=True)
    gamma_np = log_dict_np["gamma"]
    np.testing.assert_allclose(J, J_np)
    np.testing.assert_allclose(w, w_np)
    np.testing.assert_allclose(gamma, gamma_np)


def test_clean_discrete_measure(nx):
    a = nx.ones(3) / 3.0
    X = nx.from_numpy(np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]]))
    X_clean, a_clean = ot.lp._barycenter_solvers._clean_discrete_measure(X, a)
    a_true = nx.from_numpy(np.array([2 / 3, 1 / 3]))
    X_true = nx.from_numpy(np.array([[1.0, 1.0], [2.0, 2.0]]))
    assert a_clean.shape == a_true.shape
    assert X_clean.shape == X_true.shape
    np.testing.assert_allclose(a_clean, a_true)
    np.testing.assert_allclose(X_clean, X_true)

    a = nx.ones(3) / 3.0
    X = nx.from_numpy(np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]]))
    X_clean, a_clean = ot.lp._barycenter_solvers._clean_discrete_measure(X, a, nx=nx)
    a_true = nx.from_numpy(np.array([2 / 3, 1 / 3]))
    X_true = nx.from_numpy(np.array([[1.0, 1.0], [2.0, 2.0]]))
    assert a_clean.shape == a_true.shape
    assert X_clean.shape == X_true.shape
    np.testing.assert_allclose(a_clean, a_true)
    np.testing.assert_allclose(X_clean, X_true)

    n = 5
    a = nx.ones(n) / n
    v = nx.from_numpy(np.array([1.0, 2.0, 3.0]))
    X = nx.stack([v] * n, axis=0)
    X_clean, a_clean = ot.lp._barycenter_solvers._clean_discrete_measure(X, a)
    a_true = np.array([1.0])
    X_true = np.array([1.0, 2.0, 3.0]).reshape(1, 3)
    assert a_clean.shape == a_true.shape
    assert X_clean.shape == X_true.shape
    np.testing.assert_allclose(a_clean, a_true)
    np.testing.assert_allclose(X_clean, X_true)


def test_to_int_array(nx):
    a_np = np.array([1.0, 2.0, 3.0])
    a = nx.from_numpy(a_np)
    a_int = ot.lp._barycenter_solvers._to_int_array(a)
    a_np_int = a_np.astype(int)
    np.testing.assert_allclose(nx.to_numpy(a_int), a_np_int)
    ot.lp._barycenter_solvers._to_int_array(a, nx=nx)


@pytest.mark.skipif(not ot.lp._barycenter_solvers.cvxopt, reason="No cvxopt available")
def test_lp_barycenter_cvxopt():
    a1 = np.array([1.0, 0, 0])[:, None]
    a2 = np.array([0, 0, 1.0])[:, None]

    A = np.hstack((a1, a2))
    M = np.array([[0, 1.0, 4.0], [1.0, 0, 1.0], [4.0, 1.0, 0]])

    # obvious barycenter between two Diracs
    bary0 = np.array([0, 1.0, 0])

    bary = ot.lp.barycenter(A, M, [0.5, 0.5], solver=None)

    np.testing.assert_allclose(bary, bary0, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(bary.sum(), 1)


def test_warnings():
    n = 100  # nb bins
    m = 100  # nb bins

    mean1 = 30
    mean2 = 50

    # bin positions
    x = np.arange(n, dtype=np.float64)
    y = np.arange(m, dtype=np.float64)

    # Gaussian distributions
    a = gauss(n, m=mean1, s=5)  # m= mean, s= std

    b = gauss(m, m=mean2, s=10)

    # loss matrix
    M = ot.dist(x.reshape((-1, 1)), y.reshape((-1, 1))) ** (1.0 / 2)

    print("Computing {} EMD ".format(1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        print("Computing {} EMD ".format(1))
        ot.emd(a, b, M, numItermax=1)
        assert "numItermax" in str(w[-1].message)
        # assert len(w) == 1


def test_dual_variables():
    n = 500  # nb bins
    m = 600  # nb bins

    mean1 = 300
    mean2 = 400

    # bin positions
    x = np.arange(n, dtype=np.float64)
    y = np.arange(m, dtype=np.float64)

    # Gaussian distributions
    a = gauss(n, m=mean1, s=5)  # m= mean, s= std

    b = gauss(m, m=mean2, s=10)

    # loss matrix
    M = ot.dist(x.reshape((-1, 1)), y.reshape((-1, 1))) ** (1.0 / 2)

    print("Computing {} EMD ".format(1))

    # emd loss 1 proc
    ot.tic()
    G, log = ot.emd(a, b, M, log=True)
    ot.toc("1 proc : {} s")

    ot.tic()
    G2 = ot.emd(b, a, np.ascontiguousarray(M.T))
    ot.toc("1 proc : {} s")

    cost1 = (G * M).sum()
    # Check symmetry
    np.testing.assert_array_almost_equal(cost1, (M * G2.T).sum())
    # Check with closed-form solution for gaussians
    np.testing.assert_almost_equal(cost1, np.abs(mean1 - mean2))

    # Check that both cost computations are equivalent
    np.testing.assert_almost_equal(cost1, log["cost"])
    check_duality_gap(a, b, M, G, log["u"], log["v"], log["cost"])

    constraint_violation = log["u"][:, None] + log["v"][None, :] - M

    assert constraint_violation.max() < 1e-8


def _get_sparse_test_matrices(n1, n2, k=2, seed=42, nx=None):
    """Helper function to create sparse and dense test matrices."""
    from scipy.sparse import coo_array
    from ot.backend import NumpyBackend

    if nx is None:
        nx = NumpyBackend()

    rng = np.random.RandomState(seed)
    M_orig = rng.rand(n1, n2)

    mask = np.zeros((n1, n2))
    for i in range(n1):
        j_list = rng.choice(n2, min(k, n2), replace=False)
        for j in j_list:
            mask[i, j] = 1
    for j in range(n2):
        i_list = rng.choice(n1, min(k, n1), replace=False)
        for i in i_list:
            mask[i, j] = 1

    M_sparse_np = coo_array(M_orig * mask)
    rows, cols, data = M_sparse_np.row, M_sparse_np.col, M_sparse_np.data

    if nx.__name__ == "numpy":
        M_sparse = M_sparse_np
    else:
        rows_b = nx.from_numpy(rows.astype(np.int64))
        cols_b = nx.from_numpy(cols.astype(np.int64))
        data_b = nx.from_numpy(data)
        M_sparse = nx.coo_matrix(data_b, rows_b, cols_b, shape=(n1, n2))

    M_dense = nx.from_numpy(M_orig + 1e8 * (1 - mask))

    return M_sparse, M_dense


def test_emd_sparse_vs_dense(nx):
    """Test that sparse and dense EMD solvers produce identical results.

    Uses random sparse graphs with k=2 edges per row/column, which guarantees
    feasibility with uniform marginals.
    """
    # Skip for backends that don't support sparse matrices
    backend_name = nx.__class__.__name__.lower()
    if "jax" in backend_name or "tensorflow" in backend_name:
        pytest.skip("Backend does not support sparse matrices")

    n1 = 100
    n2 = 100
    k = 2

    M_sparse, M_dense = _get_sparse_test_matrices(n1, n2, k=k, seed=42, nx=nx)

    a = ot.utils.unif(n1, type_as=M_dense)
    b = ot.utils.unif(n2, type_as=M_dense)

    # Solve with both dense and sparse solvers
    G_dense, log_dense = ot.emd(a, b, M_dense, log=True)
    G_sparse, log_sparse = ot.emd(a, b, M_sparse, log=True)

    cost_dense = log_dense["cost"]
    cost_sparse = log_sparse["cost"]
    np.testing.assert_allclose(cost_dense, cost_sparse, rtol=1e-5, atol=1e-7)

    np.testing.assert_allclose(a, G_dense.sum(1), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(b, G_dense.sum(0), rtol=1e-5, atol=1e-7)

    assert nx.issparse(G_sparse), "Sparse solver should return a sparse matrix"

    G_sparse_dense = nx.todense(G_sparse)
    np.testing.assert_allclose(
        a, nx.to_numpy(nx.sum(G_sparse_dense, 1)), rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        b, nx.to_numpy(nx.sum(G_sparse_dense, 0)), rtol=1e-5, atol=1e-7
    )

    # Test coo_array element-wise multiplication (only works with coo_array, not coo_matrix)
    if nx.__name__ == "numpy":
        # This tests that we're using coo_array which supports element-wise operations
        M_sparse_np = M_sparse
        G_sparse_np = G_sparse
        loss_sparse = np.sum(G_sparse_np * M_sparse_np)
        # Verify the loss calculation is reasonable
        assert loss_sparse >= 0, "Sparse loss should be non-negative"


def test_emd2_sparse_vs_dense(nx):
    """Test that sparse and dense emd2 solvers produce identical costs.

    Uses random sparse graphs with k=2 edges per row/column, which guarantees
    feasibility with uniform marginals.
    """
    # Skip for backends that don't support sparse matrices
    backend_name = nx.__class__.__name__.lower()
    if "jax" in backend_name or "tensorflow" in backend_name:
        pytest.skip("Backend does not support sparse matrices")

    n1 = 100
    n2 = 150
    k = 2

    M_sparse, M_dense = _get_sparse_test_matrices(n1, n2, k=k, seed=43, nx=nx)

    a = ot.utils.unif(n1, type_as=M_dense)
    b = ot.utils.unif(n2, type_as=M_dense)

    # Solve with both dense and sparse solvers
    cost_dense = ot.emd2(a, b, M_dense)
    cost_sparse = ot.emd2(a, b, M_sparse)

    # Check costs match
    np.testing.assert_allclose(cost_dense, cost_sparse, rtol=1e-5, atol=1e-7)


def test_emd2_sparse_gradients():
    """Test that PyTorch sparse tensors support gradient computation."""
    if not torch:
        pytest.skip("PyTorch not available")

    n = 10
    a = torch.tensor(ot.utils.unif(n), requires_grad=True, dtype=torch.float64)
    b = torch.tensor(ot.utils.unif(n), requires_grad=True, dtype=torch.float64)

    rows, cols, costs = [], [], []
    for i in range(n):
        rows.append(i)
        cols.append(i)
        costs.append(0.1)
        for offset in [1, 2]:
            j = (i + offset) % n
            rows.append(i)
            cols.append(j)
            costs.append(float(offset))

    indices = torch.tensor(
        np.vstack([np.array(rows), np.array(cols)]), dtype=torch.int64
    )
    values = torch.tensor(costs, dtype=torch.float64)
    M_sparse = torch.sparse_coo_tensor(indices, values, (n, n), dtype=torch.float64)

    cost = ot.emd2(a, b, M_sparse)
    cost.backward()

    assert a.grad is not None
    assert b.grad is not None
    np.testing.assert_allclose(
        a.grad.sum().item(), -b.grad.sum().item(), rtol=1e-5, atol=1e-7
    )


def test_emd2_sparse_vs_dense_gradients():
    """Verify gradient w.r.t. cost matrix M equals transport plan G."""
    if not torch:
        pytest.skip("PyTorch not available")

    n = 4
    a = torch.tensor([0.25, 0.25, 0.25, 0.25], requires_grad=True, dtype=torch.float64)
    b = torch.tensor([0.25, 0.25, 0.25, 0.25], requires_grad=True, dtype=torch.float64)

    M_full = torch.tensor(
        [
            [0.1, 1.0, 2.0, 3.0],
            [1.0, 0.1, 1.0, 2.0],
            [2.0, 1.0, 0.1, 1.0],
            [3.0, 2.0, 1.0, 0.1],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    cost_dense = ot.emd2(a, b, M_full)
    cost_dense.backward()
    G_dense = ot.emd(a.detach(), b.detach(), M_full.detach())

    np.testing.assert_allclose(
        M_full.grad.numpy(), G_dense.numpy(), rtol=1e-7, atol=1e-10
    )

    a.grad = None
    b.grad = None

    rows, cols, costs = [], [], []
    for i in range(n):
        for j in range(max(0, i - 1), min(n, i + 2)):
            rows.append(i)
            cols.append(j)
            costs.append(M_full[i, j].item())

    rows_t = torch.tensor(rows, dtype=torch.int64)
    cols_t = torch.tensor(cols, dtype=torch.int64)
    M_sparse = torch.sparse_coo_tensor(
        torch.stack([rows_t, cols_t]),
        torch.tensor(costs, dtype=torch.float64),
        (n, n),
        dtype=torch.float64,
        requires_grad=True,
    )

    cost_sparse = ot.emd2(a, b, M_sparse)
    cost_sparse.backward()
    G_sparse = ot.emd(a.detach(), b.detach(), M_sparse.detach()).to_dense()

    grad_values = M_sparse.grad.coalesce().values().numpy()
    G_values = G_sparse[rows_t, cols_t].numpy()

    np.testing.assert_allclose(grad_values, G_values, rtol=1e-7, atol=1e-10)
    assert grad_values.sum() > 0
    assert np.abs(grad_values.sum() - 1.0) < 1e-7


def test_emd__dual_warmstart():
    # Test that warmstarting the network simplex from partial duals
    n = 100
    rng = np.random.RandomState(42)

    x = rng.randn(n, 2)
    y = rng.randn(n, 2)
    a = ot.utils.unif(n)
    b = ot.utils.unif(n)
    M = ot.dist(x, y)

    # run solver with limited iterations (stop early)
    G_partial, log_partial = ot.emd(a, b, M, numItermax=100, log=True)
    u_partial = log_partial["u"]
    v_partial = log_partial["v"]

    # resume from the partial duals with full iterations
    G_warm, log_warm = ot.emd(a, b, M, log=True, potentials_init=(u_partial, v_partial))

    # cold-start reference
    G_cold, log_cold = ot.emd(a, b, M, log=True)

    # costs should match
    np.testing.assert_allclose(G_warm, G_cold, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(log_warm["cost"], log_cold["cost"], rtol=1e-7)

    # Both should satisfy marginal constraints
    np.testing.assert_allclose(G_warm.sum(1), a, atol=1e-7)
    np.testing.assert_allclose(G_warm.sum(0), b, atol=1e-7)

    # Test emd2 with warmstart
    cost_warm_emd2, log_warm_emd2 = ot.emd2(
        a, b, M, log=True, potentials_init=(u_partial, v_partial)
    )
    cost_cold_emd2, log_cold_emd2 = ot.emd2(a, b, M, log=True)

    # costs should match between warmstart and cold start
    np.testing.assert_allclose(cost_warm_emd2, cost_cold_emd2, rtol=1e-7)
    np.testing.assert_allclose(cost_warm_emd2, log_cold["cost"], rtol=1e-7)


def check_duality_gap(a, b, M, G, u, v, cost):
    cost_dual = np.vdot(a, u) + np.vdot(b, v)
    # Check that dual and primal cost are equal
    np.testing.assert_almost_equal(cost_dual, cost)

    [ind1, ind2] = np.nonzero(G)

    # Check that reduced cost is zero on transport arcs
    np.testing.assert_array_almost_equal(
        (M - u.reshape(-1, 1) - v.reshape(1, -1))[ind1, ind2], np.zeros(ind1.size)
    )
