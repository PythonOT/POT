"""Tests for ot.lp.dmmot module"""

# Author: Ronak Mehta <ronakrm@cs.wisc.edu>
#         Xizheng Yu <xyu354@wisc.edu>
#
# License: MIT License

import numpy as np
import ot


def create_test_data(nx):
    n = 4
    a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)
    a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)
    A = np.vstack([a1, a2]).T
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    A, x = nx.from_numpy(A, x)
    return A, x


def test_dmmot_monge_1dgrid_loss(nx):
    A, x = create_test_data(nx)

    # Compute loss using dmmot_monge_1dgrid_loss
    primal_obj = ot.lp.dmmot_monge_1dgrid_loss(A)
    primal_obj = nx.to_numpy(primal_obj)
    expected_primal_obj = 0.13667759626298503

    np.testing.assert_allclose(
        primal_obj,
        expected_primal_obj,
        rtol=1e-7,
        err_msg="Test failed: \
                                   Expected different primal objective value",
    )

    # Compute loss using exact OT solver with absolute ground metric
    A, x = nx.to_numpy(A, x)
    M = ot.utils.dist(x, metric="cityblock")  # absolute ground metric
    bary, _ = ot.barycenter(A, M, 1e-2, weights=None, verbose=False, log=True)
    ot_obj = 0.0
    for x in A.T:
        # deal with C-contiguous error from tensorflow backend (not sure why)
        x = np.ascontiguousarray(x)
        # compute loss
        _, log = ot.lp.emd(x, np.array(bary / np.sum(bary)), M, log=True)
        ot_obj += log["cost"]

    np.testing.assert_allclose(
        primal_obj,
        ot_obj,
        rtol=1e-7,
        err_msg="Test failed: \
                                   Expected different primal objective value",
    )


def test_dmmot_monge_1dgrid_optimize(nx):
    # test discrete_mmot_converge result
    A, _ = create_test_data(nx)
    d = 2
    niters = 10
    result = ot.lp.dmmot_monge_1dgrid_optimize(A, niters, lr_init=1e-3, lr_decay=1)

    expected_obj = np.array(
        [
            [0.05553516, 0.13082618, 0.27327479, 0.54036388],
            [0.04185365, 0.09570724, 0.24384705, 0.61859206],
        ]
    )

    assert len(result) == d, "Test failed: Expected a list of length n"
    for i in range(d):
        np.testing.assert_allclose(
            result[i],
            expected_obj[i],
            atol=1e-7,
            rtol=1e-7,
            err_msg="Test failed: \
                                   Expected vectors of all zeros",
        )


def test_dmmot_zero_mass_dual_feasibility():
    """Zero-mass bins must not leave the returned dual infeasible."""
    A = np.array([[1.0, 1.0], [0.0, 0.0]])
    obj, log = ot.lp.dmmot_monge_1dgrid_loss(A, log=True)
    u, v = np.asarray(log["dual"][0]), np.asarray(log["dual"][1])
    slack = np.array([[0.0, 1.0], [1.0, 0.0]]) - u[:, None] - v[None, :]

    np.testing.assert_allclose(obj, 0.0, atol=1e-12)
    np.testing.assert_allclose(log["dual objective"], obj, atol=1e-12)
    assert np.all(slack >= -1e-12)


def test_dmmot_positive_and_near_zero_inputs_unchanged():
    """The zero-bin completion must not alter strictly positive or near-zero bins."""
    A = np.array([[0.7, 0.3], [0.3, 0.7]])
    obj, log = ot.lp.dmmot_monge_1dgrid_loss(A, log=True)
    np.testing.assert_allclose(obj, 0.4, atol=1e-12)
    np.testing.assert_allclose(
        np.column_stack(log["dual"]), [[0, 0], [-1, 1]], atol=1e-12
    )

    A_near = np.array([[0.2, 0.3], [0.3, 0.3], [0.5, 0.4], [1e-12, 2e-12]])
    _, log_near = ot.lp.dmmot_monge_1dgrid_loss(A_near, log=True)
    np.testing.assert_allclose(
        np.column_stack(log_near["dual"]),
        [[0, 0], [1, -1], [2, -2], [3, -3]],
        atol=1e-12,
    )


def test_dmmot_multiple_zero_marginals_dual_feasibility():
    """The tighter safe bound remains feasible with zeros in several marginals."""
    A = np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]])
    obj, log = ot.lp.dmmot_monge_1dgrid_loss(A, log=True)
    u, v, w = [np.asarray(x) for x in log["dual"]]
    C = np.array(
        [
            [[max(t) - min(t) for t in [(i, j, k)]] for k in range(2)]
            for i in range(2)
            for j in range(2)
        ]
    ).reshape(2, 2, 2)

    slack = C - u[:, None, None] - v[None, :, None] - w[None, None, :]
    np.testing.assert_allclose(log["dual objective"], obj, atol=1e-12)
    assert np.all(slack >= -1e-12)
