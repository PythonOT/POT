"""Tests for ot.demd module """

# Author: Ronak Mehta <ronakrm@cs.wisc.edu>
#         Xizheng Yu <xyu354@wisc.edu>
#
# License: MIT License

import numpy as np
import ot
import pytest


def create_test_data():
    np.random.seed(1234)
    d = 2
    n = 4
    a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)
    a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)
    aa = np.vstack([a1, a2])
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    return aa, x, d, n


def test_greedy_primal_dual():
    # test greedy_primal_dual object calculation
    aa, _, _, _ = create_test_data()
    result = ot.greedy_primal_dual(aa)
    expected_primal_obj = 0.13667759626298503
    np.testing.assert_allclose(result['primal objective'],
                               expected_primal_obj,
                               rtol=1e-7,
                               err_msg="Test failed: \
                               Expected different primal objective value")


def test_demd():
    # test one demd iteration result
    aa, _, d, n = create_test_data()
    primal_obj = ot.demd(aa, n, d)
    expected_primal_obj = 0.13667759626298503
    np.testing.assert_allclose(primal_obj,
                               expected_primal_obj,
                               rtol=1e-7,
                               err_msg="Test failed: \
                               Expected different primal objective value")


def test_demd_minimize():
    # test demd_minimize result
    aa, _, d, n = create_test_data()
    niters = 10
    result = ot.demd_minimize(ot.demd, aa, d, n, 2, niters, 0.001, 5)

    expected_obj = np.array([[0.05553516, 0.13082618, 0.27327479, 0.54036388],
                             [0.04185365, 0.09570724, 0.24384705, 0.61859206]])

    assert len(result) == d, "Test failed: Expected a list of length n"
    for i in range(d):
        np.testing.assert_allclose(result[i],
                                   expected_obj[i],
                                   atol=1e-7,
                                   rtol=1e-7,
                                   err_msg="Test failed: \
                                   Expected vectors of all zeros")
