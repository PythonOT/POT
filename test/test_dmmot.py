"""Tests for ot.lp.dmmot module """

# Author: Ronak Mehta <ronakrm@cs.wisc.edu>
#         Xizheng Yu <xyu354@wisc.edu>
#
# License: MIT License

import numpy as np
import ot


def create_test_data():
    np.random.seed(1234)
    d = 2
    n = 4
    a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)
    a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)
    A = np.vstack([a1, a2])
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    return A.T, x


def test_discrete_mmot():
    # test one discrete_mmot iteration result
    A, _ = create_test_data()
    primal_obj = ot.lp.discrete_mmot(A)
    expected_primal_obj = 0.13667759626298503
    np.testing.assert_allclose(primal_obj,
                               expected_primal_obj,
                               rtol=1e-7,
                               err_msg="Test failed: \
                               Expected different primal objective value")


def test_discrete_mmot_converge():
    # test discrete_mmot_converge result
    A, _ = create_test_data()
    d = 2
    niters = 10
    result = ot.lp.discrete_mmot_converge(A, niters, 0.001, 5)
    
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
