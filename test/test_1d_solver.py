"""Tests for module 1d Wasserstein solver"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
import pytest

from ot.lp import wasserstein_1d

from ot.backend import get_backend_list

backend_list = get_backend_list()


@pytest.mark.parametrize('nx', backend_list)
def test_wasserstein_1d(nx):
    from scipy.stats import wasserstein_distance

    rng = np.random.RandomState(0)

    n = 100
    x = np.linspace(0, 5, n)
    rho_u = np.abs(rng.randn(n))
    rho_u /= rho_u.sum()
    rho_v = np.abs(rng.randn(n))
    rho_v /= rho_v.sum()

    xb = nx.from_numpy(x)
    rho_ub = nx.from_numpy(rho_u)
    rho_vb = nx.from_numpy(rho_v)

    # test 1 : wasserstein_1d should be close to scipy W_1 implementation
    np.testing.assert_almost_equal(wasserstein_1d(xb, xb, rho_ub, rho_vb, p=1),
                                   wasserstein_distance(x, x, rho_u, rho_v))

    # test 2 : wasserstein_1d should be close to one when only translating the support
    np.testing.assert_almost_equal(wasserstein_1d(xb, xb + 1, p=2),
                                   1.)

    # test 3 : arrays test
    X = np.stack((np.linspace(0, 5, n), np.linspace(0, 5, n) * 10), -1)
    Xb = nx.from_numpy(X)
    res = wasserstein_1d(Xb, Xb, rho_ub, rho_vb, p=2)
    np.testing.assert_almost_equal(100 * res[0], res[1], decimal=4)
