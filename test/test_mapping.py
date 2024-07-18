"""Tests for module mapping"""
# Author: Eloi Tanguy <eloi.tanguy@u-paris.fr>
#
# License: MIT License

import numpy as np
import ot
import pytest
from ot.backend import to_numpy


try:  # test if cvxpy is installed
    import cvxpy  # noqa: F401

    nocvxpy = False
except ImportError:
    nocvxpy = True


@pytest.mark.skipif(nocvxpy, reason="No CVXPY available")
def test_ssnb_qcqp_constants():
    c1, c2, c3 = ot.mapping._ssnb_qcqp_constants(0.5, 1)
    np.testing.assert_almost_equal(c1, 1)
    np.testing.assert_almost_equal(c2, 0.5)
    np.testing.assert_almost_equal(c3, 1)


@pytest.mark.skipif(nocvxpy, reason="No CVXPY available")
def test_nearest_brenier_potential_fit(nx):
    X = nx.ones((2, 2))
    phi, G, log = ot.mapping.nearest_brenier_potential_fit(X, X, its=3, log=True)
    np.testing.assert_almost_equal(
        to_numpy(G), to_numpy(X)
    )  # image of source should be close to target
    # test without log but with X_classes, a, b and other init method
    a = nx.ones(2) / 2
    ot.mapping.nearest_brenier_potential_fit(
        X, X, X_classes=nx.ones(2), a=a, b=a, its=1, init_method="target"
    )


@pytest.mark.skipif(nocvxpy, reason="No CVXPY available")
def test_brenier_potential_predict_bounds(nx):
    X = nx.ones((2, 2))
    phi, G = ot.mapping.nearest_brenier_potential_fit(X, X, its=3)
    phi_lu, G_lu, log = ot.mapping.nearest_brenier_potential_predict_bounds(
        X, phi, G, X, log=True
    )
    # 'new' input isn't new, so should be equal to target
    np.testing.assert_almost_equal(to_numpy(G_lu[0]), to_numpy(X))
    np.testing.assert_almost_equal(to_numpy(G_lu[1]), to_numpy(X))
    # test with no log but classes
    ot.mapping.nearest_brenier_potential_predict_bounds(
        X, phi, G, X, X_classes=nx.ones(2), Y_classes=nx.ones(2)
    )


def test_joint_OT_mapping():
    """
    Complements the tests in test_da, for verbose, log and bias options
    """
    xs = np.array([[0.1, 0.2], [-0.1, 0.3]])
    ot.mapping.joint_OT_mapping_kernel(xs, xs, verbose=True)
    ot.mapping.joint_OT_mapping_linear(xs, xs, verbose=True)
    ot.mapping.joint_OT_mapping_kernel(xs, xs, log=True, bias=True)
    ot.mapping.joint_OT_mapping_linear(xs, xs, log=True, bias=True)
