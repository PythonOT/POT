
import numpy as np
import ot


# import pytest


def test_OTDA():

    n = 150  # nb bins

    xs, ys = ot.datasets.get_data_classif('3gauss', n)
    xt, yt = ot.datasets.get_data_classif('3gauss2', n)

    a, b = ot.unif(n), ot.unif(n)

    # LP problem
    da_emd = ot.da.OTDA()     # init class
    da_emd.fit(xs, xt)       # fit distributions
    da_emd.interp()    # interpolation of source samples
    da_emd.predict(xs)    # interpolation of source samples

    assert np.allclose(a, np.sum(da_emd.G, 1))
    assert np.allclose(b, np.sum(da_emd.G, 0))

    # sinkhorn regularization
    lambd = 1e-1
    da_entrop = ot.da.OTDA_sinkhorn()
    da_entrop.fit(xs, xt, reg=lambd)
    da_entrop.interp()
    da_entrop.predict(xs)

    assert np.allclose(a, np.sum(da_entrop.G, 1), rtol=1e-3, atol=1e-3)
    assert np.allclose(b, np.sum(da_entrop.G, 0), rtol=1e-3, atol=1e-3)

    # non-convex Group lasso regularization
    reg = 1e-1
    eta = 1e0
    da_lpl1 = ot.da.OTDA_lpl1()
    da_lpl1.fit(xs, ys, xt, reg=reg, eta=eta)
    da_lpl1.interp()
    da_lpl1.predict(xs)

    assert np.allclose(a, np.sum(da_lpl1.G, 1), rtol=1e-3, atol=1e-3)
    assert np.allclose(b, np.sum(da_lpl1.G, 0), rtol=1e-3, atol=1e-3)

    # True Group lasso regularization
    reg = 1e-1
    eta = 2e0
    da_l1l2 = ot.da.OTDA_l1l2()
    da_l1l2.fit(xs, ys, xt, reg=reg, eta=eta, numItermax=20, verbose=True)
    da_l1l2.interp()
    da_l1l2.predict(xs)

    assert np.allclose(a, np.sum(da_l1l2.G, 1), rtol=1e-3, atol=1e-3)
    assert np.allclose(b, np.sum(da_l1l2.G, 0), rtol=1e-3, atol=1e-3)

    # linear mapping
    da_emd = ot.da.OTDA_mapping_linear()     # init class
    da_emd.fit(xs, xt, numItermax=10)       # fit distributions
    da_emd.predict(xs)    # interpolation of source samples

    # nonlinear mapping
    da_emd = ot.da.OTDA_mapping_kernel()     # init class
    da_emd.fit(xs, xt, numItermax=10)       # fit distributions
    da_emd.predict(xs)    # interpolation of source samples
