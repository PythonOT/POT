"""Tests for module da on Domain Adaptation """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import ot


def test_otda():

    n_samples = 150  # nb samples
    np.random.seed(0)

    xs, ys = ot.datasets.get_data_classif('3gauss', n_samples)
    xt, yt = ot.datasets.get_data_classif('3gauss2', n_samples)

    a, b = ot.unif(n_samples), ot.unif(n_samples)

    # LP problem
    da_emd = ot.da.OTDA()     # init class
    da_emd.fit(xs, xt)       # fit distributions
    da_emd.interp()    # interpolation of source samples
    da_emd.predict(xs)    # interpolation of source samples

    np.testing.assert_allclose(a, np.sum(da_emd.G, 1))
    np.testing.assert_allclose(b, np.sum(da_emd.G, 0))

    # sinkhorn regularization
    lambd = 1e-1
    da_entrop = ot.da.OTDA_sinkhorn()
    da_entrop.fit(xs, xt, reg=lambd)
    da_entrop.interp()
    da_entrop.predict(xs)

    np.testing.assert_allclose(a, np.sum(da_entrop.G, 1), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(b, np.sum(da_entrop.G, 0), rtol=1e-3, atol=1e-3)

    # non-convex Group lasso regularization
    reg = 1e-1
    eta = 1e0
    da_lpl1 = ot.da.OTDA_lpl1()
    da_lpl1.fit(xs, ys, xt, reg=reg, eta=eta)
    da_lpl1.interp()
    da_lpl1.predict(xs)

    np.testing.assert_allclose(a, np.sum(da_lpl1.G, 1), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(b, np.sum(da_lpl1.G, 0), rtol=1e-3, atol=1e-3)

    # True Group lasso regularization
    reg = 1e-1
    eta = 2e0
    da_l1l2 = ot.da.OTDA_l1l2()
    da_l1l2.fit(xs, ys, xt, reg=reg, eta=eta, numItermax=20, verbose=True)
    da_l1l2.interp()
    da_l1l2.predict(xs)

    np.testing.assert_allclose(a, np.sum(da_l1l2.G, 1), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(b, np.sum(da_l1l2.G, 0), rtol=1e-3, atol=1e-3)

    # linear mapping
    da_emd = ot.da.OTDA_mapping_linear()     # init class
    da_emd.fit(xs, xt, numItermax=10)       # fit distributions
    da_emd.predict(xs)    # interpolation of source samples

    # nonlinear mapping
    da_emd = ot.da.OTDA_mapping_kernel()     # init class
    da_emd.fit(xs, xt, numItermax=10)       # fit distributions
    da_emd.predict(xs)    # interpolation of source samples
