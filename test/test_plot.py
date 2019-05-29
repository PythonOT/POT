"""Tests for module plot for visualization """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import pytest


try:  # test if matplotlib is installed
    import matplotlib
    matplotlib.use('Agg')
    nogo = False
except ImportError:
    nogo = True


@pytest.mark.skipif(nogo, reason="Matplotlib not installed")
def test_plot1D_mat():

    import ot
    import ot.plot

    n_bins = 100  # nb bins

    # bin positions
    x = np.arange(n_bins, dtype=np.float64)

    # Gaussian distributions
    a = ot.datasets.make_1D_gauss(n_bins, m=20, s=5)  # m= mean, s= std
    b = ot.datasets.make_1D_gauss(n_bins, m=60, s=10)

    # loss matrix
    M = ot.dist(x.reshape((n_bins, 1)), x.reshape((n_bins, 1)))
    M /= M.max()

    ot.plot.plot1D_mat(a, b, M, 'Cost matrix M')


@pytest.mark.skipif(nogo, reason="Matplotlib not installed")
def test_plot2D_samples_mat():

    import ot
    import ot.plot

    n_bins = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([4, 4])
    cov_t = np.array([[1, -.8], [-.8, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_bins, mu_s, cov_s)
    xt = ot.datasets.make_2D_samples_gauss(n_bins, mu_t, cov_t)

    G = 1.0 * (np.random.rand(n_bins, n_bins) < 0.01)

    ot.plot.plot2D_samples_mat(xs, xt, G, thr=1e-5)
