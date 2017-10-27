"""Tests for module plot for visualization """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import matplotlib

from ot.datasets import get_1D_gauss, get_2D_samples_gauss
from ot.utils import dist
from ot.plot import plot1D_mat, plot2D_samples_mat

matplotlib.use('Agg')


def test_plot1D_mat():

    n_bins = 100  # nb bins

    # bin positions
    x = np.arange(n_bins, dtype=np.float64)

    # Gaussian distributions
    a = get_1D_gauss(n_bins, m=20, s=5)  # m= mean, s= std
    b = get_1D_gauss(n_bins, m=60, s=10)

    # loss matrix
    M = dist(x.reshape((n_bins, 1)), x.reshape((n_bins, 1)))
    M /= M.max()

    plot1D_mat(a, b, M, 'Cost matrix M')


def test_plot2D_samples_mat():

    n_bins = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([4, 4])
    cov_t = np.array([[1, -.8], [-.8, 1]])

    xs = get_2D_samples_gauss(n_bins, mu_s, cov_s)
    xt = get_2D_samples_gauss(n_bins, mu_t, cov_t)

    G = 1.0 * (np.random.rand(n_bins, n_bins) < 0.01)

    plot2D_samples_mat(xs, xt, G, thr=1e-5)
