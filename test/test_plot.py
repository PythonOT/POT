

import numpy as np
import matplotlib
matplotlib.use('Agg')


def test_plot1D_mat():

    import ot

    n = 100  # nb bins

    # bin positions
    x = np.arange(n, dtype=np.float64)

    # Gaussian distributions
    a = ot.datasets.get_1D_gauss(n, m=20, s=5)  # m= mean, s= std
    b = ot.datasets.get_1D_gauss(n, m=60, s=10)

    # loss matrix
    M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
    M /= M.max()

    ot.plot.plot1D_mat(a, b, M, 'Cost matrix M')


def test_plot2D_samples_mat():

    import ot

    n = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([4, 4])
    cov_t = np.array([[1, -.8], [-.8, 1]])

    xs = ot.datasets.get_2D_samples_gauss(n, mu_s, cov_s)
    xt = ot.datasets.get_2D_samples_gauss(n, mu_t, cov_t)

    G = 1.0 * (np.random.rand(n, n) < 0.01)

    ot.plot.plot2D_samples_mat(xs, xt, G, thr=1e-5)
