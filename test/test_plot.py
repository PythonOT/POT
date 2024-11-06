"""Tests for module plot for visualization"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import pytest


try:  # test if matplotlib is installed
    import matplotlib

    matplotlib.use("Agg")
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

    ot.plot.plot1D_mat(a, b, M)
    ot.plot.plot1D_mat(a, b, M, plot_style="xy")

    with pytest.raises(AssertionError):
        ot.plot.plot1D_mat(a, b, M, plot_style="NotAValidStyle")


@pytest.mark.skipif(nogo, reason="Matplotlib not installed")
def test_rescale_for_imshow_plot():
    import ot
    import ot.plot

    n = 7
    a_x, b_x = -1, 3
    x = np.linspace(a_x, b_x, n)
    a_y, b_y = 2, 6
    y = np.linspace(a_y, b_y, n)

    x_rescaled, y_rescaled = ot.plot.rescale_for_imshow_plot(x, y, n)
    assert x_rescaled.shape == (n,)
    assert y_rescaled.shape == (n,)

    x_rescaled, y_rescaled = ot.plot.rescale_for_imshow_plot(
        x, y, n, m=n, a_y=a_y + 1, b_y=b_y - 1
    )
    assert x_rescaled.shape[0] <= n
    assert y_rescaled.shape[0] <= n
    with pytest.raises(AssertionError):
        ot.plot.rescale_for_imshow_plot(x[3:], y, n)


@pytest.mark.skipif(nogo, reason="Matplotlib not installed")
def test_plot2D_samples_mat():
    import ot
    import ot.plot

    n_bins = 50  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([4, 4])
    cov_t = np.array([[1, -0.8], [-0.8, 1]])

    rng = np.random.RandomState(42)
    xs = ot.datasets.make_2D_samples_gauss(n_bins, mu_s, cov_s, random_state=rng)
    xt = ot.datasets.make_2D_samples_gauss(n_bins, mu_t, cov_t, random_state=rng)

    G = 1.0 * (rng.rand(n_bins, n_bins) < 0.01)

    ot.plot.plot2D_samples_mat(xs, xt, G, thr=1e-5)
    ot.plot.plot2D_samples_mat(xs, xt, G, thr=1e-5, alpha=0.5)
