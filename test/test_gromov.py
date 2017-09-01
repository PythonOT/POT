"""Tests for module gromov  """

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
import ot


def test_gromov():
<<<<<<< HEAD
    n_samples = 50  # nb samples
=======
    n = 50  # nb samples
>>>>>>> 986f46ddde3ce2f550cb56f66620df377326423d

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

<<<<<<< HEAD
    xs = ot.datasets.get_2D_samples_gauss(n_samples, mu_s, cov_s)

    xt = [xs[n_samples - (i + 1)] for i in range(n_samples)]
    xt = np.array(xt)

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
=======
    xs = ot.datasets.get_2D_samples_gauss(n, mu_s, cov_s)

    xt = [xs[n - (i + 1)] for i in range(n)]
    xt = np.array(xt)

    p = ot.unif(n)
    q = ot.unif(n)
>>>>>>> 986f46ddde3ce2f550cb56f66620df377326423d

    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    G = ot.gromov_wasserstein(C1, C2, p, q, 'square_loss', epsilon=5e-4)

    # check constratints
    np.testing.assert_allclose(
        p, G.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(
        q, G.sum(0), atol=1e-04)  # cf convergence gromov
