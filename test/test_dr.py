"""Tests for module dr on Dimensionality Reduction """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import ot
import pytest

try:  # test if autograd and pymanopt are installed
    import ot.dr
    nogo = False
except ImportError:
    nogo = True


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_fda():

    n_samples = 90  # nb samples in source and target datasets
    np.random.seed(0)

    # generate gaussian dataset
    xs, ys = ot.datasets.make_data_classif('gaussrot', n_samples)

    n_features_noise = 8

    xs = np.hstack((xs, np.random.randn(n_samples, n_features_noise)))

    p = 1

    Pfda, projfda = ot.dr.fda(xs, ys, p)

    projfda(xs)

    np.testing.assert_allclose(np.sum(Pfda**2, 0), np.ones(p))


@pytest.mark.skipif(nogo, reason="Missing modules (autograd or pymanopt)")
def test_wda():

    n_samples = 100  # nb samples in source and target datasets
    np.random.seed(0)

    # generate gaussian dataset
    xs, ys = ot.datasets.make_data_classif('gaussrot', n_samples)

    n_features_noise = 8

    xs = np.hstack((xs, np.random.randn(n_samples, n_features_noise)))

    p = 2

    Pwda, projwda = ot.dr.wda(xs, ys, p, maxiter=10)

    projwda(xs)

    np.testing.assert_allclose(np.sum(Pwda**2, 0), np.ones(p))
