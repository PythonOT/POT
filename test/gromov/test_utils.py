""" Tests for gromov._utils.py """

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

import numpy as np
import pytest

import ot


def test_update_barycenter(nx):
    ns = 5
    nt = 10

    Xs, ys = ot.datasets.make_data_classif('3gauss', ns, random_state=42)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', nt, random_state=42)

    rng = np.random.RandomState(42)
    ys = rng.randn(Xs.shape[0], 2)
    yt = rng.randn(Xt.shape[0], 2)

    C1 = ot.dist(Xs)
    C2 = ot.dist(Xt)
    C1 /= C1.max()
    C2 /= C2.max()

    p1, p2 = ot.unif(ns), ot.unif(nt)
    n_samples = 3

    ysb, ytb, C1b, C2b, p1b, p2b = nx.from_numpy(ys, yt, C1, C2, p1, p2)

    lambdas = [.5, .5]
    Csb = [C1b, C2b]
    Ysb = [ysb, ytb]

    Tb = [nx.ones((m, n_samples), type_as=C1b) / (m * n_samples) for m in [ns, nt]]
    pb = nx.concatenate(
        [nx.sum(elem, 0)[None, :] for elem in Tb], axis=0)

    # test edge cases for the update of the barycenter with `p != None`
    # and `target=False`
    Cb = ot.gromov.update_barycenter_structure(
        [elem.T for elem in Tb], Csb, lambdas, pb, target=False)
    Xb = ot.gromov.update_barycenter_feature(
        [elem.T for elem in Tb], Ysb, lambdas, pb, target=False)

    Cbt = ot.gromov.update_barycenter_structure(
        Tb, Csb, lambdas, None, target=True)
    Xbt = ot.gromov.update_barycenter_feature(
        Tb, Ysb, lambdas, None, target=True)

    np.testing.assert_allclose(Cb, Cbt)
    np.testing.assert_allclose(Xb, Xbt)

    # test not supported metrics
    with pytest.raises(ValueError):
        Cbt = ot.gromov.update_barycenter_structure(
            Tb, Csb, lambdas, None, loss_fun='unknown', target=True)
    with pytest.raises(ValueError):
        Xbt = ot.gromov.update_barycenter_feature(
            Tb, Ysb, lambdas, None, loss_fun='unknown', target=True)
