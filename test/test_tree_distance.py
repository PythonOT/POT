"""Tests for the tree wassesterin distance"""

import pytest
import numpy as np

import ot
from ot.lp import tree_wasserstein_distance
from ot.lp import wasserstein_1d


def test_symetry_reflexivity(nx):
    n = 50
    mu = np.array([0.0, 0.0])
    sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
    points_np = ot.datasets.make_2D_samples_gauss(n, mu, sigma)
    points = nx.from_numpy(points_np)

    tree, length = ot.utils.random_tree(points)

    u = nx.rand(n)
    u = u / nx.sum(u)

    v = nx.rand(n)
    v = v / nx.sum(v)

    np.testing.assert_almost_equal(
        tree_wasserstein_distance(tree, length, u, v),
        tree_wasserstein_distance(tree, length, v, u),
    )

    np.testing.assert_almost_equal(tree_wasserstein_distance(tree, length, u, u), 0)


def test_chain(nx):
    n = 50
    mu = 0
    sigma = 1

    points_np = np.random.normal(mu, sigma, n)
    points_np = np.sort(points_np)
    points = nx.from_numpy(points_np)

    tree = nx.arange(n - 1, -1)
    tree[0] = 0

    length = nx.zeros(n)

    for i in range(1, n):
        length[i] = points[i] - points[i - 1]

    u = nx.rand(n)
    u = u / nx.sum(u)

    v = nx.rand(n)
    v = v / nx.sum(v)

    np.testing.assert_almost_equal(
        tree_wasserstein_distance(tree, length, u, v),
        wasserstein_1d(points, points, u, v),
        decimal=4,
    )
