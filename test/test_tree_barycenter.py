"""Tests for the tree wassesterin barycenter"""

import numpy as np
import pytest

import ot
from ot.lp import fixed_support_tree_barycenter
from ot.utils import list_to_array
from ot.lp import free_support_tree_barycenter


def test_fixed_symetry_reflexivity(nx):
    n = 50
    k = 30

    tree, length = ot.utils.random_tree_fixed_leaves(n, k, nx, 0)

    u = nx.rand(k)
    u = u / nx.sum(u)

    v = nx.rand(k)
    v = v / nx.sum(v)

    np.testing.assert_allclose(
        fixed_support_tree_barycenter(tree, length, nx.stack([u, v])),
        fixed_support_tree_barycenter(tree, length, nx.stack([v, u])),
        rtol=1e-2,
        atol=1e-3,
    )

    np.testing.assert_allclose(
        fixed_support_tree_barycenter(
            tree, length, nx.stack([u, u]), nb_itr=100, step=0.01
        ),
        u,
        rtol=5.0,
        atol=1e-3,
    )


def test_fixed_multiple_trees(nx):
    n = 40
    k = 20

    nb_trees = 5

    data = [ot.utils.random_tree_fixed_leaves(n, k, nx, 41) for _ in range(nb_trees)]
    trees, lengths = map(list, zip(*data))

    trees = list_to_array(trees, nx=nx)
    lengths = list_to_array(lengths, nx=nx)

    u = nx.rand(k)
    u = u / nx.sum(u)

    np.testing.assert_allclose(
        fixed_support_tree_barycenter(
            trees, lengths, nx.stack([u, u]), nb_itr=1000, step=0.001
        ),
        u,
        rtol=1e-3,
        atol=1e-3,
    )


def test_free_reflexivity_symetry(nx):
    torch = pytest.importorskip("torch")

    n = 10
    mu = np.array([0.0, 0.0])
    sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
    points_np = ot.datasets.make_2D_samples_gauss(n, mu, sigma)
    points = nx.from_numpy(points_np)

    tree, length = ot.utils.random_tree(points, 0)
    length *= 100

    u = nx.rand(n)
    u = u / nx.sum(u)

    v = nx.rand(n)
    v = v / nx.sum(v)

    np.testing.assert_allclose(
        free_support_tree_barycenter(tree, length, nx.stack([u, v])),
        free_support_tree_barycenter(tree, length, nx.stack([v, u])),
        rtol=1e-2,
        atol=1e-3,
    )

    np.testing.assert_allclose(
        free_support_tree_barycenter(
            tree, length, nx.stack([u, u]), nb_itr=100, step=0.0005
        ),
        u,
        rtol=5.0,
        atol=1e-3,
    )
