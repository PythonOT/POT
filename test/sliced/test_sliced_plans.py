"""Tests for module sliced_plans"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty <ncourty@irisa.fr>
#         Eloi Tanguy <eloi.tanguy@math.cnrs.fr>
#         Laetitia Chapel <laetitia.chapel@irisa.fr>
#
# License: MIT License

import numpy as np
import pytest

import ot
from ot.sliced import get_random_projections
from ot.backend import tf, torch


def test_sliced_permutations():
    n = 4
    n_projections = 10
    d = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(n, 2)

    thetas = ot.sliced.get_random_projections(d, n_projections, seed=0).T

    # test without provided thetas
    _, _ = ot.sliced.sliced_plans(x, y, n_projections=n_projections)

    # test with invalid shapes
    with pytest.raises(AssertionError):
        ot.sliced.sliced_plans(x[:, 1:], y, thetas=thetas)


def test_sliced_plans():
    n = 4
    m = 5
    n_projections = 10
    d = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(m, 2)

    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, m)
    b /= b.sum()

    thetas = ot.sliced.get_random_projections(d, n_projections, seed=0).T

    # test with a and b not uniform
    ot.sliced.sliced_plans(x, y, a, b, thetas=thetas)

    # test with the minkowski metric
    ot.sliced.sliced_plans(x, y, thetas=thetas, metric="minkowski")

    # test with an unsupported metric
    with pytest.raises(AssertionError):
        ot.sliced.sliced_plans(x, y, thetas=thetas, metric="mahalanobis")

    # test with a batch size
    ot.sliced.sliced_plans(x, y, a, b, thetas=thetas, batch_size=2)

    # test permutations
    n = 5
    m = 5
    n_projections = 10
    d = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(m, 2)

    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, m)
    b /= b.sum()

    # test with the minkowski metric
    ot.sliced.sliced_plans(x, y, n_projections=10, metric="minkowski")


def test_min_pivot_sliced():
    n = 10
    m = 4
    n_projections = 10
    d = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(m, 2)
    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, m)
    b /= b.sum()

    thetas = ot.sliced.get_random_projections(d, n_projections, seed=0).T

    # identity of the indiscernibles
    _, min_cost = ot.sliced.min_pivot_sliced(x, x, a, a, n_projections=10)
    np.testing.assert_almost_equal(min_cost, 0.0)

    _, min_cost = ot.sliced.min_pivot_sliced(x, y, a, b, thetas=thetas, dense=True)

    # result should be an upper-bound of W2 and relatively close
    w2 = ot.emd2(a, b, ot.dist(x, y))
    assert min_cost >= w2
    assert min_cost <= 1.5 * w2

    # test without provided thetas
    ot.sliced.min_pivot_sliced(x, y, a, b, n_projections=n_projections, log=True)

    # test with invalid shapes
    with pytest.raises(AssertionError):
        ot.sliced.min_pivot_sliced(x[:, 1:], y, thetas=thetas)

    # test the logs
    _, min_cost, log = ot.sliced.min_pivot_sliced(
        x, y, a, b, thetas=thetas, dense=False, log=True
    )
    assert len(log) == 5
    costs = log["costs"]
    assert len(costs) == thetas.shape[0]
    assert len(log["min_theta"]) == d
    assert (log["thetas"] == thetas).all()
    for c in costs:
        assert c > 0

    # test with different metrics
    ot.sliced.min_pivot_sliced(x, y, thetas=thetas, metric="minkowski")
    ot.sliced.min_pivot_sliced(x, y, thetas=thetas, metric="euclidean")
    ot.sliced.min_pivot_sliced(x, y, thetas=thetas, metric="cityblock")

    # test with an unsupported metric
    with pytest.raises(AssertionError):
        ot.sliced.min_pivot_sliced(x, y, thetas=thetas, metric="mahalanobis")

    # test with a batch size
    ot.sliced.min_pivot_sliced(x, y, a, b, thetas=thetas, batch_size=2)


def test_expected_sliced():
    n = 10
    m = 24
    n_projections = 10
    d = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(m, 2)
    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, m)
    b /= b.sum()

    thetas = ot.sliced.get_random_projections(d, n_projections, seed=0).T

    _, expected_cost = ot.sliced.expected_sliced(x, y, a, b, dense=True, thetas=thetas)
    # result should be a coarse upper-bound of W2
    w2 = ot.emd2(a, b, ot.dist(x, y))
    assert expected_cost >= w2
    assert expected_cost <= 3 * w2

    # test without provided thetas
    ot.sliced.expected_sliced(x, y, n_projections=n_projections, log=True)
    ot.sliced.expected_sliced(x, y, a, b, n_projections=n_projections, log=True)

    # test with invalid shapes
    with pytest.raises(AssertionError):
        ot.sliced.min_pivot_sliced(x[:, 1:], y, thetas=thetas)

    # with a small temperature (i.e. large beta), the cost should be close
    # to min_pivot
    _, expected_cost = ot.sliced.expected_sliced(
        x, y, a, b, thetas=thetas, dense=True, beta=1000.0
    )
    _, min_cost = ot.sliced.min_pivot_sliced(x, y, a, b, thetas=thetas, dense=True)
    np.testing.assert_almost_equal(expected_cost, min_cost, decimal=3)

    # test the logs
    _, min_cost, log = ot.sliced.expected_sliced(
        x, y, a, b, thetas=thetas, dense=False, log=True
    )
    assert len(log) == 4
    costs = log["costs"]
    assert len(costs) == thetas.shape[0]
    assert len(log["weights"]) == thetas.shape[0]
    assert (log["thetas"] == thetas).all()
    for c in costs:
        assert c > 0

    # test with the minkowski metric
    ot.sliced.expected_sliced(x, y, thetas=thetas, metric="minkowski")

    # test with an unsupported metric
    with pytest.raises(AssertionError):
        ot.sliced.expected_sliced(x, y, thetas=thetas, metric="mahalanobis")

    # test with a batch size
    ot.sliced.expected_sliced(x, y, a, b, thetas=thetas, batch_size=2)


def test_sliced_plans_backends(nx):
    n = 10
    m = 24
    n_projections = 10
    d = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    y = rng.randn(m, 2)
    a = rng.uniform(0, 1, n)
    a /= a.sum()
    b = rng.uniform(0, 1, m)
    b /= b.sum()

    x_b, y_b, a_b, b_b = nx.from_numpy(x, y, a, b)

    thetas_b = ot.sliced.get_random_projections(
        d, n_projections, seed=0, backend=nx, type_as=x_b
    ).T
    thetas = nx.to_numpy(thetas_b)

    _, expected_cost_b = ot.sliced.expected_sliced(
        x_b, y_b, a_b, b_b, dense=True, thetas=thetas_b
    )
    # result should be the same than numpy version
    _, expected_cost = ot.sliced.expected_sliced(x, y, a, b, dense=True, thetas=thetas)
    np.testing.assert_almost_equal(expected_cost_b, expected_cost)

    # for min_pivot
    _, min_cost_b = ot.sliced.min_pivot_sliced(
        x_b, y_b, a_b, b_b, dense=True, thetas=thetas_b
    )
    # result should be the same than numpy version
    _, min_cost = ot.sliced.min_pivot_sliced(x, y, a, b, dense=True, thetas=thetas)
    np.testing.assert_almost_equal(min_cost_b, min_cost)

    # for thetas
    thetas_b = ot.sliced.get_random_projections(
        d, n_projections, seed=0, backend=nx, type_as=x_b
    ).T

    # test with the minkowski metric
    ot.sliced.min_pivot_sliced(x_b, y_b, thetas=thetas_b, metric="minkowski")

    # test with a batch size
    ot.sliced.min_pivot_sliced(x_b, y_b, a_b, b_b, thetas=thetas_b, batch_size=2)
