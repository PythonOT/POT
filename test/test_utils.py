"""Tests for module utils for timing and parallel computation """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import ot
import numpy as np
import sys
import pytest


def test_proj_simplex(nx):
    n = 10
    rng = np.random.RandomState(0)

    # test on matrix when projection is done on axis 0
    x = rng.randn(n, 2)
    x1 = nx.from_numpy(x)

    # all projections should sum to 1
    proj = ot.utils.proj_simplex(x1)
    l1 = np.sum(nx.to_numpy(proj), axis=0)
    l2 = np.ones(2)
    np.testing.assert_allclose(l1, l2, atol=1e-5)

    # all projections should sum to 3
    proj = ot.utils.proj_simplex(x1, 3)
    l1 = np.sum(nx.to_numpy(proj), axis=0)
    l2 = 3 * np.ones(2)
    np.testing.assert_allclose(l1, l2, atol=1e-5)

    # tets on vector
    x = rng.randn(n)
    x1 = nx.from_numpy(x)

    # all projections should sum to 1
    proj = ot.utils.proj_simplex(x1)
    l1 = np.sum(nx.to_numpy(proj), axis=0)
    l2 = np.ones(2)
    np.testing.assert_allclose(l1, l2, atol=1e-5)


def test_parmap():

    n = 10

    def f(i):
        return 1.0 * i * i

    a = np.arange(n)

    l1 = list(map(f, a))

    l2 = list(ot.utils.parmap(f, a))

    np.testing.assert_allclose(l1, l2)


def test_tic_toc():

    import time

    ot.tic()
    time.sleep(0.1)
    t = ot.toc()
    t2 = ot.toq()

    # test timing
    # np.testing.assert_allclose(0.1, t, rtol=1e-1, atol=1e-1)
    # very slow macos github action equality not possible
    assert t > 0.09

    # test toc vs toq
    np.testing.assert_allclose(t, t2, rtol=1e-1, atol=1e-1)


def test_kernel():

    n = 100
    rng = np.random.RandomState(0)
    x = rng.randn(n, 2)

    K = ot.utils.kernel(x, x)

    # gaussian kernel  has ones on the diagonal
    np.testing.assert_allclose(np.diag(K), np.ones(n))


def test_unif():

    n = 100

    u = ot.unif(n)

    np.testing.assert_allclose(1, np.sum(u))


def test_unif_backend(nx):

    n = 100

    for tp in nx.__type_list__:
        print(nx.dtype_device(tp))

        u = ot.unif(n, type_as=tp)

        np.testing.assert_allclose(1, np.sum(nx.to_numpy(u)), atol=1e-6)


def test_dist():

    n = 10

    rng = np.random.RandomState(0)
    x = rng.randn(n, 2)

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sum(np.square(x[i, :] - x[j, :]))

    D2 = ot.dist(x, x)
    D3 = ot.dist(x)

    D4 = ot.dist(x, x, metric='minkowski', p=2)

    assert D4[0, 1] == D4[1, 0]

    # dist shoul return squared euclidean
    np.testing.assert_allclose(D, D2, atol=1e-14)
    np.testing.assert_allclose(D, D3, atol=1e-14)

    # tests that every metric runs correctly
    metrics_w = [
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
        'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
    ]  # those that support weights
    metrics = ['mahalanobis', 'seuclidean']  # do not support weights depending on scipy's version

    for metric in metrics_w:
        print(metric)
        ot.dist(x, x, metric=metric, p=3, w=np.random.random((2, )))
    for metric in metrics:
        print(metric)
        ot.dist(x, x, metric=metric, p=3)

    # weighted minkowski but with no weights
    with pytest.raises(ValueError):
        ot.dist(x, x, metric="wminkowski")


def test_dist_backends(nx):

    n = 100
    rng = np.random.RandomState(0)
    x = rng.randn(n, 2)
    x1 = nx.from_numpy(x)

    lst_metric = ['euclidean', 'sqeuclidean']

    for metric in lst_metric:

        D = ot.dist(x, x, metric=metric)
        D1 = ot.dist(x1, x1, metric=metric)

        # low atol because jax forces float32
        np.testing.assert_allclose(D, nx.to_numpy(D1), atol=1e-5)


def test_dist0():

    n = 100
    M = ot.utils.dist0(n, method='lin_square')

    # dist0 default to linear sampling with quadratic loss
    np.testing.assert_allclose(M[0, -1], (n - 1) * (n - 1))


def test_dots():

    n1, n2, n3, n4 = 100, 50, 200, 100

    rng = np.random.RandomState(0)

    A = rng.randn(n1, n2)
    B = rng.randn(n2, n3)
    C = rng.randn(n3, n4)

    X1 = ot.utils.dots(A, B, C)

    X2 = A.dot(B.dot(C))

    np.testing.assert_allclose(X1, X2)


def test_clean_zeros():

    n = 100
    nz = 50
    nz2 = 20
    u1 = ot.unif(n)
    u1[:nz] = 0
    u1 = u1 / u1.sum()
    u2 = ot.unif(n)
    u2[:nz2] = 0
    u2 = u2 / u2.sum()

    M = ot.utils.dist0(n)

    a, b, M2 = ot.utils.clean_zeros(u1, u2, M)

    assert len(a) == n - nz
    assert len(b) == n - nz2


def test_cost_normalization():

    C = np.random.rand(10, 10)

    # does nothing
    M0 = ot.utils.cost_normalization(C)
    np.testing.assert_allclose(C, M0)

    M = ot.utils.cost_normalization(C, 'median')
    np.testing.assert_allclose(np.median(M), 1)

    M = ot.utils.cost_normalization(C, 'max')
    np.testing.assert_allclose(M.max(), 1)

    M = ot.utils.cost_normalization(C, 'log')
    np.testing.assert_allclose(M.max(), np.log(1 + C).max())

    M = ot.utils.cost_normalization(C, 'loglog')
    np.testing.assert_allclose(M.max(), np.log(1 + np.log(1 + C)).max())


def test_check_params():

    res1 = ot.utils.check_params(first='OK', second=20)
    assert res1 is True

    res0 = ot.utils.check_params(first='OK', second=None)
    assert res0 is False


def test_deprecated_func():

    @ot.utils.deprecated('deprecated text for fun')
    def fun():
        pass

    def fun2():
        pass

    @ot.utils.deprecated('deprecated text for class')
    class Class():
        pass

    with pytest.warns(DeprecationWarning):
        fun()

    with pytest.warns(DeprecationWarning):
        cl = Class()
        print(cl)

    if sys.version_info < (3, 5):
        print('Not tested')
    else:
        assert ot.utils._is_deprecated(fun) is True

        assert ot.utils._is_deprecated(fun2) is False


def test_BaseEstimator():

    class Class(ot.utils.BaseEstimator):

        def __init__(self, first='spam', second='eggs'):

            self.first = first
            self.second = second

    cl = Class()

    names = cl._get_param_names()
    assert 'first' in names
    assert 'second' in names

    params = cl.get_params()
    assert 'first' in params
    assert 'second' in params

    params['first'] = 'spam again'
    cl.set_params(**params)

    with pytest.raises(ValueError):
        cl.set_params(bibi=10)

    assert cl.first == 'spam again'
