"""Tests for module utils for timing and parallel computation"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import ot
import numpy as np
import sys
import pytest


def get_LazyTensor(nx):
    n1 = 100
    n2 = 200

    rng = np.random.RandomState(42)
    a = rng.rand(n1)
    a /= a.sum()
    b = rng.rand(n2)
    b /= b.sum()

    a, b = nx.from_numpy(a, b)

    def getitem(i, j, a, b):
        return a[i, None] * b[None, j]

    # create a lazy tensor
    T = ot.utils.LazyTensor((n1, n2), getitem, a=a, b=b)

    return T, a, b


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


def test_projection_sparse_simplex():
    def double_sort_projection_sparse_simplex(X, max_nz, z=1, axis=None):
        r"""This is an equivalent but less efficient version
        of ot.utils.projection_sparse_simplex, as it uses two
        sorts instead of one.
        """

        if axis == 0:
            # For each column of X, find top max_nz values and
            # their corresponding indices. This incurs a sort.
            max_nz_indices = np.argpartition(X, kth=-max_nz, axis=0)[-max_nz:]

            max_nz_values = X[max_nz_indices, np.arange(X.shape[1])]

            # Project the top max_nz values onto the simplex.
            # This incurs a second sort.
            G_nz_values = ot.smooth.projection_simplex(max_nz_values, z=z, axis=0)

            # Put the projection of max_nz_values to their original indices
            # and set all other values zero.
            G = np.zeros_like(X)
            G[max_nz_indices, np.arange(X.shape[1])] = G_nz_values
            return G
        elif axis == 1:
            return double_sort_projection_sparse_simplex(X.T, max_nz, z, axis=0).T

        else:
            X = X.ravel().reshape(-1, 1)
            return double_sort_projection_sparse_simplex(X, max_nz, z, axis=0).ravel()

    m, n = 5, 10
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(m, n))
    max_nz = 3

    for axis in [0, 1, None]:
        slow_sparse_proj = double_sort_projection_sparse_simplex(X, max_nz, axis=axis)
        fast_sparse_proj = ot.utils.projection_sparse_simplex(X, max_nz, axis=axis)

        # check that two versions produce consistent results
        np.testing.assert_allclose(slow_sparse_proj, fast_sparse_proj)


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

    D4 = ot.dist(x, x, metric="minkowski", p=2)

    assert D4[0, 1] == D4[1, 0]

    # dist shoul return squared euclidean
    np.testing.assert_allclose(D, D2, atol=1e-14)
    np.testing.assert_allclose(D, D3, atol=1e-14)

    # tests that every metric runs correctly
    metrics_w = [
        "braycurtis",
        "canberra",
        "chebyshev",
        "cityblock",
        "correlation",
        "cosine",
        "dice",
        "euclidean",
        "hamming",
        "jaccard",
        "matching",
        "minkowski",
        "rogerstanimoto",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
        "sqeuclidean",
        "yule",
    ]  # those that support weights
    metrics = [
        "mahalanobis",
        "seuclidean",
    ]  # do not support weights depending on scipy's version

    for metric in metrics_w:
        print(metric)
        ot.dist(x, x, metric=metric, p=3, w=rng.random((2,)))
        ot.dist(
            x, x, metric=metric, p=3, w=None
        )  # check that not having any weight does not cause issues
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

    lst_metric = ["euclidean", "sqeuclidean"]

    for metric in lst_metric:
        D = ot.dist(x, x, metric=metric)
        D1 = ot.dist(x1, x1, metric=metric)

        # low atol because jax forces float32
        np.testing.assert_allclose(D, nx.to_numpy(D1), atol=1e-5)


def test_dist0():
    n = 100
    M = ot.utils.dist0(n, method="lin_square")

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


def test_cost_normalization(nx):
    rng = np.random.RandomState(0)

    C = rng.rand(10, 10)
    C1 = nx.from_numpy(C)

    # does nothing
    M0 = ot.utils.cost_normalization(C1)
    M1 = nx.to_numpy(M0)
    np.testing.assert_allclose(C, M1)

    M = ot.utils.cost_normalization(C1, "median")
    M1 = nx.to_numpy(M)
    np.testing.assert_allclose(np.median(M1), 1)

    M = ot.utils.cost_normalization(C1, "max")
    M1 = nx.to_numpy(M)
    np.testing.assert_allclose(M1.max(), 1)

    M = ot.utils.cost_normalization(C1, "log")
    M1 = nx.to_numpy(M)
    np.testing.assert_allclose(M1.max(), np.log(1 + C).max())

    M = ot.utils.cost_normalization(C1, "loglog")
    M1 = nx.to_numpy(M)
    np.testing.assert_allclose(M1.max(), np.log(1 + np.log(1 + C)).max())

    with pytest.raises(ValueError):
        ot.utils.cost_normalization(C1, "error")


def test_list_to_array(nx):
    lst = [np.array([1, 2, 3]), np.array([4, 5, 6])]

    a1, a2 = ot.utils.list_to_array(*lst)

    assert a1.shape == (3,)
    assert a2.shape == (3,)

    a, b, M = ot.utils.list_to_array([], [], [[1.0, 2.0], [3.0, 4.0]])


def test_check_params():
    res1 = ot.utils.check_params(first="OK", second=20)
    assert res1 is True

    res0 = ot.utils.check_params(first="OK", second=None)
    assert res0 is False


def test_check_random_state_error():
    with pytest.raises(ValueError):
        ot.utils.check_random_state("error")


def test_get_parameter_pair_error():
    with pytest.raises(ValueError):
        ot.utils.get_parameter_pair((1, 2, 3))  # not pair ;)


def test_deprecated_func():
    @ot.utils.deprecated("deprecated text for fun")
    def fun():
        pass

    def fun2():
        pass

    @ot.utils.deprecated("deprecated text for class")
    class Class:
        pass

    with pytest.warns(DeprecationWarning):
        fun()

    with pytest.warns(DeprecationWarning):
        cl = Class()
        print(cl)

    if sys.version_info < (3, 5):
        print("Not tested")
    else:
        assert ot.utils._is_deprecated(fun) is True

        assert ot.utils._is_deprecated(fun2) is False


def test_BaseEstimator():
    class Class(ot.utils.BaseEstimator):
        def __init__(self, first="spam", second="eggs"):
            self.first = first
            self.second = second

    cl = Class()

    names = cl._get_param_names()
    assert "first" in names
    assert "second" in names

    params = cl.get_params()
    assert "first" in params
    assert "second" in params

    params["first"] = "spam again"
    cl.set_params(**params)

    with pytest.raises(ValueError):
        cl.set_params(bibi=10)

    assert cl.first == "spam again"


def test_OTResult():
    res = ot.utils.OTResult()

    # test print
    print(res)

    # tets get citation
    print(res.citation)

    lst_attributes = [
        "lazy_plan",
        "marginal_a",
        "marginal_b",
        "marginals",
        "plan",
        "potential_a",
        "potential_b",
        "potentials",
        "sparse_plan",
        "status",
        "value",
        "value_linear",
        "value_quad",
        "log",
    ]
    for at in lst_attributes:
        print(at)
        assert getattr(res, at) is None

    list_not_implemented = ["a_to_b", "b_to_a"]
    for at in list_not_implemented:
        print(at)
        with pytest.raises(NotImplementedError):
            getattr(res, at)


def test_get_coordinate_circle():
    rng = np.random.RandomState(42)
    u = rng.rand(1, 100)
    x1, y1 = np.cos(u * (2 * np.pi)), np.sin(u * (2 * np.pi))
    x = np.concatenate([x1, y1]).T
    x_p = ot.utils.get_coordinate_circle(x)

    np.testing.assert_allclose(u[0], x_p)


def test_LazyTensor(nx):
    n1 = 100
    n2 = 200
    shape = (n1, n2)

    rng = np.random.RandomState(42)
    x1 = rng.randn(n1, 2)
    x2 = rng.randn(n2, 2)

    x1, x2 = nx.from_numpy(x1, x2)

    # i,j can be integers or slices, x1,x2 have to be passed as keyword arguments
    def getitem(i, j, x1, x2):
        return nx.dot(x1[i], x2[j].T)

    # create a lazy tensor
    T = ot.utils.LazyTensor((n1, n2), getitem, x1=x1, x2=x2)

    assert T.shape == (n1, n2)
    assert str(T) == "LazyTensor(shape=(100, 200),attributes=(x1,x2))"

    assert T.x1 is x1
    assert T.x2 is x2

    # get the full tensor (not lazy)
    assert T[:].shape == shape

    # get one component
    assert T[1, 1] == nx.dot(x1[1], x2[1].T)

    # get one row
    assert T[1].shape == (n2,)

    # get one column with slices
    assert T[::10, 5].shape == (10,)

    with pytest.raises(NotImplementedError):
        T["error"]


def test_OTResult_LazyTensor(nx):
    T, a, b = get_LazyTensor(nx)

    res = ot.utils.OTResult(lazy_plan=T, batch_size=9, backend=nx)

    np.testing.assert_allclose(nx.to_numpy(a), nx.to_numpy(res.marginal_a))
    np.testing.assert_allclose(nx.to_numpy(b), nx.to_numpy(res.marginal_b))


def test_LazyTensor_reduce(nx):
    T, a, b = get_LazyTensor(nx)

    T0 = T[:]
    s0 = nx.sum(T0)

    # total sum
    s = ot.utils.reduce_lazytensor(T, nx.sum, nx=nx)
    np.testing.assert_allclose(nx.to_numpy(s), 1)
    np.testing.assert_allclose(nx.to_numpy(s), nx.to_numpy(s0))

    s2 = ot.utils.reduce_lazytensor(T, nx.sum)
    np.testing.assert_allclose(nx.to_numpy(s), nx.to_numpy(s2))

    s2 = ot.utils.reduce_lazytensor(T, nx.sum, batch_size=500)
    np.testing.assert_allclose(nx.to_numpy(s), nx.to_numpy(s2))

    s2 = ot.utils.reduce_lazytensor(T, nx.sum, batch_size=11)
    np.testing.assert_allclose(nx.to_numpy(s), nx.to_numpy(s2))

    # sum over axis 0
    s = ot.utils.reduce_lazytensor(T, nx.sum, axis=0, nx=nx)
    np.testing.assert_allclose(nx.to_numpy(s), nx.to_numpy(b))

    # sum over axis 1
    s = ot.utils.reduce_lazytensor(T, nx.sum, axis=1, nx=nx)
    np.testing.assert_allclose(nx.to_numpy(s), nx.to_numpy(a))

    # test otehr reduction function
    s = ot.utils.reduce_lazytensor(T, nx.logsumexp, axis=1, nx=nx)
    s2 = nx.logsumexp(T[:], axis=1)
    np.testing.assert_allclose(nx.to_numpy(s), nx.to_numpy(s2))

    # test 3D tensors
    def getitem(i, j, k, a, b, c):
        return a[i, None, None] * b[None, j, None] * c[None, None, k]

    # create a lazy tensor
    n = a.shape[0]
    T = ot.utils.LazyTensor((n, n, n), getitem, a=a, b=a, c=a)

    # total sum
    s1 = ot.utils.reduce_lazytensor(T, nx.sum, axis=0, nx=nx)
    s2 = ot.utils.reduce_lazytensor(T, nx.sum, axis=1, nx=nx)

    np.testing.assert_allclose(nx.to_numpy(s1), nx.to_numpy(s2))

    with pytest.raises(NotImplementedError):
        ot.utils.reduce_lazytensor(T, nx.sum, axis=2, nx=nx, batch_size=10)


def test_lowrank_LazyTensor(nx):
    p = 5
    n1 = 100
    n2 = 200

    shape = (n1, n2)

    rng = np.random.RandomState(42)
    X1 = rng.randn(n1, p)
    X2 = rng.randn(n2, p)
    diag_d = rng.rand(p)

    X1, X2, diag_d = nx.from_numpy(X1, X2, diag_d)

    T0 = nx.dot(X1, X2.T)

    T = ot.utils.get_lowrank_lazytensor(X1, X2)

    np.testing.assert_allclose(nx.to_numpy(T[:]), nx.to_numpy(T0))

    assert T.Q is X1
    assert T.R is X2

    # get the full tensor (not lazy)
    assert T[:].shape == shape

    # get one component
    assert T[1, 1] == nx.dot(X1[1], X2[1].T)

    # get one row
    assert T[1].shape == (n2,)

    # get one column with slices
    assert T[::10, 5].shape == (10,)

    T0 = nx.dot(X1 * diag_d[None, :], X2.T)

    T = ot.utils.get_lowrank_lazytensor(X1, X2, diag_d, nx=nx)

    np.testing.assert_allclose(nx.to_numpy(T[:]), nx.to_numpy(T0))


def test_labels_to_mask_helper(nx):
    y = np.array([1, 0, 2, 2, 1])
    out = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
        ]
    )
    y = nx.from_numpy(y)
    masks = ot.utils.labels_to_masks(y)
    np.testing.assert_array_equal(out, masks)


def test_label_normalization(nx):
    y = nx.from_numpy(np.arange(5) + 1)
    out = np.arange(5)
    # labels are shifted
    y_normalized = ot.utils.label_normalization(y)
    np.testing.assert_array_equal(out, y_normalized)
    # labels are shifted but the shift if expected
    y_normalized_start = ot.utils.label_normalization(y, start=1)
    np.testing.assert_array_equal(y, y_normalized_start)


def test_proj_SDP(nx):
    t = np.pi / 8
    U = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    w = np.array([1.0, -1.0])
    S = np.stack([U @ np.diag(w) @ U.T] * 2, axis=0)
    S_nx = nx.from_numpy(S)
    R = ot.utils.proj_SDP(S_nx)

    w_expected = np.array([1.0, 0.0])
    S_expected = np.stack([U @ np.diag(w_expected) @ U.T] * 2, axis=0)
    assert np.allclose(nx.to_numpy(R), S_expected)

    R0 = ot.utils.proj_SDP(S_nx[0])
    assert np.allclose(nx.to_numpy(R0), S_expected[0])


def test_laplacian():
    n = 100
    rng = np.random.RandomState(0)
    x = rng.randn(n, 2)
    M = ot.dist(x, x)
    L = ot.utils.laplacian(M)
    assert L.shape == (n, n)


def test_kl_div(nx):
    n = 10
    rng = np.random.RandomState(0)
    # test on non-negative tensors
    x = rng.randn(n)
    x = x - x.min() + 1e-5
    y = rng.randn(n)
    y = y - y.min() + 1e-5
    xb = nx.from_numpy(x)
    yb = nx.from_numpy(y)
    kl = nx.kl_div(xb, yb)
    kl_mass = nx.kl_div(xb, yb, True)
    recovered_kl = kl_mass - nx.sum(yb - xb)
    np.testing.assert_allclose(kl, recovered_kl)
