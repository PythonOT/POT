"""Tests for module gaussian"""

# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytehnique.edu>
#
# License: MIT License

import numpy as np

import pytest

import ot
from ot.datasets import make_data_classif
from ot.utils import is_all_finite


def test_bures_wasserstein_mapping(nx):
    ns = 50
    nt = 50

    Xs, ys = make_data_classif("3gauss", ns)
    Xt, yt = make_data_classif("3gauss2", nt)
    ms = np.mean(Xs, axis=0)[None, :]
    mt = np.mean(Xt, axis=0)[None, :]
    Cs = np.cov(Xs.T)
    Ct = np.cov(Xt.T)

    Xsb, msb, mtb, Csb, Ctb = nx.from_numpy(Xs, ms, mt, Cs, Ct)

    A_log, b_log, log = ot.gaussian.bures_wasserstein_mapping(
        msb, mtb, Csb, Ctb, log=True
    )
    A, b = ot.gaussian.bures_wasserstein_mapping(msb, mtb, Csb, Ctb, log=False)

    Xst = nx.to_numpy(nx.dot(Xsb, A) + b)
    Xst_log = nx.to_numpy(nx.dot(Xsb, A_log) + b_log)

    Cst = np.cov(Xst.T)
    Cst_log = np.cov(Xst_log.T)

    np.testing.assert_allclose(Cst_log, Cst, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(Ct, Cst, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("bias", [True, False])
def test_empirical_bures_wasserstein_mapping(nx, bias):
    ns = 50
    nt = 50

    Xs, ys = make_data_classif("3gauss", ns)
    Xt, yt = make_data_classif("3gauss2", nt)

    if not bias:
        ms = np.mean(Xs, axis=0)[None, :]
        mt = np.mean(Xt, axis=0)[None, :]

        Xs = Xs - ms
        Xt = Xt - mt

    Xsb, Xtb = nx.from_numpy(Xs, Xt)

    A, b, log = ot.gaussian.empirical_bures_wasserstein_mapping(
        Xsb, Xtb, log=True, bias=bias
    )
    A_log, b_log = ot.gaussian.empirical_bures_wasserstein_mapping(
        Xsb, Xtb, log=False, bias=bias
    )

    Xst = nx.to_numpy(nx.dot(Xsb, A) + b)
    Xst_log = nx.to_numpy(nx.dot(Xsb, A_log) + b_log)

    Ct = np.cov(Xt.T)
    Cst = np.cov(Xst.T)
    Cst_log = np.cov(Xst_log.T)

    np.testing.assert_allclose(Cst_log, Cst, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(Ct, Cst, rtol=1e-2, atol=1e-2)


def test_empirical_bures_wasserstein_mapping_numerical_error_warning():
    rng = np.random.RandomState(42)
    Xs = rng.rand(766, 800) * 5
    Xt = rng.rand(295, 800) * 2
    with pytest.warns():
        A, b = ot.gaussian.empirical_bures_wasserstein_mapping(Xs, Xt, reg=1e-8)
        assert not is_all_finite(A, b)


def test_bures_wasserstein_distance(nx):
    ms, mt = np.array([0]).astype(np.float32), np.array([10]).astype(np.float32)
    Cs, Ct = np.array([[1]]).astype(np.float32), np.array([[1]]).astype(np.float32)
    msb, mtb, Csb, Ctb = nx.from_numpy(ms, mt, Cs, Ct)
    Wb_log, log = ot.gaussian.bures_wasserstein_distance(msb, mtb, Csb, Ctb, log=True)
    Wb = ot.gaussian.bures_wasserstein_distance(msb, mtb, Csb, Ctb, log=False)
    Wb2 = ot.gaussian.bures_distance(Csb, Ctb, log=False)

    np.testing.assert_allclose(
        nx.to_numpy(Wb_log), nx.to_numpy(Wb), rtol=1e-2, atol=1e-2
    )
    np.testing.assert_allclose(10, nx.to_numpy(Wb), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(0, Wb2, rtol=1e-2, atol=1e-2)


def test_bures_wasserstein_distance_batch(nx):
    n = 50
    k = 2
    X = []
    y = []
    m = []
    C = []
    for _ in range(k):
        X_, y_ = make_data_classif("3gauss", n)
        m_ = np.mean(X_, axis=0)[None, :]
        C_ = np.cov(X_.T)
        X.append(X_)
        y.append(y_)
        m.append(m_)
        C.append(C_)
    m = np.array(m)
    C = np.array(C)
    X = nx.from_numpy(*X)
    m = nx.from_numpy(m)
    C = nx.from_numpy(C)

    Wb = ot.gaussian.bures_wasserstein_distance(m[0, 0], m[1, 0], C[0], C[1], log=False)

    # Test cross vs 1
    Wb2 = ot.gaussian.bures_wasserstein_distance(
        m[0, 0][None], m[1, 0][None], C[0][None], C[1][None]
    )
    np.testing.assert_allclose(nx.to_numpy(Wb), nx.to_numpy(Wb2[0, 0]), atol=1e-5)
    np.testing.assert_equal(Wb2.shape, (1, 1))

    Wb2 = ot.gaussian.bures_wasserstein_distance(m[:, 0], m[1, 0][None], C, C[1][None])
    np.testing.assert_allclose(nx.to_numpy(Wb), nx.to_numpy(Wb2[0, 0]), atol=1e-5)
    np.testing.assert_allclose(0, nx.to_numpy(Wb2[1, 0]), atol=1e-5)
    np.testing.assert_equal(Wb2.shape, (2, 1))

    Wb2 = ot.gaussian.bures_wasserstein_distance(m[:, 0], m[:, 0], C, C)
    np.testing.assert_allclose(nx.to_numpy(Wb), nx.to_numpy(Wb2[1, 0]), atol=1e-5)
    np.testing.assert_allclose(nx.to_numpy(Wb), nx.to_numpy(Wb2[0, 1]), atol=1e-5)
    np.testing.assert_allclose(0, nx.to_numpy(Wb2[0, 0]), atol=1e-5)
    np.testing.assert_allclose(0, nx.to_numpy(Wb2[1, 1]), atol=1e-5)
    np.testing.assert_equal(Wb2.shape, (2, 2))

    # Test paired
    Wb3 = ot.gaussian.bures_wasserstein_distance(m[:, 0], m[:, 0], C, C, paired=True)
    np.testing.assert_allclose(0, nx.to_numpy(Wb3[0]), atol=1e-5)
    np.testing.assert_allclose(0, nx.to_numpy(Wb3[1]), atol=1e-5)

    m_rev = np.zeros((k, 2))
    C_rev = np.zeros((k, 2, 2))
    m_rev[0] = m[1, 0]
    m_rev[1] = m[0, 0]
    C_rev[0] = C[1]
    C_rev[1] = C[0]
    m_rev = nx.from_numpy(m_rev)
    C_rev = nx.from_numpy(C_rev)

    Wb3 = ot.gaussian.bures_wasserstein_distance(m_rev, m[:, 0], C_rev, C, paired=True)
    np.testing.assert_allclose(nx.to_numpy(Wb2)[0, 1], nx.to_numpy(Wb3)[0], atol=1e-5)
    np.testing.assert_allclose(nx.to_numpy(Wb2)[0, 1], nx.to_numpy(Wb3)[0], atol=1e-5)

    with pytest.raises(AssertionError):
        Wb3 = ot.gaussian.bures_wasserstein_distance(m[0, 0], m[:, 0], C[0], C)


@pytest.mark.parametrize("bias", [True, False])
def test_empirical_bures_wasserstein_distance(nx, bias):
    ns = 400
    nt = 400

    rng = np.random.RandomState(10)
    Xs = rng.normal(0, 1, ns)[:, np.newaxis]
    Xt = rng.normal(10 * bias, 1, nt)[:, np.newaxis]

    Xsb, Xtb = nx.from_numpy(Xs, Xt)
    Wb_log, log = ot.gaussian.empirical_bures_wasserstein_distance(
        Xsb, Xtb, log=True, bias=bias
    )
    Wb = ot.gaussian.empirical_bures_wasserstein_distance(
        Xsb, Xtb, log=False, bias=bias
    )

    np.testing.assert_allclose(
        nx.to_numpy(Wb_log), nx.to_numpy(Wb), rtol=1e-2, atol=1e-2
    )
    np.testing.assert_allclose(10 * bias, nx.to_numpy(Wb), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "method",
    [
        "fixed_point",
        "gradient_descent",
        "stochastic_gradient_descent",
        "averaged_stochastic_gradient_descent",
    ],
)
def test_bures_wasserstein_barycenter(nx, method):
    n = 50
    k = 10
    X = []
    y = []
    m = []
    C = []
    for _ in range(k):
        X_, y_ = make_data_classif("3gauss", n)
        m_ = np.mean(X_, axis=0)[None, :]
        C_ = np.cov(X_.T)
        X.append(X_)
        y.append(y_)
        m.append(m_)
        C.append(C_)
    m = np.array(m)
    C = np.array(C)
    X = nx.from_numpy(*X)
    m = nx.from_numpy(m)[:, 0]
    C = nx.from_numpy(C)

    mblog, Cblog, log = ot.gaussian.bures_wasserstein_barycenter(
        m, C, method=method, log=True
    )
    mb, Cb = ot.gaussian.bures_wasserstein_barycenter(m, C, method=method, log=False)

    np.testing.assert_allclose(Cb, Cblog, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(mb, mblog, rtol=1e-2, atol=1e-2)

    # Test weights argument
    weights = nx.ones(k) / k
    mbw, Cbw = ot.gaussian.bures_wasserstein_barycenter(
        m, C, weights=weights, method=method, log=False
    )
    np.testing.assert_allclose(Cbw, Cb, rtol=1e-1, atol=1e-1)

    # test with closed form for diagonal covariance matrices
    Cdiag = [nx.diag(nx.diag(C[i])) for i in range(k)]
    Cdiag = nx.stack(Cdiag, axis=0)
    mbdiag, Cbdiag = ot.gaussian.bures_wasserstein_barycenter(
        m, Cdiag, method=method, log=False
    )

    Cdiag_sqrt = [nx.sqrtm(C) for C in Cdiag]
    Cdiag_sqrt = nx.stack(Cdiag_sqrt, axis=0)
    Cdiag_mean = nx.mean(Cdiag_sqrt, axis=0)
    Cdiag_cf = Cdiag_mean @ Cdiag_mean

    np.testing.assert_allclose(Cbdiag, Cdiag_cf, rtol=1e-2, atol=1e-2)


def test_fixedpoint_vs_gradientdescent_bures_wasserstein_barycenter(nx):
    n = 50
    k = 10
    X = []
    y = []
    m = []
    C = []
    for _ in range(k):
        X_, y_ = make_data_classif("3gauss", n)
        m_ = np.mean(X_, axis=0)[None, :]
        C_ = np.cov(X_.T)
        X.append(X_)
        y.append(y_)
        m.append(m_)
        C.append(C_)
    m = np.array(m)
    C = np.array(C)
    X = nx.from_numpy(*X)
    m = nx.from_numpy(m)[:, 0]
    C = nx.from_numpy(C)

    mb, Cb = ot.gaussian.bures_wasserstein_barycenter(
        m, C, method="fixed_point", log=False
    )
    mb2, Cb2 = ot.gaussian.bures_wasserstein_barycenter(
        m, C, method="gradient_descent", log=False
    )

    np.testing.assert_allclose(mb, mb2, atol=1e-5)
    np.testing.assert_allclose(Cb, Cb2, atol=1e-5)

    # Test weights argument
    Cbw = ot.gaussian.bures_barycenter_fixpoint(C, weights=None)
    Cbw2 = ot.gaussian.bures_barycenter_gradient_descent(C, weights=None)
    np.testing.assert_allclose(Cbw, Cb, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(Cbw2, Cb2, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "method", ["stochastic_gradient_descent", "averaged_stochastic_gradient_descent"]
)
def test_stochastic_gd_bures_wasserstein_barycenter(nx, method):
    n = 50
    k = 10
    X = []
    y = []
    m = []
    C = []
    for _ in range(k):
        X_, y_ = make_data_classif("3gauss", n)
        m_ = np.mean(X_, axis=0)[None, :]
        C_ = np.cov(X_.T)
        X.append(X_)
        y.append(y_)
        m.append(m_)
        C.append(C_)
    m = np.array(m)
    C = np.array(C)
    X = nx.from_numpy(*X)
    m = nx.from_numpy(m)[:, 0]
    C = nx.from_numpy(C)

    mb, Cb = ot.gaussian.bures_wasserstein_barycenter(
        m, C, method="fixed_point", log=False
    )

    loss = nx.mean(ot.gaussian.bures_wasserstein_distance(mb[None], m, Cb[None], C))

    n_samples = [1, 5]
    for n in n_samples:
        mb2, Cb2 = ot.gaussian.bures_wasserstein_barycenter(
            m, C, method=method, log=False, batch_size=n
        )

        loss2 = nx.mean(
            ot.gaussian.bures_wasserstein_distance(mb2[None], m, Cb2[None], C)
        )

        np.testing.assert_allclose(mb, mb2, atol=1e-5)
        # atol big for now because too slow, need to see if
        # it can be improved...
        np.testing.assert_allclose(Cb, Cb2, atol=1e-1)
        np.testing.assert_allclose(loss, loss2, atol=1e-3)

    with pytest.raises(ValueError):
        mb2, Cb2 = ot.gaussian.bures_wasserstein_barycenter(
            m, C, method=method, log=False, batch_size=-5
        )


def test_not_implemented_method(nx):
    n = 50
    k = 10
    X = []
    y = []
    m = []
    C = []
    for _ in range(k):
        X_, y_ = make_data_classif("3gauss", n)
        m_ = np.mean(X_, axis=0)[None, :]
        C_ = np.cov(X_.T)
        X.append(X_)
        y.append(y_)
        m.append(m_)
        C.append(C_)
    m = np.array(m)
    C = np.array(C)
    X = nx.from_numpy(*X)
    m = nx.from_numpy(m)[:, 0]
    C = nx.from_numpy(C)

    not_implemented = "new_method"
    with pytest.raises(ValueError):
        mb, Cb = ot.gaussian.bures_wasserstein_barycenter(
            m, C, method=not_implemented, log=False
        )


@pytest.mark.parametrize("bias", [True, False])
def test_empirical_bures_wasserstein_barycenter(nx, bias):
    n = 50
    k = 10
    X = []
    y = []
    for _ in range(k):
        X_, y_ = make_data_classif("3gauss", n)
        X.append(X_)
        y.append(y_)

    X = nx.from_numpy(*X)

    mblog, Cblog, log = ot.gaussian.empirical_bures_wasserstein_barycenter(
        X, log=True, bias=bias
    )
    mb, Cb = ot.gaussian.empirical_bures_wasserstein_barycenter(X, log=False, bias=bias)

    np.testing.assert_allclose(Cb, Cblog, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(mb, mblog, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("d_target", [1, 2, 3, 10])
def test_gaussian_gromov_wasserstein_distance(nx, d_target):
    ns = 400
    nt = 400

    rng = np.random.RandomState(10)
    Xs, ys = make_data_classif("3gauss", ns, random_state=rng)
    Xt, yt = make_data_classif("3gauss2", nt, random_state=rng)
    Xt = np.concatenate((Xt, rng.normal(0, 1, (nt, 8))), axis=1)
    Xt = Xt[:, 0:d_target].reshape((nt, d_target))

    ms = np.mean(Xs, axis=0)[None, :]
    mt = np.mean(Xt, axis=0)[None, :]
    Cs = np.cov(Xs.T)
    Ct = np.cov(Xt.T).reshape((d_target, d_target))

    Xsb, Xtb, msb, mtb, Csb, Ctb = nx.from_numpy(Xs, Xt, ms, mt, Cs, Ct)

    Gb, log = ot.gaussian.gaussian_gromov_wasserstein_distance(Csb, Ctb, log=True)
    Ge, log = ot.gaussian.empirical_gaussian_gromov_wasserstein_distance(
        Xsb, Xtb, log=True
    )

    # no log
    Ge0 = ot.gaussian.empirical_gaussian_gromov_wasserstein_distance(
        Xsb, Xtb, log=False
    )

    np.testing.assert_allclose(nx.to_numpy(Gb), nx.to_numpy(Ge), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(nx.to_numpy(Ge), nx.to_numpy(Ge0), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("d_target", [1, 2, 3, 10])
def test_gaussian_gromov_wasserstein_mapping(nx, d_target):
    ns = 400
    nt = 400

    rng = np.random.RandomState(10)
    Xs, ys = make_data_classif("3gauss", ns, random_state=rng)
    Xt, yt = make_data_classif("3gauss2", nt, random_state=rng)
    Xt = np.concatenate((Xt, rng.normal(0, 1, (nt, 8))), axis=1)
    Xt = Xt[:, 0:d_target].reshape((nt, d_target))

    ms = np.mean(Xs, axis=0)[None, :]
    mt = np.mean(Xt, axis=0)[None, :]
    Cs = np.cov(Xs.T)
    Ct = np.cov(Xt.T).reshape((d_target, d_target))

    Xsb, Xtb, msb, mtb, Csb, Ctb = nx.from_numpy(Xs, Xt, ms, mt, Cs, Ct)

    A, b, log = ot.gaussian.gaussian_gromov_wasserstein_mapping(
        msb, mtb, Csb, Ctb, log=True
    )
    Ae, be, loge = ot.gaussian.empirical_gaussian_gromov_wasserstein_mapping(
        Xsb, Xtb, log=True
    )

    # no log + skewness
    Ae0, be0 = ot.gaussian.empirical_gaussian_gromov_wasserstein_mapping(
        Xsb, Xtb, log=False, sign_eigs="skewness"
    )

    Xst = nx.to_numpy(nx.dot(Xsb, A) + b)
    Cst = np.cov(Xst.T)

    np.testing.assert_allclose(nx.to_numpy(A), nx.to_numpy(Ae))
    if d_target <= 2:
        np.testing.assert_allclose(Ct, Cst)

    # test the other way around (target to source)
    Ai, bi, logi = ot.gaussian.gaussian_gromov_wasserstein_mapping(
        mtb, msb, Ctb, Csb, log=True
    )

    Xtt = nx.to_numpy(nx.dot(Xtb, Ai) + bi)
    Ctt = np.cov(Xtt.T)

    if d_target >= 2:
        np.testing.assert_allclose(Cs, Ctt)
