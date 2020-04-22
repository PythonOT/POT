"""Tests for module da on Domain Adaptation """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
from numpy.testing import assert_allclose, assert_equal

import ot
from ot.datasets import make_data_classif
from ot.utils import unif


def test_sinkhorn_lpl1_transport_class():
    """test_sinkhorn_transport
    """

    ns = 150
    nt = 200

    Xs, ys = make_data_classif('3gauss', ns)
    Xt, yt = make_data_classif('3gauss2', nt)

    otda = ot.da.SinkhornLpl1Transport()

    # test its computed
    otda.fit(Xs=Xs, ys=ys, Xt=Xt)
    assert hasattr(otda, "cost_")
    assert hasattr(otda, "coupling_")

    # test dimensions of coupling
    assert_equal(otda.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert_equal(otda.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))

    # test margin constraints
    mu_s = unif(ns)
    mu_t = unif(nt)
    assert_allclose(
        np.sum(otda.coupling_, axis=0), mu_t, rtol=1e-3, atol=1e-3)
    assert_allclose(
        np.sum(otda.coupling_, axis=1), mu_s, rtol=1e-3, atol=1e-3)

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    assert_equal(transp_Xs.shape, Xs.shape)

    Xs_new, _ = make_data_classif('3gauss', ns + 1)
    transp_Xs_new = otda.transform(Xs_new)

    # check that the oos method is working
    assert_equal(transp_Xs_new.shape, Xs_new.shape)

    # test inverse transform
    transp_Xt = otda.inverse_transform(Xt=Xt)
    assert_equal(transp_Xt.shape, Xt.shape)

    Xt_new, _ = make_data_classif('3gauss2', nt + 1)
    transp_Xt_new = otda.inverse_transform(Xt=Xt_new)

    # check that the oos method is working
    assert_equal(transp_Xt_new.shape, Xt_new.shape)

    # test fit_transform
    transp_Xs = otda.fit_transform(Xs=Xs, ys=ys, Xt=Xt)
    assert_equal(transp_Xs.shape, Xs.shape)

    # check label propagation
    transp_yt = otda.transform_labels(ys)
    assert_equal(transp_yt.shape[0], yt.shape[0])
    assert_equal(transp_yt.shape[1], len(np.unique(ys)))

    # check inverse label propagation
    transp_ys = otda.inverse_transform_labels(yt)
    assert_equal(transp_ys.shape[0], ys.shape[0])
    assert_equal(transp_ys.shape[1], len(np.unique(yt)))

    # test unsupervised vs semi-supervised mode
    otda_unsup = ot.da.SinkhornLpl1Transport()
    otda_unsup.fit(Xs=Xs, ys=ys, Xt=Xt)
    n_unsup = np.sum(otda_unsup.cost_)

    otda_semi = ot.da.SinkhornLpl1Transport()
    otda_semi.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)
    assert_equal(otda_semi.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    n_semisup = np.sum(otda_semi.cost_)

    # check that the cost matrix norms are indeed different
    assert n_unsup != n_semisup, "semisupervised mode not working"

    # check that the coupling forbids mass transport between labeled source
    # and labeled target samples
    mass_semi = np.sum(
        otda_semi.coupling_[otda_semi.cost_ == otda_semi.limit_max])
    assert mass_semi == 0, "semisupervised mode not working"


def test_sinkhorn_l1l2_transport_class():
    """test_sinkhorn_transport
    """

    ns = 150
    nt = 200

    Xs, ys = make_data_classif('3gauss', ns)
    Xt, yt = make_data_classif('3gauss2', nt)

    otda = ot.da.SinkhornL1l2Transport()

    # test its computed
    otda.fit(Xs=Xs, ys=ys, Xt=Xt)
    assert hasattr(otda, "cost_")
    assert hasattr(otda, "coupling_")
    assert hasattr(otda, "log_")

    # test dimensions of coupling
    assert_equal(otda.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert_equal(otda.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))

    # test margin constraints
    mu_s = unif(ns)
    mu_t = unif(nt)
    assert_allclose(
        np.sum(otda.coupling_, axis=0), mu_t, rtol=1e-3, atol=1e-3)
    assert_allclose(
        np.sum(otda.coupling_, axis=1), mu_s, rtol=1e-3, atol=1e-3)

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    assert_equal(transp_Xs.shape, Xs.shape)

    Xs_new, _ = make_data_classif('3gauss', ns + 1)
    transp_Xs_new = otda.transform(Xs_new)

    # check that the oos method is working
    assert_equal(transp_Xs_new.shape, Xs_new.shape)

    # test inverse transform
    transp_Xt = otda.inverse_transform(Xt=Xt)
    assert_equal(transp_Xt.shape, Xt.shape)

    # check label propagation
    transp_yt = otda.transform_labels(ys)
    assert_equal(transp_yt.shape[0], yt.shape[0])
    assert_equal(transp_yt.shape[1], len(np.unique(ys)))

    # check inverse label propagation
    transp_ys = otda.inverse_transform_labels(yt)
    assert_equal(transp_ys.shape[0], ys.shape[0])
    assert_equal(transp_ys.shape[1], len(np.unique(yt)))

    Xt_new, _ = make_data_classif('3gauss2', nt + 1)
    transp_Xt_new = otda.inverse_transform(Xt=Xt_new)

    # check that the oos method is working
    assert_equal(transp_Xt_new.shape, Xt_new.shape)

    # test fit_transform
    transp_Xs = otda.fit_transform(Xs=Xs, ys=ys, Xt=Xt)
    assert_equal(transp_Xs.shape, Xs.shape)

    # test unsupervised vs semi-supervised mode
    otda_unsup = ot.da.SinkhornL1l2Transport()
    otda_unsup.fit(Xs=Xs, ys=ys, Xt=Xt)
    n_unsup = np.sum(otda_unsup.cost_)

    otda_semi = ot.da.SinkhornL1l2Transport()
    otda_semi.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)
    assert_equal(otda_semi.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    n_semisup = np.sum(otda_semi.cost_)

    # check that the cost matrix norms are indeed different
    assert n_unsup != n_semisup, "semisupervised mode not working"

    # check that the coupling forbids mass transport between labeled source
    # and labeled target samples
    mass_semi = np.sum(
        otda_semi.coupling_[otda_semi.cost_ == otda_semi.limit_max])
    mass_semi = otda_semi.coupling_[otda_semi.cost_ == otda_semi.limit_max]
    assert_allclose(mass_semi, np.zeros_like(mass_semi),
                    rtol=1e-9, atol=1e-9)

    # check everything runs well with log=True
    otda = ot.da.SinkhornL1l2Transport(log=True)
    otda.fit(Xs=Xs, ys=ys, Xt=Xt)
    assert len(otda.log_.keys()) != 0


def test_sinkhorn_transport_class():
    """test_sinkhorn_transport
    """

    ns = 150
    nt = 200

    Xs, ys = make_data_classif('3gauss', ns)
    Xt, yt = make_data_classif('3gauss2', nt)

    otda = ot.da.SinkhornTransport()

    # test its computed
    otda.fit(Xs=Xs, Xt=Xt)
    assert hasattr(otda, "cost_")
    assert hasattr(otda, "coupling_")
    assert hasattr(otda, "log_")

    # test dimensions of coupling
    assert_equal(otda.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert_equal(otda.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))

    # test margin constraints
    mu_s = unif(ns)
    mu_t = unif(nt)
    assert_allclose(
        np.sum(otda.coupling_, axis=0), mu_t, rtol=1e-3, atol=1e-3)
    assert_allclose(
        np.sum(otda.coupling_, axis=1), mu_s, rtol=1e-3, atol=1e-3)

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    assert_equal(transp_Xs.shape, Xs.shape)

    Xs_new, _ = make_data_classif('3gauss', ns + 1)
    transp_Xs_new = otda.transform(Xs_new)

    # check that the oos method is working
    assert_equal(transp_Xs_new.shape, Xs_new.shape)

    # test inverse transform
    transp_Xt = otda.inverse_transform(Xt=Xt)
    assert_equal(transp_Xt.shape, Xt.shape)

    # check label propagation
    transp_yt = otda.transform_labels(ys)
    assert_equal(transp_yt.shape[0], yt.shape[0])
    assert_equal(transp_yt.shape[1], len(np.unique(ys)))

    # check inverse label propagation
    transp_ys = otda.inverse_transform_labels(yt)
    assert_equal(transp_ys.shape[0], ys.shape[0])
    assert_equal(transp_ys.shape[1], len(np.unique(yt)))

    Xt_new, _ = make_data_classif('3gauss2', nt + 1)
    transp_Xt_new = otda.inverse_transform(Xt=Xt_new)

    # check that the oos method is working
    assert_equal(transp_Xt_new.shape, Xt_new.shape)

    # test fit_transform
    transp_Xs = otda.fit_transform(Xs=Xs, Xt=Xt)
    assert_equal(transp_Xs.shape, Xs.shape)

    # test unsupervised vs semi-supervised mode
    otda_unsup = ot.da.SinkhornTransport()
    otda_unsup.fit(Xs=Xs, Xt=Xt)
    n_unsup = np.sum(otda_unsup.cost_)

    otda_semi = ot.da.SinkhornTransport()
    otda_semi.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)
    assert_equal(otda_semi.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    n_semisup = np.sum(otda_semi.cost_)

    # check that the cost matrix norms are indeed different
    assert n_unsup != n_semisup, "semisupervised mode not working"

    # check that the coupling forbids mass transport between labeled source
    # and labeled target samples
    mass_semi = np.sum(
        otda_semi.coupling_[otda_semi.cost_ == otda_semi.limit_max])
    assert mass_semi == 0, "semisupervised mode not working"

    # check everything runs well with log=True
    otda = ot.da.SinkhornTransport(log=True)
    otda.fit(Xs=Xs, ys=ys, Xt=Xt)
    assert len(otda.log_.keys()) != 0


def test_unbalanced_sinkhorn_transport_class():
    """test_sinkhorn_transport
    """

    ns = 150
    nt = 200

    Xs, ys = make_data_classif('3gauss', ns)
    Xt, yt = make_data_classif('3gauss2', nt)

    otda = ot.da.UnbalancedSinkhornTransport()

    # test its computed
    otda.fit(Xs=Xs, Xt=Xt)
    assert hasattr(otda, "cost_")
    assert hasattr(otda, "coupling_")
    assert hasattr(otda, "log_")

    # test dimensions of coupling
    assert_equal(otda.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert_equal(otda.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    assert_equal(transp_Xs.shape, Xs.shape)

    # check label propagation
    transp_yt = otda.transform_labels(ys)
    assert_equal(transp_yt.shape[0], yt.shape[0])
    assert_equal(transp_yt.shape[1], len(np.unique(ys)))

    # check inverse label propagation
    transp_ys = otda.inverse_transform_labels(yt)
    assert_equal(transp_ys.shape[0], ys.shape[0])
    assert_equal(transp_ys.shape[1], len(np.unique(yt)))

    Xs_new, _ = make_data_classif('3gauss', ns + 1)
    transp_Xs_new = otda.transform(Xs_new)

    # check that the oos method is working
    assert_equal(transp_Xs_new.shape, Xs_new.shape)

    # test inverse transform
    transp_Xt = otda.inverse_transform(Xt=Xt)
    assert_equal(transp_Xt.shape, Xt.shape)

    Xt_new, _ = make_data_classif('3gauss2', nt + 1)
    transp_Xt_new = otda.inverse_transform(Xt=Xt_new)

    # check that the oos method is working
    assert_equal(transp_Xt_new.shape, Xt_new.shape)

    # test fit_transform
    transp_Xs = otda.fit_transform(Xs=Xs, Xt=Xt)
    assert_equal(transp_Xs.shape, Xs.shape)

    # test unsupervised vs semi-supervised mode
    otda_unsup = ot.da.SinkhornTransport()
    otda_unsup.fit(Xs=Xs, Xt=Xt)
    n_unsup = np.sum(otda_unsup.cost_)

    otda_semi = ot.da.SinkhornTransport()
    otda_semi.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)
    assert_equal(otda_semi.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    n_semisup = np.sum(otda_semi.cost_)

    # check that the cost matrix norms are indeed different
    assert n_unsup != n_semisup, "semisupervised mode not working"

    # check everything runs well with log=True
    otda = ot.da.SinkhornTransport(log=True)
    otda.fit(Xs=Xs, ys=ys, Xt=Xt)
    assert len(otda.log_.keys()) != 0


def test_emd_transport_class():
    """test_sinkhorn_transport
    """

    ns = 150
    nt = 200

    Xs, ys = make_data_classif('3gauss', ns)
    Xt, yt = make_data_classif('3gauss2', nt)

    otda = ot.da.EMDTransport()

    # test its computed
    otda.fit(Xs=Xs, Xt=Xt)
    assert hasattr(otda, "cost_")
    assert hasattr(otda, "coupling_")

    # test dimensions of coupling
    assert_equal(otda.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert_equal(otda.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))

    # test margin constraints
    mu_s = unif(ns)
    mu_t = unif(nt)
    assert_allclose(
        np.sum(otda.coupling_, axis=0), mu_t, rtol=1e-3, atol=1e-3)
    assert_allclose(
        np.sum(otda.coupling_, axis=1), mu_s, rtol=1e-3, atol=1e-3)

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    assert_equal(transp_Xs.shape, Xs.shape)

    Xs_new, _ = make_data_classif('3gauss', ns + 1)
    transp_Xs_new = otda.transform(Xs_new)

    # check that the oos method is working
    assert_equal(transp_Xs_new.shape, Xs_new.shape)

    # test inverse transform
    transp_Xt = otda.inverse_transform(Xt=Xt)
    assert_equal(transp_Xt.shape, Xt.shape)

    # check label propagation
    transp_yt = otda.transform_labels(ys)
    assert_equal(transp_yt.shape[0], yt.shape[0])
    assert_equal(transp_yt.shape[1], len(np.unique(ys)))

    # check inverse label propagation
    transp_ys = otda.inverse_transform_labels(yt)
    assert_equal(transp_ys.shape[0], ys.shape[0])
    assert_equal(transp_ys.shape[1], len(np.unique(yt)))

    Xt_new, _ = make_data_classif('3gauss2', nt + 1)
    transp_Xt_new = otda.inverse_transform(Xt=Xt_new)

    # check that the oos method is working
    assert_equal(transp_Xt_new.shape, Xt_new.shape)

    # test fit_transform
    transp_Xs = otda.fit_transform(Xs=Xs, Xt=Xt)
    assert_equal(transp_Xs.shape, Xs.shape)

    # test unsupervised vs semi-supervised mode
    otda_unsup = ot.da.EMDTransport()
    otda_unsup.fit(Xs=Xs, ys=ys, Xt=Xt)
    n_unsup = np.sum(otda_unsup.cost_)

    otda_semi = ot.da.EMDTransport()
    otda_semi.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)
    assert_equal(otda_semi.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    n_semisup = np.sum(otda_semi.cost_)

    # check that the cost matrix norms are indeed different
    assert n_unsup != n_semisup, "semisupervised mode not working"

    # check that the coupling forbids mass transport between labeled source
    # and labeled target samples
    mass_semi = np.sum(
        otda_semi.coupling_[otda_semi.cost_ == otda_semi.limit_max])
    mass_semi = otda_semi.coupling_[otda_semi.cost_ == otda_semi.limit_max]

    # we need to use a small tolerance here, otherwise the test breaks
    assert_allclose(mass_semi, np.zeros_like(mass_semi),
                    rtol=1e-2, atol=1e-2)


def test_mapping_transport_class():
    """test_mapping_transport
    """

    ns = 60
    nt = 120

    Xs, ys = make_data_classif('3gauss', ns)
    Xt, yt = make_data_classif('3gauss2', nt)
    Xs_new, _ = make_data_classif('3gauss', ns + 1)

    ##########################################################################
    # kernel == linear mapping tests
    ##########################################################################

    # check computation and dimensions if bias == False
    otda = ot.da.MappingTransport(kernel="linear", bias=False)
    otda.fit(Xs=Xs, Xt=Xt)
    assert hasattr(otda, "coupling_")
    assert hasattr(otda, "mapping_")
    assert hasattr(otda, "log_")

    assert_equal(otda.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert_equal(otda.mapping_.shape, ((Xs.shape[1], Xt.shape[1])))

    # test margin constraints
    mu_s = unif(ns)
    mu_t = unif(nt)
    assert_allclose(
        np.sum(otda.coupling_, axis=0), mu_t, rtol=1e-3, atol=1e-3)
    assert_allclose(
        np.sum(otda.coupling_, axis=1), mu_s, rtol=1e-3, atol=1e-3)

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    assert_equal(transp_Xs.shape, Xs.shape)

    transp_Xs_new = otda.transform(Xs_new)

    # check that the oos method is working
    assert_equal(transp_Xs_new.shape, Xs_new.shape)

    # check computation and dimensions if bias == True
    otda = ot.da.MappingTransport(kernel="linear", bias=True)
    otda.fit(Xs=Xs, Xt=Xt)
    assert_equal(otda.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert_equal(otda.mapping_.shape, ((Xs.shape[1] + 1, Xt.shape[1])))

    # test margin constraints
    mu_s = unif(ns)
    mu_t = unif(nt)
    assert_allclose(
        np.sum(otda.coupling_, axis=0), mu_t, rtol=1e-3, atol=1e-3)
    assert_allclose(
        np.sum(otda.coupling_, axis=1), mu_s, rtol=1e-3, atol=1e-3)

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    assert_equal(transp_Xs.shape, Xs.shape)

    transp_Xs_new = otda.transform(Xs_new)

    # check that the oos method is working
    assert_equal(transp_Xs_new.shape, Xs_new.shape)

    ##########################################################################
    # kernel == gaussian mapping tests
    ##########################################################################

    # check computation and dimensions if bias == False
    otda = ot.da.MappingTransport(kernel="gaussian", bias=False)
    otda.fit(Xs=Xs, Xt=Xt)

    assert_equal(otda.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert_equal(otda.mapping_.shape, ((Xs.shape[0], Xt.shape[1])))

    # test margin constraints
    mu_s = unif(ns)
    mu_t = unif(nt)
    assert_allclose(
        np.sum(otda.coupling_, axis=0), mu_t, rtol=1e-3, atol=1e-3)
    assert_allclose(
        np.sum(otda.coupling_, axis=1), mu_s, rtol=1e-3, atol=1e-3)

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    assert_equal(transp_Xs.shape, Xs.shape)

    transp_Xs_new = otda.transform(Xs_new)

    # check that the oos method is working
    assert_equal(transp_Xs_new.shape, Xs_new.shape)

    # check computation and dimensions if bias == True
    otda = ot.da.MappingTransport(kernel="gaussian", bias=True)
    otda.fit(Xs=Xs, Xt=Xt)
    assert_equal(otda.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert_equal(otda.mapping_.shape, ((Xs.shape[0] + 1, Xt.shape[1])))

    # test margin constraints
    mu_s = unif(ns)
    mu_t = unif(nt)
    assert_allclose(
        np.sum(otda.coupling_, axis=0), mu_t, rtol=1e-3, atol=1e-3)
    assert_allclose(
        np.sum(otda.coupling_, axis=1), mu_s, rtol=1e-3, atol=1e-3)

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    assert_equal(transp_Xs.shape, Xs.shape)

    transp_Xs_new = otda.transform(Xs_new)

    # check that the oos method is working
    assert_equal(transp_Xs_new.shape, Xs_new.shape)

    # check everything runs well with log=True
    otda = ot.da.MappingTransport(kernel="gaussian", log=True)
    otda.fit(Xs=Xs, Xt=Xt)
    assert len(otda.log_.keys()) != 0


def test_linear_mapping():
    ns = 150
    nt = 200

    Xs, ys = make_data_classif('3gauss', ns)
    Xt, yt = make_data_classif('3gauss2', nt)

    A, b = ot.da.OT_mapping_linear(Xs, Xt)

    Xst = Xs.dot(A) + b

    Ct = np.cov(Xt.T)
    Cst = np.cov(Xst.T)

    np.testing.assert_allclose(Ct, Cst, rtol=1e-2, atol=1e-2)


def test_linear_mapping_class():
    ns = 150
    nt = 200

    Xs, ys = make_data_classif('3gauss', ns)
    Xt, yt = make_data_classif('3gauss2', nt)

    otmap = ot.da.LinearTransport()

    otmap.fit(Xs=Xs, Xt=Xt)
    assert hasattr(otmap, "A_")
    assert hasattr(otmap, "B_")
    assert hasattr(otmap, "A1_")
    assert hasattr(otmap, "B1_")

    Xst = otmap.transform(Xs=Xs)

    Ct = np.cov(Xt.T)
    Cst = np.cov(Xst.T)

    np.testing.assert_allclose(Ct, Cst, rtol=1e-2, atol=1e-2)


def test_jcpot_transport_class():
    """test_jcpot_transport
    """

    ns1 = 150
    ns2 = 150
    nt = 200

    Xs1, ys1 = make_data_classif('3gauss', ns1)
    Xs2, ys2 = make_data_classif('3gauss', ns2)

    Xt, yt = make_data_classif('3gauss2', nt)

    Xs = [Xs1, Xs2]
    ys = [ys1, ys2]

    otda = ot.da.JCPOTTransport(reg_e=1, max_iter=10000, tol=1e-9, verbose=True, log=True)

    # test its computed
    otda.fit(Xs=Xs, ys=ys, Xt=Xt)

    assert hasattr(otda, "coupling_")
    assert hasattr(otda, "proportions_")
    assert hasattr(otda, "log_")

    # test dimensions of coupling
    for i, xs in enumerate(Xs):
        assert_equal(otda.coupling_[i].shape, ((xs.shape[0], Xt.shape[0])))

    # test all margin constraints
    mu_t = unif(nt)

    for i in range(len(Xs)):
        # test margin constraints w.r.t. uniform target weights for each coupling matrix
        assert_allclose(
            np.sum(otda.coupling_[i], axis=0), mu_t, rtol=1e-3, atol=1e-3)

        # test margin constraints w.r.t. modified source weights for each source domain

        assert_allclose(
            np.dot(otda.log_['D1'][i], np.sum(otda.coupling_[i], axis=1)), otda.proportions_, rtol=1e-3,
            atol=1e-3)

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    [assert_equal(x.shape, y.shape) for x, y in zip(transp_Xs, Xs)]

    Xs_new, _ = make_data_classif('3gauss', ns1 + 1)
    transp_Xs_new = otda.transform(Xs_new)

    # check that the oos method is working
    assert_equal(transp_Xs_new.shape, Xs_new.shape)

    # check label propagation
    transp_yt = otda.transform_labels(ys)
    assert_equal(transp_yt.shape[0], yt.shape[0])
    assert_equal(transp_yt.shape[1], len(np.unique(ys)))

    # check inverse label propagation
    transp_ys = otda.inverse_transform_labels(yt)
    [assert_equal(x.shape[0], y.shape[0]) for x, y in zip(transp_ys, ys)]
    [assert_equal(x.shape[1], len(np.unique(y))) for x, y in zip(transp_ys, ys)]


def test_jcpot_barycenter():
    """test_jcpot_barycenter
    """

    ns1 = 150
    ns2 = 150
    nt = 200

    sigma = 0.1
    np.random.seed(1985)

    ps1 = .2
    ps2 = .9
    pt = .4

    Xs1, ys1 = make_data_classif('2gauss_prop', ns1, nz=sigma, p=ps1)
    Xs2, ys2 = make_data_classif('2gauss_prop', ns2, nz=sigma, p=ps2)
    Xt, yt = make_data_classif('2gauss_prop', nt, nz=sigma, p=pt)

    Xs = [Xs1, Xs2]
    ys = [ys1, ys2]

    prop = ot.bregman.jcpot_barycenter(Xs, ys, Xt, reg=.5, metric='sqeuclidean',
                                       numItermax=10000, stopThr=1e-9, verbose=False, log=False)

    np.testing.assert_allclose(prop, [1 - pt, pt], rtol=1e-3, atol=1e-3)


def test_emd_laplace_class():
    """test_emd_laplace_transport
    """
    ns = 150
    nt = 200

    Xs, ys = make_data_classif('3gauss', ns)
    Xt, yt = make_data_classif('3gauss2', nt)

    otda = ot.da.EMDLaplaceTransport(reg_lap=0.01, max_iter=1000, tol=1e-9, verbose=False, log=True)

    # test its computed
    otda.fit(Xs=Xs, ys=ys, Xt=Xt)

    assert hasattr(otda, "coupling_")
    assert hasattr(otda, "log_")

    # test dimensions of coupling
    assert_equal(otda.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))

    # test all margin constraints
    mu_s = unif(ns)
    mu_t = unif(nt)

    assert_allclose(
        np.sum(otda.coupling_, axis=0), mu_t, rtol=1e-3, atol=1e-3)
    assert_allclose(
        np.sum(otda.coupling_, axis=1), mu_s, rtol=1e-3, atol=1e-3)

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    [assert_equal(x.shape, y.shape) for x, y in zip(transp_Xs, Xs)]

    Xs_new, _ = make_data_classif('3gauss', ns + 1)
    transp_Xs_new = otda.transform(Xs_new)

    # check that the oos method is working
    assert_equal(transp_Xs_new.shape, Xs_new.shape)

    # test inverse transform
    transp_Xt = otda.inverse_transform(Xt=Xt)
    assert_equal(transp_Xt.shape, Xt.shape)

    Xt_new, _ = make_data_classif('3gauss2', nt + 1)
    transp_Xt_new = otda.inverse_transform(Xt=Xt_new)

    # check that the oos method is working
    assert_equal(transp_Xt_new.shape, Xt_new.shape)

    # test fit_transform
    transp_Xs = otda.fit_transform(Xs=Xs, Xt=Xt)
    assert_equal(transp_Xs.shape, Xs.shape)

    # check label propagation
    transp_yt = otda.transform_labels(ys)
    assert_equal(transp_yt.shape[0], yt.shape[0])
    assert_equal(transp_yt.shape[1], len(np.unique(ys)))

    # check inverse label propagation
    transp_ys = otda.inverse_transform_labels(yt)
    assert_equal(transp_ys.shape[0], ys.shape[0])
    assert_equal(transp_ys.shape[1], len(np.unique(yt)))
