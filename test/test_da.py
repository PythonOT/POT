"""Tests for module da on Domain Adaptation"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

import ot
from ot.datasets import make_data_classif
from ot.utils import unif

try:  # test if cudamat installed
    import sklearn  # noqa: F401

    nosklearn = False
except ImportError:
    nosklearn = True

try:  # test if cvxpy is installed
    import cvxpy  # noqa: F401

    nocvxpy = False
except ImportError:
    nocvxpy = True


def test_class_jax_tf():
    from ot.backend import tf

    backends = []
    if tf:
        backends.append(ot.backend.TensorflowBackend())

    for nx in backends:
        ns = 150
        nt = 200

        Xs, ys = make_data_classif("3gauss", ns)
        Xt, yt = make_data_classif("3gauss2", nt)

        Xs, ys, Xt, yt = nx.from_numpy(Xs, ys, Xt, yt)

        otda = ot.da.SinkhornLpl1Transport()

        with pytest.raises(TypeError):
            otda.fit(Xs=Xs, ys=ys, Xt=Xt)


@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
@pytest.mark.parametrize(
    "class_to_test",
    [
        ot.da.EMDTransport,
        ot.da.SinkhornTransport,
        ot.da.SinkhornLpl1Transport,
        ot.da.SinkhornL1l2Transport,
        ot.da.SinkhornL1l2Transport,
    ],
)
def test_log_da(nx, class_to_test):
    ns = 50
    nt = 50

    Xs, ys = make_data_classif("3gauss", ns)
    Xt, yt = make_data_classif("3gauss2", nt)

    Xs, ys, Xt, yt = nx.from_numpy(Xs, ys, Xt, yt)

    otda = class_to_test(log=True)

    # test its computed
    otda.fit(Xs=Xs, ys=ys, Xt=Xt)
    assert hasattr(otda, "log_")


@pytest.skip_backend("tf")
def test_sinkhorn_lpl1_transport_class(nx):
    """test_sinkhorn_transport"""

    ns = 50
    nt = 50

    Xs, ys = make_data_classif("3gauss", ns, random_state=42)
    Xt, yt = make_data_classif("3gauss2", nt, random_state=43)
    # prepare semi-supervised labels
    yt_semi = np.copy(yt)
    yt_semi[np.arange(0, nt, 2)] = -1

    Xs, ys, Xt, yt, yt_semi = nx.from_numpy(Xs, ys, Xt, yt, yt_semi)

    otda = ot.da.SinkhornLpl1Transport()

    # test its computed
    otda.fit(Xs=Xs, ys=ys, Xt=Xt)
    assert hasattr(otda, "cost_")
    assert not np.any(np.isnan(nx.to_numpy(otda.cost_))), "cost is finite"
    assert hasattr(otda, "coupling_")
    assert np.all(np.isfinite(nx.to_numpy(otda.coupling_))), "coupling is finite"

    # test dimensions of coupling
    assert_equal(otda.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert_equal(otda.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))

    # test margin constraints
    mu_s = unif(ns)
    mu_t = unif(nt)
    assert_allclose(
        nx.to_numpy(nx.sum(otda.coupling_, axis=0)), mu_t, rtol=1e-3, atol=1e-3
    )
    assert_allclose(
        nx.to_numpy(nx.sum(otda.coupling_, axis=1)), mu_s, rtol=1e-3, atol=1e-3
    )

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    assert_equal(transp_Xs.shape, Xs.shape)

    Xs_new = nx.from_numpy(make_data_classif("3gauss", ns + 1, random_state=44)[0])
    transp_Xs_new = otda.transform(Xs_new)

    # check that the oos method is working
    assert_equal(transp_Xs_new.shape, Xs_new.shape)

    # test inverse transform
    transp_Xt = otda.inverse_transform(Xt=Xt)
    assert_equal(transp_Xt.shape, Xt.shape)

    Xt_new = nx.from_numpy(make_data_classif("3gauss2", nt + 1, random_state=45)[0])
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
    assert np.all(
        np.isfinite(nx.to_numpy(otda_unsup.coupling_))
    ), "unsup coupling is finite"
    n_unsup = nx.sum(otda_unsup.cost_)

    otda_semi = ot.da.SinkhornLpl1Transport()
    otda_semi.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt_semi)
    assert np.all(
        np.isfinite(nx.to_numpy(otda_semi.coupling_))
    ), "semi coupling is finite"
    assert_equal(otda_semi.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    n_semisup = nx.sum(otda_semi.cost_)

    # check that the cost matrix norms are indeed different
    assert np.allclose(
        n_unsup, n_semisup, atol=1e-7
    ), "semisupervised mode is not working"

    # check that the coupling forbids mass transport between labeled source
    # and labeled target samples
    mass_semi = nx.sum(otda_semi.coupling_[otda_semi.cost_ == otda_semi.limit_max])
    assert mass_semi == 0, "semisupervised mode not working"


@pytest.skip_backend("tf")
def test_sinkhorn_l1l2_transport_class(nx):
    """test_sinkhorn_transport"""

    ns = 50
    nt = 50

    Xs, ys = make_data_classif("3gauss", ns, random_state=42)
    Xt, yt = make_data_classif("3gauss2", nt, random_state=43)
    # prepare semi-supervised labels
    yt_semi = np.copy(yt)
    yt_semi[np.arange(0, nt, 2)] = -1

    Xs, ys, Xt, yt, yt_semi = nx.from_numpy(Xs, ys, Xt, yt, yt_semi)

    otda = ot.da.SinkhornL1l2Transport(max_inner_iter=500)
    otda.fit(Xs=Xs, ys=ys, Xt=Xt)

    # test its computed
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
        nx.to_numpy(nx.sum(otda.coupling_, axis=0)), mu_t, rtol=1e-3, atol=1e-3
    )
    assert_allclose(
        nx.to_numpy(nx.sum(otda.coupling_, axis=1)), mu_s, rtol=1e-3, atol=1e-3
    )

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    assert_equal(transp_Xs.shape, Xs.shape)

    Xs_new = nx.from_numpy(make_data_classif("3gauss", ns + 1)[0])
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

    Xt_new = nx.from_numpy(make_data_classif("3gauss2", nt + 1)[0])
    transp_Xt_new = otda.inverse_transform(Xt=Xt_new)

    # check that the oos method is working
    assert_equal(transp_Xt_new.shape, Xt_new.shape)

    # test fit_transform
    transp_Xs = otda.fit_transform(Xs=Xs, ys=ys, Xt=Xt)
    assert_equal(transp_Xs.shape, Xs.shape)

    # test unsupervised vs semi-supervised mode
    otda_unsup = ot.da.SinkhornL1l2Transport()
    otda_unsup.fit(Xs=Xs, ys=ys, Xt=Xt)
    n_unsup = nx.sum(otda_unsup.cost_)

    otda_semi = ot.da.SinkhornL1l2Transport()
    otda_semi.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt_semi)
    assert_equal(otda_semi.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    n_semisup = nx.sum(otda_semi.cost_)

    # check that the cost matrix norms are indeed different
    assert np.allclose(
        n_unsup, n_semisup, atol=1e-7
    ), "semisupervised mode is not working"

    # check that the coupling forbids mass transport between labeled source
    # and labeled target samples
    mass_semi = nx.sum(otda_semi.coupling_[otda_semi.cost_ == otda_semi.limit_max])
    mass_semi = otda_semi.coupling_[otda_semi.cost_ == otda_semi.limit_max]
    assert_allclose(
        nx.to_numpy(mass_semi), np.zeros_like(mass_semi), rtol=1e-9, atol=1e-9
    )

    # check everything runs well with log=True
    otda = ot.da.SinkhornL1l2Transport(log=True)
    otda.fit(Xs=Xs, ys=ys, Xt=Xt)
    assert len(otda.log_.keys()) != 0


@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
def test_sinkhorn_transport_class(nx):
    """test_sinkhorn_transport"""

    ns = 50
    nt = 50

    Xs, ys = make_data_classif("3gauss", ns)
    Xt, yt = make_data_classif("3gauss2", nt)

    Xs, ys, Xt, yt = nx.from_numpy(Xs, ys, Xt, yt)

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
        nx.to_numpy(nx.sum(otda.coupling_, axis=0)), mu_t, rtol=1e-3, atol=1e-3
    )
    assert_allclose(
        nx.to_numpy(nx.sum(otda.coupling_, axis=1)), mu_s, rtol=1e-3, atol=1e-3
    )

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    assert_equal(transp_Xs.shape, Xs.shape)

    Xs_new = nx.from_numpy(make_data_classif("3gauss", ns + 1)[0])
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

    Xt_new = nx.from_numpy(make_data_classif("3gauss2", nt + 1)[0])
    transp_Xt_new = otda.inverse_transform(Xt=Xt_new)

    # check that the oos method is working
    assert_equal(transp_Xt_new.shape, Xt_new.shape)

    # test fit_transform
    transp_Xs = otda.fit_transform(Xs=Xs, Xt=Xt)
    assert_equal(transp_Xs.shape, Xs.shape)

    # test unsupervised vs semi-supervised mode
    otda_unsup = ot.da.SinkhornTransport()
    otda_unsup.fit(Xs=Xs, Xt=Xt)
    n_unsup = nx.sum(otda_unsup.cost_)

    otda_semi = ot.da.SinkhornTransport()
    otda_semi.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)
    assert_equal(otda_semi.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    n_semisup = nx.sum(otda_semi.cost_)

    # check that the cost matrix norms are indeed different
    assert np.allclose(
        n_unsup, n_semisup, atol=1e-7
    ), "semisupervised mode is not working"

    # check that the coupling forbids mass transport between labeled source
    # and labeled target samples
    mass_semi = nx.sum(otda_semi.coupling_[otda_semi.cost_ == otda_semi.limit_max])
    assert mass_semi == 0, "semisupervised mode not working"

    # check everything runs well with log=True
    otda = ot.da.SinkhornTransport(log=True)
    otda.fit(Xs=Xs, ys=ys, Xt=Xt)
    assert len(otda.log_.keys()) != 0

    # test diffeernt transform and inverse transform
    otda = ot.da.SinkhornTransport(out_of_sample_map="ferradans")
    transp_Xs = otda.fit_transform(Xs=Xs, Xt=Xt)
    assert_equal(transp_Xs.shape, Xs.shape)
    transp_Xt = otda.inverse_transform(Xt=Xt)
    assert_equal(transp_Xt.shape, Xt.shape)

    # test diffeernt transform
    otda = ot.da.SinkhornTransport(out_of_sample_map="continuous", method="sinkhorn")
    transp_Xs2 = otda.fit_transform(Xs=Xs, Xt=Xt)
    assert_equal(transp_Xs2.shape, Xs.shape)
    transp_Xt2 = otda.inverse_transform(Xt=Xt)
    assert_equal(transp_Xt2.shape, Xt.shape)

    np.testing.assert_almost_equal(
        nx.to_numpy(transp_Xs), nx.to_numpy(transp_Xs2), decimal=5
    )
    np.testing.assert_almost_equal(
        nx.to_numpy(transp_Xt), nx.to_numpy(transp_Xt2), decimal=5
    )

    with pytest.raises(ValueError):
        otda = ot.da.SinkhornTransport(out_of_sample_map="unknown")


@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
def test_unbalanced_sinkhorn_transport_class(nx):
    """test_sinkhorn_transport"""

    ns = 50
    nt = 50

    Xs, ys = make_data_classif("3gauss", ns)
    Xt, yt = make_data_classif("3gauss2", nt)

    Xs, ys, Xt, yt = nx.from_numpy(Xs, ys, Xt, yt)

    for log in [True, False]:
        otda = ot.da.UnbalancedSinkhornTransport(log=log)

        # test its computed
        otda.fit(Xs=Xs, Xt=Xt)
        assert hasattr(otda, "cost_")
        assert hasattr(otda, "coupling_")
        assert hasattr(otda, "log_")

        # test dimensions of coupling
        assert_equal(otda.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
        assert_equal(otda.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))
        assert not np.any(np.isnan(nx.to_numpy(otda.cost_))), "cost is finite"

        # test coupling
        assert np.all(np.isfinite(nx.to_numpy(otda.coupling_))), "coupling is finite"

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

        Xs_new = nx.from_numpy(make_data_classif("3gauss", ns + 1)[0])
        transp_Xs_new = otda.transform(Xs_new)

        # check that the oos method is working
        assert_equal(transp_Xs_new.shape, Xs_new.shape)

        # test inverse transform
        transp_Xt = otda.inverse_transform(Xt=Xt)
        assert_equal(transp_Xt.shape, Xt.shape)

        Xt_new = nx.from_numpy(make_data_classif("3gauss2", nt + 1)[0])
        transp_Xt_new = otda.inverse_transform(Xt=Xt_new)

        # check that the oos method is working
        assert_equal(transp_Xt_new.shape, Xt_new.shape)

        # test fit_transform
        transp_Xs = otda.fit_transform(Xs=Xs, Xt=Xt)
        assert_equal(transp_Xs.shape, Xs.shape)

        # test unsupervised vs semi-supervised mode
        otda_unsup = ot.da.SinkhornTransport()
        otda_unsup.fit(Xs=Xs, Xt=Xt)
        assert not np.any(np.isnan(nx.to_numpy(otda_unsup.cost_))), "cost is finite"
        n_unsup = nx.sum(otda_unsup.cost_)

        otda_semi = ot.da.SinkhornTransport()
        otda_semi.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)
        assert not np.any(np.isnan(nx.to_numpy(otda_semi.cost_))), "cost is finite"
        assert_equal(otda_semi.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
        n_semisup = nx.sum(otda_semi.cost_)

        # check that the cost matrix norms are indeed different
        assert np.allclose(
            n_unsup, n_semisup, atol=1e-7
        ), "semisupervised mode is not working"

        # check everything runs well with log=True
        otda = ot.da.SinkhornTransport(log=True)
        otda.fit(Xs=Xs, ys=ys, Xt=Xt)
        assert not np.any(np.isnan(nx.to_numpy(otda.cost_))), "cost is finite"
        assert len(otda.log_.keys()) != 0


@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
def test_emd_transport_class(nx):
    """test_sinkhorn_transport"""

    ns = 50
    nt = 50

    Xs, ys = make_data_classif("3gauss", ns)
    Xt, yt = make_data_classif("3gauss2", nt)

    Xs, ys, Xt, yt = nx.from_numpy(Xs, ys, Xt, yt)

    otda = ot.da.EMDTransport()

    # test its computed
    otda.fit(Xs=Xs, Xt=Xt)
    assert hasattr(otda, "cost_")
    assert hasattr(otda, "coupling_")

    # test dimensions of coupling
    assert_equal(otda.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert not np.any(np.isnan(nx.to_numpy(otda.cost_))), "cost is finite"
    assert_equal(otda.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert np.all(np.isfinite(nx.to_numpy(otda.coupling_))), "coupling is finite"

    # test margin constraints
    mu_s = unif(ns)
    mu_t = unif(nt)
    assert_allclose(
        nx.to_numpy(nx.sum(otda.coupling_, axis=0)), mu_t, rtol=1e-3, atol=1e-3
    )
    assert_allclose(
        nx.to_numpy(nx.sum(otda.coupling_, axis=1)), mu_s, rtol=1e-3, atol=1e-3
    )

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    assert_equal(transp_Xs.shape, Xs.shape)

    Xs_new = nx.from_numpy(make_data_classif("3gauss", ns + 1)[0])
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

    Xt_new = nx.from_numpy(make_data_classif("3gauss2", nt + 1)[0])
    transp_Xt_new = otda.inverse_transform(Xt=Xt_new)

    # check that the oos method is working
    assert_equal(transp_Xt_new.shape, Xt_new.shape)

    # test fit_transform
    transp_Xs = otda.fit_transform(Xs=Xs, Xt=Xt)
    assert_equal(transp_Xs.shape, Xs.shape)

    # test unsupervised vs semi-supervised mode
    otda_unsup = ot.da.EMDTransport()
    otda_unsup.fit(Xs=Xs, ys=ys, Xt=Xt)
    assert_equal(otda_unsup.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert not np.any(np.isnan(nx.to_numpy(otda_unsup.cost_))), "cost is finite"
    assert_equal(otda_unsup.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert np.all(np.isfinite(nx.to_numpy(otda_unsup.coupling_))), "coupling is finite"
    n_unsup = nx.sum(otda_unsup.cost_)

    otda_semi = ot.da.EMDTransport()
    otda_semi.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)
    assert_equal(otda_semi.cost_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert not np.any(np.isnan(nx.to_numpy(otda_semi.cost_))), "cost is finite"
    assert_equal(otda_semi.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))
    assert np.all(np.isfinite(nx.to_numpy(otda_semi.coupling_))), "coupling is finite"
    n_semisup = nx.sum(otda_semi.cost_)

    # check that the cost matrix norms are indeed different
    assert np.allclose(
        n_unsup, n_semisup, atol=1e-7
    ), "semisupervised mode is not working"

    # check that the coupling forbids mass transport between labeled source
    # and labeled target samples
    mass_semi = nx.sum(otda_semi.coupling_[otda_semi.cost_ == otda_semi.limit_max])
    mass_semi = otda_semi.coupling_[otda_semi.cost_ == otda_semi.limit_max]

    # we need to use a small tolerance here, otherwise the test breaks
    assert_allclose(
        nx.to_numpy(mass_semi), np.zeros(list(mass_semi.shape)), rtol=1e-2, atol=1e-2
    )


@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
@pytest.mark.parametrize("kernel", ["linear", "gaussian"])
@pytest.mark.parametrize("bias", ["unbiased", "biased"])
def test_mapping_transport_class(nx, kernel, bias):
    """test_mapping_transport"""

    ns = 20
    nt = 30

    Xs, ys = make_data_classif("3gauss", ns)
    Xt, yt = make_data_classif("3gauss2", nt)
    Xs_new, _ = make_data_classif("3gauss", ns + 1)

    Xs, Xt, Xs_new = nx.from_numpy(Xs, Xt, Xs_new)

    # Mapping tests
    bias = bias == "biased"
    otda = ot.da.MappingTransport(kernel=kernel, bias=bias)
    otda.fit(Xs=Xs, Xt=Xt)
    assert hasattr(otda, "coupling_")
    assert hasattr(otda, "mapping_")
    assert hasattr(otda, "log_")

    assert_equal(otda.coupling_.shape, ((Xs.shape[0], Xt.shape[0])))
    S = Xs.shape[0] if kernel == "gaussian" else Xs.shape[1]  # if linear
    if bias:
        S += 1
    assert_equal(otda.mapping_.shape, ((S, Xt.shape[1])))

    # test margin constraints
    mu_s = unif(ns)
    mu_t = unif(nt)
    assert_allclose(
        nx.to_numpy(nx.sum(otda.coupling_, axis=0)), mu_t, rtol=1e-3, atol=1e-3
    )
    assert_allclose(
        nx.to_numpy(nx.sum(otda.coupling_, axis=1)), mu_s, rtol=1e-3, atol=1e-3
    )

    # test transform
    transp_Xs = otda.transform(Xs=Xs)
    assert_equal(transp_Xs.shape, Xs.shape)

    transp_Xs_new = otda.transform(Xs_new)

    # check that the oos method is working
    assert_equal(transp_Xs_new.shape, Xs_new.shape)

    # check everything runs well with log=True
    otda = ot.da.MappingTransport(kernel=kernel, bias=bias, log=True)
    otda.fit(Xs=Xs, Xt=Xt)
    assert len(otda.log_.keys()) != 0


@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
def test_mapping_transport_class_specific_seed(nx):
    # check that it does not crash when derphi is very close to 0
    ns = 20
    nt = 30
    rng = np.random.RandomState(39)
    Xs, ys = make_data_classif("3gauss", ns, random_state=rng)
    Xt, yt = make_data_classif("3gauss2", nt, random_state=rng)
    otda = ot.da.MappingTransport(kernel="gaussian", bias=False)
    otda.fit(Xs=nx.from_numpy(Xs), Xt=nx.from_numpy(Xt))


@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
def test_linear_mapping_class(nx):
    ns = 50
    nt = 50

    Xs, ys = make_data_classif("3gauss", ns)
    Xt, yt = make_data_classif("3gauss2", nt)

    Xsb, Xtb = nx.from_numpy(Xs, Xt)

    for log in [True, False]:
        otmap = ot.da.LinearTransport(log=log)

        otmap.fit(Xs=Xsb, Xt=Xtb)
        assert hasattr(otmap, "A_")
        assert hasattr(otmap, "B_")
        assert hasattr(otmap, "A1_")
        assert hasattr(otmap, "B1_")

        Xst = nx.to_numpy(otmap.transform(Xs=Xsb))

        Ct = np.cov(Xt.T)
        Cst = np.cov(Xst.T)

        np.testing.assert_allclose(Ct, Cst, rtol=1e-2, atol=1e-2)

        Xts = nx.to_numpy(otmap.inverse_transform(Xt=Xtb))

        Cs = np.cov(Xs.T)
        Cts = np.cov(Xts.T)

        np.testing.assert_allclose(Cs, Cts, rtol=1e-2, atol=1e-2)


@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
def test_linear_gw_mapping_class(nx):
    ns = 50
    nt = 50

    Xs, ys = make_data_classif("3gauss", ns)
    Xt, yt = make_data_classif("3gauss2", nt)

    Xsb, Xtb = nx.from_numpy(Xs, Xt)

    for log in [True, False]:
        otmap = ot.da.LinearGWTransport(log=log)

        otmap.fit(Xs=Xsb, Xt=Xtb)
        assert hasattr(otmap, "A_")
        assert hasattr(otmap, "B_")
        assert hasattr(otmap, "A1_")
        assert hasattr(otmap, "B1_")

        Xst = nx.to_numpy(otmap.transform(Xs=Xsb))

        Ct = np.cov(Xt.T)
        Cst = np.cov(Xst.T)

        np.testing.assert_allclose(Ct, Cst, rtol=1e-2, atol=1e-2)


@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
def test_jcpot_transport_class(nx):
    """test_jcpot_transport"""

    ns1 = 50
    ns2 = 50
    nt = 50

    Xs1, ys1 = make_data_classif("3gauss", ns1)
    Xs2, ys2 = make_data_classif("3gauss", ns2)

    Xt, yt = make_data_classif("3gauss2", nt)

    Xs1, ys1, Xs2, ys2, Xt, yt = nx.from_numpy(Xs1, ys1, Xs2, ys2, Xt, yt)

    Xs = [Xs1, Xs2]
    ys = [ys1, ys2]

    for log in [True, False]:
        otda = ot.da.JCPOTTransport(
            reg_e=1, max_iter=10000, tol=1e-9, verbose=True, log=log
        )

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
                nx.to_numpy(nx.sum(otda.coupling_[i], axis=0)),
                mu_t,
                rtol=1e-3,
                atol=1e-3,
            )

            if log:
                # test margin constraints w.r.t. modified source weights for each source domain

                assert_allclose(
                    nx.to_numpy(
                        nx.dot(otda.log_["D1"][i], nx.sum(otda.coupling_[i], axis=1))
                    ),
                    nx.to_numpy(otda.proportions_),
                    rtol=1e-3,
                    atol=1e-3,
                )

        # test transform
        transp_Xs = otda.transform(Xs=Xs)
        [assert_equal(x.shape, y.shape) for x, y in zip(transp_Xs, Xs)]

        Xs_new = nx.from_numpy(make_data_classif("3gauss", ns1 + 1)[0])
        transp_Xs_new = otda.transform(Xs_new)

        # check that the oos method is working
        assert_equal(transp_Xs_new.shape, Xs_new.shape)

        # check label propagation
        transp_yt = otda.transform_labels(ys)
        assert_equal(transp_yt.shape[0], yt.shape[0])
        assert_equal(transp_yt.shape[1], len(np.unique(nx.to_numpy(*ys))))

        # check inverse label propagation
        transp_ys = otda.inverse_transform_labels(yt)
        for x, y in zip(transp_ys, ys):
            assert_equal(x.shape[0], y.shape[0])
            assert_equal(x.shape[1], len(np.unique(nx.to_numpy(y))))


def test_jcpot_barycenter(nx):
    """test_jcpot_barycenter"""

    ns1 = 50
    ns2 = 50
    nt = 50

    sigma = 0.1

    ps1 = 0.2
    ps2 = 0.9
    pt = 0.4

    Xs1, ys1 = make_data_classif("2gauss_prop", ns1, nz=sigma, p=ps1)
    Xs2, ys2 = make_data_classif("2gauss_prop", ns2, nz=sigma, p=ps2)
    Xt, _ = make_data_classif("2gauss_prop", nt, nz=sigma, p=pt)

    Xs1b, ys1b, Xs2b, ys2b, Xtb = nx.from_numpy(Xs1, ys1, Xs2, ys2, Xt)

    Xsb = [Xs1b, Xs2b]
    ysb = [ys1b, ys2b]

    prop = ot.bregman.jcpot_barycenter(
        Xsb,
        ysb,
        Xtb,
        reg=0.5,
        metric="sqeuclidean",
        numItermax=10000,
        stopThr=1e-9,
        verbose=False,
        log=False,
    )

    np.testing.assert_allclose(nx.to_numpy(prop), [1 - pt, pt], rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(nosklearn, reason="No sklearn available")
@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
def test_emd_laplace_class(nx):
    """test_emd_laplace_transport"""
    ns = 50
    nt = 50

    Xs, ys = make_data_classif("3gauss", ns)
    Xt, yt = make_data_classif("3gauss2", nt)

    Xs, ys, Xt, yt = nx.from_numpy(Xs, ys, Xt, yt)

    for log in [True, False]:
        otda = ot.da.EMDLaplaceTransport(
            reg_lap=0.01, max_iter=1000, tol=1e-9, verbose=False, log=log
        )

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
            nx.to_numpy(nx.sum(otda.coupling_, axis=0)), mu_t, rtol=1e-3, atol=1e-3
        )
        assert_allclose(
            nx.to_numpy(nx.sum(otda.coupling_, axis=1)), mu_s, rtol=1e-3, atol=1e-3
        )

        # test transform
        transp_Xs = otda.transform(Xs=Xs)
        [assert_equal(x.shape, y.shape) for x, y in zip(transp_Xs, Xs)]

        Xs_new = nx.from_numpy(make_data_classif("3gauss", ns + 1)[0])
        transp_Xs_new = otda.transform(Xs_new)

        # check that the oos method is working
        assert_equal(transp_Xs_new.shape, Xs_new.shape)

        # test inverse transform
        transp_Xt = otda.inverse_transform(Xt=Xt)
        assert_equal(transp_Xt.shape, Xt.shape)

        Xt_new = nx.from_numpy(make_data_classif("3gauss2", nt + 1)[0])
        transp_Xt_new = otda.inverse_transform(Xt=Xt_new)

        # check that the oos method is working
        assert_equal(transp_Xt_new.shape, Xt_new.shape)

        # test fit_transform
        transp_Xs = otda.fit_transform(Xs=Xs, Xt=Xt)
        assert_equal(transp_Xs.shape, Xs.shape)

        # check label propagation
        transp_yt = otda.transform_labels(ys)
        assert_equal(transp_yt.shape[0], yt.shape[0])
        assert_equal(transp_yt.shape[1], len(np.unique(nx.to_numpy(ys))))

        # check inverse label propagation
        transp_ys = otda.inverse_transform_labels(yt)
        assert_equal(transp_ys.shape[0], ys.shape[0])
        assert_equal(transp_ys.shape[1], len(np.unique(nx.to_numpy(yt))))


@pytest.mark.skipif(nocvxpy, reason="No CVXPY available")
def test_nearest_brenier_potential(nx):
    X = nx.ones((2, 2))
    for ssnb in [
        ot.da.NearestBrenierPotential(log=True),
        ot.da.NearestBrenierPotential(log=False),
    ]:
        ssnb.fit(Xs=X, Xt=X)
        G_lu = ssnb.transform(Xs=X)
        # 'new' input isn't new, so should be equal to target
        np.testing.assert_almost_equal(nx.to_numpy(G_lu[0]), nx.to_numpy(X))
        np.testing.assert_almost_equal(nx.to_numpy(G_lu[1]), nx.to_numpy(X))


@pytest.mark.skipif(nosklearn, reason="No sklearn available")
@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
def test_emd_laplace(nx):
    """Complements :code:`test_emd_laplace_class` for uncovered options in :code:`emd_laplace`"""
    ns = 50
    nt = 50

    Xs, ys = make_data_classif("3gauss", ns)
    Xt, yt = make_data_classif("3gauss2", nt)

    Xs, ys, Xt, yt = nx.from_numpy(Xs, ys, Xt, yt)
    M = ot.dist(Xs, Xt)
    with pytest.raises(ValueError):
        ot.da.emd_laplace(
            ot.unif(ns), ot.unif(nt), Xs, Xt, M, sim_param=["INVALID", "INPUT", 2]
        )
    with pytest.raises(ValueError):
        ot.da.emd_laplace(
            ot.unif(ns), ot.unif(nt), Xs, Xt, M, sim=["INVALID", "INPUT", 2]
        )

    # test all margin constraints with gaussian similarity and disp regularisation
    coupling = ot.da.emd_laplace(
        ot.unif(ns, type_as=Xs),
        ot.unif(nt, type_as=Xs),
        Xs,
        Xt,
        M,
        sim="gauss",
        reg="disp",
    )

    assert_allclose(
        nx.to_numpy(nx.sum(coupling, axis=0)), unif(nt), rtol=1e-3, atol=1e-3
    )
    assert_allclose(
        nx.to_numpy(nx.sum(coupling, axis=1)), unif(ns), rtol=1e-3, atol=1e-3
    )


@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
def test_sinkhorn_l1l2_gl_cost_vectorized(nx):
    n_samples, n_labels = 150, 3
    rng = np.random.RandomState(42)
    G = rng.rand(n_samples, n_samples)
    labels_a = rng.randint(n_labels, size=(n_samples,))
    G, labels_a = nx.from_numpy(G), nx.from_numpy(labels_a)

    # previously used implementation for the cost estimator
    lstlab = nx.unique(labels_a)

    def f(G):
        res = 0
        for i in range(G.shape[1]):
            for lab in lstlab:
                temp = G[labels_a == lab, i]
                res += nx.norm(temp)
        return res

    def df(G):
        W = nx.zeros(G.shape, type_as=G)
        for i in range(G.shape[1]):
            for lab in lstlab:
                temp = G[labels_a == lab, i]
                n = nx.norm(temp)
                if n:
                    W[labels_a == lab, i] = temp / n
        return W

    # new vectorized implementation for the cost estimator
    labels_u, labels_idx = nx.unique(labels_a, return_inverse=True)
    n_labels = labels_u.shape[0]
    unroll_labels_idx = nx.eye(n_labels, type_as=labels_u)[None, labels_idx]

    def f2(G):
        G_split = nx.repeat(G.T[:, :, None], n_labels, axis=2)
        return nx.sum(nx.norm(G_split * unroll_labels_idx, axis=1))

    def df2(G):
        G_split = nx.repeat(G.T[:, :, None], n_labels, axis=2) * unroll_labels_idx
        W = nx.norm(G_split * unroll_labels_idx, axis=1, keepdims=True)
        G_norm = G_split / nx.clip(W, 1e-12, None)
        return nx.sum(G_norm, axis=2).T

    assert np.allclose(f(G), f2(G))
    assert np.allclose(df(G), df2(G))


@pytest.skip_backend("jax")
@pytest.skip_backend("tf")
def test_sinkhorn_lpl1_vectorization(nx):
    n_samples, n_labels = 150, 3
    rng = np.random.RandomState(42)
    M = rng.rand(n_samples, n_samples)
    labels_a = rng.randint(n_labels, size=(n_samples,))
    M, labels_a = nx.from_numpy(M), nx.from_numpy(labels_a)

    # hard-coded params from the original code
    p, epsilon = 0.5, 1e-3
    T = nx.from_numpy(rng.rand(n_samples, n_samples))

    def unvectorized(transp):
        indices_labels = []
        classes = nx.unique(labels_a)
        for c in classes:
            (idxc,) = nx.where(labels_a == c)
            indices_labels.append(idxc)
        W = nx.ones(M.shape, type_as=M)
        for i, c in enumerate(classes):
            majs = nx.sum(transp[indices_labels[i]], axis=0)
            majs = p * ((majs + epsilon) ** (p - 1))
            W[indices_labels[i]] = majs
        return W

    def vectorized(transp):
        labels_u, labels_idx = nx.unique(labels_a, return_inverse=True)
        n_labels = labels_u.shape[0]
        unroll_labels_idx = nx.eye(n_labels, type_as=transp)[labels_idx]
        W = (
            nx.repeat(transp.T[:, :, None], n_labels, axis=2)
            * unroll_labels_idx[None, :, :]
        )
        W = nx.sum(W, axis=1)
        W = p * ((W + epsilon) ** (p - 1))
        W = nx.dot(W, unroll_labels_idx.T)
        return W.T

    assert np.allclose(unvectorized(T), vectorized(T))
