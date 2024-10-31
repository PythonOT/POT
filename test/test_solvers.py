"""Tests for ot solvers"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

import itertools
import numpy as np
import pytest
import sys

import ot
from ot.bregman import geomloss
from ot.backend import torch


lst_reg = [None, 1]
lst_reg_type = ["KL", "entropy", "L2", "tuple"]
lst_unbalanced = [None, 0.9]
lst_unbalanced_type = ["KL", "L2", "TV"]

lst_reg_type_gromov = ["entropy"]
lst_gw_losses = ["L2", "KL"]
lst_unbalanced_type_gromov = ["KL", "semirelaxed", "partial"]
lst_unbalanced_gromov = [None, 0.9]
lst_alpha = [0, 0.4, 0.9, 1]

lst_method_params_solve_sample = [
    {"method": "1d"},
    {"method": "1d", "metric": "euclidean"},
    {"method": "gaussian"},
    {"method": "gaussian", "reg": 1},
    {"method": "factored", "rank": 10},
    {"method": "lowrank", "rank": 10},
]

lst_parameters_solve_sample_NotImplemented = [
    {"method": "1d", "metric": "any other one"},  # fail 1d on weird metrics
    {
        "method": "gaussian",
        "metric": "euclidean",
    },  # fail gaussian on metric not euclidean
    {
        "method": "factored",
        "metric": "euclidean",
    },  # fail factored on metric not euclidean
    {
        "method": "lowrank",
        "metric": "euclidean",
    },  # fail lowrank on metric not euclidean
    {"lazy": True},  # fail lazy for non regularized
    {"lazy": True, "unbalanced": 1},  # fail lazy for non regularized unbalanced
    {
        "lazy": True,
        "reg": 1,
        "unbalanced": 1,
    },  # fail lazy for unbalanced and regularized
]

# set readable ids for each param
lst_method_params_solve_sample = [
    pytest.param(param, id=str(param)) for param in lst_method_params_solve_sample
]
lst_parameters_solve_sample_NotImplemented = [
    pytest.param(param, id=str(param))
    for param in lst_parameters_solve_sample_NotImplemented
]


def assert_allclose_sol(sol1, sol2):
    lst_attr = [
        "value",
        "value_linear",
        "plan",
        "potential_a",
        "potential_b",
        "marginal_a",
        "marginal_b",
    ]

    nx1 = sol1._backend if sol1._backend is not None else ot.backend.NumpyBackend()
    nx2 = sol2._backend if sol2._backend is not None else ot.backend.NumpyBackend()

    for attr in lst_attr:
        if getattr(sol1, attr) is not None and getattr(sol2, attr) is not None:
            try:
                np.allclose(
                    nx1.to_numpy(getattr(sol1, attr)),
                    nx2.to_numpy(getattr(sol2, attr)),
                    equal_nan=True,
                )
            except NotImplementedError:
                pass
        elif getattr(sol1, attr) is None and getattr(sol2, attr) is None:
            return True
        else:
            return False


def test_solve(nx):
    n_samples_s = 10
    n_samples_t = 7
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples_s, n_features)
    y = rng.randn(n_samples_t, n_features)
    a = ot.utils.unif(n_samples_s)
    b = ot.utils.unif(n_samples_t)

    M = ot.dist(x, y)
    reg = 1e-1

    # solve unif weights
    sol0 = ot.solve(M, reg=reg)

    print(sol0)

    # solve signe weights
    sol = ot.solve(M, a, b, reg=reg)

    # check some attributes
    sol.potentials
    sol.sparse_plan
    sol.marginals
    sol.status

    # print("dual = {}".format(sol.potentials))
    # assert_allclose_sol(sol0, sol)

    # solve in backend
    ab, bb, Mb = nx.from_numpy(a, b, M)
    solb = ot.solve(M, a, b)

    assert_allclose_sol(sol, solb)

    # test not implemented unbalanced and check raise
    with pytest.raises(NotImplementedError):
        sol0 = ot.solve(M, unbalanced=1, unbalanced_type="cryptic divergence")

    # test not implemented reg_type and check raise
    with pytest.raises(NotImplementedError):
        sol0 = ot.solve(M, reg=1, reg_type="cryptic divergence")


@pytest.mark.skipif(not torch, reason="torch no installed")
def test_solve_envelope():
    n_samples_s = 10
    n_samples_t = 7
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples_s, n_features)
    y = rng.randn(n_samples_t, n_features)
    a = ot.utils.unif(n_samples_s)
    b = ot.utils.unif(n_samples_t)
    M = ot.dist(x, y)

    a = torch.tensor(a, requires_grad=True)
    b = torch.tensor(b, requires_grad=True)
    M = torch.tensor(M, requires_grad=True)

    sol0 = ot.solve(M, a, b, reg=10, grad="envelope")
    sol0.value.backward()

    gM0 = M.grad.clone()
    ga0 = a.grad.clone()
    gb0 = b.grad.clone()

    a = torch.tensor(a, requires_grad=True)
    b = torch.tensor(b, requires_grad=True)
    M = torch.tensor(M, requires_grad=True)

    sol = ot.solve(M, a, b, reg=10, grad="autodiff")
    sol.value.backward()

    gM = M.grad.clone()
    ga = a.grad.clone()
    gb = b.grad.clone()

    # Note, gradients aer invariant to change in constant so we center them
    assert torch.allclose(gM0, gM)
    assert torch.allclose(ga0 - ga0.mean(), ga - ga.mean())
    assert torch.allclose(gb0 - gb0.mean(), gb - gb.mean())


@pytest.mark.parametrize(
    "reg,reg_type,unbalanced,unbalanced_type",
    itertools.product(lst_reg, lst_reg_type, lst_unbalanced, lst_unbalanced_type),
)
def test_solve_grid(nx, reg, reg_type, unbalanced, unbalanced_type):
    n_samples_s = 10
    n_samples_t = 7
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples_s, n_features)
    y = rng.randn(n_samples_t, n_features)
    a = ot.utils.unif(n_samples_s)
    b = ot.utils.unif(n_samples_t)

    M = ot.dist(x, y)

    try:
        if reg_type == "tuple":

            def f(G):
                return np.sum(G**2)

            def df(G):
                return 2 * G

            reg_type = (f, df)

        # solve unif weights
        sol0 = ot.solve(
            M,
            reg=reg,
            reg_type=reg_type,
            unbalanced=unbalanced,
            unbalanced_type=unbalanced_type,
        )

        # solve signe weights
        sol = ot.solve(
            M,
            a,
            b,
            reg=reg,
            reg_type=reg_type,
            unbalanced=unbalanced,
            unbalanced_type=unbalanced_type,
        )

        assert_allclose_sol(sol0, sol)

        # solve in backend
        ab, bb, Mb = nx.from_numpy(a, b, M)

        if isinstance(reg_type, tuple):

            def f(G):
                return nx.sum(G**2)

            def df(G):
                return 2 * G

            reg_type = (f, df)

        solb = ot.solve(
            Mb,
            ab,
            bb,
            reg=reg,
            reg_type=reg_type,
            unbalanced=unbalanced,
            unbalanced_type=unbalanced_type,
        )

        assert_allclose_sol(sol, solb)

    except NotImplementedError:
        pytest.skip("Not implemented")


def test_solve_not_implemented(nx):
    n_samples_s = 10
    n_samples_t = 7
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples_s, n_features)
    y = rng.randn(n_samples_t, n_features)

    M = ot.dist(x, y)

    # test not implemented and check raise
    with pytest.raises(NotImplementedError):
        ot.solve(M, reg=1.0, reg_type="cryptic divergence")
    with pytest.raises(NotImplementedError):
        ot.solve(M, unbalanced=1.0, unbalanced_type="cryptic divergence")


def test_solve_gromov(nx):
    np.random.seed(0)

    n_samples_s = 3
    n_samples_t = 5

    Ca = np.random.rand(n_samples_s, n_samples_s)
    Ca = (Ca + Ca.T) / 2

    Cb = np.random.rand(n_samples_t, n_samples_t)
    Cb = (Cb + Cb.T) / 2

    a = ot.utils.unif(n_samples_s)
    b = ot.utils.unif(n_samples_t)

    M = np.random.rand(n_samples_s, n_samples_t)

    sol0 = ot.solve_gromov(Ca, Cb)  # GW
    sol = ot.solve_gromov(Ca, Cb, a=a, b=b)  # GW
    sol0_fgw = ot.solve_gromov(Ca, Cb, M)  # FGW

    # check some attributes
    sol.potentials
    sol.marginals

    assert_allclose_sol(sol0, sol)

    # solve in backend
    ax, bx, Mx, Cax, Cbx = nx.from_numpy(a, b, M, Ca, Cb)

    solx = ot.solve_gromov(Cax, Cbx, a=ax, b=bx)  # GW
    solx_fgw = ot.solve_gromov(Cax, Cbx, Mx)  # FGW

    assert_allclose_sol(sol, solx)
    assert_allclose_sol(sol0_fgw, solx_fgw)


@pytest.mark.parametrize(
    "reg,reg_type,unbalanced,unbalanced_type,alpha,loss",
    itertools.product(
        lst_reg,
        lst_reg_type_gromov,
        lst_unbalanced_gromov,
        lst_unbalanced_type_gromov,
        lst_alpha,
        lst_gw_losses,
    ),
)
def test_solve_gromov_grid(nx, reg, reg_type, unbalanced, unbalanced_type, alpha, loss):
    np.random.seed(0)

    n_samples_s = 3
    n_samples_t = 5

    Ca = np.random.rand(n_samples_s, n_samples_s)
    Ca = (Ca + Ca.T) / 2

    Cb = np.random.rand(n_samples_t, n_samples_t)
    Cb = (Cb + Cb.T) / 2

    a = ot.utils.unif(n_samples_s)
    b = ot.utils.unif(n_samples_t)

    M = np.random.rand(n_samples_s, n_samples_t)

    try:
        sol0 = ot.solve_gromov(
            Ca,
            Cb,
            reg=reg,
            reg_type=reg_type,
            unbalanced=unbalanced,
            unbalanced_type=unbalanced_type,
            loss=loss,
        )  # GW
        sol0_fgw = ot.solve_gromov(
            Ca,
            Cb,
            M,
            reg=reg,
            reg_type=reg_type,
            unbalanced=unbalanced,
            unbalanced_type=unbalanced_type,
            alpha=alpha,
            loss=loss,
        )  # FGW

        # solve in backend
        ax, bx, Mx, Cax, Cbx = nx.from_numpy(a, b, M, Ca, Cb)

        solx = ot.solve_gromov(
            Cax,
            Cbx,
            reg=reg,
            reg_type=reg_type,
            unbalanced=unbalanced,
            unbalanced_type=unbalanced_type,
            loss=loss,
        )  # GW
        solx_fgw = ot.solve_gromov(
            Cax,
            Cbx,
            Mx,
            reg=reg,
            reg_type=reg_type,
            unbalanced=unbalanced,
            unbalanced_type=unbalanced_type,
            alpha=alpha,
            loss=loss,
        )  # FGW

        solx.value_quad

        assert_allclose_sol(sol0, solx)
        assert_allclose_sol(sol0_fgw, solx_fgw)

    except NotImplementedError:
        pytest.skip("Not implemented")


def test_solve_gromov_not_implemented(nx):
    np.random.seed(0)

    n_samples_s = 3
    n_samples_t = 5

    Ca = np.random.rand(n_samples_s, n_samples_s)
    Ca = (Ca + Ca.T) / 2

    Cb = np.random.rand(n_samples_t, n_samples_t)
    Cb = (Cb + Cb.T) / 2

    a = ot.utils.unif(n_samples_s)
    b = ot.utils.unif(n_samples_t)

    M = np.random.rand(n_samples_s, n_samples_t)

    Ca, Cb, M, a, b = nx.from_numpy(Ca, Cb, M, a, b)

    # test not implemented and check raise
    with pytest.raises(NotImplementedError):
        ot.solve_gromov(Ca, Cb, loss="weird loss")
    with pytest.raises(NotImplementedError):
        ot.solve_gromov(Ca, Cb, unbalanced=1, unbalanced_type="cryptic divergence")
    with pytest.raises(NotImplementedError):
        ot.solve_gromov(Ca, Cb, reg=1, reg_type="cryptic divergence")

    # detect partial not implemented and error detect in value
    with pytest.raises(ValueError):
        ot.solve_gromov(Ca, Cb, unbalanced_type="partial", unbalanced=1.5)
    with pytest.raises(NotImplementedError):
        ot.solve_gromov(Ca, Cb, M, unbalanced_type="partial", unbalanced=0.5)
    with pytest.raises(ValueError):
        ot.solve_gromov(Ca, Cb, reg=1, unbalanced_type="partial", unbalanced=1.5)


def test_solve_sample(nx):
    # test solve_sample when is_Lazy = False
    n = 20
    X_s = np.reshape(1.0 * np.arange(n), (n, 1))
    X_t = np.reshape(1.0 * np.arange(0, n), (n, 1))

    a = ot.utils.unif(X_s.shape[0])
    b = ot.utils.unif(X_t.shape[0])

    M = ot.dist(X_s, X_t)

    # solve with ot.solve
    sol00 = ot.solve(M, a, b)

    # solve unif weights
    sol0 = ot.solve_sample(X_s, X_t)

    # solve signe weights
    sol = ot.solve_sample(X_s, X_t, a, b)

    # check some attributes
    sol.potentials
    sol.sparse_plan
    sol.marginals
    sol.status

    assert_allclose_sol(sol0, sol)
    assert_allclose_sol(sol0, sol00)

    # solve in backend
    X_sb, X_tb, ab, bb = nx.from_numpy(X_s, X_t, a, b)
    solb = ot.solve_sample(X_sb, X_tb, ab, bb)

    assert_allclose_sol(sol, solb)

    # test not implemented unbalanced and check raise
    with pytest.raises(NotImplementedError):
        sol0 = ot.solve_sample(
            X_s, X_t, unbalanced=1, unbalanced_type="cryptic divergence"
        )

    # test not implemented reg_type and check raise
    with pytest.raises(NotImplementedError):
        sol0 = ot.solve_sample(X_s, X_t, reg=1, reg_type="cryptic divergence")


def test_solve_sample_lazy(nx):
    # test solve_sample when is_Lazy = False
    n = 20
    X_s = np.reshape(1.0 * np.arange(n), (n, 1))
    X_t = np.reshape(1.0 * np.arange(0, n), (n, 1))

    a = ot.utils.unif(X_s.shape[0])
    b = ot.utils.unif(X_t.shape[0])

    X_s, X_t, a, b = nx.from_numpy(X_s, X_t, a, b)

    M = ot.dist(X_s, X_t)

    # solve with ot.solve
    sol00 = ot.solve(M, a, b, reg=1)

    sol0 = ot.solve_sample(X_s, X_t, a, b, reg=1)

    # solve signe weights
    sol = ot.solve_sample(X_s, X_t, a, b, reg=1, lazy=True)

    assert_allclose_sol(sol0, sol00)

    np.testing.assert_allclose(sol0.plan, sol.lazy_plan[:], rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
@pytest.mark.skipif(not geomloss, reason="pytorch not installed")
@pytest.skip_backend("tf")
@pytest.skip_backend("cupy")
@pytest.skip_backend("jax")
@pytest.mark.parametrize("metric", ["sqeuclidean", "euclidean"])
def test_solve_sample_geomloss(nx, metric):
    # test solve_sample when is_Lazy = False
    n_samples_s = 13
    n_samples_t = 7
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples_s, n_features)
    y = rng.randn(n_samples_t, n_features)
    a = ot.utils.unif(n_samples_s)
    b = ot.utils.unif(n_samples_t)

    xb, yb, ab, bb = nx.from_numpy(x, y, a, b)

    sol0 = ot.solve_sample(xb, yb, ab, bb, reg=1)

    # solve signe weights
    sol = ot.solve_sample(xb, yb, ab, bb, reg=1, method="geomloss")
    assert_allclose_sol(sol0, sol)

    sol1 = ot.solve_sample(xb, yb, ab, bb, reg=1, lazy=False, method="geomloss")
    assert_allclose_sol(sol0, sol)

    sol1 = ot.solve_sample(
        xb, yb, ab, bb, reg=1, lazy=True, method="geomloss_tensorized"
    )
    np.testing.assert_allclose(
        nx.to_numpy(sol1.lazy_plan[:]),
        nx.to_numpy(sol.lazy_plan[:]),
        rtol=1e-5,
        atol=1e-5,
    )

    sol1 = ot.solve_sample(xb, yb, ab, bb, reg=1, lazy=True, method="geomloss_online")
    np.testing.assert_allclose(
        nx.to_numpy(sol1.lazy_plan[:]),
        nx.to_numpy(sol.lazy_plan[:]),
        rtol=1e-5,
        atol=1e-5,
    )

    sol1 = ot.solve_sample(
        xb, yb, ab, bb, reg=1, lazy=True, method="geomloss_multiscale"
    )
    np.testing.assert_allclose(
        nx.to_numpy(sol1.lazy_plan[:]),
        nx.to_numpy(sol.lazy_plan[:]),
        rtol=1e-5,
        atol=1e-5,
    )

    sol1 = ot.solve_sample(xb, yb, ab, bb, reg=1, lazy=True, method="geomloss")
    np.testing.assert_allclose(
        nx.to_numpy(sol1.lazy_plan[:]),
        nx.to_numpy(sol.lazy_plan[:]),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("method_params", lst_method_params_solve_sample)
def test_solve_sample_methods(nx, method_params):
    n_samples_s = 20
    n_samples_t = 7
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples_s, n_features)
    y = rng.randn(n_samples_t, n_features)
    a = ot.utils.unif(n_samples_s)
    b = ot.utils.unif(n_samples_t)

    xb, yb, ab, bb = nx.from_numpy(x, y, a, b)

    sol = ot.solve_sample(x, y, **method_params)
    solb = ot.solve_sample(xb, yb, ab, bb, **method_params)

    # check some attributes (no need )
    assert_allclose_sol(sol, solb)

    sol2 = ot.solve_sample(x, x, **method_params)
    if method_params["method"] not in ["factored", "lowrank"]:
        np.testing.assert_allclose(sol2.value, 0)


@pytest.mark.parametrize("method_params", lst_parameters_solve_sample_NotImplemented)
def test_solve_sample_NotImplemented(nx, method_params):
    n_samples_s = 20
    n_samples_t = 7
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples_s, n_features)
    y = rng.randn(n_samples_t, n_features)
    a = ot.utils.unif(n_samples_s)
    b = ot.utils.unif(n_samples_t)

    xb, yb, ab, bb = nx.from_numpy(x, y, a, b)

    with pytest.raises(NotImplementedError):
        ot.solve_sample(xb, yb, ab, bb, **method_params)
