"""Regression tests for gradient reuse in symmetric GW/FGW conditional gradient."""

import numpy as np
import pytest

import ot
from ot.backend import NumpyBackend
from ot.gromov._gw import (
    _solve_gromov_linesearch_impl,
    _solve_gromov_linesearch_with_grad,
)
from ot.gromov._utils import gwggrad, init_matrix


def _problem(n=7, m=6, seed=0, loss_fun="square_loss"):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n))
    Y = rng.normal(size=(m, m))
    C1 = np.abs(X - X.T)
    C2 = np.abs(Y - Y.T)
    p = rng.dirichlet(np.ones(n))
    q = rng.dirichlet(np.ones(m))
    M = rng.random((n, m))
    return C1, C2, p, q, M


def test_public_linesearch_keeps_three_value_contract():
    C1, C2, p, q, M = _problem()
    _, hC1, hC2 = init_matrix(C1, C2, p, q)
    G = p[:, None] * q[None, :]
    deltaG = ot.emd(p, q, M) - G
    res = ot.gromov.solve_gromov_linesearch(
        G, deltaG, 1.0, hC1, hC2, M=0.0, reg=1.0, nx=NumpyBackend(), symmetric=True
    )
    assert len(res) == 3


def test_cached_gradient_matches_gwggrad_for_all_alphas_and_losses():
    nx = NumpyBackend()
    for loss_fun in ("square_loss", "kl_loss"):
        C1, C2, p, q, M = _problem(seed=3, loss_fun=loss_fun)
        if loss_fun == "kl_loss":
            C1, C2 = C1 + 1.0, C2 + 1.0
        constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)
        G = p[:, None] * q[None, :]
        deltaG = ot.emd(p, q, M) - G
        df_G = gwggrad(constC, hC1, hC2, G, nx)
        for alpha in (0.0, 0.2, 0.5, 1.0):
            res = _solve_gromov_linesearch_with_grad(
                G,
                deltaG,
                0.0,
                hC1,
                hC2,
                0.0,
                1.0,
                df_G,
                nx=nx,
                symmetric=True,
            )
            # force the requested alpha through alpha_min/alpha_max clipping
            clipped = _solve_gromov_linesearch_with_grad(
                G,
                deltaG,
                0.0,
                hC1,
                hC2,
                0.0,
                1.0,
                df_G,
                alpha_min=alpha,
                alpha_max=alpha,
                nx=nx,
                symmetric=True,
            )
            G_next = G + clipped[0] * deltaG
            expected = gwggrad(constC, hC1, hC2, G_next, nx)
            np.testing.assert_allclose(clipped[3], expected, rtol=1e-12, atol=1e-12)
            assert len(res) == 4


def test_impl_matches_public_linesearch_exactly():
    nx = NumpyBackend()
    C1, C2, p, q, M = _problem()
    _, hC1, hC2 = init_matrix(C1, C2, p, q)
    G = p[:, None] * q[None, :]
    deltaG = ot.emd(p, q, M) - G
    for symmetric in (True, False):
        args = (G, deltaG, 1.37, hC1, hC2, M, 0.4)
        public = ot.gromov.solve_gromov_linesearch(*args, nx=nx, symmetric=symmetric)
        core = _solve_gromov_linesearch_impl(*args, None, None, nx, symmetric)
        assert public == core[:3]


@pytest.mark.parametrize(
    "armijo,symmetric", [(False, True), (False, False), (True, True)]
)
@pytest.mark.parametrize("fgw", [False, True])
def test_solver_paths_unchanged(armijo, symmetric, fgw):
    C1, C2, p, q, M = _problem(seed=8)
    kw = {
        "loss_fun": "square_loss",
        "symmetric": symmetric,
        "armijo": armijo,
        "log": True,
        "max_iter": 60,
        "tol_rel": 1e-9,
        "tol_abs": 1e-11,
    }
    if fgw:
        G, log = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, alpha=0.4, **kw)
        assert "fgw_dist" in log
    else:
        G, log = ot.gromov.gromov_wasserstein(C1, C2, p, q, **kw)
        assert "gw_dist" in log
    np.testing.assert_allclose(G.sum(axis=1), p, atol=1e-9)
    np.testing.assert_allclose(G.sum(axis=0), q, atol=1e-9)
    assert G.min() >= -1e-12
    assert {"loss", "u", "v"}.issubset(log.keys())


def test_reuse_solver_matches_explicit_gradient_reference():
    C1, C2, p, q, _ = _problem(n=9, m=9, seed=11)
    kw = {
        "loss_fun": "square_loss",
        "symmetric": True,
        "log": True,
        "max_iter": 80,
        "tol_rel": 1e-9,
        "tol_abs": 1e-11,
    }
    G_fast, log_fast = ot.gromov.gromov_wasserstein(C1, C2, p, q, **kw)

    # Reference: force the generic solver to recompute df at every iteration by
    # calling the public three-value linesearch explicitly.
    constC, hC1, hC2 = init_matrix(C1, C2, p, q)
    f = lambda T: ot.gromov.gwloss(constC, hC1, hC2, T, NumpyBackend())
    df = lambda T: gwggrad(constC, hC1, hC2, T, NumpyBackend())

    def line_search(cost, Gt, deltaG, Mi, cost_G, df_G, **kwargs):
        return ot.gromov.solve_gromov_linesearch(
            Gt,
            deltaG,
            cost_G,
            hC1,
            hC2,
            M=0.0,
            reg=1.0,
            nx=NumpyBackend(),
            symmetric=True,
            **kwargs,
        )

    G_ref, log_ref = ot.optim.cg(
        p,
        q,
        0.0,
        1.0,
        f,
        df,
        None,
        line_search,
        log=True,
        numItermax=80,
        stopThr=1e-9,
        stopThr2=1e-11,
    )
    np.testing.assert_allclose(G_fast, G_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(log_fast["loss"], log_ref["loss"], rtol=0, atol=1e-12)
