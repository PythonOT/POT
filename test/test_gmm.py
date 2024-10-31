"""Tests for module gaussian"""

# Author: Eloi Tanguy <eloi.tanguy@u-paris>
#         Remi Flamary <remi.flamary@polytehnique.edu>
#         Julie Delon <julie.delon@math.cnrs.fr>
#
# License: MIT License

import numpy as np
import pytest
from ot.utils import proj_simplex
from ot.gmm import (
    gaussian_pdf,
    gmm_pdf,
    dist_bures_squared,
    gmm_ot_loss,
    gmm_ot_plan,
    gmm_ot_apply_map,
    gmm_ot_plan_density,
)

try:
    import torch
except ImportError:
    torch = False


def get_gmms(nx=None):
    rng = np.random.RandomState(seed=42)
    ks = 3
    kt = 5
    d = 3
    m_s = rng.randn(ks, d)
    m_t = rng.randn(kt, d)
    C_s = rng.randn(ks, d, d)
    C_s = np.matmul(C_s, np.transpose(C_s, (0, 2, 1)))
    C_t = rng.randn(kt, d, d)
    C_t = np.matmul(C_t, np.transpose(C_t, (0, 2, 1)))
    w_s = proj_simplex(rng.rand(ks))
    w_t = proj_simplex(rng.rand(kt))
    if nx is not None:
        m_s = nx.from_numpy(m_s)
        m_t = nx.from_numpy(m_t)
        C_s = nx.from_numpy(C_s)
        C_t = nx.from_numpy(C_t)
        w_s = nx.from_numpy(w_s)
        w_t = nx.from_numpy(w_t)
    return m_s, m_t, C_s, C_t, w_s, w_t


def test_gaussian_pdf(nx):
    rng = np.random.RandomState(seed=42)
    n = 7
    d = 3
    x = nx.from_numpy(rng.randn(n, d))
    m, _, C, _, _, _ = get_gmms(nx)
    pdf = gaussian_pdf(x, m[0], C[0])
    assert pdf.shape == (n,)

    x = nx.from_numpy(rng.randn(n, n, d))
    pdf = gaussian_pdf(x, m[0], C[0])
    assert pdf.shape == (
        n,
        n,
    )

    with pytest.raises(AssertionError):
        gaussian_pdf(x, m[0, :-1], C[0])


def test_gmm_pdf(nx):
    rng = np.random.RandomState(seed=42)
    n = 7
    d = 3
    x = nx.from_numpy(rng.randn(n, d))
    m, _, C, _, w, _ = get_gmms(nx)
    pdf = gmm_pdf(x, m, C, w)
    assert pdf.shape == (n,)

    x = nx.from_numpy(rng.randn(n, n, d))
    pdf = gmm_pdf(x, m, C, w)
    assert pdf.shape == (
        n,
        n,
    )

    with pytest.raises(AssertionError):
        gmm_pdf(x, m[:-1], C, w)


@pytest.skip_backend("tf")  # skips because of array assignment
@pytest.skip_backend("jax")
def test_dist_bures_squared(nx):
    m_s, m_t, C_s, C_t, _, _ = get_gmms(nx)
    dist_bures_squared(m_s, m_t, C_s, C_t)
    D0 = dist_bures_squared(m_s, m_s, C_s, C_s)

    assert np.allclose(np.diag(D0), 0, atol=1e-6)

    with pytest.raises(AssertionError):
        dist_bures_squared(m_s[:, 1:], m_t, C_s, C_t)

    with pytest.raises(AssertionError):
        dist_bures_squared(m_s[1:], m_t, C_s, C_t)

    with pytest.raises(AssertionError):
        dist_bures_squared(m_s, m_t[1:], C_s, C_t)


@pytest.skip_backend("tf")  # skips because of array assignment
@pytest.skip_backend("jax")
def test_gmm_ot_loss(nx):
    m_s, m_t, C_s, C_t, w_s, w_t = get_gmms(nx)
    loss = gmm_ot_loss(m_s, m_t, C_s, C_t, w_s, w_t)

    assert loss > 0

    loss = gmm_ot_loss(m_s, m_s, C_s, C_s, w_s, w_s)

    assert np.allclose(loss, 0, atol=1e-6)

    with pytest.raises(AssertionError):
        gmm_ot_loss(m_s, m_t, C_s, C_t, w_s[1:], w_t)

    with pytest.raises(AssertionError):
        gmm_ot_loss(m_s, m_t, C_s, C_t, w_s, w_t[1:])


@pytest.skip_backend("tf")  # skips because of array assignment
@pytest.skip_backend("jax")
def test_gmm_ot_plan(nx):
    m_s, m_t, C_s, C_t, w_s, w_t = get_gmms(nx)

    plan = gmm_ot_plan(m_s, m_t, C_s, C_t, w_s, w_t)

    assert np.allclose(plan.sum(0), w_t, atol=1e-6)
    assert np.allclose(plan.sum(1), w_s, atol=1e-6)

    plan = gmm_ot_plan(m_s, m_s + 1, C_s, C_s, w_s, w_s)

    assert np.allclose(plan, np.diag(w_s), atol=1e-6)

    with pytest.raises(AssertionError):
        gmm_ot_loss(m_s, m_t, C_s, C_t, w_s[1:], w_t)

    with pytest.raises(AssertionError):
        gmm_ot_loss(m_s, m_t, C_s, C_t, w_s, w_t[1:])


def test_gmm_apply_map():
    m_s, m_t, C_s, C_t, w_s, w_t = get_gmms()
    rng = np.random.RandomState(seed=42)
    x = rng.randn(7, 3)

    for method in ["bary", "rand"]:
        gmm_ot_apply_map(x, m_s, m_t, C_s, C_t, w_s, w_t, method=method)

    plan = gmm_ot_plan(m_s, m_t, C_s, C_t, w_s, w_t)
    gmm_ot_apply_map(x, m_s, m_t, C_s, C_t, w_s, w_t, plan=plan)


@pytest.mark.skipif(not torch, reason="No torch available")
def test_gradient_gmm_ot_loss_pytorch():
    m_s, m_t, C_s, C_t, w_s, w_t = get_gmms()
    m_s = torch.tensor(m_s, requires_grad=True)
    m_t = torch.tensor(m_t, requires_grad=True)
    C_s = torch.tensor(C_s, requires_grad=True)
    C_t = torch.tensor(C_t, requires_grad=True)
    w_s = torch.tensor(w_s, requires_grad=True)
    w_t = torch.tensor(w_t, requires_grad=True)
    loss = gmm_ot_loss(m_s, m_t, C_s, C_t, w_s, w_t)
    loss.backward()
    grad_m_s = m_s.grad
    grad_C_s = C_s.grad
    grad_w_s = w_s.grad
    assert (grad_m_s**2).sum().item() > 0
    assert (grad_C_s**2).sum().item() > 0
    assert (grad_w_s**2).sum().item() > 0


def test_gmm_ot_plan_density(nx):
    m_s, m_t, C_s, C_t, w_s, w_t = get_gmms(nx)
    rng = np.random.RandomState(seed=42)
    n = 7
    x = nx.from_numpy(rng.randn(n, 3))
    y = nx.from_numpy(rng.randn(n + 1, 3))

    density = gmm_ot_plan_density(x, y, m_s, m_t, C_s, C_t, w_s, w_t)
    assert density.shape == (n, n + 1)

    plan = gmm_ot_plan(m_s, m_t, C_s, C_t, w_s, w_t)
    gmm_ot_plan_density(x, x, m_s, m_t, C_s, C_t, w_s, w_t, plan=plan)

    with pytest.raises(AssertionError):
        gmm_ot_plan_density(x[:, 1:], y, m_s, m_t, C_s, C_t, w_s, w_t)
