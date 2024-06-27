"""Tests for module gaussian"""

# Author: Eloi Tanguy <eloi.tanguy@u-paris>
#         Remi Flamary <remi.flamary@polytehnique.edu>
#         Julie Delon <julie.delon@math.cnrs.fr>
#
# License: MIT License

import numpy as np
import pytest
import ot
from ot.utils import proj_simplex
from ot.gmm import gaussian_pdf, gmm_pdf, dist_bures


def get_gmms():
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
    return m_s, m_t, C_s, C_t, w_s, w_t


def test_gaussian_pdf():
    rng = np.random.RandomState(seed=42)
    n = 7
    d = 3
    x = rng.randn(n, d)
    m, _, C, _, _, _ = get_gmms()
    p = gaussian_pdf(x, m[0], C[0])


def test_gmm_pdf():
    rng = np.random.RandomState(seed=42)
    n = 7
    d = 3
    x = rng.randn(n, d)
    m_s, _, C_s, _, w_s, _ = get_gmms()
    p = gmm_pdf(x, m_s, C_s, w_s)

def test_dist_bures():
    m_s, m_t, C_s, C_t, _, _ = get_gmms()
    D = dist_bures(m_s, m_t, C_s, C_t)
    D0 = dist_bures(m_s, m_s, C_s, C_s)
    print(D0)
    assert np.allclose(np.diag(D0), 0)
