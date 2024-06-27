# -*- coding: utf-8 -*-
"""
Optimal transport for Gaussian Mixtures
"""

# Author: Eloi Tanguy <eloi.tanguy@u-paris>
#         Remi Flamary <remi.flamary@polytehnique.edu>
#         Julie Delon <julie.delon@math.cnrs.fr>
#
# License: MIT License

import warnings

from .backend import get_backend
from .utils import dots, is_all_finite, list_to_array
from .gaussian import bures_wasserstein_distance, bures_wasserstein_mapping, gaussian_pdf
from .lp import emd2, emd
from scipy.stats import multivariate_normal


def gaussian_pdf(x, m, C):
    var = multivariate_normal(mean=m, cov=C)
    return var.pdf(x)


def gmm_pdf(x, m, C, w):
    nx = get_backend(x, m, C)
    out = nx.zeros((x.shape[0]))
    for k in range(m.shape[0]):
        out = out + w[k] * gaussian_pdf(x, m[k], C[k])
    return out


def dist_bures(m_s, m_t, C_s, C_t):
    r"""
    """
    nx = get_backend(m_s, C_s, m_t, C_t)
    k_s, k_t = m_s.shape[0], m_t.shape[0]
    # TODO assert tailles
    D = nx.zeros((k_s, k_t), type_as=m_s)
    for i in range(k_s):
        for j in range(k_t):
            D[i, j] = bures_wasserstein_distance(m_s[i], m_t[j], C_s[i], C_t[j])
    return D


def gmm_ot_loss(m_s, m_t, C_s, C_t, w_s, w_t):
    r"""
    Gaussian Mixture OT loss
    """
    get_backend(m_s, C_s, w_s, m_t, C_t, w_t)  # backed test
    # TODO assert taille w
    D = dist_bures(m_s, m_t, C_s, C_t)
    return emd2(w_s, w_t, D)


def gmm_ot_plan(m_s, m_t, C_s, C_t, w_s, w_t):
    r"""
    Gaussian Mixture OT loss
    """
    get_backend(m_s, C_s, w_s, m_t, C_t, w_t)  # backed test
    # TODO assert taille w
    D = dist_bures(m_s, m_t, C_s, C_t)
    return emd(w_s, w_t, D)


def gmm_ot_apply_map(x, m_s, m_t, C_s, C_t, w_s, w_t, plan=None,
                     method='bary'):
    r"""
    Applies the barycentric or stochastic map associated to the GMM OT from the
    source GMM to the target GMM
    """
    if plan is None:
        plan = gmm_ot_plan(m_s, m_t, C_s, C_t, w_s, w_t)
    
    # TODO asserts
    normalisation = np.expand_dims(m0.pdf(x), -1)  # from (...) to (..., 1)
    out = nx.zeros_like(x)
    for k0 in range(m0.n_components):
        for k1 in range(m1.n_components):
            g = gaussian_pdf(x, m_s[k0], C_s[k0])[:, None]
            A, b = bures_wasserstein_mapping(m_s[k0], m_t[k1], C_s[k0], C_t[k1])
            Tk0k1x = A @ x + b
            out = out + w[k0, k1] * g * Tk0k1x
    return out / normalisation