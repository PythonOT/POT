# -*- coding: utf-8 -*-
"""
Optimal transport for Gaussian Mixtures
"""

# Author: Eloi Tanguy <eloi.tanguy@u-paris>
#         Remi Flamary <remi.flamary@polytehnique.edu>
#         Julie Delon <julie.delon@math.cnrs.fr>
#
# License: MIT License

from .backend import get_backend
from .lp import emd2, emd
import numpy as np
from .lp import dist


def gaussian_pdf(x, m, C):
    r"""
    Compute the probability density function of a multivariate Gaussian distribution.

    Parameters
    ----------
    x : array-like, shape (n_samples, d)
        The input samples.
    m : array-like, shape (d,)
        The mean vector of the Gaussian distribution.
    C : array-like, shape (d, d)
        The covariance matrix of the Gaussian distribution.

    Returns
    -------
    pdf : array-like, shape (n_samples,)
        The probability density function evaluated at each sample.

    """
    _, d = x.shape
    z = (2 * np.pi) ** (-d / 2) * np.linalg.det(C) ** (-0.5)
    exp = np.exp(-0.5 * np.sum((x - m) @ np.linalg.inv(C) * (x - m), axis=1))
    return z * exp


def gmm_pdf(x, m, C, w):
    r"""
    Compute the probability density function (PDF) of a Gaussian Mixture Model (GMM) at given points.

    Parameters:
    -----------
    x : array-like, shape (n_samples, d)
        The input samples.
    m : array-like, shape (n_components, d)
        The means of the Gaussian components.
    C : array-like, shape (n_components, d, d)
        The covariance matrices of the Gaussian components.
    w : array-like, shape (n_components,)
        The weights of the Gaussian components.

    Returns:
    --------
    out : array-like, shape (n_components,)
        The PDF values at the given points.

    """
    out = np.zeros((x.shape[0]))
    for k in range(m.shape[0]):
        out = out + w[k] * gaussian_pdf(x, m[k], C[k])
    return out


def dist_bures_squared(m_s, m_t, C_s, C_t):
    """
    Compute the matrix of the squared Bures distances between the components of two Gaussian Mixture Models (GMMs).

    Parameters:
    -----------
    m_s : array-like, shape (k_s, d)
        Mean vectors of the source GMM.
    m_t : array-like, shape (k_t, d)
        Mean vectors of the target GMM.
    C_s : array-like, shape (k_s, d, d)
        Covariance matrices of the source GMM.
    C_t : array-like, shape (k_t, d, d)
        Covariance matrices of the target GMM.

    Returns:
    --------
    dist : array-like, shape (k_s, k_t)
        Matrix of squared Bures distances between the components of the source and target GMMs.

    """
    nx = get_backend(m_s, C_s, m_t, C_t)
    k_s, k_t = m_s.shape[0], m_t.shape[0]

    assert m_s.shape[0] == C_s.shape[0], \
        "Source GMM has different amount of components"

    assert m_t.shape[0] == C_t.shape[0], \
        "Target GMM has different amount of components"

    assert m_s.shape[-1] == m_t.shape[-1] == C_s.shape[-1] == C_t.shape[-1], \
        "All GMMs must have the same dimension"

    D_means = dist(m_s, m_t, metric='sqeuclidean')
    D_covs = nx.zeros((k_s, k_t), type_as=m_s)

    for i in range(k_s):
        Cs12 = nx.sqrtm(C_s[i])  # nx.sqrtm is not batchable
        for j in range(k_t):
            C = nx.sqrtm(Cs12 @ C_t[j] @ Cs12)
            D_covs[i, j] = nx.trace(C_s[i] + C_t[j] - 2 * C)

    return nx.maximum(D_means + D_covs, 0)


def gmm_ot_loss(m_s, m_t, C_s, C_t, w_s, w_t):
    """
    Compute the Gaussian Mixture Model (GMM) Optimal Transport distance between
    two GMMs.

    Parameters:
    -----------
    m_s : array-like, shape (k_s, d)
        Mean vectors of the source GMM.
    m_t : array-like, shape (k_t, d)
        Mean vectors of the target GMM.
    C_s : array-like, shape (k_s, d, d)
        Covariance matrices of the source GMM.
    C_t : array-like, shape (k_t, d, d)
        Covariance matrices of the target GMM.
    w_s : array-like, shape (k_s,)
        Weights of the source GMM components.
    w_t : array-like, shape (k_t,)
        Weights of the target GMM components.

    Returns:
    --------
    loss : float
        The GMM-OT loss.

    """
    get_backend(m_s, C_s, w_s, m_t, C_t, w_t)

    assert m_s.shape[0] == w_s.shape[0], \
        "Source GMM has different amount of components"

    assert m_t.shape[0] == w_t.shape[0], \
        "Target GMM has different amount of components"

    D = dist_bures_squared(m_s, m_t, C_s, C_t)
    return emd2(w_s, w_t, D)


def gmm_ot_plan(m_s, m_t, C_s, C_t, w_s, w_t):
    r"""
    Compute the Gaussian Mixture Model (GMM) Optimal Transport plan between
    two GMMs.

    Parameters:
    -----------
    m_s : array-like, shape (k_s, d)
        Mean vectors of the source GMM.
    m_t : array-like, shape (k_t, d)
        Mean vectors of the target GMM.
    C_s : array-like, shape (k_s, d, d)
        Covariance matrices of the source GMM.
    C_t : array-like, shape (k_t, d, d)
        Covariance matrices of the target GMM.
    w_s : array-like, shape (k_s,)
        Weights of the source GMM components.
    w_t : array-like, shape (k_t,)
        Weights of the target GMM components.

    Returns:
    --------
    plan : array-like, shape (k_s, k_t)
        The GMM-OT plan.

    """
    get_backend(m_s, C_s, w_s, m_t, C_t, w_t)

    assert m_s.shape[0] == w_s.shape[0], \
        "Source GMM has different amount of components"

    assert m_t.shape[0] == w_t.shape[0], \
        "Target GMM has different amount of components"

    D = dist_bures_squared(m_s, m_t, C_s, C_t)
    return emd(w_s, w_t, D)


def gmm_ot_apply_map(x, m_s, m_t, C_s, C_t, w_s, w_t, plan=None,
                     method='bary'):
    r"""
    Applies the barycentric or stochastic map associated to the GMM OT from the
    source GMM to the target GMM
    """

    if plan is None:
        plan = gmm_ot_plan(m_s, m_t, C_s, C_t, w_s, w_t)
        nx = get_backend(x, m_s, m_t, C_s, C_t, w_s, w_t)
    else:
        nx = get_backend(x, m_s, m_t, C_s, C_t, w_s, w_t, plan)

    if method == 'bary':
        # TODO asserts
        normalization = gmm_pdf(x, m_s, C_s, w_s)[:, None]
        out = nx.zeros(x.shape)

        for k0 in range(m_s.shape[0]):
            Cs12 = nx.sqrtm(C_s[k0])
            Cs12inv = nx.inv(Cs12)

            for k1 in range(m_t.shape[0]):
                g = gaussian_pdf(x, m_s[k0], C_s[k0])[:, None]

                M0 = nx.sqrtm(Cs12 @ C_t[k1] @ Cs12)
                A = Cs12inv @ M0 @ Cs12inv
                b = m_t[k1] - A @ m_s[k0]

                # gaussian mapping between components k0 and k1 applied to x
                Tk0k1x = x @ A + b
                out = out + plan[k0, k1] * g * Tk0k1x

        return out / normalization

    else:  # rand
        raise NotImplementedError('Mapping {} not implemented'.format(method))
