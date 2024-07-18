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
from .gaussian import bures_wasserstein_mapping


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
    Compute the probability density function (PDF) of a Gaussian Mixture Model (GMM) 
    at given points.

    Parameters
    ----------
    x : array-like, shape (n_samples, d)
        The input samples.
    m : array-like, shape (n_components, d)
        The means of the Gaussian components.
    C : array-like, shape (n_components, d, d)
        The covariance matrices of the Gaussian components.
    w : array-like, shape (n_components,)
        The weights of the Gaussian components.

    Returns
    -------
    out : array-like, shape (n_components,)
        The PDF values at the given points.

    """
    out = np.zeros((x.shape[0]))
    for k in range(m.shape[0]):
        out = out + w[k] * gaussian_pdf(x, m[k], C[k])
    return out


def dist_bures_squared(m_s, m_t, C_s, C_t):
    r"""
    Compute the matrix of the squared Bures distances between the components of
    two Gaussian Mixture Models (GMMs). Used to compute the GMM Optimal
    Transport distance [69].

    Parameters
    ----------
    m_s : array-like, shape (k_s, d)
        Mean vectors of the source GMM.
    m_t : array-like, shape (k_t, d)
        Mean vectors of the target GMM.
    C_s : array-like, shape (k_s, d, d)
        Covariance matrices of the source GMM.
    C_t : array-like, shape (k_t, d, d)
        Covariance matrices of the target GMM.

    Returns
    -------
    dist : array-like, shape (k_s, k_t)
        Matrix of squared Bures distances between the components of the source
        and target GMMs.

    References
    ----------
    .. [69] Delon, J., & Desolneux, A. (2020). A Wasserstein-type distance in the space of Gaussian mixture models. SIAM Journal on Imaging Sciences, 13(2), 936-970.

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
    r"""
    Compute the Gaussian Mixture Model (GMM) Optimal Transport distance between
    two GMMs introduced in [69].

    Parameters
    ----------
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

    Returns
    -------
    loss : float
        The GMM-OT loss.

    References
    ----------
    .. [69] Delon, J., & Desolneux, A. (2020). A Wasserstein-type distance in the space of Gaussian mixture models. SIAM Journal on Imaging Sciences, 13(2), 936-970.

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
    two GMMs introduced in [69].

    Parameters
    ----------
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

    Returns
    -------
    plan : array-like, shape (k_s, k_t)
        The GMM-OT plan.

    References
    ----------
    .. [69] Delon, J., & Desolneux, A. (2020). A Wasserstein-type distance in the space of Gaussian mixture models. SIAM Journal on Imaging Sciences, 13(2), 936-970.

    """
    get_backend(m_s, C_s, w_s, m_t, C_t, w_t)

    assert m_s.shape[0] == w_s.shape[0], \
        "Source GMM has different amount of components"

    assert m_t.shape[0] == w_t.shape[0], \
        "Target GMM has different amount of components"

    D = dist_bures_squared(m_s, m_t, C_s, C_t)
    return emd(w_s, w_t, D)


def gmm_ot_apply_map(x, m_s, m_t, C_s, C_t, w_s, w_t, plan=None,
                     method='bary', seed=None):
    r"""
    Apply Gaussian Mixture Model (GMM) optimal transport (OT) mapping to input
    data. The 'barycentric' mapping corresponds to the barycentric projection
    of the GMM-OT plan, and is called T_bary in [69]. The 'random' mapping takes
    for each input point a random pair (i,j) of components of the GMMs and
    applied the Gaussian map, it is called T_rand in [69].

    Parameters
    ----------
    x : array-like, shape (n_samples, d)
        Input data points.

    m_s : array-like, shape (k_s, d)
        Mean vectors of the source GMM components.

    m_t : array-like, shape (k_t, d)
        Mean vectors of the target GMM components.

    C_s : array-like, shape (k_s, d, d)
        Covariance matrices of the source GMM components.

    C_t : array-like, shape (k_t, d, d)
        Covariance matrices of the target GMM components.

    w_s : array-like, shape (k_s,)
        Weights of the source GMM components.

    w_t : array-like, shape (k_t,)
        Weights of the target GMM components.

    plan : array-like, shape (k_s, k_t), optional
        Optimal transport plan between the source and target GMM components.
        If not provided, it will be computed internally.

    method : {'bary', 'rand'}, optional
        Method for applying the GMM OT mapping. 'bary' uses barycentric mapping,
        while 'rand' uses random sampling. Default is 'bary'.

    seed : int, optional
        Seed for the random number generator. Only used when method='rand'.

    Returns
    -------
    out : array-like, shape (n_samples, d)
        Output data points after applying the GMM OT mapping.

    References
    ----------
    .. [69] Delon, J., & Desolneux, A. (2020). A Wasserstein-type distance in the space of Gaussian mixture models. SIAM Journal on Imaging Sciences, 13(2), 936-970.

    """

    if plan is None:
        plan = gmm_ot_plan(m_s, m_t, C_s, C_t, w_s, w_t)
        nx = get_backend(x, m_s, m_t, C_s, C_t, w_s, w_t)
    else:
        nx = get_backend(x, m_s, m_t, C_s, C_t, w_s, w_t, plan)

    k_s, k_t = m_s.shape[0], m_t.shape[0]
    d = m_s.shape[1]
    n_samples = x.shape[0]

    if method == 'bary':
        # TODO asserts
        normalization = gmm_pdf(x, m_s, C_s, w_s)[:, None]
        out = nx.zeros(x.shape)

        for i in range(k_s):
            Cs12 = nx.sqrtm(C_s[i])
            Cs12inv = nx.inv(Cs12)

            for j in range(k_t):
                g = gaussian_pdf(x, m_s[i], C_s[i])[:, None]

                M0 = nx.sqrtm(Cs12 @ C_t[j] @ Cs12)
                A = Cs12inv @ M0 @ Cs12inv
                b = m_t[j] - A @ m_s[i]

                # gaussian mapping between components i and j applied to x
                T_ij_x = x @ A + b
                out = out + plan[i, j] * g * T_ij_x

        return out / normalization

    else:  # rand
        # A[i, j] is the linear part of the gaussian mapping between components
        # i and j, b[i, j] is the translation part
        rng = np.random.RandomState(seed)

        A = nx.zeros((k_s, k_t, d, d))
        b = nx.zeros((k_s, k_t, d))

        for i in range(k_s):
            Cs12 = nx.sqrtm(C_s[i])
            Cs12inv = nx.inv(Cs12)

            for j in range(k_t):
                M0 = nx.sqrtm(Cs12 @ C_t[j] @ Cs12)
                A[i, j] = Cs12inv @ M0 @ Cs12inv
                b[i, j] = m_t[j] - A[i, j] @ m_s[i]

        normalization = gmm_pdf(x, m_s, C_s, w_s)  # (n_samples,)
        gs = np.stack(
            [gaussian_pdf(x, m_s[i], C_s[i]) for i in range(k_s)], axis=-1)
        # (n_samples, k_s)
        out = nx.zeros(x.shape)

        for i_sample in range(n_samples):
            p_mat = plan * gs[i_sample][:, None] / normalization[i_sample]
            p = p_mat.reshape(k_s * k_t)  # stack line-by-line
            # sample between 0 and k_s * k_t - 1
            ij_mat = rng.choice(k_s * k_t, p=p)
            i = ij_mat // k_t
            j = ij_mat % k_t
            out[i_sample] = A[i, j] @ x[i_sample] + b[i, j]

        return out





def gmm_ot_plan_density(x, y, m_s, m_t, C_s, C_t, w_s, w_t, plan=None, atol=1e-8):
    r"""
        Args:
            m0: gaussian mixture 0
            m1: gaussian mixture 1
            x: (..., d) array-like
            y: (..., d) array-like (same shape as x)
            atol: absolute tolerance for the condition T_kl(x) = y

        Returns:
           density of the MW2 OT plan between m0 and m1 at (x, y)
    """
    
    if plan is None:
        plan = gmm_ot_plan(m_s, m_t, C_s, C_t, w_s, w_t)

    def Tk0k1(k0, k1):
        A, b = bures_wasserstein_mapping(m_s[k0], m_t[k1], C_s[k0], C_t[k1])
        Tx = x @ A + b
        g = gaussian_pdf(x, m_s[k0], C_s[k0])
        out = plan[k0, k1] * g
        norms = np.linalg.norm(Tx - y, axis=-1)
        out[norms > atol] = 0
        return out

    mat = np.array(
        [
            [Tk0k1(k0, k1) for k1 in range(m_t.shape[0])]
            for k0 in range(m_s.shape[0])
        ])
    return np.sum(mat, axis=(0, 1))