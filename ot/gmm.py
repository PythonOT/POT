# -*- coding: utf-8 -*-
"""
Optimal transport for Gaussian Mixtures
"""

# Author: Eloi Tanguy <eloi.tanguy@math.cnrs.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Julie Delon <julie.delon@math.cnrs.fr>
#
# License: MIT License

from .backend import get_backend
from .lp import emd2, emd
import numpy as np
from .utils import dist
from .gaussian import bures_wasserstein_mapping, bures_wasserstein_barycenter


def gaussian_logpdf(x, m, C):
    r"""
    Compute the log of the probability density function of a multivariate
    Gaussian distribution.

    Parameters
    ----------
    x : array-like, shape (..., d)
        The input samples.
    m : array-like, shape (d,)
        The mean vector of the Gaussian distribution.
    C : array-like, shape (d, d)
        The covariance matrix of the Gaussian distribution.

    Returns
    -------
    pdf : array-like, shape (...,)
        The probability density function evaluated at each sample.

    """
    assert (
        x.shape[-1] == m.shape[-1] == C.shape[-1] == C.shape[-2]
    ), "Dimension mismatch"
    nx = get_backend(x, m, C)
    d = m.shape[0]
    diff = x - m
    inv_C = nx.inv(C)
    z = nx.sum(diff * (diff @ inv_C), axis=-1)
    _, log_det_C = nx.slogdet(C)
    return -0.5 * (d * np.log(2 * np.pi) + log_det_C + z)


def gaussian_pdf(x, m, C):
    r"""
    Compute the probability density function of a multivariate
    Gaussian distribution.

    Parameters
    ----------
    x : array-like, shape (..., d)
        The input samples.
    m : array-like, shape (d,)
        The mean vector of the Gaussian distribution.
    C : array-like, shape (d, d)
        The covariance matrix of the Gaussian distribution.

    Returns
    -------
    pdf : array-like, shape (...,)
        The probability density function evaluated at each sample.

    """
    return get_backend(x, m, C).exp(gaussian_logpdf(x, m, C))


def gmm_pdf(x, m, C, w):
    r"""
    Compute the probability density function (PDF) of a
    Gaussian Mixture Model (GMM) at given points.

    Parameters
    ----------
    x : array-like, shape (..., d)
        The input samples.
    m : array-like, shape (n_components, d)
        The means of the Gaussian components.
    C : array-like, shape (n_components, d, d)
        The covariance matrices of the Gaussian components.
    w : array-like, shape (n_components,)
        The weights of the Gaussian components.

    Returns
    -------
    out : array-like, shape (...,)
        The PDF values at the given points.

    """
    assert (
        m.shape[0] == C.shape[0] == w.shape[0]
    ), "All GMM parameters must have the same amount of components"
    nx = get_backend(x, m, C, w)
    out = nx.zeros((x.shape[:-1]))
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

    assert m_s.shape[0] == C_s.shape[0], "Source GMM has different amount of components"

    assert m_t.shape[0] == C_t.shape[0], "Target GMM has different amount of components"

    assert (
        m_s.shape[-1] == m_t.shape[-1] == C_s.shape[-1] == C_t.shape[-1]
    ), "All GMMs must have the same dimension"

    D_means = dist(m_s, m_t, metric="sqeuclidean")

    # C2[i, j] = Cs12[i] @ C_t[j] @ Cs12[i], shape (k_s, k_t, d, d)
    Cs12 = nx.sqrtm(C_s)  # broadcasts matrix sqrt over (k_s,)
    C2 = nx.einsum("ikl,jlm,imn->ijkn", Cs12, C_t, Cs12)
    C = nx.sqrtm(C2)  # broadcasts matrix sqrt over (k_s, k_t)

    # D_covs[i,j] = trace(C_s[i] + C_t[j] - 2C[i,j])
    trace_C_s = nx.einsum("ikk->i", C_s)[:, None]  # (k_s, 1)
    trace_C_t = nx.einsum("ikk->i", C_t)[None, :]  # (1, k_t)
    D_covs = trace_C_s + trace_C_t  # broadcasts to (k_s, k_t)
    D_covs -= 2 * nx.einsum("ijkk->ij", C)

    return nx.maximum(D_means + D_covs, 0)


def gmm_ot_loss(m_s, m_t, C_s, C_t, w_s, w_t, log=False):
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
    log: bool, optional (default=False)
        If True, returns a dictionary containing the cost and dual variables.
        Otherwise returns only the GMM optimal transportation cost.

    Returns
    -------
    loss : float or array-like
        The GMM-OT loss.
    log : dict, optional
        If input log is true, a dictionary containing the
        cost and dual variables and exit status

    References
    ----------
    .. [69] Delon, J., & Desolneux, A. (2020). A Wasserstein-type distance in the space of Gaussian mixture models. SIAM Journal on Imaging Sciences, 13(2), 936-970.

    """
    get_backend(m_s, C_s, w_s, m_t, C_t, w_t)

    assert m_s.shape[0] == w_s.shape[0], "Source GMM has different amount of components"

    assert m_t.shape[0] == w_t.shape[0], "Target GMM has different amount of components"

    D = dist_bures_squared(m_s, m_t, C_s, C_t)
    return emd2(w_s, w_t, D, log=log)


def gmm_ot_plan(m_s, m_t, C_s, C_t, w_s, w_t, log=False):
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
    log : bool, optional (default=False)
        If True, returns a dictionary containing the cost and dual variables.
        Otherwise returns only the GMM optimal transportation matrix.

    Returns
    -------
    plan : array-like, shape (k_s, k_t)
        The GMM-OT plan.
    log : dict, optional
        If input log is true, a dictionary containing the
        cost and dual variables and exit status

    References
    ----------
    .. [69] Delon, J., & Desolneux, A. (2020). A Wasserstein-type distance in the space of Gaussian mixture models. SIAM Journal on Imaging Sciences, 13(2), 936-970.

    """
    get_backend(m_s, C_s, w_s, m_t, C_t, w_t)

    assert m_s.shape[0] == w_s.shape[0], "Source GMM has different amount of components"

    assert m_t.shape[0] == w_t.shape[0], "Target GMM has different amount of components"

    D = dist_bures_squared(m_s, m_t, C_s, C_t)
    return emd(w_s, w_t, D, log=log)


def gmm_ot_apply_map(
    x, m_s, m_t, C_s, C_t, w_s, w_t, plan=None, method="bary", seed=None
):
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

    if method == "bary":
        out = nx.zeros(x.shape)
        logpdf = nx.stack(
            [gaussian_logpdf(x, m_s[k], C_s[k])[:, None] for k in range(k_s)]
        )

        # only need to compute for non-zero plan entries
        for i, j in zip(*nx.where(plan > 0)):
            Cs12 = nx.sqrtm(C_s[i])
            Cs12inv = nx.inv(Cs12)

            M0 = nx.sqrtm(Cs12 @ C_t[j] @ Cs12)
            A = Cs12inv @ M0 @ Cs12inv
            b = m_t[j] - A @ m_s[i]

            # gaussian mapping between components i and j applied to x
            T_ij_x = x @ A + b
            z = w_s[:, None, None] * nx.exp(logpdf - logpdf[i][None, :, :])
            denom = nx.sum(z, axis=0)

            out = out + plan[i, j] * T_ij_x / denom

        return out

    else:  # rand
        # A[i, j] is the linear part of the gaussian mapping between components
        # i and j, b[i, j] is the translation part
        rng = np.random.RandomState(seed)

        A = nx.zeros((k_s, k_t, d, d))
        b = nx.zeros((k_s, k_t, d))

        # only need to compute for non-zero plan entries
        for i, j in zip(*nx.where(plan > 0)):
            Cs12 = nx.sqrtm(C_s[i])
            Cs12inv = nx.inv(Cs12)

            M0 = nx.sqrtm(Cs12 @ C_t[j] @ Cs12)
            A[i, j] = Cs12inv @ M0 @ Cs12inv
            b[i, j] = m_t[j] - A[i, j] @ m_s[i]

        logpdf = nx.stack(
            [gaussian_logpdf(x, m_s[k], C_s[k]) for k in range(k_s)], axis=-1
        )
        # (n_samples, k_s)
        out = nx.zeros(x.shape)

        for i_sample in range(n_samples):
            log_g = logpdf[i_sample]
            log_diff = log_g[:, None] - log_g[None, :]
            weighted_exp = w_s[:, None] * nx.exp(log_diff)
            denom = nx.sum(weighted_exp, axis=0)[:, None] * nx.ones(plan.shape[1])
            p_mat = plan / denom

            p = p_mat.reshape(k_s * k_t)  # stack line-by-line
            # sample between 0 and k_s * k_t - 1
            ij_mat = rng.choice(k_s * k_t, p=p)
            i = ij_mat // k_t
            j = ij_mat % k_t
            out[i_sample] = A[i, j] @ x[i_sample] + b[i, j]

        return out


def gmm_ot_plan_density(x, y, m_s, m_t, C_s, C_t, w_s, w_t, plan=None, atol=1e-2):
    """
    Compute the density of the Gaussian Mixture Model - Optimal Transport
    coupling between GMMS at given points, as introduced in [69].
    Given two arrays of points x and y, the function computes the density at
    each point `(x[i], y[i])` of the product space.

    Parameters
    ----------
    x : array-like, shape (n, d)
        Entry points in source space for density computation.
    y : array-like, shape (m, d)
        Entry points in target space for density computation.
    m_s : array-like, shape (k_s, d)
        The means of the source GMM components.
    m_t : array-like, shape (k_t, d)
        The means of the target GMM components.
    C_s : array-like, shape (k_s, d, d)
        The covariance matrices of the source GMM components.
    C_t : array-like, shape (k_t, d, d)
        The covariance matrices of the target GMM components.
    w_s : array-like, shape (k_s,)
        The weights of the source GMM components.
    w_t : array-like, shape (k_t,)
        The weights of the target GMM components.
    plan : array-like, shape (k_s, k_t), optional
        The optimal transport plan between the source and target GMMs.
        If not provided, it will be computed using `gmm_ot_plan`.
    atol : float, optional
        The absolute tolerance used to determine the support of the GMM-OT
        coupling.

    Returns
    -------
    density : array-like, shape (n, m)
        The density of the GMM-OT coupling between the two GMMs.

    References
    ----------
    .. [69] Delon, J., & Desolneux, A. (2020). A Wasserstein-type distance in the space of Gaussian mixture models. SIAM Journal on Imaging Sciences, 13(2), 936-970.

    """
    assert (
        x.shape[-1] == y.shape[-1]
    ), "x (n, d) and y (m, d) must have the same dimension d"
    n, m = x.shape[0], y.shape[0]
    nx = get_backend(x, y, m_s, m_t, C_s, C_t, w_s, w_t)

    # hand-made d-variate meshgrid in ij indexing
    xx = x[:, None, :] * nx.ones((1, m, 1))  # shapes (n, m, d)
    yy = y[None, :, :] * nx.ones((n, 1, 1))  # shapes (n, m, d)

    if plan is None:
        plan = gmm_ot_plan(m_s, m_t, C_s, C_t, w_s, w_t)

    def Tk0k1(k0, k1):
        A, b = bures_wasserstein_mapping(m_s[k0], m_t[k1], C_s[k0], C_t[k1])
        Tx = xx @ A + b
        g = gaussian_pdf(xx, m_s[k0], C_s[k0])
        out = plan[k0, k1] * g
        norms = nx.norm(Tx - yy, axis=-1)
        out = out * ((norms < atol) * 1.0)
        return out

    mat = nx.stack(
        [
            nx.stack([Tk0k1(k0, k1) for k1 in range(m_t.shape[0])])
            for k0 in range(m_s.shape[0])
        ]
    )
    return nx.sum(mat, axis=(0, 1))


def gmm_barycenter_fixed_point(
    means_list,
    covs_list,
    w_list,
    means_init,
    covs_init,
    weights,
    w_bar=None,
    iterations=100,
    log=False,
    barycentric_proj_method="euclidean",
):
    r"""
    Solves the Gaussian Mixture Model OT barycenter problem (defined in [69])
    using the fixed point algorithm (proposed in [77]). The
    weights of the barycenter are not optimized, and stay the same as the input
    `w_list` or are initialized to uniform.

    The algorithm uses barycentric projections of GMM-OT plans, and these can be
    computed either through Bures Barycenters (slow but accurate,
    barycentric_proj_method='bures') or by convex combination (fast,
    barycentric_proj_method='euclidean', default).

    This is a special case of the generic free-support barycenter solver
    `ot.lp.free_support_barycenter_generic_costs`.

    Parameters
    ----------
    means_list : list of array-like
        List of K (m_k, d) GMM means.
    covs_list : list of array-like
        List of K (m_k, d, d) GMM covariances.
    w_list : list of array-like
        List of K (m_k) arrays of weights.
    means_init : array-like
        Initial (n, d) GMM means.
    covs_init : array-like
        Initial (n, d, d) GMM covariances.
    weights : array-like
        Array (K,) of the barycentre coefficients.
    w_bar : array-like, optional
        Initial weights (n) of the barycentre GMM. If None, initialized to uniform.
    iterations : int, optional
        Number of iterations (default is 100).
    log : bool, optional
        Whether to return the list of iterations (default is False).
    barycentric_proj_method : str, optional
        Method to project the barycentre weights: 'euclidean' (default) or 'bures'.

    Returns
    -------
    means : array-like
        (n, d) barycentre GMM means.
    covs : array-like
        (n, d, d) barycentre GMM covariances.
    log_dict : dict, optional
        Dictionary containing the list of iterations if log is True.

    References
    ----------
    .. [69] Delon, J., & Desolneux, A. (2020). A Wasserstein-type distance in the space of Gaussian mixture models. SIAM Journal on Imaging Sciences, 13(2), 936-970.

    .. [77] Tanguy, Eloi and Delon, Julie and Gozlan, Nathaël (2024). Computing barycenters of Measures for Generic Transport Costs. arXiv preprint 2501.04016 (2024)

    See Also
    --------
    ot.lp.free_support_barycenter_generic_costs : Compute barycenter of measures for generic transport costs.
    """
    nx = get_backend(
        means_init, covs_init, means_list[0], covs_list[0], w_list[0], weights
    )
    K = len(means_list)
    n = means_init.shape[0]
    d = means_init.shape[1]
    means_its = [nx.copy(means_init)]
    covs_its = [nx.copy(covs_init)]
    means, covs = means_init, covs_init

    if w_bar is None:
        w_bar = nx.ones(n, type_as=means) / n

    for _ in range(iterations):
        pi_list = [
            gmm_ot_plan(means, means_list[k], covs, covs_list[k], w_bar, w_list[k])
            for k in range(K)
        ]

        # filled in the euclidean case
        means_selection, covs_selection = None, None

        # in the euclidean case, the selection of Gaussians from each K sources
        # comes from a barycentric projection: it is a convex combination of the
        # selected means and covariances, which can be computed without a
        # for loop on i = 0, ..., n -1
        if barycentric_proj_method == "euclidean":
            means_selection = nx.zeros((n, K, d), type_as=means)
            covs_selection = nx.zeros((n, K, d, d), type_as=means)
            for k in range(K):
                means_selection[:, k, :] = n * pi_list[k] @ means_list[k]
                covs_selection[:, k, :, :] = (
                    nx.einsum("ij,jab->iab", pi_list[k], covs_list[k]) * n
                )

        # each component i of the barycentre will be a Bures barycentre of the
        # selected components of the K GMMs. In the 'bures' barycentric
        # projection option, the selected components are also Bures barycentres.
        for i in range(n):
            # means_selection_i (K, d) is the selected means, each comes from a
            # Gaussian barycentre along the disintegration of pi_k at i
            # covs_selection_i (K, d, d) are the selected covariances
            means_selection_i = None
            covs_selection_i = None

            # use previous computation (convex combination)
            if barycentric_proj_method == "euclidean":
                means_selection_i = means_selection[i]
                covs_selection_i = covs_selection[i]

            # compute Bures barycentre of certain components to get the
            # selection at i
            elif barycentric_proj_method == "bures":
                means_selection_i = nx.zeros((K, d), type_as=means)
                covs_selection_i = nx.zeros((K, d, d), type_as=means)
                for k in range(K):
                    w = (1 / w_bar[i]) * pi_list[k][i, :]
                    m, C = bures_wasserstein_barycenter(means_list[k], covs_list[k], w)
                    means_selection_i[k] = m
                    covs_selection_i[k] = C

            else:
                raise ValueError("Unknown barycentric_proj_method")

            means[i], covs[i] = bures_wasserstein_barycenter(
                means_selection_i, covs_selection_i, weights
            )

        if log:
            means_its.append(nx.copy(means))
            covs_its.append(nx.copy(covs))

    if log:
        return means, covs, {"means_its": means_its, "covs_its": covs_its}
    return means, covs
