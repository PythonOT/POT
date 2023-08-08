# -*- coding: utf-8 -*-
"""
Optimal transport for Gaussian distributions
"""

# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytehnique.edu>
#
# License: MIT License

from .backend import get_backend
from .utils import dots
from .utils import list_to_array


def bures_wasserstein_mapping(ms, mt, Cs, Ct, log=False):
    r"""Return OT linear operator between samples.

    The function estimates the optimal linear operator that aligns the two
    empirical distributions. This is equivalent to estimating the closed
    form mapping between two Gaussian distributions :math:`\mathcal{N}(\mu_s,\Sigma_s)`
    and :math:`\mathcal{N}(\mu_t,\Sigma_t)` as proposed in
    :ref:`[1] <references-OT-mapping-linear>` and discussed in remark 2.29 in
    :ref:`[2] <references-OT-mapping-linear>`.

    The linear operator from source to target :math:`M`

    .. math::
        M(\mathbf{x})= \mathbf{A} \mathbf{x} + \mathbf{b}

    where :

    .. math::
        \mathbf{A} &= \Sigma_s^{-1/2} \left(\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2} \right)^{1/2}
        \Sigma_s^{-1/2}

        \mathbf{b} &= \mu_t - \mathbf{A} \mu_s

    Parameters
    ----------
    ms : array-like (d,)
        mean of the source distribution
    mt : array-like (d,)
        mean of the target distribution
    Cs : array-like (d,d)
        covariance of the source distribution
    Ct : array-like (d,d)
        covariance of the target distribution
    log : bool, optional
        record log if True


    Returns
    -------
    A : (d, d) array-like
        Linear operator
    b : (1, d) array-like
        bias
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-OT-mapping-linear:
    References
    ----------
    .. [1] Knott, M. and Smith, C. S. "On the optimal mapping of
        distributions", Journal of Optimization Theory and Applications
        Vol 43, 1984

    .. [2] Peyré, G., & Cuturi, M. (2017). "Computational Optimal
        Transport", 2018.
    """
    ms, mt, Cs, Ct = list_to_array(ms, mt, Cs, Ct)
    nx = get_backend(ms, mt, Cs, Ct)

    Cs12 = nx.sqrtm(Cs)
    Cs12inv = nx.inv(Cs12)

    M0 = nx.sqrtm(dots(Cs12, Ct, Cs12))

    A = dots(Cs12inv, M0, Cs12inv)

    b = mt - nx.dot(ms, A)

    if log:
        log = {}
        log['Cs12'] = Cs12
        log['Cs12inv'] = Cs12inv
        return A, b, log
    else:
        return A, b


def empirical_bures_wasserstein_mapping(xs, xt, reg=1e-6, ws=None,
                                        wt=None, bias=True, log=False):
    r"""Return OT linear operator between samples.

    The function estimates the optimal linear operator that aligns the two
    empirical distributions. This is equivalent to estimating the closed
    form mapping between two Gaussian distributions :math:`\mathcal{N}(\mu_s,\Sigma_s)`
    and :math:`\mathcal{N}(\mu_t,\Sigma_t)` as proposed in
    :ref:`[1] <references-OT-mapping-linear>` and discussed in remark 2.29 in
    :ref:`[2] <references-OT-mapping-linear>`.

    The linear operator from source to target :math:`M`

    .. math::
        M(\mathbf{x})= \mathbf{A} \mathbf{x} + \mathbf{b}

    where :

    .. math::
        \mathbf{A} &= \Sigma_s^{-1/2} \left(\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2} \right)^{1/2}
        \Sigma_s^{-1/2}

        \mathbf{b} &= \mu_t - \mathbf{A} \mu_s

    Parameters
    ----------
    xs : array-like (ns,d)
        samples in the source domain
    xt : array-like (nt,d)
        samples in the target domain
    reg : float,optional
        regularization added to the diagonals of covariances (>0)
    ws : array-like (ns,1), optional
        weights for the source samples
    wt : array-like (ns,1), optional
        weights for the target samples
    bias: boolean, optional
        estimate bias :math:`\mathbf{b}` else :math:`\mathbf{b} = 0` (default:True)
    log : bool, optional
        record log if True


    Returns
    -------
    A : (d, d) array-like
        Linear operator
    b : (1, d) array-like
        bias
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-OT-mapping-linear:
    References
    ----------
    .. [1] Knott, M. and Smith, C. S. "On the optimal mapping of
        distributions", Journal of Optimization Theory and Applications
        Vol 43, 1984

    .. [2] Peyré, G., & Cuturi, M. (2017). "Computational Optimal
        Transport", 2018.
    """
    xs, xt = list_to_array(xs, xt)
    nx = get_backend(xs, xt)

    d = xs.shape[1]

    if bias:
        mxs = nx.mean(xs, axis=0)[None, :]
        mxt = nx.mean(xt, axis=0)[None, :]

        xs = xs - mxs
        xt = xt - mxt
    else:
        mxs = nx.zeros((1, d), type_as=xs)
        mxt = nx.zeros((1, d), type_as=xs)

    if ws is None:
        ws = nx.ones((xs.shape[0], 1), type_as=xs) / xs.shape[0]

    if wt is None:
        wt = nx.ones((xt.shape[0], 1), type_as=xt) / xt.shape[0]

    Cs = nx.dot((xs * ws).T, xs) / nx.sum(ws) + reg * nx.eye(d, type_as=xs)
    Ct = nx.dot((xt * wt).T, xt) / nx.sum(wt) + reg * nx.eye(d, type_as=xt)

    if log:
        A, b, log = bures_wasserstein_mapping(mxs, mxt, Cs, Ct, log=log)
        log['Cs'] = Cs
        log['Ct'] = Ct
        return A, b, log
    else:
        A, b = bures_wasserstein_mapping(mxs, mxt, Cs, Ct)
        return A, b


def bures_wasserstein_distance(ms, mt, Cs, Ct, log=False):
    r"""Return Bures Wasserstein distance between samples.

    The function estimates the Bures-Wasserstein distance between two
    empirical distributions source :math:`\mu_s` and target :math:`\mu_t`,
    discussed in remark 2.31 :ref:`[1] <references-bures-wasserstein-distance>`.

    The Bures Wasserstein distance between source and target distribution :math:`\mathcal{W}`

    .. math::
        \mathcal{W}(\mu_s, \mu_t)_2^2= \left\lVert \mathbf{m}_s - \mathbf{m}_t \right\rVert^2 + \mathcal{B}(\Sigma_s, \Sigma_t)^{2}

    where :

    .. math::
        \mathbf{B}(\Sigma_s, \Sigma_t)^{2} = \text{Tr}\left(\Sigma_s + \Sigma_t - 2 \sqrt{\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2}} \right)

    Parameters
    ----------
    ms : array-like (d,)
        mean of the source distribution
    mt : array-like (d,)
        mean of the target distribution
    Cs : array-like (d,d)
        covariance of the source distribution
    Ct : array-like (d,d)
        covariance of the target distribution
    log : bool, optional
        record log if True


    Returns
    -------
    W : float
        Bures Wasserstein distance
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-bures-wasserstein-distance:
    References
    ----------

    .. [1] Peyré, G., & Cuturi, M. (2017). "Computational Optimal
        Transport", 2018.
    """
    ms, mt, Cs, Ct = list_to_array(ms, mt, Cs, Ct)
    nx = get_backend(ms, mt, Cs, Ct)

    Cs12 = nx.sqrtm(Cs)

    B = nx.trace(Cs + Ct - 2 * nx.sqrtm(dots(Cs12, Ct, Cs12)))
    W = nx.sqrt(nx.norm(ms - mt)**2 + B)
    if log:
        log = {}
        log['Cs12'] = Cs12
        return W, log
    else:
        return W


def empirical_bures_wasserstein_distance(xs, xt, reg=1e-6, ws=None,
                                         wt=None, bias=True, log=False):
    r"""Return Bures Wasserstein distance from mean and covariance of distribution.

    The function estimates the Bures-Wasserstein distance between two
    empirical distributions source :math:`\mu_s` and target :math:`\mu_t`,
    discussed in remark 2.31 :ref:`[1] <references-bures-wasserstein-distance>`.

    The Bures Wasserstein distance between source and target distribution :math:`\mathcal{W}`

    .. math::
        \mathcal{W}(\mu_s, \mu_t)_2^2= \left\lVert \mathbf{m}_s - \mathbf{m}_t \right\rVert^2 + \mathcal{B}(\Sigma_s, \Sigma_t)^{2}

    where :

    .. math::
        \mathbf{B}(\Sigma_s, \Sigma_t)^{2} = \text{Tr}\left(\Sigma_s + \Sigma_t - 2 \sqrt{\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2}} \right)

    Parameters
    ----------
    xs : array-like (ns,d)
        samples in the source domain
    xt : array-like (nt,d)
        samples in the target domain
    reg : float,optional
        regularization added to the diagonals of covariances (>0)
    ws : array-like (ns), optional
        weights for the source samples
    wt : array-like (ns), optional
        weights for the target samples
    bias: boolean, optional
        estimate bias :math:`\mathbf{b}` else :math:`\mathbf{b} = 0` (default:True)
    log : bool, optional
        record log if True


    Returns
    -------
    W : float
        Bures Wasserstein distance
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-bures-wasserstein-distance:
    References
    ----------

    .. [1] Peyré, G., & Cuturi, M. (2017). "Computational Optimal
        Transport", 2018.
    """
    xs, xt = list_to_array(xs, xt)
    nx = get_backend(xs, xt)

    d = xs.shape[1]

    if bias:
        mxs = nx.mean(xs, axis=0)[None, :]
        mxt = nx.mean(xt, axis=0)[None, :]

        xs = xs - mxs
        xt = xt - mxt
    else:
        mxs = nx.zeros((1, d), type_as=xs)
        mxt = nx.zeros((1, d), type_as=xs)

    if ws is None:
        ws = nx.ones((xs.shape[0], 1), type_as=xs) / xs.shape[0]

    if wt is None:
        wt = nx.ones((xt.shape[0], 1), type_as=xt) / xt.shape[0]

    Cs = nx.dot((xs * ws).T, xs) / nx.sum(ws) + reg * nx.eye(d, type_as=xs)
    Ct = nx.dot((xt * wt).T, xt) / nx.sum(wt) + reg * nx.eye(d, type_as=xt)

    if log:
        W, log = bures_wasserstein_distance(mxs, mxt, Cs, Ct, log=log)
        log['Cs'] = Cs
        log['Ct'] = Ct
        return W, log
    else:
        W = bures_wasserstein_distance(mxs, mxt, Cs, Ct)
        return W


def gaussian_gromov_wasserstein_distance(Cov_s, Cov_t, log=False):
    r""" Return the Gaussian Gromov-Wasserstein value from [57].

    This function return the closed form value of the Gaussian Gromov-Wasserstein
    distance between two Gaussian distributions
    :math:`\mathcal{N}(\mu_s,\Sigma_s)` and :math:`\mathcal{N}(\mu_t,\Sigma_t)`
    when the OT plan is assumed to be also Gaussian. See [57] Theorem 4.1 for
    more details.

    Parameters
    ----------
    Cov_s : array-like (ds,ds)
        covariance of the source distribution
    Cov_t : array-like (dt,dt)
        covariance of the target distribution


    Returns
    -------
    G : float
        Gaussian Gromov-Wasserstein distance


    .. _references-gaussien_gromov_wasserstein_distance:
    References
    ----------
    .. [57] Delon, J., Desolneux, A., & Salmona, A. (2022). Gromov–Wasserstein
    distances between Gaussian distributions. Journal of Applied Probability,
    59(4), 1178-1198.
    """

    nx = get_backend(Cov_s, Cov_t)

    # ensure that Cov_s is the largest covariance matrix
    # that is m >= n
    if Cov_s.shape[0] < Cov_t.shape[0]:
        Cov_s, Cov_t = Cov_t, Cov_s

    n = Cov_t.shape[0]

    # compte and sort eigenvalues decerasingly
    d_s = nx.flip(nx.sort(nx.eigh(Cov_s)[0]))
    d_t = nx.flip(nx.sort(nx.eigh(Cov_t)[0]))

    # compute the gaussien Gromov-Wasserstein distance
    res = 4 * (nx.sum(d_s) - nx.sum(d_t))**2 + 8 * nx.sum((d_s[:n] - d_t)**2) + 8 * nx.sum((d_s[n:])**2)
    if log:
        log = {}
        log['d_s'] = d_s
        log['d_t'] = d_t
        return nx.sqrt(res), log
    else:
        return nx.sqrt(res)


def empirical_gaussian_gromov_wasserstein_distance(xs, xt, ws=None,
                                                   wt=None, log=False):
    r"""Return Gaussian Gromov-Wasserstein distance between samples.

    The function estimates the Gaussian Gromov-Wasserstein distance between two
    Gaussien distributions source :math:`\mu_s` and target :math:`\mu_t`, whose
    parameters are estimated from the provided samples :math:`\mathcal{X}_s` and
    :math:`\mathcal{X}_t`. See [57] Theorem 4.1 for more details.

    Parameters
    ----------
    xs : array-like (ns,d)
        samples in the source domain
    xt : array-like (nt,d)
        samples in the target domain
    ws : array-like (ns,1), optional
        weights for the source samples
    wt : array-like (ns,1), optional
        weights for the target samples
    log : bool, optional
        record log if True


    Returns
    -------
    G : float
        Gaussian Gromov-Wasserstein distance


    .. _references-gaussien_gromov_wasserstein:
    References
    ----------
    .. [57] Delon, J., Desolneux, A., & Salmona, A. (2022). Gromov–Wasserstein
    distances between Gaussian distributions. Journal of Applied Probability,
    59(4), 1178-1198.
    """
    xs, xt = list_to_array(xs, xt)
    nx = get_backend(xs, xt)

    if ws is None:
        ws = nx.ones((xs.shape[0], 1), type_as=xs) / xs.shape[0]

    if wt is None:
        wt = nx.ones((xt.shape[0], 1), type_as=xt) / xt.shape[0]

    mxs = nx.dot(ws.T, xs) / nx.sum(ws)
    mxt = nx.dot(wt.T, xt) / nx.sum(wt)

    xs = xs - mxs
    xt = xt - mxt

    Cs = nx.dot((xs * ws).T, xs) / nx.sum(ws)
    Ct = nx.dot((xt * wt).T, xt) / nx.sum(wt)

    if log:
        G, log = gaussian_gromov_wasserstein_distance(Cs, Ct, log=log)
        log['Cov_s'] = Cs
        log['Cov_t'] = Ct
        return G, log
    else:
        G = gaussian_gromov_wasserstein_distance(Cs, Ct)
        return G


def gaussian_gromov_wasserstein_mapping(mu_s, mu_t, Cov_s, Cov_t, sign_eigs=None, log=False):
    r""" Return the Gaussian Gromov-Wasserstein mapping from [57].

    This function return the closed form value of the Gaussian
    Gromov-Wasserstein mapping between two Gaussian distributions
    :math:`\mathcal{N}(\mu_s,\Sigma_s)` and :math:`\mathcal{N}(\mu_t,\Sigma_t)`
    when the OT plan is assumed to be also Gaussian. See [57] Theorem 4.1 for
    more details.

    Parameters
    ----------
    mu_s : array-like (ds,)
        mean of the source distribution
    mu_t : array-like (dt,)
        mean of the target distribution
    Cov_s : array-like (ds,ds)
        covariance of the source distribution
    Cov_t : array-like (dt,dt)
        covariance of the target distribution
    log : bool, optional
        record log if True


    Returns
    -------
    A : (dt, ds) array-like
        Linear operator
    b : (1, dt) array-like
        bias


    .. _references-gaussien_gromov_wasserstein_mapping:
    References
    ----------
    .. [57] Delon, J., Desolneux, A., & Salmona, A. (2022). Gromov–Wasserstein
    distances between Gaussian distributions. Journal of Applied Probability,
    59(4), 1178-1198.
    """

    nx = get_backend(mu_s, mu_t, Cov_s, Cov_t)

    n = Cov_t.shape[0]
    m = Cov_s.shape[0]

    # compte and sort eigenvalues/eigenvectors decreasingly
    d_s, U_s = nx.eigh(Cov_s)
    id_s = nx.flip(nx.argsort(d_s))
    d_s, U_s = d_s[id_s], U_s[:, id_s]

    d_t, U_t = nx.eigh(Cov_t)
    id_t = nx.flip(nx.argsort(d_t))
    d_t, U_t = d_t[id_t], U_t[:, id_t]

    if sign_eigs is None:
        sign_eigs = nx.ones(min(m, n), type_as=mu_s)

    if m >= n:
        A = nx.concatenate((nx.diag(sign_eigs * nx.sqrt(d_t) / nx.sqrt(d_s[:n])), nx.zeros((n, m - n), type_as=mu_s)), axis=1).T
    else:
        A = nx.concatenate((nx.diag(sign_eigs * nx.sqrt(d_t[:m]) / nx.sqrt(d_s)), nx.zeros((n - m, m), type_as=mu_s)), axis=0).T

    A = nx.dot(nx.dot(U_s, A), U_t.T)

    # compute the gaussien Gromov-Wasserstein dis
    b = mu_t - nx.dot(mu_s, A)

    if log:
        log = {}
        log['d_s'] = d_s
        log['d_t'] = d_t
        log['U_s'] = U_s
        log['U_t'] = U_t
        return A, b, log
    else:
        return A, b


def empirical_gaussian_gromov_wasserstein_mapping(xs, xt, ws=None,
                                                  wt=None, sign_eigs=None, log=False):
    r"""Return Gaussian Gromov-Wasserstein mapping between samples.

    The function estimates the Gaussian Gromov-Wasserstein mapping between two
    Gaussien distributions source :math:`\mu_s` and target :math:`\mu_t`, whose
    parameters are estimated from the provided samples :math:`\mathcal{X}_s` and
    :math:`\mathcal{X}_t`. See [57] Theorem 4.1 for more details.


    Parameters
    ----------
    xs : array-like (ns,ds)
        samples in the source domain
    xt : array-like (nt,dt)
        samples in the target domain
    ws : array-like (ns,1), optional
        weights for the source samples
    wt : array-like (ns,1), optional
        weights for the target samples
    sign_eigs : array-like (min(ds,dt),) or string, optional
        sign of the eigenvalues of the mapping matrix, by default all signs will
        be positive. If 'skewness' is provided, the sign of the eigenvalues is
        selected as the product of the sign of the skewness of the projected data.
    log : bool, optional
        record log if True


    Returns
    -------
    A : (dt, ds) array-like
        Linear operator
    b : (1, dt) array-like
        bias

    .. _references-empirical_gaussian_gromov_wasserstein_mapping:
    References
    ----------
    .. [57] Delon, J., Desolneux, A., & Salmona, A. (2022). Gromov–Wasserstein
    distances between Gaussian distributions. Journal of Applied Probability,
    59(4), 1178-1198.
    """

    xs, xt = list_to_array(xs, xt)

    nx = get_backend(xs, xt)

    m = xs.shape[1]
    n = xt.shape[1]

    if ws is None:
        ws = nx.ones((xs.shape[0], 1), type_as=xs) / xs.shape[0]

    if wt is None:
        wt = nx.ones((xt.shape[0], 1), type_as=xt) / xt.shape[0]

    # estimate mean and covariance
    mu_s = nx.dot(ws.T, xs) / nx.sum(ws)
    mu_t = nx.dot(wt.T, xt) / nx.sum(wt)

    xs = xs - mu_s
    xt = xt - mu_t

    Cov_s = nx.dot((xs * ws).T, xs) / nx.sum(ws)
    Cov_t = nx.dot((xt * wt).T, xt) / nx.sum(wt)

    # compte and sort eigenvalues/eigenvectors decreasingly
    d_s, U_s = nx.eigh(Cov_s)
    id_s = nx.flip(nx.argsort(d_s))
    d_s, U_s = d_s[id_s], U_s[:, id_s]

    d_t, U_t = nx.eigh(Cov_t)
    id_t = nx.flip(nx.argsort(d_t))
    d_t, U_t = d_t[id_t], U_t[:, id_t]

    # select the sign of the eigenvalues
    if sign_eigs is None:
        sign_eigs = nx.ones(min(m, n), type_as=mu_s)
    elif sign_eigs == 'skewness':
        size = min(m, n)
        skew_s = nx.sum((nx.dot(xs, U_s[:, :size]))**3 * ws, axis=0)
        skew_t = nx.sum((nx.dot(xt, U_t[:, :size]))**3 * wt, axis=0)
        sign_eigs = nx.sign(skew_t * skew_s)

    if m >= n:
        A = nx.concatenate((nx.diag(sign_eigs * nx.sqrt(d_t) / nx.sqrt(d_s[:n])), nx.zeros((n, m - n), type_as=mu_s)), axis=1).T
    else:
        A = nx.concatenate((nx.diag(sign_eigs * nx.sqrt(d_t[:m]) / nx.sqrt(d_s)), nx.zeros((n - m, m), type_as=mu_s)), axis=0).T

    A = nx.dot(nx.dot(U_s, A), U_t.T)

    # compute the gaussien Gromov-Wasserstein dis
    b = mu_t - nx.dot(mu_s, A)

    if log:
        log = {}
        log['d_s'] = d_s
        log['d_t'] = d_t
        log['U_s'] = U_s
        log['U_t'] = U_t
        log['Cov_s'] = Cov_s
        log['Cov_t'] = Cov_t
        return A, b, log
    else:
        return A, b
