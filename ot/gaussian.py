# -*- coding: utf-8 -*-
"""
Domain adaptation with optimal transport
"""

# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: MIT License

from .backend import get_backend
from .utils import dots
from .utils import list_to_array


def OT_mapping_linear(xs, xt, reg=1e-6, ws=None,
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

    Cs12 = nx.sqrtm(Cs)
    Cs_12 = nx.inv(Cs12)

    M0 = nx.sqrtm(dots(Cs12, Ct, Cs12))

    A = dots(Cs_12, M0, Cs_12)

    b = mxt - nx.dot(mxs, A)

    if log:
        log = {}
        log['Cs'] = Cs
        log['Ct'] = Ct
        log['Cs12'] = Cs12
        log['Cs_12'] = Cs_12
        return A, b, log
    else:
        return A, b


def bures_wasserstein_distance(xs, xt, reg=1e-6, ws=None,
                               wt=None, bias=True, log=False):
    r"""Return Bures Wasserstein distance between samples.

    The function estimates the Bures-Wasserstein distance between two
    empirical distributions source :math:`\mu_s` and target :math:`\mu_t`,
    discussed in remark 2.31 :ref:`[1] <references-bures-wasserstein-distance>`.

    The Bures Wasserstein distance between source and target distribution :math:`\mathcal{W}`

    .. math::
        \mathcal{W}(\mu_s, \mu_t)_2^2= \left\lVert \mathbf{m}_s - \mathbf{m}_t \right\rVert^2 + \mathcal{B}(\Sigma_s, \Sigma_t)^{2}

    where :

    .. math::
        \mathbf{B}(\Sigma_s, \Sigma_t)^{2} &= \text{\Sigma_s^{1/2} + \Sigma_t^{1/2} - 2 \left(\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2} \right)^{1/2}

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
    W : float
        bures Wasserstein distance
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
    Cs12 = nx.sqrtm(Cs)

    B = nx.trace(Cs + Ct - 2 * nx.sqrtm(dots(Cs12, Ct, Cs12)))
    print(nx.norm(mxs - mxt), mxs, mxt)
    W = nx.norm(mxs - mxt) + B
    if log:
        log = {}
        log['Cs'] = Cs
        log['Ct'] = Ct
        log['Cs12'] = Cs12
        return W, log
    else:
        return W
