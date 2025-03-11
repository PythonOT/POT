# -*- coding: utf-8 -*-
"""
Optimal transport for Gaussian distributions
"""

# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytehnique.edu>
#
# License: MIT License

import warnings
import numpy as np

from .backend import get_backend
from .utils import dots, is_all_finite, list_to_array, exp_bures


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
        log["Cs12"] = Cs12
        log["Cs12inv"] = Cs12inv
        return A, b, log
    else:
        return A, b


def empirical_bures_wasserstein_mapping(
    xs, xt, reg=1e-6, ws=None, wt=None, bias=True, log=False
):
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
    is_input_finite = is_all_finite(xs, xt)

    d = xs.shape[1]

    if ws is None:
        ws = nx.ones((xs.shape[0], 1), type_as=xs) / xs.shape[0]

    if wt is None:
        wt = nx.ones((xt.shape[0], 1), type_as=xt) / xt.shape[0]

    if bias:
        mxs = nx.dot(ws.T, xs) / nx.sum(ws)
        mxt = nx.dot(wt.T, xt) / nx.sum(wt)

        xs = xs - mxs
        xt = xt - mxt
    else:
        mxs = nx.zeros((1, d), type_as=xs)
        mxt = nx.zeros((1, d), type_as=xs)

    Cs = nx.dot((xs * ws).T, xs) / nx.sum(ws) + reg * nx.eye(d, type_as=xs)
    Ct = nx.dot((xt * wt).T, xt) / nx.sum(wt) + reg * nx.eye(d, type_as=xt)

    if log:
        A, b, log = bures_wasserstein_mapping(mxs, mxt, Cs, Ct, log=log)
    else:
        A, b = bures_wasserstein_mapping(mxs, mxt, Cs, Ct)

    if is_input_finite and not is_all_finite(A, b):
        warnings.warn(
            "Numerical errors were encountered in ot.gaussian.empirical_bures_wasserstein_mapping. "
            "Consider increasing the regularization parameter `reg`."
        )

    if log:
        log["Cs"] = Cs
        log["Ct"] = Ct
        return A, b, log
    else:
        return A, b


def bures_distance(Cs, Ct, paired=False, log=False, nx=None):
    r"""Return Bures distance.

    The function computes the Bures distance between :math:`\mu_s=\mathcal{N}(0,\Sigma_s)` and :math:`\mu_t=\mathcal{N}(0,\Sigma_t)`,
    given by (see e.g. Remark 2.31 :ref:`[15] <references-bures-wasserstein-distance>`):

    .. math::
        \mathbf{B}(\Sigma_s, \Sigma_t)^{2} = \text{Tr}\left(\Sigma_s + \Sigma_t - 2 \sqrt{\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2}} \right)

    Parameters
    ----------
    Cs : array-like (d,d) or (n,d,d)
        covariance of the source distribution
    Ct : array-like (d,d) or (m,d,d)
        covariance of the target distribution
    paired: bool, optional
        if True and n==m, return the paired distances and crossed distance otherwise
    log : bool, optional
        record log if True
    nx : module, optional
        The numerical backend module to use. If not provided, the backend will
        be fetched from the input matrices `Cs, Ct`.

    Returns
    -------
    W : float if Cs and Cd of shape (d,d), array-like (n,m) if Cs of shape (n,d,d) and Ct of shape (m,d,d), array-like (n,) if Cs and Ct of shape (n, d, d) and paired is True
        Bures Wasserstein distance
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-bures-wasserstein-distance:
    References
    ----------
    .. [15] Peyré, G., & Cuturi, M. (2019). Computational optimal transport: With applications to data science.
        Foundations and Trends® in Machine Learning, 11(5-6), 355-607.
    """
    Cs, Ct = list_to_array(Cs, Ct)

    if nx is None:
        nx = get_backend(Cs, Ct)

    assert Cs.shape[-1] == Ct.shape[-1], "All Gaussian must have the same dimension"

    Cs12 = nx.sqrtm(Cs)

    if len(Cs.shape) == 2 and len(Ct.shape) == 2:
        # Return float
        bw2 = nx.trace(Cs + Ct - 2 * nx.sqrtm(dots(Cs12, Ct, Cs12)))
    else:
        assert (
            len(Cs.shape) == 3 and len(Ct.shape) == 3
        ), "Both Cs and Ct should be batched"
        if paired and len(Cs) == len(Ct):
            # Return shape (n,)
            M = nx.einsum("nij, njk, nkl -> nil", Cs12, Ct, Cs12)
            bw2 = nx.trace(Cs + Ct - 2 * nx.sqrtm(M))
        else:
            # Return shape (n,m)
            M = nx.einsum("nij, mjk, nkl -> nmil", Cs12, Ct, Cs12)
            bw2 = nx.trace(Cs[:, None] + Ct[None] - 2 * nx.sqrtm(M))

    W = nx.sqrt(nx.maximum(bw2, 0))

    if log:
        log = {}
        log["Cs12"] = Cs12
        return W, log
    else:
        return W


def bures_wasserstein_distance(ms, mt, Cs, Ct, paired=False, log=False):
    r"""Return Bures Wasserstein distance between samples.

    The function computes the Bures-Wasserstein distance between :math:`\mu_s=\mathcal{N}(m_s,\Sigma_s)` and :math:`\mu_t=\mathcal{N}(m_t,\Sigma_t)`,
    as discussed in remark 2.31 :ref:`[15] <references-bures-wasserstein-distance>`.

    .. math::
        \mathcal{W}(\mu_s, \mu_t)_2^2= \left\lVert \mathbf{m}_s - \mathbf{m}_t \right\rVert^2 + \mathcal{B}(\Sigma_s, \Sigma_t)^{2}

    where :

    .. math::
        \mathbf{B}(\Sigma_s, \Sigma_t)^{2} = \text{Tr}\left(\Sigma_s + \Sigma_t - 2 \sqrt{\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2}} \right)

    Parameters
    ----------
    ms : array-like (d,) or (n,d)
        mean of the source distribution
    mt : array-like (d,) or (m,d)
        mean of the target distribution
    Cs : array-like (d,d) or (n,d,d)
        covariance of the source distribution
    Ct : array-like (d,d) or (m,d,d)
        covariance of the target distribution
    paired: bool, optional
        if True and n==m, return the paired distances and crossed distance otherwise
    log : bool, optional
        record log if True

    Returns
    -------
    W : float if ms and md of shape (d,), array-like (n,m) if ms of shape (n,d) and mt of shape (m,d), array-like (n,) if ms and mt of shape (n,d) and paired is True
        Bures Wasserstein distance
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-bures-wasserstein-distance:
    References
    ----------

    .. [15] Peyré, G., & Cuturi, M. (2019). Computational optimal transport: With applications to data science.
        Foundations and Trends® in Machine Learning, 11(5-6), 355-607.
    """
    ms, mt, Cs, Ct = list_to_array(ms, mt, Cs, Ct)
    nx = get_backend(ms, mt, Cs, Ct)

    assert (
        ms.shape[0] == Cs.shape[0]
    ), "Source Gaussians has different amount of components"

    assert (
        mt.shape[0] == Ct.shape[0]
    ), "Target Gaussians has different amount of components"

    assert (
        ms.shape[-1] == mt.shape[-1] == Cs.shape[-1] == Ct.shape[-1]
    ), "All Gaussian must have the same dimension"

    if log:
        bw, log_dict = bures_distance(Cs, Ct, paired=paired, log=log, nx=nx)
        Cs12 = log_dict["Cs12"]
    else:
        bw = bures_distance(Cs, Ct, paired=paired, nx=nx)

    if len(ms.shape) == 1 and len(mt.shape) == 1:
        # Return float
        squared_dist_m = nx.norm(ms - mt) ** 2
    else:
        assert (
            len(ms.shape) == 2 and len(mt.shape) == 2
        ), "Both ms and mt should be batched"
        if paired and len(ms.shape) == len(mt.shape):
            # Return shape (n,)
            squared_dist_m = nx.norm(ms - mt, axis=-1) ** 2
        else:
            # Return shape (n,m)
            squared_dist_m = nx.norm(ms[:, None] - mt[None], axis=-1) ** 2

    W = nx.sqrt(nx.maximum(squared_dist_m + bw**2, 0))

    if log:
        log = {}
        log["Cs12"] = Cs12
        return W, log
    else:
        return W


def empirical_bures_wasserstein_distance(
    xs, xt, reg=1e-6, ws=None, wt=None, bias=True, log=False
):
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

    if ws is None:
        ws = nx.ones((xs.shape[0], 1), type_as=xs) / xs.shape[0]

    if wt is None:
        wt = nx.ones((xt.shape[0], 1), type_as=xt) / xt.shape[0]

    if bias:
        mxs = nx.dot(ws.T, xs) / nx.sum(ws)
        mxt = nx.dot(wt.T, xt) / nx.sum(wt)

        xs = xs - mxs
        xt = xt - mxt
    else:
        mxs = nx.zeros((1, d), type_as=xs)
        mxt = nx.zeros((1, d), type_as=xs)

    Cs = nx.dot((xs * ws).T, xs) / nx.sum(ws) + reg * nx.eye(d, type_as=xs)
    Ct = nx.dot((xt * wt).T, xt) / nx.sum(wt) + reg * nx.eye(d, type_as=xt)

    if log:
        W, log = bures_wasserstein_distance(mxs[0], mxt[0], Cs, Ct, log=log)
        log["Cs"] = Cs
        log["Ct"] = Ct
        return W, log
    else:
        W = bures_wasserstein_distance(mxs[0], mxt[0], Cs, Ct)
        return W


def bures_barycenter_fixpoint(
    C, weights=None, num_iter=1000, eps=1e-7, log=False, nx=None
):
    r"""Return the (Bures-)Wasserstein barycenter between centered Gaussian distributions.

    The function estimates the (Bures)-Wasserstein barycenter between centered Gaussian distributions :math:`\big(\mathcal{N}(0,\Sigma_i)\big)_{i=1}^n`
    :ref:`[16] <references-OT-bures-barycenter-fixed-point>` by solving

    .. math::
        \Sigma_b = \mathrm{argmin}_{\Sigma \in S_d^{++}(\mathbb{R})}\ \sum_{i=1}^n w_i W_2^2\big(\mathcal{N}(0,\Sigma), \mathcal{N}(0, \Sigma_i)\big).

    The barycenter still follows a Gaussian distribution :math:`\mathcal{N}(0,\Sigma_b)`
    where :math:`\Sigma_b` is solution of the following fixed-point algorithm:

    .. math::
        \Sigma_b = \sum_{i=1}^n w_i \left(\Sigma_b^{1/2}\Sigma_i^{1/2}\Sigma_b^{1/2}\right)^{1/2}.

    Parameters
    ----------
    C : array-like (k,d,d)
        covariance of k distributions
    weights : array-like (k), optional
        weights for each distribution
    method : str
        method used for the solver, either 'fixed_point' or 'gradient_descent'
    num_iter : int, optional
        number of iteration for the fixed point algorithm
    eps : float, optional
        tolerance for the fixed point algorithm
    log : bool, optional
        record log if True
    nx : module, optional
        The numerical backend module to use. If not provided, the backend will
        be fetched from the input matrices `C`.

    Returns
    -------
    Cb : (d, d) array-like
        covariance of the barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-OT-bures-barycenter-fixed-point:
    References
    ----------
    .. [16] M. Agueh and G. Carlier, "Barycenters in the Wasserstein space",
        SIAM Journal on Mathematical Analysis, vol. 43, no. 2, pp. 904-924,
        2011.
    """
    if nx is None:
        nx = get_backend(
            *C,
        )

    if weights is None:
        weights = nx.ones(C.shape[0], type_as=C[0]) / C.shape[0]

    # Init the covariance barycenter
    Cb = nx.mean(C * weights[:, None, None], axis=0)

    for it in range(num_iter):
        # fixed point update
        Cb12 = nx.sqrtm(Cb)

        Cnew = nx.sqrtm(Cb12 @ C @ Cb12)
        Cnew *= weights[:, None, None]
        Cnew = nx.sum(Cnew, axis=0)

        # check convergence
        diff = nx.norm(Cb - Cnew)
        if diff <= eps:
            break
        Cb = Cnew

    if diff > eps:
        print("Dit not converge.")

    if log:
        log = {}
        log["num_iter"] = it
        log["final_diff"] = diff
        return Cb, log
    else:
        return Cb


def bures_barycenter_gradient_descent(
    C,
    weights=None,
    num_iter=1000,
    eps=1e-7,
    log=False,
    step_size=1,
    batch_size=None,
    averaged=False,
    nx=None,
):
    r"""Return the (Bures-)Wasserstein barycenter between centered Gaussian distributions.

    The function estimates the (Bures)-Wasserstein barycenter between centered Gaussian distributions :math:`\big(\mathcal{N}(0,\Sigma_i)\big)_{i=1}^n`
    by using a gradient descent in the Wasserstein space :ref:`[74, 75] <references-OT-bures-barycenter-gradient_descent>`
    on the objective

    .. math::
        \mathcal{L}(\Sigma) = \sum_{i=1}^n w_i W_2^2\big(\mathcal{N}(0,\Sigma), \mathcal{N}(0,\Sigma_i)\big).

    Parameters
    ----------
    C : array-like (k,d,d)
        covariance of k distributions
    weights : array-like (k), optional
        weights for each distribution
    method : str
        method used for the solver, either 'fixed_point' or 'gradient_descent'
    num_iter : int, optional
        number of iteration for the fixed point algorithm
    eps : float, optional
        tolerance for the fixed point algorithm
    log : bool, optional
        record log if True
    step_size : float, optional
        step size for the gradient descent, 1 by default
    batch_size : int, optional
        batch size if use a stochastic gradient descent
    averaged : bool, optional
        if True, use the averaged procedure of :ref:`[74] <references-OT-bures-barycenter-gradient_descent>`
    nx : module, optional
        The numerical backend module to use. If not provided, the backend will
        be fetched from the input matrices `C`.

    Returns
    -------
    Cb : (d, d) array-like
        covariance of the barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-OT-bures-barycenter-gradient_descent:
    References
    ----------
    .. [74] Chewi, S., Maunu, T., Rigollet, P., & Stromme, A. J. (2020).
        Gradient descent algorithms for Bures-Wasserstein barycenters.
        In Conference on Learning Theory (pp. 1276-1304). PMLR.

    .. [75] Altschuler, J., Chewi, S., Gerber, P. R., & Stromme, A. (2021).
        Averaging on the Bures-Wasserstein manifold: dimension-free convergence
        of gradient descent. Advances in Neural Information Processing Systems, 34, 22132-22145.
    """
    if nx is None:
        nx = get_backend(
            *C,
        )

    n = C.shape[0]

    if weights is None:
        weights = nx.ones(C.shape[0], type_as=C[0]) / n

    # Init the covariance barycenter
    Cb = nx.mean(C * weights[:, None, None], axis=0)
    Id = nx.eye(C.shape[-1], type_as=Cb)

    L_diff = []

    Cb_averaged = nx.copy(Cb)

    for it in range(num_iter):
        Cb12 = nx.sqrtm(Cb)
        Cb12_ = nx.inv(Cb12)

        if batch_size is not None and batch_size < n:  # if stochastic gradient descent
            if batch_size <= 0:
                raise ValueError(
                    "batch_size must be an integer between 0 and {}".format(n)
                )
            inds = np.random.choice(
                n, batch_size, replace=True, p=nx._to_numpy(weights)
            )
            M = nx.sqrtm(nx.einsum("ij,njk,kl -> nil", Cb12, C[inds], Cb12))
            ot_maps = nx.einsum("ij,njk,kl -> nil", Cb12_, M, Cb12_)
            grad_bw = Id - nx.mean(ot_maps, axis=0)

            # step size from [74] (page 15)
            step_size = 2 / (0.7 * (it + 2 / 0.7 + 1))
        else:  # gradient descent
            M = nx.sqrtm(nx.einsum("ij,njk,kl -> nil", Cb12, C, Cb12))
            ot_maps = nx.einsum("ij,njk,kl -> nil", Cb12_, M, Cb12_)
            grad_bw = Id - nx.sum(ot_maps * weights[:, None, None], axis=0)

        Cnew = exp_bures(Cb, -step_size * grad_bw, nx=nx)

        if averaged:
            # ot map between Cb_averaged and Cnew
            Cb_averaged12 = nx.sqrtm(Cb_averaged)
            Cb_averaged12inv = nx.inv(Cb_averaged12)
            M = nx.sqrtm(nx.einsum("ij,jk,kl->il", Cb_averaged12, Cnew, Cb_averaged12))
            ot_map = nx.einsum("ij,jk,kl->il", Cb_averaged12inv, M, Cb_averaged12inv)
            map = Id * step_size / (step_size + 1) + ot_map / (step_size + 1)
            Cb_averaged = nx.einsum("ij,jk,kl->il", map, Cb_averaged, map)

        # check convergence
        L_diff.append(nx.norm(Cb - Cnew))

        # Criteria to stop
        if np.mean(L_diff[-100:]) <= eps:
            break

        Cb = Cnew

    if averaged:
        Cb = Cb_averaged

    if log:
        dict_log = {}
        dict_log["num_iter"] = it
        dict_log["final_diff"] = L_diff[-1]
        return Cb, dict_log
    else:
        return Cb


def bures_wasserstein_barycenter(
    m,
    C,
    weights=None,
    method="fixed_point",
    num_iter=1000,
    eps=1e-7,
    log=False,
    step_size=1,
    batch_size=None,
):
    r"""Return the (Bures-)Wasserstein barycenter between Gaussian distributions.

    The function estimates the (Bures)-Wasserstein barycenter between Gaussian distributions :math:`\big(\mathcal{N}(\mu_i,\Sigma_i)\big)_{i=1}^n`
    :ref:`[16, 74, 75] <references-OT-bures_wasserstein-barycenter>` by solving

    .. math::
        (\mu_b, \Sigma_b) = \mathrm{argmin}_{\mu,\Sigma}\ \sum_{i=1}^n w_i W_2^2\big(\mathcal{N}(\mu,\Sigma), \mathcal{N}(\mu_i, \Sigma_i)\big)

    The barycenter still follows a Gaussian distribution :math:`\mathcal{N}(\mu_b,\Sigma_b)`
    where:

    .. math::
        \mu_b = \sum_{i=1}^n w_i \mu_i,

    and the barycentric covariance is the solution of the following fixed-point algorithm:

    .. math::
        \Sigma_b = \sum_{i=1}^n w_i \left(\Sigma_b^{1/2}\Sigma_i^{1/2}\Sigma_b^{1/2}\right)^{1/2}

    We propose two solvers: one based on solving the previous fixed-point problem [16]. Another based on
    gradient descent in the Bures-Wasserstein space [74,75].

    Parameters
    ----------
    m : array-like (k,d)
        mean of k distributions
    C : array-like (k,d,d)
        covariance of k distributions
    weights : array-like (k), optional
        weights for each distribution
    method : str
        method used for the solver, either 'fixed_point', 'gradient_descent', 'stochastic_gradient_descent' or
        'averaged_stochastic_gradient_descent'
    num_iter : int, optional
        number of iteration for the fixed point algorithm
    eps : float, optional
        tolerance for the fixed point algorithm
    log : bool, optional
        record log if True
    step_size : float, optional
        step size for the gradient descent, 1 by default
    batch_size : int, optional
        batch size if use a stochastic gradient descent. If not None, use method='gradient_descent'


    Returns
    -------
    mb : (d,) array-like
        mean of the barycenter
    Cb : (d, d) array-like
        covariance of the barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-OT-bures_wasserstein-barycenter:
    References
    ----------
    .. [16] M. Agueh and G. Carlier, "Barycenters in the Wasserstein space",
        SIAM Journal on Mathematical Analysis, vol. 43, no. 2, pp. 904-924,
        2011.

    .. [74] Chewi, S., Maunu, T., Rigollet, P., & Stromme, A. J. (2020).
        Gradient descent algorithms for Bures-Wasserstein barycenters.
        In Conference on Learning Theory (pp. 1276-1304). PMLR.

    .. [75] Altschuler, J., Chewi, S., Gerber, P. R., & Stromme, A. (2021).
        Averaging on the Bures-Wasserstein manifold: dimension-free convergence
        of gradient descent. Advances in Neural Information Processing Systems, 34, 22132-22145.
    """
    nx = get_backend(
        *m,
    )

    if weights is None:
        weights = nx.ones(C.shape[0], type_as=C[0]) / C.shape[0]

    # Compute the mean barycenter
    mb = nx.sum(m * weights[:, None], axis=0)

    if method == "gradient_descent":
        out = bures_barycenter_gradient_descent(
            C,
            weights=weights,
            num_iter=num_iter,
            eps=eps,
            log=log,
            step_size=step_size,
            nx=nx,
        )
    elif method == "stochastic_gradient_descent":
        out = bures_barycenter_gradient_descent(
            C,
            weights=weights,
            num_iter=num_iter,
            eps=eps,
            log=log,
            batch_size=1 if batch_size is None else batch_size,
            nx=nx,
        )
    elif method == "averaged_stochastic_gradient_descent":
        out = bures_barycenter_gradient_descent(
            C,
            weights=weights,
            num_iter=num_iter,
            eps=eps,
            log=log,
            batch_size=1 if batch_size is None else batch_size,
            averaged=True,
            nx=nx,
        )
    elif method == "fixed_point":
        out = bures_barycenter_fixpoint(
            C, weights=weights, num_iter=num_iter, eps=eps, log=log, nx=nx
        )
    else:
        raise ValueError("Unknown method '%s'." % method)

    if log:
        Cb, log = out
        return mb, Cb, log
    else:
        Cb = out
        return mb, Cb


def empirical_bures_wasserstein_barycenter(
    X, reg=1e-6, weights=None, num_iter=1000, eps=1e-7, w=None, bias=True, log=False
):
    r"""Return OT linear operator between samples.

    The function estimates the optimal barycenter of the
    empirical distributions. This is equivalent to resolving the fixed point
    algorithm for multiple Gaussian distributions :math:`\left\{\mathcal{N}(\mu,\Sigma)\right\}_{i=1}^n`
    :ref:`[1] <references-OT-mapping-linear-barycenter>`.

    The barycenter still following a Gaussian distribution :math:`\mathcal{N}(\mu_b,\Sigma_b)`
    where :

    .. math::
        \mu_b = \sum_{i=1}^n w_i \mu_i

    And the barycentric covariance is the solution of the following fixed-point algorithm:

    .. math::
        \Sigma_b = \sum_{i=1}^n w_i \left(\Sigma_b^{1/2}\Sigma_i^{1/2}\Sigma_b^{1/2}\right)^{1/2}


    Parameters
    ----------
    X : list of array-like (n,d)
        samples in each distribution
    reg : float,optional
        regularization added to the diagonals of covariances (>0)
    weights : array-like (n,), optional
        weights for each distribution
    num_iter : int, optional
        number of iteration for the fixed point algorithm
    eps : float, optional
        tolerance for the fixed point algorithm
    w : list of array-like (n,), optional
        weights for each sample in each distribution
    bias: boolean, optional
        estimate bias :math:`\mathbf{b}` else :math:`\mathbf{b} = 0` (default:True)
    log : bool, optional
        record log if True


    Returns
    -------
    mb : (d,) array-like
        mean of the barycenter
    Cb : (d, d) array-like
        covariance of the barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-OT-mapping-linear-barycenter:
    References
    ----------
    .. [1] M. Agueh and G. Carlier, "Barycenters in the Wasserstein space",
        SIAM Journal on Mathematical Analysis, vol. 43, no. 2, pp. 904-924,
        2011.
    """
    X = list_to_array(*X)
    nx = get_backend(*X)

    k = len(X)
    d = [X[i].shape[1] for i in range(k)]

    if w is None:
        w = [
            nx.ones((X[i].shape[0], 1), type_as=X[i]) / X[i].shape[0] for i in range(k)
        ]

    if bias:
        m = [nx.dot(w[i].T, X[i]) / nx.sum(w[i]) for i in range(k)]
        X = [X[i] - m[i] for i in range(k)]
    else:
        m = [nx.zeros((1, d[i]), type_as=X[i]) for i in range(k)]

    C = [
        nx.dot((X[i] * w[i]).T, X[i]) / nx.sum(w[i]) + reg * nx.eye(d[i], type_as=X[i])
        for i in range(k)
    ]
    m = nx.stack(m, axis=0)[:, 0]
    C = nx.stack(C, axis=0)

    if log:
        mb, Cb, log = bures_wasserstein_barycenter(
            m, C, weights=weights, num_iter=num_iter, eps=eps, log=log
        )

        return mb, Cb, log
    else:
        mb, Cb = bures_wasserstein_barycenter(
            m, C, weights=weights, num_iter=num_iter, eps=eps, log=log
        )
        return mb, Cb


def gaussian_gromov_wasserstein_distance(Cov_s, Cov_t, log=False):
    r"""Return the Gaussian Gromov-Wasserstein value from [57].

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
    .. [57] Delon, J., Desolneux, A., & Salmona, A. (2022). Gromov–Wasserstein distances between Gaussian distributions.
        Journal of Applied Probability, 59(4), 1178-1198.
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
    res = (
        4 * (nx.sum(d_s) - nx.sum(d_t)) ** 2
        + 8 * nx.sum((d_s[:n] - d_t) ** 2)
        + 8 * nx.sum((d_s[n:]) ** 2)
    )
    if log:
        log = {}
        log["d_s"] = d_s
        log["d_t"] = d_t
        return nx.sqrt(res), log
    else:
        return nx.sqrt(res)


def empirical_gaussian_gromov_wasserstein_distance(xs, xt, ws=None, wt=None, log=False):
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
    .. [57] Delon, J., Desolneux, A., & Salmona, A. (2022).
        Gromov–Wasserstein distances between Gaussian distributions.
        Journal of Applied Probability, 59(4), 1178-1198.
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
        log["Cov_s"] = Cs
        log["Cov_t"] = Ct
        return G, log
    else:
        G = gaussian_gromov_wasserstein_distance(Cs, Ct)
        return G


def gaussian_gromov_wasserstein_mapping(
    mu_s, mu_t, Cov_s, Cov_t, sign_eigs=None, log=False
):
    r"""Return the Gaussian Gromov-Wasserstein mapping from [57].

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
    .. [57] Delon, J., Desolneux, A., & Salmona, A. (2022).
        Gromov–Wasserstein distances between Gaussian distributions.
        Journal of Applied Probability, 59(4), 1178-1198.
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
        A = nx.concatenate(
            (
                nx.diag(sign_eigs * nx.sqrt(d_t) / nx.sqrt(d_s[:n])),
                nx.zeros((n, m - n), type_as=mu_s),
            ),
            axis=1,
        ).T
    else:
        A = nx.concatenate(
            (
                nx.diag(sign_eigs * nx.sqrt(d_t[:m]) / nx.sqrt(d_s)),
                nx.zeros((n - m, m), type_as=mu_s),
            ),
            axis=0,
        ).T

    A = nx.dot(nx.dot(U_s, A), U_t.T)

    # compute the gaussien Gromov-Wasserstein dis
    b = mu_t - nx.dot(mu_s, A)

    if log:
        log = {}
        log["d_s"] = d_s
        log["d_t"] = d_t
        log["U_s"] = U_s
        log["U_t"] = U_t
        return A, b, log
    else:
        return A, b


def empirical_gaussian_gromov_wasserstein_mapping(
    xs, xt, ws=None, wt=None, sign_eigs=None, log=False
):
    r"""Return Gaussian Gromov-Wasserstein mapping between samples.

    The function estimates the Gaussian Gromov-Wasserstein mapping between two
    Gaussian distributions source :math:`\mu_s` and target :math:`\mu_t`, whose
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
    .. [57] Delon, J., Desolneux, A., & Salmona, A. (2022).
        Gromov–Wasserstein distances between Gaussian distributions.
        Journal of Applied Probability, 59(4), 1178-1198.
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

    # compute and sort eigenvalues/eigenvectors decreasingly
    d_s, U_s = nx.eigh(Cov_s)
    id_s = nx.flip(nx.argsort(d_s))
    d_s, U_s = d_s[id_s], U_s[:, id_s]

    d_t, U_t = nx.eigh(Cov_t)
    id_t = nx.flip(nx.argsort(d_t))
    d_t, U_t = d_t[id_t], U_t[:, id_t]

    # select the sign of the eigenvalues
    if sign_eigs is None:
        sign_eigs = nx.ones(min(m, n), type_as=mu_s)
    elif sign_eigs == "skewness":
        size = min(m, n)
        skew_s = nx.sum((nx.dot(xs, U_s[:, :size])) ** 3 * ws, axis=0)
        skew_t = nx.sum((nx.dot(xt, U_t[:, :size])) ** 3 * wt, axis=0)
        sign_eigs = nx.sign(skew_t * skew_s)

    if m >= n:
        A = nx.concatenate(
            (
                nx.diag(sign_eigs * nx.sqrt(d_t) / nx.sqrt(d_s[:n])),
                nx.zeros((n, m - n), type_as=mu_s),
            ),
            axis=1,
        ).T
    else:
        A = nx.concatenate(
            (
                nx.diag(sign_eigs * nx.sqrt(d_t[:m]) / nx.sqrt(d_s)),
                nx.zeros((n - m, m), type_as=mu_s),
            ),
            axis=0,
        ).T

    A = nx.dot(nx.dot(U_s, A), U_t.T)

    # compute the gaussien Gromov-Wasserstein dis
    b = mu_t - nx.dot(mu_s, A)

    if log:
        log = {}
        log["d_s"] = d_s
        log["d_t"] = d_t
        log["U_s"] = U_s
        log["U_t"] = U_t
        log["Cov_s"] = Cov_s
        log["Cov_t"] = Cov_t
        return A, b, log
    else:
        return A, b
