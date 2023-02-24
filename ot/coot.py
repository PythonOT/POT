# -*- coding: utf-8 -*-
"""
Fused CO-Optimal Transport and entropic Fused CO-Optimal Transport solvers
"""

# Author: Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

import numpy as np
from functools import partial
from .lp import emd
from .utils import list_to_array
from .backend import get_backend
from .bregman import sinkhorn


def co_optimal_transport(X, Y, px=(None, None), py=(None, None), eps=(0, 0),
                         alpha=(1, 1), D=(None, None), warmstart=None,
                         nits_bcd=100, tol_bcd=1e-7, eval_bcd=1,
                         nits_ot=500, tol_sinkhorn=1e-7, method_sinkhorn="sinkhorn",
                         early_stopping_tol=1e-6, log=False, verbose=False):
    r"""
    Return the sample and feature transport plans between
    :math:`(\mathbf{X}, \mathbf{p}_{xs}, \mathbf{p}_{xf})` and
    :math:`(\mathbf{Y}, \mathbf{p}_{ys}, \mathbf{p}_{yf})`.

    The function solves the following CO-Optimal Transport (COOT) problem:

    .. math::
        \mathbf{COOT}_{\varepsilon} = \mathop{\arg \min}_{\mathbf{P}, \mathbf{Q}}
        \quad \sum_{i,j,k,l}
        (\mathbf{X}_{i,k} - \mathbf{Y}_{j,l})^2 \mathbf{P}_{i,j} \mathbf{Q}_{k,l}
        + \alpha_1 \sum_{i,j} \mathbf{P}_{i,j} \mathbf{D^{(s)}}_{i, j}
        + \alpha_2 \sum_{k, l} \mathbf{Q}_{k,l} \mathbf{D^{(f)}}_{k, l}
        + \varepsilon_1 \mathbf{KL}(\mathbf{P} | \mathbf{p}_{xs} \mathbf{p}_{ys}^T)
        + \varepsilon_2 \mathbf{KL}(\mathbf{Q} | \mathbf{p}_{xf} \mathbf{p}_{yf}^T)

    Where :

    - :math:`\mathbf{X}`: Data matrix in the source space
    - :math:`\mathbf{Y}`: Data matrix in the target space
    - :math:`\mathbf{D^{(s)}}`: Additional sample matrix
    - :math:`\mathbf{D^{(f)}}`: Additional feature matrix
    - :math:`\mathbf{p}_{xs}`: Distribution of the samples in the source space
    - :math:`\mathbf{p}_{xf}`: Distribution of the features in the source space
    - :math:`\mathbf{p}_{ys}`: Distribution of the samples in the target space
    - :math:`\mathbf{p}_{yf}`: Distribution of the features in the target space

    .. note:: This function allows epsilons to be zero.
    In that case, the `ot.lp.emd` solver of POT will be used.

    Parameters
    ----------
    X : (sx, fx) array-like, float
        First input matrix.
    Y : (sy, fy) array-like, float
        Second input matrix.
    px : (sx, fx) tuple, float, optional (default = (None,None))
        Histogram assigned on rows (samples) and columns (features) of X.
        Uniform distribution by default.
    py : (sy, fy) tuple, float, optional (default = (None,None))
        Histogram assigned on rows (samples) and columns (features) of Y.
        Uniform distribution by default.
    eps : (scalar, scalar) tuple, float or int (default = (0,0))
        Regularisation parameters for entropic approximation of sample and feature couplings.
        Allow the case where eps contains 0. In that case, the EMD solver is used instead of
        Sinkhorn solver.
    alpha : (scalar, scalar) tuple, float or int, optional (default = (1,1))
        Interpolation parameter for fused COOT with respect to the sample and feature couplings.
    D : tuple of matrices (sx, sy) and (fx, fy), float, optional (default = (None,None))
        Sample and feature matrices, in case of fused COOT.
    warmstart : dictionary, optional (default = None)
        Containing 4 keys:
            + "duals_sample" and "duals_feature" whose values are
            tuples of 2 vectors of size (sx, sy) and (fx, fy).
            Initialization of sample and feature dual vectors
            if using Sinkhorn algorithm. Zero vectors by default.
            + "pi_sample" and "pi_feature" whose values are matrices
            of size (sx, sy) and (fx, fy).
            Initialization of sample and feature couplings.
            Uniform distributions by default.
    nits_bcd : int, optional (default = 100)
        Number of Block Coordinate Descent (BCD) iterations to solve COOT.
    tol_bcd : float, optional (default = 1e-7)
        Tolerance of BCD scheme. If the L1-norm between the current and previous
        sample couplings is under this threshold, then stop BCD scheme.
    eval_bcd : int, optional (default = 1)
        Multiplier of iteration at which the COOT cost is evaluated. For example,
        if `eval_bcd = 8`, then the cost is calculated at iterations 8, 16, 24, etc...
    nits_ot : int, optional (default = 100)
        Number of iterations to solve each of the
        two optimal transport problems in each BCD iteration.
    tol_sinkhorn : float, optional (default = 1e-7)
        Tolerance of Sinkhorn algorithm to stop the Sinkhorn scheme for
        entropic optimal transport problem (if any) in each BCD iteration.
        Only triggered when Sinkhorn solver is used.
    method_sinkhorn : string, optional (default = "sinkhorn")
        Method used in POT's `ot.sinkhorn` solver.
        Only support "sinkhorn" and "sinkhorn_log".
    early_stopping_tol : float, optional (default = 1e-6)
        Tolerance for the early stopping. If the absolute difference between
        the last 2 recorded COOT distances is under this tolerance, then stop BCD scheme.
    log : bool, optional (default = False)
        If True then the cost and 4 dual vectors, including
        2 from sample and 2 from feature couplings, are recorded.
    verbose : bool, optional (default = False)
        If True then print the COOT cost at every multiplier of `eval_bcd`-th iteration.

    Returns
    -------
    pi_sample : (sx, sy) array-like, float
        Sample coupling matrix.
    pi_feature : (fx, fy) array-like, float
        Feature coupling matrix.

    if log is True, then return additionally a dictionary whose keys are:
        duals_sample : (sx, sy) tuple, float
            Pair of dual vectors when solving OT problem w.r.t the sample coupling.
        duals_feature : (fx, fy) tuple, float
            Pair of dual vectors when solving OT problem w.r.t the feature coupling.
        distances : list, float
            List of COOT distances

    References
    ----------
    .. [47] I. Redko, T. Vayer, R. Flamary, and N. Courty, CO-Optimal Transport,
        Advances in Neural Information Processing Systems, 33 (2020).
    """

    def compute_kl(p, log_q):
        kl = nx.sum(p * nx.log(p + 1.0 * (p == 0))) - nx.sum(p * log_q)
        return kl

    def emd_solver(cost, p1_np, p2_np):
        cost_np = nx.to_numpy(cost)
        pi_np, log = emd(a=p1_np, b=p2_np, M=cost_np,
                         numItermax=nits_ot, log=True)
        f1 = nx.from_numpy(log["u"], type_as=cost)
        f2 = nx.from_numpy(log["v"], type_as=cost)
        pi = nx.from_numpy(pi_np, type_as=cost)

        return pi, (f1, f2)

    def get_distance(ot_cost, pi_sample, pi_feature, log_pxy_samp, log_pxy_feat,
                     D_samp, alpha_samp, eps):
        eps_samp, eps_feat = eps

        # COOT part
        coot = nx.sum(ot_cost * pi_feature)
        if alpha_samp != 0:
            coot = coot + alpha_samp * nx.sum(D_samp * pi_sample)

        # Entropic part
        if eps_samp != 0:
            coot = coot + eps_samp * \
                compute_kl(pi_sample, log_pxy_samp)
        if eps_feat != 0:
            coot = coot + eps_feat * \
                compute_kl(pi_feature, log_pxy_feat)

        return coot

    # Main function

    if method_sinkhorn not in ["sinkhorn", "sinkhorn_log"]:
        raise ValueError(
            "Method {} is not supported in CO-Optimal Transport.".format(method_sinkhorn))

    X, Y = list_to_array(X, Y)
    nx = get_backend(X, Y)

    # constant input variables
    eps_samp, eps_feat = eps
    alpha_samp, alpha_feat = alpha
    if D is None:
        D = (None, None)
    D_samp, D_feat = D
    if D_samp is None or alpha_samp == 0:
        D_samp, alpha_samp = 0, 0
    if D_feat is None or alpha_feat == 0:
        D_feat, alpha_feat = 0, 0

    sx, fx = X.shape  # s for sample and f for feature
    sy, fy = Y.shape  # s for sample and f for feature

    # measures on rows and columns
    px_samp, px_feat = px
    py_samp, py_feat = py

    if px_samp is None:
        px_samp = nx.ones(sx, type_as=X) / sx
        px_samp_np = np.ones(sx) / sx  # create
    else:
        px_samp_np = nx.to_numpy(px_samp)

    if px_feat is None:
        px_feat = nx.ones(fx, type_as=X) / fx
        px_feat_np = np.ones(fx) / fx
    else:
        px_feat_np = nx.to_numpy(px_feat)

    if py_samp is None:
        py_samp = nx.ones(sy, type_as=Y) / sy
        py_samp_np = np.ones(sy) / sy
    else:
        py_samp_np = nx.to_numpy(py_samp)

    if py_feat is None:
        py_feat = nx.ones(fy, type_as=Y) / fy
        py_feat_np = np.ones(fy) / fy
    else:
        py_feat_np = nx.to_numpy(py_feat)

    pxy_samp = px_samp[:, None] * py_samp[None, :]
    pxy_feat = px_feat[:, None] * py_feat[None, :]

    # pre-calculate cost constants
    XY_sqr = (X ** 2 @ px_feat)[:, None] + (Y ** 2 @
                                            py_feat)[None, :] + alpha_samp * D_samp
    XY_sqr_T = ((X.T)**2 @ px_samp)[:, None] + ((Y.T)
                                                ** 2 @ py_samp)[None, :] + alpha_feat * D_feat

    # initialize coupling and dual vectors
    if warmstart is None:
        pi_sample, pi_feature = pxy_samp, pxy_feat  # size sx x sy and size fx x fy
        duals_samp = (nx.zeros(sx, type_as=X), nx.zeros(
            sy, type_as=Y))  # shape sx, sy
        duals_feat = (nx.zeros(fx, type_as=X), nx.zeros(
            fy, type_as=Y))  # shape fx, fy
    else:
        pi_sample, pi_feature = warmstart["pi_sample"], warmstart["pi_feature"]
        duals_samp, duals_feat = warmstart["duals_sample"], warmstart["duals_feature"]

    # create shortcuts of functions
    self_sinkhorn = partial(sinkhorn, method=method_sinkhorn,
                            numItermax=nits_ot, stopThr=tol_sinkhorn, log=True)
    self_get_distance = partial(get_distance, log_pxy_samp=nx.log(pxy_samp),
                                log_pxy_feat=nx.log(pxy_feat), D_samp=D_samp,
                                alpha_samp=alpha_samp, eps=eps)

    # initialize log
    list_coot = [float("inf")]
    err = tol_bcd + 1e-3

    for idx in range(nits_bcd):
        pi_sample_prev = nx.copy(pi_sample)

        # update sample coupling
        ot_cost = XY_sqr - 2 * X @ pi_feature @ Y.T  # size sx x sy
        if eps_samp > 0:
            pi_sample, dict_log = self_sinkhorn(
                a=px_samp, b=py_samp, M=ot_cost, reg=eps_samp, warmstart=duals_samp)
            duals_samp = (nx.log(dict_log["u"]), nx.log(dict_log["v"]))
        elif eps_samp == 0:
            pi_sample, duals_samp = emd_solver(ot_cost, px_samp_np, py_samp_np)

        # update feature coupling
        ot_cost = XY_sqr_T - 2 * X.T @ pi_sample @ Y  # size fx x fy
        if eps_feat > 0:
            pi_feature, dict_log = self_sinkhorn(
                a=px_feat, b=py_feat, M=ot_cost, reg=eps_feat, warmstart=duals_feat)
            duals_feat = (nx.log(dict_log["u"]), nx.log(dict_log["v"]))
        elif eps_feat == 0:
            pi_feature, duals_feat = emd_solver(
                ot_cost, px_feat_np, py_feat_np)

        if idx % eval_bcd == 0:
            # update error
            err = nx.sum(nx.abs(pi_sample - pi_sample_prev))
            coot = self_get_distance(ot_cost, pi_sample, pi_feature)
            list_coot.append(coot)

            if err < tol_bcd or abs(list_coot[-2] - list_coot[-1]) < early_stopping_tol:
                break

            if verbose:
                print(
                    "CO-Optimal Transport cost at iteration {}: {}".format(idx + 1, coot))

    # sanity check
    if nx.sum(nx.isnan(pi_sample)) > 0 or nx.sum(nx.isnan(pi_feature)) > 0:
        print("There is NaN in coupling.")

    if log:
        dict_log = {"duals_sample": duals_samp,
                    "duals_feature": duals_feat,
                    "distances": list_coot[1:]}

        return pi_sample, pi_feature, dict_log

    else:
        return pi_sample, pi_feature


def co_optimal_transport2(X, Y, px=(None, None), py=(None, None), eps=(0, 0),
                          alpha=(1, 1), D=(None, None), warmstart=None,
                          log=False, verbose=False, early_stopping_tol=1e-6,
                          nits_bcd=100, tol_bcd=1e-7, eval_bcd=1,
                          nits_ot=500, tol_sinkhorn=1e-7,
                          method_sinkhorn="sinkhorn"):
    r"""
    Return the CO-Optimal Transport distance between
    :math:`(\mathbf{X}, \mathbf{p}_{xs}, \mathbf{p}_{xf})` and
    :math:`(\mathbf{Y}, \mathbf{p}_{ys}, \mathbf{p}_{yf})`.

    The function solves the following CO-Optimal Transport (COOT) problem:

    .. math::
        \mathbf{COOT}_{\varepsilon} = \mathop{\arg \min}_{\mathbf{P}, \mathbf{Q}}
        \quad \sum_{i,j,k,l}
        (\mathbf{X}_{i,k} - \mathbf{Y}_{j,l})^2 \mathbf{P}_{i,j} \mathbf{Q}_{k,l}
        + \alpha_1 \sum_{i,j} \mathbf{P}_{i,j} \mathbf{D^{(s)}}_{i, j}
        + \alpha_2 \sum_{k, l} \mathbf{Q}_{k,l} \mathbf{D^{(f)}}_{k, l}
        + \varepsilon_1 \mathbf{KL}(\mathbf{P} | \mathbf{p}_{xs} \mathbf{p}_{ys}^T)
        + \varepsilon_2 \mathbf{KL}(\mathbf{Q} | \mathbf{p}_{xf} \mathbf{p}_{yf}^T)

    Where :

    - :math:`\mathbf{X}`: Data matrix in the source space
    - :math:`\mathbf{Y}`: Data matrix in the target space
    - :math:`\mathbf{D^{(s)}}`: Additional sample matrix
    - :math:`\mathbf{D^{(f)}}`: Additional feature matrix
    - :math:`\mathbf{p}_{xs}`: Distribution of the samples in the source space
    - :math:`\mathbf{p}_{xf}`: Distribution of the features in the source space
    - :math:`\mathbf{p}_{ys}`: Distribution of the samples in the target space
    - :math:`\mathbf{p}_{yf}`: Distribution of the features in the target space

    .. note:: This function allows epsilons to be zero.
    In that case, the `ot.lp.emd` solver of POT will be used.

    Parameters
    ----------
    X : (sx, fx) array-like, float
        First input matrix.
    Y : (sy, fy) array-like, float
        Second input matrix.
    px : (sx, fx) tuple, float, optional (default = (None,None))
        Histogram assigned on rows (samples) and columns (features) of X.
        Uniform distribution by default.
    py : (sy, fy) tuple, float, optional (default = (None,None))
        Histogram assigned on rows (samples) and columns (features) of Y.
        Uniform distribution by default.
    eps : (scalar, scalar) tuple, float or int (default = (0,0))
        Regularisation parameters for entropic approximation of sample and feature couplings.
        Allow the case where eps contains 0. In that case, the EMD solver is used instead of
        Sinkhorn solver.
    alpha : (scalar, scalar) tuple, float or int, optional (default = (1,1))
        Interpolation parameter for fused COOT with respect to the sample and feature couplings.
    D : tuple of matrices (sx, sy) and (fx, fy), float, optional (default = (None,None))
        Sample and feature matrices, in case of fused COOT.
    warmstart : dictionary, optional (default = None)
        Containing 4 keys:
            + "duals_sample" and "duals_feature" whose values are
            tuples of 2 vectors of size (sx, sy) and (fx, fy).
            Initialization of sample and feature dual vectors
            if using Sinkhorn algorithm. Zero vectors by default.
            + "pi_sample" and "pi_feature" whose values are matrices
            of size (sx, sy) and (fx, fy).
            Initialization of sample and feature couplings.
            Uniform distributions by default.
    nits_bcd : int, optional (default = 100)
        Number of Block Coordinate Descent (BCD) iterations to solve COOT.
    tol_bcd : float, optional (default = 1e-7)
        Tolerance of BCD scheme. If the L1-norm between the current and previous
        sample couplings is under this threshold, then stop BCD scheme.
    eval_bcd : int, optional (default = 1)
        Multiplier of iteration at which the COOT cost is evaluated. For example,
        if `eval_bcd = 8`, then the cost is calculated at iterations 8, 16, 24, etc...
    nits_ot : int, optional (default = 100)
        Number of iterations to solve each of the
        two optimal transport problems in each BCD iteration.
    tol_sinkhorn : float, optional (default = 1e-7)
        Tolerance of Sinkhorn algorithm to stop the Sinkhorn scheme for
        entropic optimal transport problem (if any) in each BCD iteration.
        Only triggered when Sinkhorn solver is used.
    method_sinkhorn : string, optional (default = "sinkhorn")
        Method used in POT's `ot.sinkhorn` solver.
        Only support "sinkhorn" and "sinkhorn_log".
    early_stopping_tol : float, optional (default = 1e-6)
        Tolerance for the early stopping. If the absolute difference between
        the last 2 recorded COOT distances is under this tolerance, then stop BCD scheme.
    log : bool, optional (default = False)
        If True then the cost and 4 dual vectors, including
        2 from sample and 2 from feature couplings, are recorded.
    verbose : bool, optional (default = False)
        If True then print the COOT cost at every multiplier of `eval_bcd`-th iteration.

    Returns
    -------
    CO-Optimal Transport distance : float

    If log is True, then also return the dictionary output of `co_optimal_transport` solver.

    References
    ----------
    .. [47] I. Redko, T. Vayer, R. Flamary, and N. Courty, CO-Optimal Transport,
        Advances in Neural Information Processing Systems, 33 (2020).
    """

    pi_sample, pi_feature, dict_log = co_optimal_transport(X=X, Y=Y, px=px, py=py, eps=eps,
                                                           alpha=alpha, D=D, warmstart=warmstart,
                                                           nits_bcd=nits_bcd, tol_bcd=tol_bcd,
                                                           eval_bcd=eval_bcd, nits_ot=nits_ot,
                                                           tol_sinkhorn=tol_sinkhorn, method_sinkhorn=method_sinkhorn,
                                                           early_stopping_tol=early_stopping_tol,
                                                           log=True, verbose=verbose)

    X, Y = list_to_array(X, Y)
    nx = get_backend(X, Y)

    sx, fx = X.shape
    sy, fy = Y.shape

    px_samp, px_feat = px
    py_samp, py_feat = py

    if px_samp is None:
        px_samp = nx.ones(sx, type_as=X) / sx
    if px_feat is None:
        px_feat = nx.ones(fx, type_as=X) / fx
    if py_samp is None:
        py_samp = nx.ones(sy, type_as=Y) / sy
    if py_feat is None:
        py_feat = nx.ones(fy, type_as=Y) / fy

    vx_samp, vy_samp = dict_log["duals_sample"]
    vx_feat, vy_feat = dict_log["duals_feature"]

    gradX = 2 * X * (px_samp[:, None] * px_feat[None, :]) - \
        2 * pi_sample @ Y @ pi_feature.T  # shape (sx, fx)
    gradY = 2 * Y * (py_samp[:, None] * py_feat[None, :]) - \
        2 * pi_sample.T @ X @ pi_feature  # shape (sy, fy)

    coot = dict_log["distances"][-1]
    coot = nx.set_gradients(coot, (px_samp, px_feat, py_samp, py_feat, X, Y),
                            (vx_samp, vx_feat, vy_samp, vy_feat, gradX, gradY))

    if log:
        return coot, dict_log

    else:
        return coot
