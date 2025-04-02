# -*- coding: utf-8 -*-
"""
CO-Optimal Transport solver
"""

# Author: Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

import warnings
from .lp import emd
from .utils import list_to_array
from .backend import get_backend
from .bregman import sinkhorn


def co_optimal_transport(
    X,
    Y,
    wx_samp=None,
    wx_feat=None,
    wy_samp=None,
    wy_feat=None,
    epsilon=0,
    alpha=0,
    M_samp=None,
    M_feat=None,
    warmstart=None,
    nits_bcd=100,
    tol_bcd=1e-7,
    eval_bcd=1,
    nits_ot=500,
    tol_sinkhorn=1e-7,
    method_sinkhorn="sinkhorn",
    early_stopping_tol=1e-6,
    log=False,
    verbose=False,
):
    r"""Compute the CO-Optimal Transport between two matrices.

    Return the sample and feature transport plans between
    :math:`(\mathbf{X}, \mathbf{w}_{xs}, \mathbf{w}_{xf})` and
    :math:`(\mathbf{Y}, \mathbf{w}_{ys}, \mathbf{w}_{yf})`.

    The function solves the following CO-Optimal Transport (COOT) problem:

    .. math::
        \mathbf{COOT}_{\alpha, \varepsilon} = \mathop{\arg \min}_{\mathbf{P}, \mathbf{Q}}
        &\quad \sum_{i,j,k,l}
        (\mathbf{X}_{i,k} - \mathbf{Y}_{j,l})^2 \mathbf{P}_{i,j} \mathbf{Q}_{k,l}
        + \alpha_s \sum_{i,j} \mathbf{P}_{i,j} \mathbf{M^{(s)}}_{i, j} \\
        &+ \alpha_f \sum_{k, l} \mathbf{Q}_{k,l} \mathbf{M^{(f)}}_{k, l}
        + \varepsilon_s \mathbf{KL}(\mathbf{P} | \mathbf{w}_{xs} \mathbf{w}_{ys}^T)
        + \varepsilon_f \mathbf{KL}(\mathbf{Q} | \mathbf{w}_{xf} \mathbf{w}_{yf}^T)

    Where :

    - :math:`\mathbf{X}`: Data matrix in the source space
    - :math:`\mathbf{Y}`: Data matrix in the target space
    - :math:`\mathbf{M^{(s)}}`: Additional sample matrix
    - :math:`\mathbf{M^{(f)}}`: Additional feature matrix
    - :math:`\mathbf{w}_{xs}`: Distribution of the samples in the source space
    - :math:`\mathbf{w}_{xf}`: Distribution of the features in the source space
    - :math:`\mathbf{w}_{ys}`: Distribution of the samples in the target space
    - :math:`\mathbf{w}_{yf}`: Distribution of the features in the target space

    .. note:: This function allows epsilon to be zero.
              In that case, the :any:`ot.lp.emd` solver of POT will be used.

    Parameters
    ----------
    X : (n_sample_x, n_feature_x) array-like, float
        First input matrix.
    Y : (n_sample_y, n_feature_y) array-like, float
        Second input matrix.
    wx_samp : (n_sample_x, ) array-like, float, optional (default = None)
        Histogram assigned on rows (samples) of matrix X.
        Uniform distribution by default.
    wx_feat : (n_feature_x, ) array-like, float, optional (default = None)
        Histogram assigned on columns (features) of matrix X.
        Uniform distribution by default.
    wy_samp : (n_sample_y, ) array-like, float, optional (default = None)
        Histogram assigned on rows (samples) of matrix Y.
        Uniform distribution by default.
    wy_feat : (n_feature_y, ) array-like, float, optional (default = None)
        Histogram assigned on columns (features) of matrix Y.
        Uniform distribution by default.
    epsilon : scalar or indexable object of length 2, float or int, optional (default = 0)
        Regularization parameters for entropic approximation of sample and feature couplings.
        Allow the case where epsilon contains 0. In that case, the EMD solver is used instead of
        Sinkhorn solver. If epsilon is scalar, then the same epsilon is applied to
        both regularization of sample and feature couplings.
    alpha : scalar or indexable object of length 2, float or int, optional (default = 0)
        Coefficient parameter of linear terms with respect to the sample and feature couplings.
        If alpha is scalar, then the same alpha is applied to both linear terms.
    M_samp : (n_sample_x, n_sample_y), float, optional (default = None)
        Sample matrix with respect to the linear term on sample coupling.
    M_feat : (n_feature_x, n_feature_y), float, optional (default = None)
        Feature matrix with respect to the linear term on feature coupling.
    warmstart : dictionary, optional (default = None)
        Contains 4 keys:
            - "duals_sample" and "duals_feature" whose values are
              tuples of 2 vectors of size (n_sample_x, n_sample_y) and (n_feature_x, n_feature_y).
              Initialization of sample and feature dual vectors
              if using Sinkhorn algorithm. Zero vectors by default.

            - "pi_sample" and "pi_feature" whose values are matrices
              of size (n_sample_x, n_sample_y) and (n_feature_x, n_feature_y).
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
    pi_samp : (n_sample_x, n_sample_y) array-like, float
        Sample coupling matrix.
    pi_feat : (n_feature_x, n_feature_y) array-like, float
        Feature coupling matrix.
    log : dictionary, optional
        Returned if `log` is True. The keys are:
            duals_sample : (n_sample_x, n_sample_y) tuple, float
                Pair of dual vectors when solving OT problem w.r.t the sample coupling.
            duals_feature : (n_feature_x, n_feature_y) tuple, float
                Pair of dual vectors when solving OT problem w.r.t the feature coupling.
            distances : list, float
                List of COOT distances along iterations.

    References
    ----------
    .. [49] I. Redko, T. Vayer, R. Flamary, and N. Courty, CO-Optimal Transport,
        Advances in Neural Information Processing ny_sampstems, 33 (2020).
    """

    # Main function

    if method_sinkhorn not in ["sinkhorn", "sinkhorn_log"]:
        raise ValueError(
            "Method {} is not supported in CO-Optimal Transport.".format(
                method_sinkhorn
            )
        )

    X, Y = list_to_array(X, Y)
    nx = get_backend(X, Y)

    if isinstance(epsilon, float) or isinstance(epsilon, int):
        eps_samp, eps_feat = epsilon, epsilon
    else:
        if len(epsilon) != 2:
            raise ValueError(
                "Epsilon must be either a scalar or an indexable object of length 2."
            )
        else:
            eps_samp, eps_feat = epsilon[0], epsilon[1]

    if isinstance(alpha, float) or isinstance(alpha, int):
        alpha_samp, alpha_feat = alpha, alpha
    else:
        if len(alpha) != 2:
            raise ValueError(
                "Alpha must be either a scalar or an indexable object of length 2."
            )
        else:
            alpha_samp, alpha_feat = alpha[0], alpha[1]

    # constant input variables
    if M_samp is None or alpha_samp == 0:
        M_samp, alpha_samp = 0, 0
    if M_feat is None or alpha_feat == 0:
        M_feat, alpha_feat = 0, 0

    nx_samp, nx_feat = X.shape
    ny_samp, ny_feat = Y.shape

    # measures on rows and columns
    if wx_samp is None:
        wx_samp = nx.ones(nx_samp, type_as=X) / nx_samp
    if wx_feat is None:
        wx_feat = nx.ones(nx_feat, type_as=X) / nx_feat
    if wy_samp is None:
        wy_samp = nx.ones(ny_samp, type_as=Y) / ny_samp
    if wy_feat is None:
        wy_feat = nx.ones(ny_feat, type_as=Y) / ny_feat

    wxy_samp = wx_samp[:, None] * wy_samp[None, :]
    wxy_feat = wx_feat[:, None] * wy_feat[None, :]

    # pre-calculate cost constants
    XY_sqr = (X**2 @ wx_feat)[:, None] + (Y**2 @ wy_feat)[None, :] + alpha_samp * M_samp
    XY_sqr_T = (
        ((X.T) ** 2 @ wx_samp)[:, None]
        + ((Y.T) ** 2 @ wy_samp)[None, :]
        + alpha_feat * M_feat
    )

    # initialize coupling and dual vectors
    if warmstart is None:
        pi_samp, pi_feat = (
            wxy_samp,
            wxy_feat,
        )  # shape nx_samp x ny_samp and nx_feat x ny_feat
        duals_samp = (
            nx.zeros(nx_samp, type_as=X),
            nx.zeros(ny_samp, type_as=Y),
        )  # shape nx_samp, ny_samp
        duals_feat = (
            nx.zeros(nx_feat, type_as=X),
            nx.zeros(ny_feat, type_as=Y),
        )  # shape nx_feat, ny_feat
    else:
        pi_samp, pi_feat = warmstart["pi_sample"], warmstart["pi_feature"]
        duals_samp, duals_feat = warmstart["duals_sample"], warmstart["duals_feature"]

    # initialize log
    list_coot = [float("inf")]
    err = tol_bcd + 1e-3

    for idx in range(nits_bcd):
        pi_samp_prev = nx.copy(pi_samp)

        # update sample coupling
        ot_cost = XY_sqr - 2 * X @ pi_feat @ Y.T  # size nx_samp x ny_samp
        if eps_samp > 0:
            pi_samp, dict_log = sinkhorn(
                a=wx_samp,
                b=wy_samp,
                M=ot_cost,
                reg=eps_samp,
                method=method_sinkhorn,
                numItermax=nits_ot,
                stopThr=tol_sinkhorn,
                log=True,
                warmstart=duals_samp,
            )
            duals_samp = (nx.log(dict_log["u"]), nx.log(dict_log["v"]))
        elif eps_samp == 0:
            pi_samp, dict_log = emd(
                a=wx_samp, b=wy_samp, M=ot_cost, numItermax=nits_ot, log=True
            )
            duals_samp = (dict_log["u"], dict_log["v"])
        # update feature coupling
        ot_cost = XY_sqr_T - 2 * X.T @ pi_samp @ Y  # size nx_feat x ny_feat
        if eps_feat > 0:
            pi_feat, dict_log = sinkhorn(
                a=wx_feat,
                b=wy_feat,
                M=ot_cost,
                reg=eps_feat,
                method=method_sinkhorn,
                numItermax=nits_ot,
                stopThr=tol_sinkhorn,
                log=True,
                warmstart=duals_feat,
            )
            duals_feat = (nx.log(dict_log["u"]), nx.log(dict_log["v"]))
        elif eps_feat == 0:
            pi_feat, dict_log = emd(
                a=wx_feat, b=wy_feat, M=ot_cost, numItermax=nits_ot, log=True
            )
            duals_feat = (dict_log["u"], dict_log["v"])

        if idx % eval_bcd == 0:
            # update error
            err = nx.sum(nx.abs(pi_samp - pi_samp_prev))

            # COOT part
            coot = nx.sum(ot_cost * pi_feat)
            if alpha_samp != 0:
                coot = coot + alpha_samp * nx.sum(M_samp * pi_samp)
            # Entropic part
            if eps_samp != 0:
                coot = coot + eps_samp * nx.kl_div(pi_samp, wxy_samp)
            if eps_feat != 0:
                coot = coot + eps_feat * nx.kl_div(pi_feat, wxy_feat)
            list_coot.append(coot)

            if err < tol_bcd or abs(list_coot[-2] - list_coot[-1]) < early_stopping_tol:
                break

            if verbose:
                print(
                    "CO-Optimal Transport cost at iteration {}: {}".format(
                        idx + 1, coot
                    )
                )

    # sanity check
    if nx.sum(nx.isnan(pi_samp)) > 0 or nx.sum(nx.isnan(pi_feat)) > 0:
        warnings.warn("There is NaN in coupling.")

    if log:
        dict_log = {
            "duals_sample": duals_samp,
            "duals_feature": duals_feat,
            "distances": list_coot[1:],
        }

        return pi_samp, pi_feat, dict_log

    else:
        return pi_samp, pi_feat


def co_optimal_transport2(
    X,
    Y,
    wx_samp=None,
    wx_feat=None,
    wy_samp=None,
    wy_feat=None,
    epsilon=0,
    alpha=0,
    M_samp=None,
    M_feat=None,
    warmstart=None,
    log=False,
    verbose=False,
    early_stopping_tol=1e-6,
    nits_bcd=100,
    tol_bcd=1e-7,
    eval_bcd=1,
    nits_ot=500,
    tol_sinkhorn=1e-7,
    method_sinkhorn="sinkhorn",
):
    r"""Compute the CO-Optimal Transport distance between two measures.

    Returns the CO-Optimal Transport distance between
    :math:`(\mathbf{X}, \mathbf{w}_{xs}, \mathbf{w}_{xf})` and
    :math:`(\mathbf{Y}, \mathbf{w}_{ys}, \mathbf{w}_{yf})`.

    The function solves the following CO-Optimal Transport (COOT) problem:

    .. math::
        \mathbf{COOT}_{\alpha, \varepsilon} = \mathop{\arg \min}_{\mathbf{P}, \mathbf{Q}}
        &\quad \sum_{i,j,k,l}
        (\mathbf{X}_{i,k} - \mathbf{Y}_{j,l})^2 \mathbf{P}_{i,j} \mathbf{Q}_{k,l}
        + \alpha_1 \sum_{i,j} \mathbf{P}_{i,j} \mathbf{M^{(s)}}_{i, j} \\
        &+ \alpha_2 \sum_{k, l} \mathbf{Q}_{k,l} \mathbf{M^{(f)}}_{k, l}
        + \varepsilon_1 \mathbf{KL}(\mathbf{P} | \mathbf{w}_{xs} \mathbf{w}_{ys}^T)
        + \varepsilon_2 \mathbf{KL}(\mathbf{Q} | \mathbf{w}_{xf} \mathbf{w}_{yf}^T)

    where :

    - :math:`\mathbf{X}`: Data matrix in the source space
    - :math:`\mathbf{Y}`: Data matrix in the target space
    - :math:`\mathbf{M^{(s)}}`: Additional sample matrix
    - :math:`\mathbf{M^{(f)}}`: Additional feature matrix
    - :math:`\mathbf{w}_{xs}`: Distribution of the samples in the source space
    - :math:`\mathbf{w}_{xf}`: Distribution of the features in the source space
    - :math:`\mathbf{w}_{ys}`: Distribution of the samples in the target space
    - :math:`\mathbf{w}_{yf}`: Distribution of the features in the target space

    .. note:: This function allows epsilon to be zero.
              In that case, the :any:`ot.lp.emd` solver of POT will be used.

    Parameters
    ----------
    X : (n_sample_x, n_feature_x) array-like, float
        First input matrix.
    Y : (n_sample_y, n_feature_y) array-like, float
        Second input matrix.
    wx_samp : (n_sample_x, ) array-like, float, optional (default = None)
        Histogram assigned on rows (samples) of matrix X.
        Uniform distribution by default.
    wx_feat : (n_feature_x, ) array-like, float, optional (default = None)
        Histogram assigned on columns (features) of matrix X.
        Uniform distribution by default.
    wy_samp : (n_sample_y, ) array-like, float, optional (default = None)
        Histogram assigned on rows (samples) of matrix Y.
        Uniform distribution by default.
    wy_feat : (n_feature_y, ) array-like, float, optional (default = None)
        Histogram assigned on columns (features) of matrix Y.
        Uniform distribution by default.
    epsilon : scalar or indexable object of length 2, float or int, optional (default = 0)
        Regularization parameters for entropic approximation of sample and feature couplings.
        Allow the case where epsilon contains 0. In that case, the EMD solver is used instead of
        Sinkhorn solver. If epsilon is scalar, then the same epsilon is applied to
        both regularization of sample and feature couplings.
    alpha : scalar or indexable object of length 2, float or int, optional (default = 0)
        Coefficient parameter of linear terms with respect to the sample and feature couplings.
        If alpha is scalar, then the same alpha is applied to both linear terms.
    M_samp : (n_sample_x, n_sample_y), float, optional (default = None)
        Sample matrix with respect to the linear term on sample coupling.
    M_feat : (n_feature_x, n_feature_y), float, optional (default = None)
        Feature matrix with respect to the linear term on feature coupling.
    warmstart : dictionary, optional (default = None)
        Contains 4 keys:
            - "duals_sample" and "duals_feature" whose values are
            tuples of 2 vectors of size (n_sample_x, n_sample_y) and (n_feature_x, n_feature_y).
            Initialization of sample and feature dual vectors
            if using Sinkhorn algorithm. Zero vectors by default.
            - "pi_sample" and "pi_feature" whose values are matrices
            of size (n_sample_x, n_sample_y) and (n_feature_x, n_feature_y).
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
    float
        CO-Optimal Transport distance.
    dict
        Contains logged information from :any:`co_optimal_transport` solver.
        Only returned if `log` parameter is True

    References
    ----------
    .. [47] I. Redko, T. Vayer, R. Flamary, and N. Courty, CO-Optimal Transport,
        Advances in Neural Information Processing ny_sampstems, 33 (2020).
    """

    pi_samp, pi_feat, dict_log = co_optimal_transport(
        X=X,
        Y=Y,
        wx_samp=wx_samp,
        wx_feat=wx_feat,
        wy_samp=wy_samp,
        wy_feat=wy_feat,
        epsilon=epsilon,
        alpha=alpha,
        M_samp=M_samp,
        M_feat=M_feat,
        warmstart=warmstart,
        nits_bcd=nits_bcd,
        tol_bcd=tol_bcd,
        eval_bcd=eval_bcd,
        nits_ot=nits_ot,
        tol_sinkhorn=tol_sinkhorn,
        method_sinkhorn=method_sinkhorn,
        early_stopping_tol=early_stopping_tol,
        log=True,
        verbose=verbose,
    )

    X, Y = list_to_array(X, Y)
    nx = get_backend(X, Y)

    nx_samp, nx_feat = X.shape
    ny_samp, ny_feat = Y.shape

    # measures on rows and columns
    if wx_samp is None:
        wx_samp = nx.ones(nx_samp, type_as=X) / nx_samp
    if wx_feat is None:
        wx_feat = nx.ones(nx_feat, type_as=X) / nx_feat
    if wy_samp is None:
        wy_samp = nx.ones(ny_samp, type_as=Y) / ny_samp
    if wy_feat is None:
        wy_feat = nx.ones(ny_feat, type_as=Y) / ny_feat

    vx_samp, vy_samp = dict_log["duals_sample"]
    vx_feat, vy_feat = dict_log["duals_feature"]

    gradX = (
        2 * X * (wx_samp[:, None] * wx_feat[None, :]) - 2 * pi_samp @ Y @ pi_feat.T
    )  # shape (nx_samp, nx_feat)
    gradY = (
        2 * Y * (wy_samp[:, None] * wy_feat[None, :]) - 2 * pi_samp.T @ X @ pi_feat
    )  # shape (ny_samp, ny_feat)

    coot = dict_log["distances"][-1]
    coot = nx.set_gradients(
        coot,
        (wx_samp, wx_feat, wy_samp, wy_feat, X, Y),
        (vx_samp, vx_feat, vy_samp, vy_feat, gradX, gradY),
    )

    if log:
        return coot, dict_log

    else:
        return coot
