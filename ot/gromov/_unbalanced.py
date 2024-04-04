# -*- coding: utf-8 -*-
"""
Unbalanced Co-Optimal Transport solver
"""

# Author: Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#         Alexis Thual <alexis.thual@cea.fr>
#
# License: MIT License

import warnings
from functools import partial
from ot.backend import get_backend
from ot.utils import list_to_array, get_parameter_pair
from ot.unbalanced import sinkhorn_unbalanced, mm_unbalanced, lbfgsb_unbalanced


def fused_unbalanced_cross_spaces_divergence(
        X, Y, wx_samp=None, wx_feat=None, wy_samp=None, wy_feat=None,
        reg_marginals=None, epsilon=0, reg_type="joint", divergence="kl",
        unbalanced_solver="scaling", alpha=0, M_samp=None, M_feat=None,
        init_pi=None, init_duals=None, max_iter=100, tol=1e-7,
        max_iter_ot=500, tol_ot=1e-7, method_sinkhorn="sinkhorn",
        log=False, verbose=False, **kwargs_solver
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
        Coeffficient parameter of linear terms with respect to the sample and feature couplings.
        If alpha is scalar, then the same alpha is applied to both linear terms.
    M_samp : (n_sample_x, n_sample_y), float, optional (default = None)
        Sample matrix with respect to the linear term on sample coupling.
    M_feat : (n_feature_x, n_feature_y), float, optional (default = None)
        Feature matrix with respect to the linear term on feature coupling.
    divergence : string, optional (default = "kl")
        If divergence = "kl", then D is the Kullback-Leibler divergence.
        If divergence = "l2", then D is the half squared Euclidean norm.
    unbalanced_solver : string, optional (default = "scaling")
        Solver for the unbalanced OT subroutine.
        If divergence = "kl", then unbalanced_solver can be: "scaling", "mm", "lbfgsb"
        If divergence = "l2", then unbalanced_solver can be "mm", "lbfgsb"
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
    max_iter : int, optional (default = 100)
        Number of Block Coordinate Descent (BCD) iterations to solve COOT.
    tol : float, optional (default = 1e-7)
        Tolerance of BCD scheme. If the L1-norm between the current and previous
        sample couplings is under this threshold, then stop BCD scheme.
    max_iter_ot : int, optional (default = 100)
        Number of iterations to solve each of the
        two optimal transport problems in each BCD iteration.
    tol_ot : float, optional (default = 1e-7)
        Tolerance of Sinkhorn algorithm to stop the Sinkhorn scheme for
        entropic optimal transport problem (if any) in each BCD iteration.
        Only triggered when Sinkhorn solver is used.
    method_sinkhorn : string, optional (default = "sinkhorn")
        Method used in POT's `ot.sinkhorn` solver when divergence = "kl" and
        unbalanced_solver = "scaling". Only support method_sinkhorn = "sinkhorn"
        and method_sinkhorn = "sinkhorn_log".
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
            linear : float
                Linear part of the cost.
            ucoot : float
                Total cost.

    References
    ----------
    .. [49] I. Redko, T. Vayer, R. Flamary, and N. Courty, CO-Optimal Transport,
        Advances in Neural Information Processing ny_sampstems, 33 (2020).
    """

    # SUPPORT FUNCTIONS FOR KL DIVERGENCE

    def approx_product_kl(pi, pi1, pi2, a, b):
        """
        Calculate
        < pi, log pi / (a \otimes b) >
        = <pi, log pi> - <pi1, log a> - <pi2, log b>
        """
        res = nx.sum(pi * nx.log(pi + 1.0 * (pi == 0))) \
            - nx.sum(pi1 * nx.log(a)) - nx.sum(pi2 * nx.log(b))
        return res

    def product_kl(pi, pi1, pi2, a, b):
        """
        Calculate KL(pi, a \otimes b)
        """
        res = approx_product_kl(pi, pi1, pi2, a, b) \
            - nx.sum(pi1) + nx.sum(a) * nx.sum(b)
        return res

    def product_div(pi, pi1, pi2, a, b):
        """
        Calculate D(pi, a \otimes b)
        """
        if divergence == "kl":
            res = product_kl(pi, pi1, pi2, a, b) \
                - nx.sum(pi1) + nx.sum(a) * nx.sum(b)
        elif divergence == "l2":
            res = (nx.sum(pi**2) + nx.sum(a**2) * nx.sum(b**2)
                   - 2 * nx.dot(a, pi @ b)) / 2
        return res

    def approx_kl(p, q):
        return nx.sum(p * nx.log(p + 1.0 * (p == 0))) - nx.sum(p * nx.log(q))

    def generalized_kl(p, q):
        return approx_kl(p, q) - nx.sum(p) + nx.sum(q)

    def quad_kl(mu, nu, alpha, beta):
        """
        Calculate the KL divergence between two product measures:
        KL(mu \otimes nu, alpha \otimes beta) =
        m_mu * KL(nu, beta) + m_nu * KL(mu, alpha) +
        (m_mu - m_alpha) * (m_nu - m_beta)

        Parameters
        ----------
        mu: vector or matrix
        nu: vector or matrix
        alpha: vector or matrix with the same size as mu
        beta: vector or matrix with the same size as nu

        Returns
        ----------
        KL divergence between two product measures
        """

        m_mu, m_nu = nx.sum(mu), nx.sum(nu)
        m_alpha, m_beta = nx.sum(alpha), nx.sum(beta)
        const = (m_mu - m_alpha) * (m_nu - m_beta)
        quad_kl = m_nu * generalized_kl(mu, alpha) + \
            m_mu * generalized_kl(nu, beta) + const

        return quad_kl

    def quad_l2(mu, nu, alpha, beta):
        norm = nx.sum(alpha**2) * nx.sum(beta**2) \
            - 2 * nx.sum(alpha * mu) * nx.sum(beta * nu) \
            + nx.sum(mu**2) * nx.sum(nu**2)

        return norm / 2

    def quad_div(mu, nu, alpha, beta):
        if divergence == "kl":
            return quad_kl(mu, nu, alpha, beta)
        elif divergence == "l2":
            return quad_l2(mu, nu, alpha, beta)

    def local_cost(data, pi, tuple_p, hyperparams):
        """
        Calculate cost matrix of the UOT problem
        """
        X_sqr, Y_sqr, X, Y, M = data
        rho_x, rho_y, eps = hyperparams
        a, b = tuple_p

        pi1, pi2 = nx.sum(pi, 1), nx.sum(pi, 0)
        A, B = X_sqr @ pi1, Y_sqr @ pi2
        uot_cost = A[:, None] + B[None, :] - 2 * X @ pi @ Y.T + M

        if divergence == "kl":
            if rho_x != float("inf") and rho_x != 0:
                uot_cost = uot_cost + rho_x * approx_kl(pi1, a)
            if rho_y != float("inf") and rho_y != 0:
                uot_cost = uot_cost + rho_y * approx_kl(pi2, b)
            if reg_type == "joint" and eps > 0:
                uot_cost = uot_cost + eps * approx_product_kl(pi, pi1, pi2, a, b)

        return uot_cost

    def total_cost(M_linear, data, tuple_pxy_samp,
                   tuple_pxy_feat, pi_samp, pi_feat, hyperparams):

        rho_x, rho_y, eps_samp, eps_feat = hyperparams
        M_samp, M_feat = M_linear
        px_samp, py_samp, pxy_samp = tuple_pxy_samp
        px_feat, py_feat, pxy_feat = tuple_pxy_feat
        X_sqr, Y_sqr, X, Y = data

        pi1_samp, pi2_samp = nx.sum(pi_samp, 1), nx.sum(pi_samp, 0)
        pi1_feat, pi2_feat = nx.sum(pi_feat, 1), nx.sum(pi_feat, 0)

        A_sqr = nx.dot(X_sqr @ pi1_feat, pi1_samp)
        B_sqr = nx.dot(Y_sqr @ pi2_feat, pi2_samp)
        AB = (X @ pi_feat @ Y.T) * pi_samp
        linear_cost = A_sqr + B_sqr - 2 * nx.sum(AB)

        if linear_cost < 0:
            raise Warning("The linear cost should be nonnegative.")

        ucoot_cost = linear_cost
        if M_samp != 0:
            ucoot_cost = ucoot_cost + nx.sum(pi_samp * M_samp)
        if M_feat != 0:
            ucoot_cost = ucoot_cost + nx.sum(pi_feat * M_feat)

        if rho_x != float("inf") and rho_x != 0:
            ucoot_cost = ucoot_cost + \
                rho_x * quad_div(pi1_samp, pi1_feat, px_samp, px_feat)
        if rho_y != float("inf") and rho_y != 0:
            ucoot_cost = ucoot_cost + \
                rho_y * quad_div(pi2_samp, pi2_feat, py_samp, py_feat)

        if reg_type == "joint" and eps_samp != 0:
            ucoot_cost = ucoot_cost + \
                eps_samp * quad_div(pi_samp, pi_feat, pxy_samp, pxy_feat)
        elif reg_type == "independent":
            if eps_samp != 0:
                div_samp = product_div(pi_samp, pi1_samp, pi2_samp, px_samp, py_samp)
                ucoot_cost = ucoot_cost + eps_samp * div_samp
            if eps_feat != 0:
                div_feat = product_div(pi_feat, pi1_feat, pi2_feat, px_feat, py_feat)
                ucoot_cost = ucoot_cost + eps_feat * div_feat

        return linear_cost, ucoot_cost

    # SUPPORT FUNCTIONS FOR SQUARED L2 NORM

    def parameters_uot_l2(pi, tuple_weights, hyperparams):
        """Compute parameters of the L2 loss."""

        rho_x, rho_y, eps = hyperparams
        wx, wy, wxy = tuple_weights

        pi1, pi2 = nx.sum(pi, 1), nx.sum(pi, 0)
        l2_pi1, l2_pi2, l2_pi = nx.sum(pi1**2), nx.sum(pi2**2), nx.sum(pi**2)

        weight_wx = nx.sum(pi1 * wx) / l2_pi1
        weight_wy = nx.sum(pi2 * wy) / l2_pi2
        weight_wxy = nx.sum(pi * wxy) / l2_pi if reg_type == "joint" else 1
        weighted_w = (
            weight_wx * wx,
            weight_wy * wy,
            weight_wxy * wxy,
        )

        new_rho = (rho_x * l2_pi1, rho_y * l2_pi2)
        new_eps = eps * l2_pi if reg_type == "joint" else eps

        return weighted_w, new_rho, new_eps

    # Unbalanced KL
    def uot_kl(wx, wy, wxy, cost, eps, rho, init_pi, init_duals):
        if unbalanced_solver == "scaling":
            pi, log = sinkhorn_unbalanced(
                a=wx, b=wy, M=cost, reg=eps, reg_m=rho, reg_type="kl",
                warmstart=init_duals, method=method_sinkhorn,
                numItermax=max_iter_ot, stopThr=tol_ot, verbose=False, log=True
            )
            duals = (log['logu'], log['logv'])

        elif unbalanced_solver == "mm":
            pi = mm_unbalanced(a=wx, b=wy, M=cost, reg_m=rho,
                               c=wxy, reg=eps, div=divergence,
                               G0=init_pi, numItermax=max_iter_ot,
                               stopThr=tol_ot, verbose=False, log=False)
            duals = (None, None)

        elif unbalanced_solver == "lbfgsb":
            pi = lbfgsb_unbalanced(a=wx, b=wy, M=cost, reg=eps, reg_m=rho,
                                   c=wxy, reg_div=divergence,
                                   regm_div=divergence,
                                   G0=init_pi, numItermax=max_iter_ot,
                                   stopThr=tol_ot, method='L-BFGS-B',
                                   verbose=False, log=False)
            duals = (None, None)

        return pi, duals

    # Unbalanced L2
    def uot_l2(wx, wy, wxy, cost, eps, rho, init_pi):
        if unbalanced_solver == "mm":
            pi = mm_unbalanced(a=wx, b=wy, M=cost, reg_m=rho, c=wxy,
                               reg=eps, div=divergence,
                               G0=init_pi, numItermax=max_iter_ot,
                               stopThr=tol_ot, verbose=False, log=False)

        elif unbalanced_solver == "lbfgsb":
            pi = lbfgsb_unbalanced(a=wx, b=wy, M=cost, reg=eps, reg_m=rho,
                                   c=wxy, reg_div=divergence,
                                   regm_div=divergence,
                                   G0=init_pi, numItermax=max_iter_ot,
                                   stopThr=tol_ot, method='L-BFGS-B',
                                   verbose=False, log=False)

        return pi

    # MAIN FUNCTION

    if reg_type not in ["joint", "independent"]:
        raise (NotImplementedError('Unknown reg_type="{}"'.format(reg_type)))
    if divergence not in ["kl", "l2"]:
        raise (NotImplementedError('Unknown divergence="{}"'.format(divergence)))
    if unbalanced_solver not in ["scaling", "mm", "lbfgsb"]:
        raise (NotImplementedError('Unknown method="{}"'.format(unbalanced_solver)))

    X, Y = list_to_array(X, Y)
    nx = get_backend(X, Y)

    # hyperparameters
    alpha_samp, alpha_feat = get_parameter_pair(alpha)
    rho_x, rho_y = get_parameter_pair(reg_marginals)
    eps_samp, eps_feat = get_parameter_pair(epsilon)
    if reg_type == "joint":  # same regularization
        eps_feat = eps_samp
    if unbalanced_solver == "scaling" and divergence == "l2":
        warnings.warn("Scaling algorithm does not support L2 norm. \
                      Divergence is set to 'kl'.")
        divergence = "kl"
    if unbalanced_solver == "scaling" and (eps_samp == 0 or eps_feat == 0):
        warnings.warn("Scaling algorithm does not support unregularized problem. \
                      Solver is set to 'mm'.")
        unbalanced_solver = "mm"

    # constant input variables
    if M_samp is None or alpha_samp == 0:
        M_samp, alpha_samp = 0, 0
    else:
        M_samp = alpha_feat * M_samp
    if M_feat is None or alpha_feat == 0:
        M_feat, alpha_feat = 0, 0
    else:
        M_feat = alpha_feat * M_feat

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

    # initialize coupling and dual vectors
    if init_pi is None:
        init_pi = (None, None)
    pi_samp, pi_feat = init_pi
    pi_samp = wxy_samp if pi_samp is None else pi_samp
    pi_feat = wxy_feat if pi_feat is None else pi_feat

    if init_duals is None:
        init_duals = (None, None)
    duals_samp, duals_feat = init_duals
    if unbalanced_solver == "scaling":
        if duals_samp is None:
            duals_samp = (nx.zeros(nx_samp, type_as=X),
                          nx.zeros(ny_samp, type_as=Y))
        if duals_feat is None:
            duals_feat = (nx.zeros(nx_feat, type_as=X),
                          nx.zeros(ny_feat, type_as=Y))

    # shortcut functions
    X_sqr, Y_sqr = X**2, Y**2
    local_cost_samp = partial(local_cost,
                              data=(X_sqr, Y_sqr, X, Y, M_samp),
                              tuple_p=(wx_feat, wy_feat),
                              hyperparams=(rho_x, rho_y, eps_feat))
    local_cost_feat = partial(local_cost,
                              data=(X_sqr.T, Y_sqr.T, X.T, Y.T, M_feat),
                              tuple_p=(wx_samp, wy_samp),
                              hyperparams=(rho_x, rho_y, eps_samp))
    parameters_uot_l2_samp = partial(
        parameters_uot_l2,
        tuple_weights=(wx_samp, wy_samp, wxy_samp),
        hyperparams=(rho_x, rho_y, eps_samp)
    )
    parameters_uot_l2_feat = partial(
        parameters_uot_l2,
        tuple_weights=(wx_feat, wy_feat, wxy_feat),
        hyperparams=(rho_x, rho_y, eps_feat)
    )

    # rescale params
    if reg_type == "joint":
        K_samp = rho_x * nx.sum(wx_samp) + \
            rho_y * nx.sum(wy_samp) + \
            eps_samp * nx.sum(wxy_samp)
        K_feat = rho_x * nx.sum(wx_feat) + \
            rho_y * nx.sum(wy_feat) + \
            eps_feat * nx.sum(wxy_feat)
        K = rho_x * nx.sum(wx_samp) * nx.sum(wx_feat) + \
            rho_y * nx.sum(wy_samp) * nx.sum(wy_feat) + \
            eps_samp * nx.sum(wxy_samp) * nx.sum(wxy_feat)
        sum_param = rho_x + rho_y + eps_samp

    # initialize log
    if log:
        dict_log = {"error": []}

    for idx in range(max_iter):
        pi_samp_prev = nx.copy(pi_samp)

        # update feature coupling
        mass = nx.sum(pi_samp)
        uot_cost = local_cost_feat(pi=pi_samp)

        if divergence == "kl":
            new_rho = (rho_x * mass, rho_y * mass)
            new_eps = mass * eps_feat if reg_type == "joint" else eps_feat
            pi_feat, duals_feat = uot_kl(wx_feat, wy_feat, wxy_feat, uot_cost,
                                         new_eps, new_rho, pi_feat, duals_feat)
        else:  # divergence == "l2"
            weighted_w, new_rho, new_eps = parameters_uot_l2_feat(pi_feat)
            weighted_wx, weighted_wy, weighted_wxy = weighted_w
            pi_feat = uot_l2(weighted_wx, weighted_wy, weighted_wxy, uot_cost,
                             new_eps, new_rho, pi_feat)
        pi_feat = nx.sqrt(mass / nx.sum(pi_feat)) * pi_feat

        # update sample coupling
        mass = nx.sum(pi_feat)
        uot_cost = local_cost_samp(pi=pi_feat)

        if divergence == "kl":
            new_rho = (rho_x * mass, rho_y * mass)
            new_eps = mass * eps_feat if reg_type == "joint" else eps_feat
            pi_samp, duals_samp = uot_kl(wx_samp, wy_samp, wxy_samp, uot_cost,
                                         new_eps, new_rho, pi_samp, duals_samp)
        else:  # divergence == "l2"
            weighted_w, new_rho, new_eps = parameters_uot_l2_samp(pi_samp)
            weighted_wx, weighted_wy, weighted_wxy = weighted_w
            pi_samp = uot_l2(weighted_wx, weighted_wy, weighted_wxy, uot_cost,
                             new_eps, new_rho, pi_samp)
        pi_samp = nx.sqrt(mass / nx.sum(pi_samp)) * pi_samp  # shape nx x ny

        if reg_type == "joint":
            _, ucoot_cost = total_cost(
                M_linear=(M_samp, M_feat),
                data=(X_sqr, Y_sqr, X, Y),
                tuple_pxy_samp=(wx_samp, wy_samp, wxy_samp),
                tuple_pxy_feat=(wx_feat, wy_feat, wxy_feat),
                pi_samp=pi_samp, pi_feat=pi_feat,
                hyperparams=(rho_x, rho_y, eps_samp, eps_feat)
            )
            m_samp, m_feat = nx.sum(pi_samp), nx.sum(pi_feat)
            scale = nx.exp((K - ucoot_cost) / (sum_param * m_samp * m_feat) - 1)
            pi_samp = nx.sqrt(scale * m_feat / m_samp) * pi_samp
            pi_feat = nx.sqrt(scale * m_samp / m_feat) * pi_feat

        # get error
        err = nx.sum(nx.abs(pi_samp - pi_samp_prev))
        dict_log["error"].append(err)
        if verbose:
            print('{:5d}|{:8e}|'.format(idx + 1, err))
        if err < tol:
            break

    # sanity check
    if nx.sum(nx.isnan(pi_samp)) > 0 or nx.sum(nx.isnan(pi_feat)) > 0:
        warnings.warn("There is NaN in coupling.")

    if log:
        linear_cost, ucoot_cost = total_cost(
            M_linear=(M_samp, M_feat),
            data=(X_sqr, Y_sqr, X, Y),
            tuple_pxy_samp=(wx_samp, wy_samp, wxy_samp),
            tuple_pxy_feat=(wx_feat, wy_feat, wxy_feat),
            pi_samp=pi_samp, pi_feat=pi_feat,
            hyperparams=(rho_x, rho_y, eps_samp, eps_feat)
        )

        dict_log["duals_sample"] = duals_samp
        dict_log["duals_feature"] = duals_feat
        dict_log["linear_cost"] = linear_cost
        dict_log["ucoot_cost"] = ucoot_cost

        return pi_samp, pi_feat, dict_log

    else:
        return pi_samp, pi_feat


def fused_unbalanced_co_optimal_transport(
        X, Y, wx_samp=None, wx_feat=None, wy_samp=None, wy_feat=None,
        reg_marginals=None, epsilon=0, divergence="kl",
        unbalanced_solver="scaling", alpha=0, M_samp=None, M_feat=None,
        init_pi=None, init_duals=None, max_iter=100, tol=1e-7,
        max_iter_ot=500, tol_ot=1e-7, method_sinkhorn="sinkhorn",
        log=False, verbose=False, **kwargs_solve
        ):

    return fused_unbalanced_cross_spaces_divergence(
        X=X, Y=Y, wx_samp=wx_samp, wx_feat=wx_feat,
        wy_samp=wy_samp, wy_feat=wy_feat, reg_marginals=reg_marginals,
        epsilon=epsilon, reg_type="independent",
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M_samp=M_samp, M_feat=M_feat, init_pi=init_pi,
        init_duals=init_duals, max_iter=max_iter, tol=tol,
        max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn=method_sinkhorn, log=log,
        verbose=verbose, **kwargs_solve
    )


def fused_unbalanced_co_optimal_transport2(
    X, Y, wx_samp=None, wx_feat=None, wy_samp=None, wy_feat=None,
    reg_marginals=None, epsilon=0, divergence="kl",
    unbalanced_solver="scaling", alpha=0, M_samp=None, M_feat=None,
    init_pi=None, init_duals=None, max_iter=100, tol=1e-7,
    max_iter_ot=500, tol_ot=1e-7, method_sinkhorn="sinkhorn",
    log=False, verbose=False, **kwargs_solve
    ):

    pi_samp, pi_feat, dict_log = fused_unbalanced_co_optimal_transport(
        X=X, Y=Y, wx_samp=wx_samp, wx_feat=wx_feat, wy_samp=wy_samp, wy_feat=wy_feat,
        reg_marginals=reg_marginals, epsilon=epsilon, divergence=divergence,
        unbalanced_solver=unbalanced_solver, alpha=alpha, M_samp=M_samp, M_feat=M_feat,
        init_pi=init_pi, init_duals=init_duals, max_iter=max_iter, tol=tol,
        max_iter_ot=max_iter_ot, tol_ot=tol_ot, method_sinkhorn=method_sinkhorn,
        log=True, verbose=verbose, **kwargs_solve
        )

    X, Y, pi_samp, pi_feat = list_to_array(X, Y, pi_samp, pi_feat)
    nx = get_backend(X, Y, pi_samp, pi_feat)

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

    # extract parameters
    rho_x, rho_y = get_parameter_pair(reg_marginals)
    alpha_samp, alpha_feat = get_parameter_pair(alpha)
    eps_samp, eps_feat = get_parameter_pair(epsilon)

    # calculate marginals
    pi1_samp, pi2_samp = nx.sum(pi_samp, 1), nx.sum(pi_samp, 0)
    pi1_feat, pi2_feat = nx.sum(pi_feat, 1), nx.sum(pi_feat, 0)
    m_samp, m_feat = nx.sum(pi1_samp), nx.sum(pi1_feat)
    m_wx_feat, m_wx_samp = nx.sum(wx_feat), nx.sum(wx_samp)
    m_wy_feat, m_wy_samp = nx.sum(wy_feat), nx.sum(wy_samp)

    # calculate subgradients
    gradX = 2 * X * (wx_samp[:, None] * wx_feat[None, :]) - \
        2 * pi_samp @ Y @ pi_feat.T  # shape (nx_samp, nx_feat)
    gradY = 2 * Y * (wy_samp[:, None] * wy_feat[None, :]) - \
        2 * pi_samp.T @ X @ pi_feat  # shape (ny_samp, ny_feat)

    gradM_samp = alpha_samp * pi_samp
    gradM_feat = alpha_feat * pi_feat

    grad_wx_samp = rho_x * (m_wx_feat - m_feat * pi1_samp / wx_samp) + \
                eps_samp * (m_wy_samp - pi1_samp / wx_samp)
    grad_wx_feat = rho_x * (m_wx_samp - m_samp * pi1_feat / wx_feat) + \
                eps_feat * (m_wy_feat - pi1_feat / wx_feat)

    grad_wy_samp = rho_y * (m_wy_feat - m_feat * pi2_samp / wy_samp) + \
                eps_samp * (m_wx_samp - pi2_samp / wy_samp)
    grad_wy_feat = rho_y * (m_wy_samp - m_samp * pi2_feat / wy_feat) + \
                eps_feat * (m_wx_feat - pi2_feat / wy_feat)

    # set gradients
    ucoot = dict_log["ucoot_cost"]
    ucoot = nx.set_gradients(ucoot,
        (X, Y, M_samp, M_feat, wx_samp, wx_feat, wy_samp, wy_feat),
        (gradX, gradY, gradM_samp, gradM_feat, grad_wx_samp, grad_wx_feat, grad_wy_samp, grad_wy_feat)
        )

    if log:
        return ucoot, dict_log

    else:
        return ucoot


def fused_unbalanced_gromov_wasserstein(
        Cx, Cy, wx=None, wy=None, reg_marginals=None, epsilon=0,
        divergence="kl", unbalanced_solver="scaling",
        alpha=0, M=None, init_duals=None, init_pi=None, max_iter=100,
        tol=1e-7, max_iter_ot=500, tol_ot=1e-7, method_sinkhorn="sinkhorn",
        log=False, verbose=False, **kwargs_solve
        ):

    alpha_samp, alpha_feat = get_parameter_pair(alpha)
    alpha = (alpha_samp / 2, alpha_feat / 2)

    pi_samp, pi_feat, dict_log = fused_unbalanced_cross_spaces_divergence(
        X=Cx, Y=Cy, wx_samp=wx, wx_feat=wx, wy_samp=wy, wy_feat=wy,
        reg_marginals=reg_marginals, epsilon=epsilon, reg_type="joint",
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M_samp=M, M_feat=M, init_pi=(init_pi, init_pi),
        init_duals=(init_duals, init_duals), max_iter=max_iter, tol=tol,
        max_iter_ot=max_iter_ot, tol_ot=tol_ot,
        method_sinkhorn=method_sinkhorn, log=True,
        verbose=verbose, **kwargs_solve
    )

    if log:
        log_fugw = {"error": dict_log["error"],
                    "duals": dict_log["duals_sample"],
                    "linear_cost": dict_log["linear_cost"],
                    "fugw_cost": dict_log["ucoot_cost"]}

        return pi_samp, pi_feat, log_fugw

    else:
        return pi_samp, pi_feat


def fused_unbalanced_gromov_wasserstein2(
        Cx, Cy, wx=None, wy=None, reg_marginals=None, epsilon=0,
        divergence="kl", unbalanced_solver="scaling",
        alpha=0, M=None, init_duals=None, init_pi=None, max_iter=100,
        tol=1e-7, max_iter_ot=500, tol_ot=1e-7, method_sinkhorn="sinkhorn",
        log=False, verbose=False, **kwargs_solve
        ):

    pi_samp, pi_feat, log_fugw = fused_unbalanced_gromov_wasserstein(
        Cx=Cx, Cy=Cy, wx=wx, wy=wy, reg_marginals=reg_marginals, epsilon=epsilon,
        divergence=divergence, unbalanced_solver=unbalanced_solver,
        alpha=alpha, M=M, init_duals=init_duals, init_pi=init_pi,
        max_iter=max_iter, tol=tol, max_iter_ot=max_iter_ot,
        tol_ot=tol_ot, method_sinkhorn=method_sinkhorn,
        log=True, verbose=verbose, **kwargs_solve
    )

    Cx, Cy, pi_samp, pi_feat = list_to_array(Cx, Cy, pi_samp, pi_feat)
    nx = get_backend(Cx, Cy, pi_samp, pi_feat)

    sx, sy = Cx.shape[0], Cy.shape[0]

    # measures on rows and columns
    if wx is None:
        wx = nx.ones(sx, type_as=Cx) / sx
    if wy is None:
        wy = nx.ones(sy, type_as=Cy) / sy

    # calculate marginals
    pi1_samp, pi2_samp = nx.sum(pi_samp, 1), nx.sum(pi_samp, 0)
    pi1_feat, pi2_feat = nx.sum(pi_feat, 1), nx.sum(pi_feat, 0)
    m_samp, m_feat = nx.sum(pi1_samp), nx.sum(pi1_feat)
    m_wx, m_wy = nx.sum(wx), nx.sum(wy)

    # calculate subgradients
    gradX = 2 * Cx * (pi1_samp[:, None] * pi1_feat[None, :]) - \
        2 * pi_samp @ Cy @ pi_feat.T  # shape (nx_samp, nx_feat)
    gradY = 2 * Cy * (pi2_samp[:, None] * pi2_feat[None, :]) - \
        2 * pi_samp.T @ Cx @ pi_feat  # shape (ny_samp, ny_feat)

    gradM = alpha / 2 * (pi_samp + pi_feat)

    rho_x, rho_y = get_parameter_pair(reg_marginals)
    grad_wx = 2 * m_wx * (rho_x + epsilon * m_wy**2) - \
            (rho_x + epsilon) * (m_feat * pi1_samp + m_samp * pi1_feat) / wx
    grad_wy = 2 * m_wy * (rho_y + epsilon * m_wx**2) - \
            (rho_y + epsilon) * (m_feat * pi2_samp + m_samp * pi2_feat) / wy

    # set gradients
    fugw = log_fugw["fugw_cost"]
    fugw = nx.set_gradients(fugw, (Cx, Cy, M, wx, wy),
                                (gradX, gradY, gradM, grad_wx, grad_wy))

    if log:
        return fugw, log_fugw

    else:
        return fugw
