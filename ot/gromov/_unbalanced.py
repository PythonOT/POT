# -*- coding: utf-8 -*-
"""
Unbalanced Co-Optimal Transport and Fused Unbalanced Gromov-Wasserstein solvers
"""

# Author: Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#         Alexis Thual <alexis.thual@cea.fr>
#
# License: MIT License

import warnings
from functools import partial
import ot
from ot.backend import get_backend
from ot.utils import list_to_array, get_parameter_pair
from ._utils import (
    fused_unbalanced_across_spaces_cost,
    uot_cost_matrix,
    uot_parameters_and_measures,
)


def fused_unbalanced_across_spaces_divergence(
    X,
    Y,
    wx_samp=None,
    wx_feat=None,
    wy_samp=None,
    wy_feat=None,
    reg_marginals=10,
    epsilon=0,
    reg_type="joint",
    divergence="kl",
    unbalanced_solver="sinkhorn",
    alpha=0,
    M_samp=None,
    M_feat=None,
    rescale_plan=True,
    init_pi=None,
    init_duals=None,
    max_iter=100,
    tol=1e-7,
    max_iter_ot=500,
    tol_ot=1e-7,
    log=False,
    verbose=False,
    **kwargs_solver,
):
    r"""Compute the fused unbalanced cross-spaces divergence between two matrices equipped
    with the distributions on rows and columns. We consider two cases of matrix:

    - (Squared) similarity matrix in Gromov-Wasserstein setting,
    whose rows and columns represent the samples.

    - Arbitrary-size matrix in Co-Optimal Transport setting,
    whose rows represent samples, and columns represent corresponding features/dimensions.

    More precisely, this function returns the sample and feature transport plans between
    :math:`(\mathbf{X}, \mathbf{w}_{xs}, \mathbf{w}_{xf})` and
    :math:`(\mathbf{Y}, \mathbf{w}_{ys}, \mathbf{w}_{yf})`,
    by solving the following problem using Block Coordinate Descent algorithm:

    .. math::

        \mathop{\arg \min}_{\mathbf{P}, \mathbf{Q}}
        &\quad \sum_{i,j,k,l}
        (\mathbf{X}_{i,k} - \mathbf{Y}_{j,l})^2 \mathbf{P}_{i,j} \mathbf{Q}_{k,l} \\
        &+ \rho_s \mathbf{Div}(\mathbf{P}_{\# 1} \mathbf{Q}_{\# 1}^T | \mathbf{w}_{xs} \mathbf{w}_{ys}^T)
        + \rho_f \mathbf{Div}(\mathbf{P}_{\# 2} \mathbf{Q}_{\# 2}^T | \mathbf{w}_{xf} \mathbf{w}_{yf}^T) \\
        &+ \alpha_s \sum_{i,j} \mathbf{P}_{i,j} \mathbf{M^{(s)}}_{i, j}
        + \alpha_f \sum_{k, l} \mathbf{Q}_{k,l} \mathbf{M^{(f)}}_{k, l}
        + \mathbf{Reg}(\mathbf{P}, \mathbf{Q})

    Where:

    - :math:`\mathbf{X}`: Source input (arbitrary-size) matrix
    - :math:`\mathbf{Y}`: Target input (arbitrary-size) matrix
    - :math:`\mathbf{M^{(s)}}`: Additional sample matrix
    - :math:`\mathbf{M^{(f)}}`: Additional feature matrix
    - :math:`\mathbf{w}_{xs}`: Distribution of the samples in the source space
    - :math:`\mathbf{w}_{xf}`: Distribution of the features in the source space
    - :math:`\mathbf{w}_{ys}`: Distribution of the samples in the target space
    - :math:`\mathbf{w}_{yf}`: Distribution of the features in the target space
    - :math:`\mathbf{Div}`: Either Kullback-Leibler divergence or half-squared L2 norm.
    - :math:`\mathbf{Reg}`: Regularizer for sample and feature couplings.

    We consider two types of regularizer:
        + Independent regularization used in unbalanced Co-Optimal Transport

        .. math::
            \mathbf{Reg}(\mathbf{P}, \mathbf{Q}) =
            \varepsilon_s \mathbf{Div}(\mathbf{P} | \mathbf{w}_{xs} \mathbf{w}_{ys}^T)
            + \varepsilon_f \mathbf{Div}(\mathbf{Q} | \mathbf{w}_{xf} \mathbf{w}_{yf}^T)

        + Joint regularization used in fused unbalanced Gromov-Wasserstein

        .. math::
            \mathbf{Reg}(\mathbf{P}, \mathbf{Q}) =
            \varepsilon \mathbf{Div}(\mathbf{P} \otimes \mathbf{Q} | (\mathbf{w}_{xs} \mathbf{w}_{ys}^T) \otimes (\mathbf{w}_{xf} \mathbf{w}_{yf}^T) )

    .. note:: This function allows epsilon to be zero. In that case, `unbalanced_method` must be either "mm" or "lbfgsb".

    Parameters
    ----------
    X : (n_sample_x, n_feature_x) array-like, float
        Source input matrix.
    Y : (n_sample_y, n_feature_y) array-like, float
        Target input matrix.
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
    reg_marginals: float or indexable object of length 1 or 2
        Marginal relaxation terms for sample and feature couplings.
        If `reg_marginals` is a scalar or an indexable object of length 1,
        then the same value is applied to both marginal relaxations.
    epsilon : scalar or indexable object of length 2, float or int, optional (default = 0)
        Regularization parameters for entropic approximation of sample and feature couplings.
        Allow the case where `epsilon` contains 0. In that case, the MM solver is used by default
        instead of Sinkhorn solver. If `epsilon` is scalar, then the same value is applied to
        both regularization of sample and feature couplings.
    reg_type: string, optional

        - If `reg_type` = "joint": then use joint regularization for couplings.

        - If `reg_type` = "independent": then use independent regularization for couplings.
    divergence : string, optional (default = "kl")

        - If `divergence` = "kl", then Div is the Kullback-Leibler divergence.

        - If `divergence` = "l2", then Div is the half squared Euclidean norm.
    unbalanced_solver : string, optional (default = "sinkhorn")
        Solver for the unbalanced OT subroutine.

        - If `divergence` = "kl", then `unbalanced_solver` can be: "sinkhorn", "sinkhorn_log", "mm", "lbfgsb"

        - If `divergence` = "l2", then `unbalanced_solver` can be "mm", "lbfgsb"
    alpha : scalar or indexable object of length 2, float or int, optional (default = 0)
        Coeffficient parameter of linear terms with respect to the sample and feature couplings.
        If alpha is scalar, then the same alpha is applied to both linear terms.
    M_samp : (n_sample_x, n_sample_y), float, optional (default = None)
        Sample matrix associated to the Wasserstein linear term on sample coupling.
    M_feat : (n_feature_x, n_feature_y), float, optional (default = None)
        Feature matrix associated to the Wasserstein linear term on feature coupling.
    rescale_plan : boolean, optional (default = True)
        If True, then rescale the sample and feature transport plans within each BCD iteration,
        so that they always have equal mass.
    init_pi : tuple of two matrices of size (n_sample_x, n_sample_y) and
        (n_feature_x, n_feature_y), optional (default = None).
        Initialization of sample and feature couplings.
        Uniform distributions by default.
    init_duals : tuple of two tuples ((n_sample_x, ), (n_sample_y, )) and ((n_feature_x, ), (n_feature_y, )), optional (default = None).
        Initialization of sample and feature dual vectors
        if using Sinkhorn algorithm. Zero vectors by default.
    max_iter : int, optional (default = 100)
        Number of Block Coordinate Descent (BCD) iterations.
    tol : float, optional (default = 1e-7)
        Tolerance of BCD scheme. If the L1-norm between the current and previous
        sample couplings is under this threshold, then stop BCD scheme.
    max_iter_ot : int, optional (default = 100)
        Number of iterations to solve each of the
        two unbalanced optimal transport problems in each BCD iteration.
    tol_ot : float, optional (default = 1e-7)
        Tolerance of unbalanced solver for each of the
        two unbalanced optimal transport problems in each BCD iteration.
    log : bool, optional (default = False)
        If True then the cost and four dual vectors, including
        two from sample and two from feature couplings, are recorded.
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

            error : array-like, float
                list of L1 norms between the current and previous sample coupling.
            duals_sample : (n_sample_x, n_sample_y) tuple, float
                Pair of dual vectors when solving OT problem w.r.t the sample coupling.
            duals_feature : (n_feature_x, n_feature_y) tuple, float
                Pair of dual vectors when solving OT problem w.r.t the feature coupling.
            linear : float
                Linear part of the cost.
            ucoot : float
                Total cost.
            backend
                The proper backend for all input arrays
    """

    # MAIN FUNCTION

    if reg_type not in ["joint", "independent"]:
        raise (NotImplementedError('Unknown reg_type="{}"'.format(reg_type)))
    if divergence not in ["kl", "l2"]:
        raise (NotImplementedError('Unknown divergence="{}"'.format(divergence)))
    if unbalanced_solver not in ["sinkhorn", "sinkhorn_log", "mm", "lbfgsb"]:
        raise (NotImplementedError('Unknown method="{}"'.format(unbalanced_solver)))

    # hyperparameters
    alpha_samp, alpha_feat = get_parameter_pair(alpha)
    rho_x, rho_y = get_parameter_pair(reg_marginals)
    eps_samp, eps_feat = get_parameter_pair(epsilon)

    if reg_type == "joint":  # same regularization
        eps_feat = eps_samp
    if unbalanced_solver in ["sinkhorn", "sinkhorn_log"] and divergence == "l2":
        warnings.warn(
            "Sinkhorn algorithm does not support L2 norm. \
                      Divergence is set to 'kl'."
        )
        divergence = "kl"
    if unbalanced_solver in ["sinkhorn", "sinkhorn_log"] and (
        eps_samp == 0 or eps_feat == 0
    ):
        warnings.warn(
            "Sinkhorn algorithm does not support unregularized problem. \
                      Solver is set to 'mm'."
        )
        unbalanced_solver = "mm"

    if init_pi is None:
        pi_samp, pi_feat = None, None
    else:
        pi_samp, pi_feat = init_pi

    if init_duals is None:
        init_duals = (None, None)
    duals_samp, duals_feat = init_duals

    arr = [X, Y]

    for tuple in [duals_samp, duals_feat]:
        if tuple is not None:
            d1, d2 = duals_feat
            if d1 is not None:
                arr.append(list_to_array(d1))
            if d2 is not None:
                arr.append(list_to_array(d2))

    nx = get_backend(
        *arr, wx_samp, wx_feat, wy_samp, wy_feat, M_samp, M_feat, pi_samp, pi_feat
    )

    # constant input variables
    if M_samp is None:
        if alpha_samp > 0:
            warnings.warn(
                "M_samp is None but alpha_samp = {} > 0. \
                          The algo will treat as if alpha_samp = 0.".format(alpha_samp)
            )
    else:
        M_samp = alpha_samp * M_samp

    if M_feat is None:
        if alpha_feat > 0:
            warnings.warn(
                "M_feat is None but alpha_feat = {} > 0. \
                          The algo will treat as if alpha_feat = 0.".format(alpha_feat)
            )
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
    pi_samp = wxy_samp if pi_samp is None else pi_samp
    pi_feat = wxy_feat if pi_feat is None else pi_feat

    if unbalanced_solver in ["sinkhorn", "sinkhorn_log"]:
        if duals_samp is None:
            duals_samp = (nx.zeros(nx_samp, type_as=X), nx.zeros(ny_samp, type_as=Y))
        if duals_feat is None:
            duals_feat = (nx.zeros(nx_feat, type_as=X), nx.zeros(ny_feat, type_as=Y))

    # shortcut functions
    X_sqr, Y_sqr = X**2, Y**2
    local_cost_samp = partial(
        uot_cost_matrix,
        data=(X_sqr, Y_sqr, X, Y, M_samp),
        tuple_p=(wx_feat, wy_feat),
        hyperparams=(rho_x, rho_y, eps_feat),
        divergence=divergence,
        reg_type=reg_type,
        nx=nx,
    )

    local_cost_feat = partial(
        uot_cost_matrix,
        data=(X_sqr.T, Y_sqr.T, X.T, Y.T, M_feat),
        tuple_p=(wx_samp, wy_samp),
        hyperparams=(rho_x, rho_y, eps_samp),
        divergence=divergence,
        reg_type=reg_type,
        nx=nx,
    )

    parameters_uot_l2_samp = partial(
        uot_parameters_and_measures,
        tuple_weights=(wx_samp, wy_samp, wxy_samp),
        hyperparams=(rho_x, rho_y, eps_samp),
        reg_type=reg_type,
        divergence=divergence,
        nx=nx,
    )

    parameters_uot_l2_feat = partial(
        uot_parameters_and_measures,
        tuple_weights=(wx_feat, wy_feat, wxy_feat),
        hyperparams=(rho_x, rho_y, eps_feat),
        reg_type=reg_type,
        divergence=divergence,
        nx=nx,
    )

    solver = partial(
        ot.solve,
        reg_type=divergence,
        unbalanced_type=divergence,
        method=unbalanced_solver,
        max_iter=max_iter_ot,
        tol=tol_ot,
        verbose=False,
    )

    # initialize log
    if log:
        dict_log = {"backend": nx, "error": []}

    for idx in range(max_iter):
        pi_samp_prev = nx.copy(pi_samp)

        # Update feature coupling
        mass = nx.sum(pi_samp)
        uot_cost = local_cost_feat(pi=pi_samp)

        if divergence == "kl":
            new_rho = (rho_x * mass, rho_y * mass)
            new_eps = mass * eps_feat if reg_type == "joint" else eps_feat
            new_wx, new_wy, new_wxy = wx_feat, wy_feat, wxy_feat
        else:  # divergence == "l2"
            new_w, new_rho, new_eps = parameters_uot_l2_feat(pi_feat)
            new_wx, new_wy, new_wxy = new_w

        res = solver(
            M=uot_cost,
            a=new_wx,
            b=new_wy,
            reg=new_eps,
            c=new_wxy,
            unbalanced=new_rho,
            plan_init=pi_feat,
            potentials_init=duals_feat,
        )
        pi_feat, duals_feat = res.plan, res.potentials

        if rescale_plan:
            pi_feat = nx.sqrt(mass / nx.sum(pi_feat)) * pi_feat

        # Update sample coupling
        mass = nx.sum(pi_feat)
        uot_cost = local_cost_samp(pi=pi_feat)

        if divergence == "kl":
            new_rho = (rho_x * mass, rho_y * mass)
            new_eps = mass * eps_feat if reg_type == "joint" else eps_feat
            new_wx, new_wy, new_wxy = wx_samp, wy_samp, wxy_samp
        else:  # divergence == "l2"
            new_w, new_rho, new_eps = parameters_uot_l2_samp(pi_samp)
            new_wx, new_wy, new_wxy = new_w

        res = solver(
            M=uot_cost,
            a=new_wx,
            b=new_wy,
            reg=new_eps,
            c=new_wxy,
            unbalanced=new_rho,
            plan_init=pi_samp,
            potentials_init=duals_samp,
        )
        pi_samp, duals_samp = res.plan, res.potentials

        if rescale_plan:
            pi_samp = nx.sqrt(mass / nx.sum(pi_samp)) * pi_samp  # shape nx x ny

        # get L1 error
        err = nx.sum(nx.abs(pi_samp - pi_samp_prev))
        if log:
            dict_log["error"].append(err)
        if verbose:
            print("{:5d}|{:8e}|".format(idx + 1, err))
        if err < tol:
            break

    # sanity check
    if nx.sum(nx.isnan(pi_samp)) > 0 or nx.sum(nx.isnan(pi_feat)) > 0:
        raise (
            ValueError(
                "There is NaN in coupling. \
                          Adjust the relaxation or regularization parameters."
            )
        )

    if log:
        linear_cost, ucoot_cost = fused_unbalanced_across_spaces_cost(
            M_linear=(M_samp, M_feat),
            data=(X_sqr, Y_sqr, X, Y),
            tuple_pxy_samp=(wx_samp, wy_samp, wxy_samp),
            tuple_pxy_feat=(wx_feat, wy_feat, wxy_feat),
            pi_samp=pi_samp,
            pi_feat=pi_feat,
            hyperparams=(rho_x, rho_y, eps_samp, eps_feat),
            divergence=divergence,
            reg_type=reg_type,
            nx=nx,
        )

        dict_log["duals_sample"] = duals_samp
        dict_log["duals_feature"] = duals_feat
        dict_log["linear_cost"] = linear_cost
        dict_log["ucoot_cost"] = ucoot_cost

        return pi_samp, pi_feat, dict_log

    else:
        return pi_samp, pi_feat


def unbalanced_co_optimal_transport(
    X,
    Y,
    wx_samp=None,
    wx_feat=None,
    wy_samp=None,
    wy_feat=None,
    reg_marginals=10,
    epsilon=0,
    divergence="kl",
    unbalanced_solver="mm",
    alpha=0,
    M_samp=None,
    M_feat=None,
    rescale_plan=True,
    init_pi=None,
    init_duals=None,
    max_iter=100,
    tol=1e-7,
    max_iter_ot=500,
    tol_ot=1e-7,
    log=False,
    verbose=False,
    **kwargs_solve,
):
    r"""Compute the unbalanced Co-Optimal Transport between two Euclidean point clouds
    (represented as matrices whose rows are samples and columns are the features/dimensions).

    More precisely, this function returns the sample and feature transport plans between
    :math:`(\mathbf{X}, \mathbf{w}_{xs}, \mathbf{w}_{xf})` and
    :math:`(\mathbf{Y}, \mathbf{w}_{ys}, \mathbf{w}_{yf})`,
    by solving the following problem using Block Coordinate Descent algorithm:

    .. math::
        \mathop{\arg \min}_{\mathbf{P}, \mathbf{Q}} &\quad \sum_{i,j,k,l}
        (\mathbf{X}_{i,k} - \mathbf{Y}_{j,l})^2 \mathbf{P}_{i,j} \mathbf{Q}_{k,l} \\
        &+ \rho_s \mathbf{Div}(\mathbf{P}_{\# 1} \mathbf{Q}_{\# 1}^T | \mathbf{w}_{xs} \mathbf{w}_{ys}^T)
        + \rho_f \mathbf{Div}(\mathbf{P}_{\# 2} \mathbf{Q}_{\# 2}^T | \mathbf{w}_{xf} \mathbf{w}_{yf}^T) \\
        &+ \alpha_s \sum_{i,j} \mathbf{P}_{i,j} \mathbf{M^{(s)}}_{i, j}
        + \alpha_f \sum_{k, l} \mathbf{Q}_{k,l} \mathbf{M^{(f)}}_{k, l} \\
        &+ \varepsilon_s \mathbf{Div}(\mathbf{P} | \mathbf{w}_{xs} \mathbf{w}_{ys}^T)
        + \varepsilon_f \mathbf{Div}(\mathbf{Q} | \mathbf{w}_{xf} \mathbf{w}_{yf}^T)

    Where:

    - :math:`\mathbf{X}`: Source input (arbitrary-size) matrix
    - :math:`\mathbf{Y}`: Target input (arbitrary-size) matrix
    - :math:`\mathbf{M^{(s)}}`: Additional sample matrix
    - :math:`\mathbf{M^{(f)}}`: Additional feature matrix
    - :math:`\mathbf{w}_{xs}`: Distribution of the samples in the source space
    - :math:`\mathbf{w}_{xf}`: Distribution of the features in the source space
    - :math:`\mathbf{w}_{ys}`: Distribution of the samples in the target space
    - :math:`\mathbf{w}_{yf}`: Distribution of the features in the target space
    - :math:`\mathbf{Div}`: Either Kullback-Leibler divergence or half-squared L2 norm.

    .. note:: This function allows `epsilon` to be zero. In that case, `unbalanced_method` must be either "mm" or "lbfgsb".

    Parameters
    ----------
    X : (n_sample_x, n_feature_x) array-like, float
        Source input matrix.
    Y : (n_sample_y, n_feature_y) array-like, float
        Target input matrix.
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
    reg_marginals: float or indexable object of length 1 or 2
        Marginal relaxation terms for sample and feature couplings.
        If `reg_marginals is a scalar` or an indexable object of length 1,
        then the same value is applied to both marginal relaxations.
    epsilon : scalar or indexable object of length 2, float or int, optional (default = 0)
        Regularization parameters for entropic approximation of sample and feature couplings.
        Allow the case where `epsilon` contains 0. In that case, the MM solver is used by default
        instead of Sinkhorn solver. If `epsilon` is scalar, then the same value is applied to
        both regularization of sample and feature couplings.
    divergence : string, optional (default = "kl")

        - If `divergence` = "kl", then Div is the Kullback-Leibler divergence.

        - If `divergence` = "l2", then Div is the half squared Euclidean norm.
    unbalanced_solver : string, optional (default = "sinkhorn")
        Solver for the unbalanced OT subroutine.

        - If `divergence` = "kl", then `unbalanced_solver` can be: "sinkhorn", "sinkhorn_log", "mm", "lbfgsb"

        - If `divergence` = "l2", then `unbalanced_solver` can be "mm", "lbfgsb"
    alpha : scalar or indexable object of length 2, float or int, optional (default = 0)
        Coeffficient parameter of linear terms with respect to the sample and feature couplings.
        If alpha is scalar, then the same alpha is applied to both linear terms.
    M_samp : (n_sample_x, n_sample_y), float, optional (default = None)
        Sample matrix associated to the Wasserstein linear term on sample coupling.
    M_feat : (n_feature_x, n_feature_y), float, optional (default = None)
        Feature matrix associated to the Wasserstein linear term on feature coupling.
    rescale_plan : boolean, optional (default = True)
        If True, then rescale the sample and feature transport plans within each BCD iteration,
        so that they always have equal mass.
    init_pi : tuple of two matrices of size (n_sample_x, n_sample_y) and
        (n_feature_x, n_feature_y), optional (default = None).
        Initialization of sample and feature couplings.
        Uniform distributions by default.
    init_duals : tuple of two tuples ((n_sample_x, ), (n_sample_y, )) and ((n_feature_x, ), (n_feature_y, )), optional (default = None).
        Initialization of sample and feature dual vectors
        if using Sinkhorn algorithm. Zero vectors by default.
    max_iter : int, optional (default = 100)
        Number of Block Coordinate Descent (BCD) iterations.
    tol : float, optional (default = 1e-7)
        Tolerance of BCD scheme. If the L1-norm between the current and previous
        sample couplings is under this threshold, then stop BCD scheme.
    max_iter_ot : int, optional (default = 100)
        Number of iterations to solve each of the
        two unbalanced optimal transport problems in each BCD iteration.
    tol_ot : float, optional (default = 1e-7)
        Tolerance of unbalanced solver for each of the
        two unbalanced optimal transport problems in each BCD iteration.
    log : bool, optional (default = False)
        If True then the cost and four dual vectors, including
        two from sample and two from feature couplings, are recorded.
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

            error : array-like, float
                list of L1 norms between the current and previous sample coupling.
            duals_sample : (n_sample_x, n_sample_y)-tuple, float
                Pair of dual vectors when solving OT problem w.r.t the sample coupling.
            duals_feature : (n_feature_x, n_feature_y)-tuple, float
                Pair of dual vectors when solving OT problem w.r.t the feature coupling.
            linear : float
                Linear part of the cost.
            ucoot : float
                Total cost.

    References
    ----------
    .. [71] Tran, H., Janati, H., Courty, N., Flamary, R., Redko, I., Demetci, P., & Singh, R.
            Unbalanced Co-Optimal Transport. AAAI Conference on Artificial Intelligence, 2023.
    """

    return fused_unbalanced_across_spaces_divergence(
        X=X,
        Y=Y,
        wx_samp=wx_samp,
        wx_feat=wx_feat,
        wy_samp=wy_samp,
        wy_feat=wy_feat,
        reg_marginals=reg_marginals,
        epsilon=epsilon,
        reg_type="independent",
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=M_samp,
        M_feat=M_feat,
        rescale_plan=rescale_plan,
        init_pi=init_pi,
        init_duals=init_duals,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=log,
        verbose=verbose,
        **kwargs_solve,
    )


def unbalanced_co_optimal_transport2(
    X,
    Y,
    wx_samp=None,
    wx_feat=None,
    wy_samp=None,
    wy_feat=None,
    reg_marginals=10,
    epsilon=0,
    divergence="kl",
    unbalanced_solver="sinkhorn",
    alpha=0,
    M_samp=None,
    M_feat=None,
    rescale_plan=True,
    init_pi=None,
    init_duals=None,
    max_iter=100,
    tol=1e-7,
    max_iter_ot=500,
    tol_ot=1e-7,
    log=False,
    verbose=False,
    **kwargs_solve,
):
    r"""Compute the unbalanced Co-Optimal Transport between two Euclidean point clouds
    (represented as matrices whose rows are samples and columns are the features/dimensions).

    More precisely, this function returns the unbalanced Co-Optimal Transport cost between
    :math:`(\mathbf{X}, \mathbf{w}_{xs}, \mathbf{w}_{xf})` and
    :math:`(\mathbf{Y}, \mathbf{w}_{ys}, \mathbf{w}_{yf})`,
    by solving the following problem using Block Coordinate Descent algorithm:

    .. math::
        \mathop{\min}_{\mathbf{P}, \mathbf{Q}} &\quad \sum_{i,j,k,l}
        (\mathbf{X}_{i,k} - \mathbf{Y}_{j,l})^2 \mathbf{P}_{i,j} \mathbf{Q}_{k,l} \\
        &+ \rho_s \mathbf{Div}(\mathbf{P}_{\# 1} \mathbf{Q}_{\# 1}^T | \mathbf{w}_{xs} \mathbf{w}_{ys}^T)
        + \rho_f \mathbf{Div}(\mathbf{P}_{\# 2} \mathbf{Q}_{\# 2}^T | \mathbf{w}_{xf} \mathbf{w}_{yf}^T) \\
        &+ \alpha_s \sum_{i,j} \mathbf{P}_{i,j} \mathbf{M^{(s)}}_{i, j}
        + \alpha_f \sum_{k, l} \mathbf{Q}_{k,l} \mathbf{M^{(f)}}_{k, l} \\
        &+ \varepsilon_s \mathbf{Div}(\mathbf{P} | \mathbf{w}_{xs} \mathbf{w}_{ys}^T)
        + \varepsilon_f \mathbf{Div}(\mathbf{Q} | \mathbf{w}_{xf} \mathbf{w}_{yf}^T)

    Where:

    - :math:`\mathbf{X}`: Source input (arbitrary-size) matrix
    - :math:`\mathbf{Y}`: Target input (arbitrary-size) matrix
    - :math:`\mathbf{M^{(s)}}`: Additional sample matrix
    - :math:`\mathbf{M^{(f)}}`: Additional feature matrix
    - :math:`\mathbf{w}_{xs}`: Distribution of the samples in the source space
    - :math:`\mathbf{w}_{xf}`: Distribution of the features in the source space
    - :math:`\mathbf{w}_{ys}`: Distribution of the samples in the target space
    - :math:`\mathbf{w}_{yf}`: Distribution of the features in the target space
    - :math:`\mathbf{Div}`: Either Kullback-Leibler divergence or half-squared L2 norm.

    .. note:: This function allows `epsilon` to be zero. In that case, `unbalanced_method` must be either "mm" or "lbfgsb".
            Also the computation of gradients is only supported for KL divergence. The case of half squared-L2 norm uses those of KL divergence.

    Parameters
    ----------
    X : (n_sample_x, n_feature_x) array-like, float
        Source input matrix.
    Y : (n_sample_y, n_feature_y) array-like, float
        Target input matrix.
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
    reg_marginals: float or indexable object of length 1 or 2
        Marginal relaxation terms for sample and feature couplings.
        If `reg_marginals` is a scalar or an indexable object of length 1,
        then the same value is applied to both marginal relaxations.
    epsilon : scalar or indexable object of length 2, float or int, optional (default = 0)
        Regularization parameters for entropic approximation of sample and feature couplings.
        Allow the case where `epsilon` contains 0. In that case, the MM solver is used by default
        instead of Sinkhorn solver. If `epsilon` is scalar, then the same value is applied to
        both regularization of sample and feature couplings.
    divergence : string, optional (default = "kl")

        - If `divergence` = "kl", then Div is the Kullback-Leibler divergence.

        - If `divergence` = "l2", then Div is the half squared Euclidean norm.
    unbalanced_solver : string, optional (default = "sinkhorn")
        Solver for the unbalanced OT subroutine.

        - If `divergence` = "kl", then `unbalanced_solver` can be: "sinkhorn", "sinkhorn_log", "mm", "lbfgsb"

        - If `divergence` = "l2", then `unbalanced_solver` can be "mm", "lbfgsb"
    alpha : scalar or indexable object of length 2, float or int, optional (default = 0)
        Coeffficient parameter of linear terms with respect to the sample and feature couplings.
        If alpha is scalar, then the same alpha is applied to both linear terms.
    M_samp : (n_sample_x, n_sample_y), float, optional (default = None)
        Sample matrix associated to the Wasserstein linear term on sample coupling.
    M_feat : (n_feature_x, n_feature_y), float, optional (default = None)
        Feature matrix associated to the Wasserstein linear term on feature coupling.
    rescale_plan : boolean, optional (default = True)
        If True, then rescale the transport plans in each BCD iteration,
        so that they always have equal mass.
    init_pi : tuple of two matrices of size (n_sample_x, n_sample_y) and
        (n_feature_x, n_feature_y), optional (default = None).
        Initialization of sample and feature couplings.
        Uniform distributions by default.
    init_duals : tuple of two tuples ((n_sample_x, ), (n_sample_y, )) and ((n_feature_x, ), (n_feature_y, )), optional (default = None).
        Initialization of sample and feature dual vectors
        if using Sinkhorn algorithm. Zero vectors by default.
    max_iter : int, optional (default = 100)
        Number of Block Coordinate Descent (BCD) iterations.
    tol : float, optional (default = 1e-7)
        Tolerance of BCD scheme. If the L1-norm between the current and previous
        sample couplings is under this threshold, then stop BCD scheme.
    max_iter_ot : int, optional (default = 100)
        Number of iterations to solve each of the
        two unbalanced optimal transport problems in each BCD iteration.
    tol_ot : float, optional (default = 1e-7)
        Tolerance of unbalanced solver for each of the
        two unbalanced optimal transport problems in each BCD iteration.
    log : bool, optional (default = False)
        If True then the cost and four dual vectors, including
        two from sample and two from feature couplings, are recorded.
    verbose : bool, optional (default = False)
        If True then print the COOT cost at every multiplier of `eval_bcd`-th iteration.

    Returns
    -------
    ucoot : float
        UCOOT cost.
    log : dictionary, optional
        Returned if `log` is True. The keys are:

            error : array-like, float
                list of L1 norms between the current and previous sample coupling.
            duals_sample : (n_sample_x, n_sample_y)-tuple, float
                Pair of dual vectors when solving OT problem w.r.t the sample coupling.
            duals_feature : (n_feature_x, n_feature_y)-tuple, float
                Pair of dual vectors when solving OT problem w.r.t the feature coupling.
            linear : float
                Linear part of UCOOT cost.
            ucoot : float
                UCOOT cost.
            backend
                The proper backend for all input arrays

    References
    ----------
    .. [71] Tran, H., Janati, H., Courty, N., Flamary, R., Redko, I., Demetci, P., & Singh, R.
            Unbalanced Co-Optimal Transport. AAAI Conference on Artificial Intelligence, 2023.
    """

    if divergence != "kl":
        warnings.warn(
            "The computation of gradients is only supported for KL divergence, not \
                      for {} divergence".format(divergence)
        )

    pi_samp, pi_feat, log_ucoot = unbalanced_co_optimal_transport(
        X=X,
        Y=Y,
        wx_samp=wx_samp,
        wx_feat=wx_feat,
        wy_samp=wy_samp,
        wy_feat=wy_feat,
        reg_marginals=reg_marginals,
        epsilon=epsilon,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=M_samp,
        M_feat=M_feat,
        rescale_plan=rescale_plan,
        init_pi=init_pi,
        init_duals=init_duals,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=True,
        verbose=verbose,
        **kwargs_solve,
    )

    nx = log_ucoot["backend"]

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
    eps_samp, eps_feat = get_parameter_pair(epsilon)

    # calculate marginals
    pi1_samp, pi2_samp = nx.sum(pi_samp, 1), nx.sum(pi_samp, 0)
    pi1_feat, pi2_feat = nx.sum(pi_feat, 1), nx.sum(pi_feat, 0)
    m_samp, m_feat = nx.sum(pi1_samp), nx.sum(pi1_feat)
    m_wx_feat, m_wx_samp = nx.sum(wx_feat), nx.sum(wx_samp)
    m_wy_feat, m_wy_samp = nx.sum(wy_feat), nx.sum(wy_samp)

    # calculate subgradients
    gradX = 2 * X * (pi1_samp[:, None] * pi1_feat[None, :]) - 2 * nx.dot(
        nx.dot(pi_samp, Y), pi_feat.T
    )  # shape (nx_samp, nx_feat)
    gradY = 2 * Y * (pi2_samp[:, None] * pi2_feat[None, :]) - 2 * nx.dot(
        nx.dot(pi_samp.T, X), pi_feat
    )  # shape (ny_samp, ny_feat)

    grad_wx_samp = rho_x * (m_wx_feat - m_feat * pi1_samp / wx_samp) + eps_samp * (
        m_wy_samp - pi1_samp / wx_samp
    )
    grad_wx_feat = rho_x * (m_wx_samp - m_samp * pi1_feat / wx_feat) + eps_feat * (
        m_wy_feat - pi1_feat / wx_feat
    )
    grad_wy_samp = rho_y * (m_wy_feat - m_feat * pi2_samp / wy_samp) + eps_samp * (
        m_wx_samp - pi2_samp / wy_samp
    )
    grad_wy_feat = rho_y * (m_wy_samp - m_samp * pi2_feat / wy_feat) + eps_feat * (
        m_wx_feat - pi2_feat / wy_feat
    )

    # set gradients
    ucoot = log_ucoot["ucoot_cost"]
    ucoot = nx.set_gradients(
        ucoot,
        (X, Y, wx_samp, wx_feat, wy_samp, wy_feat),
        (gradX, gradY, grad_wx_samp, grad_wx_feat, grad_wy_samp, grad_wy_feat),
    )

    if log:
        return ucoot, log_ucoot

    else:
        return ucoot


def fused_unbalanced_gromov_wasserstein(
    Cx,
    Cy,
    wx=None,
    wy=None,
    reg_marginals=10,
    epsilon=0,
    divergence="kl",
    unbalanced_solver="mm",
    alpha=0,
    M=None,
    init_duals=None,
    init_pi=None,
    max_iter=100,
    tol=1e-7,
    max_iter_ot=500,
    tol_ot=1e-7,
    log=False,
    verbose=False,
    **kwargs_solve,
):
    r"""Compute the lower bound of the fused unbalanced Gromov-Wasserstein (FUGW) between two similarity matrices.
    In practice, this lower bound is used interchangeably with the true FUGW.

    More precisely, this function returns the transport plan between
    :math:`(\mathbf{C^X}, \mathbf{w_X})` and :math:`(\mathbf{C^Y}, \mathbf{w_Y})`,
    by solving the following problem using Block Coordinate Descent algorithm:

    .. math::
        \mathop{\arg \min}_{\substack{\mathbf{P}, \mathbf{Q}: \\ mass(P) = mass(Q)}}
        &\quad \sum_{i,j,k,l} (\mathbf{C^X}_{i,k} - \mathbf{C^Y}_{j,l})^2 \mathbf{P}_{i,j} \mathbf{Q}_{k,l}
        + \frac{\alpha}{2} \sum_{i,j} (\mathbf{P}_{i,j} + \mathbf{Q}_{i,j}) \mathbf{M}_{i, j} \\
        &+ \rho_1 \mathbf{Div}(\mathbf{P}_{\# 1} \mathbf{Q}_{\# 1}^T | \mathbf{w_X} \mathbf{w_X}^T)
        + \rho_2 \mathbf{Div}(\mathbf{P}_{\# 2} \mathbf{Q}_{\# 2}^T | \mathbf{w_Y} \mathbf{w_Y}^T) \\
        &+ \varepsilon \mathbf{Div}(\mathbf{P} \otimes \mathbf{Q} | (\mathbf{w_X} \mathbf{w_Y}^T) \otimes (\mathbf{w_X} \mathbf{w_Y}^T))

    Where:

    - :math:`\mathbf{C^X}`: Source similarity matrix
    - :math:`\mathbf{C^Y}`: Target similarity matrix
    - :math:`\mathbf{M}`: Sample matrix corresponding to the Wasserstein term
    - :math:`\mathbf{w_X}`: Distribution of the samples in the source space
    - :math:`\mathbf{w_Y}`: Distribution of the samples in the target space
    - :math:`\mathbf{Div}`: Either Kullback-Leibler divergence or half-squared L2 norm.

    .. note:: This function allows epsilon to be zero. In that case, `unbalanced_method` must be either "mm" or "lbfgsb".

    Parameters
    ----------
    Cx : (n_sample_x, n_feature_x) array-like, float
        Source similarity matrix.
    Cy : (n_sample_y, n_feature_y) array-like, float
        Target similarity matrix.
    wx : (n_sample_x, ) array-like, float, optional (default = None)
        Histogram assigned on rows (samples) of matrix Cx.
        Uniform distribution by default.
    wy : (n_sample_y, ) array-like, float, optional (default = None)
        Histogram assigned on rows (samples) of matrix Cy.
        Uniform distribution by default.
    reg_marginals: float or indexable object of length 1 or 2
        Marginal relaxation terms for sample and feature couplings.
        If `reg_marginals` is a scalar or an indexable object of length 1,
        then the same value is applied to both marginal relaxations.
    epsilon : scalar, float or int, optional (default = 0)
        Regularization parameters for entropic approximation of sample and feature couplings.
        Allow the case where `epsilon` contains 0. In that case, the MM solver is used by default
        instead of Sinkhorn solver. If `epsilon` is scalar, then the same value is applied to
        both regularization of sample and feature couplings.
    divergence : string, optional (default = "kl")

        - If `divergence` = "kl", then Div is the Kullback-Leibler divergence.

        - If `divergence` = "l2", then Div is the half squared Euclidean norm.
    unbalanced_solver : string, optional (default = "sinkhorn")
        Solver for the unbalanced OT subroutine.

        - If `divergence` = "kl", then `unbalanced_solver` can be: "sinkhorn", "sinkhorn_log", "mm", "lbfgsb"

        - If `divergence` = "l2", then `unbalanced_solver` can be "mm", "lbfgsb"
    alpha : scalar, float or int, optional (default = 0)
        Coeffficient parameter of linear terms with respect to the sample and feature couplings.
        If alpha is scalar, then the same alpha is applied to both linear terms.
    M : (n_sample_x, n_sample_y), float, optional (default = None)
        Sample matrix associated to the Wasserstein linear term on sample coupling.
    init_pi :(n_sample_x, n_sample_y) array-like, optional (default = None)
        Initialization of sample coupling. By default = :math:`w_X w_Y^T`.
    init_duals : tuple of vectors ((n_sample_x, ), (n_sample_y, )), optional (default = None).
        Initialization of sample and feature dual vectors
        if using Sinkhorn algorithm. Zero vectors by default.
    max_iter : int, optional (default = 100)
        Number of Block Coordinate Descent (BCD) iterations.
    tol : float, optional (default = 1e-7)
        Tolerance of BCD scheme. If the L1-norm between the current and previous
        sample couplings is under this threshold, then stop BCD scheme.
    max_iter_ot : int, optional (default = 100)
        Number of iterations to solve each of the
        two unbalanced optimal transport problems in each BCD iteration.
    tol_ot : float, optional (default = 1e-7)
        Tolerance of unbalanced solver for each of the
        two unbalanced optimal transport problems in each BCD iteration.
    log : bool, optional (default = False)
        If True then the cost and four dual vectors, including
        two from sample and two from feature couplings, are recorded.
    verbose : bool, optional (default = False)
        If True then print the COOT cost at every multiplier of `eval_bcd`-th iteration.

    Returns
    -------
    pi_samp : (n_sample_x, n_sample_y) array-like, float
        Sample coupling matrix.
        In practice, we use this matrix as solution of FUGW.
    pi_samp2 : (n_sample_x, n_sample_y) array-like, float
        Second sample coupling matrix.
        In practice, we usually ignore this output.
    log : dictionary, optional
        Returned if `log` is True. The keys are:

            error : array-like, float
                list of L1 norms between the current and previous sample couplings.
            duals : (n_sample_x, n_sample_y)-tuple, float
                Pair of dual vectors when solving OT problem w.r.t the sample coupling.
            linear : float
                Linear part of FUGW cost.
            fugw_cost : float
                Total FUGW cost.
            backend
                The proper backend for all input arrays

    References
    ----------
    .. [70] Thual, A., Tran, H., Zemskova, T., Courty, N., Flamary, R., Dehaene, S., & Thirion, B.
            Aligning individual brains with Fused Unbalanced Gromov-Wasserstein.
            Advances in Neural Information Systems, 35 (2022).

    .. [72] Thibault Séjourné, François-Xavier Vialard, & Gabriel Peyré.
            The Unbalanced Gromov Wasserstein Distance: Conic Formulation and Relaxation.
            Neural Information Processing Systems, 34 (2021).
    """

    alpha = (alpha / 2, alpha / 2)

    pi_samp, pi_feat, dict_log = fused_unbalanced_across_spaces_divergence(
        X=Cx,
        Y=Cy,
        wx_samp=wx,
        wx_feat=wx,
        wy_samp=wy,
        wy_feat=wy,
        reg_marginals=reg_marginals,
        epsilon=epsilon,
        reg_type="joint",
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M_samp=M,
        M_feat=M,
        rescale_plan=True,
        init_pi=(init_pi, init_pi),
        init_duals=(init_duals, init_duals),
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=True,
        verbose=verbose,
        **kwargs_solve,
    )

    if log:
        log_fugw = {
            "error": dict_log["error"],
            "duals": dict_log["duals_sample"],
            "linear_cost": dict_log["linear_cost"],
            "fugw_cost": dict_log["ucoot_cost"],
            "backend": dict_log["backend"],
        }

        return pi_samp, pi_feat, log_fugw

    else:
        return pi_samp, pi_feat


def fused_unbalanced_gromov_wasserstein2(
    Cx,
    Cy,
    wx=None,
    wy=None,
    reg_marginals=10,
    epsilon=0,
    divergence="kl",
    unbalanced_solver="mm",
    alpha=0,
    M=None,
    init_duals=None,
    init_pi=None,
    max_iter=100,
    tol=1e-7,
    max_iter_ot=500,
    tol_ot=1e-7,
    log=False,
    verbose=False,
    **kwargs_solve,
):
    r"""Compute the lower bound of the fused unbalanced Gromov-Wasserstein (FUGW) between two similarity matrices.
    In practice, this lower bound is used interchangeably with the true FUGW.

    More precisely, this function returns the lower bound of the fused unbalanced Gromov-Wasserstein cost between
    :math:`(\mathbf{C^X}, \mathbf{w_X})` and :math:`(\mathbf{C^Y}, \mathbf{w_Y})`,
    by solving the following problem using Block Coordinate Descent algorithm:

    .. math::
        \mathop{\min}_{\substack{\mathbf{P}, \mathbf{Q}: \\ mass(P) = mass(Q)}}
        &\quad \sum_{i,j,k,l} (\mathbf{C^X}_{i,k} - \mathbf{C^Y}_{j,l})^2 \mathbf{P}_{i,j} \mathbf{Q}_{k,l}
        + \frac{\alpha}{2} \sum_{i,j} (\mathbf{P}_{i,j} + \mathbf{Q}_{i,j}) \mathbf{M}_{i, j} \\
        &+ \rho_1 \mathbf{Div}(\mathbf{P}_{\# 1} \mathbf{Q}_{\# 1}^T | \mathbf{w_X} \mathbf{w_X}^T)
        + \rho_2 \mathbf{Div}(\mathbf{P}_{\# 2} \mathbf{Q}_{\# 2}^T | \mathbf{w_Y} \mathbf{w_Y}^T) \\
        &+ \varepsilon \mathbf{Div}(\mathbf{P} \otimes \mathbf{Q} | (\mathbf{w_X} \mathbf{w_Y}^T) \otimes (\mathbf{w_X} \mathbf{w_Y}^T))

    Where:

    - :math:`\mathbf{C^X}`: Source similarity matrix
    - :math:`\mathbf{C^Y}`: Target similarity matrix
    - :math:`\mathbf{M}`: Sample matrix corresponding to the Wasserstein term
    - :math:`\mathbf{w_X}`: Distribution of the samples in the source space
    - :math:`\mathbf{w_Y}`: Distribution of the samples in the target space
    - :math:`\mathbf{Div}`: Either Kullback-Leibler divergence or half-squared L2 norm.

    .. note:: This function allows `epsilon` to be zero. In that case, unbalanced_method must be either "mm" or "lbfgsb".
            Also the computation of gradients is only supported for KL divergence, but not for half squared-L2 norm. In case of half squared-L2 norm, the calculation of KL divergence will be used.

    Parameters
    ----------
    Cx : (n_sample_x, n_feature_x) array-like, float
        Source similarity matrix.
    Cy : (n_sample_y, n_feature_y) array-like, float
        Target similarity matrix.
    wx : (n_sample_x, ) array-like, float, optional (default = None)
        Histogram assigned on rows (samples) of matrix Cx.
        Uniform distribution by default.
    wy : (n_sample_y, ) array-like, float, optional (default = None)
        Histogram assigned on rows (samples) of matrix Cy.
        Uniform distribution by default.
    reg_marginals: float or indexable object of length 1 or 2
        Marginal relaxation terms for sample and feature couplings.
        If `reg_marginals` is a scalar or an indexable object of length 1,
        then the same value is applied to both marginal relaxations.
    epsilon : scalar, float or int, optional (default = 0)
        Regularization parameters for entropic approximation of sample and feature couplings.
        Allow the case where `epsilon` contains 0. In that case, the MM solver is used by default
        instead of Sinkhorn solver. If `epsilon` is scalar, then the same value is applied to
        both regularization of sample and feature couplings.
    divergence : string, optional (default = "kl")

        - If `divergence` = "kl", then Div is the Kullback-Leibler divergence.

        - If `divergence` = "l2", then Div is the half squared Euclidean norm.
    unbalanced_solver : string, optional (default = "sinkhorn")
        Solver for the unbalanced OT subroutine.

        - If `divergence` = "kl", then `unbalanced_solver` can be: "sinkhorn", "sinkhorn_log", "mm", "lbfgsb"

        - If `divergence` = "l2", then `unbalanced_solver` can be "mm", "lbfgsb"
    alpha : scalar, float or int, optional (default = 0)
        Coeffficient parameter of linear terms with respect to the sample and feature couplings.
        If alpha is scalar, then the same alpha is applied to both linear terms.
    M : (n_sample_x, n_sample_y), float, optional (default = None)
        Sample matrix associated to the Wasserstein linear term on sample coupling.
    init_pi :(n_sample_x, n_sample_y) array-like, optional (default = None)
        Initialization of sample coupling. By default = :math:`w_X w_Y^T`.
    init_duals : tuple of vectors ((n_sample_x, ), (n_sample_y, )), optional (default = None).
        Initialization of sample and feature dual vectors
        if using Sinkhorn algorithm. Zero vectors by default.
    max_iter : int, optional (default = 100)
        Number of Block Coordinate Descent (BCD) iterations.
    tol : float, optional (default = 1e-7)
        Tolerance of BCD scheme. If the L1-norm between the current and previous
        sample couplings is under this threshold, then stop BCD scheme.
    max_iter_ot : int, optional (default = 100)
        Number of iterations to solve each of the
        two unbalanced optimal transport problems in each BCD iteration.
    tol_ot : float, optional (default = 1e-7)
        Tolerance of unbalanced solver for each of the
        two unbalanced optimal transport problems in each BCD iteration.
    log : bool, optional (default = False)
        If True then the cost and four dual vectors, including
        two from sample and two from feature couplings, are recorded.
    verbose : bool, optional (default = False)
        If True then print the COOT cost at every multiplier of `eval_bcd`-th iteration.

    Returns
    -------
    fugw : float
        Total FUGW cost
    log : dictionary, optional
        Returned if `log` is True. The keys are:

            error : array-like, float
                list of L1 norms between the current and previous sample couplings.
            duals : (n_sample_x, n_sample_y)-tuple, float
                Pair of dual vectors when solving OT problem w.r.t the sample coupling.
            linear : float
                Linear part of FUGW cost.
            fugw_cost : float
                Total FUGW cost.
            backend
                The proper backend for all input arrays

    References
    ----------
    .. [70] Thual, A., Tran, H., Zemskova, T., Courty, N., Flamary, R., Dehaene, S., & Thirion, B.
            Aligning individual brains with Fused Unbalanced Gromov-Wasserstein.
            Advances in Neural Information Systems, 35 (2022).

    .. [72] Thibault Séjourné, François-Xavier Vialard, & Gabriel Peyré.
            The Unbalanced Gromov Wasserstein Distance: Conic Formulation and Relaxation.
            Neural Information Processing Systems, 34 (2021).
    """

    if divergence != "kl":
        warnings.warn(
            "The computation of gradients is only supported for KL divergence, \
                      but not for {} divergence. The gradient of the KL case will be used.".format(
                divergence
            )
        )

    pi_samp, pi_feat, log_fugw = fused_unbalanced_gromov_wasserstein(
        Cx=Cx,
        Cy=Cy,
        wx=wx,
        wy=wy,
        reg_marginals=reg_marginals,
        epsilon=epsilon,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha,
        M=M,
        init_duals=init_duals,
        init_pi=init_pi,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=True,
        verbose=verbose,
        **kwargs_solve,
    )

    nx = log_fugw["backend"]
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
    gradX = 2 * Cx * (pi1_samp[:, None] * pi1_feat[None, :]) - 2 * nx.dot(
        nx.dot(pi_samp, Cy), pi_feat.T
    )  # shape (nx_samp, nx_feat)
    gradY = 2 * Cy * (pi2_samp[:, None] * pi2_feat[None, :]) - 2 * nx.dot(
        nx.dot(pi_samp.T, Cx), pi_feat
    )  # shape (ny_samp, ny_feat)

    gradM = alpha / 2 * (pi_samp + pi_feat)

    rho_x, rho_y = get_parameter_pair(reg_marginals)
    grad_wx = (
        2 * m_wx * (rho_x + epsilon * m_wy**2)
        - (rho_x + epsilon) * (m_feat * pi1_samp + m_samp * pi1_feat) / wx
    )
    grad_wy = (
        2 * m_wy * (rho_y + epsilon * m_wx**2)
        - (rho_y + epsilon) * (m_feat * pi2_samp + m_samp * pi2_feat) / wy
    )

    # set gradients
    fugw = log_fugw["fugw_cost"]
    fugw = nx.set_gradients(
        fugw, (Cx, Cy, M, wx, wy), (gradX, gradY, gradM, grad_wx, grad_wy)
    )

    if log:
        return fugw, log_fugw

    else:
        return fugw
