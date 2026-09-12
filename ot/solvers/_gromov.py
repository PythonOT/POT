# -*- coding: utf-8 -*-
"""
General OT solvers with unified API
"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

from ..utils import OTResult
from ..lp import emd2
from ..backend import get_backend
from ..bregman import (
    sinkhorn_log,
)
from ..gromov import (
    gromov_wasserstein2,
    fused_gromov_wasserstein2,
    entropic_gromov_wasserstein2,
    entropic_fused_gromov_wasserstein2,
    semirelaxed_gromov_wasserstein2,
    semirelaxed_fused_gromov_wasserstein2,
    entropic_semirelaxed_fused_gromov_wasserstein2,
    entropic_semirelaxed_gromov_wasserstein2,
    partial_gromov_wasserstein2,
    partial_fused_gromov_wasserstein2,
    entropic_partial_gromov_wasserstein2,
    entropic_partial_fused_gromov_wasserstein2,
    fused_unbalanced_gromov_wasserstein,
)

from ._linear import solve

import warnings


def solve_gromov(
    Ca,
    Cb,
    M=None,
    a=None,
    b=None,
    loss="L2",
    symmetric=None,
    alpha=0.5,
    reg=None,
    reg_type="entropy",
    unbalanced=None,
    unbalanced_type="KL",
    n_threads=1,
    method=None,
    max_iter=None,
    plan_init=None,
    tol=None,
    verbose=False,
):
    r"""Solve the discrete (Fused) Gromov-Wasserstein and return :any:`OTResult` object

    The function solves the following optimization problem:

    .. math::
        \min_{\mathbf{T}\geq 0} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} + \lambda_r R(\mathbf{T}) + \lambda_u U(\mathbf{T}\mathbf{1},\mathbf{a}) + \lambda_u U(\mathbf{T}^T\mathbf{1},\mathbf{b})

    The regularization is selected with `reg` (:math:`\lambda_r`) and
    `reg_type`. By default ``reg=None`` and there is no regularization. The
    unbalanced marginal penalization can be selected with `unbalanced`
    (:math:`\lambda_u`) and `unbalanced_type`. By default ``unbalanced=None``
    and the function solves the exact optimal transport problem (respecting the
    marginals).

    Parameters
    ----------
    Ca : array-like, shape (dim_a, dim_a)
        Cost matrix in the source domain
    Cb : array-like, shape (dim_b, dim_b)
        Cost matrix in the target domain
    M : array-like, shape (dim_a, dim_b), optional
        Linear cost matrix for Fused Gromov-Wasserstein (default is None).
    a : array-like, shape (dim_a,), optional
        Samples weights in the source domain (default is uniform)
    b : array-like, shape (dim_b,), optional
        Samples weights in the source domain (default is uniform)
    loss : str, optional
        Type of loss function, either ``"L2"`` or ``"KL"``, by default ``"L2"``
    symmetric : bool, optional
        Use symmetric version of the Gromov-Wasserstein problem, by default None
        tests whether the matrices are symmetric or True/False to avoid the test.
    reg : float, optional
        Regularization weight :math:`\lambda_r`, by default None (no reg., exact
        OT)
    reg_type : str, optional
        Type of regularization :math:`R`, by default "entropy" (only used when
        ``reg!=None``)
    alpha : float, optional
        Weight the quadratic term (alpha*Gromov) and the linear term
        ((1-alpha)*Wass) in the Fused Gromov-Wasserstein problem. Not used for
        Gromov problem (when M is not provided). By default ``alpha=None``
        corresponds to ``alpha=1`` for Gromov problem (``M==None``) and
        ``alpha=0.5`` for Fused Gromov-Wasserstein problem (``M!=None``)
    unbalanced : float, optional
        Unbalanced penalization weight :math:`\lambda_u`, by default None
        (balanced OT). Not implemented yet for "KL" unbalanced penalization
        function :math:`U`. Corresponds to the total transport mass for partial OT.
    unbalanced_type : str, optional
        Type of unbalanced penalization function :math:`U` either "KL", "semirelaxed",
        "partial", by default "KL".
    n_threads : int, optional
        Number of OMP threads for exact OT solver, by default 1
    method : str, optional
        Method for solving the problem when multiple algorithms are available,
        default None for automatic selection.
    max_iter : int, optional
        Maximum number of iterations, by default None (default values in each
        solvers)
    plan_init : array-like, shape (dim_a, dim_b), optional
        Initialization of the OT plan for iterative methods, by default None
    tol : float, optional
        Tolerance for solution precision, by default None (default values in
        each solvers)
    verbose : bool, optional
        Print information in the solver, by default False

    Returns
    -------
    res : OTResult()
        Result of the optimization problem. The information can be obtained as follows:

        - res.plan : OT plan :math:`\mathbf{T}`
        - res.potentials : OT dual potentials
        - res.value : Optimal value of the optimization problem
        - res.value_linear : Linear OT loss with the optimal OT plan
        - res.value_quad : Quadratic (GW) part of the OT loss with the optimal OT plan

        See :any:`OTResult` for more information.

    Notes
    -----
    The following methods are available for solving the Gromov-Wasserstein
    problem:

    - **Classical Gromov-Wasserstein (GW) problem [3]** (default parameters):

    .. math::
        \min_{\mathbf{T}\geq 0} \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j}\mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve_gromov(Ca, Cb) # uniform weights
        res = ot.solve_gromov(Ca, Cb, a=a, b=b) # given weights
        res = ot.solve_gromov(Ca, Cb, loss='KL') # KL loss

        plan = res.plan # GW plan
        value = res.value # GW value

    - **Fused Gromov-Wasserstein (FGW) problem [24]** (when ``M!=None``):

    .. math::
        \min_{\mathbf{T}\geq 0} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j}\mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve_gromov(Ca, Cb, M) # uniform weights, alpha=0.5 (default)
        res = ot.solve_gromov(Ca, Cb, M, a=a, b=b, alpha=0.1) # given weights and alpha

        plan = res.plan # FGW plan
        loss_linear_term = res.value_linear # Wasserstein part of the loss
        loss_quad_term = res.value_quad # Gromov part of the loss
        loss = res.value # FGW value

    - **Regularized (Fused) Gromov-Wasserstein (GW) problem [12]** (when  ``reg!=None``):

    .. math::
        \min_{\mathbf{T}\geq 0} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j}\mathbf{T}_{k,l} + \lambda_r R(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve_gromov(Ca, Cb, reg=1.0) # GW entropy regularization (default)
        res = ot.solve_gromov(Ca, Cb, M, a=a, b=b, reg=10, alpha=0.1) # FGW with entropy

        plan = res.plan # FGW plan
        loss_linear_term = res.value_linear # Wasserstein part of the loss
        loss_quad_term = res.value_quad # Gromov part of the loss
        loss = res.value # FGW value (including regularization)

    - **Semi-relaxed (Fused) Gromov-Wasserstein (GW) [48]** (when  ``unbalanced='semirelaxed'``):

    .. math::
        \min_{\mathbf{T}\geq 0} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j}\mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T} \geq 0

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve_gromov(Ca, Cb, unbalanced='semirelaxed') # semirelaxed GW
        res = ot.solve_gromov(Ca, Cb, unbalanced='semirelaxed', reg=1) # entropic semirelaxed GW
        res = ot.solve_gromov(Ca, Cb, M, unbalanced='semirelaxed', alpha=0.1) # semirelaxed FGW

        plan = res.plan # FGW plan
        right_marginal = res.marginal_b # right marginal of the plan

    - **Partial (Fused) Gromov-Wasserstein (GW) problem [29]** (when  ``unbalanced='partial'``):

    .. math::
        \min_{\mathbf{T}\geq 0} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j}\mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} \leq \mathbf{a}

                \mathbf{T}^T \mathbf{1} \leq \mathbf{b}

                \mathbf{T} \geq 0

                \mathbf{1}^T\mathbf{T}\mathbf{1} = m

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve_gromov(Ca, Cb, unbalanced_type='partial', unbalanced=0.8) # partial GW with m=0.8
        res = ot.solve_gromov(Ca, Cb, M, unbalanced_type='partial', unbalanced=0.8, alpha=0.5) # partial FGW with m=0.8


    .. _references-solve-gromov:
    References
    ----------

    .. [3] Mémoli, F. (2011). Gromov–Wasserstein distances and the metric
        approach to object matching. Foundations of computational mathematics,
        11(4), 417-487.

    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon (2016),
        Gromov-Wasserstein averaging of kernel and distance matrices
        International Conference on Machine Learning (ICML).

    .. [24] Vayer, T., Chapel, L., Flamary, R., Tavenard, R. and Courty, N.
        (2019). Optimal Transport for structured data with application on graphs
        Proceedings of the 36th International Conference on Machine Learning
        (ICML).

    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer,
        Nicolas Courty (2022). Semi-relaxed Gromov-Wasserstein divergence and
        applications on graphs. International Conference on Learning
        Representations (ICLR), 2022.

    .. [29] Chapel, L., Alaya, M., Gasso, G. (2020). Partial Optimal Transport
        with Applications on Positive-Unlabeled Learning, Advances in Neural
        Information Processing Systems (NeurIPS), 2020.

    """

    # detect backend
    nx = get_backend(Ca, Cb, M, a, b)

    # create uniform weights if not given
    if a is None:
        a = nx.ones(Ca.shape[0], type_as=Ca) / Ca.shape[0]
    if b is None:
        b = nx.ones(Cb.shape[1], type_as=Cb) / Cb.shape[1]

    # default values for solutions
    potentials = None
    value = None
    value_linear = None
    value_quad = None
    plan = None
    status = None
    log = None

    loss_dict = {"l2": "square_loss", "kl": "kl_loss"}

    if loss.lower() not in loss_dict.keys():
        raise (NotImplementedError('Not implemented GW loss="{}"'.format(loss)))
    loss_fun = loss_dict[loss.lower()]

    if reg is None or reg == 0:  # exact OT
        if unbalanced is None and unbalanced_type.lower() not in [
            "semirelaxed",
        ]:  # Exact balanced OT
            if unbalanced_type.lower() in ["partial"]:
                warnings.warn(
                    "Exact balanced OT is computed as `unbalanced=None` even though "
                    f"unbalanced_type = {unbalanced_type}.",
                    stacklevel=2,
                )

            if M is None or alpha == 1:  # Gromov-Wasserstein problem
                # default values for solver
                if max_iter is None:
                    max_iter = 10000
                if tol is None:
                    tol = 1e-9

                value, log = gromov_wasserstein2(
                    Ca,
                    Cb,
                    a,
                    b,
                    loss_fun=loss_fun,
                    log=True,
                    symmetric=symmetric,
                    max_iter=max_iter,
                    G0=plan_init,
                    tol_rel=tol,
                    tol_abs=tol,
                    verbose=verbose,
                )

                value_quad = value
                if alpha == 1:  # set to 0 for FGW with alpha=1
                    value_linear = 0
                plan = log["T"]
                potentials = (log["u"], log["v"])

            elif alpha == 0:  # Wasserstein problem
                # default values for EMD solver
                if max_iter is None:
                    max_iter = 1000000

                value_linear, log = emd2(
                    a,
                    b,
                    M,
                    numItermax=max_iter,
                    log=True,
                    return_matrix=True,
                    numThreads=n_threads,
                )

                value = value_linear
                potentials = (log["u"], log["v"])
                plan = log["G"]
                status = log["warning"] if log["warning"] is not None else "Converged"
                value_quad = 0

            else:  # Fused Gromov-Wasserstein problem
                # default values for solver
                if max_iter is None:
                    max_iter = 10000
                if tol is None:
                    tol = 1e-9

                value, log = fused_gromov_wasserstein2(
                    M,
                    Ca,
                    Cb,
                    a,
                    b,
                    loss_fun=loss_fun,
                    alpha=alpha,
                    log=True,
                    symmetric=symmetric,
                    max_iter=max_iter,
                    G0=plan_init,
                    tol_rel=tol,
                    tol_abs=tol,
                    verbose=verbose,
                )

                value_linear = log["lin_loss"]
                value_quad = log["quad_loss"]
                plan = log["T"]
                potentials = (log["u"], log["v"])

        elif unbalanced_type.lower() in ["semirelaxed"]:  # Semi-relaxed  OT
            if M is None or alpha == 1:  # Semi relaxed Gromov-Wasserstein problem
                # default values for solver
                if max_iter is None:
                    max_iter = 10000
                if tol is None:
                    tol = 1e-9

                value, log = semirelaxed_gromov_wasserstein2(
                    Ca,
                    Cb,
                    a,
                    loss_fun=loss_fun,
                    log=True,
                    symmetric=symmetric,
                    max_iter=max_iter,
                    G0=plan_init,
                    tol_rel=tol,
                    tol_abs=tol,
                    verbose=verbose,
                )

                value_quad = value
                if alpha == 1:  # set to 0 for FGW with alpha=1
                    value_linear = 0
                plan = log["T"]
                # potentials = (log['u'], log['v']) TODO

            else:  # Semi relaxed Fused Gromov-Wasserstein problem
                # default values for solver
                if max_iter is None:
                    max_iter = 10000
                if tol is None:
                    tol = 1e-9

                value, log = semirelaxed_fused_gromov_wasserstein2(
                    M,
                    Ca,
                    Cb,
                    a,
                    loss_fun=loss_fun,
                    alpha=alpha,
                    log=True,
                    symmetric=symmetric,
                    max_iter=max_iter,
                    G0=plan_init,
                    tol_rel=tol,
                    tol_abs=tol,
                    verbose=verbose,
                )

                value_linear = log["lin_loss"]
                value_quad = log["quad_loss"]
                plan = log["T"]
                # potentials = (log['u'], log['v']) TODO

        elif unbalanced_type.lower() in ["partial"]:  # Partial OT
            if M is None or alpha == 1.0:  # Partial Gromov-Wasserstein problem
                if unbalanced > nx.sum(a) or unbalanced > nx.sum(b):
                    raise (
                        ValueError("Partial GW mass given in `unbalanced` is too large")
                    )

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-7

                value, log = partial_gromov_wasserstein2(
                    Ca,
                    Cb,
                    a,
                    b,
                    m=unbalanced,
                    loss_fun=loss_fun,
                    log=True,
                    numItermax=max_iter,
                    G0=plan_init,
                    tol=tol,
                    symmetric=symmetric,
                    verbose=verbose,
                )

                value_quad = value
                plan = log["T"]
                # potentials = (log['u'], log['v']) TODO

            else:  # partial FGW
                if unbalanced > nx.sum(a) or unbalanced > nx.sum(b):
                    raise (
                        ValueError("Partial GW mass given in `unbalanced` is too large")
                    )
                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-7

                value, log = partial_fused_gromov_wasserstein2(
                    M,
                    Ca,
                    Cb,
                    a,
                    b,
                    m=unbalanced,
                    loss_fun=loss_fun,
                    alpha=alpha,
                    log=True,
                    numItermax=max_iter,
                    G0=plan_init,
                    tol=tol,
                    symmetric=symmetric,
                    verbose=verbose,
                )

                value_linear = log["lin_loss"]
                value_quad = log["quad_loss"]
                plan = log["T"]
                # potentials = (log['u'], log['v']) TODO

        elif unbalanced_type.lower() in ["kl", "l2"]:  # unbalanced exact OT
            if alpha == 0:  # unbalanced Wasserstein problem
                res = solve(
                    M,
                    a=a,
                    b=b,
                    reg=None,
                    reg_type=reg_type,
                    unbalanced=unbalanced,
                    unbalanced_type=unbalanced_type,
                    method=method,
                    max_iter=max_iter,
                    plan_init=plan_init,
                    tol=tol,
                    verbose=verbose,
                )

                plan = res.plan
                potentials = res.potentials
                value_linear = res.value_linear
                value = res.value
                value_quad = 0
                status = res.status

            else:
                if max_iter is None:
                    max_iter = 100
                if tol is None:
                    tol = 1e-7

                # in this function alpha weights the linear and quadratic terms : alpha * quadratic + (1 - alpha) * linear
                # while fused_unbalanced_gromov_wasserstein uses alpha as the coefficient of the linear term.
                alpha_fugw = (1 - alpha) / alpha
                reg_fugw = unbalanced / (2 * alpha)
                plan, _, log = fused_unbalanced_gromov_wasserstein(
                    Ca,
                    Cb,
                    a,
                    b,
                    reg_marginals=reg_fugw,
                    divergence=unbalanced_type.lower(),
                    alpha=alpha_fugw,
                    M=M,
                    max_iter=max_iter,
                    tol=tol,
                    log=True,
                    epsilon=0,
                )
                value_linear = log["linear_cost"] * alpha
                value = log["fugw_cost"] * alpha

        else:
            raise (
                NotImplementedError(
                    'Unknown unbalanced_type="{}"'.format(unbalanced_type)
                )
            )

    else:  # regularized OT
        if unbalanced is None and unbalanced_type.lower() not in [
            "semirelaxed",
        ]:  # Balanced regularized OT
            if unbalanced_type.lower() in ["partial"]:
                warnings.warn(
                    "Exact balanced OT is computed as `unbalanced=None` even though "
                    f"unbalanced_type = {unbalanced_type}.",
                    stacklevel=2,
                )

            if reg_type.lower() in ["entropy"] and (
                M is None or alpha == 1
            ):  # Entropic Gromov-Wasserstein problem
                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9
                if method is None:
                    method = "PGD"

                value_quad, log = entropic_gromov_wasserstein2(
                    Ca,
                    Cb,
                    a,
                    b,
                    epsilon=reg,
                    loss_fun=loss_fun,
                    log=True,
                    symmetric=symmetric,
                    solver=method,
                    max_iter=max_iter,
                    G0=plan_init,
                    tol_rel=tol,
                    tol_abs=tol,
                    verbose=verbose,
                )

                plan = log["T"]
                value_linear = 0
                value = value_quad + reg * nx.sum(plan * nx.log(plan + 1e-16))
                # potentials = (log['log_u'], log['log_v'])  #TODO

            elif (
                reg_type.lower() in ["entropy"] and M is not None and alpha == 0
            ):  # Entropic Wasserstein problem
                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                plan, log = sinkhorn_log(
                    a,
                    b,
                    M,
                    reg=reg,
                    numItermax=max_iter,
                    stopThr=tol,
                    log=True,
                    verbose=verbose,
                )

                value_linear = nx.sum(M * plan)
                value = value_linear + reg * nx.sum(plan * nx.log(plan + 1e-16))
                potentials = (log["log_u"], log["log_v"])

            elif (
                reg_type.lower() in ["entropy"] and M is not None
            ):  # Entropic Fused Gromov-Wasserstein problem
                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9
                if method is None:
                    method = "PGD"

                value_noreg, log = entropic_fused_gromov_wasserstein2(
                    M,
                    Ca,
                    Cb,
                    a,
                    b,
                    loss_fun=loss_fun,
                    alpha=alpha,
                    log=True,
                    symmetric=symmetric,
                    solver=method,
                    max_iter=max_iter,
                    G0=plan_init,
                    tol_rel=tol,
                    tol_abs=tol,
                    verbose=verbose,
                )

                value_linear = log["lin_loss"]
                value_quad = log["quad_loss"]
                plan = log["T"]
                # potentials = (log['u'], log['v'])
                value = value_noreg + reg * nx.sum(plan * nx.log(plan + 1e-16))

            else:
                raise (
                    NotImplementedError(
                        'Not implemented reg_type="{}"'.format(reg_type)
                    )
                )

        elif unbalanced_type.lower() in ["semirelaxed"]:  # Semi-relaxed  OT
            if reg_type.lower() in ["entropy"] and (
                M is None or alpha == 1
            ):  # Entropic Semi-relaxed Gromov-Wasserstein problem
                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                value_quad, log = entropic_semirelaxed_gromov_wasserstein2(
                    Ca,
                    Cb,
                    a,
                    epsilon=reg,
                    loss_fun=loss_fun,
                    log=True,
                    symmetric=symmetric,
                    max_iter=max_iter,
                    G0=plan_init,
                    tol=tol,
                    verbose=verbose,
                )

                plan = log["T"]
                value_linear = 0
                value = value_quad + reg * nx.sum(plan * nx.log(plan + 1e-16))

            else:  # Entropic Semi-relaxed FGW problem
                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                value_noreg, log = entropic_semirelaxed_fused_gromov_wasserstein2(
                    M,
                    Ca,
                    Cb,
                    a,
                    loss_fun=loss_fun,
                    alpha=alpha,
                    log=True,
                    symmetric=symmetric,
                    max_iter=max_iter,
                    G0=plan_init,
                    tol=tol,
                    verbose=verbose,
                )

                value_linear = log["lin_loss"]
                value_quad = log["quad_loss"]
                plan = log["T"]
                value = value_noreg + reg * nx.sum(plan * nx.log(plan + 1e-16))

        elif unbalanced_type.lower() in ["partial"]:  # Partial OT
            if M is None or alpha == 1.0:  # Partial Gromov-Wasserstein problem
                if unbalanced > nx.sum(a) or unbalanced > nx.sum(b):
                    raise (
                        ValueError("Partial GW mass given in `unbalanced` is too large")
                    )

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-7

                value_noreg, log = entropic_partial_gromov_wasserstein2(
                    Ca,
                    Cb,
                    a,
                    b,
                    reg=reg,
                    loss_fun=loss_fun,
                    m=unbalanced,
                    log=True,
                    numItermax=max_iter,
                    G0=plan_init,
                    tol=tol,
                    symmetric=symmetric,
                    verbose=verbose,
                )

                value_quad = value_noreg
                plan = log["T"]
                # potentials = (log['u'], log['v']) TODO
                value = value_noreg + reg * nx.sum(plan * nx.log(plan + 1e-16))
            else:  # partial FGW
                if unbalanced > nx.sum(a) or unbalanced > nx.sum(b):
                    raise (
                        ValueError("Partial GW mass given in `unbalanced` is too large")
                    )

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-7

                value_noreg, log = entropic_partial_fused_gromov_wasserstein2(
                    M,
                    Ca,
                    Cb,
                    a,
                    b,
                    reg=reg,
                    loss_fun=loss_fun,
                    alpha=alpha,
                    m=unbalanced,
                    log=True,
                    numItermax=max_iter,
                    G0=plan_init,
                    tol=tol,
                    symmetric=symmetric,
                    verbose=verbose,
                )

                value_linear = log["lin_loss"]
                value_quad = log["quad_loss"]
                plan = log["T"]
                # potentials = (log['u'], log['v']) TODO
                value = value_noreg + reg * nx.sum(plan * nx.log(plan + 1e-16))

        elif unbalanced_type.lower() in ["kl", "l2"]:
            if alpha == 0:  # regularized unbalanced Wasserstein problem
                res = solve(
                    M,
                    a=a,
                    b=b,
                    reg=reg,
                    reg_type=reg_type,
                    unbalanced=unbalanced,
                    unbalanced_type=unbalanced_type,
                    method=method,
                    max_iter=max_iter,
                    plan_init=plan_init,
                    tol=tol,
                    verbose=verbose,
                )

                plan = res.plan
                potentials = res.potentials
                value_linear = res.value_linear
                value = res.value
                value_quad = 0
                status = res.status

            else:
                if max_iter is None:
                    max_iter = 100
                if tol is None:
                    tol = 1e-7

                # in this function alpha weights the linear and quadratic terms : alpha * quadratic + (1 - alpha) * linear
                # while fused_unbalanced_gromov_wasserstein uses alpha as the coefficient of the linear term.
                alpha_fugw = (1 - alpha) / alpha
                reg_fugw = unbalanced / (2 * alpha)
                epsilon_fugw = reg / (2 * alpha)
                plan, _, log = fused_unbalanced_gromov_wasserstein(
                    Ca,
                    Cb,
                    a,
                    b,
                    reg_marginals=reg_fugw,
                    divergence=unbalanced_type.lower(),
                    alpha=alpha_fugw,
                    M=M,
                    max_iter=max_iter,
                    tol=tol,
                    log=True,
                    epsilon=epsilon_fugw,
                )
                value_linear = log["linear_cost"] * alpha
                value = log["fugw_cost"] * alpha

        else:  # unbalanced AND regularized OT
            raise (
                NotImplementedError(
                    'Not implemented reg_type="{}" and unbalanced_type="{}"'.format(
                        reg_type, unbalanced_type
                    )
                )
            )

    res = OTResult(
        potentials=potentials,
        value=value,
        value_linear=value_linear,
        value_quad=value_quad,
        plan=plan,
        status=status,
        backend=nx,
        log=log,
    )

    return res
