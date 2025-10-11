# -*- coding: utf-8 -*-
"""
General OT solvers with unified API
"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

from .utils import OTResult, dist
from .lp import emd2, wasserstein_1d
from .backend import get_backend
from .unbalanced import mm_unbalanced, sinkhorn_knopp_unbalanced, lbfgsb_unbalanced
from .bregman import (
    sinkhorn_log,
    empirical_sinkhorn2,
    empirical_sinkhorn2_geomloss,
    empirical_sinkhorn_nystroem2,
)
from .smooth import smooth_ot_dual
from .gromov import (
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
)
from .gaussian import empirical_bures_wasserstein_distance
from .factored import factored_optimal_transport
from .lowrank import lowrank_sinkhorn
from .optim import cg

import warnings


lst_method_lazy = [
    "1d",
    "gaussian",
    "lowrank",
    "nystroem",
    "factored",
    "geomloss",
    "geomloss_auto",
    "geomloss_tensorized",
    "geomloss_online",
    "geomloss_multiscale",
]


def solve(
    M,
    a=None,
    b=None,
    reg=None,
    c=None,
    reg_type="KL",
    unbalanced=None,
    unbalanced_type="KL",
    method=None,
    n_threads=1,
    max_iter=None,
    plan_init=None,
    potentials_init=None,
    tol=None,
    verbose=False,
    grad="autodiff",
):
    r"""Solve the discrete optimal transport problem and return :any:`OTResult` object

    The function solves the following general optimal transport problem

    .. math::
        \min_{\mathbf{T}\geq 0} \quad \sum_{i,j} T_{i,j}M_{i,j} + \lambda_r R(\mathbf{T}) +
        \lambda_1 U(\mathbf{T}\mathbf{1},\mathbf{a}) +
        \lambda_2 U(\mathbf{T}^T\mathbf{1},\mathbf{b})

    The regularization is selected with `reg` (:math:`\lambda_r`) and `reg_type`. By
    default ``reg=None`` and there is no regularization. The unbalanced marginal
    penalization can be selected with `unbalanced` (:math:`(\lambda_1, \lambda_2)`) and
    `unbalanced_type`. By default ``unbalanced=None`` and the function
    solves the exact optimal transport problem (respecting the marginals).

    Parameters
    ----------
    M : array-like, shape (dim_a, dim_b)
        Loss matrix
    a : array-like, shape (dim_a,), optional
        Samples weights in the source domain (default is uniform)
    b : array-like, shape (dim_b,), optional
        Samples weights in the source domain (default is uniform)
    reg : float, optional
        Regularization weight :math:`\lambda_r`, by default None (no reg., exact
        OT)
    c : array-like, shape (dim_a, dim_b), optional (default=None)
        Reference measure for the regularization.
        If None, then use :math:`\mathbf{c} = \mathbf{a} \mathbf{b}^T`.
        If :math:`\texttt{reg_type}=`'entropy', then :math:`\mathbf{c} = 1_{dim_a} 1_{dim_b}^T`.
    reg_type : str, optional
        Type of regularization :math:`R`  either "KL", "L2", "entropy",
        by default "KL". a tuple of functions can be provided for general
        solver (see :any:`cg`). This is only used when ``reg!=None``.
    unbalanced : float or indexable object of length 1 or 2
        Marginal relaxation term.
        If it is a scalar or an indexable object of length 1,
        then the same relaxation is applied to both marginal relaxations.
        The balanced OT can be recovered using :math:`unbalanced=float("inf")`.
        For semi-relaxed case, use either
        :math:`unbalanced=(float("inf"), scalar)` or
        :math:`unbalanced=(scalar, float("inf"))`.
        If unbalanced is an array,
        it must have the same backend as input arrays `(a, b, M)`.
    unbalanced_type : str, optional
        Type of unbalanced penalization function :math:`U`  either "KL", "L2",
        "TV", by default "KL".
    method : str, optional
        Method for solving the problem when multiple algorithms are available,
        default None for automatic selection.
    n_threads : int, optional
        Number of OMP threads for exact OT solver, by default 1
    max_iter : int, optional
        Maximum number of iterations, by default None (default values in each solvers)
    plan_init : array-like, shape (dim_a, dim_b), optional
        Initialization of the OT plan for iterative methods, by default None
    potentials_init : (array-like(dim_a,),array-like(dim_b,)), optional
        Initialization of the OT dual potentials for iterative methods, by default None
    tol : _type_, optional
        Tolerance for solution precision, by default None (default values in each solvers)
    verbose : bool, optional
        Print information in the solver, by default False
    grad : str, optional
        Type of gradient computation, either or 'autodiff', 'envelope' or 'last_step' used only for
        Sinkhorn solver. By default 'autodiff' provides gradients wrt all
        outputs (`plan, value, value_linear`) but with important memory cost.
        'envelope' provides gradients only for `value` and and other outputs are
        detached. This is useful for memory saving when only the value is needed. 'last_step' provides
        gradients only for the last iteration of the Sinkhorn solver, but provides gradient for both the OT plan and the objective values.
        'detach' does not compute the gradients for the Sinkhorn solver.

    Returns
    -------
    res : OTResult()
        Result of the optimization problem. The information can be obtained as follows:

        - res.plan : OT plan :math:`\mathbf{T}`
        - res.potentials : OT dual potentials
        - res.value : Optimal value of the optimization problem
        - res.value_linear : Linear OT loss with the optimal OT plan

        See :any:`OTResult` for more information.

    Notes
    -----

    The following methods are available for solving the OT problems:

    - **Classical exact OT problem [1]** (default parameters) :

    .. math::
        \min_\mathbf{T} \quad \langle \mathbf{T}, \mathbf{M} \rangle_F

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve(M, a, b)

    - **Entropic regularized OT [2]** (when ``reg!=None``):

    .. math::
        \min_\mathbf{T} \quad \langle \mathbf{T}, \mathbf{M} \rangle_F + \lambda R(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0

    can be solved with the following code:

    .. code-block:: python

        # default is ``"KL"`` regularization (``reg_type="KL"``)
        res = ot.solve(M, a, b, reg=1.0)
        # or for original Sinkhorn paper formulation [2]
        res = ot.solve(M, a, b, reg=1.0, reg_type='entropy')

        # Use envelope theorem differentiation for memory saving
        res = ot.solve(M, a, b, reg=1.0, grad='envelope') # M, a, b are torch tensors
        res.value.backward() # only the value is differentiable

    Note that by default the Sinkhorn solver uses automatic differentiation to
    compute the gradients of the values and plan. This can be changed with the
    `grad` parameter. The `envelope` mode computes the gradients only
    for the value and the other outputs are detached. This is useful for
    memory saving when only the gradient of value is needed.

    - **Quadratic regularized OT [17]** (when ``reg!=None`` and ``reg_type="L2"``):

    .. math::
        \min_\mathbf{T} \quad \langle \mathbf{T}, \mathbf{M} \rangle_F + \lambda R(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve(M,a,b,reg=1.0,reg_type='L2')

    - **Unbalanced OT [41]** (when ``unbalanced!=None``):

    .. math::
        \min_{\mathbf{T}\geq 0} \quad \sum_{i,j} T_{i,j}M_{i,j} +
        \lambda_1 U(\mathbf{T}\mathbf{1},\mathbf{a}) +
        \lambda_2 U(\mathbf{T}^T\mathbf{1},\mathbf{b})

    can be solved with the following code:

    .. code-block:: python

        # default is ``"KL"``
        res = ot.solve(M,a,b,unbalanced=1.0)
        # quadratic unbalanced OT
        res = ot.solve(M,a,b,unbalanced=1.0,unbalanced_type='L2')
        # TV = partial OT
        res = ot.solve(M,a,b,unbalanced=1.0,unbalanced_type='TV')


    - **Regularized unbalanced regularized OT [34]** (when ``unbalanced!=None`` and ``reg!=None``):

    .. math::
        \min_{\mathbf{T}\geq 0} \quad \sum_{i,j} T_{i,j}M_{i,j} + \lambda_r R(\mathbf{T}) +
        \lambda_1 U(\mathbf{T}\mathbf{1},\mathbf{a}) +
        \lambda_2 U(\mathbf{T}^T\mathbf{1},\mathbf{b})

    can be solved with the following code:

    .. code-block:: python

        # default is ``"KL"`` for both
        res = ot.solve(M,a,b,reg=1.0,unbalanced=1.0)
        # quadratic unbalanced OT with KL regularization
        res = ot.solve(M,a,b,reg=1.0,unbalanced=1.0,unbalanced_type='L2')
        # both quadratic
        res = ot.solve(M,a,b,reg=1.0, reg_type='L2',unbalanced=1.0,unbalanced_type='L2')


    .. _references-solve:
    References
    ----------

    .. [1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W.
        (2011, December).  Displacement interpolation using Lagrangian mass
        transport. In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p.
        158). ACM.

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information Processing
        Systems (NIPS) 26, 2013

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems.
        arXiv preprint arXiv:1607.05816.

    .. [17] Blondel, M., Seguy, V., & Rolet, A. (2018). Smooth and Sparse
        Optimal Transport. Proceedings of the Twenty-First International
        Conference on Artificial Intelligence and Statistics (AISTATS).

    .. [34] Feydy, J., Séjourné, T., Vialard, F. X., Amari, S. I., Trouvé,
        A., & Peyré, G. (2019, April). Interpolating between optimal transport
        and MMD using Sinkhorn divergences. In The 22nd International Conference
        on Artificial Intelligence and Statistics (pp. 2681-2690). PMLR.

    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.

    """
    # detect backend
    nx = get_backend(M, a, b, c)

    # create uniform weights if not given
    if a is None:
        a = nx.ones(M.shape[0], type_as=M) / M.shape[0]
    if b is None:
        b = nx.ones(M.shape[1], type_as=M) / M.shape[1]
    if c is None:
        c = a[:, None] * b[None, :]

    if reg is None:
        reg = 0

    # default values for solutions
    potentials = None
    value = None
    value_linear = None
    plan = None
    status = None

    if reg == 0:  # exact OT
        if unbalanced is None:  # Exact balanced OT
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

        elif unbalanced_type.lower() in ["kl", "l2"]:  # unbalanced exact OT
            # default values for exact unbalanced OT
            if max_iter is None:
                max_iter = 1000
            if tol is None:
                tol = 1e-12

            plan, log = mm_unbalanced(
                a,
                b,
                M,
                reg_m=unbalanced,
                c=c,
                reg=reg,
                div=unbalanced_type,
                numItermax=max_iter,
                stopThr=tol,
                log=True,
                verbose=verbose,
                G0=plan_init,
            )

            value_linear = log["cost"]
            value = log["total_cost"]

        elif unbalanced_type.lower() == "tv":
            if max_iter is None:
                max_iter = 1000
            if tol is None:
                tol = 1e-12
            if isinstance(reg_type, str):
                reg_type = reg_type.lower()

            plan, log = lbfgsb_unbalanced(
                a,
                b,
                M,
                reg=reg,
                reg_m=unbalanced,
                c=c,
                reg_div=reg_type,
                regm_div=unbalanced_type,
                numItermax=max_iter,
                stopThr=tol,
                verbose=verbose,
                log=True,
                G0=plan_init,
            )

            value_linear = log["cost"]
            value = log["total_cost"]

        else:
            raise (
                NotImplementedError(
                    'Unknown unbalanced_type="{}"'.format(unbalanced_type)
                )
            )

    else:  # regularized OT
        if unbalanced is None:  # Balanced regularized OT
            if isinstance(reg_type, tuple):  # general solver
                f, df = reg_type

                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                plan, log = cg(
                    a,
                    b,
                    M,
                    reg=reg,
                    f=f,
                    df=df,
                    numItermax=max_iter,
                    stopThr=tol,
                    log=True,
                    verbose=verbose,
                    G0=plan_init,
                )

                value_linear = nx.sum(M * plan)
                value = log["loss"][-1]
                potentials = (log["u"], log["v"])

            elif reg_type.lower() in ["entropy", "kl"]:
                if grad in [
                    "envelope",
                    "last_step",
                    "detach",
                ]:  # if envelope, last_step or detach then detach the input
                    M0, a0, b0 = M, a, b
                    M, a, b = nx.detach(M, a, b)

                # default values for sinkhorn
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9
                if grad == "last_step":
                    if max_iter == 0:
                        raise ValueError(
                            "The maximum number of iterations must be greater than 0 when using grad=last_step."
                        )
                    max_iter = max_iter - 1

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

                potentials = (log["log_u"], log["log_v"])

                # if last_step, compute the last step of the Sinkhorn algorithm with the non-detached inputs
                if grad == "last_step":
                    loga = nx.log(a0)
                    logb = nx.log(b0)
                    v = logb - nx.logsumexp(-M0 / reg + potentials[0][:, None], 0)
                    u = loga - nx.logsumexp(-M0 / reg + potentials[1][None, :], 1)
                    plan = nx.exp(-M0 / reg + u[:, None] + v[None, :])
                    potentials = (u, v)
                    log["niter"] = max_iter + 1
                    log["log_u"] = u
                    log["log_v"] = v
                    log["u"] = nx.exp(u)
                    log["v"] = nx.exp(v)

                value_linear = nx.sum(M * plan)

                if reg_type.lower() == "entropy":
                    value = value_linear + reg * nx.sum(plan * nx.log(plan + 1e-16))
                else:
                    value = value_linear + reg * nx.kl_div(
                        plan, a[:, None] * b[None, :]
                    )

                if grad == "envelope":  # set the gradient at convergence
                    value = nx.set_gradients(
                        value,
                        (M0, a0, b0),
                        (
                            plan,
                            reg * (potentials[0] - potentials[0].mean()),
                            reg * (potentials[1] - potentials[1].mean()),
                        ),
                    )

            elif reg_type.lower() == "l2":
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                plan, log = smooth_ot_dual(
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
                value = value_linear + reg * nx.sum(plan**2)
                potentials = (log["alpha"], log["beta"])

            else:
                raise (
                    NotImplementedError(
                        'Not implemented reg_type="{}"'.format(reg_type)
                    )
                )

        else:  # unbalanced AND regularized OT
            if (
                not isinstance(reg_type, tuple)
                and reg_type.lower() in ["kl"]
                and unbalanced_type.lower() == "kl"
            ):
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                plan, log = sinkhorn_knopp_unbalanced(
                    a,
                    b,
                    M,
                    reg=reg,
                    reg_m=unbalanced,
                    method=method,
                    reg_type=reg_type,
                    c=c,
                    warmstart=potentials_init,
                    numItermax=max_iter,
                    stopThr=tol,
                    verbose=verbose,
                    log=True,
                )

                value_linear = log["cost"]
                value = log["total_cost"]

                potentials = (log["logu"], log["logv"])

            elif (
                isinstance(reg_type, tuple)
                or reg_type.lower() in ["kl", "l2", "entropy"]
            ) and unbalanced_type.lower() in ["kl", "l2", "tv"]:
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-12
                if isinstance(reg_type, str):
                    reg_type = reg_type.lower()

                plan, log = lbfgsb_unbalanced(
                    a,
                    b,
                    M,
                    reg=reg,
                    reg_m=unbalanced,
                    c=c,
                    reg_div=reg_type,
                    regm_div=unbalanced_type,
                    numItermax=max_iter,
                    stopThr=tol,
                    verbose=verbose,
                    log=True,
                    G0=plan_init,
                )

                value_linear = log["cost"]
                value = log["total_cost"]

            else:
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
        plan=plan,
        status=status,
        backend=nx,
    )

    return res


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
        "partial", by default "KL" but note that it is not implemented yet.
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
            raise (NotImplementedError('Unbalanced_type="{}"'.format(unbalanced_type)))

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


def solve_sample(
    X_a,
    X_b,
    a=None,
    b=None,
    metric="sqeuclidean",
    reg=None,
    c=None,
    reg_type="KL",
    unbalanced=None,
    unbalanced_type="KL",
    lazy=False,
    batch_size=None,
    method=None,
    n_threads=1,
    max_iter=None,
    plan_init=None,
    rank=100,
    scaling=0.95,
    potentials_init=None,
    X_init=None,
    tol=None,
    verbose=False,
    grad="autodiff",
    random_state=None,
):
    r"""Solve the discrete optimal transport problem using the samples in the source and target domains.

    The function solves the following general optimal transport problem

    .. math::
        \min_{\mathbf{T}\geq 0} \quad \sum_{i,j} T_{i,j}M_{i,j} + \lambda_r R(\mathbf{T}) +
        \lambda_u U(\mathbf{T}\mathbf{1},\mathbf{a}) +
        \lambda_u U(\mathbf{T}^T\mathbf{1},\mathbf{b})

    where the cost matrix :math:`\mathbf{M}` is computed from the samples in the
    source and target domains such that :math:`M_{i,j} = d(x_i,y_j)` where
    :math:`d` is a metric (by default the squared Euclidean distance).

    The regularization is selected with `reg` (:math:`\lambda_r`) and `reg_type`. By
    default ``reg=None`` and there is no regularization. The unbalanced marginal
    penalization can be selected with `unbalanced` (:math:`\lambda_u`) and
    `unbalanced_type`. By default ``unbalanced=None`` and the function
    solves the exact optimal transport problem (respecting the marginals).

    Parameters
    ----------
    X_a : array-like, shape (n_samples_a, dim)
        samples in the source domain
    X_b : array-like, shape (n_samples_b, dim)
        samples in the target domain
    a : array-like, shape (dim_a,), optional
        Samples weights in the source domain (default is uniform)
    b : array-like, shape (dim_b,), optional
        Samples weights in the source domain (default is uniform)
    reg : float, optional
        Regularization weight :math:`\lambda_r`, by default None (no reg., exact
        OT)
    c : array-like, shape (dim_a, dim_b), optional (default=None)
        Reference measure for the regularization.
        If None, then use :math:`\mathbf{c} = \mathbf{a} \mathbf{b}^T`.
        If :math:`\texttt{reg_type}=`'entropy', then :math:`\mathbf{c} = 1_{dim_a} 1_{dim_b}^T`.
    reg_type : str, optional
        Type of regularization :math:`R`  either "KL", "L2", "entropy", by default "KL"
    unbalanced : float or indexable object of length 1 or 2
        Marginal relaxation term.
        If it is a scalar or an indexable object of length 1,
        then the same relaxation is applied to both marginal relaxations.
        The balanced OT can be recovered using :math:`unbalanced=float("inf")`.
        For semi-relaxed case, use either
        :math:`unbalanced=(float("inf"), scalar)` or
        :math:`unbalanced=(scalar, float("inf"))`.
        If unbalanced is an array,
        it must have the same backend as input arrays `(a, b, M)`.
    unbalanced_type : str, optional
        Type of unbalanced penalization function :math:`U`  either "KL", "L2", "TV", by default "KL"
    lazy : bool, optional
        Return :any:`OTResultlazy` object to reduce memory cost when True, by
        default False
    batch_size : int, optional
        Batch size for lazy solver, by default None (default values in each
        solvers)
    method : str, optional
        Method for solving the problem, this can be used to select the solver
        for unbalanced problems (see :any:`ot.solve`), or to select a specific
        large scale solver.
    n_threads : int, optional
        Number of OMP threads for exact OT solver, by default 1
    max_iter : int, optional
        Maximum number of iteration, by default None (default values in each solvers)
    plan_init : array-like, shape (dim_a, dim_b), optional
        Initialization of the OT plan for iterative methods, by default None
    rank : int, optional
        Rank of the OT matrix for lazy solers (method='factored') or (method='nystroem'), by default 100
    scaling : float, optional
        Scaling factor for the epsilon scaling lazy solvers (method='geomloss'), by default 0.95
    potentials_init : (array-like(dim_a,),array-like(dim_b,)), optional
        Initialization of the OT dual potentials for iterative methods, by default None
    tol : _type_, optional
        Tolerance for solution precision, by default None (default values in each solvers)
    verbose : bool, optional
        Print information in the solver, by default False
    grad : str, optional
        Type of gradient computation, either or 'autodiff' or 'envelope'  used only for
        Sinkhorn solver. By default 'autodiff' provides gradients wrt all
        outputs (`plan, value, value_linear`) but with important memory cost.
        'envelope' provides gradients only for `value` and and other outputs are
        detached. This is useful for memory saving when only the value is needed.
    random_state : int, optional
        The random state for sampling the components in each distribution for method='nystroem'.

    Returns
    -------

    res : OTResult()
        Result of the optimization problem. The information can be obtained as follows:

        - res.plan : OT plan :math:`\mathbf{T}`
        - res.potentials : OT dual potentials
        - res.value : Optimal value of the optimization problem
        - res.value_linear : Linear OT loss with the optimal OT plan
        - res.lazy_plan : Lazy OT plan (when ``lazy=True`` or lazy method)

        See :any:`OTResult` for more information.

    Notes
    -----

    The following methods are available for solving the OT problems:

    - **Classical exact OT problem [1]** (default parameters) :

    .. math::
        \min_\mathbf{T} \quad \langle \mathbf{T}, \mathbf{M} \rangle_F

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0,  M_{i,j} = d(x_i,y_j)



    can be solved with the following code:

    .. code-block:: python

        res = ot.solve_sample(xa, xb, a, b)

        # for uniform weights
        res = ot.solve_sample(xa, xb)

    - **Entropic regularized OT [2]** (when ``reg!=None``):

    .. math::
        \min_\mathbf{T} \quad \langle \mathbf{T}, \mathbf{M} \rangle_F + \lambda R(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0,  M_{i,j} = d(x_i,y_j)

    can be solved with the following code:

    .. code-block:: python

        # default is ``"KL"`` regularization (``reg_type="KL"``)
        res = ot.solve_sample(xa, xb, a, b, reg=1.0)
        # or for original Sinkhorn paper formulation [2]
        res = ot.solve_sample(xa, xb, a, b, reg=1.0, reg_type='entropy')

        # lazy solver of memory complexity O(n)
        res = ot.solve_sample(xa, xb, a, b, reg=1.0, lazy=True, batch_size=100)
        # lazy OT plan
        lazy_plan = res.lazy_plan

        # Use envelope theorem differentiation for memory saving
        res = ot.solve_sample(xa, xb, a, b, reg=1.0, grad='envelope')
        res.value.backward() # only the value is differentiable

    Note that by default the Sinkhorn solver uses automatic differentiation to
    compute the gradients of the values and plan. This can be changed with the
    `grad` parameter. The `envelope` mode computes the gradients only
    for the value and the other outputs are detached. This is useful for
    memory saving when only the gradient of value is needed.

    We also have a very efficient solver with compiled CPU/CUDA code using
    geomloss/PyKeOps that can be used with the following code:

    .. code-block:: python

        # automatic solver
        res = ot.solve_sample(xa, xb, a, b, reg=1.0, method='geomloss')

        # force O(n) memory efficient solver
        res = ot.solve_sample(xa, xb, a, b, reg=1.0, method='geomloss_online')

        # force pre-computed cost matrix
        res = ot.solve_sample(xa, xb, a, b, reg=1.0, method='geomloss_tensorized')

        # use multiscale solver
        res = ot.solve_sample(xa, xb, a, b, reg=1.0, method='geomloss_multiscale')

        # One can play with speed (small scaling factor) and precision (scaling close to 1)
        res = ot.solve_sample(xa, xb, a, b, reg=1.0, method='geomloss', scaling=0.5)

    - **Quadratic regularized OT [17]** (when ``reg!=None`` and ``reg_type="L2"``):

    .. math::
        \min_\mathbf{T} \quad \langle \mathbf{T}, \mathbf{M} \rangle_F + \lambda R(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0,  M_{i,j} = d(x_i,y_j)

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve_sample(xa, xb, a, b, reg=1.0, reg_type='L2')

    - **Unbalanced OT [41]** (when ``unbalanced!=None``):

    .. math::
        \min_{\mathbf{T}\geq 0} \quad \sum_{i,j} T_{i,j}M_{i,j} + \lambda_u U(\mathbf{T}\mathbf{1},\mathbf{a}) + \lambda_u U(\mathbf{T}^T\mathbf{1},\mathbf{b})

        \text{with} \ M_{i,j} = d(x_i,y_j)

    can be solved with the following code:

    .. code-block:: python

        # default is ``"KL"``
        res = ot.solve_sample(xa, xb, a, b, unbalanced=1.0)
        # quadratic unbalanced OT
        res = ot.solve_sample(xa, xb, a, b, unbalanced=1.0,unbalanced_type='L2')
        # TV = partial OT
        res = ot.solve_sample(xa, xb, a, b, unbalanced=1.0,unbalanced_type='TV')


    - **Regularized unbalanced regularized OT [34]** (when ``unbalanced!=None`` and ``reg!=None``):

    .. math::
        \min_{\mathbf{T}\geq 0} \quad \sum_{i,j} T_{i,j}M_{i,j} + \lambda_r R(\mathbf{T}) + \lambda_u U(\mathbf{T}\mathbf{1},\mathbf{a}) + \lambda_u U(\mathbf{T}^T\mathbf{1},\mathbf{b})

        \text{with} \ M_{i,j} = d(x_i,y_j)

    can be solved with the following code:

    .. code-block:: python

        # default is ``"KL"`` for both
        res = ot.solve_sample(xa, xb, a, b, reg=1.0, unbalanced=1.0)
        # quadratic unbalanced OT with KL regularization
        res = ot.solve_sample(xa, xb, a, b, reg=1.0, unbalanced=1.0,unbalanced_type='L2')
        # both quadratic
        res = ot.solve_sample(xa, xb, a, b, reg=1.0, reg_type='L2',
        unbalanced=1.0, unbalanced_type='L2')


    - **Factored OT [2]** (when ``method='factored'``):

    This method solve the following OT problem [40]_

    .. math::
        \mathop{\arg \min}_\mu \quad  W_2^2(\mu_a,\mu)+ W_2^2(\mu,\mu_b)

    where $\mu$ is a uniform weighted empirical distribution of  :math:`\mu_a` and :math:`\mu_b` are the empirical measures associated
    to the samples in the source and target domains, and :math:`W_2` is the
    Wasserstein distance. This problem is solved using exact OT solvers for
    `reg=None` and the Sinkhorn solver for `reg!=None`. The solution provides
    two transport plans that can be used to recover a low rank OT plan between
    the two distributions.

    .. code-block:: python

        res = ot.solve_sample(xa, xb, method='factored', rank=10)

        # recover the lazy low rank plan
        factored_solution_lazy = res.lazy_plan

        # recover the full low rank plan
        factored_solution = factored_solution_lazy[:]

    - ** Nystroem OT [76] ** (when ``method='nystroem'``):

    Corresponds to a low rank approximation of entropic OT (for a squared Euclidean cost) that runs in linear time.

    - **Gaussian Bures-Wasserstein [2]** (when ``method='gaussian'``):

    This method computes the Gaussian Bures-Wasserstein distance between two
    Gaussian distributions estimated from the empirical distributions

    .. math::
        \mathcal{W}(\mu_s, \mu_t)_2^2= \left\lVert \mathbf{m}_s - \mathbf{m}_t \right\rVert^2 + \mathcal{B}(\Sigma_s, \Sigma_t)^{2}

    where :

    .. math::
        \mathbf{B}(\Sigma_s, \Sigma_t)^{2} = \text{Tr}\left(\Sigma_s + \Sigma_t - 2 \sqrt{\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2}} \right)

    The covariances and means are estimated from the data.

    .. code-block:: python

        res = ot.solve_sample(xa, xb, method='gaussian')

        # recover the squared Gaussian Bures-Wasserstein distance
        BW_dist = res.value

    - **Wasserstein 1d [1]** (when ``method='1D'``):

    This method computes the Wasserstein distance between two 1d distributions
    estimated from the empirical distributions. For multivariate data the
    distances are computed independently for each dimension.

    .. code-block:: python

        res = ot.solve_sample(xa, xb, method='1D')

        # recover the squared Wasserstein distances
        W_dists = res.value


    .. _references-solve-sample:
    References
    ----------

    .. [1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W.
        (2011, December).  Displacement interpolation using Lagrangian mass
        transport. In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p.
        158). ACM.

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information Processing
        Systems (NIPS) 26, 2013

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems.
        arXiv preprint arXiv:1607.05816.

    .. [17] Blondel, M., Seguy, V., & Rolet, A. (2018). Smooth and Sparse
        Optimal Transport. Proceedings of the Twenty-First International
        Conference on Artificial Intelligence and Statistics (AISTATS).

    .. [34] Feydy, J., Séjourné, T., Vialard, F. X., Amari, S. I., Trouvé,
        A., & Peyré, G. (2019, April). Interpolating between optimal transport
        and MMD using Sinkhorn divergences. In The 22nd International Conference
        on Artificial Intelligence and Statistics (pp. 2681-2690). PMLR.

    .. [40] Forrow, A., Hütter, J. C., Nitzan, M., Rigollet, P., Schiebinger,
        G., & Weed, J. (2019, April). Statistical optimal transport via factored
        couplings. In The 22nd International Conference on Artificial
        Intelligence and Statistics (pp. 2454-2465). PMLR.

    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.

    .. [65] Scetbon, M., Cuturi, M., & Peyré, G. (2021).
        Low-rank Sinkhorn Factorization. In International Conference on
        Machine Learning.

    .. [80] Altschuler, J., Bach, F., Rudi, A., Niles-Weed, J. (2019).
        Massively scalable Sinkhorn distances via the Nyström method. NeurIPS.


    """

    if method is not None and method.lower() in lst_method_lazy:
        lazy0 = lazy
        lazy = True

    if not lazy:  # default non lazy solver calls ot.solve
        # compute cost matrix M and use solve function
        M = dist(X_a, X_b, metric)

        res = solve(
            M,
            a,
            b,
            reg,
            c,
            reg_type,
            unbalanced,
            unbalanced_type,
            method,
            n_threads,
            max_iter,
            plan_init,
            potentials_init,
            tol,
            verbose,
            grad,
        )

        return res

    else:
        # Detect backend
        nx = get_backend(X_a, X_b, a, b)

        # default values for solutions
        potentials = None
        value = None
        value_linear = None
        plan = None
        lazy_plan = None
        status = None
        log = None

        method = method.lower() if method is not None else ""

        if method == "1d":  # Wasserstein 1d (parallel on all dimensions)
            if metric == "sqeuclidean":
                p = 2
            elif metric in ["euclidean", "cityblock"]:
                p = 1
            else:
                raise (
                    NotImplementedError('Not implemented metric="{}"'.format(metric))
                )

            value = wasserstein_1d(X_a, X_b, a, b, p=p)
            value_linear = value

        elif method == "gaussian":  # Gaussian Bures-Wasserstein
            if metric.lower() not in ["sqeuclidean"]:
                raise (
                    NotImplementedError('Not implemented metric="{}"'.format(metric))
                )

            if reg is None:
                reg = 1e-6

            value, log = empirical_bures_wasserstein_distance(
                X_a, X_b, reg=reg, log=True
            )
            value = value**2  # return the value (squared bures distance)
            value_linear = value  # return the value

        elif method == "factored":  # Factored OT
            if metric.lower() not in ["sqeuclidean"]:
                raise (
                    NotImplementedError('Not implemented metric="{}"'.format(metric))
                )

            if max_iter is None:
                max_iter = 100
            if tol is None:
                tol = 1e-7
            if reg is None:
                reg = 0

            Q, R, X, log = factored_optimal_transport(
                X_a,
                X_b,
                reg=reg,
                r=rank,
                log=True,
                stopThr=tol,
                numItermax=max_iter,
                verbose=verbose,
            )
            log["X"] = X

            value_linear = log["costa"] + log["costb"]
            value = value_linear  # TODO add reg term
            lazy_plan = log["lazy_plan"]
            if not lazy0:  # store plan if not lazy
                plan = lazy_plan[:]

        elif method == "lowrank":
            if metric.lower() not in ["sqeuclidean"]:
                raise (
                    NotImplementedError('Not implemented metric="{}"'.format(metric))
                )

            if max_iter is None:
                max_iter = 2000
            if tol is None:
                tol = 1e-7
            if reg is None:
                reg = 0

            Q, R, g, log = lowrank_sinkhorn(
                X_a,
                X_b,
                rank=rank,
                reg=reg,
                a=a,
                b=b,
                numItermax=max_iter,
                stopThr=tol,
                log=True,
            )
            value = log["value"]
            value_linear = log["value_linear"]
            lazy_plan = log["lazy_plan"]
            if not lazy0:  # store plan if not lazy
                plan = lazy_plan[:]

        elif method == "nystroem":
            if metric.lower() not in ["sqeuclidean"]:
                raise (
                    NotImplementedError('Not implemented metric="{}"'.format(metric))
                )

            if max_iter is None:
                max_iter = 1000
            if tol is None:
                tol = 1e-7
            if reg is None:
                reg = 1.0

            value, log = empirical_sinkhorn_nystroem2(
                X_a,
                X_b,
                reg=reg,
                anchors=rank,
                a=a,
                b=b,
                numItermax=max_iter,
                verbose=verbose,
                stopThr=tol,
                random_state=random_state,
                log=True,
            )

            lazy_plan = log["lazy_plan"]
            if not lazy0:  # store plan if not lazy
                plan = lazy_plan[:]

        elif method.startswith("geomloss"):  # Geomloss solver for entropic OT
            split_method = method.split("_")
            if len(split_method) == 2:
                backend = split_method[1]
            else:
                if lazy0 is None:
                    backend = "auto"
                elif lazy0:
                    backend = "online"
                else:
                    backend = "tensorized"

            value, log = empirical_sinkhorn2_geomloss(
                X_a,
                X_b,
                reg=reg,
                a=a,
                b=b,
                metric=metric,
                log=True,
                verbose=verbose,
                scaling=scaling,
                backend=backend,
            )

            lazy_plan = log["lazy_plan"]
            if not lazy0:  # store plan if not lazy
                plan = lazy_plan[:]

            # return scaled potentials (to be consistent with other solvers)
            potentials = (
                log["f"] / (lazy_plan.blur**2),
                log["g"] / (lazy_plan.blur**2),
            )

        elif reg is None or reg == 0:  # exact OT
            if unbalanced is None:  # balanced EMD solver not available for lazy
                raise (
                    NotImplementedError(
                        "Exact OT solver with lazy=True not implemented"
                    )
                )

            else:
                raise (
                    NotImplementedError(
                        'Non regularized solver with unbalanced_type="{}" not implemented'.format(
                            unbalanced_type
                        )
                    )
                )

        else:
            if unbalanced is None:
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9
                if batch_size is None:
                    batch_size = 100

                value_linear, log = empirical_sinkhorn2(
                    X_a,
                    X_b,
                    reg,
                    a,
                    b,
                    metric=metric,
                    numIterMax=max_iter,
                    stopThr=tol,
                    isLazy=True,
                    batchSize=batch_size,
                    verbose=verbose,
                    log=True,
                )
                # compute potentials
                potentials = (log["u"], log["v"])
                lazy_plan = log["lazy_plan"]

            else:
                raise (
                    NotImplementedError(
                        'Not implemented unbalanced_type="{}" with regularization'.format(
                            unbalanced_type
                        )
                    )
                )

        res = OTResult(
            potentials=potentials,
            value=value,
            lazy_plan=lazy_plan,
            value_linear=value_linear,
            plan=plan,
            status=status,
            backend=nx,
            log=log,
        )
        return res
