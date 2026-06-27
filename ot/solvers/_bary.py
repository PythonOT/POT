# -*- coding: utf-8 -*-
"""
General OT solvers with unified API
"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

from ..utils import BaryResult
from ..lp import free_support_barycenter_generic_costs
from ..backend import get_backend

from ._linear import solve, solve_sample, lst_method_lazy


def _bary_sample_bcd(
    X_a_list,
    X_b_init,
    a_list,
    b,
    w,
    metric,
    inner_solver,
    update_masses,
    warmstart_plan,
    warmstart_potentials,
    stopping_criterion,
    max_iter_bary,
    tol_bary,
    verbose,
    log,
    nx,
):
    """Compute the barycenter using BCD.

    Parameters
    ----------
    X_a_list : list of array-like, shape (n_samples_k, dim)
        List of samples in each source distribution
    X_b_init : array-like, shape (n_samples_b, dim),
        Initialization of the barycenter samples.
    a_list : list of array-like, shape (dim_k,)
        List of samples weights in each source distribution
    b : array-like, shape (n_samples_b,)
        Weights of the barycenter samples.
    w : list of array-like, shape (N,)
        Samples barycentric weights
    metric : str
        Metric to use for the cost matrix, by default "sqeuclidean"
    inner_solver : callable with parameters (X_a, X_b, a, b, plan_init, potentials_init)
        Function to solve the inner OT problem with inputs: source and target samples (`X_a`, `X_b`),
        their respective masses (`a`, `b`), optional initial transport plan `plan_init`, optional initial
        dual potentials `potentials_init` used for instance with sinkhorn-like inner solvers.
    update_masses : bool
        Update the masses of the barycenter, depending on whether balanced or unbalanced OT is used.
    warmstart_plan : bool
        Use the previous plan as initialization for the inner solver. Set based on inner solver type in ot.bary_sample
    warmstart_potentials : bool
        Use the previous potentials as initialization for the inner solver. Set based on inner solver type in ot.bary_sample
    stopping_criterion : str
        Stopping criterion for the BCD algorithm. Can be "loss" or "bary".
    max_iter_bary : int
        Maximum number of iterations for the barycenter
    tol_bary : float
        Tolerance for the barycenter convergence
    verbose : bool
        Print information in the solver
    log : bool
        Log the loss during the iterations
    nx: backend, optional
        Backend to use for the computation. If provided it must match the backend of the input arrays.
    Returns
    -------

    res : BaryResult()
        Result of the optimization problem. The information can be obtained as follows:

        - res.X : Barycenter samples
        - res.b : Barycenter weights
        - res.value : Optimal value of the optimization problem
        - res.value_linear : Linear OT loss with the optimal OT plan
        - res.list_res: List of OTResult for each inner OT problem (one per source distribution)
        - res.log: log of the optimization process (if log=True)

        See :any:`BaryResult` for more information.

    """
    if nx is None:
        nx = get_backend(*X_a_list, X_b_init, *a_list, b, w)

    X_b = X_b_init
    # inv_b is used to compute the barycenter samples in closed-form
    # inv_b is fixed in balanced OT, or updated via estimated b in unbalanced settings.
    inv_b = nx.nan_to_num(1.0 / b, nan=1.0, posinf=1.0, neginf=1.0)

    prev_criterion = float("inf")
    n_samples = len(X_a_list)

    log_ = None
    if log:
        log_ = {"stopping_criterion": []}

    # Compute the barycenter using BCD
    for it in range(max_iter_bary):
        # Solve the inner OT problem for each source distribution
        if it == 0:  # no pre-defined warmstart used at iteration 0.
            list_res = [
                inner_solver(X_a_list[k], X_b, a_list[k], b, None, None)
                for k in range(n_samples)
            ]
        elif warmstart_plan:
            list_res = [
                inner_solver(X_a_list[k], X_b, a_list[k], b, list_res[k].plan, None)
                for k in range(n_samples)
            ]
        elif warmstart_potentials:
            list_res = [
                inner_solver(
                    X_a_list[k], X_b, a_list[k], b, None, list_res[k].potentials
                )
                for k in range(n_samples)
            ]
        else:
            list_res = [
                inner_solver(X_a_list[k], X_b, a_list[k], b, None, None)
                for k in range(n_samples)
            ]

        # Update the estimated barycenter weights in unbalanced cases
        if update_masses:
            b_estimated = sum(
                [w[k] * list_res[k].plan.sum(axis=0) for k in range(n_samples)]
            )
            inv_b = nx.nan_to_num(1.0 / b_estimated, nan=1.0, posinf=1.0, neginf=1.0)

        # Update the barycenter samples
        if metric in ["sqeuclidean", "euclidean"]:
            X_b_new = (
                sum([w[k] * list_res[k].plan.T @ X_a_list[k] for k in range(n_samples)])
                * inv_b[:, None]
            )
        else:
            raise NotImplementedError('Not implemented metric="{}"'.format(metric))

        # compute criterion
        if stopping_criterion == "loss":
            new_criterion = sum([w[k] * list_res[k].value for k in range(n_samples)])
        else:  # stopping_criterion = "bary"
            new_criterion = nx.sum((X_b_new - X_b) ** 2)

        if verbose:
            if it % 1 == 0:
                print(
                    f"BCD iteration {it}: criterion {stopping_criterion} = {new_criterion:.4f}"
                )

        if log:
            log_["stopping_criterion"].append(new_criterion)
        # Check convergence
        if abs(new_criterion - prev_criterion) / abs(prev_criterion) < tol_bary:
            print(f"BCD converged in {it} iterations")
            break

        X_b = X_b_new
        prev_criterion = new_criterion

    # compute loss values

    value_linear = sum([w[k] * list_res[k].value_linear for k in range(n_samples)])
    if stopping_criterion == "loss":
        value = new_criterion
    else:
        value = sum([w[k] * list_res[k].value for k in range(n_samples)])
    # update BaryResult
    bary_res = BaryResult(
        X=X_b,
        b=b,
        value=value,
        value_linear=value_linear,
        log=log_,
        list_res=list_res,
        backend=nx,
    )
    return bary_res


def solve_bary_sample(
    X_a_list,
    n,
    a_list=None,
    w=None,
    X_b_init=None,
    b=None,
    metric="sqeuclidean",
    reg=None,
    c=None,
    reg_type="KL",
    unbalanced=None,
    unbalanced_type="KL",
    lazy=False,
    method=None,
    auto_bary_method="L2_barycentric_proj",
    warmstart=True,
    stopping_criterion="loss",
    max_iter_bary=1000,
    tol_bary=1e-5,
    random_state=0,
    verbose=False,
    **kwargs,
):
    r"""Solve the discrete OT barycenter problem over source distributions optimizing the barycenter support using Block-Coordinate Descent.

    The function solves the following general OT barycenter problem

    .. math::
        \min_{\mathbf{X} \in \mathbb{R}^{n \times d}} \min_{\{ \mathbf{T}^{(k)} \}_k \in \mathbb{R}_+^{n_i \times n}} \quad \sum_k w_k \{ \langle \mathbf{T}^{(k)}, \mathbf{M}^{(k)} \rangle_F + \lambda_r R(\mathbf{T}^{(k)}) +
        \lambda_u U(\mathbf{T^{(k)}}\mathbf{1},\mathbf{a}^{(k)}) +
        \lambda_u U(\mathbf{T}^{(k)T}\mathbf{1},\mathbf{b}) \}

    where the cost matrices :math:`\mathbf{M}^{(k)}` from each input distribution :math:`(\mathbf{X}^{(k)}, \mathbf{a}^{(k)})`
    to the barycenter domain are computed as :math:`M^{(k)}_{i,j} = d(x^{(k)}_i,x_j)` where
    :math:`d` is a metric (by default the squared Euclidean distance). For common metrics the barycenter is computed in closed-form.
    For balanced OT, the `metric` parameter can also be any callable function, or list of functions, that computes the distance from an input to the barycenter.
    In which case, the barycenter is updated by gradient descent using the provided metric(s) and the optimal transport plan(s) at each iteration.
    The barycenter probability weights are fixed to :math:`\mathbf{b}`.

    The regularization is selected with `reg` (:math:`\lambda_r`) and `reg_type`. By
    default ``reg=None`` and there is no regularization. The unbalanced marginal
    penalization can be selected with `unbalanced` (:math:`\lambda_u`) and
    `unbalanced_type`. By default ``unbalanced=None`` and the function
    solves the exact optimal transport problem (respecting the marginals).

    Parameters
    ----------
    X_a_list : list of array-like, shape (n_samples_k, dim)
        List of N samples in each source distribution
    n : int
        number of samples in the barycenter domain
    a_list : list of array-like, shape (n_samples_k,), optional
        List of samples weights in each source distribution (default is uniform)
    w : list of array-like, shape (N,), optional
        Samples barycentric weights (default is uniform)
    X_b_init : array-like, shape (n, dim), optional
        Initialization of the barycenter samples (default is gaussian random sampling)
    b : array-like, shape (n,), optional
        Barycenter weights (default is uniform)
    metric : str, callable or list of callables optional
        Metric to use for the computation of the cost matrix, by default "sqeuclidean".
        It can be a list of callables (bary, source) of length N (number of source distributions) to use different metrics for each source distribution.
        In this case, the barycenter is updated by gradient descent using the provided metric(s) and the optimal transport plan(s) at each iteration.
        If only callable is provided the same cost function is used for all source distributions.
    reg : float, optional
        Regularization weight :math:`\lambda_r`, by default None (no reg., exact
        OT)
    c : array-like, shape (dim_a, dim_b), optional (default=None)
        Reference measure for the regularization.
        If None, then use :math:`\mathbf{c} = \mathbf{a}^{(k)} \mathbf{b}^T`.
        If :math:`\texttt{reg_type}=`'entropy', then :math:`\mathbf{c} = 1_{|a^{(k)}|} 1_{|b|}^T`.
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
    method : str, optional
        Method for solving the problem, this can be used to select the solver
        for unbalanced problems (see :any:`ot.solve`), or to select a specific
        large scale solver.
    auto_bary_method: str, optional
        For balanced OT with callable metric functions, the barycenter method to use in 'L2_barycentric_proj' (default) for Euclidean
        barycentric projection, or 'true_fixed_point' for iterates using the North West Corner multi-marginal gluing method.
    warmstart : bool, optional
        Use the previous OT or potentials as initialization for the next inner solver iteration, by default False.
    stopping_criterion : str, optional
        Stopping criterion for the outer loop of the BCD solver, by default 'loss'.
        Either 'loss' to use the optimize objective or 'bary' for variations of the barycenter w.r.t the Frobenius norm.
    max_iter_bary : int, optional
        Maximum number of iteration for the outer loop of the BCD solver, by default 1000.
    tol_bary : float, optional
        Tolerance for solution precision of the barycenter problem, by default 1e-5.
    random_state : int, optional
        Random seed for the initialization of the barycenter samples, by default 0.
        Only used if `X_init` is None.
    verbose : bool, optional
        Print information in the solver, by default False
    kwargs : optional
        Additional parameters for the inner solver (see :any:`ot.solve_sample` and :any:`ot.lp.free_support_barycenter_generic_costs`)
    Returns
    -------

    res : BaryResult()
        Result of the optimization problem. The information can be obtained as follows:

        - res.X : Barycenter samples
        - res.b : Barycenter weights
        - res.value : Optimal value of the optimization problem
        - res.value_linear : Linear OT loss with the optimal OT plan
        - res.list_res: List of OTResult for each inner OT problem (one per source distribution)
        - res.log: log of the optimization process (if log=True)

        See :any:`BaryResult` for more information.

    Notes
    -----

    The following methods are available for solving barycenter problems with respect to these inner OT problems:

    - **Classical exact OT problem [1]** (default parameters) :

    .. math::
        \forall k, \quad \min_{\mathbf{T}^{(k)}} \quad \langle \mathbf{T}^{(k)}, \mathbf{M}^{(k)} \rangle_F

        s.t. \ \mathbf{T}^{(k)} \mathbf{1} = \mathbf{a}^{(k)}

             \mathbf{T}^{(k)^T} \mathbf{1} = \mathbf{b}

             \mathbf{T}^{(k)} \geq 0,  M^{(k)}_{i,j} = d(x^{(k)}_i,x_j)



    can be solved with the following code for various cost metrics between the source distributions and the barycenter:

    .. code-block:: python

        # for squared Euclidean cost, where closed-form solutions are used to update the barycenter
        res = ot.solve_bary_sample([x1, x2], n , [a1, a2], w, metric='sqeuclidean')

        # for uniform sample weights and barycentric weights,
        res = ot.solve_bary_sample([x1, x2], n, [a1, a2], w, metric='sqeuclidean')

        # for other cost functions, where the barycenter is updated with gradient descent using Pytorch
        # refer to the documentation and examples for more details.

    - **Entropic regularized OT [2]** (when ``reg!=None``):

    .. math::
        \min_{\mathbf{T}^{(k)}} \quad \langle \mathbf{T}^{(k)}, \mathbf{M}^{(k)} \rangle_F + \lambda R(\mathbf{T}^{(k)})

        s.t. \ \mathbf{T}^{(k)} \mathbf{1} = \mathbf{a}^{(k)}

             \mathbf{T}^{(k)^T} \mathbf{1} = \mathbf{b}

             \mathbf{T}^{(k)} \geq 0,  M^{(k)}_{i,j} = d(x^{(k)}_i,x_j)



    can be solved with the following code:

    .. code-block:: python

        # default is ``"KL"`` regularization (``reg_type="KL"``)
        res = ot.solve_bary_sample([x1, x2], n , [a1, a2], w, reg=1.0)

        # or for original Sinkhorn paper formulation [2]
        res = ot.solve_bary_sample([x1, x2], n , [a1, a2], w, reg=1.0, reg_type='entropy')


    - **Quadratic regularized OT [17]** (when ``reg!=None`` and ``reg_type="L2"``):

    .. math::
        \min_{\mathbf{T}^{(k)}} \quad \langle \mathbf{T}^{(k)}, \mathbf{M}^{(k)} \rangle_F + \lambda R(\mathbf{T}^{(k))})

        s.t. \ \mathbf{T}^{(k)} \mathbf{1} = \mathbf{a}^{(k)}

             \mathbf{T}^{(k)^T} \mathbf{1} = \mathbf{b}

             \mathbf{T}^{(k)} \geq 0,  M^{(k)}_{i,j} = d(x^{(k)}_i,x_j)

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve_bary_sample([x1, x2], n , [a1, a2], w, reg=1.0, reg_type='L2')

    - **Unbalanced OT [41]** (when ``unbalanced!=None``):

    .. math::
        \min_{\mathbf{T}^{(k)}\geq 0} \quad \langle \mathbf{T}^{(k)}, \mathbf{M}^{(k)} \rangle_F + \lambda_u U(\mathbf{T}^{(k)}\mathbf{1},\mathbf{a}^{(k)}) + \lambda_u U(\mathbf{T}^{(k)^T}\mathbf{1},\mathbf{b})

    can be solved with the following code:

    .. code-block:: python

        # default is ``"KL"``
        res = ot.solve_bary_sample([x1, x2], n , [a1, a2], w, unbalanced=1.0)

        # quadratic unbalanced OT
        res = ot.solve_bary_sample([x1, x2], n , [a1, a2], w, unbalanced=1.0, unbalanced_type='L2')
        # TV = partial OT
        res = ot.solve_bary_sample([x1, x2], n , [a1, a2], w, unbalanced=1.0, unbalanced_type='TV')


    - **Regularized unbalanced regularized OT [34]** (when ``unbalanced!=None`` and ``reg!=None``):

    .. math::
        \min_{\mathbf{T}^{(k)} \geq 0} \quad \langle \mathbf{T}^{(k)}, \mathbf{M}^{(k)} \rangle_F + \lambda_r R(\mathbf{T}^{(k)}) + \lambda_u U(\mathbf{T}^{(k)}\mathbf{1},\mathbf{a}^{(k)}) + \lambda_u U(\mathbf{T}^{(k)^T}\mathbf{1},\mathbf{b})


    can be solved with the following code:

    .. code-block:: python

        # default is ``"KL"`` for both
        res = ot.solve_bary_sample([x1, x2], n , [a1, a2], w, reg=1.0, unbalanced=1.0)
        # quadratic unbalanced OT with KL regularization
        res = ot.solve_bary_sample([x1, x2], n , [a1, a2], w, reg=1.0, unbalanced=1.0, unbalanced_type='L2')
        # both quadratic
        res = ot.solve_bary_sample([x1, x2], n , [a1, a2], w, reg=1.0, reg_type='L2', unbalanced=1.0, unbalanced_type='L2')

    .. _references-solve_bary_sample:
    References
    ----------

    .. [20] Cuturi, Marco, and Arnaud Doucet. "Fast computation of Wasserstein barycenters." International Conference on Machine Learning. 2014.

    .. [43] Álvarez-Esteban, Pedro C., et al. "A fixed-point approach to barycenters in Wasserstein space." Journal of Mathematical Analysis and Applications 441.2 (2016): 744-762.

    .. [77] Tanguy, Eloi and Delon, Julie and Gozlan, Nathaël (2024). Computing
        barycenters of Measures for Generic Transport Costs. arXiv preprint
        2501.04016 (2024)

    """

    if method is not None and method.lower() in lst_method_lazy:
        raise NotImplementedError(
            f"method {method} operating on lazy tensors is not implemented yet"
        )

    if stopping_criterion not in ["loss", "bary"]:
        raise ValueError(
            "stopping_criterion must be either 'loss' or 'bary', got {}".format(
                stopping_criterion
            )
        )

    n_samples = len(X_a_list)

    if lazy:
        raise (NotImplementedError("Barycenter solver with lazy=True not implemented"))
    else:
        # default non lazy solver calls ot.solve_sample within _bary_sample_bcd
        # Detect backend
        nx = get_backend(*X_a_list, X_b_init, b, w)

        # check sample weights
        if a_list is None:
            a_list = [
                nx.ones((X_a_list[k].shape[0],), type_as=X_a_list[k])
                / X_a_list[k].shape[0]
                for k in range(n_samples)
            ]

        # check samples barycentric weights
        if w is None:
            w = nx.ones(n_samples, type_as=X_a_list[0]) / n_samples

        # check X_b_init
        if X_b_init is None:
            nx.seed(random_state)
            mean_ = nx.concatenate(
                [nx.mean(X_a_list[k], axis=0) for k in range(n_samples)],
                axis=0,
            )
            mean_ = nx.mean(mean_, axis=0)
            std_ = nx.concatenate(
                [nx.std(X_a_list[k], axis=0) for k in range(n_samples)],
                axis=0,
            )
            std_ = nx.mean(std_, axis=0)
            X_b_init = (
                std_ * nx.randn(n, X_a_list[0].shape[1], type_as=X_a_list[0]) + mean_
            )

        else:
            if (X_b_init.shape[0] != n) or (X_b_init.shape[1] != X_a_list[0].shape[1]):
                raise ValueError("X_b_init must have shape (n, dim)")

        # check b
        if b is None:
            b = nx.ones((n,), type_as=X_a_list[0]) / n

        if callable(metric) or (
            isinstance(metric, list) and all(callable(m) for m in metric)
        ):
            if reg is not None or unbalanced is not None:
                raise NotImplementedError(
                    "Custom callable metric only available for balanced OT (reg=None and unbalanced=None)"
                )
            else:
                if auto_bary_method == "true_fixed_point":
                    ground_bary = kwargs.get("ground_bary", None)
                    if ground_bary is None:
                        raise ValueError(
                            "ground_bary must be provided in kwargs for true_fixed_point method with callable metrics"
                        )

                outputs = free_support_barycenter_generic_costs(
                    X_a_list,
                    a_list,
                    X_b_init,
                    metric,
                    ground_bary=None,
                    a=b,
                    numItermax=max_iter_bary,
                    method=auto_bary_method,
                    stopThr=tol_bary,
                    log=True,
                    **kwargs,
                )
                if auto_bary_method == "L2_barycentric_proj":
                    X_b, log_ = outputs

                elif auto_bary_method == "true_fixed_point":
                    X_b, b, log_ = (
                        outputs  # potentially modify the masses of the barycenter with the true fixed point method
                    )

                # compute the pairwise transport plans and losses
                metric_list = (
                    metric if isinstance(metric, list) else [metric] * n_samples
                )
                list_res = [
                    solve(
                        M=metric_list[k](
                            X_b, X_a_list[k]
                        ).T,  # in the free support setting, the cost matrix is computed from the barycenter to the source distribution, so we transpose it here to be consistent with the inner_solver interface (X_a, X_b,
                        a=a_list[k],
                        b=b,
                        reg=None,
                        unbalanced=None,
                    )
                    for k in range(n_samples)
                ]

                value_linear = sum(
                    w[k] * list_res[k].value_linear for k in range(n_samples)
                )
                res = BaryResult(
                    X=X_b,
                    b=b,
                    value=value_linear,
                    value_linear=value_linear,
                    log=log_,
                    list_res=list_res,
                    backend=nx,
                )
                return res
        else:  # check metric
            if metric not in ["sqeuclidean", "euclidean"]:
                raise NotImplementedError(
                    'Not implemented BCD with closed-form on the barycenter samples with metric="{}"'.format(
                        metric
                    )
                )

            if warmstart:
                # covers exact OT, regularized OT, unbalanced and regularized OT, but not unbalanced OT without regularization (not implemented)
                warmstart_plan = False
                warmstart_potentials = True
                if reg is not None:
                    if (
                        not isinstance(reg_type, tuple)
                        and reg_type.lower() not in ["kl"]
                        and unbalanced_type.lower() != "kl"
                    ) or isinstance(reg_type, tuple):
                        warmstart_plan = True
                        warmstart_potentials = False

            else:
                warmstart_plan = False
                warmstart_potentials = False

            def inner_solver(X_a, X_b, a, b, plan_init, potentials_init):
                return solve_sample(
                    X_a=X_a,
                    X_b=X_b,
                    a=a,
                    b=b,
                    metric=metric,
                    reg=reg,
                    c=c,
                    reg_type=reg_type,
                    unbalanced=unbalanced,
                    unbalanced_type=unbalanced_type,
                    method=method,
                    plan_init=plan_init,
                    potentials_init=potentials_init,
                    verbose=False,
                    **kwargs,
                )

            # compute the barycenter using BCD
            update_masses = unbalanced is not None
            res = _bary_sample_bcd(
                X_a_list,
                X_b_init,
                a_list,
                b,
                w,
                metric,
                inner_solver,
                update_masses,
                warmstart_plan,
                warmstart_potentials,
                stopping_criterion,
                max_iter_bary,
                tol_bary,
                verbose,
                True,  # log set to True by default
                nx,
            )

            return res
