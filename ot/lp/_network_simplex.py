# -*- coding: utf-8 -*-
"""
Solvers for the original linear program OT problem.

"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import warnings
from scipy.sparse import issparse as scipy_issparse

from ..utils import list_to_array, check_number_threads
from ..backend import get_backend
from .emd_wrap import emd_c, emd_c_sparse, check_result


def center_ot_dual(alpha0, beta0, a=None, b=None):
    r"""Center dual OT potentials w.r.t. their weights

    The main idea of this function is to find unique dual potentials
    that ensure some kind of centering/fairness. The main idea is to find dual potentials that lead to the same final objective value for both source and targets (see below for more details). It will help having
    stability when multiple calling of the OT solver with small changes.

    Basically we add another constraint to the potential that will not
    change the objective value but will ensure unicity. The constraint
    is the following:

    .. math::
        \alpha^T \mathbf{a} = \beta^T \mathbf{b}

    in addition to the OT problem constraints.

    since :math:`\sum_i a_i=\sum_j b_j` this can be solved by adding/removing
    a constant from both  :math:`\alpha_0` and :math:`\beta_0`.

    .. math::
        c &= \frac{\beta_0^T \mathbf{b} - \alpha_0^T \mathbf{a}}{\mathbf{1}^T \mathbf{b} + \mathbf{1}^T \mathbf{a}}

        \alpha &= \alpha_0 + c

        \beta &= \beta_0 + c

    Parameters
    ----------
    alpha0 : (ns,) numpy.ndarray, float64
        Source dual potential
    beta0 : (nt,) numpy.ndarray, float64
        Target dual potential
    a : (ns,) numpy.ndarray, float64
        Source histogram (uniform weight if empty list)
    b : (nt,) numpy.ndarray, float64
        Target histogram (uniform weight if empty list)

    Returns
    -------
    alpha : (ns,) numpy.ndarray, float64
        Source centered dual potential
    beta : (nt,) numpy.ndarray, float64
        Target centered dual potential

    """
    # if no weights are provided, use uniform
    if a is None:
        a = np.ones(alpha0.shape[0]) / alpha0.shape[0]
    if b is None:
        b = np.ones(beta0.shape[0]) / beta0.shape[0]

    # compute constant that balances the weighted sums of the duals
    c = (b.dot(beta0) - a.dot(alpha0)) / (a.sum() + b.sum())

    # update duals
    alpha = alpha0 + c
    beta = beta0 - c

    return alpha, beta


def estimate_dual_null_weights(alpha0, beta0, a, b, M):
    r"""Estimate feasible values for 0-weighted dual potentials

    The feasible values are computed efficiently but rather coarsely.

    .. warning::
        This function is necessary because the C++ solver in `emd_c`
        discards all samples in the distributions with
        zeros weights. This means that while the primal variable (transport
        matrix) is exact, the solver only returns feasible dual potentials
        on the samples with weights different from zero.

    First we compute the constraints violations:

    .. math::
        \mathbf{V} = \alpha + \beta^T - \mathbf{M}

    Next we compute the max amount of violation per row (:math:`\alpha`) and
    columns (:math:`beta`)

    .. math::
        \mathbf{v^a}_i = \max_j \mathbf{V}_{i,j}

        \mathbf{v^b}_j = \max_i \mathbf{V}_{i,j}

    Finally we update the dual potential with 0 weights if a
    constraint is violated

    .. math::
        \alpha_i = \alpha_i - \mathbf{v^a}_i \quad \text{ if } \mathbf{a}_i=0 \text{ and } \mathbf{v^a}_i>0

        \beta_j = \beta_j - \mathbf{v^b}_j \quad \text{ if } \mathbf{b}_j=0 \text{ and } \mathbf{v^b}_j > 0

    In the end the dual potentials are centered using function
    :py:func:`ot.lp.center_ot_dual`.

    Note that all those updates do not change the objective value of the
    solution but provide dual potentials that do not violate the constraints.

    Parameters
    ----------
    alpha0 : (ns,) numpy.ndarray, float64
        Source dual potential
    beta0 : (nt,) numpy.ndarray, float64
        Target dual potential
    alpha0 : (ns,) numpy.ndarray, float64
        Source dual potential
    beta0 : (nt,) numpy.ndarray, float64
        Target dual potential
    a : (ns,) numpy.ndarray, float64
        Source distribution (uniform weights if empty list)
    b : (nt,) numpy.ndarray, float64
        Target distribution (uniform weights if empty list)
    M : (ns,nt) numpy.ndarray, float64
        Loss matrix (c-order array with type float64)

    Returns
    -------
    alpha : (ns,) numpy.ndarray, float64
        Source corrected dual potential
    beta : (nt,) numpy.ndarray, float64
        Target corrected dual potential

    """

    # binary indexing of non-zeros weights
    asel = a != 0
    bsel = b != 0

    # compute dual constraints violation
    constraint_violation = alpha0[:, None] + beta0[None, :] - M

    # Compute largest violation per line and columns
    aviol = np.max(constraint_violation, 1)
    bviol = np.max(constraint_violation, 0)

    # update corrects violation of
    alpha_up = -1 * ~asel * np.maximum(aviol, 0)
    beta_up = -1 * ~bsel * np.maximum(bviol, 0)

    alpha = alpha0 + alpha_up
    beta = beta0 + beta_up

    return center_ot_dual(alpha, beta, a, b)


def emd(
    a,
    b,
    M,
    numItermax=100000,
    log=False,
    center_dual=True,
    numThreads=1,
    check_marginals=True,
):
    r"""Solves the Earth Movers distance problem and returns the OT matrix


    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights

    .. warning:: Note that the :math:`\mathbf{M}` matrix in numpy needs to be a C-order
        numpy.array in float64 format. It will be converted if not in this
        format

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.

    .. note:: This function will cast the computed transport plan to the data type
        of the provided input with the following priority: :math:`\mathbf{a}`,
        then :math:`\mathbf{b}`, then :math:`\mathbf{M}` if marginals are not provided.
        Casting to an integer tensor might result in a loss of precision.
        If this behaviour is unwanted, please make sure to provide a
        floating point input.

    .. note:: An error will be raised if the vectors :math:`\mathbf{a}` and :math:`\mathbf{b}` do not sum to the same value.

    Uses the algorithm proposed in :ref:`[1] <references-emd>`.

    Parameters
    ----------
    a : (ns,) array-like, float
        Source histogram (uniform weight if empty list)
    b : (nt,) array-like, float
        Target histogram (uniform weight if empty list)
    M : (ns,nt) array-like or sparse matrix, float
        Loss matrix. Can be:

        - Dense array (c-order array in numpy with type float64)
        - Sparse matrix in backend's format (scipy.sparse.coo_matrix for NumPy backend,
          torch.sparse_coo_tensor for PyTorch backend, etc.)

    numItermax : int, optional (default=100000)
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.
    log: bool, optional (default=False)
        If True, returns a dictionary containing the cost and dual variables.
        Otherwise returns only the optimal transportation matrix.
    center_dual: boolean, optional (default=True)
        If True, centers the dual potential using function
        :py:func:`ot.lp.center_ot_dual`.
    numThreads: int or "max", optional (default=1, i.e. OpenMP is not used)
        If compiled with OpenMP, chooses the number of threads to parallelize.
        "max" selects the highest number possible.
    check_marginals: bool, optional (default=True)
        If True, checks that the marginals mass are equal. If False, skips the
        check.

    .. note:: The solver automatically detects sparse format using the backend's
        :py:meth:`issparse` method. For sparse inputs:

        - Uses a memory-efficient sparse EMD algorithm
        - Returns the transport plan as a sparse matrix in the backend's format
        - Supports scipy.sparse (NumPy), torch.sparse (PyTorch), etc.
        - JAX and TensorFlow backends don't support sparse matrices


    Returns
    -------
    gamma: array-like or sparse matrix, shape (ns, nt)
        Optimal transportation matrix for the given parameters.

        - For dense inputs: returns a dense array
        - For sparse inputs: returns a sparse matrix in the backend's format
          (e.g., scipy.sparse.coo_matrix for NumPy, torch.sparse_coo_tensor for PyTorch)

    log: dict, optional
        If input log is True, a dictionary containing the cost, dual variables,
        and exit status.


    Examples
    --------

    Simple example with obvious solution. The function emd accepts lists and
    perform automatic conversion to numpy arrays

    >>> import ot
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ot.emd(a, b, M)
    array([[0.5, 0. ],
           [0. , 0.5]])


    .. _references-emd:
    References
    ----------
    .. [1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W. (2011,
        December).  Displacement interpolation using Lagrangian mass transport.
        In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p. 158). ACM.

    See Also
    --------
    ot.bregman.sinkhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT
    """

    edge_sources = None
    edge_targets = None
    edge_costs = None
    n1, n2 = None, None

    # Get backend to check if M is sparse
    a, b = list_to_array(a, b)
    nx = get_backend(a, b)

    # Check if M is sparse (either backend sparse or scipy.sparse)
    is_sparse = nx.issparse(M) or scipy_issparse(M)

    if is_sparse:
        # Check if backend supports sparse matrices
        backend_name = nx.__class__.__name__
        if backend_name in ["JaxBackend", "TensorflowBackend"]:
            raise NotImplementedError(
                f"Sparse optimal transport is not supported for {backend_name}. "
                "JAX does not have native sparse matrix support, and TensorFlow's "
                "sparse implementation is incomplete. Please convert your sparse "
                "matrix to dense format using M.toarray() or equivalent before calling emd()."
            )

        # Extract COO data using backend method - returns numpy arrays
        edge_sources, edge_targets, edge_costs, (n1, n2) = nx.sparse_coo_data(M)

        # Ensure correct dtypes for C++ solver
        if edge_sources.dtype != np.uint64:
            edge_sources = edge_sources.astype(np.uint64)
        if edge_targets.dtype != np.uint64:
            edge_targets = edge_targets.astype(np.uint64)
        if edge_costs.dtype != np.float64:
            edge_costs = edge_costs.astype(np.float64)

    elif isinstance(M, tuple):
        raise ValueError(
            "Tuple format for sparse cost matrix is not supported. "
            "Please use backend-appropriate sparse COO format (e.g., scipy.sparse.coo_matrix, torch.sparse_coo_tensor, etc.)."
        )
    else:
        is_sparse = False
        a, b, M = list_to_array(a, b, M)

    if len(a) != 0:
        type_as = a
    elif len(b) != 0:
        type_as = b
    else:
        type_as = a

    # Set n1, n2 if not already set (dense case)
    if n1 is None:
        n1, n2 = M.shape

    if len(a) == 0:
        a = nx.ones((n1,), type_as=type_as) / n1
    if len(b) == 0:
        b = nx.ones((n2,), type_as=type_as) / n2

    if is_sparse:
        a, b = nx.to_numpy(a, b)
    else:
        M, a, b = nx.to_numpy(M, a, b)
        M = np.asarray(M, dtype=np.float64, order="C")

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    assert (
        a.shape[0] == n1 and b.shape[0] == n2
    ), "Dimension mismatch, check dimensions of M with a and b"

    # ensure that same mass
    if check_marginals:
        np.testing.assert_almost_equal(
            a.sum(0),
            b.sum(0),
            err_msg="a and b vector must have the same sum",
            decimal=6,
        )
    b = b * a.sum() / b.sum()

    asel = a != 0
    bsel = b != 0

    numThreads = check_number_threads(numThreads)

    # ============================================================================
    # CALL SOLVER (sparse or dense)
    # ============================================================================
    if is_sparse:
        # Sparse solver - never build full matrix
        flow_sources, flow_targets, flow_values, cost, u, v, result_code = emd_c_sparse(
            a, b, edge_sources, edge_targets, edge_costs, numItermax
        )
    else:
        # Dense solver
        G, cost, u, v, result_code = emd_c(a, b, M, numItermax, numThreads)

    # ============================================================================
    # POST-PROCESS DUAL VARIABLES AND CREATE TRANSPORT PLAN
    # ============================================================================

    # Center dual potentials
    if center_dual:
        u, v = center_ot_dual(u, v, a, b)

    # Handle null weights
    if np.any(~asel) or np.any(~bsel):
        if is_sparse:
            u, v = center_ot_dual(u, v, a, b)
        else:
            u, v = estimate_dual_null_weights(u, v, a, b, M)

    result_code_string = check_result(result_code)

    # Create transport plan in backend format
    if is_sparse:
        # Convert flow to sparse matrix using backend's coo_matrix method
        flow_values_backend = nx.from_numpy(flow_values, type_as=type_as)
        flow_sources_backend = nx.from_numpy(
            flow_sources.astype(np.int64), type_as=type_as
        )
        flow_targets_backend = nx.from_numpy(
            flow_targets.astype(np.int64), type_as=type_as
        )

        G = nx.coo_matrix(
            flow_values_backend,
            flow_sources_backend,
            flow_targets_backend,
            shape=(n1, n2),
            type_as=type_as,
        )
    else:
        # Warn about integer casting for dense case
        if not nx.is_floating_point(type_as):
            warnings.warn(
                "Input histogram consists of integer. The transport plan will be "
                "casted accordingly, possibly resulting in a loss of precision. "
                "If this behaviour is unwanted, please make sure your input "
                "histogram consists of floating point elements.",
                stacklevel=2,
            )
        G = nx.from_numpy(G, type_as=type_as)

    # Return results
    if log:
        log_dict = {
            "cost": cost,
            "u": nx.from_numpy(u, type_as=type_as),
            "v": nx.from_numpy(v, type_as=type_as),
            "warning": result_code_string,
            "result_code": result_code,
        }
        return G, log_dict
    else:
        return G


def emd2(
    a,
    b,
    M,
    processes=1,
    numItermax=100000,
    log=False,
    center_dual=True,
    numThreads=1,
    check_marginals=True,
    return_matrix=False,
):
    r"""Solves the Earth Movers distance problem and returns the loss

    .. math::
        \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.

    .. note:: This function will cast the computed transport plan and
        transportation loss to the data type of the provided input with the
        following priority: :math:`\mathbf{a}`, then :math:`\mathbf{b}`,
        then :math:`\mathbf{M}` if marginals are not provided.
        Casting to an integer tensor might result in a loss of precision.
        If this behaviour is unwanted, please make sure to provide a
        floating point input.

    .. note:: An error will be raised if the vectors :math:`\mathbf{a}` and :math:`\mathbf{b}` do not sum to the same value.

    Uses the algorithm proposed in :ref:`[1] <references-emd2>`.

    Parameters
    ----------
    a : (ns,) array-like, float64
        Source histogram (uniform weight if empty list)
    b : (nt,) array-like, float64
        Target histogram (uniform weight if empty list)
    M : (ns,nt) array-like or sparse matrix, float64
        Loss matrix. Can be:

        - Dense array (c-order array in numpy with type float64)
        - Sparse matrix in backend's format (scipy.sparse.coo_matrix for NumPy backend,
          torch.sparse_coo_tensor for PyTorch backend, etc.)

    processes : int, optional (default=1)
        Nb of processes used for multiple emd computation (deprecated)
    numItermax : int, optional (default=100000)
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.
    log: boolean, optional (default=False)
        If True, returns a dictionary containing dual
        variables. Otherwise returns only the optimal transportation cost.
    return_matrix: boolean, optional (default=False)
        If True, returns the optimal transportation matrix in the log.
    center_dual: boolean, optional (default=True)
        If True, centers the dual potential using function
        :py:func:`ot.lp.center_ot_dual`.
    numThreads: int or "max", optional (default=1, i.e. OpenMP is not used)
        If compiled with OpenMP, chooses the number of threads to parallelize.
        "max" selects the highest number possible.
    check_marginals: bool, optional (default=True)
        If True, checks that the marginals mass are equal. If False, skips the
        check.

    .. note:: The solver automatically detects sparse format using the backend's
        :py:meth:`issparse` method. For sparse inputs:

        - Uses a memory-efficient sparse EMD algorithm
        - Edges not included are treated as having infinite cost (forbidden)
        - Supports scipy.sparse (NumPy), torch.sparse (PyTorch), etc.
        - JAX and TensorFlow backends don't support sparse matrices


    Returns
    -------
    W: float, array-like
        Optimal transportation loss for the given parameters
    log: dict
        If input log is true, a dictionary containing dual
        variables and exit status


    Examples
    --------

    Simple example with obvious solution. The function emd accepts lists and
    perform automatic conversion to numpy arrays


    >>> import ot
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ot.emd2(a,b,M)
    0.0


    .. _references-emd2:
    References
    ----------
    .. [1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W.
        (2011, December).  Displacement interpolation using Lagrangian mass
        transport. In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p.
        158). ACM.

    See Also
    --------
    ot.bregman.sinkhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT
    """

    edge_sources = None
    edge_targets = None
    edge_costs = None
    n1, n2 = None, None

    # Get backend to check if M is sparse
    a, b = list_to_array(a, b)
    nx = get_backend(a, b)

    # Check if M is sparse (either backend sparse or scipy.sparse)
    from scipy.sparse import issparse as scipy_issparse

    is_sparse = nx.issparse(M) or scipy_issparse(M)

    if is_sparse:
        # Check if backend supports sparse matrices
        backend_name = nx.__class__.__name__
        if backend_name in ["JaxBackend", "TensorflowBackend"]:
            raise NotImplementedError(
                f"Sparse optimal transport is not supported for {backend_name}. "
                "JAX does not have native sparse matrix support, and TensorFlow's "
                "sparse implementation is incomplete. Please convert your sparse "
                "matrix to dense format using M.toarray() or equivalent before calling emd2()."
            )

        # Extract COO data using backend method - returns numpy arrays
        edge_sources, edge_targets, edge_costs, (n1, n2) = nx.sparse_coo_data(M)

        # Ensure correct dtypes for C++ solver
        if edge_sources.dtype != np.uint64:
            edge_sources = edge_sources.astype(np.uint64)
        if edge_targets.dtype != np.uint64:
            edge_targets = edge_targets.astype(np.uint64)
        if edge_costs.dtype != np.float64:
            edge_costs = edge_costs.astype(np.float64)

    elif isinstance(M, tuple):
        raise ValueError(
            "Tuple format for sparse cost matrix is not supported. "
            "Please use backend-appropriate sparse COO format (e.g., scipy.sparse.coo_matrix, torch.sparse_coo_tensor, etc.)."
        )
    else:
        # Dense matrix
        is_sparse = False
        a, b, M = list_to_array(a, b, M)

    if len(a) != 0:
        type_as = a
    elif len(b) != 0:
        type_as = b
    else:
        type_as = a  # Can't use M for sparse case

    # Set n1, n2 if not already set (dense case)
    if n1 is None:
        n1, n2 = M.shape

    # if empty array given then use uniform distributions
    if len(a) == 0:
        a = nx.ones((n1,), type_as=type_as) / n1
    if len(b) == 0:
        b = nx.ones((n2,), type_as=type_as) / n2

    a0, b0 = a, b
    M0 = None if is_sparse else M

    if is_sparse:
        edge_costs_original = nx.from_numpy(edge_costs, type_as=type_as)
    else:
        edge_costs_original = None

    if is_sparse:
        a, b = nx.to_numpy(a, b)
    else:
        M, a, b = nx.to_numpy(M, a, b)
        M = np.asarray(M, dtype=np.float64, order="C")

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    assert (
        a.shape[0] == n1 and b.shape[0] == n2
    ), "Dimension mismatch, check dimensions of M with a and b"

    # ensure that same mass
    if check_marginals:
        np.testing.assert_almost_equal(
            a.sum(0),
            b.sum(0, keepdims=True),
            err_msg="a and b vector must have the same sum",
            decimal=6,
        )
    b = b * a.sum(0) / b.sum(0, keepdims=True)

    asel = a != 0

    numThreads = check_number_threads(numThreads)

    # ============================================================================
    # DEFINE SOLVER FUNCTION (works for both sparse and dense)
    # ============================================================================
    def f(b):
        bsel = b != 0

        # Call appropriate solver
        if is_sparse:
            # Solve sparse EMD
            flow_sources, flow_targets, flow_values, cost, u, v, result_code = (
                emd_c_sparse(a, b, edge_sources, edge_targets, edge_costs, numItermax)
            )
        else:
            # Solve dense EMD
            G, cost, u, v, result_code = emd_c(a, b, M, numItermax, numThreads)

        # Center dual potentials
        if center_dual:
            u, v = center_ot_dual(u, v, a, b)

        # Handle null weights
        if np.any(~asel) or np.any(~bsel):
            if is_sparse:
                u, v = center_ot_dual(u, v, a, b)
            else:
                u, v = estimate_dual_null_weights(u, v, a, b, M)

        # Prepare cost with gradients
        if is_sparse:
            # Build gradient mapping for sparse case
            edge_to_idx = {
                (edge_sources[k], edge_targets[k]): k for k in range(len(edge_sources))
            }

            grad_edge_costs = np.zeros(len(edge_costs), dtype=np.float64)
            for idx in range(len(flow_sources)):
                src, tgt, flow = flow_sources[idx], flow_targets[idx], flow_values[idx]
                edge_idx = edge_to_idx.get((src, tgt), -1)
                if edge_idx >= 0:
                    grad_edge_costs[edge_idx] = flow

            cost = nx.set_gradients(
                nx.from_numpy(cost, type_as=type_as),
                (a0, b0, edge_costs_original),
                (
                    nx.from_numpy(u - np.mean(u), type_as=type_as),
                    nx.from_numpy(v - np.mean(v), type_as=type_as),
                    nx.from_numpy(grad_edge_costs, type_as=type_as),
                ),
            )
        else:
            # Dense case: warn about integer casting
            if not nx.is_floating_point(type_as):
                warnings.warn(
                    "Input histogram consists of integer. The transport plan will be "
                    "casted accordingly, possibly resulting in a loss of precision. "
                    "If this behaviour is unwanted, please make sure your input "
                    "histogram consists of floating point elements.",
                    stacklevel=2,
                )

            G_backend = nx.from_numpy(G, type_as=type_as)
            cost = nx.set_gradients(
                nx.from_numpy(cost, type_as=type_as),
                (a0, b0, M0),
                (
                    nx.from_numpy(u - np.mean(u), type_as=type_as),
                    nx.from_numpy(v - np.mean(v), type_as=type_as),
                    G_backend,
                ),
            )

        check_result(result_code)

        # Return results
        if log or return_matrix:
            log_dict = {
                "u": nx.from_numpy(u, type_as=type_as),
                "v": nx.from_numpy(v, type_as=type_as),
                "warning": check_result(result_code),
                "result_code": result_code,
            }

            if return_matrix:
                if is_sparse:
                    G = np.zeros((len(a), len(b)), dtype=np.float64)
                    G[flow_sources, flow_targets] = flow_values
                    log_dict["G"] = nx.from_numpy(G, type_as=type_as)
                else:
                    log_dict["G"] = G_backend

            return [cost, log_dict]
        else:
            return cost

    if len(b.shape) == 1:
        return f(b)
    nb = b.shape[1]

    if processes > 1:
        warnings.warn(
            "The 'processes' parameter has been deprecated. "
            "Multiprocessing should be done outside of POT."
        )
    res = list(map(f, [b[:, i].copy() for i in range(nb)]))

    return res
