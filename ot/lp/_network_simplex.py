# -*- coding: utf-8 -*-
"""
Solvers for the original linear program OT problem.

"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import warnings

from ..utils import list_to_array, check_number_threads
from ..backend import get_backend
from .emd_wrap import emd_c, emd_c_sparse, emd_c_lazy, check_result


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


def _compute_active_subset(a, b, M, row_mask, col_mask):
    """Return filtered inputs restricted to the rows/columns with mass."""
    need_filter = np.any(~row_mask) or np.any(~col_mask)
    if not need_filter:
        return need_filter, None, None, a, b, M

    row_idx = np.flatnonzero(row_mask)
    col_idx = np.flatnonzero(col_mask)
    M_solver = np.asarray(M[np.ix_(row_idx, col_idx)], dtype=np.float64, order="C")
    return need_filter, row_idx, col_idx, a[row_idx], b[col_idx], M_solver


def _inflate_dense_solution(
    G, u, v, need_filter, row_idx, col_idx, row_mask, col_mask, n_rows, n_cols
):
    """Embed the filtered dense solution back into the full support."""
    if not need_filter:
        return G, u, v

    G_full = np.zeros((n_rows, n_cols), dtype=G.dtype)
    G_full[np.ix_(row_idx, col_idx)] = G

    u_full = np.zeros((n_rows,), dtype=u.dtype)
    v_full = np.zeros((n_cols,), dtype=v.dtype)
    u_full[row_mask] = u
    v_full[col_mask] = v
    return G_full, u_full, v_full


def _prepare_warmstart(potentials_init, need_filter, row_mask, col_mask):
    """Return warm-start potentials filtered to the active support."""
    if potentials_init is None:
        return None, None

    alpha_init, beta_init = potentials_init
    alpha_init = np.asarray(alpha_init, dtype=np.float64)
    beta_init = np.asarray(beta_init, dtype=np.float64)
    if need_filter:
        alpha_init = alpha_init[row_mask]
        beta_init = beta_init[col_mask]
    return alpha_init, beta_init


def emd(
    a,
    b,
    M,
    numItermax=100000,
    log=False,
    center_dual=True,
    numThreads=1,
    check_marginals=True,
    potentials_init=None,
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
    potentials_init: tuple of two arrays (alpha, beta), optional (default=None)
        Warmstart dual potentials to accelerate convergence. Should be a tuple
        (alpha, beta) where alpha is shape (ns,) and beta is shape (nt,).
        These potentials are used to guide initial pivots in the network simplex.
        Typically obtained from a previous EMD solve or Sinkhorn approximation.

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

    n1, n2 = None, None

    # Convert lists to arrays, using M to detect backend when a,b are empty
    a, b, M = list_to_array(a, b, M)
    nx = get_backend(a, b, M)

    # Check if M is sparse using backend's issparse method
    is_sparse = nx.issparse(M)

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

    if a is not None and len(a) != 0:
        type_as = a
    elif b is not None and len(b) != 0:
        type_as = b
    else:
        type_as = a if a is not None else b

    # Set n1, n2 if not already set (dense case)
    if n1 is None:
        n1, n2 = M.shape

    if a is None or len(a) == 0:
        a = nx.ones((n1,), type_as=type_as) / n1
    if b is None or len(b) == 0:
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
        row_mask = asel
        col_mask = bsel
        need_filter = np.any(~row_mask) or np.any(~col_mask)

        if need_filter:
            row_idx = np.flatnonzero(row_mask)
            col_idx = np.flatnonzero(col_mask)
            a_solver = a[row_idx]
            b_solver = b[col_idx]
            M_solver = M[np.ix_(row_idx, col_idx)]
            M_solver = np.asarray(M_solver, dtype=np.float64, order="C")
        else:
            row_idx = None
            col_idx = None
            a_solver = a
            b_solver = b
            M_solver = M

        # Prepare warmstart if provided
        alpha_init = None
        beta_init = None
        if potentials_init is not None:
            alpha_init, beta_init = potentials_init
            alpha_init = np.asarray(alpha_init, dtype=np.float64)
            beta_init = np.asarray(beta_init, dtype=np.float64)
            if need_filter:
                alpha_init = alpha_init[row_mask]
                beta_init = beta_init[col_mask]

        # Dense solver
        G, cost, u, v, result_code = emd_c(
            a_solver, b_solver, M_solver, numItermax, numThreads, alpha_init, beta_init
        )

        if need_filter:
            G_full = np.zeros((n1, n2), dtype=G.dtype)
            G_full[np.ix_(row_idx, col_idx)] = G
            G = G_full

            u_full = np.zeros((n1,), dtype=u.dtype)
            v_full = np.zeros((n2,), dtype=v.dtype)
            u_full[row_mask] = u
            v_full[col_mask] = v
            u, v = u_full, v_full

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
    return_matrix=False,
    center_dual=True,
    numThreads=1,
    check_marginals=True,
    potentials_init=None,
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
        following priority : :math:`\mathbf{a}`, then :math:`\mathbf{b}`,
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
        If True, returns a dictionary containing the cost and dual
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
    potentials_init: tuple of two arrays (alpha, beta), optional (default=None)
        Warmstart dual potentials to accelerate convergence. Should be a tuple
        (alpha, beta) where alpha is shape (ns,) and beta is shape (nt,).
        These potentials are used to guide initial pivots in the network simplex.
        Typically obtained from a previous EMD solve or Sinkhorn approximation.

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
        If input log is true, a dictionary containing the cost, dual
        variables (u, v), exit status, and optionally the optimal
        transportation matrix (G) if return_matrix is True


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

    n1, n2 = None, None

    a, b, M = list_to_array(a, b, M)
    nx = get_backend(a, b, M)

    # Check if M is sparse using backend's issparse method
    is_sparse = nx.issparse(M)

    # Save original sparse tensor for gradient tracking (before conversion to numpy)
    M_original_sparse = None

    if is_sparse:
        # Check if backend supports sparse matrices
        backend_name = nx.__class__.__name__
        if backend_name in ["JaxBackend", "TensorflowBackend"]:
            raise NotImplementedError()

        # Save original M for gradient tracking (before numpy conversion)
        M_original_sparse = M

        edge_sources, edge_targets, edge_costs, (n1, n2) = nx.sparse_coo_data(M)

        if edge_sources.dtype != np.uint64:
            edge_sources = edge_sources.astype(np.uint64)
        if edge_targets.dtype != np.uint64:
            edge_targets = edge_targets.astype(np.uint64)
        if edge_costs.dtype != np.float64:
            edge_costs = edge_costs.astype(np.float64)

    if a is not None and len(a) != 0:
        type_as = a
    elif b is not None and len(b) != 0:
        type_as = b
    else:
        type_as = a if a is not None else b

    # Set n1, n2 if not already set (dense case)
    if n1 is None:
        n1, n2 = M.shape

    # if empty array given then use uniform distributions
    if a is None or len(a) == 0:
        a = nx.ones((n1,), type_as=type_as) / n1
    if b is None or len(b) == 0:
        b = nx.ones((n2,), type_as=type_as) / n2

    a0, b0 = a, b
    M0 = None if is_sparse else M

    if is_sparse:
        # Use the original sparse tensor (preserves gradients for PyTorch)
        edge_costs_original = M_original_sparse
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
    # DEFINE SOLVER FUNCTION
    # ============================================================================
    def f(b):
        bsel = b != 0

        if is_sparse:
            # Solve sparse EMD
            flow_sources, flow_targets, flow_values, cost, u, v, result_code = (
                emd_c_sparse(a, b, edge_sources, edge_targets, edge_costs, numItermax)
            )
        else:
            (
                need_filter,
                row_idx,
                col_idx,
                a_solver,
                b_solver,
                M_solver,
            ) = _compute_active_subset(a, b, M, asel, bsel)

            alpha_init, beta_init = _prepare_warmstart(
                potentials_init, need_filter, asel, bsel
            )

            # Solve dense EMD
            G, cost, u, v, result_code = emd_c(
                a_solver,
                b_solver,
                M_solver,
                numItermax,
                numThreads,
                alpha_init,
                beta_init,
            )

            G, u, v = _inflate_dense_solution(
                G, u, v, need_filter, row_idx, col_idx, asel, bsel, n1, n2
            )

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

            # Convert gradient to sparse format matching edge_costs_original
            grad_edge_costs_backend = nx.from_numpy(grad_edge_costs, type_as=type_as)
            if nx.issparse(edge_costs_original):
                # Reconstruct sparse gradient tensor with same structure as original
                grad_M_sparse = nx.coo_matrix(
                    grad_edge_costs_backend,
                    nx.from_numpy(edge_sources.astype(np.int64), type_as=type_as),
                    nx.from_numpy(edge_targets.astype(np.int64), type_as=type_as),
                    shape=(n1, n2),
                    type_as=type_as,
                )
            else:
                grad_M_sparse = grad_edge_costs_backend

            cost = nx.set_gradients(
                nx.from_numpy(cost, type_as=type_as),
                (a0, b0, edge_costs_original),
                (
                    nx.from_numpy(u - np.mean(u), type_as=type_as),
                    nx.from_numpy(v - np.mean(v), type_as=type_as),
                    grad_M_sparse,
                ),
            )

            # Build transport plan in backend sparse format
            flow_values_backend = nx.from_numpy(flow_values, type_as=type_as)
            flow_sources_backend = nx.from_numpy(
                flow_sources.astype(np.int64), type_as=type_as
            )
            flow_targets_backend = nx.from_numpy(
                flow_targets.astype(np.int64), type_as=type_as
            )

            G_backend = nx.coo_matrix(
                flow_values_backend,
                flow_sources_backend,
                flow_targets_backend,
                shape=(n1, n2),
                type_as=type_as,
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
                "cost": cost,
                "u": nx.from_numpy(u, type_as=type_as),
                "v": nx.from_numpy(v, type_as=type_as),
                "warning": check_result(result_code),
                "result_code": result_code,
            }
            if return_matrix:
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


def emd2_lazy(
    X_a,
    X_b,
    a=None,
    b=None,
    metric="sqeuclidean",
    numItermax=100000,
    log=False,
    return_matrix=True,
    center_dual=True,
    check_marginals=True,
    potentials_init=None,
):
    r"""Solves the Earth Movers distance problem with lazy cost computation and returns the loss

    .. math::
        \min_\gamma \quad \langle \gamma, \mathbf{M}(\mathbf{X}_a, \mathbf{X}_b) \rangle_F

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    where :

    - :math:`\mathbf{M}(\mathbf{X}_a, \mathbf{X}_b)` is computed on-the-fly from coordinates
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights

    .. note:: This function computes distances on-the-fly during the network simplex algorithm,
        avoiding the O(ns*nt) memory cost of pre-computing the full cost matrix. Memory usage
        is O(ns+nt) instead.

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.

    Parameters
    ----------
    X_a : (ns, dim) array-like, float64
        Source sample coordinates
    X_b : (nt, dim) array-like, float64
        Target sample coordinates
    a : (ns,) array-like, float64, optional
        Source histogram (uniform weight if None)
    b : (nt,) array-like, float64, optional
        Target histogram (uniform weight if None)
    metric : str, optional (default='sqeuclidean')
        Distance metric for cost computation. Options:

        - 'sqeuclidean': Squared Euclidean distance
        - 'euclidean': Euclidean distance
        - 'cityblock': Manhattan/L1 distance

    numItermax : int, optional (default=100000)
        Maximum number of iterations before stopping if not converged
    log: boolean, optional (default=False)
        If True, returns a dictionary containing the cost, dual variables,
        and optionally the transport plan (sparse format)
    return_matrix: boolean, optional (default=False)
        If True, returns the optimal transportation matrix in the log (sparse format)
    center_dual: boolean, optional (default=True)
        If True, centers the dual potential using :py:func:`ot.lp.center_ot_dual`
    check_marginals: bool, optional (default=True)
        If True, checks that the marginals mass are equal
    potentials_init : tuple of (ns,) and (nt,) arrays, optional
        Initial dual potentials (u, v) to warmstart the solver. If provided,
        the solver starts from these potentials instead of a cold start.

    Returns
    -------
    W: float
        Optimal transportation loss
    log: dict
        If input log is True, a dictionary containing:

        - cost: the optimal transportation cost
        - u, v: dual variables
        - warning: solver status message
        - result_code: solver return code
        - G: (optional) sparse transport plan if return_matrix=True

    See Also
    --------
    ot.emd2 : EMD with pre-computed cost matrix
    ot.lp.emd_c_lazy : Low-level C++ lazy solver
    """

    a, b, X_a, X_b = list_to_array(a, b, X_a, X_b)
    nx = get_backend(a, b, X_a, X_b)

    n1, n2 = X_a.shape[0], X_b.shape[0]

    if X_a.shape[1] != X_b.shape[1]:
        raise ValueError(
            f"X_a and X_b must have the same number of dimensions, "
            f"got {X_a.shape[1]} and {X_b.shape[1]}"
        )

    if a is not None and len(a) != 0:
        type_as = a
    elif b is not None and len(b) != 0:
        type_as = b
    else:
        type_as = X_a

    if a is None or len(a) == 0:
        a = nx.ones((n1,), type_as=type_as) / n1
    if b is None or len(b) == 0:
        b = nx.ones((n2,), type_as=type_as) / n2

    a0, b0 = a, b

    # Convert to numpy for C++ backend
    X_a_np = nx.to_numpy(X_a)
    X_b_np = nx.to_numpy(X_b)
    a_np = nx.to_numpy(a)
    b_np = nx.to_numpy(b)

    X_a_np = np.asarray(X_a_np, dtype=np.float64, order="C")
    X_b_np = np.asarray(X_b_np, dtype=np.float64, order="C")
    a_np = np.asarray(a_np, dtype=np.float64)
    b_np = np.asarray(b_np, dtype=np.float64)

    assert (
        a_np.shape[0] == n1 and b_np.shape[0] == n2
    ), "Dimension mismatch, check dimensions of X_a/X_b with a and b"

    if check_marginals:
        np.testing.assert_almost_equal(
            a_np.sum(),
            b_np.sum(),
            err_msg="a and b vector must have the same sum",
            decimal=6,
        )
    b_np = b_np * a_np.sum() / b_np.sum()

    # Handle warmstart potentials
    alpha_init_np = None
    beta_init_np = None
    if potentials_init is not None:
        alpha_init, beta_init = potentials_init
        alpha_init_np = nx.to_numpy(alpha_init)
        beta_init_np = nx.to_numpy(beta_init)
        alpha_init_np = np.asarray(alpha_init_np, dtype=np.float64, order="C")
        beta_init_np = np.asarray(beta_init_np, dtype=np.float64, order="C")

    G, cost, u, v, result_code = emd_c_lazy(
        a_np, b_np, X_a_np, X_b_np, metric, numItermax, alpha_init_np, beta_init_np
    )

    if center_dual:
        u, v = center_ot_dual(u, v, a_np, b_np)

    if not nx.is_floating_point(type_as):
        warnings.warn(
            "Input histogram consists of integer. The transport plan will be "
            "casted accordingly, possibly resulting in a loss of precision. "
            "If this behaviour is unwanted, please make sure your input "
            "histogram consists of floating point elements.",
            stacklevel=2,
        )

    G_backend = nx.from_numpy(G, type_as=type_as)

    cost_backend = nx.set_gradients(
        nx.from_numpy(cost, type_as=type_as),
        (a0, b0),
        (
            nx.from_numpy(u - np.mean(u), type_as=type_as),
            nx.from_numpy(v - np.mean(v), type_as=type_as),
        ),
    )

    check_result(result_code)

    if log or return_matrix:
        log_dict = {
            "cost": cost_backend,
            "u": nx.from_numpy(u, type_as=type_as),
            "v": nx.from_numpy(v, type_as=type_as),
            "warning": check_result(result_code),
            "result_code": result_code,
        }
        if return_matrix:
            log_dict["G"] = G_backend
        return cost_backend, log_dict
    else:
        return cost_backend
