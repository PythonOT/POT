"""
Exact EMD-L1 solver for histograms sharing a Cartesian grid support.
"""

# Author: Tom Vercauteren
#
# License: MIT License

import numpy as np

from ..backend import get_backend
from ..utils import list_to_array
from .emd_wrap import check_result, emd_1d_sorted, emd_c_grid_l1

# Mirrors ot::ProblemType in ot/lp/EMD.h (INFEASIBLE=0, OPTIMAL=1, ...). Not
# otherwise exposed to Python, since it is a Cython ``cdef enum``.
_RESULT_INFEASIBLE = 0
_RESULT_OPTIMAL = 1

_EMPTY_U64 = np.empty(0, dtype=np.uint64)
_EMPTY_F64 = np.empty(0, dtype=np.float64)


def _finalize_native_cost(cost, nx):
    """For the numpy backend, ``nx.sum`` etc. return a numpy scalar (e.g.
    ``np.float64``), whereas every other POT solver returns a plain Python
    float in that case (e.g. numpy scalars repr as ``np.float64(3.0)``,
    breaking naive equality/display expectations). Normalize to a plain
    float there; other backends keep their native scalar tensor as is.
    """
    if isinstance(cost, np.generic):
        return float(cost)
    return nx.detach(cost)


def _emd_grid_l1_1d_cost(A, B, nx):
    r"""Closed-form, fully backend-native cost for a 1D grid.

    A 1D grid with a shared, sorted, unit-spaced support has a classic
    :math:`\mathcal{O}(n)` closed form for the L1 (cityblock) Wasserstein
    cost: the L1 norm of the difference of cumulative sums. Every step is a
    generic backend reduction (``nx.cumsum``, ``nx.abs``, ``nx.sum``), so
    this never leaves the device `A`/`B` are already on -- unlike every
    other path in :any:`emd_grid_l1`, which needs a CPU round-trip.

    Returns ``(cost, result_code)``; `cost` is already a backend-native
    scalar with `A`'s dtype and device, detached from any autodiff graph
    (see :any:`_finalize_native_cost`).
    """
    if A.shape[0] == 0 or nx.any(A < 0) or nx.any(B < 0):
        return _finalize_native_cost(0.0 * nx.sum(A), nx), _RESULT_INFEASIBLE

    total_a = nx.sum(A)
    total_b = nx.sum(B)
    if abs(float(nx.to_numpy(total_a)) - float(nx.to_numpy(total_b))) > 1e-8 * max(
        1.0, float(nx.to_numpy(total_a))
    ):
        return _finalize_native_cost(0.0 * total_a, nx), _RESULT_INFEASIBLE

    cum_diff = nx.cumsum(A, 0) - nx.cumsum(B, 0)
    cost = nx.sum(nx.abs(cum_diff[:-1]))
    return _finalize_native_cost(cost, nx), _RESULT_OPTIMAL


def _emd_grid_l1_1d_plan(A, B, nx):
    r"""Cost and sparse transportation plan for a 1D grid.

    A 1D grid is just a sorted, shared support, for which POT already has an
    exact :math:`\mathcal{O}(n)` solver (:any:`ot.lp.emd_1d_sorted`, the same
    routine backing :any:`ot.lp.emd_1d`). Calling it directly here, with the
    grid's integer positions passed in as already sorted, skips both the
    network simplex setup of the general grid solver and the argsort/pairwise
    distance overhead that the generic, arbitrary-support :any:`ot.lp.emd_1d`
    would otherwise pay.

    Unlike :any:`_emd_grid_l1_1d_cost`, recovering the actual coupling needs
    the compiled O(n) merge behind :any:`ot.lp.emd_1d_sorted`, which requires
    plain CPU numpy arrays -- so, backend-compatible via `nx`, this converts
    `A`/`B` right here, immediately before that one call.
    """
    a = np.ascontiguousarray(nx.to_numpy(A), dtype=np.float64)
    b = np.ascontiguousarray(nx.to_numpy(B), dtype=np.float64)
    n = a.shape[0]

    if n == 0 or np.any(a < 0) or np.any(b < 0):
        return _EMPTY_U64, _EMPTY_U64, _EMPTY_F64, 0.0, _RESULT_INFEASIBLE

    total_a = a.sum()
    total_b = b.sum()
    if abs(total_a - total_b) > 1e-8 * max(1.0, total_a):
        return _EMPTY_U64, _EMPTY_U64, _EMPTY_F64, 0.0, _RESULT_INFEASIBLE

    x = np.arange(n, dtype=np.float64)
    plan_values, indices, cost = emd_1d_sorted(a, b, x, x, metric="cityblock")

    # The merge algorithm behind emd_1d_sorted can emit exact-zero-mass
    # entries at ties (e.g. two identical histograms); drop them so the plan
    # only lists actual mass movements, matching the general grid solver.
    nonzero = plan_values > 0
    plan_sources = np.ascontiguousarray(indices[nonzero, 0], dtype=np.uint64)
    plan_targets = np.ascontiguousarray(indices[nonzero, 1], dtype=np.uint64)
    plan_values = np.ascontiguousarray(plan_values[nonzero], dtype=np.float64)
    return plan_sources, plan_targets, plan_values, cost, _RESULT_OPTIMAL


def emd_grid_l1(
    A, B, numItermax=100000, return_plan=False, log=False, check_marginals=True
):
    r"""Solves the Earth Mover's Distance with the cityblock ground metric
    between two histograms sharing the same :math:`d`-dimensional Cartesian
    grid support.

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    where :math:`M_{i,j}` is the cityblock (:math:`\ell_1`) distance between
    grid nodes :math:`i` and :math:`j` on the integer grid of shape
    `A.shape`, and :math:`\mathbf{a}`, :math:`\mathbf{b}` are `A`, `B`
    flattened in C order.

    Rather than solving the min-cost flow on the full bipartite graph between
    the :math:`n=\prod(\text{A.shape})` source and target bins (as
    :any:`ot.emd` would), this reduces the problem to a min-cost flow on the
    much sparser grid adjacency graph (:math:`\mathcal{O}(d \cdot n)` arcs
    instead of :math:`\mathcal{O}(n^2)`), using the graph formulation of Ling
    & Okada [1]_. Unlike [1]_, which introduces a bespoke tree-based solver
    for that reduced graph, this reuses POT's existing (off-the-shelf)
    network simplex LP solver on it. This is exact and typically one to two
    orders of magnitude faster than :any:`ot.emd` for grid histograms.

    .. note:: The min-cost flow solved internally is a Beckmann-style flow on
        the grid's adjacency graph: it moves mass between *neighbouring*
        cells and does not by itself say which source bin any given unit of
        mass originally came from. Recovering the actual transportation plan
        :math:`\gamma` (a coupling between source and target bins) requires
        decomposing that flow into paths, which has a non-negligible cost of
        its own. Set `return_plan` to request it; leave it False (the
        default) when only the transport cost is needed.

    .. note:: For a 1D grid (``A.ndim == 1``), this instead delegates to
        POT's :math:`\mathcal{O}(n)` sorted-support solver
        (:any:`ot.lp.emd_1d_sorted`), which is exact here since the grid's
        integer positions are already a shared, sorted support. This avoids
        the network simplex setup entirely for that case. Better still, when
        `return_plan` is False, the cost itself has a closed form that runs
        as generic backend reductions with no CPU round-trip at all, so a 1D
        grid on a GPU array is solved entirely on-device.

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. Beyond the 1D, cost-only case above,
        the algorithm uses a C++/Cython CPU solver, so GPU arrays are copied
        to CPU before solving (and the transportation plan's bin indices, if
        requested, are returned as CPU arrays). Gradients are not currently
        supported: `cost` is always detached from any computation graph the
        inputs were part of.

    Parameters
    ----------
    A : array-like, float64, shape (n_1, ..., n_d)
        Source histogram on a :math:`d`-dimensional Cartesian grid
    B : array-like, float64, shape (n_1, ..., n_d)
        Target histogram, on the same grid as `A`
    numItermax : int, optional (default=100000)
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.
    return_plan : bool, optional (default=False)
        If True, additionally recovers an explicit transportation plan (a
        sparse coupling), returned in `log`. Computing it has a cost of its
        own (a CPU round-trip, and either a network simplex solve or an O(n)
        merge), so it is skipped by default when only the transport cost
        `cost` is needed.
    log : bool, optional (default=False)
        If True, also returns a dictionary with the solver status and,
        if `return_plan` is True, the sparse transportation plan.
    check_marginals : bool, optional (default=True)
        If True, checks that `A` and `B` have the same total mass.

    Returns
    -------
    cost : float
        Optimal transportation cost.
    log : dict, optional
        If input `log` is True, a dictionary containing the solver status
        (`warning`, `result_code`) and, if `return_plan` is True, the sparse
        transportation plan `G` (built via the backend's `coo_matrix`, same
        as :any:`ot.emd2_lazy`'s `return_matrix`; a real sparse matrix for
        NumPy/PyTorch/TensorFlow/CuPy, silently densified for JAX, which has
        no sparse array type) of shape :math:`(n, n)` with
        :math:`n=\prod(\text{A.shape})`, indexing into `A.reshape(-1)` and
        `B.reshape(-1)`.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([1.0, 0.0, 0.0, 0.0])
    >>> B = np.array([0.0, 0.0, 0.0, 1.0])
    >>> emd_grid_l1(A, B)
    3.0

    References
    ----------
    .. [1] Ling, H., & Okada, K. (2007). An efficient earth mover's distance
        algorithm for robust histogram comparison. IEEE Transactions on
        Pattern Analysis and Machine Intelligence, 29(5), 840-853.

    See Also
    --------
    ot.emd : Exact OT solver with a general, precomputed cost matrix
    """
    A, B = list_to_array(A, B)
    nx = get_backend(A, B)

    if A.shape != B.shape:
        raise ValueError(
            f"A and B must have the same shape, got {A.shape} and {B.shape}"
        )

    if check_marginals:
        np.testing.assert_allclose(
            nx.to_numpy(nx.sum(A)),
            nx.to_numpy(nx.sum(B)),
            rtol=1e-7,
            atol=1e-7,
            err_msg="A and B must have the same total mass",
        )

    if A.ndim == 1:
        # A 1D grid is backend-native either way (works on any backend's
        # arrays via `nx`, GPU included). Only the cost-only case below
        # avoids a CPU round-trip entirely though: recovering the plan needs
        # the compiled O(n) merge in _emd_grid_l1_1d_plan, which is CPU-only.
        if not return_plan:
            cost, result_code = _emd_grid_l1_1d_cost(A, B, nx)
            if log:
                return cost, {
                    "warning": check_result(result_code),
                    "result_code": result_code,
                }
            check_result(result_code)
            return cost

        plan_sources, plan_targets, plan_values, cost, result_code = (
            _emd_grid_l1_1d_plan(A, B, nx)
        )
    else:
        shape = np.array(A.shape, dtype=np.int64)
        # The C++ solver only understands flattened (CPU) numpy arrays:
        # `to_numpy` also does the GPU -> CPU copy for backends such as
        # torch or jax.
        a_np = np.ascontiguousarray(nx.to_numpy(A), dtype=np.float64).reshape(-1)
        b_np = np.ascontiguousarray(nx.to_numpy(B), dtype=np.float64).reshape(-1)
        plan_sources, plan_targets, plan_values, cost, result_code = emd_c_grid_l1(
            a_np, b_np, shape, numItermax, return_plan
        )

    cost = nx.from_numpy(cost, type_as=A)

    if log:
        log_dict = {
            "warning": check_result(result_code),
            "result_code": result_code,
        }
        if return_plan:
            # A.size is a numpy property but a torch method: go through
            # A.shape (uniformly a tuple of ints/plain integers across
            # backends) instead.
            n = int(np.prod(A.shape))
            plan_values_b = nx.from_numpy(plan_values, type_as=A)
            plan_sources_b = nx.from_numpy(plan_sources.astype(np.int64), type_as=A)
            plan_targets_b = nx.from_numpy(plan_targets.astype(np.int64), type_as=A)
            log_dict["G"] = nx.coo_matrix(
                plan_values_b,
                plan_sources_b,
                plan_targets_b,
                shape=(n, n),
                type_as=A,
            )
        return cost, log_dict

    check_result(result_code)
    return cost
