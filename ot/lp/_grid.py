"""
Exact EMD-L1 solver for histograms sharing a Cartesian grid support.
"""

# Author: Tom Vercauteren
#
# License: MIT License

import numpy as np

from ..backend import get_backend
from ..utils import list_to_array
from .emd_wrap import check_result, emd_c_grid_l1

# Mirrors ot::ProblemType in ot/lp/EMD.h (INFEASIBLE=0, OPTIMAL=1, ...). Not
# otherwise exposed to Python, since it is a Cython ``cdef enum``.
_RESULT_INFEASIBLE = 0
_RESULT_OPTIMAL = 1


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


def _emd_grid_l1_1d(A, B, return_plan, return_alpha, nx):
    r"""Fully backend-native cost, gradient, and (optionally) transportation
    plan for a 1D grid.

    A 1D grid with a shared, sorted, unit-spaced support has a classic
    closed form for both the L1 (cityblock) Wasserstein cost and its
    gradient: with :math:`\text{CumA}(k) = \sum_{i \leq k} A_i` (and
    likewise for `B`),

    .. math::
        \text{cost} = \sum_{k=0}^{n-2} |\text{CumA}(k) - \text{CumB}(k)|,
        \quad
        \alpha_i = \frac{\partial \text{cost}}{\partial A_i} =
        \sum_{k=i}^{n-2} \text{sign}(\text{CumA}(k) - \text{CumB}(k))

    (the gradient is a reverse/suffix cumulative sum of the sign of the CDF
    difference, an application of the envelope theorem to this LP). Both
    share the same ``nx.cumsum(A) - nx.cumsum(B)``, so are computed together
    here in :math:`\mathcal{O}(n)`, with no CPU round-trip: every step is a
    generic backend reduction (``nx.cumsum``, ``nx.sign``, ``nx.flip``, ...).
    Unlike the cost, `alpha` is not otherwise needed, so it is only computed
    when `return_alpha` is True.

    When `return_plan` is True, :any:`_emd_grid_l1_1d_monotone_plan` recovers
    the actual transportation plan too, via a fully vectorized merge of the
    two CDFs (also backend-native, :math:`\mathcal{O}(n \log n)` for the
    sort it needs); left `None` otherwise, since it is not needed for the
    cost or the gradient.

    Returns ``(cost, alpha, plan_sources, plan_targets, plan_values,
    result_code)``, all backend-native, matching `A`'s dtype and device
    (`alpha`, `plan_*` are `None` when not requested/not needed, see below).
    `alpha` (raw, uncentred; the caller centres it) is `None` if
    `return_alpha` is False. `plan_*` are `None` if `return_plan` is False.
    """
    n = A.shape[0]

    if n == 0 or nx.any(A < 0) or nx.any(B < 0):
        zero_cost = _finalize_native_cost(0.0 * nx.sum(A), nx)
        zero_alpha = nx.zeros(A.shape, type_as=A) if return_alpha else None
        return zero_cost, zero_alpha, None, None, None, _RESULT_INFEASIBLE

    total_a = nx.sum(A)
    total_b = nx.sum(B)
    if abs(float(nx.to_numpy(total_a)) - float(nx.to_numpy(total_b))) > 1e-8 * max(
        1.0, float(nx.to_numpy(total_a))
    ):
        zero_cost = _finalize_native_cost(0.0 * total_a, nx)
        zero_alpha = nx.zeros(A.shape, type_as=A) if return_alpha else None
        return zero_cost, zero_alpha, None, None, None, _RESULT_INFEASIBLE

    cum_a = nx.cumsum(A, 0)
    cum_b = nx.cumsum(B, 0)
    cum_diff = cum_a - cum_b
    cost = _finalize_native_cost(nx.sum(nx.abs(cum_diff[:-1])), nx)

    if return_alpha:
        sign = nx.sign(cum_diff[:-1])
        suffix = nx.flip(nx.cumsum(nx.flip(sign, 0), 0), 0)
        alpha = nx.concatenate([suffix, nx.zeros((1,), type_as=A)], axis=0)
    else:
        alpha = None

    if not return_plan:
        return cost, alpha, None, None, None, _RESULT_OPTIMAL

    # cum_a, cum_b are reused as is (the plan needs the raw, unpinned CDFs
    # too), avoiding a second nx.cumsum(A) / nx.cumsum(B).
    plan_sources, plan_targets, plan_values = _emd_grid_l1_1d_monotone_plan(
        cum_a, cum_b, nx
    )
    return cost, alpha, plan_sources, plan_targets, plan_values, _RESULT_OPTIMAL


def _emd_grid_l1_1d_monotone_plan(cum_a, cum_b, nx):
    r"""Exact monotone 1D transportation plan on a shared grid support, in
    sparse COO form.

    Works by merging the two CDFs: sorting their :math:`2n` combined
    breakpoints together, the cumulative count of A-breakpoints and
    B-breakpoints seen so far at each step of the merge gives that step's
    (source bin, target bin) pair, and the gap to the previous breakpoint
    gives the mass moved. A source bin can touch more than two target bins
    in general (and vice versa), which is why this has to be indexed by
    merge step rather than by bin -- the same reason the classic sequential
    merge algorithm behind :any:`ot.lp.emd_1d_sorted` needs to walk source
    and target pointers independently.

    Takes the raw (unpinned) CDFs `cum_a`, `cum_b` (``nx.cumsum(A, 0)``,
    ``nx.cumsum(B, 0)``) rather than `A`, `B` themselves, since the caller
    (:any:`_emd_grid_l1_1d`) already needs them for the cost and can pass
    them along here, avoiding computing them twice. Assumes `A`, `B` are
    feasible (nonnegative, matching total mass); the caller is responsible
    for checking this first.

    Returns backend-native ``(plan_sources, plan_targets, plan_values)``,
    matching `cum_a`'s dtype and device, with exactly :math:`2n-1` entries.
    Some entries may carry zero flow (e.g. at ties between the two CDFs,
    such as when `A` and `B` are identical): these are harmless, explicit
    zeros in the sparse coupling built from them, and filtering them out
    would need a backend-native "keep the nonzero entries" op that does not
    exist as a generic, shape-static reduction, so it is not worth it for
    what is already a tiny, fixed-size result.
    """
    n = cum_a.shape[0]

    # Pin the last entry of each CDF to exactly 1: avoids floating-point
    # drift making the tie at the very end (both CDFs must reach the same
    # total mass) not an exact tie, which would otherwise show up as a
    # spurious tiny "flow" entry below.
    one = nx.reshape(0.0 * cum_a[-1] + 1.0, (1,))
    cum_a = nx.concatenate([cum_a[:-1], one], axis=0)
    cum_b = nx.concatenate([cum_b[:-1], one], axis=0)
    all_cum = nx.concatenate([cum_a, cum_b], axis=0)  # length 2n

    # One-hot markers of where each of the 2n merged breakpoints came from,
    # carried through the sort below to recover, at each step, how many of
    # each type preceded it.
    ones_n = nx.ones((n,), type_as=cum_a)
    zeros_n = nx.zeros((n,), type_as=cum_a)
    origin_source = nx.concatenate([ones_n, zeros_n], axis=0)
    origin_target = nx.concatenate([zeros_n, ones_n], axis=0)

    perm = nx.argsort(all_cum, axis=-1)
    sort_val = nx.take_along_axis(all_cum, perm, axis=-1)
    sort_source = nx.take_along_axis(origin_source, perm, axis=-1)
    sort_target = nx.take_along_axis(origin_target, perm, axis=-1)

    origin = nx.stack([sort_source, sort_target], axis=0)  # (2, 2n)
    excl_cumsum = nx.cumsum(origin, axis=-1) - origin
    idx = nx.clip(excl_cumsum, None, n - 1)  # (2, 2n): [0] source, [1] target

    left = nx.zero_pad(sort_val[:-1], [(1, 0)], value=0.0)
    flow = nx.clip(sort_val - left, 0.0, None)

    # The very last merge step is the simultaneous end of both CDFs (both
    # pinned to 1 above), carrying no flow; drop it, leaving exactly 2n-1
    # entries, as expected for a monotone coupling of two n-atom measures.
    return idx[0, :-1], idx[1, :-1], flow[:-1]


def emd_grid_l1(
    A,
    B,
    numItermax=100000,
    return_plan=False,
    log=False,
    check_marginals=True,
    grad="envelope",
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

    .. note:: For a 1D grid (``A.ndim == B.ndim == 1``), this instead uses a
        closed form (see :any:`_emd_grid_l1_1d`), exact here since the
        grid's integer positions are already a shared, sorted support. This
        avoids the network simplex setup entirely for that case, and runs
        as generic backend reductions with no CPU round-trip at all -- for
        the cost and gradient, but also for the plan, when requested -- so
        a 1D grid on a GPU array is solved entirely on-device.

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. Beyond the 1D case above, the
        algorithm uses a C++/Cython CPU solver, so GPU arrays are copied
        to CPU before solving (and the transportation plan's bin indices, if
        requested, are returned as CPU arrays). `cost` is always detached
        from any computation graph the inputs were part of: this does not
        support automatic differentiation. The exact gradient of `cost`
        with respect to `A` and `B` is instead exposed explicitly as
        `alpha`/`beta` in `log` (see below).

    .. note:: `log["alpha"]` and `log["beta"]`, when present, are the
        (centred) dual potentials, which give the gradient of `cost` with
        respect to `A` and `B` directly: :math:`\partial \text{cost}/\partial
        A_i = \text{alpha}_i` and :math:`\partial \text{cost}/\partial B_i =
        \text{beta}_i = -\text{alpha}_i` (a single graph is used, not a
        bipartite source/target split, so there is one potential array, not
        two). For :math:`d \geq 2` these are LEMON's network-simplex node
        potentials, an unavoidable byproduct of the solve itself (a free
        application of the envelope theorem to this LP), so they are always
        computed and returned in `log` regardless of `grad`. For a 1D grid
        they are instead a closed form (see :any:`_emd_grid_l1_1d`) that
        needs its own, separate :math:`\mathcal{O}(n)` pass -- not otherwise
        free -- so `grad` controls whether it runs there. Either way they
        are defined up to an additive constant; centring (subtracting their
        mean) picks the canonical representative, the Riemannian gradient of
        `cost` on the probability simplex.

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
        own (for :math:`d \geq 2`, a network simplex solve, plus a CPU
        round-trip since that solver is CPU-only; for a 1D grid, an
        :math:`\mathcal{O}(n \log n)` merge, entirely on-device), so it is
        skipped by default when only the transport cost `cost` is needed.
    log : bool, optional (default=False)
        If True, also returns a dictionary with the solver status and,
        if `return_plan` is True, the sparse transportation plan.
    check_marginals : bool, optional (default=True)
        If True, checks that `A` and `B` have the same total mass.
    grad : {'envelope', None}, optional (default='envelope')
        Controls whether the dual potentials `alpha`/`beta` (the gradient of
        `cost`) are computed for a 1D grid, where doing so has a cost of its
        own (see the note above). For :math:`d \geq 2` they come for free as
        a byproduct of the network-simplex solve, so this has no effect
        there: they are always computed and returned in `log`. If
        `'envelope'` (the default), also compute them for a 1D grid, via the
        envelope theorem applied to that grid's closed form. If None, skip
        that computation for a 1D grid, and `log` will not contain `alpha`
        or `beta` in that case. Has no effect unless `log` is True.

    Returns
    -------
    cost : float
        Optimal transportation cost.
    log : dict, optional
        If input `log` is True, a dictionary containing the solver status
        (`warning`, `result_code`) and, if `return_plan` is True, the sparse
        transportation plan `G` (built via the backend's
        `coo_matrix`, same as :any:`ot.emd2_lazy`'s `return_matrix`; a real
        sparse matrix for NumPy/PyTorch/TensorFlow/CuPy, silently densified
        for JAX, which has no sparse array type) of shape :math:`(n, n)`
        with :math:`n=\prod(\text{A.shape})`, indexing into `A.reshape(-1)`
        and `B.reshape(-1)`. Unless `A.ndim == 1` and `grad` is None, it also
        contains the (centred) dual potentials `alpha`, `beta` (the gradient
        of `cost` with respect to `A`, `B`; see the note above).

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
    if grad not in (None, "envelope"):
        raise ValueError(f"grad must be None or 'envelope', got {grad!r}")

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
        # A 1D grid is backend-native throughout -- cost, gradient, and
        # (when requested) the transportation plan -- so it gets its own,
        # fully self-contained branch. See _emd_grid_l1_1d. Unlike the
        # general (d >= 2) path below, the gradient is not a free byproduct
        # here, so it is only computed when actually requested.
        return_alpha = log and grad == "envelope"
        cost, alpha, plan_sources, plan_targets, plan_values, result_code = (
            _emd_grid_l1_1d(A, B, return_plan, return_alpha, nx)
        )

        if log:
            log_dict = {
                "warning": check_result(result_code),
                "result_code": result_code,
            }
            if alpha is not None:
                # `alpha` is only defined up to an additive constant;
                # centring it picks the canonical representative, the
                # Riemannian gradient of the cost on the probability
                # simplex.
                alpha = alpha - nx.mean(alpha)
                log_dict["alpha"] = alpha
                log_dict["beta"] = -alpha
            if return_plan and result_code == _RESULT_OPTIMAL:
                # plan_sources/targets/values are already backend-native
                # (matching A), so this is a plain packaging step, with no
                # conversion needed.
                n = A.shape[0]
                log_dict["G"] = nx.coo_matrix(
                    plan_values,
                    plan_sources,
                    plan_targets,
                    shape=(n, n),
                    type_as=A,
                )
            return cost, log_dict

        check_result(result_code)
        return cost

    # ndim >= 2: the general grid solver, backed by network simplex in C++.
    shape = np.array(A.shape, dtype=np.int64)
    # The C++ solver only understands flattened (CPU) numpy arrays:
    # `to_numpy` also does the GPU -> CPU copy for backends such as torch or
    # jax.
    a_np = np.ascontiguousarray(nx.to_numpy(A), dtype=np.float64).reshape(-1)
    b_np = np.ascontiguousarray(nx.to_numpy(B), dtype=np.float64).reshape(-1)
    (
        plan_sources,
        plan_targets,
        plan_values,
        alpha,
        cost,
        result_code,
    ) = emd_c_grid_l1(a_np, b_np, shape, numItermax, return_plan)

    # `alpha` (the node potentials) is only defined up to an additive
    # constant; centring it picks the canonical representative, the
    # Riemannian gradient of the cost on the probability simplex.
    # beta = -alpha since this is a single graph, not a bipartite
    # source/target split (supply[i] = A[i] - B[i] for every node).
    alpha = alpha - alpha.mean()
    beta = -alpha

    cost = nx.from_numpy(cost, type_as=A)

    if log:
        log_dict = {
            "warning": check_result(result_code),
            "result_code": result_code,
            "alpha": nx.from_numpy(alpha, type_as=A),
            "beta": nx.from_numpy(beta, type_as=A),
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
