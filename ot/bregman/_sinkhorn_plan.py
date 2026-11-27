"""Dense, opt-in full-plan stopping for positive NumPy Sinkhorn.

The estimate is derived in exact arithmetic; float64 is not outward rounded.
See https://github.com/PythonOT/POT/issues/934 for the proof and benchmarks.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from numbers import Real
from threading import local
import warnings

import numpy as np

# One opportunistic entry per thread. Larger kernels are built per call.
_AUTO_BYTES = 32 * 1024 * 1024
_cache = local()

# A byte comparison uses no matrix-sized temporary and exits on the first difference.
# np.array_equal is the portable fallback when a C runtime symbol is unavailable.
try:
    _memcmp = ctypes.CDLL(None).memcmp
except (AttributeError, OSError, TypeError):
    _memcmp = None
else:
    _memcmp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    _memcmp.restype = ctypes.c_int


def _plan_factor(low, high, n):
    # log cross-ratio diameter <= 2 log(high/low). Guard before dividing.
    cap = float(n - 1)
    if high >= (2 * cap - 1) * low:
        return cap
    return 0.5 * (1 + high / low)


@dataclass(eq=False)
class _Kernel:
    matrix: np.ndarray
    column_factor: float
    row_factor: float
    uniform_columns: np.ndarray | None = None

    def cold_columns(self, u):
        if self.uniform_columns is None:
            self.uniform_columns = self.matrix.T @ u
        return self.uniform_columns


def _build_kernel(M, reg, *, shared):
    # The caller selects strict or ordinary NumPy warning semantics.
    K = np.empty_like(M)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)
    low, high = K.min(), K.max()
    if not (low > 1e-280 and high < 1e100):
        # The standard solver can consume this already-computed kernel.
        return K
    factor = _plan_factor(low, high, M.shape[1])
    ratio = low / high
    tau = (1 - ratio) / (1 + ratio)
    propagated = min(1 + _plan_factor(low, high, M.shape[0]), factor * tau)
    if shared:
        K.setflags(write=False)
    return _Kernel(K, factor, propagated)


@dataclass(frozen=True, eq=False)
class _Prepared:
    cost: np.ndarray
    kernel: object


@dataclass
class _Entry:
    signature: tuple
    prepared: _Prepared | None = None
    skip: int = 0


def _same_cost(M, saved):
    # Both are live C-contiguous CPU float64 ndarrays with matching shape, checked by the caller.
    if _memcmp is None:
        return np.array_equal(M, saved)
    return _memcmp(M.ctypes.data, saved.ctypes.data, M.nbytes) == 0


def _auto_kernel(M, reg):
    if 2 * M.nbytes + M.shape[1] * 8 > _AUTO_BYTES:
        return _build_kernel(M, reg, shared=False)
    rows, cols = M.shape
    signature = (M.shape, reg, M[0, 0], M[rows // 2, cols // 2], M[-1, -1])
    entry = getattr(_cache, "entry", None)
    if entry is None or entry.signature != signature:
        # Samples screen candidates only. No kernel is ever reused on this basis.
        _cache.entry = _Entry(signature)
    elif entry.skip:
        entry.skip -= 1
    elif entry.prepared is None:
        cost = M.copy(order="C")
        cost.setflags(write=False)
        kernel = _build_kernel(cost, reg, shared=True)
        if isinstance(kernel, _Kernel):
            entry.prepared = _Prepared(cost, kernel)
        return kernel
    elif _same_cost(M, entry.prepared.cost):
        return entry.prepared.kernel
    else:
        # If the samples match while other entries change, defer another snapshot.
        entry.prepared, entry.skip = None, 8
    return _build_kernel(M, reg, shared=False)


def _materialize(K, u, v):
    # An unshared kernel can be overwritten. Every shared-kernel solve creates its own result.
    out = K if K.flags.writeable else None
    plan = np.multiply(K, u[:, None], out=out)
    plan *= v
    return plan


def _residual_norm(residual):
    """Euclidean norm for a real, contiguous one-dimensional residual."""
    return np.float64(math.sqrt(residual.dot(residual)))


def _try_marginal(a, b, M, reg, limit, tol, warn, *, warmstart=None, log=False):
    """Return (plan, None) on success or (None, reusable unscaled kernel) on fallback."""
    if (
        type(a) is not np.ndarray
        or type(b) is not np.ndarray
        or type(M) is not np.ndarray
        or a.dtype != np.float64
        or b.dtype != np.float64
        or M.dtype != np.float64
    ):
        return None, None
    if (
        a.ndim != 1
        or b.ndim != 1
        or M.shape != (a.size, b.size)
        or min(a.size, b.size) < 2
        or not M.flags.c_contiguous
        or type(limit) not in (int, np.int32, np.int64)
        or limit < 11
        or type(tol) not in (float, np.float64)
        or not 1e-13 <= tol < 1
        or type(reg) not in (float, int, np.float64)
        or not np.isfinite(reg)
        or reg <= 0
    ):
        return None, None
    min_a, min_b = a.min(), b.min()
    if (
        not min_a > 0
        or not min_b > 0
        or not abs(a.sum() - 1.0) <= 8e-15
        or not abs(b.sum() - 1.0) <= 8e-15
    ):
        return None, None
    if warmstart is not None:
        if not isinstance(warmstart, (tuple, list)) or len(warmstart) != 2:
            return None, None
        for seed, marginal in zip(warmstart, (a, b)):
            if (
                type(seed) is not np.ndarray
                or seed.dtype != np.float64
                or seed.shape != marginal.shape
            ):
                return None, None
        # Outside this range let the standard loop handle floating-point
        # exceptions and rollback with both original scaling vectors.
        if (
            min(min_a, min_b) < 1e-100
            or not np.abs(warmstart[0]).max() < 200.0
            or not np.abs(warmstart[1]).max() < 200.0
        ):
            return None, None
    kernel = _auto_kernel(M, reg)
    if not isinstance(kernel, _Kernel):
        return None, kernel
    if warmstart is not None:
        return _warm_marginal(
            a, b, kernel.matrix, np.exp(warmstart[0]), limit, tol, log, warn
        )
    K = kernel.matrix
    wide = b.size >= 4 * a.size
    col_budget = (0.25 * tol) / kernel.column_factor
    row_budget = (0.25 * tol) / max(kernel.row_factor, 1e-300)
    u = np.full(a.size, 1.0 / a.size)
    col_product = kernel.cold_columns(u)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        for it in range(limit):
            v = b / col_product
            kv = K @ v
            checkpoint = it % 10 == 0
            extra = it % 10 in (4, 7, 9) and it + 10 - it % 10 < limit
            accept = wide and extra and 0.5 * np.abs(u * kv - a).sum() < row_budget
            u = a / kv
            if not accept:
                col_product = K.T @ u
                if checkpoint:
                    accept = _residual_norm(v * col_product - b) < tol
                elif extra and not wide:
                    accept = 0.5 * np.abs(v * col_product - b).sum() < col_budget
            if accept:
                plan = _materialize(K, u, v)
                z = plan.sum(axis=0) - b
                if (
                    _residual_norm(z) < tol
                    and _residual_norm(plan.sum(axis=1) - a) < tol
                    and (wide or checkpoint or 0.5 * np.abs(z).sum() < col_budget)
                ):
                    return plan, None
                # _materialize can overwrite an unshared kernel.
                return None, None if K.flags.writeable else K
            if checkpoint and not np.isfinite(u).all():
                return None, K
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            return None, K
        plan = _materialize(K, u, v)
        if not _residual_norm(plan.sum(axis=1) - a) < max(tol, 1e-12):
            return None, None if K.flags.writeable else K
    if warn:
        warnings.warn(
            "Sinkhorn did not converge. You might want to "
            "increase the number of iterations `numItermax` "
            "or the regularization parameter `reg`.",
            stacklevel=3,
        )
    return plan, None


def _warm_marginal(a, b, K, u, limit, tol, log, warn):
    """Check the ordinary marginal tolerance at updates 1–4, 11, 21, ... ."""
    errors = [] if log else None
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        col_product = K.T @ u
        for it in range(limit):
            v = b / col_product
            u = a / (K @ v)
            col_product = K.T @ u
            ordinary = it % 10 == 0
            if it >= 4 and not ordinary:
                continue
            err = _residual_norm(v * col_product - b)
            if not math.isfinite(err):
                return None, K
            if log and ordinary:
                errors.append(err)
            if err < tol:
                break
        else:
            # Replay a nonfinite failure from the original inputs so that POT
            # emits its usual numerical warning and returns its rollback state.
            if not (
                np.isfinite(v).all()
                and np.isfinite(u).all()
                and np.isfinite(col_product).all()
                and (u > 0).all()
                and (v > 0).all()
                and (col_product > 0).all()
            ):
                return None, K
        plan = _materialize(K, u, v)
        row_error = _residual_norm(plan.sum(axis=1) - a)
        col_error = _residual_norm(plan.sum(axis=0) - b)
        converged = (ordinary or it < 4) and err < tol
        if not (
            row_error < tol
            and math.isfinite(col_error)
            and (not converged or col_error < tol)
        ):
            return None, None if K.flags.writeable else K
    if converged:
        if log and not ordinary:
            errors.append(col_error)
    elif warn:
        warnings.warn(
            "Sinkhorn did not converge. You might want to "
            "increase the number of iterations `numItermax` "
            "or the regularization parameter `reg`.",
            stacklevel=3,
        )
    if log:
        return (plan, {"err": errors, "niter": it, "u": u, "v": v}), None
    return plan, None


def _direction_bound(z, col, u, v, old_z, old_u, old_v, factor, target):
    direction = z - old_z
    squared_length = direction @ direction
    if not np.isfinite(squared_length) or squared_length <= 0:
        return np.inf
    t = max(0.0, float(-(z @ direction) / squared_length))
    if not np.isfinite(t) or not t:
        return np.inf

    # This nonnegative summand suffices to determine both rejection and retry
    # when the full bound is at least four times the target.
    residual_part = factor * 0.5 * np.abs(z + t * direction).sum()
    if residual_part >= 4.0 * target:
        return np.inf

    row_ratio = old_u / u
    col_ratio = old_v / v
    max_row_ratio = row_ratio.max()
    m = max(0.0, t * (max_row_ratio * col_ratio.max() - 1))
    if not np.isfinite(m) or m >= 1:
        return np.inf
    centered = col_ratio - (col @ col_ratio) / col.sum()
    chi_upper = (t * max_row_ratio) ** 2 * (col @ (centered * centered))
    entropy_part = np.sqrt(chi_upper * min(0.5, 0.25 / (1 - m)))
    bound = entropy_part + residual_part
    return bound if np.isfinite(bound) else np.inf


def _solve_full_plan(a, b, M, reg, limit, tol, target, *, log, warmstart):
    """Solve the opt-in problem with the validated residual-based scheduler."""
    if not all(type(x) is np.ndarray and x.dtype == np.float64 for x in (a, b, M)):
        raise ValueError("plan_tol requires NumPy float64 arrays.")
    if (
        a.ndim != 1
        or b.ndim != 1
        or M.ndim != 2
        or M.shape != (a.size, b.size)
        or min(a.size, b.size) < 2
        or not M.flags.c_contiguous
    ):
        raise ValueError(
            "plan_tol requires one histogram pair and a C-contiguous cost matrix with at least two rows and columns."
        )
    if not all(
        isinstance(x, Real)
        and not isinstance(x, (bool, np.bool_))
        and np.isfinite(x)
        and x > 0
        for x in (reg, tol, target)
    ):
        raise ValueError(
            "plan_tol requires positive finite tolerances and regularization."
        )
    if (
        not isinstance(limit, (int, np.integer))
        or isinstance(limit, (bool, np.bool_))
        or limit < 1
    ):
        raise ValueError("plan_tol requires a positive integer numItermax.")
    if (
        not np.all(np.isfinite(a) & (a > 0))
        or not np.all(np.isfinite(b) & (b > 0))
        or abs(a.sum() - 1.0) > 8e-15
        or abs(b.sum() - 1.0) > 8e-15
    ):
        raise ValueError("plan_tol requires strictly positive probability marginals.")
    # The kernel range check already rejects NaN and both infinities in M.
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        kernel = _auto_kernel(M, reg)
    if not isinstance(kernel, _Kernel):
        raise ValueError(
            "plan_tol requires a numerically safe, strictly positive Gibbs kernel."
        )

    K = kernel.matrix
    c = min(kernel.column_factor, len(a))
    wide = len(b) >= 4 * len(a)
    use_direction = M.size > 4096
    pre_budget = target / max(kernel.row_factor, 1e-300)
    if warmstart is None:
        u = np.full(len(a), 1.0 / len(a))
        col_product = kernel.cold_columns(u)
    else:
        if (
            not isinstance(warmstart, (tuple, list))
            or len(warmstart) != 2
            or not all(
                type(x) is np.ndarray
                and x.dtype == np.float64
                and x.shape == (n,)
                and np.isfinite(x).all()
                for x, n in zip(warmstart, (len(a), len(b)))
            )
        ):
            raise ValueError(
                "plan_tol warmstart requires a pair of finite NumPy float64 log-scalings."
            )
        with np.errstate(over="ignore", under="ignore"):
            u = np.exp(warmstart[0])
        if not (np.isfinite(u).all() and (u > 0).all()):
            raise ArithmeticError(
                "Sinkhorn plan_tol warmstart scaling is outside the float64 range."
            )
        col_product = K.T @ u
    previous = None
    retry_below = np.inf
    errors = []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        for it in range(limit):
            v = b / col_product
            kv = K @ v
            ordinary = it % 10 == 0
            extra = it % 10 in (4, 7, 9) and it + 10 - it % 10 < limit
            probe = (
                ordinary
                or extra
                or (warmstart is not None and it < 4 and use_direction)
            )
            reason = None
            if wide and probe:
                before = 0.5 * np.abs(u * kv - a).sum()
                if before < pre_budget:
                    reason = "pre"
                    estimate = kernel.row_factor * before
            u = a / kv
            if reason is None:
                col_product = K.T @ u
                if not probe:
                    continue
                cols_before = v * col_product
                residual = cols_before - b
                logged_l2 = None
                if log and ordinary:
                    logged_l2 = np.linalg.norm(residual)
                    errors.append(logged_l2)
                l1 = np.abs(residual).sum()
                e = 0.5 * l1
                estimate = c * e
                if estimate < target and (
                    l1 < tol
                    or (
                        logged_l2 if logged_l2 is not None else np.linalg.norm(residual)
                    )
                    < tol
                ):
                    reason = "direct"
                elif (
                    use_direction
                    and it + 1 >= 11
                    and previous is not None
                    and e < target
                    and e <= retry_below
                ):
                    old_z, old_u, old_v, old_e, old_it = previous
                    gap = it - old_it
                    fast_enough = (
                        e < old_e and estimate * (e / old_e) ** (4 / gap) <= target
                    )
                    if not fast_enough:
                        estimate = _direction_bound(
                            residual, cols_before, u, v, old_z, old_u, old_v, c, target
                        )
                        if estimate < target:
                            reason = "vector"
                        else:
                            retry_below = e * max(0.25, min(0.9, target / estimate))
                if reason is None:
                    if use_direction and it + 1 >= 8:
                        previous = residual, u, v, e, it
                    if not np.isfinite(u).all() or not np.isfinite(v).all():
                        raise ArithmeticError(
                            "Sinkhorn plan_tol could not be verified after numerical failure."
                        )
                    continue

            # Recheck the materialized result, not only the scaling residual.
            plan = _materialize(K, u, v)
            cols = plan.sum(axis=0)
            residual = cols - b
            col_l2 = np.linalg.norm(residual)
            if reason == "direct":
                estimate = c * 0.5 * np.abs(residual).sum()
            elif reason == "vector":
                estimate = _direction_bound(
                    residual, cols, u, v, old_z, old_u, old_v, c, target
                )
            if not (
                estimate < target
                and col_l2 < tol
                and np.linalg.norm(plan.sum(axis=1) - a) < tol
            ):
                raise ArithmeticError(
                    "Sinkhorn plan_tol could not be verified on the materialized plan."
                )
            if log:
                if reason == "pre" and ordinary:
                    errors.append(col_l2)
                return plan, {
                    "niter": it,
                    "err": errors,
                    "u": u,
                    "v": v,
                    "plan_tv_estimate": float(estimate),
                    "plan_exit": reason,
                }
            return plan
    raise ArithmeticError("Sinkhorn plan_tol was not met within numItermax.")
