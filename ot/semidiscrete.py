# -*- coding: utf-8 -*-
"""
Semi-discrete optimal transport: continuous source, discrete target.

Backend-agnostic semi-dual solver based on the Projected Averaged SGD
of [1]_, with an optional decreasing entropic regularization schedule
(DRAG, [2]_). Works with any backend supported by :mod:`ot.backend`
(NumPy, PyTorch, JAX, CuPy, TensorFlow).

References
----------
.. [1] Genans, Godichon-Baggioni, Vialard, Wintenberger (2025).
   "Stochastic Optimization in Semi-Discrete Optimal Transport:
   Convergence Analysis and Minimax Rate." NeurIPS 2025.
.. [2] Genans, Godichon-Baggioni, Vialard, Wintenberger (2025).
   "Decreasing Entropic Regularization Averaged Gradient for
   Semi-Discrete Optimal Transport." NeurIPS 2025.
"""

# Author: Ferdinand Genans <genans.ferdinand@gmail.com>
#
# License: MIT License

import math

import numpy as np

from .backend import get_backend


def _quadratic_cost(x, y, nx):
    r"""Default cost: :math:`\tfrac{1}{2} \|x - y\|^2`."""
    x_sq = nx.sum(x**2, axis=1)[:, None]
    y_sq = nx.sum(y**2, axis=1)[None, :]
    cross = nx.einsum("ij,kj->ik", x, y)
    return 0.5 * (x_sq + y_sq - 2.0 * cross)


def _setup(target_positions, target_weights, cost):
    """Resolve backend, default weights and default cost."""
    nx = get_backend(target_positions)
    m = target_positions.shape[0]
    if target_weights is None:
        target_weights = nx.full((m,), 1.0 / m, type_as=target_positions)
    if cost is None:

        def cost(x, y):
            return _quadratic_cost(x, y, nx)

    return nx, m, target_weights, nx.log(target_weights), cost


def _atom_weights(score, reg, log_b, nx):
    """Row-stochastic weights ``(batch, m)`` from ``score = g - C``.

    Softmax of ``score / reg + log_b`` when ``reg > 0``, one-hot of
    ``argmax(score, axis=1)`` when ``reg == 0``.
    """
    if reg > 0:
        log_w = score / reg + log_b[None, :]
        log_w = log_w - nx.logsumexp(log_w, axis=1)[:, None]
        return nx.exp(log_w)
    m = score.shape[1]
    idx = nx.argmax(score, axis=1)
    arange_m = nx.from_numpy(np.arange(m), type_as=score)
    mask = idx[:, None] == arange_m[None, :]
    one = nx.full((1,), 1.0, type_as=score)
    zero = nx.full((1,), 0.0, type_as=score)
    return nx.where(mask, one, zero)


def atom_weights(
    target_positions,
    source_samples,
    semi_dual_potential,
    target_weights=None,
    cost=None,
    reg=0.0,
):
    r"""Row-stochastic atom-assignment weights induced by ``semi_dual_potential``.

    Returns an array ``w`` of shape ``(n_samples, n_atoms)`` such that
    ``w[i, j]`` is the (entropic) probability that sample ``x_i`` is
    transported to atom ``y_j``.
    """
    nx, _, _, log_b, cost_fn = _setup(target_positions, target_weights, cost)
    score = semi_dual_potential[None, :] - cost_fn(source_samples, target_positions)
    return _atom_weights(score, reg, log_b, nx)


def ot_map(
    target_positions,
    source_samples,
    semi_dual_potential,
    target_weights=None,
    cost=None,
    reg=0.0,
):
    r"""Transport map :math:`T(x) = \sum_j w_j(x)\, y_j` induced by the potential."""
    w = atom_weights(
        target_positions,
        source_samples,
        semi_dual_potential,
        target_weights=target_weights,
        cost=cost,
        reg=reg,
    )
    return w @ target_positions


def c_transform(
    target_positions,
    source_samples,
    semi_dual_potential,
    target_weights=None,
    cost=None,
    reg=0.0,
):
    r"""Pointwise (entropic) c-transform of ``semi_dual_potential``.

    - ``reg == 0``:  :math:`\varphi_g(x) = \min_j\, c(x, y_j) - g_j`.
    - ``reg > 0``:   :math:`\varphi_g(x) = -\varepsilon \log \sum_j b_j
      \exp\!\big((g_j - c(x, y_j))/\varepsilon\big)`.
    """
    nx, _, _, log_b, cost_fn = _setup(target_positions, target_weights, cost)
    score = semi_dual_potential[None, :] - cost_fn(source_samples, target_positions)
    if reg == 0:
        return -nx.max(score, axis=1)
    return -reg * nx.logsumexp(score / reg + log_b[None, :], axis=1)


def solve_semidiscrete(
    target_positions,
    source_sampler,
    target_weights=None,
    cost=None,
    reg=0.0,
    n_iter=10_000,
    batch_size=32,
    lr0=None,
    lr_exponent=2.0 / 3.0,
    init_potential=None,
    decreasing_reg=True,
    decreasing_reg_initial_eps=0.1,
    decreasing_reg_exponent=0.5,
    proj_bound=None,
    polyak_average=True,
    log=False,
):
    r"""Solve semi-discrete OT by Polyak-averaged SGD on the semi-dual.

    Maximizes the semi-dual :math:`g \mapsto \langle g, b \rangle + \mathbb{E}_X[\varphi_g(X)]`
    by averaged stochastic gradient ascent with projection and decreasing
    regularization, which corresponds to the DRAG algorithm [1]_.
    Here :math:`\varphi_g` denotes the (entropic) c-transform of :math:`g`,

    .. math::
        \varphi_g(x) = \begin{cases}
            \min_j \big(c(x, y_j) - g_j\big) & \text{if } \mathrm{reg} = 0, \\
            -\varepsilon \log \sum_j b_j \exp\!\big((g_j - c(x, y_j))/\varepsilon\big)
            & \text{if } \mathrm{reg} = \varepsilon > 0,
        \end{cases}

    cf. :func:`c_transform`.

    With ``decreasing_reg=True`` the regularization at iteration ``t`` is
    :math:`\varepsilon_t = \max(\text{reg},\, \varepsilon_0 / t^\alpha)` — large
    at first for smoothness, then annealed towards ``reg``. This is the
    DRAG schedule of [1]_.

    Parameters
    ----------
    target_positions : array-like, shape (n_atoms, d)
        Positions of the target atoms. The backend of this array drives
        all subsequent computations.
    source_sampler : callable
        ``source_sampler(batch_size)`` returns a ``(batch_size, d)`` array
        of source samples, in the same backend as ``target_positions``.
    target_weights : array-like, shape (n_atoms,), optional
        Atom weights. Defaults to uniform.
    cost : callable, optional
        ``cost(x, y)`` returns the ``(n_samples, n_atoms)`` cost matrix.
        Defaults to ``0.5 * ||x - y||^2``.
    reg : float, default=0.0
        Entropic regularization (target value when ``decreasing_reg=True``).
    n_iter : int, default=10000
    batch_size : int, default=32
    lr0 : float, optional
        Initial learning rate. Defaults to ``sqrt(n_atoms * batch_size)``.
    lr_exponent : float, default=2/3
        Step size decays as ``lr0 / t**lr_exponent``.
    init_potential : array-like, shape (n_atoms,), optional
        Starting iterate; defaults to zero. Not mutated.
    decreasing_reg : bool, default=True
        Enable the DRAG decreasing-regularization schedule.
    decreasing_reg_initial_eps : float, default=0.1
        Initial regularization in the DRAG schedule.
    decreasing_reg_exponent : float, default=0.5
        Decay exponent of the DRAG schedule.
    proj_bound : float, optional
        If given, clip each iterate to ``[-proj_bound, proj_bound]``.
    polyak_average : bool, default=True
        If True, return the uniform average of the iterates; else the last.
    log : bool, default=False
        If True, also return a small ``dict`` with the last iterate.

    Returns
    -------
    semi_dual_potential : array, shape (n_atoms,)
    info : dict, optional
        Returned only when ``log=True``.

    References
    ----------
    .. [1] Genans, Godichon-Baggioni, Vialard, Wintenberger (2025).
    "Decreasing Entropic Regularization Averaged Gradient for
    Semi-Discrete Optimal Transport." NeurIPS 2025.

    Examples
    --------
    >>> import numpy as np
    >>> from ot.semidiscrete import solve_semidiscrete
    >>> rng = np.random.default_rng(0)
    >>> target = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
    >>> g = solve_semidiscrete(
    ...     target, lambda b: rng.random((b, 1)),
    ...     n_iter=500, batch_size=32, proj_bound=1.0,
    ... )
    """
    nx, m, b, log_b, cost_fn = _setup(target_positions, target_weights, cost)

    if init_potential is None:
        g = nx.zeros((m,), type_as=target_positions)
    else:
        g = init_potential + nx.zeros((m,), type_as=target_positions)

    if lr0 is None:
        lr0 = math.sqrt(m * batch_size)

    g_avg = nx.zeros((m,), type_as=target_positions) if polyak_average else None

    for t in range(1, n_iter + 1):
        if decreasing_reg:
            reg_t = max(reg, decreasing_reg_initial_eps / (t**decreasing_reg_exponent))
        else:
            reg_t = reg

        x = source_sampler(batch_size)
        score = g[None, :] - cost_fn(x, target_positions)
        w = _atom_weights(score, reg_t, log_b, nx)
        grad = nx.mean(w, axis=0) - b

        lr_t = lr0 / (t**lr_exponent)
        g = g - lr_t * grad
        if proj_bound is not None:
            g = nx.clip(g, -proj_bound, proj_bound)
        if polyak_average:
            g_avg = g_avg + (g - g_avg) / t

    result = g_avg if polyak_average else g
    if log:
        return result, {
            "n_iter": n_iter,
            "batch_size": batch_size,
            "proj_bound": proj_bound,
            "polyak_average": polyak_average,
            "last_potential": g,
        }
    return result


__all__ = [
    "solve_semidiscrete",
    "atom_weights",
    "ot_map",
    "c_transform",
]
