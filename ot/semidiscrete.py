# -*- coding: utf-8 -*-
"""
Semi-discrete optimal transport: continuous source, discrete target.

Backend-agnostic semi-dual solver based on the Projected Averaged SGD
of [1]_, with an optional decreasing entropic regularization schedule
(DRAG, [2]_). Works with any backend supported by :mod:`ot.backend`
(NumPy, PyTorch, JAX, CuPy, TensorFlow).

References
----------
.. [1] Genans, F., Godichon-Baggioni, A., Vialard, F.-X., Wintenberger, O.
   (2025). "Stochastic Optimization in Semi-Discrete Optimal Transport:
   Convergence Analysis and Minimax Rate." NeurIPS 2025.
.. [2] Genans, F., Godichon-Baggioni, A., Vialard, F.-X., Wintenberger, O.
   (2025). "Decreasing Entropic Regularization Averaged Gradient for
   Semi-Discrete Optimal Transport." NeurIPS 2025.
"""

# Author: Ferdinand Genans <genans.ferdinand@gmail.com>
#
# License: MIT License

import math

import numpy as np

from .backend import get_backend
from .utils import dist


def _resolve_metric(metric):
    r"""Turn ``metric`` into a callable ``(x, y) -> (n_samples, n_atoms)`` matrix.

    ``None`` defaults to ``'sqeuclidean'``. A string is forwarded to
    :func:`ot.dist`; a callable is returned unchanged.
    """
    if metric is None:
        metric = "sqeuclidean"
    if callable(metric):
        return metric
    return lambda x, y: dist(x, y, metric=metric)


def _setup(X_target, a_target, metric):
    """Resolve backend, default weights and metric callable."""
    nx = get_backend(X_target)
    m = X_target.shape[0]
    if a_target is None:
        a_target = nx.full((m,), 1.0 / m, type_as=X_target)
    return nx, m, a_target, nx.log(a_target), _resolve_metric(metric)


def _resolve_sampler(sampler_source, X_target, nx):
    r"""Turn ``sampler_source`` into a callable ``batch_size -> (batch_size, d)``.

    A callable is returned unchanged. A string selects a built-in sampler
    drawing in the same backend as ``X_target``:

    - ``'unif'`` / ``'unif_cube'``: uniform on the unit cube :math:`[0, 1]^d`;
    - ``'ball'`` / ``'unif_ball'``: uniform on the unit ball;
    - ``'normal'``: standard Gaussian :math:`\mathcal{N}(0, I_d)`.
    """
    if callable(sampler_source):
        return sampler_source
    d = X_target.shape[1]
    if sampler_source in ("unif", "unif_cube"):
        return lambda b: nx.rand(b, d, type_as=X_target)
    if sampler_source in ("ball", "unif_ball"):

        def sampler(b):
            z = nx.randn(b, d, type_as=X_target)
            r = nx.rand(b, 1, type_as=X_target) ** (1.0 / d)
            return r * z / nx.sqrt(nx.sum(z**2, axis=1))[:, None]

        return sampler
    if sampler_source == "normal":
        return lambda b: nx.randn(b, d, type_as=X_target)
    raise ValueError(
        f"Unknown sampler_source {sampler_source!r}. Expected a callable or one "
        "of 'unif', 'unif_cube', 'ball', 'unif_ball', 'normal'."
    )


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


def semidiscrete_atom_weights(
    X_target,
    X_source,
    semi_dual_potential,
    a_target=None,
    metric=None,
    reg=0.0,
):
    r"""(Entropic) assignment weights of source samples to target atoms.

    For target atoms :math:`(y_j)_{j=1}^M` (``X_target``) with weights
    :math:`(w_j)_j` (``a_target``), a ground cost :math:`c` (``metric``) and a
    semi-dual potential :math:`g` (``semi_dual_potential``), this returns, for
    each source sample :math:`x` (a row of ``X_source``), the (entropic)
    assignment weights :math:`\chi^\varepsilon(x, g)` ([2]_, Sec. 2.2):

    .. math::
        \chi^{\mathrm{reg}}_j(x, g) =
        \begin{cases}
        \displaystyle
        \frac{w_j \exp\!\big((g_j - c(x, y_j))/\mathrm{reg}\big)}
                {\sum_{k=1}^M w_k \exp\!\big((g_k - c(x, y_k))/\mathrm{reg}\big)},
        & \mathrm{reg} > 0, \\[1.2em]
        \mathbf{1}\big[\, j = \arg\min_k\, c(x, y_k) - g_k \,\big]
        & \mathrm{reg} = 0.
        \end{cases}

    :math:`\mathbb{E}_{X}[\chi^\varepsilon(X, g)]` is the atom marginal that
    the semi-dual gradient matches to :math:`w` ([1]_); cf.
    :func:`solve_semidiscrete`.

    Parameters
    ----------
    X_target : array-like, shape (n_atoms, d)
        Target atom positions :math:`y_j`.
    X_source : array-like, shape (n_samples, d)
        Source samples :math:`x` to assign.
    semi_dual_potential : array-like, shape (n_atoms,)
        Semi-dual potential :math:`g`, e.g. from :func:`solve_semidiscrete`.
    a_target : array-like, shape (n_atoms,), optional
        Atom weights :math:`w_j`. Defaults to uniform.
    metric : str or callable, optional
        Ground cost. A string is passed to :func:`ot.dist` (e.g.
        ``'sqeuclidean'``, ``'euclidean'``); a callable ``metric(x, y)``
        must return the ``(n_samples, n_atoms)`` cost matrix. Defaults to
        ``'sqeuclidean'`` (:math:`\|x - y\|^2`).
    reg : float, default=0.0
        Entropic regularization :math:`\varepsilon`. ``0`` gives hard,
        one-hot assignments; ``> 0`` the softmax above.

    Returns
    -------
    w : array, shape (n_samples, n_atoms)
        Row-stochastic assignment weights: ``w[i, j]`` is the (entropic)
        probability that sample :math:`x_i` is sent to atom :math:`y_j`.
        Each row sums to 1.

    References
    ----------
    .. [1] Genans, F., Godichon-Baggioni, A., Vialard, F.-X., Wintenberger, O.
       (2025). "Stochastic Optimization in Semi-Discrete Optimal Transport:
       Convergence Analysis and Minimax Rate." NeurIPS 2025.
    .. [2] Genans, F., Godichon-Baggioni, A., Vialard, F.-X., Wintenberger, O.
       (2025). "Decreasing Entropic Regularization Averaged Gradient for
       Semi-Discrete Optimal Transport." NeurIPS 2025.
    """
    nx, _, _, log_b, metric_fn = _setup(X_target, a_target, metric)
    score = semi_dual_potential[None, :] - metric_fn(X_source, X_target)
    return _atom_weights(score, reg, log_b, nx)


def semidiscrete_ot_map(
    X_target,
    X_source,
    semi_dual_potential,
    a_target=None,
    metric=None,
    reg=0.0,
):
    r"""Semi-discrete OT map (barycentric projection) induced by a potential.

    For each source sample :math:`x` (a row of ``X_source``), the transported
    position uses the (entropic) assignment weights
    :math:`\chi^\varepsilon(x, g)` of :func:`semidiscrete_atom_weights`:

    .. math::
        T(x) = \begin{cases}
            \displaystyle \sum_{j=1}^M \chi^\varepsilon_j(x, g)\, y_j
                & \varepsilon = \mathrm{reg} > 0, \\[1em]
            y_j \ \text{ for } x \in \mathbb{L}_j(g)
                & \varepsilon = 0.
        \end{cases}

    For :math:`\varepsilon > 0` this is the smoothed barycentric projection;
    for :math:`\varepsilon = 0` the weights are one-hot and :math:`T` is the
    Monge map of the (generalized) Brenier theorem ([1]_), sending :math:`x`
    to the atom of its Laguerre cell

    .. math::
        \mathbb{L}_j(g) = \big\{\, x : g^{c}(x) = c(x, y_j) - g_j \,\big\},

    cf. [2]_, Sec. 2.2.

    Parameters
    ----------
    X_target : array-like, shape (n_atoms, d)
        Target atom positions :math:`y_j`.
    X_source : array-like, shape (n_samples, d)
        Source samples :math:`x` to transport.
    semi_dual_potential : array-like, shape (n_atoms,)
        Semi-dual potential :math:`g`, e.g. from :func:`solve_semidiscrete`.
    a_target : array-like, shape (n_atoms,), optional
        Atom weights :math:`w_j`. Defaults to uniform.
    metric : str or callable, optional
        Ground cost. A string is passed to :func:`ot.dist` (e.g.
        ``'sqeuclidean'``, ``'euclidean'``); a callable ``metric(x, y)``
        must return the ``(n_samples, n_atoms)`` cost matrix. Defaults to
        ``'sqeuclidean'`` (:math:`\|x - y\|^2`).
    reg : float, default=0.0
        Entropic regularization :math:`\varepsilon`. ``0`` gives the hard
        Monge map; ``> 0`` the smoothed barycentric map.

    Returns
    -------
    T : array, shape (n_samples, d)
        Transported source positions :math:`T(x_i)`.

    References
    ----------
    .. [1] Genans, F., Godichon-Baggioni, A., Vialard, F.-X., Wintenberger, O.
       (2025). "Stochastic Optimization in Semi-Discrete Optimal Transport:
       Convergence Analysis and Minimax Rate." NeurIPS 2025.
    .. [2] Genans, F., Godichon-Baggioni, A., Vialard, F.-X., Wintenberger, O.
       (2025). "Decreasing Entropic Regularization Averaged Gradient for
       Semi-Discrete Optimal Transport." NeurIPS 2025.
    """
    w = semidiscrete_atom_weights(
        X_target,
        X_source,
        semi_dual_potential,
        a_target=a_target,
        metric=metric,
        reg=reg,
    )
    return w @ X_target


def semidiscrete_c_transform(
    X_target,
    X_source,
    semi_dual_potential,
    a_target=None,
    metric=None,
    reg=0.0,
):
    r"""(Entropic) :math:`c`-transform of a semi-dual potential.

    The vectorial :math:`(c, \varepsilon)`-transform
    :math:`g^{c,\varepsilon}` of the potential :math:`g`
    (``semi_dual_potential``), evaluated at the source samples ([1]_, Eq. (3);
    entropic form in [2]_, Eq. (4)):

    .. math::
        g^{c,\varepsilon}(x) = \begin{cases}
            \min_{j}\, \big(c(x, y_j) - g_j\big) & \varepsilon = 0, \\[4pt]
            -\varepsilon \log \sum_{j=1}^M w_j
                \exp\!\big((g_j - c(x, y_j))/\varepsilon\big) & \varepsilon > 0.
        \end{cases}

    For :math:`\varepsilon = 0` this is the standard :math:`c`-transform
    :math:`g^{c}(x) = \min_j (c(x, y_j) - g_j)` whose expectation gives the
    concave semi-dual :math:`H(g) = \mathbb{E}_X[g^{c}(X)] + \langle g, w\rangle`
    maximized by :func:`solve_semidiscrete` ([1]_, Eq. (2)).

    Parameters
    ----------
    X_target : array-like, shape (n_atoms, d)
        Target atom positions :math:`y_j`.
    X_source : array-like, shape (n_samples, d)
        Source samples :math:`x` at which to evaluate the transform.
    semi_dual_potential : array-like, shape (n_atoms,)
        Semi-dual potential :math:`g`, e.g. from :func:`solve_semidiscrete`.
    a_target : array-like, shape (n_atoms,), optional
        Atom weights :math:`w_j`. Defaults to uniform.
    metric : str or callable, optional
        Ground cost. A string is passed to :func:`ot.dist` (e.g.
        ``'sqeuclidean'``, ``'euclidean'``); a callable ``metric(x, y)``
        must return the ``(n_samples, n_atoms)`` cost matrix. Defaults to
        ``'sqeuclidean'`` (:math:`\|x - y\|^2`).
    reg : float, default=0.0
        Entropic regularization :math:`\varepsilon`. ``0`` gives the hard
        :math:`\min`; ``> 0`` the soft log-sum-exp.

    Returns
    -------
    phi : array, shape (n_samples,)
        The :math:`c`-transform :math:`g^{c,\varepsilon}(x_i)` at each sample.

    References
    ----------
    .. [1] Genans, F., Godichon-Baggioni, A., Vialard, F.-X., Wintenberger, O.
       (2025). "Stochastic Optimization in Semi-Discrete Optimal Transport:
       Convergence Analysis and Minimax Rate." NeurIPS 2025.
    .. [2] Genans, F., Godichon-Baggioni, A., Vialard, F.-X., Wintenberger, O.
       (2025). "Decreasing Entropic Regularization Averaged Gradient for
       Semi-Discrete Optimal Transport." NeurIPS 2025.
    """
    nx, _, _, log_b, metric_fn = _setup(X_target, a_target, metric)
    score = semi_dual_potential[None, :] - metric_fn(X_source, X_target)
    if reg == 0:
        return -nx.max(score, axis=1)
    return -reg * nx.logsumexp(score / reg + log_b[None, :], axis=1)


def solve_semidiscrete(
    X_target,
    sampler_source="unif",
    a_target=None,
    metric=None,
    reg=0.0,
    max_iter=10_000,
    batch_size=32,
    lr0=None,
    lr_exponent=2.0 / 3.0,
    init_potential=None,
    decreasing_reg=True,
    decreasing_reg_initial_eps=0.1,
    decreasing_reg_exponent=0.5,
    max_cost=None,
    polyak_average=True,
    log=False,
):
    r"""Solve semi-discrete OT by (projected) averaged SGD on the semi-dual.

    Maximizes the concave semi-dual objective ([1]_, Eq. (4), in its negative convex form)

    .. math::
        H(g) = \mathbb{E}_{X}[g^{c}(X)] + \langle g, w\rangle ,

    over the potential :math:`g \in \mathbb{R}^M`, where :math:`g^{c}` is the
    (entropic) :math:`c`-transform of :math:`g` (see
    :func:`semidiscrete_c_transform`).

    The base solver is the projected averaged SGD of [1]_ (Algorithm 1); the
    projection ``max_cost`` clips each iterate to the localizing set
    :math:`\{|g_j| \le \texttt{max\_cost}\}` ([1]_, Sec. 3.1). When
    ``decreasing_reg=True``, the entropic regularization is annealed along the
    iterations following the DRAG schedule of [2]_ (Algorithm 1), which
    accelerates convergence.

    With ``decreasing_reg=True`` the regularization at iteration ``t`` is
    :math:`\varepsilon_t = \max(\text{reg},\, \varepsilon_0 / t^\alpha)` — large
    at first for smoothness, then annealed towards ``reg``. This is the
    DRAG schedule of [2]_.

    Parameters
    ----------
    X_target : array-like, shape (n_atoms, d)
        Positions of the target atoms. The backend of this array drives
        all subsequent computations.
    sampler_source : str or callable, default='unif'
        Source distribution to sample from: either a callable
        ``sampler_source(batch_size)`` returning a ``(batch_size, d)`` batch in
        the backend of ``X_target``, or a built-in name -- one of ``'unif'``
        (``'unif_cube'``), ``'ball'`` (``'unif_ball'``) or ``'normal'``.
    a_target : array-like, shape (n_atoms,), optional
        Atom weights. Defaults to uniform.
    metric : str or callable, optional
        Ground cost, see ``metric`` in :func:`semidiscrete_atom_weights`.
        Defaults to ``'sqeuclidean'``.
    reg : float, default=0.0
        Entropic regularization (target value when ``decreasing_reg=True``).
    max_iter : int, default=10000
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
    max_cost : float, optional
        If given, clip each iterate to ``[-max_cost, max_cost]``.
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
    .. [1] Genans, F., Godichon-Baggioni, A., Vialard, F.-X., Wintenberger, O.
       (2025). "Stochastic Optimization in Semi-Discrete Optimal Transport:
       Convergence Analysis and Minimax Rate." NeurIPS 2025.
    .. [2] Genans, F., Godichon-Baggioni, A., Vialard, F.-X., Wintenberger, O.
       (2025). "Decreasing Entropic Regularization Averaged Gradient for
       Semi-Discrete Optimal Transport." NeurIPS 2025.

    Examples
    --------
    >>> import numpy as np
    >>> from ot.semidiscrete import solve_semidiscrete
    >>> rng = np.random.default_rng(0)
    >>> target = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
    >>> g = solve_semidiscrete(
    ...     target, lambda b: rng.random((b, 1)),
    ...     max_iter=500, batch_size=32, max_cost=1.0,
    ... )
    """
    nx, m, b, log_b, metric_fn = _setup(X_target, a_target, metric)
    sampler_source = _resolve_sampler(sampler_source, X_target, nx)

    if init_potential is None:
        g = nx.zeros((m,), type_as=X_target)
    else:
        g = init_potential + nx.zeros((m,), type_as=X_target)

    if lr0 is None:
        lr0 = math.sqrt(m * batch_size)

    g_avg = nx.zeros((m,), type_as=X_target) if polyak_average else None

    for t in range(1, max_iter + 1):
        if decreasing_reg:
            reg_t = max(reg, decreasing_reg_initial_eps / (t**decreasing_reg_exponent))
        else:
            reg_t = reg

        x = sampler_source(batch_size)
        score = g[None, :] - metric_fn(x, X_target)
        w = _atom_weights(score, reg_t, log_b, nx)
        grad = nx.mean(w, axis=0) - b

        lr_t = lr0 / (t**lr_exponent)
        g = g - lr_t * grad
        if max_cost is not None:
            g = nx.clip(g, -max_cost, max_cost)
        if polyak_average:
            g_avg = g_avg + (g - g_avg) / t

    result = g_avg if polyak_average else g
    if log:
        return result, {
            "max_iter": max_iter,
            "batch_size": batch_size,
            "max_cost": max_cost,
            "polyak_average": polyak_average,
            "last_potential": g,
        }
    return result


__all__ = [
    "solve_semidiscrete",
    "semidiscrete_atom_weights",
    "semidiscrete_ot_map",
    "semidiscrete_c_transform",
]
