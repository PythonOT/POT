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
from .emd_wrap import emd_c, check_result


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
    M : (ns,nt) array-like, float
        Loss matrix (c-order array in numpy with type float64)
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


    Returns
    -------
    gamma: array-like, shape (ns, nt)
        Optimal transportation matrix for the given
        parameters
    log: dict, optional
        If input log is true, a dictionary containing the
        cost and dual variables and exit status


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

    a, b, M = list_to_array(a, b, M)
    nx = get_backend(M, a, b)

    if len(a) != 0:
        type_as = a
    elif len(b) != 0:
        type_as = b
    else:
        type_as = M

    # if empty array given then use uniform distributions
    if len(a) == 0:
        a = nx.ones((M.shape[0],), type_as=type_as) / M.shape[0]
    if len(b) == 0:
        b = nx.ones((M.shape[1],), type_as=type_as) / M.shape[1]

    # convert to numpy
    M, a, b = nx.to_numpy(M, a, b)

    # ensure float64
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64, order="C")

    # if empty array given then use uniform distributions
    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    assert (
        a.shape[0] == M.shape[0] and b.shape[0] == M.shape[1]
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

    G, cost, u, v, result_code = emd_c(a, b, M, numItermax, numThreads)

    if center_dual:
        u, v = center_ot_dual(u, v, a, b)

    if np.any(~asel) or np.any(~bsel):
        u, v = estimate_dual_null_weights(u, v, a, b, M)

    result_code_string = check_result(result_code)
    if not nx.is_floating_point(type_as):
        warnings.warn(
            "Input histogram consists of integer. The transport plan will be "
            "casted accordingly, possibly resulting in a loss of precision. "
            "If this behaviour is unwanted, please make sure your input "
            "histogram consists of floating point elements.",
            stacklevel=2,
        )
    if log:
        log = {}
        log["cost"] = cost
        log["u"] = nx.from_numpy(u, type_as=type_as)
        log["v"] = nx.from_numpy(v, type_as=type_as)
        log["warning"] = result_code_string
        log["result_code"] = result_code
        return nx.from_numpy(G, type_as=type_as), log
    return nx.from_numpy(G, type_as=type_as)


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
    M : (ns,nt) array-like, float64
        Loss matrix (for numpy c-order array with type float64)
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

    a, b, M = list_to_array(a, b, M)
    nx = get_backend(M, a, b)

    if len(a) != 0:
        type_as = a
    elif len(b) != 0:
        type_as = b
    else:
        type_as = M

    # if empty array given then use uniform distributions
    if len(a) == 0:
        a = nx.ones((M.shape[0],), type_as=type_as) / M.shape[0]
    if len(b) == 0:
        b = nx.ones((M.shape[1],), type_as=type_as) / M.shape[1]

    # store original tensors
    a0, b0, M0 = a, b, M

    # convert to numpy
    M, a, b = nx.to_numpy(M, a, b)

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64, order="C")

    assert (
        a.shape[0] == M.shape[0] and b.shape[0] == M.shape[1]
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

    if log or return_matrix:

        def f(b):
            bsel = b != 0

            G, cost, u, v, result_code = emd_c(a, b, M, numItermax, numThreads)

            if center_dual:
                u, v = center_ot_dual(u, v, a, b)

            if np.any(~asel) or np.any(~bsel):
                u, v = estimate_dual_null_weights(u, v, a, b, M)

            result_code_string = check_result(result_code)
            log = {}
            if not nx.is_floating_point(type_as):
                warnings.warn(
                    "Input histogram consists of integer. The transport plan will be "
                    "casted accordingly, possibly resulting in a loss of precision. "
                    "If this behaviour is unwanted, please make sure your input "
                    "histogram consists of floating point elements.",
                    stacklevel=2,
                )
            G = nx.from_numpy(G, type_as=type_as)
            if return_matrix:
                log["G"] = G
            log["u"] = nx.from_numpy(u, type_as=type_as)
            log["v"] = nx.from_numpy(v, type_as=type_as)
            log["warning"] = result_code_string
            log["result_code"] = result_code
            cost = nx.set_gradients(
                nx.from_numpy(cost, type_as=type_as),
                (a0, b0, M0),
                (log["u"] - nx.mean(log["u"]), log["v"] - nx.mean(log["v"]), G),
            )
            return [cost, log]
    else:

        def f(b):
            bsel = b != 0
            G, cost, u, v, result_code = emd_c(a, b, M, numItermax, numThreads)

            if center_dual:
                u, v = center_ot_dual(u, v, a, b)

            if np.any(~asel) or np.any(~bsel):
                u, v = estimate_dual_null_weights(u, v, a, b, M)

            if not nx.is_floating_point(type_as):
                warnings.warn(
                    "Input histogram consists of integer. The transport plan will be "
                    "casted accordingly, possibly resulting in a loss of precision. "
                    "If this behaviour is unwanted, please make sure your input "
                    "histogram consists of floating point elements.",
                    stacklevel=2,
                )
            G = nx.from_numpy(G, type_as=type_as)
            cost = nx.set_gradients(
                nx.from_numpy(cost, type_as=type_as),
                (a0, b0, M0),
                (
                    nx.from_numpy(u - np.mean(u), type_as=type_as),
                    nx.from_numpy(v - np.mean(v), type_as=type_as),
                    G,
                ),
            )

            check_result(result_code)
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
