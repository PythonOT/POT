# -*- coding: utf-8 -*-
"""
Partial (Fused) Gromov-Wasserstein solvers.
"""

# Author: Laetitia Chapel <laetitia.chapel@irisa.fr>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#         Yikun Bai < yikun.bai@vanderbilt.edu >
#
# License: MIT License

from ..utils import list_to_array, unif
from ..backend import get_backend, NumpyBackend
from ..partial import entropic_partial_wasserstein
from ._utils import _transform_matrix, gwloss, gwggrad
from ..optim import partial_cg, solve_1d_linesearch_quad

import numpy as np
import warnings


def partial_gromov_wasserstein(
    C1,
    C2,
    p=None,
    q=None,
    m=None,
    loss_fun="square_loss",
    nb_dummies=1,
    G0=None,
    thres=1,
    numItermax=1e4,
    tol=1e-8,
    symmetric=None,
    warn=True,
    log=False,
    verbose=False,
    **kwargs,
):
    r"""
    Returns the Partial Gromov-Wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})`
    and :math:`(\mathbf{C_2}, \mathbf{q})`.

    The function solves the following optimization problem using Conditional Gradient:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) T_{i,j} T_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

             \mathbf{1}^T \mathbf{T}^T \mathbf{1} = m &\leq \min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}

    where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space.
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space.
    - :math:`\mathbf{p}`: Distribution in the source space.
    - :math:`\mathbf{q}`: Distribution in the target space.
    - `m` is the amount of mass to be transported
    - `L`: Loss function to account for the misfit between the similarity matrices.

    The formulation of the problem has been proposed in
    :ref:`[29] <references-partial-gromov-wasserstein>`

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.
    .. note:: All computations in the conjugate gradient solver are done with
        numpy to limit memory overhead.
    .. note:: This function will cast the computed transport plan to the data
        type of the provided input :math:`\mathbf{C}_1`. Casting to an integer
        tensor might result in a loss of precision. If this behaviour is
        unwanted, please make sure to provide a floating point input.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric costfr matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    m : float, optional
        Amount of mass to be transported
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    loss_fun : str, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'.
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : array-like, shape (ns, nt), optional
        Initialization of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    warn: bool, optional.
        Whether to raise a warning when EMD did not converge.
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal transport matrix between the two spaces.

    log : dict
        Convergence information and loss.

    Examples
    --------
    >>> from ot.gromov import partial_gromov_wasserstein
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b),2)
    array([[0.  , 0.25, 0.  , 0.  ],
           [0.25, 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b, m=0.25),2)
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.  ]])


    .. _references-partial-gromov-wasserstein:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    """
    arr = [C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(C2.shape[0], type_as=C1)
    if G0 is not None:
        G0_ = G0
        arr.append(G0)

    nx = get_backend(*arr)
    p0, q0, C10, C20 = p, q, C1, C2

    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)
    if symmetric is None:
        symmetric = np.allclose(C1, C1.T, atol=1e-10) and np.allclose(
            C2, C2.T, atol=1e-10
        )

    if m is None:
        m = min(np.sum(p), np.sum(q))
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater than 0.")
    elif m > min(np.sum(p), np.sum(q)):
        raise ValueError(
            "Problem infeasible. Parameter m should lower or"
            " equal than min(|a|_1, |b|_1)."
        )

    if G0 is None:
        G0 = (
            np.outer(p, q) * m / (np.sum(p) * np.sum(q))
        )  # make sure |G0|=m, G01_m\leq p, G0.T1_n\leq q.

    else:
        G0 = nx.to_numpy(G0_)
        # Check marginals of G0
        assert np.all(G0.sum(1) <= p)
        assert np.all(G0.sum(0) <= q)

    q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)

    # cg for GW is implemented using numpy on CPU
    np_ = NumpyBackend()

    fC1, fC2, hC1, hC2 = _transform_matrix(C1, C2, loss_fun, np_)
    fC2t = fC2.T
    if not symmetric:
        fC1t, hC1t, hC2t = fC1.T, hC1.T, hC2.T

    ones_p = np_.ones(p.shape[0], type_as=p)
    ones_q = np_.ones(q.shape[0], type_as=q)

    def f(G):
        pG = G.sum(1)
        qG = G.sum(0)
        constC1 = np.outer(np.dot(fC1, pG), ones_q)
        constC2 = np.outer(ones_p, np.dot(qG, fC2t))
        return gwloss(constC1 + constC2, hC1, hC2, G, np_)

    if symmetric:

        def df(G):
            pG = G.sum(1)
            qG = G.sum(0)
            constC1 = np.outer(np.dot(fC1, pG), ones_q)
            constC2 = np.outer(ones_p, np.dot(qG, fC2t))
            return gwggrad(constC1 + constC2, hC1, hC2, G, np_)
    else:

        def df(G):
            pG = G.sum(1)
            qG = G.sum(0)
            constC1 = np.outer(np.dot(fC1, pG), ones_q)
            constC2 = np.outer(ones_p, np.dot(qG, fC2t))
            constC1t = np.outer(np.dot(fC1t, pG), ones_q)
            constC2t = np.outer(ones_p, np.dot(qG, fC2))

            return 0.5 * (
                gwggrad(constC1 + constC2, hC1, hC2, G, np_)
                + gwggrad(constC1t + constC2t, hC1t, hC2t, G, np_)
            )

    def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
        df_Gc = df(deltaG + G)
        return solve_partial_gromov_linesearch(
            G, deltaG, cost_G, df_G, df_Gc, M=0.0, reg=1.0, nx=np_, **kwargs
        )

    if not nx.is_floating_point(C10):
        warnings.warn(
            "Input structure matrix consists of integers. The transport plan will be "
            "casted accordingly, possibly resulting in a loss of precision. "
            "If this behaviour is unwanted, please make sure your input "
            "structure matrix consists of floating point elements.",
            stacklevel=2,
        )

    if log:
        res, log = partial_cg(
            p,
            q,
            p_extended,
            q_extended,
            0.0,
            1.0,
            f,
            df,
            G0,
            line_search,
            log=True,
            numItermax=numItermax,
            stopThr=tol,
            stopThr2=0.0,
            warn=warn,
            **kwargs,
        )
        log["partial_gw_dist"] = nx.from_numpy(log["loss"][-1], type_as=C10)
        return nx.from_numpy(res, type_as=C10), log
    else:
        return nx.from_numpy(
            partial_cg(
                p,
                q,
                p_extended,
                q_extended,
                0.0,
                1.0,
                f,
                df,
                G0,
                line_search,
                log=False,
                numItermax=numItermax,
                stopThr=tol,
                stopThr2=0.0,
                **kwargs,
            ),
            type_as=C10,
        )


def partial_gromov_wasserstein2(
    C1,
    C2,
    p=None,
    q=None,
    m=None,
    loss_fun="square_loss",
    nb_dummies=1,
    G0=None,
    thres=1,
    numItermax=1e4,
    tol=1e-7,
    symmetric=None,
    warn=False,
    log=False,
    verbose=False,
    **kwargs,
):
    r"""
    Returns the Partial Gromov-Wasserstein discrepancy between
    :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`.

    The function solves the following optimization problem using Conditional Gradient:

    .. math::
        \mathbf{PGW} = \mathop{\min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) T_{i,j} T_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

             \mathbf{1}^T \mathbf{T}^T \mathbf{1} = m &\leq \min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}

    where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space.
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space.
    - :math:`\mathbf{p}`: Distribution in the source space.
    - :math:`\mathbf{q}`: Distribution in the target space.
    - `m` is the amount of mass to be transported
    - `L`: Loss function to account for the misfit between the similarity matrices.


    The formulation of the problem has been proposed in
    :ref:`[29] <references-partial-gromov-wasserstein2>`

    Note that when using backends, this loss function is differentiable wrt the
    matrices (C1, C2).

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.
    .. note:: All computations in the conjugate gradient solver are done with
        numpy to limit memory overhead.
    .. note:: This function will cast the computed transport plan to the data
        type of the provided input :math:`\mathbf{C}_1`. Casting to an integer
        tensor might result in a loss of precision. If this behaviour is
        unwanted, please make sure to provide a floating point input.

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric cost matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    m : float, optional
        Amount of mass to be transported
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    loss_fun : str, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'.
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : ndarray, shape (ns, nt), optional
        Initialization of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    warn: bool, optional.
        Whether to raise a warning when EMD did not converge.
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    .. warning::
        When dealing with a large number of points, the EMD solver may face
        some instabilities, especially when the mass associated to the dummy
        point is large. To avoid them, increase the number of dummy points
        (allows a smoother repartition of the mass over the points).


    Returns
    -------
    partial_gw_dist : float
        partial GW discrepancy
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> from ot.gromov import partial_gromov_wasserstein2
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(partial_gromov_wasserstein2(C1, C2, a, b),2)
    3.38
    >>> np.round(partial_gromov_wasserstein2(C1, C2, a, b, m=0.25),2)
    0.0


    .. _references-partial-gromov-wasserstein2:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    """
    # simple get_backend as the full one will be handled in gromov_wasserstein
    nx = get_backend(C1, C2)

    # init marginals if set as None
    if p is None:
        p = unif(C1.shape[0], type_as=C1)
    if q is None:
        q = unif(C2.shape[0], type_as=C1)

    T, log_pgw = partial_gromov_wasserstein(
        C1,
        C2,
        p,
        q,
        m,
        loss_fun,
        nb_dummies,
        G0,
        thres,
        numItermax,
        tol,
        symmetric,
        warn,
        True,
        verbose,
        **kwargs,
    )

    log_pgw["T"] = T
    pgw = log_pgw["partial_gw_dist"]

    if loss_fun == "square_loss":
        gC1 = 2 * C1 * nx.outer(p, p) - 2 * nx.dot(T, nx.dot(C2, T.T))
        gC2 = 2 * C2 * nx.outer(q, q) - 2 * nx.dot(T.T, nx.dot(C1, T))
    elif loss_fun == "kl_loss":
        gC1 = nx.log(C1 + 1e-15) * nx.outer(p, p) - nx.dot(
            T, nx.dot(nx.log(C2 + 1e-15), T.T)
        )
        gC2 = -nx.dot(T.T, nx.dot(C1, T)) / (C2 + 1e-15) + nx.outer(q, q)

    pgw = nx.set_gradients(pgw, (C1, C2), (gC1, gC2))

    if log:
        return pgw, log_pgw
    else:
        return pgw


def partial_fused_gromov_wasserstein(
    M,
    C1,
    C2,
    p=None,
    q=None,
    m=None,
    loss_fun="square_loss",
    alpha=0.5,
    nb_dummies=1,
    G0=None,
    thres=1,
    numItermax=1e4,
    tol=1e-8,
    symmetric=None,
    warn=True,
    log=False,
    verbose=False,
    **kwargs,
):
    r"""
    Returns the Partial Fused Gromov-Wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{F_1}, \mathbf{p})`
    and :math:`(\mathbf{C_2}, \mathbf{F_2}, \mathbf{q})`, with pairwise
    distance matrix :math:`\mathbf{M}` between node feature matrices.

    The function solves the following optimization problem using Conditional Gradient:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_\mathbf{T} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) T_{i,j} T_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

             \mathbf{1}^T \mathbf{T}^T \mathbf{1} = m &\leq \min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}

    where :

    - :math:`\mathbf{M}`: metric cost matrix between features across domains
    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space.
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space.
    - :math:`\mathbf{p}`: Distribution in the source space.
    - :math:`\mathbf{q}`: Distribution in the target space.
    - `m` is the amount of mass to be transported
    - `L`: Loss function to account for the misfit between the similarity matrices.

    The formulation of the problem has been proposed in
    :ref:`[29] <references-partial-gromov-wasserstein>`

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.
    .. note:: All computations in the conjugate gradient solver are done with
        numpy to limit memory overhead.
    .. note:: This function will cast the computed transport plan to the data
        type of the provided input :math:`\mathbf{C}_1`. Casting to an integer
        tensor might result in a loss of precision. If this behaviour is
        unwanted, please make sure to provide a floating point input.

    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric costfr matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    m : float, optional
        Amount of mass to be transported
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    loss_fun : str, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'.
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : array-like, shape (ns, nt), optional
        Initialization of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    warn: bool, optional.
        Whether to raise a warning when EMD did not converge.
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal transport matrix between the two spaces.

    log : dict
        Convergence information and loss.


    .. _references-partial-gromov-wasserstein:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.
    """
    arr = [M, C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(C2.shape[0], type_as=C1)
    if G0 is not None:
        G0_ = G0
        arr.append(G0)

    nx = get_backend(*arr)
    p0, q0, M0, C10, C20, alpha0 = p, q, M, C1, C2, alpha

    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    M = nx.to_numpy(M0)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)
    alpha = nx.to_numpy(alpha0)

    if symmetric is None:
        symmetric = np.allclose(C1, C1.T, atol=1e-10) and np.allclose(
            C2, C2.T, atol=1e-10
        )

    if m is None:
        m = min(np.sum(p), np.sum(q))
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater than 0.")
    elif m > min(np.sum(p), np.sum(q)):
        raise ValueError(
            "Problem infeasible. Parameter m should lower or"
            " equal than min(|a|_1, |b|_1)."
        )

    if G0 is None:
        G0 = (
            np.outer(p, q) * m / (np.sum(p) * np.sum(q))
        )  # make sure |G0|=m, G01_m\leq p, G0.T1_n\leq q.

    else:
        G0 = nx.to_numpy(G0_)
        # Check marginals of G0
        assert np.all(G0.sum(1) <= p)
        assert np.all(G0.sum(0) <= q)

    q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)

    # cg for GW is implemented using numpy on CPU
    np_ = NumpyBackend()

    fC1, fC2, hC1, hC2 = _transform_matrix(C1, C2, loss_fun, np_)
    fC2t = fC2.T
    if not symmetric:
        fC1t, hC1t, hC2t = fC1.T, hC1.T, hC2.T

    ones_p = np_.ones(p.shape[0], type_as=p)
    ones_q = np_.ones(q.shape[0], type_as=q)

    def f(G):
        pG = G.sum(1)
        qG = G.sum(0)
        constC1 = np.outer(np.dot(fC1, pG), ones_q)
        constC2 = np.outer(ones_p, np.dot(qG, fC2t))
        return gwloss(constC1 + constC2, hC1, hC2, G, np_)

    if symmetric:

        def df(G):
            pG = G.sum(1)
            qG = G.sum(0)
            constC1 = np.outer(np.dot(fC1, pG), ones_q)
            constC2 = np.outer(ones_p, np.dot(qG, fC2t))
            return gwggrad(constC1 + constC2, hC1, hC2, G, np_)
    else:

        def df(G):
            pG = G.sum(1)
            qG = G.sum(0)
            constC1 = np.outer(np.dot(fC1, pG), ones_q)
            constC2 = np.outer(ones_p, np.dot(qG, fC2t))
            constC1t = np.outer(np.dot(fC1t, pG), ones_q)
            constC2t = np.outer(ones_p, np.dot(qG, fC2))

            return 0.5 * (
                gwggrad(constC1 + constC2, hC1, hC2, G, np_)
                + gwggrad(constC1t + constC2t, hC1t, hC2t, G, np_)
            )

    def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
        df_Gc = df(deltaG + G)
        return solve_partial_gromov_linesearch(
            G,
            deltaG,
            cost_G,
            df_G,
            df_Gc,
            M=(1 - alpha) * M,
            reg=alpha,
            nx=np_,
            **kwargs,
        )

    if not nx.is_floating_point(C10):
        warnings.warn(
            "Input structure matrix consists of integers. The transport plan will be "
            "casted accordingly, possibly resulting in a loss of precision. "
            "If this behaviour is unwanted, please make sure your input "
            "structure matrix consists of floating point elements.",
            stacklevel=2,
        )

    if log:
        res, log = partial_cg(
            p,
            q,
            p_extended,
            q_extended,
            (1 - alpha) * M,
            alpha,
            f,
            df,
            G0,
            line_search,
            log=True,
            numItermax=numItermax,
            stopThr=tol,
            stopThr2=0.0,
            warn=warn,
            **kwargs,
        )
        log["partial_fgw_dist"] = nx.from_numpy(log["loss"][-1], type_as=C10)
        return nx.from_numpy(res, type_as=C10), log
    else:
        return nx.from_numpy(
            partial_cg(
                p,
                q,
                p_extended,
                q_extended,
                (1 - alpha) * M,
                alpha,
                f,
                df,
                G0,
                line_search,
                log=False,
                numItermax=numItermax,
                stopThr=tol,
                stopThr2=0.0,
                **kwargs,
            ),
            type_as=C10,
        )


def partial_fused_gromov_wasserstein2(
    M,
    C1,
    C2,
    p=None,
    q=None,
    m=None,
    loss_fun="square_loss",
    alpha=0.5,
    nb_dummies=1,
    G0=None,
    thres=1,
    numItermax=1e4,
    tol=1e-7,
    symmetric=None,
    warn=False,
    log=False,
    verbose=False,
    **kwargs,
):
    r"""
    Returns the Partial Fused Gromov-Wasserstein discrepancy between :math:`(\mathbf{C_1}, \mathbf{F_1}, \mathbf{p})`
    and :math:`(\mathbf{C_2}, \mathbf{F_2}, \mathbf{q})`, with pairwise
    distance matrix :math:`\mathbf{M}` between node feature matrices.

    The function solves the following optimization problem using Conditional Gradient:

    .. math::
        \mathbf{PFGW}_{\alpha} = \mathop{\min}_\mathbf{T} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) T_{i,j} T_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

             \mathbf{1}^T \mathbf{T}^T \mathbf{1} = m &\leq \min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}

    where :

    - :math:`\mathbf{M}`: metric cost matrix between features across domains
    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space.
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space.
    - :math:`\mathbf{p}`: Distribution in the source space.
    - :math:`\mathbf{q}`: Distribution in the target space.
    - `m` is the amount of mass to be transported
    - `L`: Loss function to account for the misfit between the similarity matrices.


    The formulation of the problem has been proposed in
    :ref:`[29] <references-partial-gromov-wasserstein2>`

    Note that when using backends, this loss function is differentiable wrt the
    matrices (M, C1, C2).

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.
    .. note:: All computations in the conjugate gradient solver are done with
        numpy to limit memory overhead.
    .. note:: This function will cast the computed transport plan to the data
        type of the provided input :math:`\mathbf{C}_1`. Casting to an integer
        tensor might result in a loss of precision. If this behaviour is
        unwanted, please make sure to provide a floating point input.

    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric cost matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    m : float, optional
        Amount of mass to be transported
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    loss_fun : str, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'.
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : ndarray, shape (ns, nt), optional
        Initialization of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    warn: bool, optional.
        Whether to raise a warning when EMD did not converge.
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    .. warning::
        When dealing with a large number of points, the EMD solver may face
        some instabilities, especially when the mass associated to the dummy
        point is large. To avoid them, increase the number of dummy points
        (allows a smoother repartition of the mass over the points).


    Returns
    -------
    partial_fgw_dist : float
        partial FGW discrepancy
    log : dict
        log dictionary returned only if `log` is `True`


    .. _references-partial-gromov-wasserstein2:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.
    """
    # simple get_backend as the full one will be handled in gromov_wasserstein
    nx = get_backend(M, C1, C2)

    # init marginals if set as None
    if p is None:
        p = unif(C1.shape[0], type_as=C1)
    if q is None:
        q = unif(C2.shape[0], type_as=C1)

    T, log_pfgw = partial_fused_gromov_wasserstein(
        M,
        C1,
        C2,
        p,
        q,
        m,
        loss_fun,
        alpha,
        nb_dummies,
        G0,
        thres,
        numItermax,
        tol,
        symmetric,
        warn,
        True,
        verbose,
        **kwargs,
    )

    log_pfgw["T"] = T
    pfgw = log_pfgw["partial_fgw_dist"]

    # compute separate terms for gradients and log
    lin_term = nx.sum(T * M)
    log_pfgw["quad_loss"] = pfgw - (1 - alpha) * lin_term
    log_pfgw["lin_loss"] = lin_term * (1 - alpha)
    pgw_term = log_pfgw["quad_loss"] / alpha

    if loss_fun == "square_loss":
        gC1 = 2 * C1 * nx.outer(p, p) - 2 * nx.dot(T, nx.dot(C2, T.T))
        gC2 = 2 * C2 * nx.outer(q, q) - 2 * nx.dot(T.T, nx.dot(C1, T))
    elif loss_fun == "kl_loss":
        gC1 = nx.log(C1 + 1e-15) * nx.outer(p, p) - nx.dot(
            T, nx.dot(nx.log(C2 + 1e-15), T.T)
        )
        gC2 = -nx.dot(T.T, nx.dot(C1, T)) / (C2 + 1e-15) + nx.outer(q, q)

    if isinstance(alpha, int) or isinstance(alpha, float):
        pfgw = nx.set_gradients(
            pfgw, (M, C1, C2), ((1 - alpha) * T, alpha * gC1, alpha * gC2)
        )
    else:
        pfgw = nx.set_gradients(
            pfgw,
            (M, C1, C2, alpha),
            ((1 - alpha) * T, alpha * gC1, alpha * gC2, pgw_term - lin_term),
        )
    if log:
        return pfgw, log_pfgw
    else:
        return pfgw


def solve_partial_gromov_linesearch(
    G,
    deltaG,
    cost_G,
    df_G,
    df_Gc,
    M,
    reg,
    alpha_min=None,
    alpha_max=None,
    nx=None,
    **kwargs,
):
    """
    Solve the linesearch in the FW iterations of partial (F)GW following eq.5 of :ref:`[29]`.

    Parameters
    ----------

    G : array-like, shape(ns, nt)
        The transport map at a given iteration of the FW
    deltaG : array-like, shape (ns, nt)
        Difference between the optimal map `Gc` found by linearization in the
        FW algorithm and the value at a given iteration
    cost_G : float
        Value of the cost at `G`
    df_G : array-like, shape (ns, nt)
        Gradient of the GW cost at `G`
    df_Gc : array-like, shape (ns, nt)
        Gradient of the GW cost at `Gc`
    M : array-like, shape (ns, nt)
        Cost matrix between the features.
    reg : float
        Regularization parameter.
    alpha_min : float, optional
        Minimum value for alpha
    alpha_max : float, optional
        Maximum value for alpha
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    alpha : float
        The optimal step size of the FW
    fc : int
        nb of function call. Useless here
    cost_G : float
        The value of the cost for the next iteration
    df_G : array-like, shape (ns, nt)
        Updated gradient of the GW cost

    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    """
    if nx is None:
        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(G, deltaG, df_G, df_Gc)
        else:
            nx = get_backend(G, deltaG, df_G, df_Gc, M)

    df_deltaG = df_Gc - df_G
    cost_deltaG = 0.5 * nx.sum(df_deltaG * deltaG)

    a = reg * cost_deltaG
    # formula to check for partial FGW
    b = nx.sum(M * deltaG) + reg * nx.sum(df_G * deltaG)
    alpha = solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # the new cost is deduced from the line search quadratic function
    cost_G = cost_G + a * (alpha**2) + b * alpha

    # update the gradient for next cg iteration
    df_G = df_G + alpha * df_deltaG
    return alpha, 1, cost_G, df_G


def entropic_partial_gromov_wasserstein(
    C1,
    C2,
    p=None,
    q=None,
    reg=1.0,
    m=None,
    loss_fun="square_loss",
    G0=None,
    numItermax=1000,
    tol=1e-7,
    symmetric=None,
    log=False,
    verbose=False,
):
    r"""
    Returns the partial Gromov-Wasserstein transport between
    :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`

    The function solves the following optimization problem:

    .. math::
        \mathbf{T} = \mathop{\arg \min}_{\mathbf{T}} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l})
        T_{i,j} T_{k,l} + \mathrm{reg} \Omega(\mathbf{T})

    .. math::
        s.t. \ \mathbf{T} &\geq 0

             \mathbf{T} \mathbf{1} &\leq \mathbf{a}

             \mathbf{T}^T \mathbf{1} &\leq \mathbf{b}

             \mathbf{1}^T \mathbf{T}^T \mathbf{1} = m
             &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}

    where :

    - :math:`\mathbf{C_1}` is the metric cost matrix in the source space
    - :math:`\mathbf{C_2}` is the metric cost matrix in the target space
    - :math:`\mathbf{p}` and :math:`\mathbf{q}` are the sample weights
    - `L`: quadratic loss function
    - :math:`\Omega` is the entropic regularization term,
      :math:`\Omega(\mathbf{T})=\sum_{i,j} T_{i,j}\log(T_{i,j})`
    - `m` is the amount of mass to be transported

    The formulation of the GW problem has been proposed in
    :ref:`[12] <references-entropic-partial-gromov-wasserstein>` and the
    partial GW in :ref:`[29] <references-entropic-partial-gromov-wasserstein>`

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    reg: float, optional. Default is 1.
        entropic regularization parameter
    m : float, optional
        Amount of mass to be transported (default:
        :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    loss_fun : str, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'.
    G0 : array-like, shape (ns, nt), optional
        Initialization of the transportation matrix
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations

    Examples
    --------
    >>> from ot.gromov import entropic_partial_gromov_wasserstein
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(entropic_partial_gromov_wasserstein(C1, C2, a, b, 1e2), 2)
    array([[0.12, 0.13, 0.  , 0.  ],
           [0.13, 0.12, 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])
    >>> np.round(entropic_partial_gromov_wasserstein(C1, C2, a, b, 1e2,0.25), 2)
    array([[0.02, 0.03, 0.  , 0.03],
           [0.03, 0.03, 0.  , 0.03],
           [0.  , 0.  , 0.03, 0.  ],
           [0.02, 0.02, 0.  , 0.03]])

    Returns
    -------
    T : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    .. _references-entropic-partial-gromov-wasserstein:
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    See Also
    --------
    ot.partial.partial_gromov_wasserstein: exact Partial Gromov-Wasserstein
    """

    arr = [C1, C2, G0]
    if p is not None:
        p = list_to_array(p)
        arr.append(p)
    if q is not None:
        q = list_to_array(q)
        arr.append(q)

    nx = get_backend(*arr)

    if p is None:
        p = nx.ones(C1.shape[0], type_as=C1) / C1.shape[0]
    if q is None:
        q = nx.ones(C2.shape[0], type_as=C2) / C2.shape[0]

    if m is None:
        m = min(nx.sum(p), nx.sum(q))
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater than 0.")
    elif m > min(nx.sum(p), nx.sum(q)):
        raise ValueError(
            "Problem infeasible. Parameter m should lower or"
            " equal than min(|a|_1, |b|_1)."
        )

    if G0 is None:
        G0 = (
            nx.outer(p, q) * m / (nx.sum(p) * nx.sum(q))
        )  # make sure |G0|=m, G01_m\leq p, G0.T1_n\leq q.

    else:
        # Check marginals of G0
        assert nx.any(nx.sum(G0, 1) <= p)
        assert nx.any(nx.sum(G0, 0) <= q)

    if symmetric is None:
        symmetric = np.allclose(C1, C1.T, atol=1e-10) and np.allclose(
            C2, C2.T, atol=1e-10
        )

    # Setup gradient computation
    fC1, fC2, hC1, hC2 = _transform_matrix(C1, C2, loss_fun, nx)
    fC2t = fC2.T
    if not symmetric:
        fC1t, hC1t, hC2t = fC1.T, hC1.T, hC2.T

    ones_p = nx.ones(p.shape[0], type_as=p)
    ones_q = nx.ones(q.shape[0], type_as=q)

    def f(G):
        pG = nx.sum(G, 1)
        qG = nx.sum(G, 0)
        constC1 = nx.outer(nx.dot(fC1, pG), ones_q)
        constC2 = nx.outer(ones_p, nx.dot(qG, fC2t))
        return gwloss(constC1 + constC2, hC1, hC2, G, nx)

    if symmetric:

        def df(G):
            pG = nx.sum(G, 1)
            qG = nx.sum(G, 0)
            constC1 = nx.outer(nx.dot(fC1, pG), ones_q)
            constC2 = nx.outer(ones_p, nx.dot(qG, fC2t))
            return gwggrad(constC1 + constC2, hC1, hC2, G, nx)
    else:

        def df(G):
            pG = nx.sum(G, 1)
            qG = nx.sum(G, 0)
            constC1 = nx.outer(nx.dot(fC1, pG), ones_q)
            constC2 = nx.outer(ones_p, nx.dot(qG, fC2t))
            constC1t = nx.outer(nx.dot(fC1t, pG), ones_q)
            constC2t = nx.outer(ones_p, nx.dot(qG, fC2))

            return 0.5 * (
                gwggrad(constC1 + constC2, hC1, hC2, G, nx)
                + gwggrad(constC1t + constC2t, hC1t, hC2t, G, nx)
            )

    cpt = 0
    err = 1

    loge = {"err": []}

    while err > tol and cpt < numItermax:
        Gprev = G0
        M_entr = df(G0)
        G0 = entropic_partial_wasserstein(p, q, M_entr, reg, m)
        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                loge["err"].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        "{:5s}|{:12s}|{:12s}".format("It.", "Err", "Loss")
                        + "\n"
                        + "-" * 31
                    )
                print("{:5d}|{:8e}|{:8e}".format(cpt, err, f(G0)))

        cpt += 1

    if log:
        loge["partial_gw_dist"] = f(G0)
        return G0, loge
    else:
        return G0


def entropic_partial_gromov_wasserstein2(
    C1,
    C2,
    p=None,
    q=None,
    reg=1.0,
    m=None,
    loss_fun="square_loss",
    G0=None,
    numItermax=1000,
    tol=1e-7,
    symmetric=None,
    log=False,
    verbose=False,
):
    r"""
    Returns the partial Gromov-Wasserstein discrepancy between
    :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`

    The function solves the following optimization problem:

    .. math::
        PGW = \min_{\mathbf{T}} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k},
             \mathbf{C_2}_{j,l})
             T_{i,j}T_{k,l} + \mathrm{reg} \Omega(\mathbf{T})

    .. math::
        s.t. \ \mathbf{T} &\geq 0

             \mathbf{T} \mathbf{1} &\leq \mathbf{a}

             \mathbf{T}^T \mathbf{1} &\leq \mathbf{b}

             \mathbf{1}^T \mathbf{T}^T \mathbf{1} = m &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}

    where :

    - :math:`\mathbf{C_1}` is the metric cost matrix in the source space
    - :math:`\mathbf{C_2}` is the metric cost matrix in the target space
    - :math:`\mathbf{p}` and :math:`\mathbf{q}` are the sample weights
    - `L`: Loss function to account for the misfit between the similarity matrices.
    - :math:`\Omega` is the entropic regularization term,
      :math:`\Omega(\mathbf{T})=\sum_{i,j} T_{i,j}\log(T_{i,j})`
    - `m` is the amount of mass to be transported

    The formulation of the GW problem has been proposed in
    :ref:`[12] <references-entropic-partial-gromov-wasserstein2>` and the
    partial GW in :ref:`[29] <references-entropic-partial-gromov-wasserstein2>`


    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    reg: float
        entropic regularization parameter
    m : float, optional
        Amount of mass to be transported (default:
        :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    loss_fun : str, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'.
    G0 : ndarray, shape (ns, nt), optional
        Initialization of the transportation matrix
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations


    Returns
    -------
    partial_gw_dist: float
        Partial Gromov-Wasserstein distance
    log : dict
        log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> from ot.gromov import entropic_partial_gromov_wasserstein2
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(entropic_partial_gromov_wasserstein2(C1, C2, a, b, 1e2), 2)
    3.75


    .. _references-entropic-partial-gromov-wasserstein2:
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.
    """

    partial_gw, log_gw = entropic_partial_gromov_wasserstein(
        C1, C2, p, q, reg, m, loss_fun, G0, numItermax, tol, symmetric, True, verbose
    )

    log_gw["T"] = partial_gw

    if log:
        return log_gw["partial_gw_dist"], log_gw
    else:
        return log_gw["partial_gw_dist"]


def entropic_partial_fused_gromov_wasserstein(
    M,
    C1,
    C2,
    p=None,
    q=None,
    reg=1.0,
    m=None,
    loss_fun="square_loss",
    alpha=0.5,
    G0=None,
    numItermax=1000,
    tol=1e-7,
    symmetric=None,
    log=False,
    verbose=False,
):
    r"""
    Returns the entropic partial Fused Gromov-Wasserstein transport between
    :math:`(\mathbf{C_1}, \mathbf{F_1}, \mathbf{p})` and
    :math:`(\mathbf{C_2}, \mathbf{F_2}, \mathbf{q})`, with pairwise
    distance matrix :math:`\mathbf{M}` between node feature matrices.

    The function solves the following optimization problem:

    .. math::
        \mathbf{T} = \mathop{\arg \min}_{\mathbf{T}} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F
        + \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l})
        T_{i,j} T_{k,l} + \mathrm{reg} \Omega(\mathbf{T})

    .. math::
        s.t. \ \mathbf{T} &\geq 0

             \mathbf{T} \mathbf{1} &\leq \mathbf{a}

             \mathbf{T}^T \mathbf{1} &\leq \mathbf{b}

             \mathbf{1}^T \mathbf{T}^T \mathbf{1} = m
             &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}

    where :

    - :math:`\mathbf{M}`: metric cost matrix between features across domains
    - :math:`\mathbf{C_1}` is the metric cost matrix in the source space
    - :math:`\mathbf{C_2}` is the metric cost matrix in the target space
    - :math:`\mathbf{p}` and :math:`\mathbf{q}` are the sample weights
    - `L`: quadratic loss function
    - :math:`\Omega` is the entropic regularization term,
      :math:`\Omega(\mathbf{T})=\sum_{i,j} T_{i,j}\log(T_{i,j})`
    - `m` is the amount of mass to be transported

    The formulation of the FGW problem has been proposed in
    :ref:`[24] <references-entropic-partial-fused-gromov-wasserstein>` and the
    partial GW in :ref:`[29] <references-entropic-partial-fused-gromov-wasserstein>`

    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    reg: float, optional. Default is 1.
        entropic regularization parameter
    m : float, optional
        Amount of mass to be transported (default:
        :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    loss_fun : str, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'.
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    G0 : array-like, shape (ns, nt), optional
        Initialization of the transportation matrix
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations

    Returns
    -------
    T : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    .. _references-entropic-partial-fused-gromov-wasserstein:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.

    .. [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    See Also
    --------
    ot.gromov.partial_fused_gromov_wasserstein: exact Partial Fused Gromov-Wasserstein
    """

    arr = [M, C1, C2, G0]
    if p is not None:
        p = list_to_array(p)
        arr.append(p)
    if q is not None:
        q = list_to_array(q)
        arr.append(q)

    nx = get_backend(*arr)

    if p is None:
        p = nx.ones(C1.shape[0], type_as=C1) / C1.shape[0]
    if q is None:
        q = nx.ones(C2.shape[0], type_as=C2) / C2.shape[0]

    if m is None:
        m = min(nx.sum(p), nx.sum(q))
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater" " than 0.")
    elif m > min(nx.sum(p), nx.sum(q)):
        raise ValueError(
            "Problem infeasible. Parameter m should lower or"
            " equal than min(|a|_1, |b|_1)."
        )

    if G0 is None:
        G0 = (
            nx.outer(p, q) * m / (nx.sum(p) * nx.sum(q))
        )  # make sure |G0|=m, G01_m\leq p, G0.T1_n\leq q.

    else:
        # Check marginals of G0
        assert nx.any(nx.sum(G0, 1) <= p)
        assert nx.any(nx.sum(G0, 0) <= q)

    if symmetric is None:
        symmetric = np.allclose(C1, C1.T, atol=1e-10) and np.allclose(
            C2, C2.T, atol=1e-10
        )

    # Setup gradient computation
    fC1, fC2, hC1, hC2 = _transform_matrix(C1, C2, loss_fun, nx)
    fC2t = fC2.T
    if not symmetric:
        fC1t, hC1t, hC2t = fC1.T, hC1.T, hC2.T

    ones_p = nx.ones(p.shape[0], type_as=p)
    ones_q = nx.ones(q.shape[0], type_as=q)

    def f(G):
        pG = nx.sum(G, 1)
        qG = nx.sum(G, 0)
        constC1 = nx.outer(nx.dot(fC1, pG), ones_q)
        constC2 = nx.outer(ones_p, nx.dot(qG, fC2t))
        return alpha * gwloss(constC1 + constC2, hC1, hC2, G, nx) + (
            1 - alpha
        ) * nx.sum(G * M)

    if symmetric:

        def df(G):
            pG = nx.sum(G, 1)
            qG = nx.sum(G, 0)
            constC1 = nx.outer(nx.dot(fC1, pG), ones_q)
            constC2 = nx.outer(ones_p, nx.dot(qG, fC2t))
            return alpha * gwggrad(constC1 + constC2, hC1, hC2, G, nx) + (
                1 - alpha
            ) * nx.sum(G * M)
    else:

        def df(G):
            pG = nx.sum(G, 1)
            qG = nx.sum(G, 0)
            constC1 = nx.outer(nx.dot(fC1, pG), ones_q)
            constC2 = nx.outer(ones_p, nx.dot(qG, fC2t))
            constC1t = nx.outer(nx.dot(fC1t, pG), ones_q)
            constC2t = nx.outer(ones_p, nx.dot(qG, fC2))

            return 0.5 * alpha * (
                gwggrad(constC1 + constC2, hC1, hC2, G, nx)
                + gwggrad(constC1t + constC2t, hC1t, hC2t, G, nx)
            ) + (1 - alpha) * nx.sum(G * M)

    cpt = 0
    err = 1

    loge = {"err": []}

    while err > tol and cpt < numItermax:
        Gprev = G0
        M_entr = df(G0)
        G0 = entropic_partial_wasserstein(p, q, M_entr, reg, m)
        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                loge["err"].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        "{:5s}|{:12s}|{:12s}".format("It.", "Err", "Loss")
                        + "\n"
                        + "-" * 31
                    )
                print("{:5d}|{:8e}|{:8e}".format(cpt, err, f(G0)))

        cpt += 1

    if log:
        loge["partial_fgw_dist"] = f(G0)
        return G0, loge
    else:
        return G0


def entropic_partial_fused_gromov_wasserstein2(
    M,
    C1,
    C2,
    p=None,
    q=None,
    reg=1.0,
    m=None,
    loss_fun="square_loss",
    alpha=0.5,
    G0=None,
    numItermax=1000,
    tol=1e-7,
    symmetric=None,
    log=False,
    verbose=False,
):
    r"""
    Returns the entropic partial Fused Gromov-Wasserstein discrepancy between
    :math:`(\mathbf{C_1}, \mathbf{F_1}, \mathbf{p})` and
    :math:`(\mathbf{C_2}, \mathbf{F_2}, \mathbf{q})`, with pairwise
    distance matrix :math:`\mathbf{M}` between node feature matrices.

    The function solves the following optimization problem:

    .. math::
        PGW = \min_{\mathbf{T}} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F
        + \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) T_{i,j} T_{k,l}
        + \mathrm{reg} \cdot\Omega(\mathbf{T})

    .. math::
        s.t. \ \mathbf{T} &\geq 0

             \mathbf{T} \mathbf{1} &\leq \mathbf{a}

             \mathbf{T}^T \mathbf{1} &\leq \mathbf{b}

             \mathbf{1}^T \mathbf{T}^T \mathbf{1} = m &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}

    where :

    - :math:`\mathbf{M}`: metric cost matrix between features across domains
    - :math:`\mathbf{C_1}` is the metric cost matrix in the source space
    - :math:`\mathbf{C_2}` is the metric cost matrix in the target space
    - :math:`\mathbf{p}` and :math:`\mathbf{q}` are the sample weights
    - `L`: Loss function to account for the misfit between the similarity matrices.
    - :math:`\Omega` is the entropic regularization term,
      :math:`\Omega(\mathbf{T})=\sum_{i,j} T_{i,j}\log(T_{i,j})`
    - `m` is the amount of mass to be transported

    The formulation of the FGW problem has been proposed in
    :ref:`[24] <references-entropic-partial-fused-gromov-wasserstein2>` and the
    partial GW in :ref:`[29] <references-entropic-partial-fused-gromov-wasserstein2>`

    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    reg: float
        entropic regularization parameter
    m : float, optional
        Amount of mass to be transported (default:
        :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    loss_fun : str, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'.
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    G0 : ndarray, shape (ns, nt), optional
        Initialization of the transportation matrix
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations


    Returns
    -------
    partial_fgw_dist: float
        Partial Entropic Fused Gromov-Wasserstein discrepancy
    log : dict
        log dictionary returned only if `log` is `True`

    .. _references-entropic-partial-fused-gromov-wasserstein2:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.

    .. [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.
    """
    nx = get_backend(M, C1, C2)

    T, log_pfgw = entropic_partial_fused_gromov_wasserstein(
        M,
        C1,
        C2,
        p,
        q,
        reg,
        m,
        loss_fun,
        alpha,
        G0,
        numItermax,
        tol,
        symmetric,
        True,
        verbose,
    )

    log_pfgw["T"] = T

    # setup for ot.solve_gromov
    lin_term = nx.sum(T * M)
    log_pfgw["quad_loss"] = log_pfgw["partial_fgw_dist"] - (1 - alpha) * lin_term
    log_pfgw["lin_loss"] = lin_term * (1 - alpha)

    if log:
        return log_pfgw["partial_fgw_dist"], log_pfgw
    else:
        return log_pfgw["partial_fgw_dist"]
