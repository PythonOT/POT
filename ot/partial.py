# -*- coding: utf-8 -*-
"""
Partial OT solvers
"""

# Author: Laetitia Chapel <laetitia.chapel@irisa.fr>
# License: MIT License

import numpy as np

from .lp import emd


def partial_wasserstein_lagrange(a, b, M, reg_m=None, nb_dummies=1, log=False,
                                 **kwargs):
    r"""
    Solves the partial optimal transport problem for the quadratic cost
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = \arg\min_\gamma <\gamma,(M-\lambda)>_F

        s.t.
             \gamma\geq 0 \\
             \gamma 1 \leq a\\
             \gamma^T 1 \leq b\\
             1^T \gamma^T 1 = m \leq \min\{\|a\|_1, \|b\|_1\}


    or equivalently (see Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X.
    (2018). An interpolating distance between optimal transport and Fisher–Rao
    metrics. Foundations of Computational Mathematics, 18(1), 1-44.)

    .. math::
        \gamma = \arg\min_\gamma <\gamma,M>_F  + \sqrt(\lambda/2)
        (\|\gamma 1 - a\|_1 + \|\gamma^T 1 - b\|_1)

        s.t.
             \gamma\geq 0 \\


    where :

    - M is the metric cost matrix
    - a and b are source and target unbalanced distributions
    - :math:`\lambda` is the lagragian cost. Tuning its value allows attaining
      a given mass to be transported m

    The formulation of the problem has been proposed in [28]_


    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Unnormalized histogram of dimension dim_a
    b : np.ndarray (dim_b,)
        Unnormalized histograms of dimension dim_b
    M : np.ndarray (dim_a, dim_b)
        cost matrix for the quadratic cost
    reg_m : float, optional
        Lagragian cost
    nb_dummies : int, optional, default:1
        number of reservoir points to be added (to avoid numerical
        instabilities, increase its value if an error is raised)
    log : bool, optional
        record log if True
    **kwargs : dict
        parameters can be directly passed to the emd solver

    .. warning::
        When dealing with a large number of points, the EMD solver may face
        some instabilities, especially when the mass associated to the dummy
        point is large. To avoid them, increase the number of dummy points
        (allows a smoother repartition of the mass over the points).

    Returns
    -------
    gamma : (dim_a x dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------

    >>> import ot
    >>> a = [.1, .2]
    >>> b = [.1, .1]
    >>> M = [[0., 1.], [2., 3.]]
    >>> np.round(partial_wasserstein_lagrange(a,b,M), 2)
    array([[0.1, 0. ],
           [0. , 0.1]])
    >>> np.round(partial_wasserstein_lagrange(a,b,M,reg_m=2), 2)
    array([[0.1, 0. ],
           [0. , 0. ]])

    References
    ----------

    .. [28] Caffarelli, L. A., & McCann, R. J. (2010) Free boundaries in
       optimal transport and Monge-Ampere obstacle problems. Annals of
       mathematics, 673-730.

    See Also
    --------
    ot.partial.partial_wasserstein : Partial Wasserstein with fixed mass
    """

    if np.sum(a) > 1 or np.sum(b) > 1:
        raise ValueError("Problem infeasible. Check that a and b are in the "
                         "simplex")

    if reg_m is None:
        reg_m = np.max(M) + 1
    if reg_m < -np.max(M):
        return np.zeros((len(a), len(b)))

    eps = 1e-20
    M = np.asarray(M, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)

    M_star = M - reg_m  # modified cost matrix

    # trick to fasten the computation: select only the subset of columns/lines
    # that can have marginals greater than 0 (that is to say M < 0)
    idx_x = np.where(np.min(M_star, axis=1) < eps)[0]
    idx_y = np.where(np.min(M_star, axis=0) < eps)[0]

    # extend a, b, M with "reservoir" or "dummy" points
    M_extended = np.zeros((len(idx_x) + nb_dummies, len(idx_y) + nb_dummies))
    M_extended[:len(idx_x), :len(idx_y)] = M_star[np.ix_(idx_x, idx_y)]

    a_extended = np.append(a[idx_x], [(np.sum(a) - np.sum(a[idx_x]) +
                                       np.sum(b)) / nb_dummies] * nb_dummies)
    b_extended = np.append(b[idx_y], [(np.sum(b) - np.sum(b[idx_y]) +
                                       np.sum(a)) / nb_dummies] * nb_dummies)

    gamma_extended, log_emd = emd(a_extended, b_extended, M_extended, log=True,
                                  **kwargs)
    gamma = np.zeros((len(a), len(b)))
    gamma[np.ix_(idx_x, idx_y)] = gamma_extended[:-nb_dummies, :-nb_dummies]

    if log_emd['warning'] is not None:
        raise ValueError("Error in the EMD resolution: try to increase the"
                         " number of dummy points")
    log_emd['cost'] = np.sum(gamma * M)
    if log:
        return gamma, log_emd
    else:
        return gamma


def partial_wasserstein(a, b, M, m=None, nb_dummies=1, log=False, **kwargs):
    r"""
    Solves the partial optimal transport problem for the quadratic cost
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = \arg\min_\gamma <\gamma,M>_F

        s.t.
             \gamma\geq 0 \\
             \gamma 1 \leq a\\
             \gamma^T 1 \leq b\\
             1^T \gamma^T 1 = m \leq \min\{\|a\|_1, \|b\|_1\}


    where :

    - M is the metric cost matrix
    - a and b are source and target unbalanced distributions
    - m is the amount of mass to be transported

    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Unnormalized histogram of dimension dim_a
    b : np.ndarray (dim_b,)
        Unnormalized histograms of dimension dim_b
    M : np.ndarray (dim_a, dim_b)
        cost matrix for the quadratic cost
    m : float, optional
        amount of mass to be transported
    nb_dummies : int, optional, default:1
        number of reservoir points to be added (to avoid numerical
        instabilities, increase its value if an error is raised)
    log : bool, optional
        record log if True
    **kwargs : dict
        parameters can be directly passed to the emd solver


    .. warning::
        When dealing with a large number of points, the EMD solver may face
        some instabilities, especially when the mass associated to the dummy
        point is large. To avoid them, increase the number of dummy points
        (allows a smoother repartition of the mass over the points).


    Returns
    -------
    :math:`gamma` : (dim_a x dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------

    >>> import ot
    >>> a = [.1, .2]
    >>> b = [.1, .1]
    >>> M = [[0., 1.], [2., 3.]]
    >>> np.round(partial_wasserstein(a,b,M), 2)
    array([[0.1, 0. ],
           [0. , 0.1]])
    >>> np.round(partial_wasserstein(a,b,M,m=0.1), 2)
    array([[0.1, 0. ],
           [0. , 0. ]])

    References
    ----------
    ..  [28] Caffarelli, L. A., & McCann, R. J. (2010) Free boundaries in
        optimal transport and Monge-Ampere obstacle problems. Annals of
        mathematics, 673-730.
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2019). "Partial Gromov-
        Wasserstein with Applications on Positive-Unlabeled Learning".
        arXiv preprint arXiv:2002.08276.

    See Also
    --------
    ot.partial.partial_wasserstein_lagrange: Partial Wasserstein with
    regularization on the marginals
    ot.partial.entropic_partial_wasserstein: Partial Wasserstein with a
    entropic regularization parameter
    """

    if m is None:
        return partial_wasserstein_lagrange(a, b, M, log=log, **kwargs)
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    elif m > np.min((np.sum(a), np.sum(b))):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|a|_1, |b|_1).")

    b_extended = np.append(b, [(np.sum(a) - m) / nb_dummies] * nb_dummies)
    a_extended = np.append(a, [(np.sum(b) - m) / nb_dummies] * nb_dummies)
    M_extended = np.zeros((len(a_extended), len(b_extended)))
    M_extended[-1, -1] = np.max(M) * 1e5
    M_extended[:len(a), :len(b)] = M

    gamma, log_emd = emd(a_extended, b_extended, M_extended, log=True,
                         **kwargs)
    if log_emd['warning'] is not None:
        raise ValueError("Error in the EMD resolution: try to increase the"
                         " number of dummy points")
    log_emd['partial_w_dist'] = np.sum(M * gamma[:len(a), :len(b)])

    if log:
        return gamma[:len(a), :len(b)], log_emd
    else:
        return gamma[:len(a), :len(b)]


def partial_wasserstein2(a, b, M, m=None, nb_dummies=1, log=False, **kwargs):
    r"""
    Solves the partial optimal transport problem for the quadratic cost
    and returns the partial GW discrepancy

    The function considers the following problem:

    .. math::
        \gamma = \arg\min_\gamma <\gamma,M>_F

        s.t.
             \gamma\geq 0 \\
             \gamma 1 \leq a\\
             \gamma^T 1 \leq b\\
             1^T \gamma^T 1 = m \leq \min\{\|a\|_1, \|b\|_1\}


    where :

    - M is the metric cost matrix
    - a and b are source and target unbalanced distributions
    - m is the amount of mass to be transported

    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Unnormalized histogram of dimension dim_a
    b : np.ndarray (dim_b,)
        Unnormalized histograms of dimension dim_b
    M : np.ndarray (dim_a, dim_b)
        cost matrix for the quadratic cost
    m : float, optional
        amount of mass to be transported
    nb_dummies : int, optional, default:1
        number of reservoir points to be added (to avoid numerical
        instabilities, increase its value if an error is raised)
    log : bool, optional
        record log if True
    **kwargs : dict
        parameters can be directly passed to the emd solver


    .. warning::
        When dealing with a large number of points, the EMD solver may face
        some instabilities, especially when the mass associated to the dummy
        point is large. To avoid them, increase the number of dummy points
        (allows a smoother repartition of the mass over the points).


    Returns
    -------
    :math:`gamma` : (dim_a x dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------

    >>> import ot
    >>> a=[.1, .2]
    >>> b=[.1, .1]
    >>> M=[[0., 1.], [2., 3.]]
    >>> np.round(partial_wasserstein2(a, b, M), 1)
    0.3
    >>> np.round(partial_wasserstein2(a,b,M,m=0.1), 1)
    0.0

    References
    ----------
    ..  [28] Caffarelli, L. A., & McCann, R. J. (2010) Free boundaries in
        optimal transport and Monge-Ampere obstacle problems. Annals of
        mathematics, 673-730.
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2019). "Partial Gromov-
        Wasserstein with Applications on Positive-Unlabeled Learning".
        arXiv preprint arXiv:2002.08276.
    """

    partial_gw, log_w = partial_wasserstein(a, b, M, m, nb_dummies, log=True,
                                            **kwargs)

    log_w['T'] = partial_gw

    if log:
        return np.sum(partial_gw * M), log_w
    else:
        return np.sum(partial_gw * M)


def gwgrad_partial(C1, C2, T):
    """Compute the GW gradient. Note: we can not use the trick in [12]_  as
    the marginals may not sum to 1.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    numpy.array of shape (n_p+nb_dummies, n_u)
        gradient

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    cC1 = np.dot(C1 ** 2 / 2, np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1)))
    cC2 = np.dot(np.dot(np.ones(C1.shape[0]).reshape(1, -1), T), C2 ** 2 / 2)
    constC = cC1 + cC2
    A = -np.dot(C1, T).dot(C2.T)
    tens = constC + A
    return tens * 2


def gwloss_partial(C1, C2, T):
    """Compute the GW loss.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    GW loss
    """
    g = gwgrad_partial(C1, C2, T) * 0.5
    return np.sum(g * T)


def partial_gromov_wasserstein(C1, C2, p, q, m=None, nb_dummies=1, G0=None,
                               thres=1, numItermax=1000, tol=1e-7,
                               log=False, verbose=False, **kwargs):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F

        s.t. \gamma 1 \leq a \\
             \gamma^T 1 \leq b \\
             \gamma\geq 0 \\
             1^T \gamma^T 1 = m \leq \min\{\|a\|_1, \|b\|_1\} \\

    where :

    - M is the metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)
        =\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are the sample weights
    - m is the amount of mass to be transported

    The formulation of the problem has been proposed in [29]_


    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    m : float, optional
        Amount of mass to be transported (default: min (|p|_1, |q|_1))
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : ndarray, shape (ns, nt), optional
        Initialisation of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    Returns
    -------
    gamma : (dim_a x dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
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
           [0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.25]])

    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2019). "Partial Gromov-
        Wasserstein with Applications on Positive-Unlabeled Learning".
        arXiv preprint arXiv:2002.08276.

    """

    if m is None:
        m = np.min((np.sum(p), np.sum(q)))
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    elif m > np.min((np.sum(p), np.sum(q))):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|a|_1, |b|_1).")

    if G0 is None:
        G0 = np.outer(p, q)

    dim_G_extended = (len(p) + nb_dummies, len(q) + nb_dummies)
    q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)

    cpt = 0
    err = 1
    eps = 1e-20
    if log:
        log = {'err': []}

    while (err > tol and cpt < numItermax):

        Gprev = G0

        M = gwgrad_partial(C1, C2, G0)
        M[M < eps] = np.quantile(M, thres)

        M_emd = np.zeros(dim_G_extended)
        M_emd[:len(p), :len(q)] = M
        M_emd[-nb_dummies:, -nb_dummies:] = np.max(M) * 1e5
        M_emd = np.asarray(M_emd, dtype=np.float64)

        Gc, logemd = emd(p_extended, q_extended, M_emd, log=True, **kwargs)

        if logemd['warning'] is not None:
            raise ValueError("Error in the EMD resolution: try to increase the"
                             " number of dummy points")

        G0 = Gc[:len(p), :len(q)]

        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}|{:12s}'.format(
                        'It.', 'Err', 'Loss') + '\n' + '-' * 31)
                print('{:5d}|{:8e}|{:8e}'.format(cpt, err,
                                                 gwloss_partial(C1, C2, G0)))

        cpt += 1

    if log:
        log['partial_gw_dist'] = gwloss_partial(C1, C2, G0)
        return G0[:len(p), :len(q)], log
    else:
        return G0[:len(p), :len(q)]


def partial_gromov_wasserstein2(C1, C2, p, q, m=None, nb_dummies=1, G0=None,
                                thres=1, numItermax=1000, tol=1e-7,
                                log=False, verbose=False, **kwargs):
    r"""
    Solves the partial optimal transport problem
    and returns the partial Gromov-Wasserstein discrepancy

    The function considers the following problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F

        s.t. \gamma 1 \leq a \\
             \gamma^T 1 \leq b \\
             \gamma\geq 0 \\
             1^T \gamma^T 1 = m \leq \min\{\|a\|_1, \|b\|_1\} \\

    where :

    - M is the metric cost matrix
    - :math:`\Omega`  is the entropic regularization term
        :math:`\Omega=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are the sample weights
    - m is the amount of mass to be transported

    The formulation of the problem has been proposed in [29]_


    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    m : float, optional
        Amount of mass to be transported (default: min (|p|_1, |q|_1))
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : ndarray, shape (ns, nt), optional
        Initialisation of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
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
    partial_gw_dist : (dim_a x dim_b) ndarray
        partial GW discrepancy
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(partial_gromov_wasserstein2(C1, C2, a, b),2)
    1.69
    >>> np.round(partial_gromov_wasserstein2(C1, C2, a, b, m=0.25),2)
    0.0

    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2019). "Partial Gromov-
        Wasserstein with Applications on Positive-Unlabeled Learning".
        arXiv preprint arXiv:2002.08276.

    """

    partial_gw, log_gw = partial_gromov_wasserstein(C1, C2, p, q, m,
                                                    nb_dummies, G0, thres,
                                                    numItermax, tol, True,
                                                    verbose, **kwargs)

    log_gw['T'] = partial_gw

    if log:
        return log_gw['partial_gw_dist'], log_gw
    else:
        return log_gw['partial_gw_dist']


def entropic_partial_wasserstein(a, b, M, reg, m=None, numItermax=1000,
                                 stopThr=1e-100, verbose=False, log=False):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 \leq a \\
             \gamma^T 1 \leq b \\
             \gamma\geq 0 \\
             1^T \gamma^T 1 = m \leq \min\{\|a\|_1, \|b\|_1\} \\

    where :

    - M is the metric cost matrix
    - :math:`\Omega`  is the entropic regularization term
        :math:`\Omega=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are the sample weights
    - m is the amount of mass to be transported

    The formulation of the problem has been proposed in [3]_ (prop. 5)


    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Unnormalized histogram of dimension dim_a
    b : np.ndarray (dim_b,)
        Unnormalized histograms of dimension dim_b
    M : np.ndarray (dim_a, dim_b)
        cost matrix
    reg : float
        Regularization term > 0
    m : float, optional
        Amount of mass to be transported
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (dim_a x dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> a = [.1, .2]
    >>> b = [.1, .1]
    >>> M = [[0., 1.], [2., 3.]]
    >>> np.round(entropic_partial_wasserstein(a, b, M, 1, 0.1), 2)
    array([[0.06, 0.02],
           [0.01, 0.  ]])

    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
       (2015). Iterative Bregman projections for regularized transportation
       problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    See Also
    --------
    ot.partial.partial_wasserstein: exact Partial Wasserstein
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    dim_a, dim_b = M.shape
    dx = np.ones(dim_a, dtype=np.float64)
    dy = np.ones(dim_b, dtype=np.float64)

    if len(a) == 0:
        a = np.ones(dim_a, dtype=np.float64) / dim_a
    if len(b) == 0:
        b = np.ones(dim_b, dtype=np.float64) / dim_b

    if m is None:
        m = np.min((np.sum(a), np.sum(b))) * 1.0
    if m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    if m > np.min((np.sum(a), np.sum(b))):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|a|_1, |b|_1).")

    log_e = {'err': []}

    # Next 3 lines equivalent to K=np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)
    np.multiply(K, m / np.sum(K), out=K)

    err, cpt = 1, 0

    while (err > stopThr and cpt < numItermax):
        Kprev = K
        K1 = np.dot(np.diag(np.minimum(a / np.sum(K, axis=1), dx)), K)
        K2 = np.dot(K1, np.diag(np.minimum(b / np.sum(K1, axis=0), dy)))
        K = K2 * (m / np.sum(K2))

        if np.any(np.isnan(K)) or np.any(np.isinf(K)):
            print('Warning: numerical errors at iteration', cpt)
            break
        if cpt % 10 == 0:
            err = np.linalg.norm(Kprev - K)
            if log:
                log_e['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 11)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt = cpt + 1
    log_e['partial_w_dist'] = np.sum(M * K)
    if log:
        return K, log_e
    else:
        return K


def entropic_partial_gromov_wasserstein(C1, C2, p, q, reg, m=None, G0=None,
                                        numItermax=1000, tol=1e-7, log=False,
                                        verbose=False):
    r"""
    Returns the partial Gromov-Wasserstein transport between (C1,p) and (C2,q)

    The function solves the following optimization problem:

    .. math::
        GW = \arg\min_{\gamma} \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})\cdot
            \gamma_{i,j}\cdot\gamma_{k,l} + reg\cdot\Omega(\gamma)

        s.t.
             \gamma\geq 0 \\
             \gamma 1 \leq a\\
             \gamma^T 1 \leq b\\
             1^T \gamma^T 1 = m \leq \min\{\|a\|_1, \|b\|_1\}

    where :

    - C1 is the metric cost matrix in the source space
    - C2 is the metric cost matrix in the target space
    - p and q are the sample weights
    - L  : quadratic loss function
    - :math:`\Omega`  is the entropic regularization term
        :math:`\Omega=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - m is the amount of mass to be transported

    The formulation of the GW problem has been proposed in [12]_ and the
    partial GW in [29]_.

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    reg: float
        entropic regularization parameter
    m : float, optional
        Amount of mass to be transported (default: min (|p|_1, |q|_1))
    G0 : ndarray, shape (ns, nt), optional
        Initialisation of the transportation matrix
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations

    Examples
    --------
    >>> import ot
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(entropic_partial_gromov_wasserstein(C1, C2, a, b,50), 2)
    array([[0.12, 0.13, 0.  , 0.  ],
           [0.13, 0.12, 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])
    >>> np.round(entropic_partial_gromov_wasserstein(C1, C2, a, b, 50, m=0.25), 2)
    array([[0.02, 0.03, 0.  , 0.03],
           [0.03, 0.03, 0.  , 0.03],
           [0.  , 0.  , 0.03, 0.  ],
           [0.02, 0.02, 0.  , 0.03]])

    Returns
    -------
    :math: `gamma` : (dim_a x dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2019). "Partial Gromov-
        Wasserstein with Applications on Positive-Unlabeled Learning".
        arXiv preprint arXiv:2002.08276.

    See Also
    --------
    ot.partial.partial_gromov_wasserstein: exact Partial Gromov-Wasserstein
    """

    if G0 is None:
        G0 = np.outer(p, q)

    if m is None:
        m = np.min((np.sum(p), np.sum(q)))
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    elif m > np.min((np.sum(p), np.sum(q))):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|a|_1, |b|_1).")

    cpt = 0
    err = 1

    loge = {'err': []}

    while (err > tol and cpt < numItermax):
        Gprev = G0
        M_entr = gwgrad_partial(C1, C2, G0)
        G0 = entropic_partial_wasserstein(p, q, M_entr, reg, m)
        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                loge['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}|{:12s}'.format(
                        'It.', 'Err', 'Loss') + '\n' + '-' * 31)
                print('{:5d}|{:8e}|{:8e}'.format(cpt, err,
                                                 gwloss_partial(C1, C2, G0)))

        cpt += 1

    if log:
        loge['partial_gw_dist'] = gwloss_partial(C1, C2, G0)
        return G0, loge
    else:
        return G0


def entropic_partial_gromov_wasserstein2(C1, C2, p, q, reg, m=None, G0=None,
                                         numItermax=1000, tol=1e-7, log=False,
                                         verbose=False):
    r"""
    Returns the partial Gromov-Wasserstein discrepancy between (C1,p) and
    (C2,q)

    The function solves the following optimization problem:

    .. math::
        GW = \arg\min_{\gamma} \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})\cdot
            \gamma_{i,j}\cdot\gamma_{k,l} + reg\cdot\Omega(\gamma)

        s.t.
             \gamma\geq 0 \\
             \gamma 1 \leq a\\
             \gamma^T 1 \leq b\\
             1^T \gamma^T 1 = m \leq \min\{\|a\|_1, \|b\|_1\}

    where :

    - C1 is the metric cost matrix in the source space
    - C2 is the metric cost matrix in the target space
    - p and q are the sample weights
    - L  : quadratic loss function
    - :math:`\Omega`  is the entropic regularization term
        :math:`\Omega=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - m is the amount of mass to be transported

    The formulation of the GW problem has been proposed in [12]_ and the
    partial GW in [29]_.


    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    reg: float
        entropic regularization parameter
    m : float, optional
        Amount of mass to be transported (default: min (|p|_1, |q|_1))
    G0 : ndarray, shape (ns, nt), optional
        Initialisation of the transportation matrix
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations


    Returns
    -------
    partial_gw_dist: float
        Gromov-Wasserstein distance
    log : dict
        log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(entropic_partial_gromov_wasserstein2(C1, C2, a, b,50), 2)
    1.87

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2019). "Partial Gromov-
        Wasserstein with Applications on Positive-Unlabeled Learning".
        arXiv preprint arXiv:2002.08276.
    """

    partial_gw, log_gw = entropic_partial_gromov_wasserstein(C1, C2, p, q, reg,
                                                             m, G0, numItermax,
                                                             tol, True,
                                                             verbose)

    log_gw['T'] = partial_gw

    if log:
        return log_gw['partial_gw_dist'], log_gw
    else:
        return log_gw['partial_gw_dist']
