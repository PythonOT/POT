# -*- coding: utf-8 -*-
"""
Gromov-Wasserstein and Fused-Gromov-Wasserstein solvers
"""

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@unice.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#
# License: MIT License

import numpy as np


from .bregman import sinkhorn
from .utils import dist, UndefinedParameter
from .optim import cg
from .lp import emd_1d, emd
from .utils import check_random_state

from scipy.sparse import issparse


def init_matrix(C1, C2, p, q, loss_fun='square_loss'):
    r"""Return loss matrices and tensors for Gromov-Wasserstein fast computation
    Returns the value of \mathcal{L}(C1,C2) \otimes T with the selected loss
    function as the loss function of Gromow-Wasserstein discrepancy.
    The matrices are computed as described in Proposition 1 in [12]
    Where :
        * C1 : Metric cost matrix in the source space
        * C2 : Metric cost matrix in the target space
        * T : A coupling between those two spaces
    The square-loss function L(a,b)=|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            * f1(a)=(a^2)
            * f2(b)=(b^2)
            * h1(a)=a
            * h2(b)=2*b
    The kl-loss function L(a,b)=a*log(a/b)-a+b is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            * f1(a)=a*log(a)-a
            * f2(b)=b
            * h1(a)=a
            * h2(b)=log(b)
    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    T :  ndarray, shape (ns, nt)
        Coupling between source and target spaces
    p : ndarray, shape (ns,)
    Returns
    -------
    constC : ndarray, shape (ns, nt)
        Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
        h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
        h2(C) matrix in Eq. (6)
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """

    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2)

        def f2(b):
            return (b**2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * np.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return np.log(b + 1e-15)

    constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                     np.ones(len(q)).reshape(1, -1))
    constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                     np.dot(q.reshape(1, -1), f2(C2).T))
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2


def tensor_product(constC, hC1, hC2, T):
    r"""Return the tensor for Gromov-Wasserstein fast computation
    The tensor is computed as described in Proposition 1 Eq. (6) in [12].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
        Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
        h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
        h2(C) matrix in Eq. (6)
    Returns
    -------
    tens : ndarray, shape (ns, nt)
        \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    A = -np.dot(hC1, T).dot(hC2.T)
    tens = constC + A
    # tens -= tens.min()
    return tens


def gwloss(constC, hC1, hC2, T):
    """Return the Loss for Gromov-Wasserstein
    The loss is computed as described in Proposition 1 Eq. (6) in [12].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
        Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
        h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
        h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
        Current value of transport matrix T
    Returns
    -------
    loss : float
        Gromov Wasserstein loss
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """

    tens = tensor_product(constC, hC1, hC2, T)

    return np.sum(tens * T)


def gwggrad(constC, hC1, hC2, T):
    """Return the gradient for Gromov-Wasserstein
    The gradient is computed as described in Proposition 2 in [12].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
        Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
        h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
        h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
        Current value of transport matrix T
    Returns
    -------
    grad : ndarray, shape (ns, nt)
           Gromov Wasserstein gradient
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    return 2 * tensor_product(constC, hC1, hC2,
                              T)  # [12] Prop. 2 misses a 2 factor


def update_square_loss(p, lambdas, T, Cs):
    """
    Updates C according to the L2 Loss kernel with the S Ts couplings
    calculated at each iteration
    Parameters
    ----------
    p : ndarray, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the S spaces' weights.
    T : list of S np.ndarray of shape (ns,N)
        The S Ts couplings calculated at each iteration.
    Cs : list of S ndarray, shape(ns,ns)
        Metric cost matrices.
    Returns
    ----------
    C : ndarray, shape (nt, nt)
        Updated C matrix.
    """
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s])
                  for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.divide(tmpsum, ppt)


def update_kl_loss(p, lambdas, T, Cs):
    """
    Updates C according to the KL Loss kernel with the S Ts couplings calculated at each iteration
    Parameters
    ----------
    p  : ndarray, shape (N,)
        Weights in the targeted barycenter.
    lambdas : list of the S spaces' weights
    T : list of S np.ndarray of shape (ns,N)
        The S Ts couplings calculated at each iteration.
    Cs : list of S ndarray, shape(ns,ns)
        Metric cost matrices.
    Returns
    ----------
    C : ndarray, shape (ns,ns)
        updated C matrix
    """
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s])
                  for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.exp(np.divide(tmpsum, ppt))


def gromov_wasserstein(C1, C2, p, q, loss_fun, log=False, armijo=False, **kwargs):
    r"""
    Returns the gromov-wasserstein transport between (C1,p) and (C2,q)
    The function solves the following optimization problem:
    .. math::
        GW = \min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
    Where :
    - C1 : Metric cost matrix in the source space
    - C2 : Metric cost matrix in the target space
    - p  : distribution in the source space
    - q  : distribution in the target space
    - L  : loss function to account for the misfit between the similarity matrices
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
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    armijo : bool, optional
        If True the steps of the line-search is found via an armijo research. Else closed form is used.
        If there is convergence issues use False.
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver
    Returns
    -------
    T : ndarray, shape (ns, nt)
        Doupling between the two spaces that minimizes:
            \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
    log : dict
        Convergence information and loss.
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    .. [13] Mémoli, Facundo. Gromov–Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.
    """

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    G0 = p[:, None] * q[None, :]

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    if log:
        res, log = cg(p, q, 0, 1, f, df, G0, log=True, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)
        log['gw_dist'] = gwloss(constC, hC1, hC2, res)
        return res, log
    else:
        return cg(p, q, 0, 1, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)


def gromov_wasserstein2(C1, C2, p, q, loss_fun, log=False, armijo=False, **kwargs):
    r"""
    Returns the gromov-wasserstein discrepancy between (C1,p) and (C2,q)
    The function solves the following optimization problem:
    .. math::
        GW = \min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
    Where :
    - C1 : Metric cost matrix in the source space
    - C2 : Metric cost matrix in the target space
    - p  : distribution in the source space
    - q  : distribution in the target space
    - L  : loss function to account for the misfit between the similarity matrices
    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric cost matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space.
    q :  ndarray, shape (nt,)
        Distribution in the target space.
    loss_fun :  str
        loss function used for the solver either 'square_loss' or 'kl_loss'
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    armijo : bool, optional
        If True the steps of the line-search is found via an armijo research. Else closed form is used.
        If there is convergence issues use False.
    Returns
    -------
    gw_dist : float
        Gromov-Wasserstein distance
    log : dict
        convergence information and Coupling marix
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    .. [13] Mémoli, Facundo. Gromov–Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.
    """

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    G0 = p[:, None] * q[None, :]

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)
    res, log_gw = cg(p, q, 0, 1, f, df, G0, log=True, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)
    log_gw['gw_dist'] = gwloss(constC, hC1, hC2, res)
    log_gw['T'] = res
    if log:
        return log_gw['gw_dist'], log_gw
    else:
        return log_gw['gw_dist']


def fused_gromov_wasserstein(M, C1, C2, p, q, loss_fun='square_loss', alpha=0.5, armijo=False, log=False, **kwargs):
    r"""
    Computes the FGW transport between two graphs see [24]
    .. math::
        \gamma = arg\min_\gamma (1-\\alpha)*<\gamma,M>_F + \\alpha* \sum_{i,j,k,l}
        L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
        s.t. \gamma 1 = p
             \gamma^T 1= q
             \gamma\geq 0
    where :
    - M is the (ns,nt) metric cost matrix
    - p and q are source and target weights (sum to 1)
    - L is a loss function to account for the misfit between the similarity matrices
    The algorithm used for solving the problem is conditional gradient as discussed in  [24]_
    Parameters
    ----------
    M : ndarray, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix representative of the structure in the source space
    C2 : ndarray, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    loss_fun : str, optional
        Loss function used for the solver
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    armijo : bool, optional
        If True the steps of the line-search is found via an armijo research. Else closed form is used.
        If there is convergence issues use False.
    log : bool, optional
        record log if True
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver
    Returns
    -------
    gamma : ndarray, shape (ns, nt)
        Optimal transportation matrix for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.
    """

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    G0 = p[:, None] * q[None, :]

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    if log:
        res, log = cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)
        log['fgw_dist'] = log['loss'][::-1][0]
        return res, log
    else:
        return cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)


def fused_gromov_wasserstein2(M, C1, C2, p, q, loss_fun='square_loss', alpha=0.5, armijo=False, log=False, **kwargs):
    r"""
    Computes the FGW distance between two graphs see [24]
    .. math::
        \min_\gamma (1-\\alpha)*<\gamma,M>_F + \\alpha* \sum_{i,j,k,l}
        L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
        s.t. \gamma 1 = p
             \gamma^T 1= q
             \gamma\geq 0
    where :
    - M is the (ns,nt) metric cost matrix
    - p and q are source and target weights (sum to 1)
    - L is a loss function to account for the misfit between the similarity matrices
    The algorithm used for solving the problem is conditional gradient as discussed in  [1]_
    Parameters
    ----------
    M : ndarray, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix respresentative of the structure in the source space.
    C2 : ndarray, shape (nt, nt)
        Metric cost matrix espresentative of the structure in the target space.
    p :  ndarray, shape (ns,)
        Distribution in the source space.
    q :  ndarray, shape (nt,)
        Distribution in the target space.
    loss_fun : str, optional
        Loss function used for the solver.
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    armijo : bool, optional
        If True the steps of the line-search is found via an armijo research.
        Else closed form is used. If there is convergence issues use False.
    log : bool, optional
        Record log if True.
    **kwargs : dict
        Parameters can be directly pased to the ot.optim.cg solver.
    Returns
    -------
    gamma : ndarray, shape (ns, nt)
        Optimal transportation matrix for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
        and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    G0 = p[:, None] * q[None, :]

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    res, log = cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)
    if log:
        log['fgw_dist'] = log['loss'][::-1][0]
        log['T'] = res
        return log['fgw_dist'], log
    else:
        return log['fgw_dist']


def GW_distance_estimation(C1, C2, p, q, loss_fun, T,
                           nb_samples_p=None, nb_samples_q=None, std=True, random_state=None):
    r"""
        Returns an approximation of the gromov-wasserstein cost between (C1,p) and (C2,q)
        with a fixed transport plan T.
        The function gives an unbiased approximation of the following equation:
        .. math::
            GW = \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
        Where :
        - C1 : Metric cost matrix in the source space
        - C2 : Metric cost matrix in the target space
        - L  : Loss function to account for the misfit between the similarity matrices
        - T  : Matrix with marginal p and q
        Parameters
        ----------
        C1 : ndarray, shape (ns, ns)
            Metric cost matrix in the source space
        C2 : ndarray, shape (nt, nt)
            Metric costfr matrix in the target space
        p :  ndarray, shape (ns,)
            Distribution in the source space
        q :  ndarray, shape (nt,)
            Distribution in the target space
        loss_fun :  function: \mathcal{R} \times \mathcal{R} \shortarrow \mathcal{R}
            Loss function used for the distance, the transport plan does not depend on the loss function
        T : csr or ndarray, shape (ns, nt)
            Transport plan matrix, either a sparse csr matrix or
        nb_samples_p : int, optional
            nb_samples_p is the number of samples (without replacement) along the first dimension of T.
        nb_samples_q : int, optional
            nb_samples_q is the number of samples along the second dimension of T, for each sample along the first.
        std : bool, optional
            Standard deviation associated with the prediction of the gromov-wasserstein cost.
        random_state : int or RandomState instance, optional
            Fix the seed for to allow reproducibility
        Returns
        -------
         : float
            Gromov-wasserstein cost
        References
        ----------
        .. [14] Kerdoncuff, Tanguy, Emonet, Rémi, Sebban, Marc
            "Sampled Gromov Wasserstein."
            Machine Learning Journal (MLJ). 2021.
        """
    generator = check_random_state(random_state)

    len_p = len(p)
    len_q = len(q)

    # It is always better to sample from the biggest distribution first.
    if len_p < len_q:
        p, q = q, p
        len_p, len_q = len_q, len_p
        C1, C2 = C2, C1
        T = T.T

    if nb_samples_p is None:
        if issparse(T):
            # If T is sparse, it probably mean that PoGroW was used, thus the number of sample is reduced
            nb_samples_p = min(int(5 * (len_p * np.log(len_p)) ** 0.5), len_p)
        else:
            nb_samples_p = len_p
    else:
        # The number of sample along the first dimension is without replacement.
        nb_samples_p = min(nb_samples_p, len_p)
    if nb_samples_q is None:
        nb_samples_q = 1
    if std:
        nb_samples_q = max(2, nb_samples_q)

    index_k = np.zeros((nb_samples_p, nb_samples_q), dtype=int)
    index_l = np.zeros((nb_samples_p, nb_samples_q), dtype=int)
    list_value_sample = np.zeros((nb_samples_p, nb_samples_p, nb_samples_q))

    index_i = generator.choice(len_p, size=nb_samples_p, p=p, replace=False)
    index_j = generator.choice(len_p, size=nb_samples_p, p=p, replace=False)

    for i in range(nb_samples_p):
        if issparse(T):
            T_indexi = T[index_i[i], :].toarray()[0]
            T_indexj = T[index_j[i], :].toarray()[0]
        else:
            T_indexi = T[index_i[i], :]
            T_indexj = T[index_j[i], :]
        # For each of the row sampled, the column is sampled.
        index_k[i] = generator.choice(len_q, size=nb_samples_q, p=T_indexi / T_indexi.sum(), replace=True)
        index_l[i] = generator.choice(len_q, size=nb_samples_q, p=T_indexj / T_indexj.sum(), replace=True)

    for n in range(nb_samples_q):
        list_value_sample[:, :, n] = loss_fun(C1[np.ix_(index_i, index_j)], C2[np.ix_(index_k[:, n], index_l[:, n])])

    if std:
        std_value = np.sum(np.std(list_value_sample, axis=2) ** 2) ** 0.5
        return np.mean(list_value_sample), std_value / (nb_samples_p * nb_samples_p)
    else:
        return np.mean(list_value_sample)


def pointwise_gromov_wasserstein(C1, C2, p, q, loss_fun,
                                 alpha=1, max_iter=100, threshold_plan=0, log=False, verbose=False, random_state=None):
    r"""
        Returns the gromov-wasserstein transport between (C1,p) and (C2,q) using a stochastic Frank-Wolfe.
        This method as a O(max_iter \times PN^2) time complexity with P the number of Sinkhorn iterations.
        The function solves the following optimization problem:
        .. math::
            GW = arg\min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
            s.t. T 1 = p
                 T^T 1= q
                 T\geq 0
        Where :
        - C1 : Metric cost matrix in the source space
        - C2 : Metric cost matrix in the target space
        - p  : distribution in the source space
        - q  : distribution in the target space
        - L  : loss function to account for the misfit between the similarity matrices
        Parameters
        ----------
        C1 : ndarray, shape (ns, ns)
            Metric cost matrix in the source space
        C2 : ndarray, shape (nt, nt)
            Metric costfr matrix in the target space
        p :  ndarray, shape (ns,)
            Distribution in the source space
        q :  ndarray, shape (nt,)
            Distribution in the target space
        loss_fun :  function: \mathcal{R} \times \mathcal{R} \shortarrow \mathcal{R}
            Loss function used for the distance, the transport plan does not depend on the loss function
        alpha : float
            Step of the Frank-Wolfe algorithm, should be between 0 and 1
        max_iter : int, optional
            Max number of iterations
        threshold_plan : float, optional
            Deleting very small value in the transport plan. If above zero, it violate the marginal constraints.
        verbose : bool, optional
            Print information along iterations
        log : bool, optional
            Gives the distance estimated and the standard deviation
        random_state : int or RandomState instance, optional
            Fix the seed for to allow reproducibility
        Returns
        -------
        T : ndarray, shape (ns, nt)
            Optimal coupling between the two spaces
        References
        ----------
        .. [14] Kerdoncuff, Tanguy, Emonet, Rémi, Sebban, Marc
            "Sampled Gromov Wasserstein."
            Machine Learning Journal (MLJ). 2021.
        """
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    len_p = len(p)
    len_q = len(q)

    generator = check_random_state(random_state)

    index = np.zeros(2, dtype=int)

    # Initialize with default marginal
    index[0] = generator.choice(len_p, size=1, p=p)
    index[1] = generator.choice(len_q, size=1, p=q)
    T = emd_1d(C1[index[0]], C2[index[1]], a=p, b=q, dense=False).tocsr()

    best_gw_dist_estimated = np.inf
    for cpt in range(max_iter):
        index[0] = generator.choice(len_p, size=1, p=p)
        T_index0 = T[index[0], :].toarray()[0]
        index[1] = generator.choice(len_q, size=1, p=T_index0 / T_index0.sum())

        if alpha == 1:
            T = emd_1d(C1[index[0]], C2[index[1]], a=p, b=q, dense=False).tocsr()
        else:
            new_T = emd_1d(C1[index[0]], C2[index[1]], a=p, b=q, dense=False).tocsr()
            T = (1 - alpha) * T + alpha * new_T
            # To limit the number of non 0, the values bellow the threshold are set to 0.
            T.data[T.data < threshold_plan] = 0
            T.eliminate_zeros()

        if cpt % 10 == 0 or cpt == (max_iter - 1):
            gw_dist_estimated = GW_distance_estimation(C1=C1, C2=C2, loss_fun=loss_fun,
                                                       p=p, q=q, T=T, std=False, random_state=generator)

            if gw_dist_estimated < best_gw_dist_estimated:
                best_gw_dist_estimated = gw_dist_estimated
                best_T = T.copy()

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Best gw estimated') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, best_gw_dist_estimated))

    if log:
        log = {}
        log["gw_dist_estimated"], log["gw_dist_std"] = GW_distance_estimation(C1=C1, C2=C2, loss_fun=loss_fun,
                                                                              p=p, q=q, T=best_T,
                                                                              random_state=generator)
        return best_T, log
    return best_T


def sampled_gromov_wasserstein(C1, C2, p, q, loss_fun,
                               nb_samples_grad=100, epsilon=1, max_iter=500, log=False, verbose=False,
                               random_state=None):
    r"""
        Returns the gromov-wasserstein transport between (C1,p) and (C2,q) using a 1-stochastic Frank-Wolfe.
        This method as a O(max_iter \times Nlog(N)) time complexity by relying on the 1D Optimal Transport solver.
        The function solves the following optimization problem:
        .. math::
            GW = arg\min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
            s.t. T 1 = p
                 T^T 1= q
                 T\geq 0
        Where :
        - C1 : Metric cost matrix in the source space
        - C2 : Metric cost matrix in the target space
        - p  : distribution in the source space
        - q  : distribution in the target space
        - L  : loss function to account for the misfit between the similarity matrices
        Parameters
        ----------
        C1 : ndarray, shape (ns, ns)
            Metric cost matrix in the source space
        C2 : ndarray, shape (nt, nt)
            Metric costfr matrix in the target space
        p :  ndarray, shape (ns,)
            Distribution in the source space
        q :  ndarray, shape (nt,)
            Distribution in the target space
        loss_fun :  function: \mathcal{R} \times \mathcal{R} \shortarrow \mathcal{R}
            Loss function used for the distance, the transport plan does not depend on the loss function
        nb_samples_grad : int
            Number of samples to approximate the gradient
        epsilon : float
            Weight of the Kullback-Leiber regularization
        max_iter : int, optional
            Max number of iterations
        verbose : bool, optional
            Print information along iterations
        log : bool, optional
            Gives the distance estimated and the standard deviation
        random_state : int or RandomState instance, optional
            Fix the seed for to allow reproducibility
        Returns
        -------
        T : ndarray, shape (ns, nt)
            Optimal coupling between the two spaces
        References
        ----------
        .. [14] Kerdoncuff, Tanguy, Emonet, Rémi, Sebban, Marc
            "Sampled Gromov Wasserstein."
            Machine Learning Journal (MLJ). 2021.
        """
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    len_p = len(p)
    len_q = len(q)

    generator = check_random_state(random_state)

    # The most natural way to define nb_sample is with a simple integer.
    if isinstance(nb_samples_grad, int):
        if nb_samples_grad > len_p:
            # As the sampling along the first dimension is done without replacement, the rest is reported to the second
            # dimension.
            nb_samples_grad_p, nb_samples_grad_q = len_p, nb_samples_grad // len_p
        else:
            nb_samples_grad_p, nb_samples_grad_q = nb_samples_grad, 1
    else:
        nb_samples_grad_p, nb_samples_grad_q = nb_samples_grad
    T = np.outer(p, q)
    # continue_loop allows to stop the loop if there is several successive small modification of T.
    continue_loop = 0

    # The gradient of GW is more complex if the two matrices are not symmetric.
    C_are_symmetric = np.allclose(C1, C1.T, rtol=1e-10, atol=1e-10) and np.allclose(C2, C2.T, rtol=1e-10, atol=1e-10)

    for cpt in range(max_iter):
        index0 = generator.choice(len_p, size=nb_samples_grad_p, p=p, replace=False)
        Lik = 0
        for i, index0_i in enumerate(index0):
            index1 = generator.choice(len_q,
                                      size=nb_samples_grad_q,
                                      p=T[index0_i, :] / T[index0_i, :].sum(),
                                      replace=False)
            # If the matrices C are not symmetric, the gradient has 2 terms, thus the term is chosen randomly.
            if (not C_are_symmetric) and generator.rand(1) > 0.5:
                Lik += np.mean(loss_fun(np.expand_dims(C1[:, np.repeat(index0[i], nb_samples_grad_q)], 1),
                                        np.expand_dims(C2[:, index1], 0)),
                               axis=2)
            else:
                Lik += np.mean(loss_fun(np.expand_dims(C1[np.repeat(index0[i], nb_samples_grad_q), :], 2),
                                        np.expand_dims(C2[index1, :], 1)),
                               axis=0)

        max_Lik = np.max(Lik)
        if max_Lik == 0:
            continue
        # This division by the max is here to facilitate the choice of epsilon.
        Lik /= max_Lik

        if epsilon > 0:
            # Set to infinity all the numbers bellow exp(-200) to avoid log of 0.
            log_T = np.log(np.clip(T, np.exp(-200), 1))
            log_T[log_T == -200] = -np.inf
            Lik = Lik - epsilon * log_T

            try:
                new_T = sinkhorn(a=p, b=q, M=Lik, reg=epsilon)
            except (RuntimeWarning, UserWarning):
                print("Warning catched in Sinkhorn: Return last stable T")
                break
        else:
            new_T = emd(a=p, b=q, M=Lik)

        change_T = ((T - new_T) ** 2).mean()
        if change_T <= 10e-20:
            continue_loop += 1
            if continue_loop > 100:  # Number max of low modifications of T
                T = new_T.copy()
                break
        else:
            continue_loop = 0

        if verbose and cpt % 10 == 0:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format('It.', '||T_n - T_{n+1}||') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, change_T))
        T = new_T.copy()

    if log:
        log = {}
        log["gw_dist_estimated"], log["gw_dist_std"] = GW_distance_estimation(C1=C1, C2=C2, loss_fun=loss_fun,
                                                                              p=p, q=q, T=T, random_state=generator)
        return T, log
    return T


def entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon,
                                max_iter=1000, tol=1e-9, verbose=False, log=False):
    r"""
    Returns the gromov-wasserstein transport between (C1,p) and (C2,q)
    (C1,p) and (C2,q)
    The function solves the following optimization problem:
    .. math::
        GW = arg\min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))
        s.t. T 1 = p
             T^T 1= q
             T\geq 0
    Where :
    - C1 : Metric cost matrix in the source space
    - C2 : Metric cost matrix in the target space
    - p  : distribution in the source space
    - q  : distribution in the target space
    - L  : loss function to account for the misfit between the similarity matrices
    - H  : entropy
    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p :  ndarray, shape (ns,)
        Distribution in the source space
    q :  ndarray, shape (nt,)
        Distribution in the target space
    loss_fun :  string
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.
    Returns
    -------
    T : ndarray, shape (ns, nt)
        Optimal coupling between the two spaces
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """

    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)

    T = np.outer(p, q)  # Initialization

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    cpt = 0
    err = 1

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # compute the gradient
        tens = gwggrad(constC, hC1, hC2, T)

        T = sinkhorn(p, q, tens, epsilon)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        log['gw_dist'] = gwloss(constC, hC1, hC2, T)
        return T, log
    else:
        return T


def entropic_gromov_wasserstein2(C1, C2, p, q, loss_fun, epsilon,
                                 max_iter=1000, tol=1e-9, verbose=False, log=False):
    r"""
    Returns the entropic gromov-wasserstein discrepancy between the two measured similarity matrices
    (C1,p) and (C2,q)
    The function solves the following optimization problem:
    .. math::
        GW = \min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))
    Where :
    - C1 : Metric cost matrix in the source space
    - C2 : Metric cost matrix in the target space
    - p  : distribution in the source space
    - q  : distribution in the target space
    - L  : loss function to account for the misfit between the similarity matrices
    - H  : entropy
    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p :  ndarray, shape (ns,)
        Distribution in the source space
    q :  ndarray, shape (nt,)
        Distribution in the target space
    loss_fun : str
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.
    Returns
    -------
    gw_dist : float
        Gromov-Wasserstein distance
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    gw, logv = entropic_gromov_wasserstein(
        C1, C2, p, q, loss_fun, epsilon, max_iter, tol, verbose, log=True)

    logv['T'] = gw

    if log:
        return logv['gw_dist'], logv
    else:
        return logv['gw_dist']


def entropic_gromov_barycenters(N, Cs, ps, p, lambdas, loss_fun, epsilon,
                                max_iter=1000, tol=1e-9, verbose=False, log=False, init_C=None):
    r"""
    Returns the gromov-wasserstein barycenters of S measured similarity matrices
    (Cs)_{s=1}^{s=S}
    The function solves the following optimization problem:
    .. math::
        C = argmin_{C\in R^{NxN}} \sum_s \lambda_s GW(C,C_s,p,p_s)
    Where :
    - :math:`C_s` : metric cost matrix
    - :math:`p_s`  : distribution
    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S np.ndarray of shape (ns,ns)
        Metric cost matrices
    ps : list of S np.ndarray of shape (ns,)
        Sample weights in the S spaces
    p : ndarray, shape(N,)
        Weights in the targeted barycenter
    lambdas : list of float
        List of the S spaces' weights.
    loss_fun : callable
        Tensor-matrix multiplication function based on specific loss function.
    update : callable
        function(p,lambdas,T,Cs) that updates C according to a specific Kernel
        with the S Ts couplings calculated at each iteration
    epsilon : float
        Regularization term >0
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | ndarray, shape (N, N)
        Random initial value for the C matrix provided by user.
    Returns
    -------
    C : ndarray, shape (N, N)
        Similarity matrix in the barycenter space (permutated arbitrarily)
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """

    S = len(Cs)

    Cs = [np.asarray(Cs[s], dtype=np.float64) for s in range(S)]
    lambdas = np.asarray(lambdas, dtype=np.float64)

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        # XXX use random state
        xalea = np.random.randn(N, 2)
        C = dist(xalea, xalea)
        C /= C.max()
    else:
        C = init_C

    cpt = 0
    err = 1

    error = []

    while (err > tol) and (cpt < max_iter):
        Cprev = C

        T = [entropic_gromov_wasserstein(Cs[s], C, ps[s], p, loss_fun, epsilon,
                                         max_iter, 1e-5, verbose, log) for s in range(S)]
        if loss_fun == 'square_loss':
            C = update_square_loss(p, lambdas, T, Cs)

        elif loss_fun == 'kl_loss':
            C = update_kl_loss(p, lambdas, T, Cs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(C - Cprev)
            error.append(err)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    return C


def gromov_barycenters(N, Cs, ps, p, lambdas, loss_fun,
                       max_iter=1000, tol=1e-9, verbose=False, log=False, init_C=None):
    r"""
    Returns the gromov-wasserstein barycenters of S measured similarity matrices
    (Cs)_{s=1}^{s=S}
    The function solves the following optimization problem with block
    coordinate descent:
    .. math::
        C = argmin_C\in R^NxN \sum_s \lambda_s GW(C,Cs,p,ps)
    Where :
    - Cs : metric cost matrix
    - ps  : distribution
    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S np.ndarray of shape (ns, ns)
        Metric cost matrices
    ps : list of S np.ndarray of shape (ns,)
        Sample weights in the S spaces
    p : ndarray, shape (N,)
        Weights in the targeted barycenter
    lambdas : list of float
        List of the S spaces' weights
    loss_fun :  tensor-matrix multiplication function based on specific loss function
    update : function(p,lambdas,T,Cs) that updates C according to a specific Kernel
             with the S Ts couplings calculated at each iteration
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshol on error (>0).
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | ndarray, shape(N,N)
        Random initial value for the C matrix provided by user.
    Returns
    -------
    C : ndarray, shape (N, N)
        Similarity matrix in the barycenter space (permutated arbitrarily)
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    S = len(Cs)

    Cs = [np.asarray(Cs[s], dtype=np.float64) for s in range(S)]
    lambdas = np.asarray(lambdas, dtype=np.float64)

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        # XXX : should use a random state and not use the global seed
        xalea = np.random.randn(N, 2)
        C = dist(xalea, xalea)
        C /= C.max()
    else:
        C = init_C

    cpt = 0
    err = 1

    error = []

    while(err > tol and cpt < max_iter):
        Cprev = C

        T = [gromov_wasserstein(Cs[s], C, ps[s], p, loss_fun,
                                numItermax=max_iter, stopThr=1e-5, verbose=verbose, log=log) for s in range(S)]
        if loss_fun == 'square_loss':
            C = update_square_loss(p, lambdas, T, Cs)

        elif loss_fun == 'kl_loss':
            C = update_kl_loss(p, lambdas, T, Cs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(C - Cprev)
            error.append(err)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    return C


def fgw_barycenters(N, Ys, Cs, ps, lambdas, alpha, fixed_structure=False, fixed_features=False,
                    p=None, loss_fun='square_loss', max_iter=100, tol=1e-9,
                    verbose=False, log=False, init_C=None, init_X=None):
    """Compute the fgw barycenter as presented eq (5) in [24].
    Parameters
    ----------
    N : integer
        Desired number of samples of the target barycenter
    Ys: list of ndarray, each element has shape (ns,d)
        Features of all samples
    Cs : list of ndarray, each element has shape (ns,ns)
        Structure matrices of all samples
    ps : list of ndarray, each element has shape (ns,)
        Masses of all samples.
    lambdas : list of float
        List of the S spaces' weights
    alpha : float
        Alpha parameter for the fgw distance
    fixed_structure : bool
        Whether to fix the structure of the barycenter during the updates
    fixed_features : bool
        Whether to fix the feature of the barycenter during the updates
    loss_fun : str
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshol on error (>0).
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : ndarray, shape (N,N), optional
        Initialization for the barycenters' structure matrix. If not set
        a random init is used.
    init_X : ndarray, shape (N,d), optional
        Initialization for the barycenters' features. If not set a
        random init is used.
    Returns
    -------
    X : ndarray, shape (N, d)
        Barycenters' features
    C : ndarray, shape (N, N)
        Barycenters' structure matrix
    log_: dict
        Only returned when log=True. It contains the keys:
        T : list of (N,ns) transport matrices
        Ms : all distance matrices between the feature of the barycenter and the
        other features dist(X,Ys) shape (N,ns)
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
        and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    S = len(Cs)
    d = Ys[0].shape[1]  # dimension on the node features
    if p is None:
        p = np.ones(N) / N

    Cs = [np.asarray(Cs[s], dtype=np.float64) for s in range(S)]
    Ys = [np.asarray(Ys[s], dtype=np.float64) for s in range(S)]

    lambdas = np.asarray(lambdas, dtype=np.float64)

    if fixed_structure:
        if init_C is None:
            raise UndefinedParameter('If C is fixed it must be initialized')
        else:
            C = init_C
    else:
        if init_C is None:
            xalea = np.random.randn(N, 2)
            C = dist(xalea, xalea)
        else:
            C = init_C

    if fixed_features:
        if init_X is None:
            raise UndefinedParameter('If X is fixed it must be initialized')
        else:
            X = init_X
    else:
        if init_X is None:
            X = np.zeros((N, d))
        else:
            X = init_X

    T = [np.outer(p, q) for q in ps]

    Ms = [np.asarray(dist(X, Ys[s]), dtype=np.float64) for s in range(len(Ys))]  # Ms is N,ns

    cpt = 0
    err_feature = 1
    err_structure = 1

    if log:
        log_ = {}
        log_['err_feature'] = []
        log_['err_structure'] = []
        log_['Ts_iter'] = []

    while((err_feature > tol or err_structure > tol) and cpt < max_iter):
        Cprev = C
        Xprev = X

        if not fixed_features:
            Ys_temp = [y.T for y in Ys]
            X = update_feature_matrix(lambdas, Ys_temp, T, p).T

        Ms = [np.asarray(dist(X, Ys[s]), dtype=np.float64) for s in range(len(Ys))]

        if not fixed_structure:
            if loss_fun == 'square_loss':
                T_temp = [t.T for t in T]
                C = update_sructure_matrix(p, lambdas, T_temp, Cs)

        T = [fused_gromov_wasserstein(Ms[s], C, Cs[s], p, ps[s], loss_fun, alpha,
                                      numItermax=max_iter, stopThr=1e-5, verbose=verbose) for s in range(S)]

        # T is N,ns
        err_feature = np.linalg.norm(X - Xprev.reshape(N, d))
        err_structure = np.linalg.norm(C - Cprev)

        if log:
            log_['err_feature'].append(err_feature)
            log_['err_structure'].append(err_structure)
            log_['Ts_iter'].append(T)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err_structure))
            print('{:5d}|{:8e}|'.format(cpt, err_feature))

        cpt += 1

    if log:
        log_['T'] = T  # from target to Ys
        log_['p'] = p
        log_['Ms'] = Ms

    if log:
        return X, C, log_
    else:
        return X, C


def update_sructure_matrix(p, lambdas, T, Cs):
    """Updates C according to the L2 Loss kernel with the S Ts couplings.
    It is calculated at each iteration
    Parameters
    ----------
    p : ndarray, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the S spaces' weights.
    T : list of S ndarray of shape (ns, N)
        The S Ts couplings calculated at each iteration.
    Cs : list of S ndarray, shape (ns, ns)
         Metric cost matrices.
    Returns
    -------
    C : ndarray, shape (nt, nt)
        Updated C matrix.
    """
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s]) for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.divide(tmpsum, ppt)


def update_feature_matrix(lambdas, Ys, Ts, p):
    """Updates the feature with respect to the S Ts couplings.
    See "Solving the barycenter problem with Block Coordinate Descent (BCD)"
    in [24] calculated at each iteration
    Parameters
    ----------
    p : ndarray, shape (N,)
        masses in the targeted barycenter
    lambdas : list of float
        List of the S spaces' weights
    Ts : list of S np.ndarray(ns,N)
        the S Ts couplings calculated at each iteration
    Ys : list of S ndarray, shape(d,ns)
        The features.
    Returns
    -------
    X : ndarray, shape (d, N)
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    p = np.array(1. / p).reshape(-1,)

    tmpsum = sum([lambdas[s] * np.dot(Ys[s], Ts[s].T) * p[None, :] for s in range(len(Ts))])

    return tmpsum