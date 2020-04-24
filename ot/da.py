# -*- coding: utf-8 -*-
"""
Domain adaptation with optimal transport
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#         Michael Perrot <michael.perrot@univ-st-etienne.fr>
#         Nathalie Gayraud <nat.gayraud@gmail.com>
#         Ievgen Redko <ievgen.redko@univ-st-etienne.fr>
#
# License: MIT License

import numpy as np
import scipy.linalg as linalg

from .bregman import sinkhorn, jcpot_barycenter
from .lp import emd
from .utils import unif, dist, kernel, cost_normalization, label_normalization, laplacian, dots
from .utils import check_params, BaseEstimator
from .unbalanced import sinkhorn_unbalanced
from .optim import cg
from .optim import gcg


def sinkhorn_lpl1_mm(a, labels_a, b, M, reg, eta=0.1, numItermax=10,
                     numInnerItermax=200, stopInnerThr=1e-9, verbose=False,
                     log=False):
    """
    Solve the entropic regularization optimal transport problem with nonconvex
    group lasso regularization

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega_e(\gamma)
        + \eta \Omega_g(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega_e` is the entropic regularization term :math:`\Omega_e
      (\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\Omega_g` is the group lasso  regularization term
      :math:`\Omega_g(\gamma)=\sum_{i,c} \|\gamma_{i,\mathcal{I}_c}\|^{1/2}_1`
      where  :math:`\mathcal{I}_c` are the index of samples from class c
      in the source domain.
    - a and b are source and target weights (sum to 1)

    The algorithm used for solving the problem is the generalized conditional
    gradient as proposed in  [5]_ [7]_


    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    labels_a : np.ndarray (ns,)
        labels of samples in the source domain
    b : np.ndarray (nt,)
        samples weights in the target domain
    M : np.ndarray (ns,nt)
        loss matrix
    reg : float
        Regularization term for entropic regularization >0
    eta : float, optional
        Regularization term  for group lasso regularization >0
    numItermax : int, optional
        Max number of iterations
    numInnerItermax : int, optional
        Max number of iterations (inner sinkhorn solver)
    stopInnerThr : float, optional
        Stop threshold on error (inner sinkhorn solver) (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
       "Optimal Transport for Domain Adaptation," in IEEE
       Transactions on Pattern Analysis and Machine Intelligence ,
       vol.PP, no.99, pp.1-1
    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015).
       Generalized conditional gradient: analysis of convergence
       and applications. arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.bregman.sinkhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT

    """
    p = 0.5
    epsilon = 1e-3

    indices_labels = []
    classes = np.unique(labels_a)
    for c in classes:
        idxc, = np.where(labels_a == c)
        indices_labels.append(idxc)

    W = np.zeros(M.shape)

    for cpt in range(numItermax):
        Mreg = M + eta * W
        transp = sinkhorn(a, b, Mreg, reg, numItermax=numInnerItermax,
                          stopThr=stopInnerThr)
        # the transport has been computed. Check if classes are really
        # separated
        W = np.ones(M.shape)
        for (i, c) in enumerate(classes):
            majs = np.sum(transp[indices_labels[i]], axis=0)
            majs = p * ((majs + epsilon) ** (p - 1))
            W[indices_labels[i]] = majs

    return transp


def sinkhorn_l1l2_gl(a, labels_a, b, M, reg, eta=0.1, numItermax=10,
                     numInnerItermax=200, stopInnerThr=1e-9, verbose=False,
                     log=False):
    """
    Solve the entropic regularization optimal transport problem with group
    lasso regularization

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega_e(\gamma)+
        \eta \Omega_g(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega_e` is the entropic regularization term
      :math:`\Omega_e(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\Omega_g` is the group lasso regulaization term
      :math:`\Omega_g(\gamma)=\sum_{i,c} \|\gamma_{i,\mathcal{I}_c}\|^2`
      where  :math:`\mathcal{I}_c` are the index of samples from class
      c in the source domain.
    - a and b are source and target weights (sum to 1)

    The algorithm used for solving the problem is the generalised conditional
    gradient as proposed in  [5]_ [7]_


    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    labels_a : np.ndarray (ns,)
        labels of samples in the source domain
    b : np.ndarray (nt,)
        samples in the target domain
    M : np.ndarray (ns,nt)
        loss matrix
    reg : float
        Regularization term for entropic regularization >0
    eta : float, optional
        Regularization term  for group lasso regularization >0
    numItermax : int, optional
        Max number of iterations
    numInnerItermax : int, optional
        Max number of iterations (inner sinkhorn solver)
    stopInnerThr : float, optional
        Stop threshold on error (inner sinkhorn solver) (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
       "Optimal Transport for Domain Adaptation," in IEEE Transactions
       on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1
    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015).
       Generalized conditional gradient: analysis of convergence and
       applications. arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.optim.gcg : Generalized conditional gradient for OT problems

    """
    lstlab = np.unique(labels_a)

    def f(G):
        res = 0
        for i in range(G.shape[1]):
            for lab in lstlab:
                temp = G[labels_a == lab, i]
                res += np.linalg.norm(temp)
        return res

    def df(G):
        W = np.zeros(G.shape)
        for i in range(G.shape[1]):
            for lab in lstlab:
                temp = G[labels_a == lab, i]
                n = np.linalg.norm(temp)
                if n:
                    W[labels_a == lab, i] = temp / n
        return W

    return gcg(a, b, M, reg, eta, f, df, G0=None, numItermax=numItermax,
               numInnerItermax=numInnerItermax, stopThr=stopInnerThr,
               verbose=verbose, log=log)


def joint_OT_mapping_linear(xs, xt, mu=1, eta=0.001, bias=False, verbose=False,
                            verbose2=False, numItermax=100, numInnerItermax=10,
                            stopInnerThr=1e-6, stopThr=1e-5, log=False,
                            **kwargs):
    """Joint OT and linear mapping estimation as proposed in [8]

    The function solves the following optimization problem:

    .. math::
        \min_{\gamma,L}\quad \|L(X_s) -n_s\gamma X_t\|^2_F +
          \mu<\gamma,M>_F + \eta  \|L -I\|^2_F

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (ns,nt) squared euclidean cost matrix between samples in
       Xs and Xt (scaled by ns)
    - :math:`L` is a dxd linear operator that approximates the barycentric
      mapping
    - :math:`I` is the identity matrix (neutral linear mapping)
    - a and b are uniform source and target weights

    The problem consist in solving jointly an optimal transport matrix
    :math:`\gamma` and a linear mapping that fits the barycentric mapping
    :math:`n_s\gamma X_t`.

    One can also estimate a mapping with constant bias (see supplementary
    material of [8]) using the bias optional argument.

    The algorithm used for solving the problem is the block coordinate
    descent that alternates between updates of G (using conditionnal gradient)
    and the update of L using a classical least square solver.


    Parameters
    ----------
    xs : np.ndarray (ns,d)
        samples in the source domain
    xt : np.ndarray (nt,d)
        samples in the target domain
    mu : float,optional
        Weight for the linear OT loss (>0)
    eta : float, optional
        Regularization term  for the linear mapping L (>0)
    bias : bool,optional
        Estimate linear mapping with constant bias
    numItermax : int, optional
        Max number of BCD iterations
    stopThr : float, optional
        Stop threshold on relative loss decrease (>0)
    numInnerItermax : int, optional
        Max number of iterations (inner CG solver)
    stopInnerThr : float, optional
        Stop threshold on error (inner CG solver) (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    L : (d x d) ndarray
        Linear mapping matrix (d+1 x d if bias)
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
        "Mapping estimation for discrete optimal transport",
        Neural Information Processing Systems (NIPS), 2016.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    ns, nt, d = xs.shape[0], xt.shape[0], xt.shape[1]

    if bias:
        xs1 = np.hstack((xs, np.ones((ns, 1))))
        xstxs = xs1.T.dot(xs1)
        Id = np.eye(d + 1)
        Id[-1] = 0
        I0 = Id[:, :-1]

        def sel(x):
            return x[:-1, :]
    else:
        xs1 = xs
        xstxs = xs1.T.dot(xs1)
        Id = np.eye(d)
        I0 = Id

        def sel(x):
            return x

    if log:
        log = {'err': []}

    a, b = unif(ns), unif(nt)
    M = dist(xs, xt) * ns
    G = emd(a, b, M)

    vloss = []

    def loss(L, G):
        """Compute full loss"""
        return np.sum((xs1.dot(L) - ns * G.dot(xt)) ** 2) + mu * \
            np.sum(G * M) + eta * np.sum(sel(L - I0) ** 2)

    def solve_L(G):
        """ solve L problem with fixed G (least square)"""
        xst = ns * G.dot(xt)
        return np.linalg.solve(xstxs + eta * Id, xs1.T.dot(xst) + eta * I0)

    def solve_G(L, G0):
        """Update G with CG algorithm"""
        xsi = xs1.dot(L)

        def f(G):
            return np.sum((xsi - ns * G.dot(xt)) ** 2)

        def df(G):
            return -2 * ns * (xsi - ns * G.dot(xt)).dot(xt.T)

        G = cg(a, b, M, 1.0 / mu, f, df, G0=G0,
               numItermax=numInnerItermax, stopThr=stopInnerThr)
        return G

    L = solve_L(G)

    vloss.append(loss(L, G))

    if verbose:
        print('{:5s}|{:12s}|{:8s}'.format(
            'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
        print('{:5d}|{:8e}|{:8e}'.format(0, vloss[-1], 0))

    # init loop
    if numItermax > 0:
        loop = 1
    else:
        loop = 0
    it = 0

    while loop:

        it += 1

        # update G
        G = solve_G(L, G)

        # update L
        L = solve_L(G)

        vloss.append(loss(L, G))

        if it >= numItermax:
            loop = 0

        if abs(vloss[-1] - vloss[-2]) / abs(vloss[-2]) < stopThr:
            loop = 0

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}'.format(
                    'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
            print('{:5d}|{:8e}|{:8e}'.format(
                it, vloss[-1], (vloss[-1] - vloss[-2]) / abs(vloss[-2])))
    if log:
        log['loss'] = vloss
        return G, L, log
    else:
        return G, L


def joint_OT_mapping_kernel(xs, xt, mu=1, eta=0.001, kerneltype='gaussian',
                            sigma=1, bias=False, verbose=False, verbose2=False,
                            numItermax=100, numInnerItermax=10,
                            stopInnerThr=1e-6, stopThr=1e-5, log=False,
                            **kwargs):
    """Joint OT and nonlinear mapping estimation with kernels as proposed in [8]

    The function solves the following optimization problem:

    .. math::
        \min_{\gamma,L\in\mathcal{H}}\quad \|L(X_s) -
        n_s\gamma X_t\|^2_F + \mu<\gamma,M>_F + \eta  \|L\|^2_\mathcal{H}

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (ns,nt) squared euclidean cost matrix between samples in
      Xs and Xt (scaled by ns)
    - :math:`L` is a ns x d linear operator on a kernel matrix that
      approximates the barycentric mapping
    - a and b are uniform source and target weights

    The problem consist in solving jointly an optimal transport matrix
    :math:`\gamma` and the nonlinear mapping that fits the barycentric mapping
    :math:`n_s\gamma X_t`.

    One can also estimate a mapping with constant bias (see supplementary
    material of [8]) using the bias optional argument.

    The algorithm used for solving the problem is the block coordinate
    descent that alternates between updates of G (using conditionnal gradient)
    and the update of L using a classical kernel least square solver.


    Parameters
    ----------
    xs : np.ndarray (ns,d)
        samples in the source domain
    xt : np.ndarray (nt,d)
        samples in the target domain
    mu : float,optional
        Weight for the linear OT loss (>0)
    eta : float, optional
        Regularization term  for the linear mapping L (>0)
    kerneltype : str,optional
        kernel used by calling function ot.utils.kernel (gaussian by default)
    sigma : float, optional
        Gaussian kernel bandwidth.
    bias : bool,optional
        Estimate linear mapping with constant bias
    verbose : bool, optional
        Print information along iterations
    verbose2 : bool, optional
        Print information along iterations
    numItermax : int, optional
        Max number of BCD iterations
    numInnerItermax : int, optional
        Max number of iterations (inner CG solver)
    stopInnerThr : float, optional
        Stop threshold on error (inner CG solver) (>0)
    stopThr : float, optional
        Stop threshold on relative loss decrease (>0)
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    L : (ns x d) ndarray
        Nonlinear mapping matrix (ns+1 x d if bias)
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
       "Mapping estimation for discrete optimal transport",
       Neural Information Processing Systems (NIPS), 2016.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    ns, nt = xs.shape[0], xt.shape[0]

    K = kernel(xs, xs, method=kerneltype, sigma=sigma)
    if bias:
        K1 = np.hstack((K, np.ones((ns, 1))))
        Id = np.eye(ns + 1)
        Id[-1] = 0
        Kp = np.eye(ns + 1)
        Kp[:ns, :ns] = K

        # ls regu
        # K0 = K1.T.dot(K1)+eta*I
        # Kreg=I

        # RKHS regul
        K0 = K1.T.dot(K1) + eta * Kp
        Kreg = Kp

    else:
        K1 = K
        Id = np.eye(ns)

        # ls regul
        # K0 = K1.T.dot(K1)+eta*I
        # Kreg=I

        # proper kernel ridge
        K0 = K + eta * Id
        Kreg = K

    if log:
        log = {'err': []}

    a, b = unif(ns), unif(nt)
    M = dist(xs, xt) * ns
    G = emd(a, b, M)

    vloss = []

    def loss(L, G):
        """Compute full loss"""
        return np.sum((K1.dot(L) - ns * G.dot(xt)) ** 2) + mu * \
            np.sum(G * M) + eta * np.trace(L.T.dot(Kreg).dot(L))

    def solve_L_nobias(G):
        """ solve L problem with fixed G (least square)"""
        xst = ns * G.dot(xt)
        return np.linalg.solve(K0, xst)

    def solve_L_bias(G):
        """ solve L problem with fixed G (least square)"""
        xst = ns * G.dot(xt)
        return np.linalg.solve(K0, K1.T.dot(xst))

    def solve_G(L, G0):
        """Update G with CG algorithm"""
        xsi = K1.dot(L)

        def f(G):
            return np.sum((xsi - ns * G.dot(xt)) ** 2)

        def df(G):
            return -2 * ns * (xsi - ns * G.dot(xt)).dot(xt.T)

        G = cg(a, b, M, 1.0 / mu, f, df, G0=G0,
               numItermax=numInnerItermax, stopThr=stopInnerThr)
        return G

    if bias:
        solve_L = solve_L_bias
    else:
        solve_L = solve_L_nobias

    L = solve_L(G)

    vloss.append(loss(L, G))

    if verbose:
        print('{:5s}|{:12s}|{:8s}'.format(
            'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
        print('{:5d}|{:8e}|{:8e}'.format(0, vloss[-1], 0))

    # init loop
    if numItermax > 0:
        loop = 1
    else:
        loop = 0
    it = 0

    while loop:

        it += 1

        # update G
        G = solve_G(L, G)

        # update L
        L = solve_L(G)

        vloss.append(loss(L, G))

        if it >= numItermax:
            loop = 0

        if abs(vloss[-1] - vloss[-2]) / abs(vloss[-2]) < stopThr:
            loop = 0

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}'.format(
                    'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
            print('{:5d}|{:8e}|{:8e}'.format(
                it, vloss[-1], (vloss[-1] - vloss[-2]) / abs(vloss[-2])))
    if log:
        log['loss'] = vloss
        return G, L, log
    else:
        return G, L


def OT_mapping_linear(xs, xt, reg=1e-6, ws=None,
                      wt=None, bias=True, log=False):
    """ return OT linear operator between samples

    The function estimates the optimal linear operator that aligns the two
    empirical distributions. This is equivalent to estimating the closed
    form mapping between two Gaussian distributions :math:`N(\mu_s,\Sigma_s)`
    and :math:`N(\mu_t,\Sigma_t)` as proposed in [14] and discussed in remark
    2.29 in [15].

    The linear operator from source to target :math:`M`

    .. math::
        M(x)=Ax+b

    where :

    .. math::
        A=\Sigma_s^{-1/2}(\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2})^{1/2}
        \Sigma_s^{-1/2}
    .. math::
        b=\mu_t-A\mu_s

    Parameters
    ----------
    xs : np.ndarray (ns,d)
        samples in the source domain
    xt : np.ndarray (nt,d)
        samples in the target domain
    reg : float,optional
        regularization added to the diagonals of convariances (>0)
    ws : np.ndarray (ns,1), optional
        weights for the source samples
    wt : np.ndarray (ns,1), optional
        weights for the target samples
    bias: boolean, optional
        estimate bias b else b=0 (default:True)
    log : bool, optional
        record log if True


    Returns
    -------
    A : (d x d) ndarray
        Linear operator
    b : (1 x d) ndarray
        bias
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [14] Knott, M. and Smith, C. S. "On the optimal mapping of
        distributions", Journal of Optimization Theory and Applications
        Vol 43, 1984

    .. [15]  Peyré, G., & Cuturi, M. (2017). "Computational Optimal
        Transport", 2018.


    """

    d = xs.shape[1]

    if bias:
        mxs = xs.mean(0, keepdims=True)
        mxt = xt.mean(0, keepdims=True)

        xs = xs - mxs
        xt = xt - mxt
    else:
        mxs = np.zeros((1, d))
        mxt = np.zeros((1, d))

    if ws is None:
        ws = np.ones((xs.shape[0], 1)) / xs.shape[0]

    if wt is None:
        wt = np.ones((xt.shape[0], 1)) / xt.shape[0]

    Cs = (xs * ws).T.dot(xs) / ws.sum() + reg * np.eye(d)
    Ct = (xt * wt).T.dot(xt) / wt.sum() + reg * np.eye(d)

    Cs12 = linalg.sqrtm(Cs)
    Cs_12 = linalg.inv(Cs12)

    M0 = linalg.sqrtm(Cs12.dot(Ct.dot(Cs12)))

    A = Cs_12.dot(M0.dot(Cs_12))

    b = mxt - mxs.dot(A)

    if log:
        log = {}
        log['Cs'] = Cs
        log['Ct'] = Ct
        log['Cs12'] = Cs12
        log['Cs_12'] = Cs_12
        return A, b, log
    else:
        return A, b


def emd_laplace(a, b, xs, xt, M, sim='knn', sim_param=None, reg='pos', eta=1, alpha=.5,
                numItermax=100, stopThr=1e-9, numInnerItermax=100000,
                stopInnerThr=1e-9, log=False, verbose=False):
    r"""Solve the optimal transport problem (OT) with Laplacian regularization

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + eta\Omega_\alpha(\gamma)

        s.t.\ \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0

    where:

    - a and b are source and target weights (sum to 1)
    - xs and xt are source and target samples
    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega_\alpha` is the Laplacian regularization term
      :math:`\Omega_\alpha = (1-\alpha)/n_s^2\sum_{i,j}S^s_{i,j}\|T(\mathbf{x}^s_i)-T(\mathbf{x}^s_j)\|^2+\alpha/n_t^2\sum_{i,j}S^t_{i,j}^'\|T(\mathbf{x}^t_i)-T(\mathbf{x}^t_j)\|^2`
      with :math:`S^s_{i,j}, S^t_{i,j}` denoting source and target similarity matrices and :math:`T(\cdot)` being a barycentric mapping

    The algorithm used for solving the problem is the conditional gradient algorithm as proposed in [5].

    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,)
        samples weights in the target domain
    xs : np.ndarray (ns,d)
        samples in the source domain
    xt : np.ndarray (nt,d)
        samples in the target domain
    M : np.ndarray (ns,nt)
        loss matrix
    sim : string, optional
        Type of similarity ('knn' or 'gauss') used to construct the Laplacian.
    sim_param : int or float, optional
        Parameter (number of the nearest neighbors for sim='knn'
        or bandwidth for sim='gauss') used to compute the Laplacian.
    reg : string
        Type of Laplacian regularization
    eta : float
        Regularization term for Laplacian regularization
    alpha : float
        Regularization term  for source domain's importance in regularization
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (inner emd solver) (>0)
    numInnerItermax : int, optional
        Max number of iterations (inner CG solver)
    stopInnerThr : float, optional
        Stop threshold on error (inner CG solver) (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
       "Optimal Transport for Domain Adaptation," in IEEE
       Transactions on Pattern Analysis and Machine Intelligence ,
       vol.PP, no.99, pp.1-1
    .. [30] R. Flamary, N. Courty, D. Tuia, A. Rakotomamonjy,
        "Optimal transport with Laplacian regularization: Applications to domain adaptation and shape matching,"
         in NIPS Workshop on Optimal Transport and Machine Learning OTML, 2014.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    if not isinstance(sim_param, (int, float, type(None))):
        raise ValueError(
            'Similarity parameter should be an int or a float. Got {type} instead.'.format(type=type(sim_param).__name__))

    if sim == 'gauss':
        if sim_param is None:
            sim_param = 1 / (2 * (np.mean(dist(xs, xs, 'sqeuclidean')) ** 2))
        sS = kernel(xs, xs, method=sim, sigma=sim_param)
        sT = kernel(xt, xt, method=sim, sigma=sim_param)

    elif sim == 'knn':
        if sim_param is None:
            sim_param = 3

        from sklearn.neighbors import kneighbors_graph

        sS = kneighbors_graph(X=xs, n_neighbors=int(sim_param)).toarray()
        sS = (sS + sS.T) / 2
        sT = kneighbors_graph(xt, n_neighbors=int(sim_param)).toarray()
        sT = (sT + sT.T) / 2
    else:
        raise ValueError('Unknown similarity type {sim}. Currently supported similarity types are "knn" and "gauss".'.format(sim=sim))

    lS = laplacian(sS)
    lT = laplacian(sT)

    def f(G):
        return alpha * np.trace(np.dot(xt.T, np.dot(G.T, np.dot(lS, np.dot(G, xt))))) \
            + (1 - alpha) * np.trace(np.dot(xs.T, np.dot(G, np.dot(lT, np.dot(G.T, xs)))))

    ls2 = lS + lS.T
    lt2 = lT + lT.T
    xt2 = np.dot(xt, xt.T)

    if reg == 'disp':
        Cs = -eta * alpha / xs.shape[0] * dots(ls2, xs, xt.T)
        Ct = -eta * (1 - alpha) / xt.shape[0] * dots(xs, xt.T, lt2)
        M = M + Cs + Ct

    def df(G):
        return alpha * np.dot(ls2, np.dot(G, xt2))\
            + (1 - alpha) * np.dot(xs, np.dot(xs.T, np.dot(G, lt2)))

    return cg(a, b, M, reg=eta, f=f, df=df, G0=None, numItermax=numItermax, numItermaxEmd=numInnerItermax,
              stopThr=stopThr, stopThr2=stopInnerThr, verbose=verbose, log=log)


def distribution_estimation_uniform(X):
    """estimates a uniform distribution from an array of samples X

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The array of samples

    Returns
    -------
    mu : array-like, shape (n_samples,)
        The uniform distribution estimated from X
    """

    return unif(X.shape[0])


class BaseTransport(BaseEstimator):

    """Base class for OTDA objects

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    the fit method should:

    - estimate a cost matrix and store it in a `cost_` attribute
    - estimate a coupling matrix and store it in a `coupling_`
    attribute
    - estimate distributions from source and target data and store them in
    mu_s and mu_t attributes
    - store Xs and Xt in attributes to be used later on in transform and
    inverse_transform methods

    transform method should always get as input a Xs parameter
    inverse_transform method should always get as input a Xt parameter

    transform_labels method should always get as input a ys parameter
    inverse_transform_labels method should always get as input a yt parameter
    """

    def fit(self, Xs=None, ys=None, Xt=None, yt=None):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt)

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The training class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt):

            # pairwise distance
            self.cost_ = dist(Xs, Xt, metric=self.metric)
            self.cost_ = cost_normalization(self.cost_, self.norm)

            if (ys is not None) and (yt is not None):

                if self.limit_max != np.infty:
                    self.limit_max = self.limit_max * np.max(self.cost_)

                # assumes labeled source samples occupy the first rows
                # and labeled target samples occupy the first columns
                classes = [c for c in np.unique(ys) if c != -1]
                for c in classes:
                    idx_s = np.where((ys != c) & (ys != -1))
                    idx_t = np.where(yt == c)

                    # all the coefficients corresponding to a source sample
                    # and a target sample :
                    # with different labels get a infinite
                    for j in idx_t[0]:
                        self.cost_[idx_s[0], j] = self.limit_max

            # distribution estimation
            self.mu_s = self.distribution_estimation(Xs)
            self.mu_t = self.distribution_estimation(Xt)

            # store arrays of samples
            self.xs_ = Xs
            self.xt_ = Xt

        return self

    def fit_transform(self, Xs=None, ys=None, Xt=None, yt=None):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt) and transports source samples Xs onto target
        ones Xt

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels for training samples
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The source samples samples.
        """

        return self.fit(Xs, ys, Xt, yt).transform(Xs, ys, Xt, yt)

    def transform(self, Xs=None, ys=None, Xt=None, yt=None, batch_size=128):
        """Transports source samples Xs onto target ones Xt

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The source input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels for source samples
        Xt : array-like, shape (n_target_samples, n_features)
            The target input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels for target. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The transport source samples.
        """

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs):

            if np.array_equal(self.xs_, Xs):

                # perform standard barycentric mapping
                transp = self.coupling_ / np.sum(self.coupling_, 1)[:, None]

                # set nans to 0
                transp[~ np.isfinite(transp)] = 0

                # compute transported samples
                transp_Xs = np.dot(transp, self.xt_)
            else:
                # perform out of sample mapping
                indices = np.arange(Xs.shape[0])
                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xs = []
                for bi in batch_ind:
                    # get the nearest neighbor in the source domain
                    D0 = dist(Xs[bi], self.xs_)
                    idx = np.argmin(D0, axis=1)

                    # transport the source samples
                    transp = self.coupling_ / np.sum(
                        self.coupling_, 1)[:, None]
                    transp[~ np.isfinite(transp)] = 0
                    transp_Xs_ = np.dot(transp, self.xt_)

                    # define the transported points
                    transp_Xs_ = transp_Xs_[idx, :] + Xs[bi] - self.xs_[idx, :]

                    transp_Xs.append(transp_Xs_)

                transp_Xs = np.concatenate(transp_Xs, axis=0)

            return transp_Xs

    def transform_labels(self, ys=None):
        """Propagate source labels ys to obtain estimated target labels as in [27]

        Parameters
        ----------
        ys : array-like, shape (n_source_samples,)
            The source class labels

        Returns
        -------
        transp_ys : array-like, shape (n_target_samples, nb_classes)
            Estimated soft target labels.

        References
        ----------

        .. [27] Ievgen Redko, Nicolas Courty, Rémi Flamary, Devis Tuia
           "Optimal transport for multi-source domain adaptation under target shift",
           International Conference on Artificial Intelligence and Statistics (AISTATS), 2019.

        """

        # check the necessary inputs parameters are here
        if check_params(ys=ys):

            ysTemp = label_normalization(np.copy(ys))
            classes = np.unique(ysTemp)
            n = len(classes)
            D1 = np.zeros((n, len(ysTemp)))

            # perform label propagation
            transp = self.coupling_ / np.sum(self.coupling_, 1)[:, None]

            # set nans to 0
            transp[~ np.isfinite(transp)] = 0

            for c in classes:
                D1[int(c), ysTemp == c] = 1

            # compute propagated labels
            transp_ys = np.dot(D1, transp)

            return transp_ys.T

    def inverse_transform(self, Xs=None, ys=None, Xt=None, yt=None,
                          batch_size=128):
        """Transports target samples Xt onto source samples Xs

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The source input samples.
        ys : array-like, shape (n_source_samples,)
            The source class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The target input samples.
        yt : array-like, shape (n_target_samples,)
            The target class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform

        Returns
        -------
        transp_Xt : array-like, shape (n_source_samples, n_features)
            The transported target samples.
        """

        # check the necessary inputs parameters are here
        if check_params(Xt=Xt):

            if np.array_equal(self.xt_, Xt):

                # perform standard barycentric mapping
                transp_ = self.coupling_.T / np.sum(self.coupling_, 0)[:, None]

                # set nans to 0
                transp_[~ np.isfinite(transp_)] = 0

                # compute transported samples
                transp_Xt = np.dot(transp_, self.xs_)
            else:
                # perform out of sample mapping
                indices = np.arange(Xt.shape[0])
                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xt = []
                for bi in batch_ind:
                    D0 = dist(Xt[bi], self.xt_)
                    idx = np.argmin(D0, axis=1)

                    # transport the target samples
                    transp_ = self.coupling_.T / np.sum(
                        self.coupling_, 0)[:, None]
                    transp_[~ np.isfinite(transp_)] = 0
                    transp_Xt_ = np.dot(transp_, self.xs_)

                    # define the transported points
                    transp_Xt_ = transp_Xt_[idx, :] + Xt[bi] - self.xt_[idx, :]

                    transp_Xt.append(transp_Xt_)

                transp_Xt = np.concatenate(transp_Xt, axis=0)

            return transp_Xt

    def inverse_transform_labels(self, yt=None):
        """Propagate target labels yt to obtain estimated source labels ys

        Parameters
        ----------
        yt : array-like, shape (n_target_samples,)

        Returns
        -------
        transp_ys : array-like, shape (n_source_samples, nb_classes)
            Estimated soft source labels.
        """

        # check the necessary inputs parameters are here
        if check_params(yt=yt):

            ytTemp = label_normalization(np.copy(yt))
            classes = np.unique(ytTemp)
            n = len(classes)
            D1 = np.zeros((n, len(ytTemp)))

            # perform label propagation
            transp = self.coupling_ / np.sum(self.coupling_, 1)[:, None]

            # set nans to 0
            transp[~ np.isfinite(transp)] = 0

            for c in classes:
                D1[int(c), ytTemp == c] = 1

            # compute propagated samples
            transp_ys = np.dot(D1, transp.T)

            return transp_ys.T


class LinearTransport(BaseTransport):
    """ OT linear operator between empirical distributions

    The function estimates the optimal linear operator that aligns the two
    empirical distributions. This is equivalent to estimating the closed
    form mapping between two Gaussian distributions :math:`N(\mu_s,\Sigma_s)`
    and :math:`N(\mu_t,\Sigma_t)` as proposed in [14] and discussed in
    remark 2.29 in [15].

    The linear operator from source to target :math:`M`

    .. math::
        M(x)=Ax+b

    where :

    .. math::
        A=\Sigma_s^{-1/2}(\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2})^{1/2}
        \Sigma_s^{-1/2}
    .. math::
        b=\mu_t-A\mu_s

    Parameters
    ----------
    reg : float,optional
        regularization added to the daigonals of convariances (>0)
    bias: boolean, optional
        estimate bias b else b=0 (default:True)
    log : bool, optional
        record log if True

    References
    ----------

    .. [14] Knott, M. and Smith, C. S. "On the optimal mapping of
        distributions", Journal of Optimization Theory and Applications
        Vol 43, 1984

    .. [15]  Peyré, G., & Cuturi, M. (2017). "Computational Optimal
        Transport", 2018.

    """

    def __init__(self, reg=1e-8, bias=True, log=False,
                 distribution_estimation=distribution_estimation_uniform):
        self.bias = bias
        self.log = log
        self.reg = reg
        self.distribution_estimation = distribution_estimation

    def fit(self, Xs=None, ys=None, Xt=None, yt=None):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt)

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """

        self.mu_s = self.distribution_estimation(Xs)
        self.mu_t = self.distribution_estimation(Xt)

        # coupling estimation
        returned_ = OT_mapping_linear(Xs, Xt, reg=self.reg,
                                      ws=self.mu_s.reshape((-1, 1)),
                                      wt=self.mu_t.reshape((-1, 1)),
                                      bias=self.bias, log=self.log)

        # deal with the value of log
        if self.log:
            self.A_, self.B_, self.log_ = returned_
        else:
            self.A_, self.B_, = returned_
            self.log_ = dict()

        # re compute inverse mapping
        self.A1_ = linalg.inv(self.A_)
        self.B1_ = -self.B_.dot(self.A1_)

        return self

    def transform(self, Xs=None, ys=None, Xt=None, yt=None, batch_size=128):
        """Transports source samples Xs onto target ones Xt

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The transport source samples.
        """

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs):
            transp_Xs = Xs.dot(self.A_) + self.B_

            return transp_Xs

    def inverse_transform(self, Xs=None, ys=None, Xt=None, yt=None,
                          batch_size=128):
        """Transports target samples Xt onto target samples Xs

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform

        Returns
        -------
        transp_Xt : array-like, shape (n_source_samples, n_features)
            The transported target samples.
        """

        # check the necessary inputs parameters are here
        if check_params(Xt=Xt):
            transp_Xt = Xt.dot(self.A1_) + self.B1_

            return transp_Xt


class SinkhornTransport(BaseTransport):

    """Domain Adapatation OT method based on Sinkhorn Algorithm

    Parameters
    ----------
    reg_e : float, optional (default=1)
        Entropic regularization parameter
    max_iter : int, float, optional (default=1000)
        The minimum number of iteration before stopping the optimization
        algorithm if no it has not converged
    tol : float, optional (default=10e-9)
        The precision required to stop the optimization algorithm.
    verbose : bool, optional (default=False)
        Controls the verbosity of the optimization algorithm
    log : int, optional (default=False)
        Controls the logs of the optimization algorithm
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    distribution_estimation : callable, optional (defaults to the uniform)
        The kind of distribution estimation to employ
    out_of_sample_map : string, optional (default="ferradans")
        The kind of out of sample mapping to apply to transport samples
        from a domain into another one. Currently the only possible option is
        "ferradans" which uses the method proposed in [6].
    limit_max: float, optional (defaul=np.infty)
        Controls the semi supervised mode. Transport between labeled source
        and target samples of different classes will exhibit an cost defined
        by this variable

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling
    log_ : dictionary
        The dictionary of log, empty dic if parameter log is not True

    References
    ----------
    .. [1] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
           "Optimal Transport for Domain Adaptation," in IEEE Transactions
           on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1
    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal
           Transport, Advances in Neural Information Processing Systems (NIPS)
           26, 2013
    .. [6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
            Regularized discrete optimal transport. SIAM Journal on Imaging
            Sciences, 7(3), 1853-1882.
    """

    def __init__(self, reg_e=1., max_iter=1000,
                 tol=10e-9, verbose=False, log=False,
                 metric="sqeuclidean", norm=None,
                 distribution_estimation=distribution_estimation_uniform,
                 out_of_sample_map='ferradans', limit_max=np.infty):
        self.reg_e = reg_e
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.metric = metric
        self.norm = norm
        self.limit_max = limit_max
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map

    def fit(self, Xs=None, ys=None, Xt=None, yt=None):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt)

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """

        super(SinkhornTransport, self).fit(Xs, ys, Xt, yt)

        # coupling estimation
        returned_ = sinkhorn(
            a=self.mu_s, b=self.mu_t, M=self.cost_, reg=self.reg_e,
            numItermax=self.max_iter, stopThr=self.tol,
            verbose=self.verbose, log=self.log)

        # deal with the value of log
        if self.log:
            self.coupling_, self.log_ = returned_
        else:
            self.coupling_ = returned_
            self.log_ = dict()

        return self


class EMDTransport(BaseTransport):

    """Domain Adapatation OT method based on Earth Mover's Distance

    Parameters
    ----------
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    log : int, optional (default=False)
        Controls the logs of the optimization algorithm
    distribution_estimation : callable, optional (defaults to the uniform)
        The kind of distribution estimation to employ
    out_of_sample_map : string, optional (default="ferradans")
        The kind of out of sample mapping to apply to transport samples
        from a domain into another one. Currently the only possible option is
        "ferradans" which uses the method proposed in [6].
    limit_max: float, optional (default=10)
        Controls the semi supervised mode. Transport between labeled source
        and target samples of different classes will exhibit an infinite cost
        (10 times the maximum value of the cost matrix)
    max_iter : int, optional (default=100000)
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling

    References
    ----------
    .. [1] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
           "Optimal Transport for Domain Adaptation," in IEEE Transactions
           on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1
    .. [6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
            Regularized discrete optimal transport. SIAM Journal on Imaging
            Sciences, 7(3), 1853-1882.
    """

    def __init__(self, metric="sqeuclidean", norm=None, log=False,
                 distribution_estimation=distribution_estimation_uniform,
                 out_of_sample_map='ferradans', limit_max=10,
                 max_iter=100000):
        self.metric = metric
        self.norm = norm
        self.log = log
        self.limit_max = limit_max
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map
        self.max_iter = max_iter

    def fit(self, Xs, ys=None, Xt=None, yt=None):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt)

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """

        super(EMDTransport, self).fit(Xs, ys, Xt, yt)

        returned_ = emd(
            a=self.mu_s, b=self.mu_t, M=self.cost_, numItermax=self.max_iter,
            log=self.log)

        # coupling estimation
        if self.log:
            self.coupling_, self.log_ = returned_
        else:
            self.coupling_ = returned_
            self.log_ = dict()
        return self


class SinkhornLpl1Transport(BaseTransport):

    """Domain Adapatation OT method based on sinkhorn algorithm +
    LpL1 class regularization.

    Parameters
    ----------
    reg_e : float, optional (default=1)
        Entropic regularization parameter
    reg_cl : float, optional (default=0.1)
        Class regularization parameter
    max_iter : int, float, optional (default=10)
        The minimum number of iteration before stopping the optimization
        algorithm if no it has not converged
    max_inner_iter : int, float, optional (default=200)
        The number of iteration in the inner loop
    log : bool, optional (default=False)
        Controls the logs of the optimization algorithm
    tol : float, optional (default=10e-9)
        Stop threshold on error (inner sinkhorn solver) (>0)
    verbose : bool, optional (default=False)
        Controls the verbosity of the optimization algorithm
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    distribution_estimation : callable, optional (defaults to the uniform)
        The kind of distribution estimation to employ
    out_of_sample_map : string, optional (default="ferradans")
        The kind of out of sample mapping to apply to transport samples
        from a domain into another one. Currently the only possible option is
        "ferradans" which uses the method proposed in [6].
    limit_max: float, optional (defaul=np.infty)
        Controls the semi supervised mode. Transport between labeled source
        and target samples of different classes will exhibit a cost defined by
        limit_max.

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling

    References
    ----------

    .. [1] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
       "Optimal Transport for Domain Adaptation," in IEEE
       Transactions on Pattern Analysis and Machine Intelligence ,
       vol.PP, no.99, pp.1-1
    .. [2] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015).
       Generalized conditional gradient: analysis of convergence
       and applications. arXiv preprint arXiv:1510.06567.
    .. [6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
            Regularized discrete optimal transport. SIAM Journal on Imaging
            Sciences, 7(3), 1853-1882.
    """

    def __init__(self, reg_e=1., reg_cl=0.1,
                 max_iter=10, max_inner_iter=200, log=False,
                 tol=10e-9, verbose=False,
                 metric="sqeuclidean", norm=None,
                 distribution_estimation=distribution_estimation_uniform,
                 out_of_sample_map='ferradans', limit_max=np.infty):
        self.reg_e = reg_e
        self.reg_cl = reg_cl
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.tol = tol
        self.log = log
        self.verbose = verbose
        self.metric = metric
        self.norm = norm
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map
        self.limit_max = limit_max

    def fit(self, Xs, ys=None, Xt=None, yt=None):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt)

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt, ys=ys):
            super(SinkhornLpl1Transport, self).fit(Xs, ys, Xt, yt)

            returned_ = sinkhorn_lpl1_mm(
                a=self.mu_s, labels_a=ys, b=self.mu_t, M=self.cost_,
                reg=self.reg_e, eta=self.reg_cl, numItermax=self.max_iter,
                numInnerItermax=self.max_inner_iter, stopInnerThr=self.tol,
                verbose=self.verbose, log=self.log)

        # deal with the value of log
        if self.log:
            self.coupling_, self.log_ = returned_
        else:
            self.coupling_ = returned_
            self.log_ = dict()
        return self


class EMDLaplaceTransport(BaseTransport):

    """Domain Adapatation OT method based on Earth Mover's Distance with Laplacian regularization

    Parameters
    ----------
    reg_type : string optional (default='pos')
        Type of the regularization term: 'pos' and 'disp' for
        regularization term defined in [2] and [6], respectively.
    reg_lap : float, optional (default=1)
        Laplacian regularization parameter
    reg_src : float, optional (default=0.5)
        Source relative importance in regularization
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    similarity : string, optional (default="knn")
        The similarity to use either knn or gaussian
    similarity_param : int or float, optional (default=None)
        Parameter for the similarity: number of nearest neighbors or bandwidth
        if similarity="knn" or "gaussian", respectively. If None is provided,
        it is set to 3 or the average pairwise squared Euclidean distance, respectively.
    max_iter : int, optional (default=100)
        Max number of BCD iterations
    tol : float, optional (default=1e-5)
        Stop threshold on relative loss decrease (>0)
    max_inner_iter : int, optional (default=10)
        Max number of iterations (inner CG solver)
    inner_tol : float, optional (default=1e-6)
        Stop threshold on error (inner CG solver) (>0)
    log : int, optional (default=False)
        Controls the logs of the optimization algorithm
    distribution_estimation : callable, optional (defaults to the uniform)
        The kind of distribution estimation to employ
    out_of_sample_map : string, optional (default="ferradans")
        The kind of out of sample mapping to apply to transport samples
        from a domain into another one. Currently the only possible option is
        "ferradans" which uses the method proposed in [6].

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling

    References
    ----------
    .. [1] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
           "Optimal Transport for Domain Adaptation," in IEEE Transactions
           on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1
    .. [2] R. Flamary, N. Courty, D. Tuia, A. Rakotomamonjy,
        "Optimal transport with Laplacian regularization: Applications to domain adaptation and shape matching,"
         in NIPS Workshop on Optimal Transport and Machine Learning OTML, 2014.
    .. [6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
            Regularized discrete optimal transport. SIAM Journal on Imaging
            Sciences, 7(3), 1853-1882.
    """

    def __init__(self, reg_type='pos', reg_lap=1., reg_src=1., metric="sqeuclidean",
                 norm=None, similarity="knn", similarity_param=None, max_iter=100, tol=1e-9,
                 max_inner_iter=100000, inner_tol=1e-9, log=False, verbose=False,
                 distribution_estimation=distribution_estimation_uniform,
                 out_of_sample_map='ferradans'):
        self.reg = reg_type
        self.reg_lap = reg_lap
        self.reg_src = reg_src
        self.metric = metric
        self.norm = norm
        self.similarity = similarity
        self.sim_param = similarity_param
        self.max_iter = max_iter
        self.tol = tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol
        self.log = log
        self.verbose = verbose
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map

    def fit(self, Xs, ys=None, Xt=None, yt=None):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt)

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """

        super(EMDLaplaceTransport, self).fit(Xs, ys, Xt, yt)

        returned_ = emd_laplace(a=self.mu_s, b=self.mu_t, xs=self.xs_,
                                xt=self.xt_, M=self.cost_, sim=self.similarity, sim_param=self.sim_param, reg=self.reg, eta=self.reg_lap,
                                alpha=self.reg_src, numItermax=self.max_iter, stopThr=self.tol, numInnerItermax=self.max_inner_iter,
                                stopInnerThr=self.inner_tol, log=self.log, verbose=self.verbose)

        # coupling estimation
        if self.log:
            self.coupling_, self.log_ = returned_
        else:
            self.coupling_ = returned_
            self.log_ = dict()
        return self


class SinkhornL1l2Transport(BaseTransport):

    """Domain Adapatation OT method based on sinkhorn algorithm +
    l1l2 class regularization.

    Parameters
    ----------
    reg_e : float, optional (default=1)
        Entropic regularization parameter
    reg_cl : float, optional (default=0.1)
        Class regularization parameter
    max_iter : int, float, optional (default=10)
        The minimum number of iteration before stopping the optimization
        algorithm if no it has not converged
    max_inner_iter : int, float, optional (default=200)
        The number of iteration in the inner loop
    tol : float, optional (default=10e-9)
        Stop threshold on error (inner sinkhorn solver) (>0)
    verbose : bool, optional (default=False)
        Controls the verbosity of the optimization algorithm
    log : bool, optional (default=False)
        Controls the logs of the optimization algorithm
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    distribution_estimation : callable, optional (defaults to the uniform)
        The kind of distribution estimation to employ
    out_of_sample_map : string, optional (default="ferradans")
        The kind of out of sample mapping to apply to transport samples
        from a domain into another one. Currently the only possible option is
        "ferradans" which uses the method proposed in [6].
    limit_max: float, optional (default=10)
        Controls the semi supervised mode. Transport between labeled source
        and target samples of different classes will exhibit an infinite cost
        (10 times the maximum value of the cost matrix)

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling
    log_ : dictionary
        The dictionary of log, empty dic if parameter log is not True

    References
    ----------

    .. [1] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
       "Optimal Transport for Domain Adaptation," in IEEE
       Transactions on Pattern Analysis and Machine Intelligence ,
       vol.PP, no.99, pp.1-1
    .. [2] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015).
       Generalized conditional gradient: analysis of convergence
       and applications. arXiv preprint arXiv:1510.06567.
    .. [6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
            Regularized discrete optimal transport. SIAM Journal on Imaging
            Sciences, 7(3), 1853-1882.
    """

    def __init__(self, reg_e=1., reg_cl=0.1,
                 max_iter=10, max_inner_iter=200,
                 tol=10e-9, verbose=False, log=False,
                 metric="sqeuclidean", norm=None,
                 distribution_estimation=distribution_estimation_uniform,
                 out_of_sample_map='ferradans', limit_max=10):
        self.reg_e = reg_e
        self.reg_cl = reg_cl
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.metric = metric
        self.norm = norm
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map
        self.limit_max = limit_max

    def fit(self, Xs, ys=None, Xt=None, yt=None):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt)

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt, ys=ys):

            super(SinkhornL1l2Transport, self).fit(Xs, ys, Xt, yt)

            returned_ = sinkhorn_l1l2_gl(
                a=self.mu_s, labels_a=ys, b=self.mu_t, M=self.cost_,
                reg=self.reg_e, eta=self.reg_cl, numItermax=self.max_iter,
                numInnerItermax=self.max_inner_iter, stopInnerThr=self.tol,
                verbose=self.verbose, log=self.log)

            # deal with the value of log
            if self.log:
                self.coupling_, self.log_ = returned_
            else:
                self.coupling_ = returned_
                self.log_ = dict()

        return self


class MappingTransport(BaseEstimator):

    """MappingTransport: DA methods that aims at jointly estimating a optimal
    transport coupling and the associated mapping

    Parameters
    ----------
    mu : float, optional (default=1)
        Weight for the linear OT loss (>0)
    eta : float, optional (default=0.001)
        Regularization term for the linear mapping L (>0)
    bias : bool, optional (default=False)
        Estimate linear mapping with constant bias
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    kernel : string, optional (default="linear")
        The kernel to use either linear or gaussian
    sigma : float, optional (default=1)
        The gaussian kernel parameter
    max_iter : int, optional (default=100)
        Max number of BCD iterations
    tol : float, optional (default=1e-5)
        Stop threshold on relative loss decrease (>0)
    max_inner_iter : int, optional (default=10)
        Max number of iterations (inner CG solver)
    inner_tol : float, optional (default=1e-6)
        Stop threshold on error (inner CG solver) (>0)
    log : bool, optional (default=False)
        record log if True
    verbose : bool, optional (default=False)
        Print information along iterations
    verbose2 : bool, optional (default=False)
        Print information along iterations

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling
    mapping_ : array-like, shape (n_features (+ 1), n_features)
        (if bias) for kernel == linear
        The associated mapping
        array-like, shape (n_source_samples (+ 1), n_features)
        (if bias) for kernel == gaussian
    log_ : dictionary
        The dictionary of log, empty dic if parameter log is not True

    References
    ----------

    .. [8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
            "Mapping estimation for discrete optimal transport",
            Neural Information Processing Systems (NIPS), 2016.

    """

    def __init__(self, mu=1, eta=0.001, bias=False, metric="sqeuclidean",
                 norm=None, kernel="linear", sigma=1, max_iter=100, tol=1e-5,
                 max_inner_iter=10, inner_tol=1e-6, log=False, verbose=False,
                 verbose2=False):
        self.metric = metric
        self.norm = norm
        self.mu = mu
        self.eta = eta
        self.bias = bias
        self.kernel = kernel
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol
        self.log = log
        self.verbose = verbose
        self.verbose2 = verbose2

    def fit(self, Xs=None, ys=None, Xt=None, yt=None):
        """Builds an optimal coupling and estimates the associated mapping
        from source and target sets of samples (Xs, ys) and (Xt, yt)

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self
        """

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt):

            self.xs_ = Xs
            self.xt_ = Xt

            if self.kernel == "linear":
                returned_ = joint_OT_mapping_linear(
                    Xs, Xt, mu=self.mu, eta=self.eta, bias=self.bias,
                    verbose=self.verbose, verbose2=self.verbose2,
                    numItermax=self.max_iter,
                    numInnerItermax=self.max_inner_iter, stopThr=self.tol,
                    stopInnerThr=self.inner_tol, log=self.log)

            elif self.kernel == "gaussian":
                returned_ = joint_OT_mapping_kernel(
                    Xs, Xt, mu=self.mu, eta=self.eta, bias=self.bias,
                    sigma=self.sigma, verbose=self.verbose,
                    verbose2=self.verbose, numItermax=self.max_iter,
                    numInnerItermax=self.max_inner_iter,
                    stopInnerThr=self.inner_tol, stopThr=self.tol,
                    log=self.log)

            # deal with the value of log
            if self.log:
                self.coupling_, self.mapping_, self.log_ = returned_
            else:
                self.coupling_, self.mapping_ = returned_
                self.log_ = dict()

        return self

    def transform(self, Xs):
        """Transports source samples Xs onto target ones Xt

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The transport source samples.
        """

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs):

            if np.array_equal(self.xs_, Xs):
                # perform standard barycentric mapping
                transp = self.coupling_ / np.sum(self.coupling_, 1)[:, None]

                # set nans to 0
                transp[~ np.isfinite(transp)] = 0

                # compute transported samples
                transp_Xs = np.dot(transp, self.xt_)
            else:
                if self.kernel == "gaussian":
                    K = kernel(Xs, self.xs_, method=self.kernel,
                               sigma=self.sigma)
                elif self.kernel == "linear":
                    K = Xs
                if self.bias:
                    K = np.hstack((K, np.ones((Xs.shape[0], 1))))
                transp_Xs = K.dot(self.mapping_)

            return transp_Xs


class UnbalancedSinkhornTransport(BaseTransport):

    """Domain Adapatation unbalanced OT method based on sinkhorn algorithm

    Parameters
    ----------
    reg_e : float, optional (default=1)
        Entropic regularization parameter
    reg_m : float, optional (default=0.1)
        Mass regularization parameter
    method : str
        method used for the solver either 'sinkhorn',  'sinkhorn_stabilized' or
        'sinkhorn_epsilon_scaling', see those function for specific parameters
    max_iter : int, float, optional (default=10)
        The minimum number of iteration before stopping the optimization
        algorithm if no it has not converged
    tol : float, optional (default=10e-9)
        Stop threshold on error (inner sinkhorn solver) (>0)
    verbose : bool, optional (default=False)
        Controls the verbosity of the optimization algorithm
    log : bool, optional (default=False)
        Controls the logs of the optimization algorithm
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    distribution_estimation : callable, optional (defaults to the uniform)
        The kind of distribution estimation to employ
    out_of_sample_map : string, optional (default="ferradans")
        The kind of out of sample mapping to apply to transport samples
        from a domain into another one. Currently the only possible option is
        "ferradans" which uses the method proposed in [6].
    limit_max: float, optional (default=10)
        Controls the semi supervised mode. Transport between labeled source
        and target samples of different classes will exhibit an infinite cost
        (10 times the maximum value of the cost matrix)

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling
    log_ : dictionary
        The dictionary of log, empty dic if parameter log is not True

    References
    ----------

    .. [1] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
    Scaling algorithms for unbalanced transport problems. arXiv preprint
    arXiv:1607.05816.
    .. [6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
            Regularized discrete optimal transport. SIAM Journal on Imaging
            Sciences, 7(3), 1853-1882.
    """

    def __init__(self, reg_e=1., reg_m=0.1, method='sinkhorn',
                 max_iter=10, tol=1e-9, verbose=False, log=False,
                 metric="sqeuclidean", norm=None,
                 distribution_estimation=distribution_estimation_uniform,
                 out_of_sample_map='ferradans', limit_max=10):
        self.reg_e = reg_e
        self.reg_m = reg_m
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.metric = metric
        self.norm = norm
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map
        self.limit_max = limit_max

    def fit(self, Xs, ys=None, Xt=None, yt=None):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt)

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt):

            super(UnbalancedSinkhornTransport, self).fit(Xs, ys, Xt, yt)

            returned_ = sinkhorn_unbalanced(
                a=self.mu_s, b=self.mu_t, M=self.cost_,
                reg=self.reg_e, reg_m=self.reg_m, method=self.method,
                numItermax=self.max_iter, stopThr=self.tol,
                verbose=self.verbose, log=self.log)

            # deal with the value of log
            if self.log:
                self.coupling_, self.log_ = returned_
            else:
                self.coupling_ = returned_
                self.log_ = dict()

        return self


class JCPOTTransport(BaseTransport):

    """Domain Adapatation OT method for multi-source target shift based on Wasserstein barycenter algorithm.

    Parameters
    ----------
    reg_e : float, optional (default=1)
        Entropic regularization parameter
    max_iter : int, float, optional (default=10)
        The minimum number of iteration before stopping the optimization
        algorithm if no it has not converged
    tol : float, optional (default=10e-9)
        Stop threshold on error (inner sinkhorn solver) (>0)
    verbose : bool, optional (default=False)
        Controls the verbosity of the optimization algorithm
    log : bool, optional (default=False)
        Controls the logs of the optimization algorithm
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    distribution_estimation : callable, optional (defaults to the uniform)
        The kind of distribution estimation to employ
    out_of_sample_map : string, optional (default="ferradans")
        The kind of out of sample mapping to apply to transport samples
        from a domain into another one. Currently the only possible option is
        "ferradans" which uses the method proposed in [6].

    Attributes
    ----------
    coupling_ : list of array-like objects, shape K x (n_source_samples, n_target_samples)
        A set of optimal couplings between each source domain and the target domain
    proportions_ : array-like, shape (n_classes,)
        Estimated class proportions in the target domain
    log_ : dictionary
        The dictionary of log, empty dic if parameter log is not True

    References
    ----------

    .. [1] Ievgen Redko, Nicolas Courty, Rémi Flamary, Devis Tuia
       "Optimal transport for multi-source domain adaptation under target shift",
       International Conference on Artificial Intelligence and Statistics (AISTATS),
       vol. 89, p.849-858, 2019.

    .. [6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
            Regularized discrete optimal transport. SIAM Journal on Imaging
            Sciences, 7(3), 1853-1882.


    """

    def __init__(self, reg_e=.1, max_iter=10,
                 tol=10e-9, verbose=False, log=False,
                 metric="sqeuclidean",
                 out_of_sample_map='ferradans'):
        self.reg_e = reg_e
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.metric = metric
        self.out_of_sample_map = out_of_sample_map

    def fit(self, Xs, ys=None, Xt=None, yt=None):
        """Building coupling matrices from a list of source and target sets of samples
        (Xs, ys) and (Xt, yt)

        Parameters
        ----------
        Xs : list of K array-like objects, shape K x (nk_source_samples, n_features)
            A list of the training input samples.
        ys : list of K array-like objects, shape K x (nk_source_samples,)
            A list of the class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt, ys=ys):

            self.xs_ = Xs
            self.xt_ = Xt

            returned_ = jcpot_barycenter(Xs=Xs, Ys=ys, Xt=Xt, reg=self.reg_e,
                                         metric=self.metric, distrinumItermax=self.max_iter, stopThr=self.tol,
                                         verbose=self.verbose, log=True)

            self.coupling_ = returned_[1]['gamma']

            # deal with the value of log
            if self.log:
                self.proportions_, self.log_ = returned_
            else:
                self.proportions_ = returned_
                self.log_ = dict()

        return self

    def transform(self, Xs=None, ys=None, Xt=None, yt=None, batch_size=128):
        """Transports source samples Xs onto target ones Xt

        Parameters
        ----------
        Xs : list of K array-like objects, shape K x (nk_source_samples, n_features)
            A list of the training input samples.
        ys : list of K array-like objects, shape K x (nk_source_samples,)
            A list of the class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform
        """

        transp_Xs = []

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs):

            if all([np.allclose(x, y) for x, y in zip(self.xs_, Xs)]):

                # perform standard barycentric mapping for each source domain

                for coupling in self.coupling_:
                    transp = coupling / np.sum(coupling, 1)[:, None]

                    # set nans to 0
                    transp[~ np.isfinite(transp)] = 0

                    # compute transported samples
                    transp_Xs.append(np.dot(transp, self.xt_))
            else:

                # perform out of sample mapping
                indices = np.arange(Xs.shape[0])
                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xs = []

                for bi in batch_ind:
                    transp_Xs_ = []

                    # get the nearest neighbor in the sources domains
                    xs = np.concatenate(self.xs_, axis=0)
                    idx = np.argmin(dist(Xs[bi], xs), axis=1)

                    # transport the source samples
                    for coupling in self.coupling_:
                        transp = coupling / np.sum(
                            coupling, 1)[:, None]
                        transp[~ np.isfinite(transp)] = 0
                        transp_Xs_.append(np.dot(transp, self.xt_))

                    transp_Xs_ = np.concatenate(transp_Xs_, axis=0)

                    # define the transported points
                    transp_Xs_ = transp_Xs_[idx, :] + Xs[bi] - xs[idx, :]
                    transp_Xs.append(transp_Xs_)

                transp_Xs = np.concatenate(transp_Xs, axis=0)

            return transp_Xs

    def transform_labels(self, ys=None):
        """Propagate source labels ys to obtain target labels as in [27]

        Parameters
        ----------
        ys : list of K array-like objects, shape K x (nk_source_samples,)
            A list of the class labels

        Returns
        -------
        yt : array-like, shape (n_target_samples, nb_classes)
            Estimated soft target labels.
        """

        # check the necessary inputs parameters are here
        if check_params(ys=ys):
            yt = np.zeros((len(np.unique(np.concatenate(ys))), self.xt_.shape[0]))
            for i in range(len(ys)):
                ysTemp = label_normalization(np.copy(ys[i]))
                classes = np.unique(ysTemp)
                n = len(classes)
                ns = len(ysTemp)

                # perform label propagation
                transp = self.coupling_[i] / np.sum(self.coupling_[i], 1)[:, None]

                # set nans to 0
                transp[~ np.isfinite(transp)] = 0

                if self.log:
                    D1 = self.log_['D1'][i]
                else:
                    D1 = np.zeros((n, ns))

                    for c in classes:
                        D1[int(c), ysTemp == c] = 1

                # compute propagated labels
                yt = yt + np.dot(D1, transp) / len(ys)

            return yt.T

    def inverse_transform_labels(self, yt=None):
        """Propagate source labels ys to obtain target labels

        Parameters
        ----------
        yt : array-like, shape (n_source_samples,)
            The target class labels

        Returns
        -------
        transp_ys : list of K array-like objects, shape K x (nk_source_samples, nb_classes)
            A list of estimated soft source labels
        """

        # check the necessary inputs parameters are here
        if check_params(yt=yt):
            transp_ys = []
            ytTemp = label_normalization(np.copy(yt))
            classes = np.unique(ytTemp)
            n = len(classes)
            D1 = np.zeros((n, len(ytTemp)))

            for c in classes:
                D1[int(c), ytTemp == c] = 1

            for i in range(len(self.xs_)):

                # perform label propagation
                transp = self.coupling_[i] / np.sum(self.coupling_[i], 1)[:, None]

                # set nans to 0
                transp[~ np.isfinite(transp)] = 0

                # compute propagated labels
                transp_ys.append(np.dot(D1, transp.T).T)

            return transp_ys
