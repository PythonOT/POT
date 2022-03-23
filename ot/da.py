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

from .backend import get_backend
from .bregman import sinkhorn, jcpot_barycenter
from .lp import emd
from .utils import unif, dist, kernel, cost_normalization, label_normalization, laplacian, dots
from .utils import list_to_array, check_params, BaseEstimator
from .unbalanced import sinkhorn_unbalanced
from .optim import cg
from .optim import gcg


def sinkhorn_lpl1_mm(a, labels_a, b, M, reg, eta=0.1, numItermax=10,
                     numInnerItermax=200, stopInnerThr=1e-9, verbose=False,
                     log=False):
    r"""
    Solve the entropic regularization optimal transport problem with nonconvex
    group lasso regularization

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot \Omega_e(\gamma) + \eta \ \Omega_g(\gamma)

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0


    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\Omega_e` is the entropic regularization term :math:`\Omega_e
      (\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\Omega_g` is the group lasso  regularization term
      :math:`\Omega_g(\gamma)=\sum_{i,c} \|\gamma_{i,\mathcal{I}_c}\|^{1/2}_1`
      where  :math:`\mathcal{I}_c` are the index of samples from class `c`
      in the source domain.
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is the generalized conditional
    gradient as proposed in :ref:`[5, 7] <references-sinkhorn-lpl1-mm>`.


    Parameters
    ----------
    a : array-like (ns,)
        samples weights in the source domain
    labels_a : array-like (ns,)
        labels of samples in the source domain
    b : array-like (nt,)
        samples weights in the target domain
    M : array-like (ns,nt)
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
    gamma : (ns, nt) array-like
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-sinkhorn-lpl1-mm:
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
    a, labels_a, b, M = list_to_array(a, labels_a, b, M)
    nx = get_backend(a, labels_a, b, M)

    p = 0.5
    epsilon = 1e-3

    indices_labels = []
    classes = nx.unique(labels_a)
    for c in classes:
        idxc, = nx.where(labels_a == c)
        indices_labels.append(idxc)

    W = nx.zeros(M.shape, type_as=M)
    for cpt in range(numItermax):
        Mreg = M + eta * W
        transp = sinkhorn(a, b, Mreg, reg, numItermax=numInnerItermax,
                          stopThr=stopInnerThr)
        # the transport has been computed. Check if classes are really
        # separated
        W = nx.ones(M.shape, type_as=M)
        for (i, c) in enumerate(classes):
            majs = nx.sum(transp[indices_labels[i]], axis=0)
            majs = p * ((majs + epsilon) ** (p - 1))
            W[indices_labels[i]] = majs

    return transp


def sinkhorn_l1l2_gl(a, labels_a, b, M, reg, eta=0.1, numItermax=10,
                     numInnerItermax=200, stopInnerThr=1e-9, verbose=False,
                     log=False):
    r"""
    Solve the entropic regularization optimal transport problem with group
    lasso regularization

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot \Omega_e(\gamma) + \eta \ \Omega_g(\gamma)

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0


    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\Omega_e` is the entropic regularization term
      :math:`\Omega_e(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\Omega_g` is the group lasso regulaization term
      :math:`\Omega_g(\gamma)=\sum_{i,c} \|\gamma_{i,\mathcal{I}_c}\|^2`
      where  :math:`\mathcal{I}_c` are the index of samples from class
      `c` in the source domain.
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is the generalised conditional
    gradient as proposed in :ref:`[5, 7] <references-sinkhorn-l1l2-gl>`.


    Parameters
    ----------
    a : array-like (ns,)
        samples weights in the source domain
    labels_a : array-like (ns,)
        labels of samples in the source domain
    b : array-like (nt,)
        samples in the target domain
    M : array-like (ns,nt)
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
    gamma : (ns, nt) array-like
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-sinkhorn-l1l2-gl:
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
    a, labels_a, b, M = list_to_array(a, labels_a, b, M)
    nx = get_backend(a, labels_a, b, M)

    lstlab = nx.unique(labels_a)

    def f(G):
        res = 0
        for i in range(G.shape[1]):
            for lab in lstlab:
                temp = G[labels_a == lab, i]
                res += nx.norm(temp)
        return res

    def df(G):
        W = nx.zeros(G.shape, type_as=G)
        for i in range(G.shape[1]):
            for lab in lstlab:
                temp = G[labels_a == lab, i]
                n = nx.norm(temp)
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
    r"""Joint OT and linear mapping estimation as proposed in
    :ref:`[8] <references-joint-OT-mapping-linear>`.

    The function solves the following optimization problem:

    .. math::
        \min_{\gamma,L}\quad \|L(\mathbf{X_s}) - n_s\gamma \mathbf{X_t} \|^2_F +
          \mu \langle \gamma, \mathbf{M} \rangle_F + \eta \|L - \mathbf{I}\|^2_F

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) squared euclidean cost matrix between samples in
      :math:`\mathbf{X_s}` and :math:`\mathbf{X_t}` (scaled by :math:`n_s`)
    - :math:`L` is a :math:`d\times d` linear operator that approximates the barycentric
      mapping
    - :math:`\mathbf{I}` is the identity matrix (neutral linear mapping)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are uniform source and target weights

    The problem consist in solving jointly an optimal transport matrix
    :math:`\gamma` and a linear mapping that fits the barycentric mapping
    :math:`n_s\gamma \mathbf{X_t}`.

    One can also estimate a mapping with constant bias (see supplementary
    material of :ref:`[8] <references-joint-OT-mapping-linear>`) using the bias optional argument.

    The algorithm used for solving the problem is the block coordinate
    descent that alternates between updates of :math:`\mathbf{G}` (using conditionnal gradient)
    and the update of :math:`\mathbf{L}` using a classical least square solver.


    Parameters
    ----------
    xs : array-like (ns,d)
        samples in the source domain
    xt : array-like (nt,d)
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
    gamma : (ns, nt) array-like
        Optimal transportation matrix for the given parameters
    L : (d, d) array-like
        Linear mapping matrix ((:math:`d+1`, `d`) if bias)
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-joint-OT-mapping-linear:
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
    xs, xt = list_to_array(xs, xt)
    nx = get_backend(xs, xt)

    ns, nt, d = xs.shape[0], xt.shape[0], xt.shape[1]

    if bias:
        xs1 = nx.concatenate((xs, nx.ones((ns, 1), type_as=xs)), axis=1)
        xstxs = nx.dot(xs1.T, xs1)
        Id = nx.eye(d + 1, type_as=xs)
        Id[-1] = 0
        I0 = Id[:, :-1]

        def sel(x):
            return x[:-1, :]
    else:
        xs1 = xs
        xstxs = nx.dot(xs1.T, xs1)
        Id = nx.eye(d, type_as=xs)
        I0 = Id

        def sel(x):
            return x

    if log:
        log = {'err': []}

    a = unif(ns, type_as=xs)
    b = unif(nt, type_as=xt)
    M = dist(xs, xt) * ns
    G = emd(a, b, M)

    vloss = []

    def loss(L, G):
        """Compute full loss"""
        return (
            nx.sum((nx.dot(xs1, L) - ns * nx.dot(G, xt)) ** 2)
            + mu * nx.sum(G * M)
            + eta * nx.sum(sel(L - I0) ** 2)
        )

    def solve_L(G):
        """ solve L problem with fixed G (least square)"""
        xst = ns * nx.dot(G, xt)
        return nx.solve(xstxs + eta * Id, nx.dot(xs1.T, xst) + eta * I0)

    def solve_G(L, G0):
        """Update G with CG algorithm"""
        xsi = nx.dot(xs1, L)

        def f(G):
            return nx.sum((xsi - ns * nx.dot(G, xt)) ** 2)

        def df(G):
            return -2 * ns * nx.dot(xsi - ns * nx.dot(G, xt), xt.T)

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
    r"""Joint OT and nonlinear mapping estimation with kernels as proposed in
    :ref:`[8] <references-joint-OT-mapping-kernel>`.

    The function solves the following optimization problem:

    .. math::
        \min_{\gamma, L\in\mathcal{H}}\quad \|L(\mathbf{X_s}) -
        n_s\gamma \mathbf{X_t}\|^2_F + \mu \langle \gamma, \mathbf{M} \rangle_F +
        \eta \|L\|^2_\mathcal{H}

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0


    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) squared euclidean cost matrix between samples in
      :math:`\mathbf{X_s}` and :math:`\mathbf{X_t}` (scaled by :math:`n_s`)
    - :math:`L` is a :math:`n_s \times d` linear operator on a kernel matrix that
      approximates the barycentric mapping
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are uniform source and target weights

    The problem consist in solving jointly an optimal transport matrix
    :math:`\gamma` and the nonlinear mapping that fits the barycentric mapping
    :math:`n_s\gamma \mathbf{X_t}`.

    One can also estimate a mapping with constant bias (see supplementary
    material of :ref:`[8] <references-joint-OT-mapping-kernel>`) using the bias optional argument.

    The algorithm used for solving the problem is the block coordinate
    descent that alternates between updates of :math:`\mathbf{G}` (using conditionnal gradient)
    and the update of :math:`\mathbf{L}` using a classical kernel least square solver.


    Parameters
    ----------
    xs : array-like (ns,d)
        samples in the source domain
    xt : array-like (nt,d)
        samples in the target domain
    mu : float,optional
        Weight for the linear OT loss (>0)
    eta : float, optional
        Regularization term  for the linear mapping L (>0)
    kerneltype : str,optional
        kernel used by calling function :py:func:`ot.utils.kernel` (gaussian by default)
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
    gamma : (ns, nt) array-like
        Optimal transportation matrix for the given parameters
    L : (ns, d) array-like
        Nonlinear mapping matrix ((:math:`n_s+1`, `d`) if bias)
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-joint-OT-mapping-kernel:
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
    xs, xt = list_to_array(xs, xt)
    nx = get_backend(xs, xt)

    ns, nt = xs.shape[0], xt.shape[0]

    K = kernel(xs, xs, method=kerneltype, sigma=sigma)
    if bias:
        K1 = nx.concatenate((K, nx.ones((ns, 1), type_as=xs)), axis=1)
        Id = nx.eye(ns + 1, type_as=xs)
        Id[-1] = 0
        Kp = nx.eye(ns + 1, type_as=xs)
        Kp[:ns, :ns] = K

        # ls regu
        # K0 = K1.T.dot(K1)+eta*I
        # Kreg=I

        # RKHS regul
        K0 = nx.dot(K1.T, K1) + eta * Kp
        Kreg = Kp

    else:
        K1 = K
        Id = nx.eye(ns, type_as=xs)

        # ls regul
        # K0 = K1.T.dot(K1)+eta*I
        # Kreg=I

        # proper kernel ridge
        K0 = K + eta * Id
        Kreg = K

    if log:
        log = {'err': []}

    a = unif(ns, type_as=xs)
    b = unif(nt, type_as=xt)
    M = dist(xs, xt) * ns
    G = emd(a, b, M)

    vloss = []

    def loss(L, G):
        """Compute full loss"""
        return (
            nx.sum((nx.dot(K1, L) - ns * nx.dot(G, xt)) ** 2)
            + mu * nx.sum(G * M)
            + eta * nx.trace(dots(L.T, Kreg, L))
        )

    def solve_L_nobias(G):
        """ solve L problem with fixed G (least square)"""
        xst = ns * nx.dot(G, xt)
        return nx.solve(K0, xst)

    def solve_L_bias(G):
        """ solve L problem with fixed G (least square)"""
        xst = ns * nx.dot(G, xt)
        return nx.solve(K0, nx.dot(K1.T, xst))

    def solve_G(L, G0):
        """Update G with CG algorithm"""
        xsi = nx.dot(K1, L)

        def f(G):
            return nx.sum((xsi - ns * nx.dot(G, xt)) ** 2)

        def df(G):
            return -2 * ns * nx.dot(xsi - ns * nx.dot(G, xt), xt.T)

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
    r"""Return OT linear operator between samples.

    The function estimates the optimal linear operator that aligns the two
    empirical distributions. This is equivalent to estimating the closed
    form mapping between two Gaussian distributions :math:`\mathcal{N}(\mu_s,\Sigma_s)`
    and :math:`\mathcal{N}(\mu_t,\Sigma_t)` as proposed in
    :ref:`[14] <references-OT-mapping-linear>` and discussed in remark 2.29 in
    :ref:`[15] <references-OT-mapping-linear>`.

    The linear operator from source to target :math:`M`

    .. math::
        M(\mathbf{x})= \mathbf{A} \mathbf{x} + \mathbf{b}

    where :

    .. math::
        \mathbf{A} &= \Sigma_s^{-1/2} \left(\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2} \right)^{1/2}
        \Sigma_s^{-1/2}

        \mathbf{b} &= \mu_t - \mathbf{A} \mu_s

    Parameters
    ----------
    xs : array-like (ns,d)
        samples in the source domain
    xt : array-like (nt,d)
        samples in the target domain
    reg : float,optional
        regularization added to the diagonals of covariances (>0)
    ws : array-like (ns,1), optional
        weights for the source samples
    wt : array-like (ns,1), optional
        weights for the target samples
    bias: boolean, optional
        estimate bias :math:`\mathbf{b}` else :math:`\mathbf{b} = 0` (default:True)
    log : bool, optional
        record log if True


    Returns
    -------
    A : (d, d) array-like
        Linear operator
    b : (1, d) array-like
        bias
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-OT-mapping-linear:
    References
    ----------
    .. [14] Knott, M. and Smith, C. S. "On the optimal mapping of
        distributions", Journal of Optimization Theory and Applications
        Vol 43, 1984

    .. [15] Peyré, G., & Cuturi, M. (2017). "Computational Optimal
        Transport", 2018.


    """
    xs, xt = list_to_array(xs, xt)
    nx = get_backend(xs, xt)

    d = xs.shape[1]

    if bias:
        mxs = nx.mean(xs, axis=0)[None, :]
        mxt = nx.mean(xt, axis=0)[None, :]

        xs = xs - mxs
        xt = xt - mxt
    else:
        mxs = nx.zeros((1, d), type_as=xs)
        mxt = nx.zeros((1, d), type_as=xs)

    if ws is None:
        ws = nx.ones((xs.shape[0], 1), type_as=xs) / xs.shape[0]

    if wt is None:
        wt = nx.ones((xt.shape[0], 1), type_as=xt) / xt.shape[0]

    Cs = nx.dot((xs * ws).T, xs) / nx.sum(ws) + reg * nx.eye(d, type_as=xs)
    Ct = nx.dot((xt * wt).T, xt) / nx.sum(wt) + reg * nx.eye(d, type_as=xt)

    Cs12 = nx.sqrtm(Cs)
    Cs_12 = nx.inv(Cs12)

    M0 = nx.sqrtm(dots(Cs12, Ct, Cs12))

    A = dots(Cs_12, M0, Cs_12)

    b = mxt - nx.dot(mxs, A)

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
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \eta \cdot \Omega_\alpha(\gamma)

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    where:

    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)
    - :math:`\mathbf{x_s}` and :math:`\mathbf{x_t}` are source and target samples
    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\Omega_\alpha` is the Laplacian regularization term

    .. math::
        \Omega_\alpha = \frac{1 - \alpha}{n_s^2} \sum_{i,j}
        \mathbf{S^s}_{i,j} \|T(\mathbf{x}^s_i) - T(\mathbf{x}^s_j) \|^2 +
        \frac{\alpha}{n_t^2} \sum_{i,j}
        \mathbf{S^t}_{i,j} \|T(\mathbf{x}^t_i) - T(\mathbf{x}^t_j) \|^2


    with :math:`\mathbf{S^s}_{i,j}, \mathbf{S^t}_{i,j}` denoting source and target similarity
    matrices and :math:`T(\cdot)` being a barycentric mapping.

    The algorithm used for solving the problem is the conditional gradient algorithm as proposed in
    :ref:`[5] <references-emd-laplace>`.

    Parameters
    ----------
    a : array-like (ns,)
        samples weights in the source domain
    b : array-like (nt,)
        samples weights in the target domain
    xs : array-like (ns,d)
        samples in the source domain
    xt : array-like (nt,d)
        samples in the target domain
    M : array-like (ns,nt)
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
    gamma : (ns, nt) array-like
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-emd-laplace:
    References
    ----------
    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
       "Optimal Transport for Domain Adaptation," in IEEE
       Transactions on Pattern Analysis and Machine Intelligence,
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

    a, b, xs, xt, M = list_to_array(a, b, xs, xt, M)
    nx = get_backend(a, b, xs, xt, M)

    if sim == 'gauss':
        if sim_param is None:
            sim_param = 1 / (2 * (nx.mean(dist(xs, xs, 'sqeuclidean')) ** 2))
        sS = kernel(xs, xs, method=sim, sigma=sim_param)
        sT = kernel(xt, xt, method=sim, sigma=sim_param)

    elif sim == 'knn':
        if sim_param is None:
            sim_param = 3

        from sklearn.neighbors import kneighbors_graph

        sS = nx.from_numpy(kneighbors_graph(
            X=nx.to_numpy(xs), n_neighbors=int(sim_param)
        ).toarray(), type_as=xs)
        sS = (sS + sS.T) / 2
        sT = nx.from_numpy(kneighbors_graph(
            X=nx.to_numpy(xt), n_neighbors=int(sim_param)
        ).toarray(), type_as=xt)
        sT = (sT + sT.T) / 2
    else:
        raise ValueError('Unknown similarity type {sim}. Currently supported similarity types are "knn" and "gauss".'.format(sim=sim))

    lS = laplacian(sS)
    lT = laplacian(sT)

    def f(G):
        return (
            alpha * nx.trace(dots(xt.T, G.T, lS, G, xt))
            + (1 - alpha) * nx.trace(dots(xs.T, G, lT, G.T, xs))
        )

    ls2 = lS + lS.T
    lt2 = lT + lT.T
    xt2 = nx.dot(xt, xt.T)

    if reg == 'disp':
        Cs = -eta * alpha / xs.shape[0] * dots(ls2, xs, xt.T)
        Ct = -eta * (1 - alpha) / xt.shape[0] * dots(xs, xt.T, lt2)
        M = M + Cs + Ct

    def df(G):
        return (
            alpha * dots(ls2, G, xt2)
            + (1 - alpha) * dots(xs, xs.T, G, lt2)
        )

    return cg(a, b, M, reg=eta, f=f, df=df, G0=None, numItermax=numItermax, numItermaxEmd=numInnerItermax,
              stopThr=stopThr, stopThr2=stopInnerThr, verbose=verbose, log=log)


def distribution_estimation_uniform(X):
    r"""estimates a uniform distribution from an array of samples :math:`\mathbf{X}`

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The array of samples

    Returns
    -------
    mu : array-like, shape (n_samples,)
        The uniform distribution estimated from :math:`\mathbf{X}`
    """

    return unif(X.shape[0], type_as=X)


class BaseTransport(BaseEstimator):

    """Base class for OTDA objects

    .. note::
        All estimators should specify all the parameters that can be set
        at the class level in their ``__init__`` as explicit keyword
        arguments (no ``*args`` or ``**kwargs``).

    The fit method should:

    - estimate a cost matrix and store it in a `cost_` attribute
    - estimate a coupling matrix and store it in a `coupling_` attribute
    - estimate distributions from source and target data and store them in
      `mu_s` and `mu_t` attributes
    - store `Xs` and `Xt` in attributes to be used later on in `transform` and
      `inverse_transform` methods

    `transform` method should always get as input a `Xs` parameter

    `inverse_transform` method should always get as input a `Xt` parameter

    `transform_labels` method should always get as input a `ys` parameter

    `inverse_transform_labels` method should always get as input a `yt` parameter
    """

    def fit(self, Xs=None, ys=None, Xt=None, yt=None):
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The training class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """
        nx = self._get_backend(Xs, ys, Xt, yt)

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt):

            # pairwise distance
            self.cost_ = dist(Xs, Xt, metric=self.metric)
            self.cost_ = cost_normalization(self.cost_, self.norm)

            if (ys is not None) and (yt is not None):

                if self.limit_max != np.infty:
                    self.limit_max = self.limit_max * nx.max(self.cost_)

                # assumes labeled source samples occupy the first rows
                # and labeled target samples occupy the first columns
                classes = [c for c in nx.unique(ys) if c != -1]
                for c in classes:
                    idx_s = nx.where((ys != c) & (ys != -1))
                    idx_t = nx.where(yt == c)

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
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`
        and transports source samples :math:`\mathbf{X_s}` onto target ones :math:`\mathbf{X_t}`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels for training samples
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The source samples samples.
        """

        return self.fit(Xs, ys, Xt, yt).transform(Xs, ys, Xt, yt)

    def transform(self, Xs=None, ys=None, Xt=None, yt=None, batch_size=128):
        r"""Transports source samples :math:`\mathbf{X_s}` onto target ones :math:`\mathbf{X_t}`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The source input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels for source samples
        Xt : array-like, shape (n_target_samples, n_features)
            The target input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels for target. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The transport source samples.
        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs):

            if nx.array_equal(self.xs_, Xs):

                # perform standard barycentric mapping
                transp = self.coupling_ / nx.sum(self.coupling_, axis=1)[:, None]

                # set nans to 0
                transp[~ nx.isfinite(transp)] = 0

                # compute transported samples
                transp_Xs = nx.dot(transp, self.xt_)
            else:
                # perform out of sample mapping
                indices = nx.arange(Xs.shape[0])
                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xs = []
                for bi in batch_ind:
                    # get the nearest neighbor in the source domain
                    D0 = dist(Xs[bi], self.xs_)
                    idx = nx.argmin(D0, axis=1)

                    # transport the source samples
                    transp = self.coupling_ / nx.sum(
                        self.coupling_, axis=1)[:, None]
                    transp[~ nx.isfinite(transp)] = 0
                    transp_Xs_ = nx.dot(transp, self.xt_)

                    # define the transported points
                    transp_Xs_ = transp_Xs_[idx, :] + Xs[bi] - self.xs_[idx, :]

                    transp_Xs.append(transp_Xs_)

                transp_Xs = nx.concatenate(transp_Xs, axis=0)

            return transp_Xs

    def transform_labels(self, ys=None):
        r"""Propagate source labels :math:`\mathbf{y_s}` to obtain estimated target labels as in
        :ref:`[27] <references-basetransport-transform-labels>`.

        Parameters
        ----------
        ys : array-like, shape (n_source_samples,)
            The source class labels

        Returns
        -------
        transp_ys : array-like, shape (n_target_samples, nb_classes)
            Estimated soft target labels.


        .. _references-basetransport-transform-labels:
        References
        ----------
        .. [27] Ievgen Redko, Nicolas Courty, Rémi Flamary, Devis Tuia
           "Optimal transport for multi-source domain adaptation under target shift",
           International Conference on Artificial Intelligence and Statistics (AISTATS), 2019.

        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if check_params(ys=ys):

            ysTemp = label_normalization(nx.copy(ys))
            classes = nx.unique(ysTemp)
            n = len(classes)
            D1 = nx.zeros((n, len(ysTemp)), type_as=self.coupling_)

            # perform label propagation
            transp = self.coupling_ / nx.sum(self.coupling_, axis=0)[None, :]

            # set nans to 0
            transp[~ nx.isfinite(transp)] = 0

            for c in classes:
                D1[int(c), ysTemp == c] = 1

            # compute propagated labels
            transp_ys = nx.dot(D1, transp)

            return transp_ys.T

    def inverse_transform(self, Xs=None, ys=None, Xt=None, yt=None,
                          batch_size=128):
        r"""Transports target samples :math:`\mathbf{X_t}` onto source samples :math:`\mathbf{X_s}`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The source input samples.
        ys : array-like, shape (n_source_samples,)
            The source class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The target input samples.
        yt : array-like, shape (n_target_samples,)
            The target class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform

        Returns
        -------
        transp_Xt : array-like, shape (n_source_samples, n_features)
            The transported target samples.
        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if check_params(Xt=Xt):

            if nx.array_equal(self.xt_, Xt):

                # perform standard barycentric mapping
                transp_ = self.coupling_.T / nx.sum(self.coupling_, 0)[:, None]

                # set nans to 0
                transp_[~ nx.isfinite(transp_)] = 0

                # compute transported samples
                transp_Xt = nx.dot(transp_, self.xs_)
            else:
                # perform out of sample mapping
                indices = nx.arange(Xt.shape[0])
                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xt = []
                for bi in batch_ind:
                    D0 = dist(Xt[bi], self.xt_)
                    idx = nx.argmin(D0, axis=1)

                    # transport the target samples
                    transp_ = self.coupling_.T / nx.sum(
                        self.coupling_, 0)[:, None]
                    transp_[~ nx.isfinite(transp_)] = 0
                    transp_Xt_ = nx.dot(transp_, self.xs_)

                    # define the transported points
                    transp_Xt_ = transp_Xt_[idx, :] + Xt[bi] - self.xt_[idx, :]

                    transp_Xt.append(transp_Xt_)

                transp_Xt = nx.concatenate(transp_Xt, axis=0)

            return transp_Xt

    def inverse_transform_labels(self, yt=None):
        r"""Propagate target labels :math:`\mathbf{y_t}` to obtain estimated source labels
        :math:`\mathbf{y_s}`

        Parameters
        ----------
        yt : array-like, shape (n_target_samples,)

        Returns
        -------
        transp_ys : array-like, shape (n_source_samples, nb_classes)
            Estimated soft source labels.
        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if check_params(yt=yt):

            ytTemp = label_normalization(nx.copy(yt))
            classes = nx.unique(ytTemp)
            n = len(classes)
            D1 = nx.zeros((n, len(ytTemp)), type_as=self.coupling_)

            # perform label propagation
            transp = self.coupling_ / nx.sum(self.coupling_, 1)[:, None]

            # set nans to 0
            transp[~ nx.isfinite(transp)] = 0

            for c in classes:
                D1[int(c), ytTemp == c] = 1

            # compute propagated samples
            transp_ys = nx.dot(D1, transp.T)

            return transp_ys.T


class LinearTransport(BaseTransport):
    r""" OT linear operator between empirical distributions

    The function estimates the optimal linear operator that aligns the two
    empirical distributions. This is equivalent to estimating the closed
    form mapping between two Gaussian distributions :math:`\mathcal{N}(\mu_s,\Sigma_s)`
    and :math:`\mathcal{N}(\mu_t,\Sigma_t)` as proposed in
    :ref:`[14] <references-lineartransport>` and discussed in remark 2.29 in
    :ref:`[15] <references-lineartransport>`.

    The linear operator from source to target :math:`M`

    .. math::
        M(\mathbf{x})= \mathbf{A} \mathbf{x} + \mathbf{b}

    where :

    .. math::
        \mathbf{A} &= \Sigma_s^{-1/2} \left(\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2} \right)^{1/2}
        \Sigma_s^{-1/2}

        \mathbf{b} &= \mu_t - \mathbf{A} \mu_s

    Parameters
    ----------
    reg : float,optional
        regularization added to the daigonals of covariances (>0)
    bias: boolean, optional
        estimate bias :math:`\mathbf{b}` else :math:`\mathbf{b} = 0` (default:True)
    log : bool, optional
        record log if True


    .. _references-lineartransport:
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
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """
        nx = self._get_backend(Xs, ys, Xt, yt)

        self.mu_s = self.distribution_estimation(Xs)
        self.mu_t = self.distribution_estimation(Xt)

        # coupling estimation
        returned_ = OT_mapping_linear(Xs, Xt, reg=self.reg,
                                      ws=nx.reshape(self.mu_s, (-1, 1)),
                                      wt=nx.reshape(self.mu_t, (-1, 1)),
                                      bias=self.bias, log=self.log)

        # deal with the value of log
        if self.log:
            self.A_, self.B_, self.log_ = returned_
        else:
            self.A_, self.B_, = returned_
            self.log_ = dict()

        # re compute inverse mapping
        self.A1_ = nx.inv(self.A_)
        self.B1_ = -nx.dot(self.B_, self.A1_)

        return self

    def transform(self, Xs=None, ys=None, Xt=None, yt=None, batch_size=128):
        r"""Transports source samples :math:`\mathbf{X_s}` onto target ones :math:`\mathbf{X_t}`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The transport source samples.
        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs):
            transp_Xs = nx.dot(Xs, self.A_) + self.B_

            return transp_Xs

    def inverse_transform(self, Xs=None, ys=None, Xt=None, yt=None,
                          batch_size=128):
        r"""Transports target samples :math:`\mathbf{X_t}` onto source samples :math:`\mathbf{X_s}`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform

        Returns
        -------
        transp_Xt : array-like, shape (n_source_samples, n_features)
            The transported target samples.
        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if check_params(Xt=Xt):
            transp_Xt = nx.dot(Xt, self.A1_) + self.B1_

            return transp_Xt


class SinkhornTransport(BaseTransport):

    """Domain Adaptation OT method based on Sinkhorn Algorithm

    Parameters
    ----------
    reg_e : float, optional (default=1)
        Entropic regularization parameter
    max_iter : int, float, optional (default=1000)
        The minimum number of iteration before stopping the optimization
        algorithm if it has not converged
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
        "ferradans" which uses the method proposed in :ref:`[6] <references-sinkhorntransport>`.
    limit_max: float, optional (default=np.infty)
        Controls the semi supervised mode. Transport between labeled source
        and target samples of different classes will exhibit an cost defined
        by this variable

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling
    log_ : dictionary
        The dictionary of log, empty dict if parameter log is not True


    .. _references-sinkhorntransport:
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
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

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

    """Domain Adaptation OT method based on Earth Mover's Distance

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
        "ferradans" which uses the method proposed in :ref:`[6] <references-emdtransport>`.
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


    .. _references-emdtransport:
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
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

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
    r"""Domain Adaptation OT method based on sinkhorn algorithm +
    LpL1 class regularization.

    Parameters
    ----------
    reg_e : float, optional (default=1)
        Entropic regularization parameter
    reg_cl : float, optional (default=0.1)
        Class regularization parameter
    max_iter : int, float, optional (default=10)
        The minimum number of iteration before stopping the optimization
        algorithm if it has not converged
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
        "ferradans" which uses the method proposed in :ref:`[6] <references-sinkhornlpl1transport>`.
    limit_max: float, optional (default=np.infty)
        Controls the semi supervised mode. Transport between labeled source
        and target samples of different classes will exhibit a cost defined by
        limit_max.

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling


    .. _references-sinkhornlpl1transport:
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
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

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

    """Domain Adaptation OT method based on Earth Mover's Distance with Laplacian regularization

    Parameters
    ----------
    reg_type : string optional (default='pos')
        Type of the regularization term: 'pos' and 'disp' for
        regularization term defined in :ref:`[2] <references-emdlaplacetransport>` and
        :ref:`[6] <references-emdlaplacetransport>`, respectively.
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
        "ferradans" which uses the method proposed in :ref:`[6] <references-emdlaplacetransport>`.

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling


    .. _references-emdlaplacetransport:
    References
    ----------
    .. [1] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
           "Optimal Transport for Domain Adaptation," in IEEE Transactions
           on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

    .. [2] R. Flamary, N. Courty, D. Tuia, A. Rakotomamonjy,
        "Optimal transport with Laplacian regularization: Applications to domain adaptation and shape matching,"
        in NIPS Workshop on Optimal Transport and Machine Learning OTML, 2014.

    .. [6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
        Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.
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
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

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

    """Domain Adaptation OT method based on sinkhorn algorithm +
    L1L2 class regularization.

    Parameters
    ----------
    reg_e : float, optional (default=1)
        Entropic regularization parameter
    reg_cl : float, optional (default=0.1)
        Class regularization parameter
    max_iter : int, float, optional (default=10)
        The minimum number of iteration before stopping the optimization
        algorithm if it has not converged
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
        "ferradans" which uses the method proposed in :ref:`[6] <references-sinkhornl1l2transport>`.
    limit_max: float, optional (default=10)
        Controls the semi supervised mode. Transport between labeled source
        and target samples of different classes will exhibit an infinite cost
        (10 times the maximum value of the cost matrix)

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling
    log_ : dictionary
        The dictionary of log, empty dict if parameter log is not True


    .. _references-sinkhornl1l2transport:
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
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

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
        Regularization term for the linear mapping `L` (>0)
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
    mapping_ :
        The associated mapping

        - array-like, shape (`n_features` (+ 1), `n_features`),
          (if bias) for kernel == linear

        - array-like, shape (`n_source_samples` (+ 1), `n_features`),
          (if bias) for kernel == gaussian
    log_ : dictionary
        The dictionary of log, empty dict if parameter log is not True


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
        r"""Builds an optimal coupling and estimates the associated mapping
        from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self
        """
        self._get_backend(Xs, ys, Xt, yt)

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
        r"""Transports source samples :math:`\mathbf{X_s}` onto target ones :math:`\mathbf{X_t}`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The transport source samples.
        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs):

            if nx.array_equal(self.xs_, Xs):
                # perform standard barycentric mapping
                transp = self.coupling_ / nx.sum(self.coupling_, 1)[:, None]

                # set nans to 0
                transp[~ nx.isfinite(transp)] = 0

                # compute transported samples
                transp_Xs = nx.dot(transp, self.xt_)
            else:
                if self.kernel == "gaussian":
                    K = kernel(Xs, self.xs_, method=self.kernel,
                               sigma=self.sigma)
                elif self.kernel == "linear":
                    K = Xs
                if self.bias:
                    K = nx.concatenate(
                        [K, nx.ones((Xs.shape[0], 1), type_as=K)], axis=1
                    )
                transp_Xs = nx.dot(K, self.mapping_)

            return transp_Xs


class UnbalancedSinkhornTransport(BaseTransport):

    """Domain Adaptation unbalanced OT method based on sinkhorn algorithm

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
        algorithm if it has not converged
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
        "ferradans" which uses the method proposed in :ref:`[6] <references-unbalancedsinkhorntransport>`.
    limit_max: float, optional (default=10)
        Controls the semi supervised mode. Transport between labeled source
        and target samples of different classes will exhibit an infinite cost
        (10 times the maximum value of the cost matrix)

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling
    log_ : dictionary
        The dictionary of log, empty dict if parameter log is not True


    .. _references-unbalancedsinkhorntransport:
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
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

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

    """Domain Adaptation OT method for multi-source target shift based on Wasserstein barycenter algorithm.

    Parameters
    ----------
    reg_e : float, optional (default=1)
        Entropic regularization parameter
    max_iter : int, float, optional (default=10)
        The minimum number of iteration before stopping the optimization
        algorithm if it has not converged
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
        "ferradans" which uses the method proposed in :ref:`[6] <references-jcpottransport>`.

    Attributes
    ----------
    coupling_ : list of array-like objects, shape K x (n_source_samples, n_target_samples)
        A set of optimal couplings between each source domain and the target domain
    proportions_ : array-like, shape (n_classes,)
        Estimated class proportions in the target domain
    log_ : dictionary
        The dictionary of log, empty dict if parameter log is not True


    .. _references-jcpottransport:
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
        r"""Building coupling matrices from a list of source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : list of K array-like objects, shape K x (nk_source_samples, n_features)
            A list of the training input samples.
        ys : list of K array-like objects, shape K x (nk_source_samples,)
            A list of the class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """
        self._get_backend(*Xs, *ys, Xt, yt)

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
        r"""Transports source samples :math:`\mathbf{X_s}` onto target ones :math:`\mathbf{X_t}`

        Parameters
        ----------
        Xs : list of K array-like objects, shape K x (nk_source_samples, n_features)
            A list of the training input samples.
        ys : list of K array-like objects, shape K x (nk_source_samples,)
            A list of the class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform
        """
        nx = self.nx

        transp_Xs = []

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs):

            if all([nx.allclose(x, y) for x, y in zip(self.xs_, Xs)]):

                # perform standard barycentric mapping for each source domain

                for coupling in self.coupling_:
                    transp = coupling / nx.sum(coupling, 1)[:, None]

                    # set nans to 0
                    transp[~ nx.isfinite(transp)] = 0

                    # compute transported samples
                    transp_Xs.append(nx.dot(transp, self.xt_))
            else:

                # perform out of sample mapping
                indices = nx.arange(Xs.shape[0])
                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xs = []

                for bi in batch_ind:
                    transp_Xs_ = []

                    # get the nearest neighbor in the sources domains
                    xs = nx.concatenate(self.xs_, axis=0)
                    idx = nx.argmin(dist(Xs[bi], xs), axis=1)

                    # transport the source samples
                    for coupling in self.coupling_:
                        transp = coupling / nx.sum(coupling, 1)[:, None]
                        transp[~ nx.isfinite(transp)] = 0
                        transp_Xs_.append(nx.dot(transp, self.xt_))

                    transp_Xs_ = nx.concatenate(transp_Xs_, axis=0)

                    # define the transported points
                    transp_Xs_ = transp_Xs_[idx, :] + Xs[bi] - xs[idx, :]
                    transp_Xs.append(transp_Xs_)

                transp_Xs = nx.concatenate(transp_Xs, axis=0)

            return transp_Xs

    def transform_labels(self, ys=None):
        r"""Propagate source labels :math:`\mathbf{y_s}` to obtain target labels as in
        :ref:`[27] <references-jcpottransport-transform-labels>`

        Parameters
        ----------
        ys : list of K array-like objects, shape K x (nk_source_samples,)
            A list of the class labels

        Returns
        -------
        yt : array-like, shape (n_target_samples, nb_classes)
            Estimated soft target labels.


        .. _references-jcpottransport-transform-labels:
        References
        ----------
        .. [27] Ievgen Redko, Nicolas Courty, Rémi Flamary, Devis Tuia
           "Optimal transport for multi-source domain adaptation under target shift",
           International Conference on Artificial Intelligence and Statistics (AISTATS), 2019.
        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if check_params(ys=ys):
            yt = nx.zeros(
                (len(nx.unique(nx.concatenate(ys))), self.xt_.shape[0]),
                type_as=ys[0]
            )
            for i in range(len(ys)):
                ysTemp = label_normalization(nx.copy(ys[i]))
                classes = nx.unique(ysTemp)
                n = len(classes)
                ns = len(ysTemp)

                # perform label propagation
                transp = self.coupling_[i] / nx.sum(self.coupling_[i], 1)[:, None]

                # set nans to 0
                transp[~ nx.isfinite(transp)] = 0

                if self.log:
                    D1 = self.log_['D1'][i]
                else:
                    D1 = nx.zeros((n, ns), type_as=transp)

                    for c in classes:
                        D1[int(c), ysTemp == c] = 1

                # compute propagated labels
                yt = yt + nx.dot(D1, transp) / len(ys)

            return yt.T

    def inverse_transform_labels(self, yt=None):
        r"""Propagate target labels :math:`\mathbf{y_t}` to obtain estimated source labels
        :math:`\mathbf{y_s}`

        Parameters
        ----------
        yt : array-like, shape (n_target_samples,)
            The target class labels

        Returns
        -------
        transp_ys : list of K array-like objects, shape K x (nk_source_samples, nb_classes)
            A list of estimated soft source labels
        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if check_params(yt=yt):
            transp_ys = []
            ytTemp = label_normalization(nx.copy(yt))
            classes = nx.unique(ytTemp)
            n = len(classes)
            D1 = nx.zeros((n, len(ytTemp)), type_as=self.coupling_[0])

            for c in classes:
                D1[int(c), ytTemp == c] = 1

            for i in range(len(self.xs_)):

                # perform label propagation
                transp = self.coupling_[i] / nx.sum(self.coupling_[i], 1)[:, None]

                # set nans to 0
                transp[~ nx.isfinite(transp)] = 0

                # compute propagated labels
                transp_ys.append(nx.dot(D1, transp.T).T)

            return transp_ys
