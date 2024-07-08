# -*- coding: utf-8 -*-
"""
Gromov-Wasserstein, Fused-Gromov-Wasserstein and unbalanced Gromov-Wasserstein utils.
"""

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@unice.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#         Huy Tran <huytran82125@gmail.com>
#
# License: MIT License


from ..utils import list_to_array
from ..backend import get_backend
from ot.unbalanced import sinkhorn_unbalanced, mm_unbalanced, lbfgsb_unbalanced


def init_matrix(C1, C2, p, q, loss_fun='square_loss', nx=None):
    r"""Return loss matrices and tensors for Gromov-Wasserstein fast computation

    Returns the value of :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` with the
    selected loss function as the loss function of Gromov-Wasserstein discrepancy.

    The matrices are computed as described in Proposition 1 in :ref:`[12] <references-init-matrix>`

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{T}`: A coupling between those two spaces

    The square-loss function :math:`L(a, b) = |a - b|^2` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a^2

                        f_2(b) &= b^2

                        h_1(a) &= a

                        h_2(b) &= 2b

    The kl-loss function :math:`L(a, b) = a \log\left(\frac{a}{b}\right) - a + b` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a \log(a) - a

                        f_2(b) &= b

                        h_1(a) &= a

                        h_2(b) &= \log(b)

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,)
        Probability distribution in the source space
    q : array-like, shape (nt,)
        Probability distribution in the target space
    loss_fun : str, optional
        Name of loss function to use: either 'square_loss' or 'kl_loss' (default='square_loss')
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)


    .. _references-init-matrix:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if nx is None:
        C1, C2, p, q = list_to_array(C1, C2, p, q)
        nx = get_backend(C1, C2, p, q)

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
            return a * nx.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return nx.log(b + 1e-15)
    else:
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

    constC1 = nx.dot(
        nx.dot(f1(C1), nx.reshape(p, (-1, 1))),
        nx.ones((1, len(q)), type_as=q)
    )
    constC2 = nx.dot(
        nx.ones((len(p), 1), type_as=p),
        nx.dot(nx.reshape(q, (1, -1)), f2(C2).T)
    )
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2


def tensor_product(constC, hC1, hC2, T, nx=None):
    r"""Return the tensor for Gromov-Wasserstein fast computation

    The tensor is computed as described in Proposition 1 Eq. (6) in :ref:`[12] <references-tensor-product>`

    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    tens : array-like, shape (`ns`, `nt`)
        :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` tensor-matrix multiplication result


    .. _references-tensor-product:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if nx is None:
        constC, hC1, hC2, T = list_to_array(constC, hC1, hC2, T)
        nx = get_backend(constC, hC1, hC2, T)

    A = - nx.dot(
        nx.dot(hC1, T), hC2.T
    )
    tens = constC + A
    # tens -= tens.min()
    return tens


def gwloss(constC, hC1, hC2, T, nx=None):
    r"""Return the Loss for Gromov-Wasserstein

    The loss is computed as described in Proposition 1 Eq. (6) in :ref:`[12] <references-gwloss>`

    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    T : array-like, shape (ns, nt)
        Current value of transport matrix :math:`\mathbf{T}`
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    loss : float
        Gromov-Wasserstein loss


    .. _references-gwloss:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    tens = tensor_product(constC, hC1, hC2, T, nx)
    if nx is None:
        tens, T = list_to_array(tens, T)
        nx = get_backend(tens, T)

    return nx.sum(tens * T)


def gwggrad(constC, hC1, hC2, T, nx=None):
    r"""Return the gradient for Gromov-Wasserstein

    The gradient is computed as described in Proposition 2 in :ref:`[12] <references-gwggrad>`

    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    T : array-like, shape (ns, nt)
        Current value of transport matrix :math:`\mathbf{T}`
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    grad : array-like, shape (`ns`, `nt`)
        Gromov-Wasserstein gradient


    .. _references-gwggrad:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    return 2 * tensor_product(constC, hC1, hC2,
                              T, nx)  # [12] Prop. 2 misses a 2 factor


def update_square_loss(p, lambdas, T, Cs, nx=None):
    r"""
    Updates :math:`\mathbf{C}` according to the L2 Loss kernel with the `S`
    :math:`\mathbf{T}_s` couplings calculated at each iteration of the GW
    barycenter problem in :ref:`[12]`:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    p : array-like, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the `S` spaces' weights.
    T : list of S array-like of shape (N, ns)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape(ns,ns)
        Metric cost matrices.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    ----------
    C : array-like, shape (`nt`, `nt`)
        Updated :math:`\mathbf{C}` matrix.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if nx is None:
        nx = get_backend(p, *T, *Cs)

    # Correct order mistake in Equation 14 in [12]
    tmpsum = sum([
        lambdas[s] * nx.dot(
            nx.dot(T[s], Cs[s]),
            T[s].T
        ) for s in range(len(T))
    ])
    ppt = nx.outer(p, p)

    return tmpsum / ppt


def update_kl_loss(p, lambdas, T, Cs, nx=None):
    r"""
    Updates :math:`\mathbf{C}` according to the KL Loss kernel with the `S`
    :math:`\mathbf{T}_s` couplings calculated at each iteration of the GW
    barycenter problem in :ref:`[12]`:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution


    Parameters
    ----------
    p  : array-like, shape (N,)
        Weights in the targeted barycenter.
    lambdas : list of float
        List of the `S` spaces' weights
    T : list of S array-like of shape (N, ns)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape(ns,ns)
        Metric cost matrices.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    ----------
    C : array-like, shape (`ns`, `ns`)
        updated :math:`\mathbf{C}` matrix

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if nx is None:
        nx = get_backend(p, *T, *Cs)

    # Correct order mistake in Equation 15 in [12]
    tmpsum = sum([
        lambdas[s] * nx.dot(
            nx.dot(T[s], nx.log(nx.maximum(Cs[s], 1e-15))),
            T[s].T
        ) for s in range(len(T))
    ])
    ppt = nx.outer(p, p)

    return nx.exp(tmpsum / ppt)


def update_feature_matrix(lambdas, Ys, Ts, p, nx=None):
    r"""Updates the feature with respect to the `S` :math:`\mathbf{T}_s` couplings.


    See "Solving the barycenter problem with Block Coordinate Descent (BCD)"
    in :ref:`[24] <references-update-feature-matrix>` calculated at each iteration

    Parameters
    ----------
    p : array-like, shape (N,)
        masses in the targeted barycenter
    lambdas : list of float
        List of the `S` spaces' weights
    Ts : list of S array-like, shape (N, ns)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration
    Ys : list of S array-like, shape (d,ns)
        The features.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    X : array-like, shape (`d`, `N`)


    .. _references-update-feature-matrix:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    if nx is None:
        nx = get_backend(*Ys, *Ts, p)

    p = 1. / p
    tmpsum = sum([
        lambdas[s] * nx.dot(Ys[s], Ts[s].T) * p[None, :]
        for s in range(len(Ts))
    ])
    return tmpsum


def init_matrix_semirelaxed(C1, C2, p, loss_fun='square_loss', nx=None):
    r"""Return loss matrices and tensors for semi-relaxed Gromov-Wasserstein fast computation

    Returns the value of :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` with the
    selected loss function as the loss function of semi-relaxed Gromov-Wasserstein discrepancy.

    The matrices are computed as described in Proposition 1 in :ref:`[12] <references-init-matrix>`
    and adapted to the semi-relaxed problem where the second marginal is not a constant anymore.

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{T}`: A coupling between those two spaces

    The square-loss function :math:`L(a, b) = |a - b|^2` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a^2

                        f_2(b) &= b^2

                        h_1(a) &= a

                        h_2(b) &= 2b

    The kl-loss function :math:`L(a, b) = a \log\left(\frac{a}{b}\right) - a + b` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a \log(a) - a

                        f_2(b) &= b

                        h_1(a) &= a

                        h_2(b) &= \log(b)
    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,)
    loss_fun : str, optional
        Name of loss function to use: either 'square_loss' or 'kl_loss' (default='square_loss')
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6) adapted to srGW
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    fC2t: array-like, shape (nt, nt)
        :math:`\mathbf{f2}(\mathbf{C2})^\top` matrix in Eq. (6)


    .. _references-init-matrix:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [48]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.
    """
    if nx is None:
        C1, C2, p = list_to_array(C1, C2, p)
        nx = get_backend(C1, C2, p)

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
            return a * nx.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return nx.log(b + 1e-15)
    else:
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

    constC = nx.dot(nx.dot(f1(C1), nx.reshape(p, (-1, 1))),
                    nx.ones((1, C2.shape[0]), type_as=p))

    hC1 = h1(C1)
    hC2 = h2(C2)
    fC2t = f2(C2).T
    return constC, hC1, hC2, fC2t


############################################################################
# Methods related to fused unbalanced GW and unbalanced Co-Optimal Transport.
############################################################################

# Calculate Div(pi, a \otimes b) for Div = KL and squared L2.
# Calculate KL(pi, a \otimes b) using
# the marginal distributions pi1, pi2 of pi
def approx_shortcut_kl(pi, pi1, pi2, a, b, nx=None):
    """
    Implement:
    < pi, log pi / (a \otimes b) >
    = <pi, log pi> - <pi1, log a> - <pi2, log b>.
    """

    if nx is None:
        pi, pi1, pi2, a, b = list_to_array(pi, pi1, pi2, a, b)
        nx = get_backend(pi, pi1, pi2, a, b)

    res = nx.sum(pi * nx.log(pi + 1.0 * (pi == 0))) \
        - nx.sum(pi1 * nx.log(a)) - nx.sum(pi2 * nx.log(b))

    return res


def div(pi, pi1, pi2, a, b, divergence, nx=None):
    """
    Calculate Div(pi, a \otimes b).
    """

    if nx is None:
        pi, pi1, pi2, a, b = list_to_array(pi, pi1, pi2, a, b)
        nx = get_backend(pi, pi1, pi2, a, b)

    if divergence == "kl":
        res = approx_shortcut_kl(pi, pi1, pi2, a, b, nx) \
            - nx.sum(pi1) + nx.sum(a) * nx.sum(b)
    elif divergence == "l2":
        res = (nx.sum(pi**2) + nx.sum(a**2) * nx.sum(b**2)
               - 2 * nx.dot(a, nx.dot(pi, b))) / 2

    return res


# Support functions for KL and squared L2 between product measures:
# Calculate Div(mu \otimes nu, alpha \otimes beta).
def approx_kl(p, q, nx=None):
    if nx is None:
        p, q = list_to_array(p, q)
        nx = get_backend(p, q)

    return nx.sum(p * nx.log(p + 1.0 * (p == 0))) - nx.sum(p * nx.log(q))


def product_kl(mu, nu, alpha, beta, nx=None):
    """
    Calculate the KL divergence between two product measures:
    KL(mu \otimes nu, alpha \otimes beta) =
    m_mu * KL(nu, beta) + m_nu * KL(mu, alpha) +
    (m_mu - m_alpha) * (m_nu - m_beta)

    Parameters
    ----------
    mu: vector or matrix
    nu: vector or matrix
    alpha: vector or matrix with the same size as mu
    beta: vector or matrix with the same size as nu

    Returns
    ----------
    KL divergence between two product measures
    """

    def kl(p, q, nx=None):
        return approx_kl(p, q, nx=nx) - nx.sum(p) + nx.sum(q)

    if nx is None:
        mu, nu, alpha, beta = list_to_array(mu, nu, alpha, beta)
        nx = get_backend(mu, nu, alpha, beta)

    m_mu, m_nu = nx.sum(mu), nx.sum(nu)
    m_alpha, m_beta = nx.sum(alpha), nx.sum(beta)
    const = (m_mu - m_alpha) * (m_nu - m_beta)
    res = m_nu * kl(mu, alpha, nx) + m_mu * kl(nu, beta, nx) + const

    return res


def product_l2(mu, nu, alpha, beta, nx=None):
    """
    norm = ||mu \otimes nu - alpha \otimes beta ||^2
    = ||a||^2 ||b||^2 + ||mu||^2 ||nu||^2 - 2 < alpha, mu > < beta, nu >.
    L2(mu \otimes nu, alpha \otimes beta) = norm / 2.
    """

    if nx is None:
        mu, nu, alpha, beta = list_to_array(mu, nu, alpha, beta)
        nx = get_backend(mu, nu, alpha, beta)

    norm = nx.sum(alpha**2) * nx.sum(beta**2) \
        - 2 * nx.sum(alpha * mu) * nx.sum(beta * nu) \
        + nx.sum(mu**2) * nx.sum(nu**2)

    return norm / 2


def product_div(mu, nu, alpha, beta, divergence, nx=None):
    if nx is None:
        mu, nu, alpha, beta = list_to_array(mu, nu, alpha, beta)
        nx = get_backend(mu, nu, alpha, beta)

    if divergence == "kl":
        return product_kl(mu, nu, alpha, beta, nx)
    elif divergence == "l2":
        return product_l2(mu, nu, alpha, beta, nx)


# Support functions for BCD schemes
def local_cost(data, pi, tuple_p, hyperparams, divergence, reg_type, nx=None):
    """
    Calculate cost matrix of the UOT subroutine
    """

    X_sqr, Y_sqr, X, Y, M = data
    rho_x, rho_y, eps = hyperparams
    a, b = tuple_p

    if nx is None:
        X, Y, a, b = list_to_array(X, Y, a, b)
        nx = get_backend(X, Y, a, b)

    pi1, pi2 = nx.sum(pi, 1), nx.sum(pi, 0)
    A, B = nx.dot(X_sqr, pi1), nx.dot(Y_sqr, pi2)
    uot_cost = A[:, None] + B[None, :] - 2 * nx.dot(nx.dot(X, pi), Y.T)
    if M is not None:
        uot_cost = uot_cost + M

    if divergence == "kl":
        if rho_x != float("inf") and rho_x != 0:
            uot_cost = uot_cost + rho_x * approx_kl(pi1, a, nx)
        if rho_y != float("inf") and rho_y != 0:
            uot_cost = uot_cost + rho_y * approx_kl(pi2, b, nx)
        if reg_type == "joint" and eps > 0:
            uot_cost = uot_cost + eps * approx_shortcut_kl(pi, pi1, pi2, a, b, nx)

    return uot_cost


def total_cost(M_linear, data, tuple_pxy_samp, tuple_pxy_feat,
               pi_samp, pi_feat, hyperparams, divergence, reg_type, nx=None):

    rho_x, rho_y, eps_samp, eps_feat = hyperparams
    M_samp, M_feat = M_linear
    px_samp, py_samp, pxy_samp = tuple_pxy_samp
    px_feat, py_feat, pxy_feat = tuple_pxy_feat
    X_sqr, Y_sqr, X, Y = data

    if nx is None:
        X, Y = list_to_array(X, Y)
        nx = get_backend(X, Y)

    pi1_samp, pi2_samp = nx.sum(pi_samp, 1), nx.sum(pi_samp, 0)
    pi1_feat, pi2_feat = nx.sum(pi_feat, 1), nx.sum(pi_feat, 0)

    A_sqr = nx.dot(nx.dot(X_sqr, pi1_feat), pi1_samp)
    B_sqr = nx.dot(nx.dot(Y_sqr, pi2_feat), pi2_samp)
    AB = nx.dot(nx.dot(X, pi_feat), Y.T) * pi_samp
    linear_cost = A_sqr + B_sqr - 2 * nx.sum(AB)

    ucoot_cost = linear_cost
    if M_samp is not None:
        ucoot_cost = ucoot_cost + nx.sum(pi_samp * M_samp)
    if M_feat is not None:
        ucoot_cost = ucoot_cost + nx.sum(pi_feat * M_feat)

    if rho_x != float("inf") and rho_x != 0:
        ucoot_cost = ucoot_cost + \
            rho_x * product_div(pi1_samp, pi1_feat, px_samp, px_feat, divergence, nx)
    if rho_y != float("inf") and rho_y != 0:
        ucoot_cost = ucoot_cost + \
            rho_y * product_div(pi2_samp, pi2_feat, py_samp, py_feat, divergence, nx)

    if reg_type == "joint" and eps_samp != 0:
        div_cost = product_div(pi_samp, pi_feat, pxy_samp, pxy_feat, divergence, nx)
        ucoot_cost = ucoot_cost + eps_samp * div_cost
    elif reg_type == "independent":
        if eps_samp != 0:
            div_samp = div(pi_samp, pi1_samp, pi2_samp, px_samp, py_samp, divergence, nx)
            ucoot_cost = ucoot_cost + eps_samp * div_samp
        if eps_feat != 0:
            div_feat = div(pi_feat, pi1_feat, pi2_feat, px_feat, py_feat, divergence, nx)
            ucoot_cost = ucoot_cost + eps_feat * div_feat

    return linear_cost, ucoot_cost


# Support functions for squared L2 norm
def parameters_uot_l2(pi, tuple_weights, hyperparams, reg_type, nx=None):
    """Compute parameters of the L2 loss."""

    rho_x, rho_y, eps = hyperparams
    wx, wy, wxy = tuple_weights

    if nx is None:
        wx, wy, wxy = list_to_array(wx, wy, wxy)
        nx = get_backend(wx, wy, wxy)

    pi1, pi2 = nx.sum(pi, 1), nx.sum(pi, 0)
    l2_pi1, l2_pi2, l2_pi = nx.sum(pi1**2), nx.sum(pi2**2), nx.sum(pi**2)

    weighted_wx = wx * nx.sum(pi1 * wx) / l2_pi1
    weighted_wy = wy * nx.sum(pi2 * wy) / l2_pi2
    weighted_wxy = wxy * nx.sum(pi * wxy) / l2_pi if reg_type == "joint" else wxy
    weighted_w = (weighted_wx, weighted_wy, weighted_wxy)

    new_rho = (rho_x * l2_pi1, rho_y * l2_pi2)
    new_eps = eps * l2_pi if reg_type == "joint" else eps

    return weighted_w, new_rho, new_eps


# UOT solver for KL and squared L2
def uot_solver(wx, wy, wxy, cost, eps, rho,
               init_pi, init_duals, divergence, unbalanced_solver,
               method_sinkhorn="sinkhorn", max_iter_ot=500, tol_ot=1e-7):

    if unbalanced_solver == "scaling":
        pi, log = sinkhorn_unbalanced(
            a=wx, b=wy, M=cost, reg=eps, reg_m=rho, reg_type="kl",
            warmstart=init_duals, method=method_sinkhorn,
            numItermax=max_iter_ot, stopThr=tol_ot, verbose=False, log=True)
        duals = (log['logu'], log['logv'])

    elif unbalanced_solver == "mm":
        pi = mm_unbalanced(a=wx, b=wy, M=cost, reg_m=rho,
                           c=wxy, reg=eps, div=divergence,
                           G0=init_pi, numItermax=max_iter_ot,
                           stopThr=tol_ot, verbose=False, log=False)
        duals = (None, None)

    elif unbalanced_solver == "lbfgsb":
        pi = lbfgsb_unbalanced(a=wx, b=wy, M=cost, reg=eps, reg_m=rho,
                               c=wxy, reg_div=divergence,
                               regm_div=divergence,
                               G0=init_pi, numItermax=max_iter_ot,
                               stopThr=tol_ot, method='L-BFGS-B',
                               verbose=False, log=False)
        duals = (None, None)

    return pi, duals
