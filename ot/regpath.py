# -*- coding: utf-8 -*-
"""
Regularization path OT solvers
"""

# Author: Haoran Wu <haoran.wu@univ-ubs.fr>
# License: MIT License

import numpy as np
import scipy.sparse as sp


def recast_ot_as_lasso(a, b, C):
    r"""This function recasts the l2-penalized UOT problem as a Lasso problem

    Recall the l2-penalized UOT problem defined in [Chapel et al., 2021]
    .. math::
        UOT = \min_T <C, T> + \lambda \|T 1_m - a\|_2^2 +
                \lambda \|T^T 1_n - b\|_2^2
        s.t.
            T \geq 0
    where :
    - C is the (dim_a, dim_b) metric cost matrix
    - :math:`\lambda` is the l2-regularization coefficient
    - a and b are source and target distributions
    - T is the transport plan to optimize

    The problem above can be reformulated to a non-negative penalized
    linear regression problem, particularly Lasso
    .. math::
        UOT2 = \min_t \gamma c^T t + 0.5 * \|H t - y\|_2^2
        s.t.
            t \geq 0
    where :
    - c is a (dim_a * dim_b, ) metric cost vector (flattened version of C)
    - :math:`\gamma = 1/\lambda` is the l2-regularization coefficient
    - y is the concatenation of vectors a and b, defined as y^T = [a^T b^T]
    - H is a (dim_a + dim_b, dim_a * dim_b) metric matrix,
        see [Chapel et al., 2021] for the design of H. The matrix product H t
        computes both the source marginal and the target marginal.
    - t is a (dim_a * dim_b, ) metric vector (flattened version of T)
    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Histogram of dimension dim_a
    b : np.ndarray (dim_b,)
        Histogram of dimension dim_b
    C : np.ndarray, shape (dim_a, dim_b)
        Cost matrix
    Returns
    -------
    H : np.ndarray (dim_a+dim_b, dim_a*dim_b)
        Auxiliary matrix constituted by 0 and 1
    y : np.ndarray (ns + nt, )
        Concatenation of histogram a and histogram b
    c : np.ndarray (ns * nt, )
        Flattened array of cost matrix
    Examples
    --------
    >>> import ot
    >>> a = np.array([0.2, 0.3, 0.5])
    >>> b = np.array([0.1, 0.9])
    >>> C = np.array([[16., 25.], [28., 16.], [40., 36.]])
    >>> H, y, c = ot.regpath.recast_ot_as_lasso(a, b, C)
    >>> H.toarray()
    array([[1., 1., 0., 0., 0., 0.],
           [0., 0., 1., 1., 0., 0.],
           [0., 0., 0., 0., 1., 1.],
           [1., 0., 1., 0., 1., 0.],
           [0., 1., 0., 1., 0., 1.]])
    >>> y
    array([0.2, 0.3, 0.5, 0.1, 0.9])
    >>> c
    array([16., 25., 28., 16., 40., 36.])

    References
    ----------
    [Chapel et al., 2021]:
        Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression.
    """

    dim_a = np.shape(a)[0]
    dim_b = np.shape(b)[0]
    y = np.concatenate((a, b))
    c = C.flatten()
    jHa = np.arange(dim_a * dim_b)
    iHa = np.repeat(np.arange(dim_a), dim_b)
    jHb = np.arange(dim_a * dim_b)
    iHb = np.tile(np.arange(dim_b), dim_a) + dim_a
    j = np.concatenate((jHa, jHb))
    i = np.concatenate((iHa, iHb))
    H = sp.csc_matrix((np.ones(dim_a * dim_b * 2), (i, j)),
                      shape=(dim_a + dim_b, dim_a * dim_b))
    return H, y, c


def recast_semi_relaxed_as_lasso(a, b, C):
    r"""This function recasts the semi-relaxed l2-UOT problem as Lasso problem

    .. math::
        semi-relaxed UOT = \min_T <C, T> + \lambda \|T 1_m - a\|_2^2
        s.t.
            T^T 1_n = b
            t \geq 0
    where :
    - C is the (dim_a, dim_b) metric cost matrix
    - :math:`\lambda` is the l2-regularization coefficient
    - a and b are source and target distributions
    - T is the transport plan to optimize

    The problem above can be reformulated as follows
    .. math::
        semi-relaxed UOT2 = \min_t \gamma c^T t + 0.5 * \|H_r t - a\|_2^2
        s.t.
            H_c t = b
            t \geq 0
    where :
    - c is a (dim_a * dim_b, ) metric cost vector (flattened version of C)
    - :math:`\gamma = 1/\lambda` is the l2-regularization coefficient
    - H_r is  a (dim_a, dim_a * dim_b) metric matrix,
        which computes the sum along the rows of transport plan T
    - H_c is a (dim_b, dim_a * dim_b) metric matrix,
        which computes the sum along the columns of transport plan T
    - t is a (dim_a * dim_b, ) metric vector (flattened version of T)
    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Histogram of dimension dim_a
    b : np.ndarray (dim_b,)
        Histogram of dimension dim_b
    C : np.ndarray, shape (dim_a, dim_b)
        Cost matrix
    Returns
    -------
    Hr : np.ndarray (dim_a, dim_a * dim_b)
        Auxiliary matrix constituted by 0 and 1, which computes
        the sum along the rows of transport plan T
    Hc : np.ndarray (dim_b, dim_a * dim_b)
        Auxiliary matrix constituted by 0 and 1, which computes
        the sum along the columns of transport plan T
    c : np.ndarray (ns * nt, )
        Flattened array of cost matrix
    Examples
    --------
    >>> import ot
    >>> a = np.array([0.2, 0.3, 0.5])
    >>> b = np.array([0.1, 0.9])
    >>> C = np.array([[16., 25.], [28., 16.], [40., 36.]])
    >>> Hr,Hc,c = ot.regpath.recast_semi_relaxed_as_lasso(a, b, C)
    >>> Hr.toarray()
    array([[1., 1., 0., 0., 0., 0.],
           [0., 0., 1., 1., 0., 0.],
           [0., 0., 0., 0., 1., 1.]])
    >>> Hc.toarray()
    array([[1., 0., 1., 0., 1., 0.],
           [0., 1., 0., 1., 0., 1.]])
    >>> c
    array([16., 25., 28., 16., 40., 36.])
    """

    dim_a = np.shape(a)[0]
    dim_b = np.shape(b)[0]

    c = C.flatten()
    jHr = np.arange(dim_a * dim_b)
    iHr = np.repeat(np.arange(dim_a), dim_b)
    jHc = np.arange(dim_a * dim_b)
    iHc = np.tile(np.arange(dim_b), dim_a)

    Hr = sp.csc_matrix((np.ones(dim_a * dim_b), (iHr, jHr)),
                       shape=(dim_a, dim_a * dim_b))
    Hc = sp.csc_matrix((np.ones(dim_a * dim_b), (iHc, jHc)),
                       shape=(dim_b, dim_a * dim_b))

    return Hr, Hc, c


def ot_next_gamma(phi, delta, HtH, Hty, c, active_index, current_gamma):
    r""" This function computes the next value of gamma if a variable
    will be added in next iteration of the regularization path

    We look for the largest value of gamma such that
    the gradient of an inactive variable vanishes
    .. math::
        \max_{i \in \bar{A}} \frac{h_i^T(H_A \phi - y)}{h_i^T H_A \delta - c_i}
    where :
    - A is the current active set
    - h_i is the ith column of auxiliary matrix H
    - H_A is the sub-matrix constructed by the columns of H
        whose indices belong to the active set A
    - c_i is the ith element of cost vector c
    - y is the concatenation of source and target distribution
    - :math:`\phi` is the intercept of the solutions in current iteration
    - :math:`\delta` is the slope of the solutions in current iteration
    Parameters
    ----------
    phi : np.ndarray (|A|, )
        Intercept of the solutions in current iteration (t is piecewise linear)
    delta : np.ndarray (|A|, )
        Slope of the solutions in current iteration (t is piecewise linear)
    HtH : np.ndarray (dim_a * dim_b, dim_a * dim_b)
        Matrix product of H^T H
    Hty : np.ndarray (dim_a + dim_b, )
        Matrix product of H^T y
    c: np.ndarray (dim_a * dim_b, )
        Flattened array of cost matrix C
    active_index : list
        Indices of active variables
    current_gamma : float
        Value of regularization coefficient at the start of current iteration
    Returns
    -------
    next_gamma : float
        Value of gamma if a variable is added to active set in next iteration
    next_active_index : int
        Index of variable to be activated
    References
    ----------
    [Chapel et al., 2021]:
        Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression.
    """
    M = (HtH[:, active_index].dot(phi) - Hty) / \
        (HtH[:, active_index].dot(delta) - c + 1e-16)
    M[active_index] = 0
    M[M > (current_gamma - 1e-10 * current_gamma)] = 0
    return np.max(M), np.argmax(M)


def semi_relaxed_next_gamma(phi, delta, phi_u, delta_u, HrHr, Hc, Hra,
                            c, active_index, current_gamma):
    r""" This function computes the next value of gamma when a variable is
    active in the regularization path of semi-relaxed UOT.

    By taking the Lagrangian form of the problem, we obtain a similar update
    as the two-sided relaxed UOT
    .. math::
        \max_{i \in \bar{A}} \frac{h_{r i}^T(H_{r A} \phi - a) + h_{c i}^T
            \phi_u}{h_{r i}^T H_{r A} \delta + h_{c i} \delta_u - c_i}
    where :
    - A is the current active set
    - h_{r i} is the ith column of the matrix H_r
    - h_{c i} is the ith column of the matrix H_c
    - H_{r A} is the sub-matrix constructed by the columns of H_r
        whose indices belong to the active set A
    - c_i is the ith element of cost vector c
    - y is the concatenation of source and target distribution
    - :math:`\phi` is the intercept of the solutions in current iteration
    - :math:`\delta` is the slope of the solutions in current iteration
    - :math:`\phi_u` is the intercept of Lagrange parameter in current
        iteration
    - :math:`\delta_u` is the slope of Lagrange parameter in current iteration
    Parameters
    ----------
    phi : np.ndarray (|A|, )
        Intercept of the solutions in current iteration (t is piecewise linear)
    delta : np.ndarray (|A|, )
        Slope of the solutions in current iteration (t is piecewise linear)
    phi_u : np.ndarray (dim_b, )
        Intercept of the Lagrange parameter in current iteration (also linear)
    delta_u : np.ndarray (dim_b, )
        Slope of the Lagrange parameter in current iteration (also linear)
    HrHr : np.ndarray (dim_a * dim_b, dim_a * dim_b)
        Matrix product of H_r^T H_r
    Hc : np.ndarray (dim_b, dim_a * dim_b)
        Matrix that computes the sum along the columns of transport plan T
    Hra : np.ndarray (dim_a * dim_b, )
        Matrix product of H_r^T a
    c: np.ndarray (dim_a * dim_b, )
        Flattened array of cost matrix C
    active_index : list
        Indices of active variables
    current_gamma : float
        Value of regularization coefficient at the start of current iteration
    Returns
    -------
    next_gamma : float
        Value of gamma if a variable is added to active set in next iteration
    next_active_index : int
        Index of variable to be activated
    References
    ----------
    [Chapel et al., 2021]:
        Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression.
    """

    M = (HrHr[:, active_index].dot(phi) - Hra + Hc.T.dot(phi_u)) / \
        (HrHr[:, active_index].dot(delta) - c + Hc.T.dot(delta_u) + 1e-16)
    M[active_index] = 0
    M[M > (current_gamma - 1e-10 * current_gamma)] = 0
    return np.max(M), np.argmax(M)


def compute_next_removal(phi, delta, current_gamma):
    r""" This function computes the next value of gamma if a variable
    is removed in next iteration of regularization path

    We look for the largest value of gamma such that
    an element of current solution vanishes
    .. math::
        \max_{j \in A} \frac{\phi_j}{\delta_j}
    where :
    - A is the current active set
    - phi_j is the jth element of the intercept of current solution
    - delta_j is the jth elemnt of the slope of current solution
    Parameters
    ----------
    phi : np.ndarray (|A|, )
        Intercept of the solutions in current iteration (t is piecewise linear)
    delta : np.ndarray (|A|, )
        Slope of the solutions in current iteration (t is piecewise linear)
    current_gamma : float
        Value of regularization coefficient at the start of current iteration
    Returns
    -------
    next_removal_gamma : float
        Value of gamma if a variable is removed in next iteration
    next_removal_index : int
        Index of the variable to remove in next iteration
    References
    ----------
    [Chapel et al., 2021]:
        Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression.
    """
    r_candidate = phi / (delta - 1e-16)
    r_candidate[r_candidate >= (1 - 1e-8) * current_gamma] = 0
    return np.max(r_candidate), np.argmax(r_candidate)


def complement_schur(M_current, b, d, id_pop):
    r""" This function computes the inverse of matrix in regularization path
    using Schur complement

    Two cases may arise: Firstly one variable is added to the active set
    .. math::
        M_{k+1}^{-1} =
        \begin{bmatrix}
            M_{k}^{-1} + s^{-1} M_{k}^{-1} b b^T M_{k}^{-1} & -s^{-1} \\
            - s^{-1} b^T M_{k}^{-1} & s^{-1}
        \end{bmatrix}
    where :
    - :math:`M_k^{-1}` is the inverse of matrix in previous iteration and
        :math:`M_k` is the upper left block matrix in Schur formulation
    - b is the upper right block matrix in Schur formulation. In our case,
        b is reduced to a column vector and b^T is the lower left block matrix
    - s is the Schur complement, given by
        :math:`s = d - b^T M_{k}^{-1} b` in our case

    Secondly, one variable is removed from the active set
    .. math::
        M_{k+1}^{-1} = M^{-1}_{A_k \backslash q} -
                       \frac{r_{-q,q} r^{T}_{-q,q}}{r_{q,q}}
    where :
    - q is the index of column and row to delete
    - :math:`M^{-1}_{A_k \backslash q}` is the previous inverse matrix
        without qth column and qth row
    - r_{-q,q} is the qth column of :math:`M^{-1}_{k}` without the qth element
    - r_{q, q} is the element of qth column and qth row in :math:`M^{-1}_{k}`
    Parameters
    ----------
    M_current : np.ndarray (|A|-1, |A|-1)
        Inverse matrix in previous iteration
    b : np.ndarray (|A|-1, )
        Upper right matrix in Schur complement, a column vector in our case
    d : float
        Lower right matrix in Schur complement, a scalar in our case
    id_pop
        Index of the variable to be removed,  equal to -1
        if none of the variables is deleted in current iteration
    Returns
    -------
    M : np.ndarray (|A|, |A|)
        Inverse matrix needed in current iteration
    References
    ----------
    [Chapel et al., 2021]:
        Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression.
    """
    if b is None:
        b = M_current[id_pop, :]
        b = np.delete(b, id_pop)
        M_del = np.delete(M_current, id_pop, 0)
        a = M_del[:, id_pop]
        M_del = np.delete(M_del, id_pop, 1)
        M = M_del - np.outer(a, b) / M_current[id_pop, id_pop]
    else:
        n = b.shape[0] + 1
        if np.shape(b)[0] == 0:
            M = np.array([[0.5]])
        else:
            X = M_current.dot(b)
            s = d - b.T.dot(X)
            M = np.zeros((n, n))
            M[:-1, :-1] = M_current + X.dot(X.T) / s
            X_ravel = X.ravel()
            M[-1, :-1] = -X_ravel / s
            M[:-1, -1] = -X_ravel / s
            M[-1, -1] = 1 / s
    return M


def construct_augmented_H(active_index, m, Hc, HrHr):
    r""" This function construct an augmented matrix for the first iteration of
    semi-relaxed regularization path

    .. math::
        Augmented_H =
        \begin{bmatrix}
            0 & H_{c A} \\
            H_{c A}^T & H_{r A}^T H_{r A}
        \end{bmatrix}
    where :
    - H_{r A} is the sub-matrix constructed by the columns of H_r
        whose indices belong to the active set A
    - H_{c A} is the sub-matrix constructed by the columns of H_c
        whose indices belong to the active set A
    Parameters
    ----------
    active_index : list
        Indices of active variables
    m : int
        Length of the target distribution
    Hc : np.ndarray (dim_b, dim_a * dim_b)
        Matrix that computes the sum along the columns of transport plan T
    HrHr : np.ndarray (dim_a * dim_b, dim_a * dim_b)
        Matrix product of H_r^T H_r
    Returns
    -------
    H_augmented : np.ndarray (dim_b + |A|, dim_b + |A|)
        Augmented matrix for the first iteration of the semi-relaxed
        regularization path
    """
    Hc_sub = Hc[:, active_index].toarray()
    HrHr_sub = HrHr[:, active_index]
    HrHr_sub = HrHr_sub[active_index, :].toarray()
    H_augmented = np.block([[np.zeros((m, m)), Hc_sub], [Hc_sub.T, HrHr_sub]])
    return H_augmented


def fully_relaxed_path(a: np.array, b: np.array, C: np.array, reg=1e-4,
                       itmax=50000):
    r"""This function gives the regularization path of l2-penalized UOT problem

    The problem to optimize is the Lasso reformulation of the l2-penalized UOT:
    .. math::
        \min_t \gamma c^T t + 0.5 * \|H t - y\|_2^2
        s.t.
            t \geq 0
    where :
    - c is a (dim_a * dim_b, ) metric cost vector (flattened version of C)
    - :math:`\gamma = 1/\lambda` is the l2-regularization coefficient
    - y is the concatenation of vectors a and b, defined as y^T = [a^T b^T]
    - H is a (dim_a + dim_b, dim_a * dim_b) metric matrix,
        see [Chapel et al., 2021] for the design of H. The matrix product Ht
        computes both the source marginal and the target marginal.
    - t is a (dim_a * dim_b, ) metric vector (flattened version of T)
    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Histogram of dimension dim_a
    b : np.ndarray (dim_b,)
        Histogram of dimension dim_b
    C : np.ndarray, shape (dim_a, dim_b)
        Cost matrix
    reg: float
        l2-regularization coefficient
    itmax: int
        Maximum number of iteration
    Returns
    -------
    t : np.ndarray (dim_a*dim_b, )
        Flattened vector of optimal transport matrix
    t_list : list
        List of solutions in regularization path
    gamma_list : list
        List of regularization coefficient in regularization path
    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> n = 3
    >>> xs = np.array([1., 2., 3.]).reshape((n, 1))
    >>> xt = np.array([5., 6., 7.]).reshape((n, 1))
    >>> C = ot.dist(xs, xt)
    >>> C /= C.max()
    >>> a = np.array([0.2, 0.5, 0.3])
    >>> b = np.array([0.2, 0.5, 0.3])
    >>> t, _, _ = ot.regpath.fully_relaxed_path(a, b, C, 1e-4)
    >>> t
    array([1.99958333e-01, 0.00000000e+00, 0.00000000e+00, 3.88888889e-05,
           4.99938889e-01, 0.00000000e+00, 0.00000000e+00, 3.88888889e-05,
           2.99958333e-01])

    References
    ----------
    [Chapel et al., 2021]:
        Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression.
    """

    n = np.shape(a)[0]
    m = np.shape(b)[0]
    H, y, c = recast_ot_as_lasso(a, b, C)
    HtH = H.T.dot(H)
    Hty = H.T.dot(y)
    n_iter = 1

    # initialization
    M0 = Hty / c
    gamma_list = [np.max(M0)]
    active_index = [np.argmax(M0)]
    t_list = [np.zeros((n * m,))]
    H_inv = np.array([[]])
    add_col = np.array([])
    id_pop = -1

    while n_iter < itmax and gamma_list[-1] > reg:
        H_inv = complement_schur(H_inv, add_col, 2., id_pop)
        current_gamma = gamma_list[-1]

        # compute the intercept and slope of solutions in current iteration
        # t = phi - gamma * delta
        phi = H_inv.dot(Hty[active_index])
        delta = H_inv.dot(c[active_index])
        gamma, ik = ot_next_gamma(phi, delta, HtH, Hty, c, active_index,
                                  current_gamma)

        # compute the next lambda when removing a point from the active set
        alt_gamma, id_pop = compute_next_removal(phi, delta, current_gamma)

        # if the positivity constraint is violated, we remove id_pop
        # from active set, otherwise we add ik to active set
        if alt_gamma > gamma:
            gamma = alt_gamma
        else:
            id_pop = -1

        # compute the solution of current segment
        tA = phi - gamma * delta
        sol = np.zeros((n * m, ))
        sol[active_index] = tA

        if id_pop != -1:
            active_index.pop(id_pop)
            add_col = None
        else:
            active_index.append(ik)
            add_col = HtH[active_index[:-1], ik].toarray()

        gamma_list.append(gamma)
        t_list.append(sol)
        n_iter += 1

    if itmax <= n_iter:
        print('maximum iteration has been reached !')

    # correct the last solution and gamma
    if len(t_list) > 1:
        t_final = (t_list[-2] + (t_list[-1] - t_list[-2]) *
                   (reg - gamma_list[-2]) / (gamma_list[-1] - gamma_list[-2]))
        t_list[-1] = t_final
        gamma_list[-1] = reg
    else:
        gamma_list[-1] = reg
        print('Regularization path does not exist !')

    return t_list[-1], t_list, gamma_list


def semi_relaxed_path(a: np.array, b: np.array, C: np.array, reg=1e-4,
                      itmax=50000):
    r"""This function gives the regularization path of semi-relaxed
    l2-UOT problem

    The problem to optimize is the Lasso reformulation of the l2-penalized UOT:
    .. math::
        \min_t \gamma c^T t + 0.5 * \|H_r t - a\|_2^2
        s.t.
            H_c t = b
            t \geq 0
    where :
    - c is a (dim_a * dim_b, ) metric cost vector (flattened version of C)
    - :math:`\gamma = 1/\lambda` is the l2-regularization coefficient
    - H_r is  a (dim_a, dim_a * dim_b) metric matrix,
        which computes the sum along the rows of transport plan T
    - H_c is a (dim_b, dim_a * dim_b) metric matrix,
        which computes the sum along the columns of transport plan T
    - t is a (dim_a * dim_b, ) metric vector (flattened version of T)
    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Histogram of dimension dim_a
    b : np.ndarray (dim_b,)
        Histogram of dimension dim_b
    C : np.ndarray, shape (dim_a, dim_b)
        Cost matrix
    reg: float (optional)
        l2-regularization coefficient
    itmax: int (optional)
        Maximum number of iteration
    Returns
    -------
    t : np.ndarray (dim_a*dim_b, )
        Flattened vector of optimal transport matrix
    t_list : list
        List of solutions in regularization path
    gamma_list : list
        List of regularization coefficient in regularization path
    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> n = 3
    >>> xs = np.array([1., 2., 3.]).reshape((n, 1))
    >>> xt = np.array([5., 6., 7.]).reshape((n, 1))
    >>> C = ot.dist(xs, xt)
    >>> C /= C.max()
    >>> a = np.array([0.2, 0.5, 0.3])
    >>> b = np.array([0.2, 0.5, 0.3])
    >>> t, _, _ = ot.regpath.semi_relaxed_path(a, b, C, 1e-4)
    >>> t
    array([1.99980556e-01, 0.00000000e+00, 0.00000000e+00, 1.94444444e-05,
           4.99980556e-01, 0.00000000e+00, 0.00000000e+00, 1.94444444e-05,
           3.00000000e-01])

    References
    ----------
    [Chapel et al., 2021]:
        Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression.
    """

    n = np.shape(a)[0]
    m = np.shape(b)[0]
    Hr, Hc, c = recast_semi_relaxed_as_lasso(a, b, C)
    Hra = Hr.T.dot(a)
    HrHr = Hr.T.dot(Hr)
    n_iter = 1
    active_index = []

    # initialization
    for j in range(np.shape(C)[1]):
        i = np.argmin(C[:, j])
        active_index.append(i * m + j)
    gamma_list = []
    t_list = []
    current_gamma = np.Inf
    augmented_H0 = construct_augmented_H(active_index, m, Hc, HrHr)
    add_col = np.array([])
    id_pop = -1

    while n_iter < itmax and current_gamma > reg:
        if n_iter == 1:
            H_inv = np.linalg.inv(augmented_H0)
        else:
            H_inv = complement_schur(H_inv, add_col, 1., id_pop + m)
        # compute the intercept and slope of solutions in current iteration
        augmented_phi = H_inv.dot(np.concatenate((b, Hra[active_index])))
        augmented_delta = H_inv[:, m:].dot(c[active_index])
        phi = augmented_phi[m:]
        delta = augmented_delta[m:]
        phi_u = augmented_phi[0:m]
        delta_u = augmented_delta[0:m]
        gamma, ik = semi_relaxed_next_gamma(phi, delta, phi_u, delta_u,
                                            HrHr, Hc, Hra, c, active_index,
                                            current_gamma)

        # compute the next lambda when removing a point from the active set
        alt_gamma, id_pop = compute_next_removal(phi, delta, current_gamma)

        # if the positivity constraint is violated, we remove id_pop
        # from active set, otherwise we add ik to active set
        if alt_gamma > gamma:
            gamma = alt_gamma
        else:
            id_pop = -1

        # compute the solution of current segment
        tA = phi - gamma * delta
        sol = np.zeros((n * m, ))
        sol[active_index] = tA
        if id_pop != -1:
            active_index.pop(id_pop)
            add_col = None
        else:
            active_index.append(ik)
            add_col = np.concatenate((Hc.toarray()[:, ik],
                                      HrHr.toarray()[active_index[:-1], ik]))
            add_col = add_col[:, np.newaxis]

        gamma_list.append(gamma)
        t_list.append(sol)
        current_gamma = gamma
        n_iter += 1

    if itmax <= n_iter:
        print('maximum iteration has been reached !')

    # correct the last solution and gamma
    if len(t_list) > 1:
        t_final = (t_list[-2] + (t_list[-1] - t_list[-2]) *
                   (reg - gamma_list[-2]) / (gamma_list[-1] - gamma_list[-2]))
        t_list[-1] = t_final
        gamma_list[-1] = reg
    else:
        gamma_list[-1] = reg
        print('Regularization path does not exist !')

    return t_list[-1], t_list, gamma_list


def regularization_path(a: np.array, b: np.array, C: np.array, reg=1e-4,
                        semi_relaxed=False, itmax=50000):
    r"""This function combines both the semi-relaxed and the fully-relaxed
    regularization paths of l2-UOT problem

    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Histogram of dimension dim_a
    b : np.ndarray (dim_b,)
        Histogram of dimension dim_b
    C : np.ndarray, shape (dim_a, dim_b)
        Cost matrix
    reg: float (optional)
        l2-regularization coefficient
    semi_relaxed : bool (optional)
        Give the semi-relaxed path if true
    itmax: int (optional)
        Maximum number of iteration
    Returns
    -------
    t : np.ndarray (dim_a*dim_b, )
        Flattened vector of optimal transport matrix
    t_list : list
        List of solutions in regularization path
    gamma_list : list
        List of regularization coefficient in regularization path
    References
    ----------
    [Chapel et al., 2021]:
        Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression.
    """
    if semi_relaxed:
        t, t_list, gamma_list = semi_relaxed_path(a, b, C, reg=reg,
                                                  itmax=itmax)
    else:
        t, t_list, gamma_list = fully_relaxed_path(a, b, C, reg=reg,
                                                   itmax=itmax)
    return t, t_list, gamma_list


def compute_transport_plan(gamma, gamma_list, Pi_list):
    r""" Given the regularization path, this function computes the transport
    plan for any value of gamma by the piecewise linearity of the path

    .. math::
        t(\gamma) = \phi(\gamma) - \gamma \delta(\gamma)
    where :
    - :math:`\gamma` is the regularization coefficient
    - :math:`\phi(\gamma)` is the corresponding intercept
    - :math:`\delta(\gamma)` is the corresponding slope
    - t is a (dim_a * dim_b, ) vector (flattened version of transport matrix)
    Parameters
    ----------
    gamma : float
        Regularization coefficient
    gamma_list : list
        List of regularization coefficients in regularization path
    Pi_list : list
        List of solutions in regularization path
    Returns
    -------
    t : np.ndarray (dim_a*dim_b, )
        Transport vector corresponding to the given value of gamma
    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> n = 3
    >>> xs = np.array([1., 2., 3.]).reshape((n, 1))
    >>> xt = np.array([5., 6., 7.]).reshape((n, 1))
    >>> C = ot.dist(xs, xt)
    >>> C /= C.max()
    >>> a = np.array([0.2, 0.5, 0.3])
    >>> b = np.array([0.2, 0.5, 0.3])
    >>> t, pi_list, g_list = ot.regpath.regularization_path(a, b, C, reg=1e-4)
    >>> gamma = 1
    >>> t2 = ot.regpath.compute_transport_plan(gamma, g_list, pi_list)
    >>> t2
    array([0.        , 0.        , 0.        , 0.19722222, 0.05555556,
           0.        , 0.        , 0.24722222, 0.        ])

    References
    ----------
    [Chapel et al., 2021]:
        Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression.
    """

    if gamma >= gamma_list[0]:
        Pi = Pi_list[0]
    elif gamma <= gamma_list[-1]:
        Pi = Pi_list[-1]
    else:
        idx = np.where(gamma <= np.array(gamma_list))[0][-1]
        gamma_k0 = gamma_list[idx]
        gamma_k1 = gamma_list[idx + 1]
        pi_k0 = Pi_list[idx]
        pi_k1 = Pi_list[idx + 1]
        Pi = pi_k0 + (pi_k1 - pi_k0) * (gamma - gamma_k0) \
            / (gamma_k1 - gamma_k0)
    return Pi
