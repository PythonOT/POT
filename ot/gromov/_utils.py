# -*- coding: utf-8 -*-
"""
Gromov-Wasserstein and Fused-Gromov-Wasserstein utils.
"""

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@unice.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License


from ..utils import list_to_array, euclidean_distances
from ..backend import get_backend
from ..lp import emd

try:
    from networkx.algorithms.community import asyn_fluidc
    from networkx import from_numpy_array
    networkx_import = True
except ImportError:
    networkx_import = False

try:
    from sklearn.cluster import SpectralClustering, KMeans
    sklearn_import = True
except ImportError:
    sklearn_import = False

import numpy as np
import warnings


def _transform_matrix(C1, C2, loss_fun='square_loss', nx=None):
    r"""Return transformed structure matrices for Gromov-Wasserstein fast computation

    Returns the matrices involved in the computation of :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2})`
    with the selected loss function as the loss function of Gromov-Wasserstein discrepancy.

    The matrices are computed as described in Proposition 1 in :ref:`[12] <references-init-matrix>`

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space

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
    loss_fun : str, optional
        Name of loss function to use: either 'square_loss' or 'kl_loss' (default='square_loss')
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    fC1 : array-like, shape (ns, ns)
        :math:`\mathbf{f1}(\mathbf{C1})` matrix in Eq. (6)
    fC2 : array-like, shape (nt, nt)
        :math:`\mathbf{f2}(\mathbf{C2})` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)


    .. _references-transform_matrix:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if nx is None:
        C1, C2 = list_to_array(C1, C2)
        nx = get_backend(C1, C2)

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
            return a * nx.log(a + 1e-16) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return nx.log(b + 1e-16)
    else:
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

    fC1 = f1(C1)
    fC2 = f2(C2)
    hC1 = h1(C1)
    hC2 = h2(C2)

    return fC1, fC2, hC1, hC2


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

    fC1, fC2, hC1, hC2 = _transform_matrix(C1, C2, loss_fun, nx)
    constC1 = nx.dot(
        nx.dot(fC1, nx.reshape(p, (-1, 1))),
        nx.ones((1, len(q)), type_as=q)
    )
    constC2 = nx.dot(
        nx.ones((len(p), 1), type_as=p),
        nx.dot(nx.reshape(q, (1, -1)), fC2.T)
    )
    constC = constC1 + constC2

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
        Probability distribution in the source space
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

    fC1, fC2, hC1, hC2 = _transform_matrix(C1, C2, loss_fun, nx)

    constC = nx.dot(nx.dot(fC1, nx.reshape(p, (-1, 1))),
                    nx.ones((1, C2.shape[0]), type_as=p))

    fC2t = fC2.T
    return constC, hC1, hC2, fC2t


def semirelaxed_init_plan(C1, C2, p, M=None, alpha=1., method='product',
                          use_target=True, random_state=0, nx=None):
    r"""
    Heuristics to initialize the semi-relaxed (F)GW transport plan
    :math:`\mathbf{T} \in \mathcal{U}_{nt}(\mathbf{p})`, between a graph
    :math:`(\mathbf{C1}, \mathbf{p})` and a structure matrix :math:`\mathbf{C2}`,
    where :math:`\mathcal{U}_{nt}(\mathbf{p}) = \{\mathbf{T} \in \mathbb{R}_{+}^{ns * nt}, \mathbf{T} \mathbf{1}_{nt} = \mathbf{p} \}`.
    Available methods are:
        - "product" or "random_product": :math:`\mathbf{T} = \mathbf{pq}^{T}`
          with :math:`\mathbf{q}` uniform or randomly samples in the nt probability simplex.

        - "random": random sampling in :math:`\mathcal{U}_{nt}(\mathbf{p})`.

        - "fluid": Fluid algorithm from networkx for graph partitioning.

        - "spectral", "kmeans" : Spectral or Kmeans clustering from sklearn.

        - "fluid_soft", "spectral_soft", "kmeans_soft": :math:`\mathbf{T}_0` given
          by corresponding clustering with target marginal :math:`\mathbf{q}_0`, further
          centered as :math:`\mathbf{T} = (\mathbf{T}_0 + \mathbf{pq}_0^T) / 2` .

    If a metric cost matrix between features across domains :math:`\mathbf{M}`
    is a provided, it will be used as cost matrix in a semi-relaxed Wasserstein
    problem providing :math:`\mathbf{T}_M \in \mathcal{U}_{nt}(\mathbf{p})`. Then
    the outputed transport plan is :math:`\alpha \mathbf{T}  + (1 - \alpha ) \mathbf{T}_{M}`.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space.
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space.
    p : array-like, shape (ns,), optional.
        Probability distribution in the source space. If let to None, uniform
        weights are assumed on C1.
    M : array-like, shape (ns, nt), optional.
        Metric cost matrix between features across domains.
    alpha : float, optional
        Trade-off parameter (0 <= alpha <= 1)
    method : str, optional
        Method to initialize the transport plan. The default is 'product'.
    use_target : bool, optional.
        Whether or not to use the target structure/features to further align
        transport plan provided by the `method`.
    random_state: int, optional
        Random seed used for stochastic methods.
    nx : backend, optional
        POT backend.

    Returns
    -------
    T : array-like, shape (ns, ns)
        Admissible transport plan for the sr(F)GW problems.

    References
    ----------
    .. [48]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.

    """
    list_partitioning_methods = [
        'fluid', 'spectral', 'kmeans', 'fluid_soft', 'spectral_soft',
        'kmeans_soft']

    if method not in list_partitioning_methods + ['product', 'random_product', 'random']:
        raise ValueError(f'Unsupported initialization method = {method}.')

    if (method in ['kmeans', 'kmeans_soft']) and (not sklearn_import):
        raise ValueError(f'Scikit-learn must be installed to use method = {method}')

    if (method in ['fluid', 'fluid_soft']) and (not networkx_import):
        raise ValueError(f'Networkx must be installed to use method = {method}')

    if nx is None:
        nx = get_backend(C1, C2, p, M)

    n = C1.shape[0]
    m = C2.shape[0]
    min_size = min(n, m)

    if method in list_partitioning_methods:
        if n > m:  # partition C1 to deduce map from C1 to C2
            C_to_partition = nx.to_numpy(C1)
        elif m > n:  # partition C2 to deduce map from C1 to C2
            C_to_partition = nx.to_numpy(C2)
        else:  # equal size -> simple Wasserstein alignment
            C_to_partition = None
            warnings.warn(
                "Both structures have the same size so no partitioning is"
                "performed to initialize the transport plan even though"
                f"initialization method is {method}",
                stacklevel=2
            )

        def get_transport_from_partition(part):
            if n > m:  # partition C1 to deduce map from C1 to C2
                T_ = nx.eye(m, type_as=C1)[part]
                T_ = p[:, None] * T_
                q = nx.sum(T_, 0)

                factored_C1 = nx.dot(nx.dot(T_.T, C1), T_) / nx.outer(q, q)

                # alignment of both structure seen as feature matrices
                if use_target:
                    M_structure = euclidean_distances(factored_C1, C2)
                    T_emd = emd(q, q, M_structure)
                    inv_q = 1. / q

                    T = nx.dot(T_, inv_q[:, None] * T_emd)
                else:
                    T = T_

            elif m > n:
                T_ = nx.eye(n, type_as=C1)[part] / m  # assume uniform masses on C2
                q = nx.sum(T_, 0)

                factored_C2 = nx.dot(nx.dot(T_.T, C2), T_) / nx.outer(q, q)

                # alignment of both structure seen as feature matrices
                M_structure = euclidean_distances(factored_C2, C1)
                T_emd = emd(q, p, M_structure)
                inv_q = 1. / q

                T = nx.dot(T_, inv_q[:, None] * T_emd).T
                q = nx.sum(T, 0)  # uniform one
            else:
                # alignment of both structure seen as feature matrices
                M_structure = euclidean_distances(C1, C2)
                q = p
                T = emd(p, q, M_structure)

            return T, q

    # Handle initialization via structure information

    if method == 'product':
        q = nx.ones(m, type_as=C1) / m
        T = nx.outer(p, q)

    elif method == 'random_product':
        np.random.seed(random_state)
        q = np.random.uniform(0, m, size=(m,))
        q = q / q.sum()
        q = nx.from_numpy(q, type_as=p)
        T = nx.outer(p, q)

    elif method == 'random':
        np.random.seed(random_state)
        U = np.random.uniform(0, n * m, size=(n, m))
        U = (p / U.sum(1))[:, None] * U
        T = nx.from_numpy(U, type_as=C1)

    elif method in ['fluid', 'fluid_soft']:
        # compute fluid partitioning on the biggest graph
        if C_to_partition is None:
            T, q = get_transport_from_partition(None)
        else:
            graph = from_numpy_array(C_to_partition)
            part_sets = asyn_fluidc(graph, min_size, seed=random_state)
            part = np.zeros(C_to_partition.shape[0], dtype=int)
            for iset_, set_ in enumerate(part_sets):
                set_ = list(set_)
                part[set_] = iset_
            part = nx.from_numpy(part)

            T, q = get_transport_from_partition(part)

        if 'soft' in method:
            T = (T + nx.outer(p, q)) / 2.

    elif method in ['spectral', 'spectral_soft']:
        # compute spectral partitioning on the biggest graph
        if C_to_partition is None:
            T, q = get_transport_from_partition(None)
        else:
            sc = SpectralClustering(n_clusters=min_size,
                                    random_state=random_state,
                                    affinity='precomputed').fit(C_to_partition)
            part = sc.labels_
            T, q = get_transport_from_partition(part)

        if 'soft' in method:
            T = (T + nx.outer(p, q)) / 2.

    elif method in ['kmeans', 'kmeans_soft']:
        # compute spectral partitioning on the biggest graph
        if C_to_partition is None:
            T, q = get_transport_from_partition(None)
        else:
            km = KMeans(n_clusters=min_size, random_state=random_state,
                        n_init=1).fit(C_to_partition)

            part = km.labels_
            T, q = get_transport_from_partition(part)

        if 'soft' in method:
            T = (T + nx.outer(p, q)) / 2.

    if (M is not None):
        # Add feature information solving a semi-relaxed Wasserstein problem
        # get minimum by rows as binary mask
        TM = nx.ones(1, type_as=p) * (M == nx.reshape(nx.min(M, axis=1), (-1, 1)))
        TM *= nx.reshape((p / nx.sum(TM, axis=1)), (-1, 1))

        T = alpha * T + (1 - alpha) * TM

    return T


def update_barycenter_structure(
        Ts, Cs, lambdas, p=None, loss_fun='square_loss', target=True,
        check_zeros=True, nx=None):
    r"""
    Updates :math:`\mathbf{C}` according to the inner loss L with the `S`
    :math:`\mathbf{T}_s` couplings calculated at each iteration of variants of
    the GW barycenter problem (e.g GW :ref:`[12]`, srGW :ref:`[48]`).
    If `target=True` it solves for:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad
        \sum_s \lambda_s \sum_{i,j,k,l}
        L(\mathbf{C}^{(s)}_{i,k}, \mathbf{C}_{j,l}) \mathbf{T}^{(s)}_{i,j} \mathbf{T}^{(s)}_{k,l}

    Else it solves the symmetric problem:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad
        \sum_s \lambda_s \sum_{i,j,k,l}
        L(\mathbf{C}_{j,l}, \mathbf{C}^{(s)}_{i,k}) \mathbf{T}^{(s)}_{i,j} \mathbf{T}^{(s)}_{k,l}

    Where :

    - :math:`\mathbf{C}^{(s)}`: pairwise matrix in the s^{th} source space .
    - :math:`\mathbf{C}`: pairwise matrix in the target space.
    - :math:`L`: inner divergence for the GW loss

    Parameters
    ----------
    Ts : list of S array-like of shape (ns, N) if `target=True` else (N, ns).
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape(ns, ns)
        Metric cost matrices.
    lambdas : list of float,
        List of the `S` spaces' weights.
    p : array-like, shape (N,) or (S,N)
        Masses or list of masses in the targeted barycenter.
    loss_fun : str, optional. Default is 'square_loss'
        Name of loss function to use in ['square_loss', 'kl_loss'].
    target: bool, optional. Default is True.
        Whether the barycenter is positioned as target (True) or source (False).
    check_zeros: bool, optional. Default is True.
        Whether to check if marginals on the barycenter contains zeros or not.
        Can be set to False to gain time if marginals are known to be positive.
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

    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.

    """

    if nx is None:
        arr = [*Ts, *Cs, p]
        nx = get_backend(*arr)

    S = len(Ts)

    if p is None:
        p = nx.concatenate(
            [nx.sum(Ts[s], int(not target))[None, :] for s in range(S)],
            axis=0)

    # compute coefficients for the barycenter coming from marginals

    if len(p.shape) == 1:  # shared target masses potentially with zeros
        if check_zeros:
            inv_p = nx.nan_to_num(1. / p, nan=1., posinf=1., neginf=1.)
        else:
            inv_p = 1. / p

        prod = nx.outer(inv_p, inv_p)

    else:
        quotient = sum([lambdas[s] * nx.outer(p[s], p[s]) for s in range(S)])
        if check_zeros:
            prod = nx.nan_to_num(1. / quotient, nan=1., posinf=1., neginf=1.)
        else:
            prod = 1. / quotient

    # compute coefficients for the barycenter coming from Ts and Cs

    if loss_fun == 'square_loss':
        if target:
            list_structures = [lambdas[s] * nx.dot(
                nx.dot(Ts[s].T, Cs[s]), Ts[s]) for s in range(S)]
        else:
            list_structures = [lambdas[s] * nx.dot(
                nx.dot(Ts[s], Cs[s]), Ts[s].T) for s in range(S)]

        return sum(list_structures) * prod

    elif loss_fun == 'kl_loss':
        if target:
            list_structures = [lambdas[s] * nx.dot(
                nx.dot(Ts[s].T, Cs[s]), Ts[s])
                for s in range(S)]

            return sum(list_structures) * prod
        else:
            list_structures = [lambdas[s] * nx.dot(
                nx.dot(Ts[s], nx.log(nx.maximum(Cs[s], 1e-16))), Ts[s].T)
                for s in range(S)]

            return nx.exp(sum(list_structures) * prod)

    else:
        raise ValueError(f"not supported loss_fun = {loss_fun}")


def update_barycenter_feature(
        Ts, Ys, lambdas, p=None, loss_fun='square_loss', target=True,
        check_zeros=True, nx=None):
    r"""Updates the feature with respect to the `S` :math:`\mathbf{T}_s`
    couplings calculated at each iteration of variants of the FGW
    barycenter problem with inner wasserstein loss `loss_fun`
    (e.g FGW :ref:`[24]`, srFGW :ref:`[48]`).
    If `target=True` the barycenter is considered as the target else as the source.

    Parameters
    ----------
    Ts : list of S array-like of shape (ns, N) if `target=True` else (N, ns).
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Ys : list of S array-like, shape (ns, d)
        Feature matrices.
    lambdas : list of float
        List of the `S` spaces' weights
    p : array-like, shape (N,) or (S,N)
        Masses or list of masses in the targeted barycenter.
    loss_fun : str, optional. Default is 'square_loss'
        Name of loss function to use in ['square_loss'].
    target: bool, optional. Default is True.
        Whether the barycenter is positioned as target (True) or source (False).
    check_zeros: bool, optional. Default is True.
        Whether to check if marginals on the barycenter contains zeros or not.
        Can be set to False to gain time if marginals are known to be positive.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    X : array-like, shape (N, d)

    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.

    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.
    """
    if nx is None:
        arr = [*Ts, *Ys, p]
        nx = get_backend(*arr)

    if loss_fun != 'square_loss':
        raise ValueError(f"not supported loss_fun = {loss_fun}")

    S = len(Ts)

    if target:
        list_features = [lambdas[s] * nx.dot(Ts[s].T, Ys[s]) for s in range(S)]
    else:
        list_features = [lambdas[s] * nx.dot(Ts[s], Ys[s]) for s in range(S)]

    if p is None:
        p = nx.concatenate(
            [nx.sum(Ts[s], int(not target))[None, :] for s in range(S)],
            axis=0)

    if len(p.shape) == 1:  # shared target masses potentially with zeros
        if check_zeros:
            inv_p = nx.nan_to_num(1. / p, nan=1., posinf=1., neginf=1.)
        else:
            inv_p = 1. / p
    else:
        p_sum = sum([lambdas[s] * p[s] for s in range(S)])
        if check_zeros:
            inv_p = nx.nan_to_num(1. / p_sum, nan=1., posinf=1., neginf=1.)
        else:
            inv_p = 1. / p_sum

    return sum(list_features) * inv_p[:, None]
