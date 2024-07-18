# -*- coding: utf-8 -*-
"""
Gromov-Wasserstein and Fused-Gromov-Wasserstein stochastic estimators.
"""

# Author: Rémi Flamary <remi.flamary@unice.fr>
#         Tanguy Kerdoncuff <tanguy.kerdoncuff@laposte.net>
#
# License: MIT License

import numpy as np


from ..bregman import sinkhorn
from ..utils import list_to_array, check_random_state
from ..lp import emd_1d, emd
from ..backend import get_backend


def GW_distance_estimation(
    C1,
    C2,
    p,
    q,
    loss_fun,
    T,
    nb_samples_p=None,
    nb_samples_q=None,
    std=True,
    random_state=None,
):
    r"""
    Returns an approximation of the Gromov-Wasserstein loss between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`
    with a fixed transport plan :math:`\mathbf{T}`. To recover an approximation of the Gromov-Wasserstein distance as defined in [13] compute :math:`d_{GW} = \frac{1}{2} \sqrt{\mathbf{GW}}`.

    The function gives an unbiased approximation of the following equation:

    .. math::

        \mathbf{GW} = \sum_{i,j,k,l} L(\mathbf{C_{1}}_{i,k}, \mathbf{C_{2}}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - `L` : Loss function to account for the misfit between the similarity matrices
    - :math:`\mathbf{T}`: Matrix with marginal :math:`\mathbf{p}` and :math:`\mathbf{q}`

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun :  function: :math:`\mathbb{R} \times \mathbb{R} \mapsto \mathbb{R}`
        Loss function used for the distance, the transport plan does not depend on the loss function
    T : csr or array-like, shape (ns, nt)
        Transport plan matrix, either a sparse csr or a dense matrix
    nb_samples_p : int, optional
        `nb_samples_p` is the number of samples (without replacement) along the first dimension of :math:`\mathbf{T}`
    nb_samples_q : int, optional
        `nb_samples_q` is the number of samples along the second dimension of :math:`\mathbf{T}`, for each sample along the first
    std : bool, optional
        Standard deviation associated with the prediction of the gromov-wasserstein cost
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

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
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    generator = check_random_state(random_state)

    len_p = p.shape[0]
    len_q = q.shape[0]

    # It is always better to sample from the biggest distribution first.
    if len_p < len_q:
        p, q = q, p
        len_p, len_q = len_q, len_p
        C1, C2 = C2, C1
        T = T.T

    if nb_samples_p is None:
        if nx.issparse(T):
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

    index_i = generator.choice(
        len_p, size=nb_samples_p, p=nx.to_numpy(p), replace=False
    )
    index_j = generator.choice(
        len_p, size=nb_samples_p, p=nx.to_numpy(p), replace=False
    )

    for i in range(nb_samples_p):
        if nx.issparse(T):
            T_indexi = nx.reshape(nx.todense(T[index_i[i], :]), (-1,))
            T_indexj = nx.reshape(nx.todense(T[index_j[i], :]), (-1,))
        else:
            T_indexi = T[index_i[i], :]
            T_indexj = T[index_j[i], :]
        # For each of the row sampled, the column is sampled.
        index_k[i] = generator.choice(
            len_q,
            size=nb_samples_q,
            p=nx.to_numpy(T_indexi / nx.sum(T_indexi)),
            replace=True,
        )
        index_l[i] = generator.choice(
            len_q,
            size=nb_samples_q,
            p=nx.to_numpy(T_indexj / nx.sum(T_indexj)),
            replace=True,
        )

    list_value_sample = nx.stack(
        [
            loss_fun(
                C1[np.ix_(index_i, index_j)], C2[np.ix_(index_k[:, n], index_l[:, n])]
            )
            for n in range(nb_samples_q)
        ],
        axis=2,
    )

    if std:
        std_value = nx.sum(nx.std(list_value_sample, axis=2) ** 2) ** 0.5
        return nx.mean(list_value_sample), std_value / (nb_samples_p * nb_samples_p)
    else:
        return nx.mean(list_value_sample)


def pointwise_gromov_wasserstein(
    C1,
    C2,
    p,
    q,
    loss_fun,
    alpha=1,
    max_iter=100,
    threshold_plan=0,
    log=False,
    verbose=False,
    random_state=None,
):
    r"""
    Returns the gromov-wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})` using a stochastic Frank-Wolfe.
    This method has a :math:`\mathcal{O}(\mathrm{max\_iter} \times PN^2)` time complexity with `P` the number of Sinkhorn iterations.

    The function solves the following optimization problem:

    .. math::
        \mathbf{GW} = \mathop{\arg \min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

                \mathbf{T}^T \mathbf{1} &= \mathbf{q}

                \mathbf{T} &\geq 0

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun :  function: :math:`\mathbb{R} \times \mathbb{R} \mapsto \mathbb{R}`
        Loss function used for the distance, the transport plan does not depend on the loss function
    alpha : float
        Step of the Frank-Wolfe algorithm, should be between 0 and 1
    max_iter : int, optional
        Max number of iterations
    threshold_plan : float, optional
        Deleting very small values in the transport plan. If above zero, it violates the marginal constraints.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Gives the distance estimated and the standard deviation
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces

    References
    ----------
    .. [14] Kerdoncuff, Tanguy, Emonet, Rémi, Sebban, Marc
        "Sampled Gromov Wasserstein."
        Machine Learning Journal (MLJ). 2021.

    """
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    len_p = p.shape[0]
    len_q = q.shape[0]

    generator = check_random_state(random_state)

    index = np.zeros(2, dtype=int)

    # Initialize with default marginal
    index[0] = generator.choice(len_p, size=1, p=nx.to_numpy(p))
    index[1] = generator.choice(len_q, size=1, p=nx.to_numpy(q))
    T = nx.tocsr(emd_1d(C1[index[0]], C2[index[1]], a=p, b=q, dense=False))

    best_gw_dist_estimated = np.inf
    for cpt in range(max_iter):
        index[0] = generator.choice(len_p, size=1, p=nx.to_numpy(p))
        T_index0 = nx.reshape(nx.todense(T[index[0], :]), (-1,))
        index[1] = generator.choice(
            len_q, size=1, p=nx.to_numpy(T_index0 / nx.sum(T_index0))
        )

        if alpha == 1:
            T = nx.tocsr(emd_1d(C1[index[0]], C2[index[1]], a=p, b=q, dense=False))
        else:
            new_T = nx.tocsr(emd_1d(C1[index[0]], C2[index[1]], a=p, b=q, dense=False))
            T = (1 - alpha) * T + alpha * new_T
            # To limit the number of non 0, the values below the threshold are set to 0.
            T = nx.eliminate_zeros(T, threshold=threshold_plan)

        if cpt % 10 == 0 or cpt == (max_iter - 1):
            gw_dist_estimated = GW_distance_estimation(
                C1=C1,
                C2=C2,
                loss_fun=loss_fun,
                p=p,
                q=q,
                T=T,
                std=False,
                random_state=generator,
            )

            if gw_dist_estimated < best_gw_dist_estimated:
                best_gw_dist_estimated = gw_dist_estimated
                best_T = nx.copy(T)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        "{:5s}|{:12s}".format("It.", "Best gw estimated")
                        + "\n"
                        + "-" * 19
                    )
                print("{:5d}|{:8e}|".format(cpt, best_gw_dist_estimated))

    if log:
        log = {}
        log["gw_dist_estimated"], log["gw_dist_std"] = GW_distance_estimation(
            C1=C1, C2=C2, loss_fun=loss_fun, p=p, q=q, T=best_T, random_state=generator
        )
        return best_T, log
    return best_T


def sampled_gromov_wasserstein(
    C1,
    C2,
    p,
    q,
    loss_fun,
    nb_samples_grad=100,
    epsilon=1,
    max_iter=500,
    log=False,
    verbose=False,
    random_state=None,
):
    r"""
    Returns the gromov-wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})` using a 1-stochastic Frank-Wolfe.
    This method has a :math:`\mathcal{O}(\mathrm{max\_iter} \times N \log(N))` time complexity by relying on the 1D Optimal Transport solver.

    The function solves the following optimization problem:

    .. math::
        \mathbf{GW} = \mathop{\arg \min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

                \mathbf{T}^T \mathbf{1} &= \mathbf{q}

                \mathbf{T} &\geq 0

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun :  function: :math:`\mathbb{R} \times \mathbb{R} \mapsto \mathbb{R}`
        Loss function used for the distance, the transport plan does not depend on the loss function
    nb_samples_grad : int
        Number of samples to approximate the gradient
    epsilon : float
        Weight of the Kullback-Leibler regularization
    max_iter : int, optional
        Max number of iterations
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Gives the distance estimated and the standard deviation
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces

    References
    ----------
    .. [14] Kerdoncuff, Tanguy, Emonet, Rémi, Sebban, Marc
        "Sampled Gromov Wasserstein."
        Machine Learning Journal (MLJ). 2021.

    """
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    len_p = p.shape[0]
    len_q = q.shape[0]

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
    T = nx.outer(p, q)
    # continue_loop allows to stop the loop if there is several successive small modification of T.
    continue_loop = 0

    # The gradient of GW is more complex if the two matrices are not symmetric.
    C_are_symmetric = nx.allclose(C1, C1.T, rtol=1e-10, atol=1e-10) and nx.allclose(
        C2, C2.T, rtol=1e-10, atol=1e-10
    )

    for cpt in range(max_iter):
        index0 = generator.choice(
            len_p, size=nb_samples_grad_p, p=nx.to_numpy(p), replace=False
        )
        Lik = 0
        for i, index0_i in enumerate(index0):
            index1 = generator.choice(
                len_q,
                size=nb_samples_grad_q,
                p=nx.to_numpy(T[index0_i, :] / nx.sum(T[index0_i, :])),
                replace=False,
            )
            # If the matrices C are not symmetric, the gradient has 2 terms, thus the term is chosen randomly.
            if (not C_are_symmetric) and generator.rand(1) > 0.5:
                Lik += nx.mean(
                    loss_fun(
                        C1[:, [index0[i]] * nb_samples_grad_q][:, None, :],
                        C2[:, index1][None, :, :],
                    ),
                    axis=2,
                )
            else:
                Lik += nx.mean(
                    loss_fun(
                        C1[[index0[i]] * nb_samples_grad_q, :][:, :, None],
                        C2[index1, :][:, None, :],
                    ),
                    axis=0,
                )

        max_Lik = nx.max(Lik)
        if max_Lik == 0:
            continue
        # This division by the max is here to facilitate the choice of epsilon.
        Lik /= max_Lik

        if epsilon > 0:
            # Set to infinity all the numbers below exp(-200) to avoid log of 0.
            log_T = nx.log(nx.clip(T, np.exp(-200), 1))
            log_T = nx.where(log_T == -200, -np.inf, log_T)
            Lik = Lik - epsilon * log_T

            try:
                new_T = sinkhorn(a=p, b=q, M=Lik, reg=epsilon)
            except (RuntimeWarning, UserWarning):
                print("Warning caught in Sinkhorn: Return last stable T")
                break
        else:
            new_T = emd(a=p, b=q, M=Lik)

        change_T = nx.mean((T - new_T) ** 2)
        if change_T <= 10e-20:
            continue_loop += 1
            if continue_loop > 100:  # Number max of low modifications of T
                T = nx.copy(new_T)
                break
        else:
            continue_loop = 0

        if verbose and cpt % 10 == 0:
            if cpt % 200 == 0:
                print(
                    "{:5s}|{:12s}".format("It.", "||T_n - T_{n+1}||") + "\n" + "-" * 19
                )
            print("{:5d}|{:8e}|".format(cpt, change_T))
        T = nx.copy(new_T)

    if log:
        log = {}
        log["gw_dist_estimated"], log["gw_dist_std"] = GW_distance_estimation(
            C1=C1, C2=C2, loss_fun=loss_fun, p=p, q=q, T=T, random_state=generator
        )
        return T, log
    return T
