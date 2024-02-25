"""
Quantized Gromov-Wasserstein solvers.
"""

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

import numpy as np
import warnings

try:
    from networkx.algorithms.community import asyn_fluidc, louvain_communities
    from networkx import from_numpy_array, pagerank
    networkx_import = True
except ImportError:
    networkx_import = False

try:
    from sklearn.cluster import SpectralClustering, KMeans
    sklearn_import = True
except ImportError:
    sklearn_import = False

import random

from ..utils import list_to_array
from ..utils import unif
from ..backend import get_backend
from ..lp import emd_1d
from ._gw import gromov_wasserstein
from ._utils import init_matrix, gwloss


def _get_partition(C, npart, part_method='random', random_state=0, nx=None):
    """
    Partitioning a given structure matrix either 'random', or using one
    of {'louvain', 'fluid'} algorithms from networkx, or {'spectral', 'kmeans'}
    clustering from scikit-learn.

    Parameters
    ----------
    C : array-like, shape (n, n) or (n, d)
        structure matrix if `part_method` in {'random', 'louvain', 'fluid', 'spectral'}
        or feature matrix if `part_method="kmeans"`.
    npart : int,
        number of partitions/clusters smaller than the number of nodes in C.
    part_method : str, optional. Default is 'random'.
        Partitioning algorithm to use among {'random', 'louvain', 'fluid', 'spectral', 'kmeans'}.
        If the louvain algorithm is used, the requested number of partitions is ignored.
    random_state: int, optional
        Random seed for the partitioning algorithm
    nx : backend, optional
        POT backend

    Returns
    -------
    part : array-like, shape (n,)
        Array of partition assignment for each node.

    References
    ----------
    .. [66] Chowdhury, S., Miller, D., & Needham, T. (2021).
        Quantized gromov-wasserstein. ECML PKDD 2021. Springer International Publishing.

    """
    if nx is None:
        nx = get_backend(C)

    n = C.shape[0]
    C0 = C

    if npart >= n:
        warnings.warn(
            "Requested number of partitions higher than the number of nodes"
            "hence we enforce each node to be a partition.",
            stacklevel=2
        )

        part = np.arange(n)

    elif npart == 1:
        part = np.zeros(n)

    elif part_method == 'random':
        # randomly partition the space
        random.seed(random_state)
        part = list_to_array(random.choices(np.arange(npart), k=C.shape[0]))

    elif part_method == 'louvain':
        C = nx.to_numpy(C0)
        graph = from_numpy_array(C)
        part_sets = louvain_communities(graph, seed=random_state)
        part = np.zeros(n)
        for iset_, set_ in enumerate(part_sets):
            set_ = list(set_)
            part[set_] = iset_

    elif part_method == 'fluid':
        C = nx.to_numpy(C0)
        graph = from_numpy_array(C)
        part_sets = asyn_fluidc(graph, npart, seed=random_state)
        part = np.zeros(n)
        for iset_, set_ in enumerate(part_sets):
            set_ = list(set_)
            part[set_] = iset_

    elif part_method == 'spectral':
        C = nx.to_numpy(C0)
        sc = SpectralClustering(n_clusters=npart,
                                random_state=random_state,
                                affinity='precomputed').fit(C)
        part = sc.labels_

    elif part_method == 'kmeans':
        C = nx.to_numpy(C0)
        km = KMeans(n_clusters=npart, random_state=random_state,
                    ).fit(C)
        part = km.labels_

    else:
        raise ValueError(
            f"""
            Unknown `part_method='{part_method}'`. Use one of:
            {'random', 'louvain', 'fluid', 'spectral', 'kmeans'}.
            """)
    return nx.from_numpy(part, type_as=C0)


def _get_representants(C, part, rep_method='random', random_state=0, nx=None):
    """
    Get representants for each partition of a given structure matrix either.

    Parameters
    ----------
    C : array-like, shape (n, n) or (n, d)
        structure matrix if `part_method` in {'random', 'louvain', 'fluid', 'spectral'}
        or feature matrix if `part_method="kmeans"`.
    part : array-like, shape (n,)
        Array of partition assignment for each node.
    rep_method : str, optional. Default is 'pagerank'.
        Selection method for representant in each partition. Can be either 'random'
        i.e random sampling within each partition, or 'pagerank' to select a
        node with maximal pagerank.
    random_state: int, optional
        Random seed for the partitioning algorithm
    nx : backend, optional
        POT backend

    Returns
    -------
    rep_indices : list, shape (npart,)
        indices for representative node of each partition sorted
        according to partition identifiers.

    References
    ----------
    .. [66] Chowdhury, S., Miller, D., & Needham, T. (2021).
        Quantized gromov-wasserstein. ECML PKDD 2021. Springer International Publishing.

    """
    if nx is None:
        nx = get_backend(C, part)

    rep_indices = []
    part_ids = nx.unique(part)
    n_part_ids = part_ids.shape[0]
    if n_part_ids == C.shape[0]:
        rep_indices = nx.arange(n_part_ids)

    elif rep_method == 'random':
        random.seed(random_state)
        for id_, part_id in enumerate(part_ids):
            indices = nx.where(part == part_id)[0]
            rep_indices.append(random.choice(indices))

    elif rep_method == 'pagerank':
        C0, part0 = C, part
        C = nx.to_numpy(C0)
        part = nx.to_numpy(part0)
        part_ids = np.unique(part)

        for id_ in part_ids:
            indices = np.where(part == id_)[0]
            C_id = C[indices, :][:, indices]
            graph = from_numpy_array(C_id)
            pagerank_values = list(pagerank(graph).values())
            rep_idx = np.argmax(pagerank_values)
            rep_indices.append(indices[rep_idx])

    elif rep_method == 'kmeans':
        rep_indices = []
        part_ids = nx.unique(part)
        for id_ in part_ids:
            indices = nx.where(part == id_)[0]
            C_id = C[indices, :]
            centroid = nx.mean(C_id, axis=0)
            dists = nx.sum((C_id - centroid[None, :]) ** 2, axis=1)
            closest_idx = nx.argmin(dists)
            rep_indices.append(indices[closest_idx])

    else:
        raise ValueError(
            f"""
            Unknown `rep_method='{rep_method}'`. Use one of:
            {'random', 'pagerank', 'kmeans'}.
            """)

    return rep_indices


def _formate_partitioned_graph(C, p, part, rep_indices, nx=None):
    """
    Formate a measurable space :math:`(\mathbf{C}, \mathbf{p})` into a partitioned
    measurable space taking into account partitions and representants
    :math:`\mathcal{P} = \left{(\mathbf{P_{i}}, \mathbf{r_{i}})\right}_i`.

    Parameters
    ----------
    C : array-like, shape (n, n)
        Structure matrix.
    p : array-like, shape (n,),
        Node distribution.
    part : array-like, shape (n,)
        Array of partition assignment for each node.
    rep_indices : list of array-like of ints, shape (npart,)
        indices for representative node of each partition sorted according to
        partition identifiers.
    nx : backend, optional
        POT backend

    Returns
    -------
    CR : array-like, shape (npart, npart)
        Structure matrix between partition representants.
    list_R : list of npart arrays,
        List of relations between a representant and nodes in its partition, for each partition.
    list_p : list of npart arrays,
        List of node distributions within each partition.

    References
    ----------
    .. [66] Chowdhury, S., Miller, D., & Needham, T. (2021).
        Quantized gromov-wasserstein. ECML PKDD 2021. Springer International Publishing.

    """
    if nx is None:
        nx = get_backend(C, p, part)

    CR = C[rep_indices, :][:, rep_indices]

    list_R, list_p = [], []

    part_ids = nx.unique(part)

    for id_, part_id in enumerate(part_ids):
        indices = nx.where(part == part_id)[0]
        list_R.append(C[rep_indices[id_], indices])
        list_p.append(p[indices])

    return CR, list_R, list_p


def quantized_gromov_wasserstein(
        C1, C2, npart1, npart2, A1=None, A2=None, p=None, q=None, part_method='fluid',
        rep_method='random', log=False, armijo=False, max_iter=1e4,
        tol_rel=1e-9, tol_abs=1e-9, random_state=0, **kwargs):
    r"""
    Returns the quantized Gromov-Wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})`
    and :math:`(\mathbf{C_2}, \mathbf{q})`, whose samples are assigned to partitions and representants
    :math:`\mathcal{P_1} = \{(\mathbf{P_{1, i}}, \mathbf{r_{1, i}})\}_{i \leq npart1}`
    and :math:`\mathcal{P_2} = \{(\mathbf{P_{2, j}}, \mathbf{r_{2, j}})\}_{j \leq npart2}`.

    The function estimates the following optimization problem:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

             \mathbf{T}_{|\mathbf{P_{1, i}}, \mathbf{P_{2, j}}} &= T^{g}_{ij} \mathbf{T}^{(i,j)}

    using a two-step strategy computing: i) a global alignment :math:`\mathbf{T}^{g}`
    between representants across spaces; ii) local alignments
    :math:`\mathbf{T}^{(i, j)}` between partitions :math:`\mathbf{P_{1, i}}`
    and :math:`\mathbf{P_{2, j}}` seen as 1D measures.

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - :math:`L`: quadratic loss function to account for the misfit between the similarity matrices

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.
    .. note:: All computations in the conjugate gradient solver are done with
        numpy to limit memory overhead.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Structure matrix in the source space
    C2 : array-like, shape (nt, nt)
        Structure matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    A1 : array-like, shape (ns, ns) or (ns, ds), optional. Default is None.
        Structure matrix or feature matrix in the source space to perform the partitioning.
    A2 : array-like, shape (nt, nt) or (nt, dt), optional. Default is None.
        Structure matrix or feature matrix in in the target space to perform the partitioning.
    part_method : str, optional. Default is 'random'.
        Partitioning algorithm to use among {'random', 'louvain', 'fluid', 'spectral', 'kmeans'}.
        If the louvain algorithm is used, the requested number of partitions is ignored.
    rep_method : str, optional. Default is 'random'.
        Selection method for representant in each partition. Can be either 'random'
        i.e random sampling within each partition, 'pagerank' to select a
        node with maximal pagerank or 'kmeans' to select neirest point to centroid.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research. Else closed form is used.
        If there are convergence issues use False.
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan is :math:`\mathbf{pq}^\top`.
        Otherwise G0 is used as initialization and must satisfy marginal constraints.
    max_iter : int, optional
        Max number of iterations
    tol_rel : float, optional
        Stop threshold on relative error (>0)
    tol_abs : float, optional
        Stop threshold on absolute error (>0)
    random_state: int, optional
        Random seed for the partitioning algorithm
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    T_global: array-like, shape (`npart1`, `npart2`)
        Gromov-Wasserstein alignment :math:`\mathbf{T}^{g}` between representants.
    Ts_local: dict of local OT matrices.
        Dictionary with keys :math:`(i, j)` corresponding to 1D OT between
        :math:`\mathbf{P_{1, i}}` and :math:`\mathbf{P_{2, j}}` if :math:`T^{g}_{ij} \neq 0`.
    T: array-like, shape `(ns, nt)`
        Coupling between the two spaces.
    log : dict
        Convergence information for inner problems and qGW loss.

    References
    ----------
    .. [66] Chowdhury, S., Miller, D., & Needham, T. (2021).
        Quantized gromov-wasserstein. ECML PKDD 2021. Springer International Publishing.

    """
    if (part_method in ['fluid', 'louvain']) and (not networkx_import):
        warnings.warn(
            f"""
            Networkx is not installed, so part_method={part_method}
            is not available and by default set to `random`. Consider
            installing Networkx to make this functionality available.
            """
        )
        part_method = 'random'

    if (rep_method == 'pagerank') and (not networkx_import):
        warnings.warn(
            """
            Networkx is not installed, so rep_method=pagerank
            is not available and by default set to `random`. Consider
            installing Networkx to make this functionality available.
            """
        )
        rep_method = 'random'

    if (part_method == 'kmeans' or rep_method == 'kmeans') and (not sklearn_import):
        warnings.warn(
            """
            Scikit-Learn is not installed, so rep_method=kmeans
            is not available and by default set to `random`. Consider
            installing Scikit-Learn to make this functionality available.
            """
        )
        part_method = 'random'
        rep_method = 'random'

    arr = [C1, C2]
    if A1 is not None:
        arr.append(A1)
    else:
        A1 = C1
    if A2 is not None:
        arr.append(A2)
    else:
        A2 = C2
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(C2.shape[0], type_as=C1)

    nx = get_backend(*arr)

    # compute partition over A1 and A2
    part1 = _get_partition(A1, npart1, part_method=part_method, random_state=random_state, nx=nx)
    part2 = _get_partition(A2, npart2, part_method=part_method, random_state=random_state, nx=nx)

    # get representants for each partition
    rep_indices1 = _get_representants(A1, part1, rep_method=rep_method, random_state=random_state, nx=nx)
    rep_indices2 = _get_representants(A2, part2, rep_method=rep_method, random_state=random_state, nx=nx)

    # formate partitions over C1 and C2
    CR1, list_R1, list_p1 = _formate_partitioned_graph(C1, p, part1, rep_indices1, nx=nx)
    CR2, list_R2, list_p2 = _formate_partitioned_graph(C2, q, part2, rep_indices2, nx=nx)

    # call to partitioned quantized gromov-wasserstein solver

    res = quantized_gromov_wasserstein_partitioned(
        CR1, CR2, list_R1, list_R2, list_p1, list_p2, build_OT=True,
        log=log, armijo=armijo, max_iter=max_iter, tol_rel=tol_rel,
        tol_abs=tol_abs, nx=nx, **kwargs)
    if log:
        T_global, Ts_local, T, log_ = res

        # compute the qGW distance
        constC, hC1, hC2 = init_matrix(C1, C2, p, q, 'square_loss', nx)
        log_['qGW_dist'] = gwloss(constC, hC1, hC2, T, nx)

        return T_global, Ts_local, T, log_

    else:
        T_global, Ts_local, T = res

        return T_global, Ts_local, T


def quantized_gromov_wasserstein_partitioned(
        CR1, CR2, list_R1, list_R2, list_p1, list_p2, build_OT=False, log=False,
        armijo=False, max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, nx=None,
        **kwargs):
    r"""
    Returns the quantized Gromov-Wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})`
    and :math:`(\mathbf{C_2}, \mathbf{q})`, whose samples are assigned to partitions and representants
    :math:`\mathcal{P_1} = \{(\mathbf{P_{1, i}}, \mathbf{r_{1, i}})\}_{i \leq npart1}`
    and :math:`\mathcal{P_2} = \{(\mathbf{P_{2, j}}, \mathbf{r_{2, j}})\}_{j \leq npart2}`.
    The latter must be precomputed and encoded e.g for the source as: :math:`\mathbf{CR_1}`
    structure matrix between representants; `list_R1` a list of relations between representants
    and their associated samples; `list_p1` a list of nodes distribution within each partition.

    The function estimates the following optimization problem:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

             \mathbf{T}_{|\mathbf{P_{1, i}}, \mathbf{P_{2, j}}} &= T^{g}_{ij} \mathbf{T}^{(i,j)}

    using a two-step strategy computing: i) a global alignment :math:`\mathbf{T}^{g}`
    between representants across spaces; ii) local alignments
    :math:`\mathbf{T}^{(i, j)}` between partitions :math:`\mathbf{P_{1, i}}`
    and :math:`\mathbf{P_{2, j}}` seen as 1D measures.

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - :math:`L`: quadratic loss function to account for the misfit between the similarity matrices

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.
    .. note:: All computations in the Gromov-Wasserstein conjugate gradient solver
        are done with numpy to limit memory overhead.

    Parameters
    ----------
    CR1 : array-like, shape (npart1, npart1)
        Structure matrix between partition representants in the source space.
    CR2 : array-like, shape (npart2, npart2)
        Structure matrix between partition representants in the target space.
    list_R1 : list of npart1 arrays,
        List of relations between representants and their associated samples in the source space.
    list_R2 : list of npart2 arrays,
        List of relations between representants and their associated samples in the target space.
    list_p1 : list of npart1 arrays,
        List of node distributions within each partition of the source space.
    list_p : list of npart2 arrays,
        List of node distributions within each partition of the target space.
    build_OT: bool, optional. Default is False
        Either to build or not the OT between non-partitioned structures.
    log : bool, optional. Default is False
        record log if True
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research. Else closed form is used.
        If there are convergence issues use False.
    max_iter : int, optional
        Max number of iterations
    tol_rel : float, optional
        Stop threshold on relative error (>0)
    tol_abs : float, optional
        Stop threshold on absolute error (>0)
    nx : backend, optional
        POT backend

    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    T_global: array-like, shape (`npart1`, `npart2`)
        Gromov-Wasserstein alignment :math:`\mathbf{T}^{g}` between representants.
    Ts_local: dict of local OT matrices.
        Dictionary with keys :math:`(i, j)` corresponding to 1D OT between
        :math:`\mathbf{P_{1, i}}` and :math:`\mathbf{P_{2, j}}` if :math:`T^{g}_{ij} \neq 0`.
    T: array-like, shape `(ns, nt)`
        Coupling between the two spaces if `build_OT=True` else None.
    log : dict, if `log=True`.
        Convergence information and losses of inner OT problems.

    References
    ----------
    .. [66] Chowdhury, S., Miller, D., & Needham, T. (2021).
        Quantized gromov-wasserstein. ECML PKDD 2021. Springer International Publishing.

    """
    if nx is None:
        arr = [CR1, CR2, *list_R1, *list_R2, *list_p1, *list_p2]

        nx = get_backend(*arr)

    npart1 = len(list_R1)
    npart2 = len(list_R2)

    # compute marginals for global alignment
    pR1 = nx.from_numpy(list_to_array([nx.sum(p) for p in list_p1]))
    pR2 = nx.from_numpy(list_to_array([nx.sum(q) for q in list_p2]))

    # compute global alignment
    res_gw = gromov_wasserstein(
        CR1, CR2, pR1, pR2, loss_fun='square_loss', symmetric=True, log=log,
        armijo=armijo, G0=None, max_iter=max_iter, tol_rel=tol_rel, tol_abs=tol_abs)

    if log:
        log_ = {}
        T_global, log_gw = res_gw
        log_['gw_dist_CR'] = log_gw['gw_dist']
    else:
        T_global = res_gw

    # compute local alignments
    Ts_local = {}
    list_p1_norm = [p / nx.sum(p) for p in list_p1]
    list_p2_norm = [q / nx.sum(q) for q in list_p2]

    for i in range(npart1):
        for j in range(npart2):
            if T_global[i, j] != 0.:
                res_1d = emd_1d(list_R1[i], list_R2[j], list_p1_norm[i], list_p2_norm[j],
                                metric='sqeuclidean', p=1., log=log)
                if log:
                    T_local, log_local = res_1d
                    Ts_local[(i, j)] = T_local
                    log_[f'cost ({i},{j})'] = log_local['cost']
                else:
                    Ts_local[(i, j)] = res_1d

    if build_OT:
        T_rows = []
        for i in range(npart1):
            list_Ti = []
            for j in range(npart2):
                if T_global[i, j] == 0.:
                    T_local = nx.zeros((list_R1[i].shape[0], list_R2[j].shape[0]), type_as=T_global)
                else:
                    T_local = T_global[i, j] * Ts_local[(i, j)]
                list_Ti.append(T_local)

            Ti = nx.concatenate(list_Ti, axis=1)
            T_rows.append(Ti)
        T = nx.concatenate(T_rows, axis=0)

    else:
        T = None

    if log:
        return T_global, Ts_local, T, log_

    else:
        return T_global, Ts_local, T
