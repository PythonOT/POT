"""
Quantized (Fused) Gromov-Wasserstein solvers.
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

from ..utils import list_to_array, unif, dist
from ..backend import get_backend
from ..lp import emd_1d
from ._gw import gromov_wasserstein, fused_gromov_wasserstein
from ._utils import init_matrix, gwloss


def quantized_fused_gromov_wasserstein_partitioned(
    CR1,
    CR2,
    list_R1,
    list_R2,
    list_p1,
    list_p2,
    part1=None,
    part2=None,
    MR=None,
    alpha=1.0,
    build_OT=False,
    log=False,
    armijo=False,
    max_iter=1e4,
    tol_rel=1e-9,
    tol_abs=1e-9,
    nx=None,
    **kwargs,
):
    r"""
    Returns the quantized Fused Gromov-Wasserstein transport between
    :math:`(\mathbf{C_1}, \mathbf{F_1}, \mathbf{p})` and :math:`(\mathbf{C_2},
    \mathbf{F_2}, \mathbf{q})`, whose samples are assigned to partitions and representants
    :math:`\mathcal{P_1} = \{(\mathbf{P_{1, i}}, \mathbf{r_{1, i}})\}_{i \leq npart1}`
    and :math:`\mathcal{P_2} = \{(\mathbf{P_{2, j}}, \mathbf{r_{2, j}})\}_{j \leq npart2}`.
    The latter must be precomputed and encoded e.g for the source as: :math:`\mathbf{CR_1}`
    structure matrix between representants; `list_R1` a list of relations between
    representants and their associated samples; `list_p1` a list of nodes
    distribution within each partition; :math:`\mathbf{FR_1}` feature matrix
    of representants.

    The function estimates the following optimization problem:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_\mathbf{T} \quad \alpha \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}
        + (1-\alpha) \langle \mathbf{T}, M\rangle_F
        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

             \mathbf{T}_{|\mathbf{P_{1, i}}, \mathbf{P_{2, j}}} &= T^{g}_{ij} \mathbf{T}^{(i,j)}

    using a two-step strategy computing: i) a global alignment :math:`\mathbf{T}^{g}`
    between representants joint structure and feature spaces; ii) local alignments
    :math:`\mathbf{T}^{(i, j)}` between partitions :math:`\mathbf{P_{1, i}}`
    and :math:`\mathbf{P_{2, j}}` seen as 1D measures.

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{F_1}`: Feature matrix in the source space
    - :math:`\mathbf{F_2}`: Feature matrix in the target space
    - :math:`\mathbf{M}`: Pairwise similarity matrix between features
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
    part1 : list of npart1 arrays, optional. Default is None.
        List of arrays containing the indices of nodes in each partition of the source space.
        Required as input if `build_OT=True`.
    part2 : list of npart2 arrays, optional. Default is None.
        List of arrays containing the indices of nodes in each partition of the target space.
        Required as input if `build_OT=True`.
    MR : array-like, shape (npart1, npart2), optional. (Default is None)
        Metric cost matrix between features of representants across spaces.
    alpha: float, optional. Default is None.
        FGW trade-off parameter in :math:`]0, 1]` between structure and features.
        If `alpha = 1` features are ignored hence computing qGW.
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
    .. [68] Chowdhury, S., Miller, D., & Needham, T. (2021).
        Quantized gromov-wasserstein. ECML PKDD 2021. Springer International Publishing.

    """
    if nx is None:
        arr = [CR1, CR2, *list_R1, *list_R2, *list_p1, *list_p2, MR]
        if build_OT:
            arr += [*part1, *part2]

        nx = get_backend(*arr)

    npart1 = len(list_R1)
    npart2 = len(list_R2)

    if (npart1 != len(list_p1)) or (npart2 != len(list_p2)):
        raise ValueError(
            f"""
            Inconsistent number of partitions between list_R1 ({npart1}), list_p1 ({len(list_p1)}), 
            and list_R2 ({npart2}), list_p2 ({len(list_p2)}).
            """
        )
    if build_OT and ((npart1 != len(part1)) or (npart2 != len(part2))):
        raise ValueError(
            f"""
            Inconsistent number of partitions in part1 ({len(part1)})
            and part2 ({len(part2)}).
            """
        )

    # compute marginals for global alignment
    pR1 = nx.from_numpy(list_to_array([nx.sum(p) for p in list_p1]), type_as=CR1)
    pR2 = nx.from_numpy(list_to_array([nx.sum(q) for q in list_p2]), type_as=CR2)

    # compute global alignment
    if alpha == 1.0:
        res_global = gromov_wasserstein(
            CR1,
            CR2,
            pR1,
            pR2,
            loss_fun="square_loss",
            log=log,
            armijo=armijo,
            max_iter=max_iter,
            tol_rel=tol_rel,
            tol_abs=tol_abs,
        )

        if log:
            T_global, dist_global = res_global[0], res_global[1]["gw_dist"]
        else:
            T_global = res_global

    elif (alpha < 1.0) and (alpha > 0.0):
        res_global = fused_gromov_wasserstein(
            MR,
            CR1,
            CR2,
            pR1,
            pR2,
            "square_loss",
            alpha=alpha,
            log=log,
            armijo=armijo,
            max_iter=max_iter,
            tol_rel=tol_rel,
            tol_abs=tol_abs,
        )

        if log:
            T_global, dist_global = res_global[0], res_global[1]["fgw_dist"]
        else:
            T_global = res_global

    else:
        raise ValueError(
            f"""
            `alpha='{alpha}'` should be in ]0, 1].
            """
        )

    if log:
        log_ = {}
        log_["global dist"] = dist_global

    # compute local alignments
    Ts_local = {}
    list_p1_norm = [p / nx.sum(p) for p in list_p1]
    list_p2_norm = [q / nx.sum(q) for q in list_p2]

    for i in range(npart1):
        for j in range(npart2):
            if T_global[i, j] != 0.0:
                res_1d = emd_1d(
                    list_R1[i],
                    list_R2[j],
                    list_p1_norm[i],
                    list_p2_norm[j],
                    metric="sqeuclidean",
                    p=1.0,
                    log=log,
                )
                if log:
                    T_local, log_local = res_1d
                    Ts_local[(i, j)] = T_local
                    log_[f"local dist ({i},{j})"] = log_local["cost"]
                else:
                    Ts_local[(i, j)] = res_1d

    if build_OT:
        # Do memory-efficient alternatives by backend for mutable vs non-mutable tensors
        if nx.__name__ in ("numpy", "torch"):
            T = _build_full_transport_by_assignment(
                T_global,
                Ts_local,
                part1,
                part2,
                nx,
            )
        else:
            T = _build_full_transport_by_concatenation(
                T_global,
                Ts_local,
                list_R1,
                list_R2,
                part1,
                part2,
                nx,
            )
    else:
        T = None

    if log:
        return T_global, Ts_local, T, log_

    else:
        return T_global, Ts_local, T


def _build_full_transport_by_assignment(
    T_global,
    Ts_local,
    part1,
    part2,
    nx,
):
    """
    Build the full transport matrix by assigning local transport blocks.

    The local couplings are written directly into the rows and columns
    identified by each source and target partition. This preserves the
    original sample ordering without materializing a partition-ordered
    intermediate matrix.

    Parameters
    ----------
    T_global : array-like, shape (npart1, npart2)
        Global transport between source and target partition representants.
    Ts_local : dict
        Dictionary of local transport matrices keyed by ``(i, j)``.
    part1 : list of array-like
        Source partition indices in the original sample ordering.
    part2 : list of array-like
        Target partition indices in the original sample ordering.
    nx : backend
        POT backend used to create and assign the transport matrix.

    Returns
    -------
    T : array-like, shape (ns, nt)
        Full transport matrix between the original source and target samples.
    """
    ns = sum(indices.shape[0] for indices in part1)
    nt = sum(indices.shape[0] for indices in part2)

    T = nx.zeros((ns, nt), type_as=T_global)

    for i, rows in enumerate(part1):
        for j, cols in enumerate(part2):
            if T_global[i, j] != 0.0:
                T_block = T_global[i, j] * Ts_local[(i, j)]
                T[rows[:, None], cols[None, :]] = T_block

    return T


def _build_full_transport_by_concatenation(
    T_global,
    Ts_local,
    list_R1,
    list_R2,
    part1,
    part2,
    nx,
):
    """
    Build the full transport matrix by concatenating partition blocks.

    Local couplings are first concatenated in source and target partition
    order, then rows and columns are reordered according to the partition
    indices so that the result uses the original sample ordering.

    Parameters
    ----------
    T_global : array-like, shape (npart1, npart2)
        Global transport between source and target partition representants.
    Ts_local : dict
        Dictionary of local transport matrices keyed by ``(i, j)``.
    list_R1 : list of array-like
        Source representative-to-sample relations, used to determine block
        dimensions.
    list_R2 : list of array-like
        Target representative-to-sample relations, used to determine block
        dimensions.
    part1 : list of array-like
        Source partition indices in the original sample ordering.
    part2 : list of array-like
        Target partition indices in the original sample ordering.
    nx : backend
        POT backend used to concatenate and reorder the transport matrix.

    Returns
    -------
    T : array-like, shape (ns, nt)
        Full transport matrix between the original source and target samples.
    """
    T_rows = []

    for i in range(len(list_R1)):
        T_blocks = []

        for j in range(len(list_R2)):
            if T_global[i, j] == 0.0:
                T_block = nx.zeros(
                    (list_R1[i].shape[0], list_R2[j].shape[0]),
                    type_as=T_global,
                )
            else:
                T_block = T_global[i, j] * Ts_local[(i, j)]

            T_blocks.append(T_block)

        T_rows.append(nx.concatenate(T_blocks, axis=1))

    T = nx.concatenate(T_rows, axis=0)

    perm1 = nx.concatenate(part1, axis=0)
    perm2 = nx.concatenate(part2, axis=0)

    T = T[nx.argsort(perm1)]
    T = nx.transpose(T)
    T = T[nx.argsort(perm2)]
    T = nx.transpose(T)

    return T


def get_graph_partition(
    C, npart, part_method="random", F=None, alpha=1.0, random_state=0, nx=None
):
    r"""
    Partitioning a given graph with structure matrix :math:`\mathbf{C} \in R^{n \times n}`
    into `npart` partitions either 'random', or using one of {'louvain', 'fluid'}
    algorithms from networkx, or 'spectral' clustering from scikit-learn,
    or (Fused) Gromov-Wasserstein projections from POT.

    Parameters
    ----------
    C : array-like, shape (n, n)
        Structure matrix.
    npart : int,
        number of partitions/clusters smaller than the number of nodes in
        :math:`\mathbf{C}`.
    part_method : str, optional. Default is 'random'.
        Partitioning algorithm to use among {'random', 'louvain', 'fluid', 'spectral', 'GW', 'FGW'}.
        'random' for random sampling of points; 'louvain' and 'fluid' for graph
        partitioning algorithm that works well on adjacency matrix, If the
        louvain algorithm is used, `npart` is ignored; 'spectral' for spectral
        clustering; '(F)GW' for (F)GW projection using sr(F)GW solvers.
    F : array-like, shape (n, d), optional. (Default is None)
        Optional feature matrix aligned with the graph structure. Only used if
        `part_method="FGW"`.
    alpha : float, optional. (Default is 1.)
        Trade-off parameter between feature and structure matrices, taking
        values in [0, 1] and only used if `F != None` and `part_method="FGW"`.
    random_state: int, optional
        Random seed for the partitioning algorithm.
    nx : backend, optional
        POT backend.

    Returns
    -------
    part : list of array-like, length npart
        List of arrays containing the indices of nodes in each partition.

    References
    ----------
    .. [68] Chowdhury, S., Miller, D., & Needham, T. (2021).
        Quantized gromov-wasserstein. ECML PKDD 2021. Springer International Publishing.

    """
    if nx is None:
        nx = get_backend(C)

    n = C.shape[0]
    C0 = C

    if (alpha != 1.0) and (F is None):
        raise ValueError("`alpha != 1` but node features are not provided.")

    if npart >= n:
        warnings.warn(
            "Requested number of partitions higher than the number of nodes"
            "hence we enforce each node to be a partition.",
            stacklevel=2,
        )

        part = list(nx.arange(n)[:, None])

    elif npart == 1:
        part = [nx.arange(n)]

    elif part_method == "random":
        # randomly partition the space
        random.seed(random_state)
        part_assignments = random.choices(np.arange(npart), k=n)
        part = [
            nx.from_numpy(np.where(np.array(part_assignments) == i)[0])
            for i in range(npart)
        ]

    elif part_method in ["louvain", "fluid"]:
        C = nx.to_numpy(C0)
        graph = from_numpy_array(C)
        if part_method == "louvain":
            part_sets = louvain_communities(graph, seed=random_state)
        else:
            part_sets = asyn_fluidc(graph, npart, seed=random_state)
        part = [
            nx.from_numpy(np.array(list(nodes)).astype(np.int64)) for nodes in part_sets
        ]

    elif part_method == "spectral":
        C = nx.to_numpy(C0)
        sc = SpectralClustering(
            n_clusters=npart, random_state=random_state, affinity="precomputed"
        ).fit(C)
        labels = sc.labels_
        part = [nx.from_numpy(np.where(labels == i)[0]) for i in range(npart)]

    elif part_method in ["GW", "FGW"]:
        raise ValueError(f"`part_method == {part_method}` not implemented yet.")

    else:
        raise ValueError(
            f"""
            Unknown `part_method='{part_method}'`. Use one of:
            {"random", "louvain", "fluid", "spectral", "GW", "FGW"}.
            """
        )

    return part


def get_graph_representants(C, part, rep_method="pagerank", random_state=0, nx=None):
    r"""
    Get representative node for each partition given by :math:`\mathbf{part} \in R^{n}`
    of a graph with structure matrix :math:`\mathbf{C} \in R^{n \times n}`.
    Selection is either done randomly or using 'pagerank' algorithm from networkx.

    Parameters
    ----------
    C : array-like, shape (n, n)
        structure matrix.
    part : list of array-like, length npart
        List of arrays containing the indices of nodes in each partition.
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
    rep_indices : array-like, shape (npart,)
        Array of indices for representative node of each partition sorted
        according to partition order in `part` with same type as `C`.

    References
    ----------
    .. [68] Chowdhury, S., Miller, D., & Needham, T. (2021).
        Quantized gromov-wasserstein. ECML PKDD 2021. Springer International Publishing.

    """
    if nx is None:
        nx = get_backend(C, *part)

    rep_indices = []
    n = C.shape[0]
    n_part = len(part)
    if n_part == n:
        rep_indices = [indices[0] for indices in part]

    elif rep_method == "random":
        random.seed(random_state)
        for indices in part:
            rep_indices.append(random.choice(indices))

    elif rep_method == "pagerank":
        C0, part0 = C, part
        C = nx.to_numpy(C0)
        part = [nx.to_numpy(indices) for indices in part]

        for indices in part:
            C_id = C[indices, :][:, indices]
            graph = from_numpy_array(C_id)
            pagerank_values = list(pagerank(graph).values())
            rep_idx = np.argmax(pagerank_values)
            rep_indices.append(indices[rep_idx])
        C, part = C0, part0
    else:
        raise ValueError(
            f"""
            Unknown `rep_method='{rep_method}'`. Use one of:
            {"random", "pagerank"}.
            """
        )
    rep_indices = nx.from_numpy(list_to_array(rep_indices).astype(np.int64))
    return rep_indices


def format_partitioned_graph(
    C, p, part, rep_indices, F=None, M=None, alpha=1.0, nx=None
):
    r"""
    Format an attributed graph :math:`(\mathbf{C}, \mathbf{F}, \mathbf{p})`
    with structure matrix :math:`(\mathbf{C} \in R^{n \times n}`, feature matrix
    :math:`(\mathbf{F} \in R^{n \times d}` and node relative importance
    :math:`(\mathbf{p} \in \Sigma_n`, into a partitioned attributed graph
    taking into account partitions and representants
    :math:`\mathcal{P} = \left\{(\mathbf{P_{i}}, \mathbf{r_{i}})\right\}_i`.

    Parameters
    ----------
    C : array-like, shape (n, n)
        Structure matrix.
    p : array-like, shape (n,),
        Node distribution.
    part : list of array-like, length npart
        List of arrays containing the indices of nodes in each partition.
    rep_indices : array-like, shape (npart,)
        Array of indices for representative node of each partition sorted
        according to partition order in `part` with same type as `C`.
    F : array-like, shape (n, d), optional. (Default is None)
        Optional feature matrix aligned with the graph structure.
    M : array-like, shape (n, n), optional. (Default is None)
        Optional pairwise similarity matrix between features.
    alpha: float, optional. Default is 1.
        Trade-off parameter in :math:`]0, 1]` between structure and features.
        If `alpha = 1` features are ignored. This trade-off is taken into account
        into the outputted relations between nodes and representants.
    nx : backend, optional
        POT backend

    Returns
    -------
    CR : array-like, shape (npart, npart)
        Structure matrix between partition representants.
    list_R : list of npart arrays,
        List of relations between a representant and the nodes in its partition,
        for each partition.
    list_p : list of npart arrays,
        List of node distributions within each partition.
    FR : array-like, shape (npart, d), if `F != None`.
        Feature matrix of representants.

    References
    ----------
    .. [68] Chowdhury, S., Miller, D., & Needham, T. (2021).
        Quantized gromov-wasserstein. ECML PKDD 2021. Springer International Publishing.

    """
    if nx is None:
        nx = get_backend(C, p, *part, rep_indices, F, M)

    if alpha != 1.0:
        if (M is None) or (F is None):
            raise ValueError(
                f"""
                `alpha == {alpha} != 1` but features information is not properly provided.
                """
            )

    CR = C[rep_indices, :][:, rep_indices]

    if alpha != 1.0:
        C_new = alpha * C + (1 - alpha) * M
    else:
        C_new = C

    list_R, list_p = [], []

    for cluster_id, indices in enumerate(part):
        list_R.append(C_new[rep_indices[cluster_id], indices])
        list_p.append(p[indices])

    if F is None:
        return CR, list_R, list_p
    else:
        FR = F[rep_indices, :]

        return CR, list_R, list_p, FR


def quantized_fused_gromov_wasserstein(
    C1,
    C2,
    npart1,
    npart2,
    p=None,
    q=None,
    C1_aux=None,
    C2_aux=None,
    F1=None,
    F2=None,
    alpha=1.0,
    part_method="fluid",
    rep_method="random",
    log=False,
    armijo=False,
    max_iter=1e4,
    tol_rel=1e-9,
    tol_abs=1e-9,
    random_state=0,
    **kwargs,
):
    r"""
    Returns the quantized Fused Gromov-Wasserstein transport between
    :math:`(\mathbf{C_1}, \mathbf{F_1}, \mathbf{p})` and :math:`(\mathbf{C_2},
    \mathbf{F_2}, \mathbf{q})`, whose samples are assigned to partitions and
    representants :math:`\mathcal{P_1} = \{(\mathbf{P_{1, i}}, \mathbf{r_{1, i}})\}_{i \leq npart1}`
    and :math:`\mathcal{P_2} = \{(\mathbf{P_{2, j}}, \mathbf{r_{2, j}})\}_{j \leq npart2}`.

    The function estimates the following optimization problem:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_\mathbf{T} \quad \alpha \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}
        + (1-\alpha) \langle \mathbf{T}, \mathbf{D}(\mathbf{F_1}, \mathbf{F}_2) \rangle_F
        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

             \mathbf{T}_{|\mathbf{P_{1, i}}, \mathbf{P_{2, j}}} &= T^{g}_{ij} \mathbf{T}^{(i,j)}

    using a two-step strategy computing: i) a global alignment :math:`\mathbf{T}^{g}`
    between representants across joint structure and feature spaces;
    ii) local alignments :math:`\mathbf{T}^{(i, j)}` between partitions
    :math:`\mathbf{P_{1, i}}` and :math:`\mathbf{P_{2, j}}` seen as 1D measures.

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{F_1}`: Feature matrix in the source space
    - :math:`\mathbf{F_2}`: Feature matrix in the target space
    - :math:`\mathbf{D}(\mathbf{F_1}, \mathbf{F_2})`: Pairwise euclidean distance matrix between features
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
        Structure matrix in the source space.
    C2 : array-like, shape (nt, nt)
        Structure matrix in the target space.
    npart1 : int,
        number of partition in the source space.
    npart2 : int,
        number of partition in the target space.
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    C1_aux : array-like, shape (ns, ns), optional. Default is None.
        Auxiliary structure matrix in the source space to perform the partitioning
        and representant selection.
    C2_aux : array-like, shape (nt, nt), optional. Default is None.
        Auxiliary structure matrix in the target space to perform the partitioning
        and representant selection.
    F1 : array-like, shape (ns, d), optional. Default is None.
        Feature matrix in the source space.
    F2 : array-like, shape (nt, d), optional. Default is None.
        Feature matrix in the target space
    alpha: float, optional. Default is 1.
        FGW trade-off parameter in :math:`]0, 1]` between structure and features.
        If `alpha = 1` features are ignored hence computing qGW, if `alpha=0`
        structures are ignored and we compute the quantized Wasserstein transport.
    part_method : str, optional. Default is 'spectral'.
        Partitioning algorithm to use among {'random', 'louvain', 'fluid',
        'spectral', 'louvain_fused', 'fluid_fused', 'spectral_fused', 'GW', 'FGW'}.
        If part_method in {'louvain_fused', 'fluid_fused', 'spectral_fused'},
        corresponding graph partitioning algorithm {'louvain', 'fluid', 'spectral'}
        will be used on the modified structure matrix
        :math:`\alpha \mathbf{C} + (1 - \alpha) \mathbf{D}(\mathbf{F})` where
        :math:`\mathbf{D}(\mathbf{F})` is the pairwise euclidean matrix between features.
        If part_method in {'GW', 'FGW'}, a (F)GW projection is used.
        If the louvain algorithm is used, the requested number of partitions is
        ignored.
    rep_method : str, optional. Default is 'pagerank'.
        Selection method for node representant in each partition.
        Can be either 'random' i.e random sampling within each partition,
        {'pagerank', 'pagerank_fused'} to select a node with maximal pagerank w.r.t
        :math:`\mathbf{C}` or :math:`\alpha \mathbf{C} + (1 - \alpha) \mathbf{D}(\mathbf{F})`.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
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
    random_state: int, optional
        Random seed for the partitioning algorithm
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    T_global: array-like, shape (`npart1`, `npart2`)
        Fused Gromov-Wasserstein alignment :math:`\mathbf{T}^{g}` between representants.
    Ts_local: dict of local OT matrices.
        Dictionary with keys :math:`(i, j)` corresponding to 1D OT between
        :math:`\mathbf{P_{1, i}}` and :math:`\mathbf{P_{2, j}}` if :math:`T^{g}_{ij} \neq 0`.
    T: array-like, shape `(ns, nt)`
        Coupling between the two spaces.
    log : dict
        Convergence information for inner problems and qGW loss.

    References
    ----------
    .. [68] Chowdhury, S., Miller, D., & Needham, T. (2021).
        Quantized gromov-wasserstein. ECML PKDD 2021. Springer International Publishing.

    """
    if part_method in ["fluid", "louvain", "fluid_fused", "louvain_fused"] or (
        rep_method in ["pagerank", "pagerank_fused"]
    ):
        if not networkx_import:
            warnings.warn(
                f"""
                Networkx is not installed, so part_method={part_method} and/or
                rep_method={rep_method} cannot be used and are set to `random`
                default methods. Consider installing Networkx to fix this.
                """
            )
        part_method = "random"
        rep_method = "random"

    if (part_method in ["spectral", "spectral_fused"]) and (not sklearn_import):
        warnings.warn(
            f"""
            Scikit-learn is not installed, so part_method={part_method} and/or
            rep_method={rep_method} cannot be used and are set to `random`
            default methods. Consider installing Scikit-learn to fix this.
            """
        )
        part_method = "random"
        rep_method = "random"

    if ("fused" in part_method) or ("fused" in rep_method) or (part_method == "FGW"):
        if (F1 is None) or (F2 is None):
            raise ValueError(
                f"""
                `part_method='{part_method}'` and/or `rep_method='{rep_method}'`
                require feature matrices which are not provided as inputs.
                """
            )

    arr = [C1, C2, C1_aux, C2_aux, F1, F2]
    if C1_aux is None:
        C1_aux = C1
    if C2_aux is None:
        C2_aux = C2
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(C2.shape[0], type_as=C1)

    nx = get_backend(*arr)

    DF1 = None
    DF2 = None
    # compute attributed graph partitions potentially using the auxiliary structure
    if "fused" in part_method:
        DF1 = dist(F1, F1)
        DF2 = dist(F2, F2)
        C1_new = alpha * C1_aux + (1 - alpha) * DF1
        C2_new = alpha * C2_aux + (1 - alpha) * DF2

        part_method_ = part_method[:-6]
        part1 = get_graph_partition(
            C1_new, npart1, part_method_, random_state=random_state, nx=nx
        )
        part2 = get_graph_partition(
            C2_new, npart2, part_method_, random_state=random_state, nx=nx
        )

    else:
        part1 = get_graph_partition(
            C1_aux, npart1, part_method, F1, alpha, random_state, nx
        )
        part2 = get_graph_partition(
            C2_aux, npart2, part_method, F2, alpha, random_state, nx
        )

    if "fused" in rep_method:
        if DF1 is None:
            DF1 = dist(F1, F1)
            DF2 = dist(F2, F2)
            C1_new = alpha * C1_aux + (1 - alpha) * DF1
            C2_new = alpha * C2_aux + (1 - alpha) * DF2

        rep_method_ = rep_method[:-6]

        rep_indices1 = get_graph_representants(
            C1_new, part1, rep_method_, random_state, nx
        )
        rep_indices2 = get_graph_representants(
            C2_new, part2, rep_method_, random_state, nx
        )

    else:
        rep_indices1 = get_graph_representants(
            C1_aux, part1, rep_method, random_state, nx
        )
        rep_indices2 = get_graph_representants(
            C2_aux, part2, rep_method, random_state, nx
        )

    # format partitions over (C1, F1) and (C2, F2)
    if (F1 is None) and (F2 is None):
        CR1, list_R1, list_p1 = format_partitioned_graph(
            C1, p, part1, rep_indices1, nx=nx
        )
        CR2, list_R2, list_p2 = format_partitioned_graph(
            C2, q, part2, rep_indices2, nx=nx
        )

        MR = None
    else:
        if DF1 is None:
            DF1 = dist(F1, F1)
            DF2 = dist(F2, F2)

        CR1, list_R1, list_p1, FR1 = format_partitioned_graph(
            C1, p, part1, rep_indices1, F1, DF1, alpha, nx
        )
        CR2, list_R2, list_p2, FR2 = format_partitioned_graph(
            C2, q, part2, rep_indices2, F2, DF2, alpha, nx
        )

        MR = dist(FR1, FR2)
    # call to partitioned quantized fused gromov-wasserstein solver

    res = quantized_fused_gromov_wasserstein_partitioned(
        CR1,
        CR2,
        list_R1,
        list_R2,
        list_p1,
        list_p2,
        part1,
        part2,
        MR,
        alpha,
        build_OT=True,
        log=log,
        armijo=armijo,
        max_iter=max_iter,
        tol_rel=tol_rel,
        tol_abs=tol_abs,
        nx=nx,
        **kwargs,
    )

    if log:
        T_global, Ts_local, T, log_ = res

        # compute the transport cost on structures
        constC, hC1, hC2 = init_matrix(C1, C2, p, q, "square_loss", nx)
        structure_cost = gwloss(constC, hC1, hC2, T, nx)

        if alpha != 1.0:
            M = dist(F1, F2)
            feature_cost = nx.sum(M * T)
        else:
            feature_cost = 0.0

        log_["qFGW_dist"] = alpha * structure_cost + (1 - alpha) * feature_cost
        return T_global, Ts_local, T, log_

    else:
        T_global, Ts_local, T = res

        return T_global, Ts_local, T


def get_partition_and_representants_samples(
    X, npart, method="kmeans", random_state=0, nx=None
):
    r"""
    Compute `npart` partitions and representants over samples :math:`\mathbf{X} \in R^{n \times d}`
    using either a random or a kmeans algorithm.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Samples endowed with an euclidean geometry.
    npart : int,
        number of partitions smaller than the number of samples in
        :math:`\mathbf{X}`.
    method : str, optional. Default is 'kmeans'.
        Partitioning and representant selection algorithms to use among
        {'random', 'kmeans'}. 'random' for random sampling of points; 'kmeans'
        for k-means clustering using scikit-learn implementation where closest
        points to centroids are considered as representants.
    random_state: int, optional
        Random seed for the partitioning algorithm.
    nx : backend, optional
        POT backend.

    Returns
    -------
    part : list of array-like, length npart
        List of arrays containing the indices of nodes in each partition.

    rep_indices : list, shape (npart,)
        indices for representative node of each partition sorted
        according to partition identifiers.

    References
    ----------
    .. [68] Chowdhury, S., Miller, D., & Needham, T. (2021).
        Quantized gromov-wasserstein. ECML PKDD 2021. Springer International Publishing.

    """
    if nx is None:
        nx = get_backend(X)

    n = X.shape[0]
    X0 = X

    if npart >= n:
        warnings.warn(
            "Requested number of partitions higher than the number of nodes"
            "hence we enforce each node to be a partition.",
            stacklevel=2,
        )

        part = list(nx.arange(n)[:, None])
        rep_indices = [i for i in range(n)]

    elif npart == 1:
        random.seed(random_state)
        part = [nx.arange(n)]
        rep_indices = [random.choice(np.arange(n))]

    elif method == "random":
        # randomly partition the space
        random.seed(random_state)
        part_assignments = random.choices(np.arange(npart), k=X.shape[0])
        part = [
            nx.from_numpy(np.where(np.array(part_assignments) == i)[0])
            for i in range(npart)
        ]

        # randomly select representant in each partition
        rep_indices = []
        for indices_array in part:
            indices = nx.to_numpy(indices_array)
            rep_indices.append(random.choice(indices))

    elif method == "kmeans":
        X = nx.to_numpy(X0)
        km = KMeans(n_clusters=npart, random_state=random_state).fit(X)
        labels = km.labels_
        part = [
            nx.from_numpy(np.where(labels == i)[0].astype(np.int64))
            for i in range(npart)
        ]

        rep_indices = []
        for i in range(npart):
            indices = np.where(labels == i)[0]
            dists = dist(X[indices], km.cluster_centers_[i][None, :])
            best_idx = indices[dists.argmin()]
            rep_indices.append(best_idx)

    else:
        raise ValueError(
            f"""
            Unknown `method='{method}'`. Use one of: {"random", "kmeans"}
            """
        )

    rep_indices = nx.from_numpy(list_to_array(rep_indices), type_as=part[0])
    # print('part:', type(part), type(part[0]), part[0].dtype)
    # print('rep_indices:', type(rep_indices), rep_indices.dtype)
    return part, rep_indices


def format_partitioned_samples(X, p, part, rep_indices, F=None, alpha=1.0, nx=None):
    r"""
    Format an attributed graph :math:`(\mathbf{D}(\mathbf{X}), \mathbf{F}, \mathbf{p})`
    with euclidean structure matrix :math:`(\mathbf{D}(\mathbf{X}) \in R^{n \times n}`,
    feature matrix :math:`(\mathbf{F} \in R^{n \times d}` and node relative importance
    :math:`(\mathbf{p} \in \Sigma_n`, into a partitioned attributed graph
    taking into account partitions and representants
    :math:`\mathcal{P} = \left\{(\mathbf{P_{i}}, \mathbf{r_{i}})\right\}_i`.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Structure matrix.
    p : array-like, shape (n,),
        Node distribution.
    part : list of array-like, length npart
        List of arrays containing the indices of nodes in each partition.
    rep_indices : array-like, shape (npart,)
        Array of indices for representative node of each partition sorted
        according to partition identifiers.
    F : array-like, shape (n, p), optional. (Default is None)
        Optional feature matrix aligned with the samples.
    alpha: float, optional. Default is 1.
        Trade-off parameter in :math:`]0, 1]` between structure and features.
        If `alpha = 1` features are ignored. This trade-off is taken into account
        into the outputted relations between nodes and representants.
    nx : backend, optional
        POT backend

    Returns
    -------
    CR : array-like, shape (npart, npart)
        Structure matrix between partition representants.
    list_R : list of npart arrays,
        List of relations between a representant and nodes in its partition,
        for each partition.
    list_p : list of npart arrays,
        List of node distributions within each partition.
    FR : array-like, shape (npart, d), if `F != None`.
        Feature matrix of representants.

    References
    ----------
    .. [68] Chowdhury, S., Miller, D., & Needham, T. (2021).
        Quantized gromov-wasserstein. ECML PKDD 2021. Springer International Publishing.

    """
    if nx is None:
        arr = [X, p, *part, rep_indices]
        if F is not None:
            arr.append(F)

        nx = get_backend(*arr)

    if alpha != 1.0:
        if F is None:
            raise ValueError(
                f"""
                `alpha == {alpha} != 1` but features information is not properly provided.
                """
            )

    XR = X[rep_indices, :]
    CR = dist(XR, XR)

    list_R, list_p = [], []

    for cluster_id, indices in enumerate(part):
        structure_R = dist(X[indices], X[rep_indices[cluster_id]][None, :])

        if alpha != 1:
            features_R = dist(F[indices], F[rep_indices[cluster_id]][None, :])
        else:
            features_R = 0.0

        list_R.append(alpha * structure_R + (1 - alpha) * features_R)
        list_p.append(p[indices])

    if F is None:
        return CR, list_R, list_p
    else:
        FR = F[rep_indices, :]

        return CR, list_R, list_p, FR


def quantized_fused_gromov_wasserstein_samples(
    X1,
    X2,
    npart1,
    npart2,
    p=None,
    q=None,
    F1=None,
    F2=None,
    alpha=1.0,
    method="kmeans",
    log=False,
    armijo=False,
    max_iter=1e4,
    tol_rel=1e-9,
    tol_abs=1e-9,
    random_state=0,
    **kwargs,
):
    r"""
    Returns the quantized Fused Gromov-Wasserstein transport between samples
    endowed with their respective euclidean geometry :math:`(\mathbf{D}(\mathbf{X_1}), \mathbf{F_1}, \mathbf{p})`
    and :math:`(\mathbf{D}(\mathbf{X_1}), \mathbf{F_2}, \mathbf{q})`, whose samples are assigned to partitions and
    representants :math:`\mathcal{P_1} = \{(\mathbf{P_{1, i}}, \mathbf{r_{1, i}})\}_{i \leq npart1}`
    and :math:`\mathcal{P_2} = \{(\mathbf{P_{2, j}}, \mathbf{r_{2, j}})\}_{j \leq npart2}`.

    The function estimates the following optimization problem:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_\mathbf{T} \quad \alpha \sum_{i,j,k,l}
        L(\mathbf{D}(\mathbf{X_1})_{i,k}, \mathbf{D}(\mathbf{X_2})_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}
        + (1-\alpha) \langle \mathbf{T}, \mathbf{D}(\mathbf{F_1}, \mathbf{F}_2) \rangle_F
        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

             \mathbf{T}_{|\mathbf{P_{1, i}}, \mathbf{P_{2, j}}} &= T^{g}_{ij} \mathbf{T}^{(i,j)}

    using a two-step strategy computing: i) a global alignment :math:`\mathbf{T}^{g}`
    between representants across joint structure and feature spaces;
    ii) local alignments :math:`\mathbf{T}^{(i, j)}` between partitions
    :math:`\mathbf{P_{1, i}}` and :math:`\mathbf{P_{2, j}}` seen as 1D measures.

    Where :

    - :math:`\mathbf{X_1}`: Samples in the source space
    - :math:`\mathbf{X_2}`: Samples in the target space
    - :math:`\mathbf{F_1}`: Feature matrix in the source space
    - :math:`\mathbf{F_2}`: Feature matrix in the target space
    - :math:`\mathbf{D}(\mathbf{F_1}, \mathbf{F_2})`: Pairwise euclidean distance matrix between features
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
    X1 : array-like, shape (ns, ds)
        Samples in the source space.
    X2 : array-like, shape (nt, dt)
        Samples in the target space.
    npart1 : int,
        number of partition in the source space.
    npart2 : int,
        number of partition in the target space.
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    F1 : array-like, shape (ns, d), optional. Default is None.
        Feature matrix in the source space.
    F2 : array-like, shape (nt, d), optional. Default is None.
        Feature matrix in the target space
    alpha: float, optional. Default is 1.
        FGW trade-off parameter in :math:`]0, 1]` between structure and features.
        If `alpha = 1` features are ignored hence computing qGW, if `alpha=0`
        structures are ignored and we compute the quantized Wasserstein transport.
    method : str, optional. Default is 'kmeans'.
        Partitioning and representant selection algorithms to use among
        {'random', 'kmeans', 'kmeans_fused'}.
        If `part_method == 'kmeans_fused'`, kmeans is performed on augmented
        samples :math:`[\alpha \mathbf{X}; (1 - \alpha) \mathbf{F}]`.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
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
    random_state: int, optional
        Random seed for the partitioning algorithm
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    T_global: array-like, shape (`npart1`, `npart2`)
        Fused Gromov-Wasserstein alignment :math:`\mathbf{T}^{g}` between representants.
    Ts_local: dict of local OT matrices.
        Dictionary with keys :math:`(i, j)` corresponding to 1D OT between
        :math:`\mathbf{P_{1, i}}` and :math:`\mathbf{P_{2, j}}` if :math:`T^{g}_{ij} \neq 0`.
    T: array-like, shape `(ns, nt)`
        Coupling between the two spaces.
    log : dict
        Convergence information for inner problems and qGW loss.

    References
    ----------
    .. [68] Chowdhury, S., Miller, D., & Needham, T. (2021).
        Quantized gromov-wasserstein. ECML PKDD 2021. Springer International Publishing.

    """

    if (method in ["kmeans", "kmeans_fused"]) and (not sklearn_import):
        warnings.warn(
            f"""
            Scikit-learn is not installed, so method={method} cannot be used
            and is set to `random` default methods. Consider installing
            Scikit-learn to fix this.
            """
        )
        method = "random"

    if ("fused" in method) and ((F1 is None) or (F2 is None)):
        raise ValueError(
            f"""
            `method='{method}'` requires feature matrices which are not provided as inputs.
            """
        )

    arr = [X1, X2, F1, F2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(X1.shape[0], type_as=X1)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(X2.shape[0], type_as=X1)

    nx = get_backend(*arr)

    # compute attributed partitions and representants
    if ("fused" in method) and (alpha != 1.0):
        X1_new = nx.concatenate([alpha * X1, (1 - alpha) * F1], axis=1)
        X2_new = nx.concatenate([alpha * X2, (1 - alpha) * F2], axis=1)
        method_ = method[:-6]
    else:
        X1_new, X2_new = X1, X2
        method_ = method
    part1, rep_indices1 = get_partition_and_representants_samples(
        X1_new, npart1, method_, random_state, nx
    )
    part2, rep_indices2 = get_partition_and_representants_samples(
        X2_new, npart2, method_, random_state, nx
    )
    # format partitions over (C1, F1) and (C2, F2)

    if (F1 is None) and (F2 is None):
        CR1, list_R1, list_p1 = format_partitioned_samples(
            X1, p, part1, rep_indices1, nx=nx
        )
        CR2, list_R2, list_p2 = format_partitioned_samples(
            X2, q, part2, rep_indices2, nx=nx
        )

        MR = None
    else:
        CR1, list_R1, list_p1, FR1 = format_partitioned_samples(
            X1, p, part1, rep_indices1, F1, alpha, nx
        )
        CR2, list_R2, list_p2, FR2 = format_partitioned_samples(
            X2, q, part2, rep_indices2, F2, alpha, nx
        )

        MR = dist(FR1, FR2)

    # call to partitioned quantized fused gromov-wasserstein solver

    res = quantized_fused_gromov_wasserstein_partitioned(
        CR1,
        CR2,
        list_R1,
        list_R2,
        list_p1,
        list_p2,
        part1,
        part2,
        MR,
        alpha,
        build_OT=True,
        log=log,
        armijo=armijo,
        max_iter=max_iter,
        tol_rel=tol_rel,
        tol_abs=tol_abs,
        nx=nx,
        **kwargs,
    )

    if log:
        T_global, Ts_local, T, log_ = res

        C1 = dist(X1, X1)
        C2 = dist(X2, X2)

        # compute the transport cost on structures
        constC, hC1, hC2 = init_matrix(C1, C2, p, q, "square_loss", nx)
        structure_cost = gwloss(constC, hC1, hC2, T, nx)

        if alpha != 1.0:
            M = dist(F1, F2)
            feature_cost = nx.sum(M * T)
        else:
            feature_cost = 0.0

        log_["qFGW_dist"] = alpha * structure_cost + (1 - alpha) * feature_cost
        return T_global, Ts_local, T, log_

    else:
        T_global, Ts_local, T = res

        return T_global, Ts_local, T
