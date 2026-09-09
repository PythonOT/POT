from ..backend import get_backend
import numpy as np
from ..utils import proj_simplex
from ..utils import list_to_array

from .solver_tree import topological_sort
from .solver_tree import tree_wasserstein_distance

# Author : Ali Boudjema


def get_B_matrix(tree, length, nb_leafs):
    nx = get_backend(length)

    rows = []
    col = []
    data = []

    for cur_leaf in range(nb_leafs):
        cur_node = cur_leaf

        while cur_node != tree[cur_node]:
            rows.append(cur_node)
            col.append(cur_leaf)
            data.append(length[cur_node])
            cur_node = tree[cur_node]

    nb_rows = len(tree)
    nb_col = nb_leafs

    data = list_to_array(data, nx=nx)
    rows = list_to_array(rows, nx=nx)
    col = list_to_array(col, nx=nx)

    B = nx.coo_matrix(data, rows, col, shape=(nb_rows, nb_col), type_as=length)

    return B


def get_gradient(cur_B, B_mes_sorted, B, nb_mes, nb_nodes, nx):
    idx = nx.zeros(nb_nodes, type_as=cur_B)

    for node in range(nb_nodes):
        idx[node] = nx.searchsorted(B_mes_sorted[:, node], cur_B[node]) + 1

    z = -nb_mes + 2 * idx - 2

    g = nx.transpose(B) @ z
    g /= nb_mes

    return g


def pre_process_trees(tree_list, length_list, measures):
    nx = get_backend(length_list, measures)

    nb_leafs = measures.shape[2]

    prepared_trees = []

    for tree, length, mes in zip(tree_list, length_list, measures):
        B = get_B_matrix(tree, length, nb_leafs)
        nb_mes = mes.shape[0]

        B_mes_list = []

        for id_mes in range(nb_mes):
            col = nx.reshape(mes[id_mes], (-1, 1))

            res = B @ col

            res_dense = nx.todense(res)

            res_1d = nx.reshape(res_dense, (-1,))

            B_mes_list.append(res_1d)

        B_mes = nx.stack(B_mes_list, axis=0)

        B_mes_sorted = nx.sort(B_mes, axis=0)

        prepared_trees.append(
            {
                "B": B,
                "B_mes_sorted": B_mes_sorted,
                "nb_nodes": tree.shape[0],
                "nb_mes": nb_mes,
            }
        )

    return prepared_trees


def fixed_support_tree_barycenter(
    tree_list, length_list, measures, nb_itr=100, step=0.01, tol=1e-5, init_measure=None
):
    """
    Computes the Tree-Wasserstein (or Tree-Sliced) barycenter for one or multiple trees,
    with the constraint that the support of the barycenter is fixed at the leaves.
    It is assumed that the leaves correspond to the first nodes of the tree (indices 0 to k-1).
    While the number of leaves (k) must be strictly identical across all structures to ensure
    consistent alignment, the total number of nodes (leaves + internal nodes) can
    freely vary from one tree to another.

    If a single tree structure is provided (e.g., 1D arrays for tree_list/length_list
    and 2D for measures), the function automatically expands their dimensions to 3D
    internal structures to handle them uniformly as a multi-tree setting with t=1.

    Parameters
    -----------
    tree_list : array_like
        A single tree of shape (n_t,) or an array of t trees where each tree has shape (n_t,).
        n_t is the total number of nodes in that specific tree. tree[i] contains the index
        of the parent of node i (with tree[root] == root).
    length_list : array_like
        The edge weights corresponding to tree_list. A single array of shape (n_t,) or an array
        of t arrays, where the t-th array has shape (n_t,) and contains the length of the edge
        connecting node i to its parent.
    measures : array_like, shape (t, m, k) or (m,k)
        The input probability distributions mapped to the leaves.
        k is the fixed number of leaves shared by all trees, and m is the number
        of measures. Accepts a 2D array of shape (m, k) for a single tree, or a 3D array
        of shape (t, m, k) in a multi-tree setting.
    nb_itr : int, optional
        the maximal number of iterations for the subgradient descent
    step : float, optional
        the step size of the descent
    tol : float, optional
        Convergence tolerance. The descent stops if the L2 norm of the difference
        between two consecutive iterations is smaller than tol.
    init_measure : array_like, shape (k), optional
        The starting point of the descent, default is None

    Returns
    -------
    cur_mes : array_like, shape (k)
        The computed fixed-support barycenter supported on the k leaves.

    References
    ----------
    "Fixed Support Tree-Sliced Wasserstein Barycenter"
    """
    nx = get_backend(tree_list, length_list, measures)

    if tree_list.ndim == 1:
        tree_list = np.reshape(tree_list, (1, *tree_list.shape))
        length_list = nx.reshape(length_list, (1, *length_list.shape))

    if measures.ndim == 2:
        measures = nx.reshape(measures, (1, *measures.shape))
        measures = nx.tile(measures, (tree_list.shape[0], 1, 1))

    assert (
        tree_list.shape[0] == length_list.shape[0] == measures.shape[0]
        and tree_list.shape[1] == length_list.shape[1]
    ), "dimension error in the input"

    prepared_trees = pre_process_trees(tree_list, length_list, measures)

    nb_leafs = measures.shape[2]

    if init_measure is None:
        cur_mes = nx.ones(nb_leafs) / nb_leafs
    else:
        cur_mes = init_measure

    nb_tree = len(prepared_trees)

    for itr in range(nb_itr):
        old_mes = nx.copy(cur_mes)
        g_total = nx.zeros(nb_leafs)

        for tree_data in prepared_trees:
            B = tree_data["B"]
            B_mes_sorted = tree_data["B_mes_sorted"]
            nb_nodes = tree_data["nb_nodes"]
            nb_mes = tree_data["nb_mes"]

            cur_B = B @ cur_mes

            g_tree = get_gradient(cur_B, B_mes_sorted, B, nb_mes, nb_nodes, nx)
            g_total += g_tree

        g_mean = g_total / nb_tree

        g_mean /= nx.norm(g_mean) + 1e-12

        cur_mes -= step * g_mean

        cur_mes = proj_simplex(cur_mes)

        if np.linalg.norm(cur_mes - old_mes) < tol:
            break

    return cur_mes


def free_support_tree_barycenter(
    tree, length, measures, nb_itr=100, step=0.01, weights=None
):
    """Computes the tree wasserstein barycenter for a tree with no constraints on the
    support of the measures

    Parameters
    ----------
    tree : array_like
        A tree of shape (n), the number of nodes. tree[i] contains the index
        of the parent of node i (with tree[root] == root).
    length : array_like
        The edge weights corresponding to tree. A single array of shape (n_t) containing
        the length of the edge connecting node i to its parent.
    measures : array_like, shape (m,n)
        The input probability distributions mapped to the nodes.
    nb_itr : int, optional
        the maximal number of iterations for the subgradient descent
    step : float, optional
        the step size of the descent
    weights : array_like, shape(m), optional
        The weight of each measure, set to uniform if none
    """

    import torch

    nb_nodes = tree.shape[0]

    barycenter = torch.ones(nb_nodes) / nb_nodes
    topo_order = topological_sort(tree)

    tree = np.asarray(tree)
    tree = torch.from_numpy(tree)

    length = np.asarray(length)
    length = torch.from_numpy(length)

    measures = np.asarray(measures)
    measures = torch.from_numpy(measures)

    nb_mes = measures.shape[0]

    if weights is None:
        w = torch.ones(nb_mes) / nb_mes
    else:
        w = weights

    for itr in range(nb_itr):
        barycenter.requires_grad_(True)

        loss = sum(
            cur_w
            * tree_wasserstein_distance(tree, length, cur_mes, barycenter, topo_order)
            for cur_w, cur_mes in zip(w, measures)
        )

        loss.backward()

        grad = barycenter.grad

        if grad is None or torch.norm(grad) < 1e-12:
            barycenter = barycenter.detach()
            break

        with torch.no_grad():
            scaled_grad = step * grad
            scaled_grad = scaled_grad - torch.max(scaled_grad)

            barycenter_next = barycenter * torch.exp(-scaled_grad)

            barycenter = barycenter_next / (torch.sum(barycenter_next) + 1e-15)

        barycenter = barycenter.detach()

    return barycenter
