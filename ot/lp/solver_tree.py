from ..backend import get_backend
import numpy as np
from collections import deque

"""
Solver for the tree wasserstein distance
"""

# Author : Ali Boudjema


def topological_sort(tree):
    r"""
    Computes a topological order of the given tree

    Parameters
    -----------
    tree: array_like, shape(n)
        ancestor of each node in the tree (ancestor of root is root)
    """

    n = tree.shape[0]

    in_degree = np.zeros(n, dtype=int)

    for cur_node in range(n):
        if cur_node != tree[cur_node]:
            in_degree[tree[cur_node]] += 1

    queue = deque()

    for cur_node in range(n):
        if in_degree[cur_node] == 0:
            queue.append(cur_node)

    topo_order = []

    while queue:
        cur_node = queue.popleft()
        topo_order.append(cur_node)

        ancestor = tree[cur_node]

        if cur_node != ancestor:
            in_degree[ancestor] -= 1

            if in_degree[ancestor] == 0:
                queue.append(ancestor)

    return np.array(topo_order)


def tree_wasserstein_distance(
    tree, length, u_weights, v_weights, topo_order=None, return_plans=False
):
    r"""
    Computes the tree wasserstein distance for a given tree between two empirical distributions

    Parameters
    ----------
    tree : array_like, shape(n)
        parent of each node in the tree (parent of root is root)
    length : array_like, shape(n)
        length of the edge above each node (length of root is 0)
    u_weights : array_like, shape(n)
        weights of the first empirical distributions
    v_weights : array_like, shape(n)
        weights of the second empirical distributions
    topo_order : array_like, shape(n), optional
        topological order of the tree
    return_plans : bool, optional
        if True, returns the optimal transport plan between the
        two distributions, default is False

    Returns
    -------
    cost : float
        The tree wasserstein distance
    plans : coo_matrix, optional
        If return_plans is True, returns a coo_matrix containing the plan

    Reference
    ---------
    The proof of this algorithm uses the formula (3) in the article
    Tree-Sliced Variants of Wasserstein Distances
    """

    n = tree.shape[0]

    assert (
        n == length.shape[0] == u_weights.shape[0] == v_weights.shape[0]
    ), "dimension error in the input"

    if topo_order is None:
        topo_order = topological_sort(tree)

    nx = get_backend(tree, length, u_weights, v_weights)

    mass_dict = {}

    for cur in range(n):
        if u_weights[cur] != v_weights[cur]:
            mass_dict[cur] = {cur: u_weights[cur] - v_weights[cur]}
        else:
            mass_dict[cur] = {}

    source_plan = []
    sink_plan = []
    mass_plan = []

    virt_size = [len(mass_dict[k]) for k in range(n)]

    cost = 0

    depth = nx.zeros(n)

    for i in range(n - 2, -1, -1):
        cur_node = topo_order[i]
        depth[cur_node] = depth[tree[cur_node]] + length[cur_node]

    for cur in topo_order:
        dict_cur = mass_dict[cur]
        p = int(tree[cur])

        if cur != p:
            dict_p = mass_dict[p]

            if virt_size[cur] > virt_size[p]:
                mass_dict[cur], mass_dict[p] = dict_p, dict_cur
                dict_cur, dict_p = dict_p, dict_cur
                virt_size[cur], virt_size[p] = virt_size[p], virt_size[cur]

            while len(dict_cur) > 0 and len(dict_p) > 0:
                node_scur = next(iter(dict_cur))
                amount_scur = dict_cur[node_scur]

                node_sp = next(iter(dict_p))
                amount_sp = dict_p[node_sp]

                if (amount_scur > 0) != (amount_sp > 0):
                    match_amount = min(abs(amount_scur), abs(amount_sp))

                    source = node_scur if amount_scur > 0 else node_sp
                    sink = node_sp if amount_scur > 0 else node_scur

                    source_plan.append(source)
                    sink_plan.append(sink)
                    mass_plan.append(match_amount)

                    length_path = depth[source] + depth[sink] - 2 * depth[p]
                    cost = cost + match_amount * length_path

                    if amount_scur > 0:
                        dict_cur[node_scur] = dict_cur[node_scur] - match_amount
                        dict_p[node_sp] = dict_p[node_sp] + match_amount
                    else:
                        dict_cur[node_scur] = dict_cur[node_scur] + match_amount
                        dict_p[node_sp] = dict_p[node_sp] - match_amount

                    if dict_cur[node_scur] == 0:
                        del dict_cur[node_scur]

                    if dict_p[node_sp] == 0:
                        del dict_p[node_sp]

                else:
                    dict_p[node_scur] = amount_scur
                    del dict_cur[node_scur]

            if len(dict_p) == 0:
                mass_dict[cur], mass_dict[p] = dict_p, dict_cur
                dict_cur, dict_p = dict_p, dict_cur

            virt_size[p] += virt_size[cur]

    if mass_plan:
        mass_plan = nx.stack(mass_plan, axis=0)
    source_plan = nx.from_numpy(np.asarray(source_plan), type_as=length)
    sink_plan = nx.from_numpy(np.asarray(sink_plan), type_as=length)

    plans = nx.coo_matrix(
        mass_plan, source_plan, sink_plan, shape=(n, n), type_as=length
    )

    if return_plans:
        return cost, plans
    else:
        return cost
