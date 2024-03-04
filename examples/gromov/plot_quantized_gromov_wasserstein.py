# -*- coding: utf-8 -*-
"""
===============================================
Quantized Gromov-Wasserstein example
===============================================

This example is designed to show how to use the quantized Gromov-Wasserstein
solvers [66]. POT provides a wrapper `quantized_gromov_wasserstein` operating other
graphs, and a generic solver `quantized_gromov_wasserstein_partitioned` that allows
the user to precompute any partitioning and representant selection methods.

We generate two graphs following Stochastic Block Models encoded as shortest path
matrices. Then show how to compute their quantized gromov-wasserstein
matchings using both solvers.

[66] Chowdhury, S., Miller, D., & Needham, T. (2021). Quantized gromov-wasserstein.
ECML PKDD 2021. Springer International Publishing.
"""

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 1

import numpy as np
import matplotlib.pylab as pl
import networkx
from networkx.generators.community import stochastic_block_model as sbm
from scipy.sparse.csgraph import shortest_path

from ot.gromov import quantized_gromov_wasserstein, quantized_gromov_wasserstein_partitioned
from ot.gromov._quantized import _get_partition, _get_representants, _formate_partitioned_graph

#############################################################################
#
# Generate two graphs following Stochastic Block models of 2 and 3 clusters.
# --------------------------------------------------------------------------


N2 = 30  # 2 communities
N3 = 45  # 3 communities
p2 = [[0.8, 0.1],
      [0.1, 0.7]]
p3 = [[0.8, 0.1, 0.],
      [0.1, 0.75, 0.1],
      [0., 0.1, 0.7]]
G2 = sbm(seed=0, sizes=[N2 // 2, N2 // 2], p=p2)
G3 = sbm(seed=0, sizes=[N3 // 3, N3 // 3, N3 // 3], p=p3)


C2 = networkx.to_numpy_array(G2)
C3 = networkx.to_numpy_array(G3)

spC2 = shortest_path(C2)
spC3 = shortest_path(C3)

h2 = np.ones(C2.shape[0]) / C2.shape[0]
h3 = np.ones(C3.shape[0]) / C3.shape[0]

# Add weights on the edges for visualization later on
weight_intra_G2 = 5
weight_inter_G2 = 0.5
weight_intra_G3 = 1.
weight_inter_G3 = 1.5

weightedG2 = networkx.Graph()
part_G2 = [G2.nodes[i]['block'] for i in range(N2)]

for node in G2.nodes():
    weightedG2.add_node(node)
for i, j in G2.edges():
    if part_G2[i] == part_G2[j]:
        weightedG2.add_edge(i, j, weight=weight_intra_G2)
    else:
        weightedG2.add_edge(i, j, weight=weight_inter_G2)

weightedG3 = networkx.Graph()
part_G3 = [G3.nodes[i]['block'] for i in range(N3)]

for node in G3.nodes():
    weightedG3.add_node(node)
for i, j in G3.edges():
    if part_G3[i] == part_G3[j]:
        weightedG3.add_edge(i, j, weight=weight_intra_G3)
    else:
        weightedG3.add_edge(i, j, weight=weight_inter_G3)

#############################################################################
#
# Compute their quantized Gromov-Wasserstein distance using the wrapper
# ---------------------------------------------------------

# 0) qGW(spC2, h2, spC3, h3) while partitioning the adjacency matrices C2 and C3
#   in 2 and 3 clusters respectively, using the Fluid algorithm and selecting
#   representant in each partition using maximal pagerank.
#   Notice that C2 and C3 are optional and if they are not specified these
#   pre-processing algorithms will be applied to spC2 and spC3.

part_method = 'louvain'
rep_method = 'pagerank'
OT_global, OTs_local, OT, log = quantized_gromov_wasserstein(
    spC2, spC3, 2, 3, C2, C3, h2, h3, part_method=part_method,
    rep_method=rep_method, log=True)

qGW_dist = log['qGW_dist']

#############################################################################
#
# Compute their quantized Gromov-Wasserstein distance using any partitioning and representant selection methods
# ---------------------------------------------------------

# 1-a) Partition C2 and C3 in 2 and 3 clusters respectively using the Fluid
#    algorithm implementation from networkx. Encode these partitions via vectors of assignments.

part2 = _get_partition(C2, npart=2, part_method=part_method)
part3 = _get_partition(C3, npart=3, part_method=part_method)

# 1-b) Select representant in each partition using the Pagerank algorithm
#     implementation from networkx.

rep_indices2 = _get_representants(C2, part2, rep_method=rep_method)
rep_indices3 = _get_representants(C3, part3, rep_method=rep_method)

# 1-c) Formate partitions. CR (2, 2) relations between representants in each space.
# list_R contain relations between samples and representants within each partition.
# list_h contain samples relative importance within each partition.

CR2, list_R2, list_h2 = _formate_partitioned_graph(spC2, h2, part2, rep_indices2)
CR3, list_R3, list_h3 = _formate_partitioned_graph(spC3, h3, part3, rep_indices3)

# 1-d) call to partitioned quantized gromov-wasserstein solver

OT_global_, OTs_local_, OT_, log_ = quantized_gromov_wasserstein_partitioned(
    CR2, CR3, list_R2, list_R3, list_h2, list_h3, build_OT=True, log=True)


#############################################################################
#
# Visualization of the quantized Gromov-Wasserstein matching
# --------------------------------------------------------------
#
# We color nodes of the graph based on the respective partition of each graph.
# On the first plot we illustrate the qGW matching between both shortest path matrices.
# While the GW matching across representants of each space is illustrated on the right.


def draw_graph(G, C, nodes_color_part, rep_indices, pos=None,
               edge_color='black', alpha_edge=0.7, node_size=None,
               shiftx=0, seed=0, highlight_rep=False):

    if (pos is None):
        pos = networkx.spring_layout(G, scale=1., seed=seed)

    if shiftx != 0:
        for k, v in pos.items():
            v[0] = v[0] + shiftx

    width_edge = 1.5

    if not highlight_rep:
        networkx.draw_networkx_edges(
            G, pos, width=width_edge, alpha=alpha_edge, edge_color=edge_color)
    else:
        for edge in G.edges:
            if (edge[0] in rep_indices) and (edge[1] in rep_indices):
                networkx.draw_networkx_edges(
                    G, pos, edgelist=[edge], width=width_edge, alpha=alpha_edge,
                    edge_color=edge_color)
            else:
                networkx.draw_networkx_edges(
                    G, pos, edgelist=[edge], width=width_edge, alpha=0.2,
                    edge_color=edge_color)

    for node, node_color in enumerate(nodes_color_part):
        local_node_shape, local_node_size = 'o', node_size
        if node in rep_indices:
            local_node_shape, local_node_size = '*', 6 * node_size

        alpha = 0.9
        if highlight_rep:
            alpha = 0.9 if node in rep_indices else 0.2

        networkx.draw_networkx_nodes(G, pos, nodelist=[node], alpha=alpha,
                                     node_shape=local_node_shape,
                                     node_size=local_node_size,
                                     node_color=node_color)

    return pos


def draw_transp_colored_qGW(
        G1, C1, G2, C2, part1, part2, rep_indices1, rep_indices2, T,
        pos1=None, pos2=None, shiftx=4, switchx=False, node_size=70,
        seed_G1=0, seed_G2=0, highlight_rep=False):
    starting_color = 0
    # get graphs partition and their coloring
    unique_colors1 = ['C%s' % (starting_color + i) for i in np.unique(part1)]
    nodes_color_part1 = []
    for cluster in part1:
        nodes_color_part1.append(unique_colors1[cluster])

    starting_color = len(unique_colors1) + 1
    unique_colors2 = ['C%s' % (starting_color + i) for i in np.unique(part2)]
    nodes_color_part2 = []
    for cluster in part2:
        nodes_color_part2.append(unique_colors2[cluster])

    pos1 = draw_graph(
        G1, C1, nodes_color_part1, rep_indices1, pos=pos1, node_size=node_size,
        shiftx=0, seed=seed_G1, highlight_rep=highlight_rep)
    pos2 = draw_graph(
        G2, C2, nodes_color_part2, rep_indices2, pos=pos2, node_size=node_size,
        shiftx=shiftx, seed=seed_G2, highlight_rep=highlight_rep)

    if not highlight_rep:
        for k1, v1 in pos1.items():
            max_Tk1 = np.max(T[k1, :])
            for k2, v2 in pos2.items():
                if (T[k1, k2] > 0):
                    pl.plot([pos1[k1][0], pos2[k2][0]],
                            [pos1[k1][1], pos2[k2][1]],
                            '-', lw=0.7, alpha=T[k1, k2] / max_Tk1,
                            color=nodes_color_part1[k1])

    else:  # OT is only between representants
        for id1, node_id1 in enumerate(rep_indices1):
            max_Tk1 = np.max(T[id1, :])
            for id2, node_id2 in enumerate(rep_indices2):
                if (T[id1, id2] > 0):
                    pl.plot([pos1[node_id1][0], pos2[node_id2][0]],
                            [pos1[node_id1][1], pos2[node_id2][1]],
                            '-', lw=0.8, alpha=T[id1, id2] / max_Tk1,
                            color=nodes_color_part1[node_id1])
    return pos1, pos2


node_size = 40
fontsize = 10
seed_G2 = 0
seed_G3 = 3

part2_ = part2.astype(np.int32)
part3_ = part3.astype(np.int32)

pl.figure(1, figsize=(8, 3))
pl.clf()
pl.axis('off')
pl.subplot(1, 2, 1)
pl.title(r'qGW$(\mathbf{spC_2}, \mathbf{spC_3}) =%s$' % (np.round(qGW_dist, 3)), fontsize=fontsize)

pos1, pos2 = draw_transp_colored_qGW(
    weightedG2, C2, weightedG3, C3, part2_, part3_, rep_indices2, rep_indices3,
    T=OT_, shiftx=1.5, node_size=node_size, seed_G1=seed_G2, seed_G2=seed_G3)

pl.tight_layout()

pl.subplot(1, 2, 2)
pl.title(r' GW$(\mathbf{CR_2}, \mathbf{CR_3}) =%s$' % (np.round(log_['gw_dist_CR'], 3)), fontsize=fontsize)

pos1, pos2 = draw_transp_colored_qGW(
    weightedG2, C2, weightedG3, C3, part2_, part3_, rep_indices2, rep_indices3,
    T=OT_global, shiftx=1.5, node_size=node_size, seed_G1=seed_G2, seed_G2=seed_G3,
    highlight_rep=True)

pl.tight_layout()
pl.show()
