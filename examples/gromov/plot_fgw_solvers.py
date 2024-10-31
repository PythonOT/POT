# -*- coding: utf-8 -*-
"""
==============================
Comparison of Fused Gromov-Wasserstein solvers
==============================

This example illustrates the computation of FGW for attributed graphs
using 4 different solvers to estimate the distance based on Conditional
Gradient [24], Sinkhorn projections [12, 51] and alternated Bregman
projections [63, 64].

We generate two graphs following Stochastic Block Models further endowed with
node features and compute their FGW matchings.

[12] Gabriel Peyré, Marco Cuturi, and Justin Solomon (2016),
"Gromov-Wasserstein averaging of kernel and distance matrices".
International Conference on Machine Learning (ICML).

[24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
and Courty Nicolas
"Optimal Transport for structured data with application on graphs"
International Conference on Machine Learning (ICML). 2019.

[51] Xu, H., Luo, D., Zha, H., & Duke, L. C. (2019).
"Gromov-wasserstein learning for graph matching and node embedding".
In International Conference on Machine Learning (ICML), 2019.

[63] Li, J., Tang, J., Kong, L., Liu, H., Li, J., So, A. M. C., & Blanchet, J.
"A Convergent Single-Loop Algorithm for Relaxation of Gromov-Wasserstein in
Graph Data". International Conference on Learning Representations (ICLR), 2023.

[64] Ma, X., Chu, X., Wang, Y., Lin, Y., Zhao, J., Ma, L., & Zhu, W.
"Fused Gromov-Wasserstein Graph Mixup for Graph-level Classifications".
In Thirty-seventh Conference on Neural Information Processing Systems
(NeurIPS), 2023.

"""

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 1

import numpy as np
import matplotlib.pylab as pl
from ot.gromov import (
    fused_gromov_wasserstein,
    entropic_fused_gromov_wasserstein,
    BAPG_fused_gromov_wasserstein,
)
import networkx
from networkx.generators.community import stochastic_block_model as sbm
from time import time

#############################################################################
#
# Generate two graphs following Stochastic Block models of 2 and 3 clusters.
# ---------------------------------------------
np.random.seed(0)

N2 = 20  # 2 communities
N3 = 30  # 3 communities
p2 = [[1.0, 0.1], [0.1, 0.9]]
p3 = [[1.0, 0.1, 0.0], [0.1, 0.95, 0.1], [0.0, 0.1, 0.9]]
G2 = sbm(seed=0, sizes=[N2 // 2, N2 // 2], p=p2)
G3 = sbm(seed=0, sizes=[N3 // 3, N3 // 3, N3 // 3], p=p3)
part_G2 = [G2.nodes[i]["block"] for i in range(N2)]
part_G3 = [G3.nodes[i]["block"] for i in range(N3)]

C2 = networkx.to_numpy_array(G2)
C3 = networkx.to_numpy_array(G3)


# We add node features with given mean - by clusters
# and inversely proportional to clusters' intra-connectivity

F2 = np.zeros((N2, 1))
for i, c in enumerate(part_G2):
    F2[i, 0] = np.random.normal(loc=c, scale=0.01)

F3 = np.zeros((N3, 1))
for i, c in enumerate(part_G3):
    F3[i, 0] = np.random.normal(loc=2.0 - c, scale=0.01)

# Compute pairwise euclidean distance between node features
M = (F2**2).dot(np.ones((1, N3))) + np.ones((N2, 1)).dot((F3**2).T) - 2 * F2.dot(F3.T)

h2 = np.ones(C2.shape[0]) / C2.shape[0]
h3 = np.ones(C3.shape[0]) / C3.shape[0]

#############################################################################
#
# Compute their Fused Gromov-Wasserstein distances
# ---------------------------------------------

alpha = 0.5


# Conditional Gradient algorithm
print("Conditional Gradient \n")
start_cg = time()
T_cg, log_cg = fused_gromov_wasserstein(
    M, C2, C3, h2, h3, "square_loss", alpha=alpha, tol_rel=1e-9, verbose=True, log=True
)
end_cg = time()
time_cg = 1000 * (end_cg - start_cg)

# Proximal Point algorithm with Kullback-Leibler as proximal operator
print("Proximal Point Algorithm \n")
start_ppa = time()
T_ppa, log_ppa = entropic_fused_gromov_wasserstein(
    M,
    C2,
    C3,
    h2,
    h3,
    "square_loss",
    alpha=alpha,
    epsilon=1.0,
    solver="PPA",
    tol=1e-9,
    log=True,
    verbose=True,
    warmstart=False,
    numItermax=10,
)
end_ppa = time()
time_ppa = 1000 * (end_ppa - start_ppa)

# Projected Gradient algorithm with entropic regularization
print("Projected Gradient Descent \n")
start_pgd = time()
T_pgd, log_pgd = entropic_fused_gromov_wasserstein(
    M,
    C2,
    C3,
    h2,
    h3,
    "square_loss",
    alpha=alpha,
    epsilon=0.01,
    solver="PGD",
    tol=1e-9,
    log=True,
    verbose=True,
    warmstart=False,
    numItermax=10,
)
end_pgd = time()
time_pgd = 1000 * (end_pgd - start_pgd)

# Alternated Bregman Projected Gradient algorithm with Kullback-Leibler as proximal operator
print("Bregman Alternated Projected Gradient \n")
start_bapg = time()
T_bapg, log_bapg = BAPG_fused_gromov_wasserstein(
    M,
    C2,
    C3,
    h2,
    h3,
    "square_loss",
    alpha=alpha,
    epsilon=1.0,
    tol=1e-9,
    marginal_loss=True,
    verbose=True,
    log=True,
)
end_bapg = time()
time_bapg = 1000 * (end_bapg - start_bapg)

print(
    "Fused Gromov-Wasserstein distance estimated with Conditional Gradient solver: "
    + str(log_cg["fgw_dist"])
)
print(
    "Fused Gromov-Wasserstein distance estimated with Proximal Point solver: "
    + str(log_ppa["fgw_dist"])
)
print(
    "Entropic Fused Gromov-Wasserstein distance estimated with Projected Gradient solver: "
    + str(log_pgd["fgw_dist"])
)
print(
    "Fused Gromov-Wasserstein distance estimated with Projected Gradient solver: "
    + str(log_bapg["fgw_dist"])
)

# compute OT sparsity level
T_cg_sparsity = 100 * (T_cg == 0.0).astype(np.float64).sum() / (N2 * N3)
T_ppa_sparsity = 100 * (T_ppa == 0.0).astype(np.float64).sum() / (N2 * N3)
T_pgd_sparsity = 100 * (T_pgd == 0.0).astype(np.float64).sum() / (N2 * N3)
T_bapg_sparsity = 100 * (T_bapg == 0.0).astype(np.float64).sum() / (N2 * N3)

# Methods using Sinkhorn/Bregman projections tend to produce feasibility errors on the
# marginal constraints

err_cg = np.linalg.norm(T_cg.sum(1) - h2) + np.linalg.norm(T_cg.sum(0) - h3)
err_ppa = np.linalg.norm(T_ppa.sum(1) - h2) + np.linalg.norm(T_ppa.sum(0) - h3)
err_pgd = np.linalg.norm(T_pgd.sum(1) - h2) + np.linalg.norm(T_pgd.sum(0) - h3)
err_bapg = np.linalg.norm(T_bapg.sum(1) - h2) + np.linalg.norm(T_bapg.sum(0) - h3)

#############################################################################
#
# Visualization of the Fused Gromov-Wasserstein matchings
# ---------------------------------------------
#
# We color nodes of the graph on the right - then project its node colors
# based on the optimal transport plan from the FGW matchings
# We adjust the intensity of links across domains proportionaly to the mass
# sent, adding a minimal intensity of 0.1 if mass sent is not zero.
# For each matching, all node sizes are proportionnal to their mass computed
# from marginals of the OT plan to illustrate potential feasibility errors.
# NB: colors refer to clusters - not to node features

# Add weights on the edges for visualization later on
weight_intra_G2 = 5
weight_inter_G2 = 0.5
weight_intra_G3 = 1.0
weight_inter_G3 = 1.5

weightedG2 = networkx.Graph()
part_G2 = [G2.nodes[i]["block"] for i in range(N2)]

for node in G2.nodes():
    weightedG2.add_node(node)
for i, j in G2.edges():
    if part_G2[i] == part_G2[j]:
        weightedG2.add_edge(i, j, weight=weight_intra_G2)
    else:
        weightedG2.add_edge(i, j, weight=weight_inter_G2)

weightedG3 = networkx.Graph()
part_G3 = [G3.nodes[i]["block"] for i in range(N3)]

for node in G3.nodes():
    weightedG3.add_node(node)
for i, j in G3.edges():
    if part_G3[i] == part_G3[j]:
        weightedG3.add_edge(i, j, weight=weight_intra_G3)
    else:
        weightedG3.add_edge(i, j, weight=weight_inter_G3)


def draw_graph(
    G,
    C,
    nodes_color_part,
    Gweights=None,
    pos=None,
    edge_color="black",
    node_size=None,
    shiftx=0,
    seed=0,
):
    if pos is None:
        pos = networkx.spring_layout(G, scale=1.0, seed=seed)

    if shiftx != 0:
        for k, v in pos.items():
            v[0] = v[0] + shiftx

    alpha_edge = 0.7
    width_edge = 1.8
    if Gweights is None:
        networkx.draw_networkx_edges(
            G, pos, width=width_edge, alpha=alpha_edge, edge_color=edge_color
        )
    else:
        # We make more visible connections between activated nodes
        n = len(Gweights)
        edgelist_activated = []
        edgelist_deactivated = []
        for i in range(n):
            for j in range(n):
                if Gweights[i] * Gweights[j] * C[i, j] > 0:
                    edgelist_activated.append((i, j))
                elif C[i, j] > 0:
                    edgelist_deactivated.append((i, j))

        networkx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist_activated,
            width=width_edge,
            alpha=alpha_edge,
            edge_color=edge_color,
        )
        networkx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist_deactivated,
            width=width_edge,
            alpha=0.1,
            edge_color=edge_color,
        )

    if Gweights is None:
        for node, node_color in enumerate(nodes_color_part):
            networkx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                node_size=node_size,
                alpha=1,
                node_color=node_color,
            )
    else:
        scaled_Gweights = Gweights / (0.5 * Gweights.max())
        nodes_size = node_size * scaled_Gweights
        for node, node_color in enumerate(nodes_color_part):
            networkx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                node_size=nodes_size[node],
                alpha=1,
                node_color=node_color,
            )
    return pos


def draw_transp_colored_GW(
    G1,
    C1,
    G2,
    C2,
    part_G1,
    p1,
    p2,
    T,
    pos1=None,
    pos2=None,
    shiftx=4,
    switchx=False,
    node_size=70,
    seed_G1=0,
    seed_G2=0,
):
    starting_color = 0
    # get graphs partition and their coloring
    part1 = part_G1.copy()
    unique_colors = ["C%s" % (starting_color + i) for i in np.unique(part1)]
    nodes_color_part1 = []
    for cluster in part1:
        nodes_color_part1.append(unique_colors[cluster])

    nodes_color_part2 = []
    # T: getting colors assignment from argmin of columns
    for i in range(len(G2.nodes())):
        j = np.argmax(T[:, i])
        nodes_color_part2.append(nodes_color_part1[j])
    pos1 = draw_graph(
        G1,
        C1,
        nodes_color_part1,
        Gweights=p1,
        pos=pos1,
        node_size=node_size,
        shiftx=0,
        seed=seed_G1,
    )
    pos2 = draw_graph(
        G2,
        C2,
        nodes_color_part2,
        Gweights=p2,
        pos=pos2,
        node_size=node_size,
        shiftx=shiftx,
        seed=seed_G2,
    )

    for k1, v1 in pos1.items():
        max_Tk1 = np.max(T[k1, :])
        for k2, v2 in pos2.items():
            if T[k1, k2] > 0:
                pl.plot(
                    [pos1[k1][0], pos2[k2][0]],
                    [pos1[k1][1], pos2[k2][1]],
                    "-",
                    lw=0.7,
                    alpha=min(T[k1, k2] / max_Tk1 + 0.1, 1.0),
                    color=nodes_color_part1[k1],
                )
    return pos1, pos2


node_size = 40
fontsize = 13
seed_G2 = 0
seed_G3 = 4

pl.figure(2, figsize=(15, 3.5))
pl.clf()
pl.subplot(141)
pl.axis("off")

pl.title(
    "(CG) FGW=%s\n \n OT sparsity = %s \n marg. error = %s \n runtime = %s"
    % (
        np.round(log_cg["fgw_dist"], 3),
        str(np.round(T_cg_sparsity, 2)) + " %",
        np.round(err_cg, 4),
        str(np.round(time_cg, 2)) + " ms",
    ),
    fontsize=fontsize,
)

pos1, pos2 = draw_transp_colored_GW(
    weightedG2,
    C2,
    weightedG3,
    C3,
    part_G2,
    p1=T_cg.sum(1),
    p2=T_cg.sum(0),
    T=T_cg,
    shiftx=1.5,
    node_size=node_size,
    seed_G1=seed_G2,
    seed_G2=seed_G3,
)

pl.subplot(142)
pl.axis("off")

pl.title(
    "(PPA) FGW=%s\n \n OT sparsity = %s \n marg. error = %s \n runtime = %s"
    % (
        np.round(log_ppa["fgw_dist"], 3),
        str(np.round(T_ppa_sparsity, 2)) + " %",
        np.round(err_ppa, 4),
        str(np.round(time_ppa, 2)) + " ms",
    ),
    fontsize=fontsize,
)

pos1, pos2 = draw_transp_colored_GW(
    weightedG2,
    C2,
    weightedG3,
    C3,
    part_G2,
    p1=T_ppa.sum(1),
    p2=T_ppa.sum(0),
    T=T_ppa,
    pos1=pos1,
    pos2=pos2,
    shiftx=0.0,
    node_size=node_size,
    seed_G1=0,
    seed_G2=0,
)

pl.subplot(143)
pl.axis("off")

pl.title(
    "(PGD) Entropic FGW=%s\n \n OT sparsity = %s \n marg. error = %s \n runtime = %s"
    % (
        np.round(log_pgd["fgw_dist"], 3),
        str(np.round(T_pgd_sparsity, 2)) + " %",
        np.round(err_pgd, 4),
        str(np.round(time_pgd, 2)) + " ms",
    ),
    fontsize=fontsize,
)

pos1, pos2 = draw_transp_colored_GW(
    weightedG2,
    C2,
    weightedG3,
    C3,
    part_G2,
    p1=T_pgd.sum(1),
    p2=T_pgd.sum(0),
    T=T_pgd,
    pos1=pos1,
    pos2=pos2,
    shiftx=0.0,
    node_size=node_size,
    seed_G1=0,
    seed_G2=0,
)


pl.subplot(144)
pl.axis("off")

pl.title(
    "(BAPG) FGW=%s\n \n OT sparsity = %s \n marg. error = %s \n runtime = %s"
    % (
        np.round(log_bapg["fgw_dist"], 3),
        str(np.round(T_bapg_sparsity, 2)) + " %",
        np.round(err_bapg, 4),
        str(np.round(time_bapg, 2)) + " ms",
    ),
    fontsize=fontsize,
)

pos1, pos2 = draw_transp_colored_GW(
    weightedG2,
    C2,
    weightedG3,
    C3,
    part_G2,
    p1=T_bapg.sum(1),
    p2=T_bapg.sum(0),
    T=T_bapg,
    pos1=pos1,
    pos2=pos2,
    shiftx=0.0,
    node_size=node_size,
    seed_G1=0,
    seed_G2=0,
)

pl.tight_layout()

pl.show()
