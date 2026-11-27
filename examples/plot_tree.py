"""
========================================
Tree wasserstein distance and barycenter
========================================

Illustrates the use of the tree wasserstein distance calculator, and barycenter solvers
for both the fixed-support and the free support versions
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ot
import torch

import networkx as netx
import random
import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F

# Utilitaries functions

norm = colors.SymLogNorm(linthresh=1e-3, vmin=0.0, vmax=1.0)


def tree_to_networkx(tree, length):
    G = netx.Graph()

    nb_nodes = tree.shape[0]

    root = -1

    for i in range(nb_nodes):
        parent = int(tree[i])
        if i != parent:
            G.add_edge(i, parent, weight=length[i])
        else:
            root = i

    return G, root


# Source - https://stackoverflow.com/a/29597209
# Posted by Joel, modified by community. See post 'Timeline' for change history
# Retrieved 2026-07-16, License - CC BY-SA 4.0
def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """
    if not netx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, netx.DiGraph):
            root = next(
                iter(netx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
        G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, netx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def print_tree(tree, length, measure=None, ax=None):
    G, root = tree_to_networkx(tree, length)

    pos = hierarchy_pos(G, root=root)

    nb_nodes = tree.shape[0]

    if measure is None:
        measure_color = [1 / nb_nodes for _ in range(nb_nodes)]
    else:
        measure_color = [measure[node] for node in G.nodes()]

    netx.draw_networkx_edges(G, pos=pos, ax=ax)
    netx.draw_networkx_labels(G, pos=pos, ax=ax, font_color="white")

    measure_color = norm(measure_color)

    nodes = netx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_color=measure_color,
        cmap=plt.cm.inferno,
        node_size=300,
        vmin=0.0,
        vmax=1.0,
    )

    return nodes


# Gradient descent to transport a measure into an other
n = 7
m = 7

s = ot.datasets.make_2D_samples_gauss(n, [0.0, 0.0], [[1.0, 0], [0, -1.0]], 0)
c = ot.datasets.make_2D_samples_gauss(m, [10.0, 10.0], [[1.0, 0], [0, -1.0]], 0)

source_point = torch.tensor(s, dtype=torch.float32)
target_point = torch.tensor(c, dtype=torch.float32)

points = torch.cat([source_point, target_point])

tree, length = ot.utils.random_tree(points, 30)

source_mes = torch.zeros(n + m)
source_mes[:n] = torch.ones(n) / n

source_mes_init = source_mes.clone()

target_mes = torch.zeros(n + m)
target_mes[n:] = torch.ones(m) / m

source_mes = source_mes.detach().requires_grad_(True)

step = 0.001

for i in range(1000):
    loss = ot.lp.tree_wasserstein_distance(tree, length, source_mes, target_mes)

    loss.backward()

    with torch.no_grad():
        sourceNouv = source_mes - step * source_mes.grad
        sourceNouv = ot.utils.proj_simplex(sourceNouv.cpu().numpy())

    source_mes = torch.tensor(sourceNouv)
    source_mes.requires_grad_(True)

source_mes = source_mes.detach().numpy()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 7))

print_tree(tree, length, source_mes_init, ax1)
ax1.set_title("Source measure")

print_tree(tree, length, target_mes, ax2)
ax2.set_title("Target measure")

last_nodes = print_tree(tree, length, source_mes, ax3)
ax3.set_title("Final measure")

plt.suptitle(
    "Gradient descent to minimize the TWD between source measure and target measure"
)

real_values = [
    0.0,
    0.001,
    0.01,
    0.02,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
]

transformed_values = norm(real_values)

cbar = fig.colorbar(
    last_nodes,
    ax=[ax1, ax2, ax3],
    orientation="horizontal",
    shrink=0.7,
    pad=0.03,
    ticks=transformed_values,
)
cbar.ax.set_xticklabels([f"{v:.3f}" for v in real_values])
cbar.set_label("Probability mass", labelpad=15)

plt.show()

# Fixed-support barycenter

n = 10
k = 5
l = 3

tree, length = ot.utils.random_tree_fixed_leaves(n, k, np, 0)

source_mes = np.random.rand(l)
source_mes = np.pad(source_mes, (0, k - l))
source_mes /= np.sum(source_mes)

target_mes = np.random.rand(k - l)
target_mes = np.pad(target_mes, (l, 0))
target_mes /= np.sum(target_mes)

barycenter = ot.lp.fixed_support_tree_barycenter(
    tree, length, np.stack([source_mes, target_mes])
)

source_mes = np.pad(source_mes, (0, n - k))
target_mes = np.pad(target_mes, (0, n - k))
barycenter = np.pad(barycenter, (0, n - k))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 7))

print_tree(tree, length, source_mes, ax1)
ax1.set_title("Measure 1")

print_tree(tree, length, target_mes, ax2)
ax2.set_title("Measure 2")

last_nodes = print_tree(tree, length, barycenter, ax3)
ax3.set_title("Barycenter")

plt.suptitle("Fixed support barycenter")

cbar = fig.colorbar(
    last_nodes,
    ax=[ax1, ax2, ax3],
    orientation="horizontal",
    shrink=0.7,
    pad=0.03,
    ticks=transformed_values,
)
cbar.ax.set_xticklabels([f"{v:.3f}" for v in real_values])
cbar.set_label("Probability mass", labelpad=15)

plt.show()

# Free support barycenter
n = 7
m = 7

s = ot.datasets.make_2D_samples_gauss(n, [0.0, 0.0], [[1.0, 0], [0, -1.0]], 0)
c = ot.datasets.make_2D_samples_gauss(m, [10.0, 10.0], [[1.0, 0], [0, -1.0]], 0)

source_point = torch.tensor(s, dtype=torch.float32)
target_point = torch.tensor(c, dtype=torch.float32)

points = torch.cat([source_point, target_point])

tree, length = ot.utils.random_tree(points, 30)

source_mes = torch.zeros(n + m)
target_mes = torch.zeros(n + m)
source_mes[12] = 1.0
target_mes[11] = 1.0

barycenter = ot.lp.free_support_tree_barycenter(
    tree, length, torch.stack([source_mes, target_mes]), 1000, 5e-1
)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 7))

print_tree(tree, length, source_mes, ax1)
ax1.set_title("Measure 1")

print_tree(tree, length, target_mes, ax2)
ax2.set_title("Measure 2")

last_nodes = print_tree(tree, length, barycenter, ax3)
ax3.set_title("Barycenter")

plt.suptitle("Free support barycenter")

cbar = fig.colorbar(
    last_nodes,
    ax=[ax1, ax2, ax3],
    orientation="horizontal",
    shrink=0.7,
    pad=0.03,
    ticks=transformed_values,
)
cbar.ax.set_xticklabels([f"{v:.3f}" for v in real_values])
cbar.set_label("Probability mass", labelpad=15)

plt.show()
