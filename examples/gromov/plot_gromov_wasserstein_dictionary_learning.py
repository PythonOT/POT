# -*- coding: utf-8 -*-
"""
==========================
Gromov-Wasserstein Dictionary Learning example
==========================

This example is designed to show how to perform a (Fused) Gromov-Wasserstein
linear dictionary learning in POT.
"""

# Author: Cédric Vincent-Cuaz <cedric.vincent-cuaz@inria.fr>
#
# License: MIT License

import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from ot.gromov import gromov_wasserstein_linear_unmixing, gromov_wasserstein_dictionary_learning, fused_gromov_wasserstein_linear_unmixing, fused_gromov_wasserstein_dictionary_learning
import ot
import networkx
from networkx.generators.community import stochastic_block_model as sbm
# %%
# =============================================================================
# Generate a dataset composed of graphs following Stochastic Block models of 1, 2 and 3 clusters.
# =============================================================================

np.random.seed(42)

N = 150  # number of graphs in the dataset
clusters = [1, 2, 3]  # for every quantity of clusters, 50 graphs following SBM will be generated with variable connectivity matrix
nlabels = len(clusters)
dataset = []
labels = []
cluster_assignments = []  # used below for toy exemple on attributed graphs
for n_cluster in clusters:
    for i in range(N // len(clusters)):
        n_nodes = int(np.random.uniform(low=25, high=50))
        p_inter = np.random.uniform(low=0.1, high=0.2)
        p_intra = np.random.uniform(low=0.8, high=0.9)

        if n_cluster > 1:
            P = p_inter * np.ones((n_cluster, n_cluster))
            np.fill_diagonal(P, p_intra)
        else:
            P = p_intra * np.eye(1)
        sizes = np.round(n_nodes * np.ones(n_cluster) / n_cluster).astype(np.int32)
        G = sbm(sizes, P, seed=i, directed=False)
        C = networkx.to_numpy_array(G)
        cluster_assignment = [G.nodes[i]['block'] for i in G.nodes]
        dataset.append(C)
        labels.append(n_cluster)
        cluster_assignments.append(np.array(cluster_assignment))
# Visualize samples


def plot_graph(x, C, binary=True, color='C0', s=None):
    for j in range(C.shape[0]):
        for i in range(j):
            if binary:
                if C[i, j] > 0:
                    pl.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], alpha=0.2, color='k')
            else:  # connection intensity proportional to C[i,j]
                pl.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], alpha=C[i, j], color='k')

    pl.scatter(x[:, 0], x[:, 1], c=color, s=s, zorder=10, edgecolors='k', cmap='tab10', vmax=9)


pl.figure(1, (15, 10))
pl.clf()
pl.suptitle('Samples from the graph dataset')
for idx_c, c in enumerate(clusters):
    C = dataset[(c - 1) * 50]  # sample with c clusters
    # get 2d position for nodes
    x = MDS(dissimilarity='precomputed', random_state=0).fit_transform(1 - C)
    pl.subplot(2, nlabels, c)
    pl.title('(graph) sample from label ' + str(c))
    plot_graph(x, C, binary=True, color='C0')
    pl.axis("off")
    pl.subplot(2, nlabels, nlabels + c)
    pl.title('(adjacency matrix) sample from label ' + str(c))
    pl.imshow(C, interpolation='nearest')
    pl.axis("off")
pl.show()

# %%
# =============================================================================
# Infer the gromov-wasserstein dictionary from the dataset
# =============================================================================

ps = [ot.unif(C.shape[0]) for C in dataset]
D = 3
nt = 6
q = ot.unif(nt)
reg = 0.01
Cdictionary_learned, log = gromov_wasserstein_dictionary_learning(dataset, ps, D, nt, q,
                                                                  epochs=15, batch_size=16, learning_rate=0.01, reg=reg,
                                                                  projection='nonnegative_symmetric', use_log=True, use_adam_optimizer=True, verbose=True)
# visualize loss evolution
pl.figure(2, (5, 5))
pl.clf()
pl.title('loss evolution by epoch')
pl.plot(log['loss_epochs'])
pl.xlabel('epochs')
pl.ylabel('cumulated reconstruction error')
pl.show()

# visualize learned dictionary atoms
pl.figure(3, (15, 10))
pl.clf()
pl.suptitle('Learned Gromov-Wasserstein dictionary atoms')
for idx_atom, atom in enumerate(Cdictionary_learned):
    scaled_atom = (atom - atom.min()) / (atom.max() - atom.min())
    x = MDS(dissimilarity='precomputed', random_state=0).fit_transform(1 - scaled_atom)
    pl.subplot(2, D, idx_atom + 1)
    pl.title('(graph) atom ' + str(c))
    plot_graph(x, atom / atom.max(), binary=False, color='C0')
    pl.axis("off")
    pl.subplot(2, D, D + idx_atom + 1)
    pl.title('(matrix) atom ' + str(idx_atom + 1))
    pl.imshow(scaled_atom, interpolation='nearest')
    pl.colorbar()
    pl.axis("off")
pl.show()

unmixings = []
reconstruction_errors = []
for C in dataset:
    p = ot.unif(C.shape[0])
    unmixing, Cembedded, OT, reconstruction_error = gromov_wasserstein_linear_unmixing(C, Cdictionary_learned, p, q, reg=reg,
                                                                                       tol_outer=10**(-6), tol_inner=10**(-6), max_iter_outer=20, max_iter_inner=200)
    unmixings.append(unmixing)
    reconstruction_errors.append(reconstruction_error)
unmixings = np.array(unmixings)
print('cumulated reconstruction error:', np.array(reconstruction_errors).sum())

fig = plt.figure(4, (5, 5))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('unmixings on gromov-wasserstein dictionary')
for cluster in range(nlabels):
    start, end = 50 * cluster, 50 * (cluster + 1)
    ax.scatter(unmixings[start:end, 0], unmixings[start:end, 1], unmixings[start:end, 2], c='C' + str(cluster), marker='o')

ax.set_xlabel('atom 1')
ax.set_ylabel('atom 2')
ax.set_zlabel('atom 3')

plt.show()

# %% Add node features to graphs in the dataset of SBM

# For each graph, nodes belonging to each cluster are assigned to the label 0 or 1
dataset_features = []
for i in range(len(dataset)):
    n = dataset[i].shape[0]
    F = np.zeros((n, 1))
    n_clusters = np.unique(cluster_assignments[i])
    for c in n_clusters:
        nodes_c = np.argwhere(cluster_assignments[i] == c)[:, 0]
        mid = len(nodes_c) // 2
        F[nodes_c[:mid], 0] = 0.
        F[nodes_c[mid:], 0] = 1.
    dataset_features.append(F)

pl.figure(5, (15, 10))
pl.clf()
pl.suptitle('Samples from the attributed graph dataset')
for idx_c, c in enumerate(clusters):
    C = dataset[(c - 1) * 50]  # sample with c clusters
    F = dataset_features[(c - 1) * 50]
    colors = ['C' + str(int(F[i, 0])) for i in range(F.shape[0])]
    # get 2d position for nodes
    x = MDS(dissimilarity='precomputed', random_state=0).fit_transform(1 - C)
    pl.subplot(2, nlabels, c)
    pl.title('(graph) sample from label ' + str(c))
    plot_graph(x, C, binary=True, color=colors)
    pl.axis("off")
    pl.subplot(2, nlabels, nlabels + c)
    pl.title('(adjacency matrix) sample from label ' + str(c))
    pl.imshow(C, interpolation='nearest')
    pl.axis("off")

pl.show()
# %%
# =============================================================================
# Infer the fused gromov-wasserstein dictionary from the dataset of attributed graphs
# =============================================================================

ps = [ot.unif(C.shape[0]) for C in dataset]
D = 3  # 6 atoms instead of 3
nt = 6
q = ot.unif(nt)
reg = 0.01
alpha = 0.5  # trade-off parameter between structure and feature information of Fused Gromov-Wasserstein

Cdictionary_learned, Ydictionary_learned, log = fused_gromov_wasserstein_dictionary_learning(dataset, dataset_features, ps, D, nt, q, alpha,
                                                                                             epochs=20, batch_size=16, learning_rate_C=0.1, learning_rate_Y=0.1, reg=reg,
                                                                                             projection='nonnegative_symmetric', use_log=True, use_adam_optimizer=False, verbose=True)
# visualize loss evolution
pl.figure(6, (5, 5))
pl.clf()
pl.title('loss evolution by epoch')
pl.plot(log['loss_epochs'])
pl.xlabel('epochs')
pl.ylabel('cumulated reconstruction error')
pl.show()

# visualize learned dictionary atoms
pl.figure(7, (15, 10))
pl.clf()
pl.suptitle('Learned Graph Dictionary')
for idx_atom, atom in enumerate(Cdictionary_learned):
    scaled_atom = (atom - atom.min()) / (atom.max() - atom.min())
    F = Ydictionary_learned[idx_atom]
    colors = []
    colors = ['C0' if F[i, 0] < 0.5 else 'C1' for i in range(F.shape[0])]
    x = MDS(dissimilarity='precomputed', random_state=0).fit_transform(1 - scaled_atom)
    pl.subplot(2, D, idx_atom + 1)
    pl.title('(attributed graph) atom ' + str(c))
    plot_graph(x, (atom / atom.max())**2, binary=False, color=colors)
    pl.axis("off")
    pl.subplot(2, D, D + idx_atom + 1)
    pl.title('(matrix) atom ' + str(idx_atom + 1))
    pl.imshow(scaled_atom, interpolation='nearest')
    pl.colorbar()
    pl.axis("off")
pl.show()

unmixings = []
reconstruction_errors = []
for i in range(len(dataset)):
    C = dataset[i]
    Y = dataset_features[i]
    p = ot.unif(C.shape[0])
    unmixing, Cembedded, Yembedded, OT, reconstruction_error = fused_gromov_wasserstein_linear_unmixing(C, Y, Cdictionary_learned, Ydictionary_learned, p, q, alpha=alpha,
                                                                                                        reg=reg, tol_outer=10**(-6), tol_inner=10**(-6), max_iter_outer=30, max_iter_inner=300)
    unmixings.append(unmixing)
    reconstruction_errors.append(reconstruction_error)
unmixings = np.array(unmixings)
print('cumulated reconstruction error:', np.array(reconstruction_errors).sum())

fig = plt.figure(8, (5, 5))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('unmixings on fused gromov-wasserstein dictionary')
for cluster in range(nlabels):
    start, end = 50 * cluster, 50 * (cluster + 1)
    ax.scatter(unmixings[start:end, 0], unmixings[start:end, 1], unmixings[start:end, 2], c='C' + str(cluster), marker='o')

ax.set_xlabel('atom 1')
ax.set_ylabel('atom 2')
ax.set_zlabel('atom 3')

plt.show()
