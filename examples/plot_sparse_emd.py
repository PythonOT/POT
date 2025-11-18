# -*- coding: utf-8 -*-
"""
============================================
Sparse Optimal Transport
============================================

In many real-world optimal transport (OT) problems, the transport plan is naturally sparse: only a small fraction of all possible source-target pairs actually exchange mass. In such cases, using sparse OT solvers can provide significant computational speedups and memory savings compared to dense solvers, which compute and store the full transport matrix.

The figure below illustrates the advantages of sparse OT solvers over dense ones in terms of speed and memory usage for different sparsity levels of the transport plan.

.. image:: /_static/images/comparison.png
    :align: center
    :width: 80%
    :alt: Dense vs Sparse OT: Speed and Memory Advantages
"""


# Author: Nathan Neike <nathan.neike@example.com>
# License: MIT License
# sphinx_gallery_thumbnail_number = 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import ot


##############################################################################
# Generate minimal example data
# ------------------------------
#
# We create a simple example with 2 source points and 2 target points to
# illustrate the concept of sparse optimal transport.

# %%

X = np.array([[0, 0], [1, 0]])
Y = np.array([[0, 1], [1, 1]])
a = np.array([0.5, 0.5])
b = np.array([0.5, 0.5])


##############################################################################
# Build sparse cost matrix
# -------------------------
#
# Instead of allowing all possible edges (dense OT), we only allow two edges:
# source 0 -> target 0 and source 1 -> target 1. This is specified using a
# sparse matrix format (COO).

# %%

# Only allow two edges: source 0 -> target 0, source 1 -> target 1
rows = [0, 1]
cols = [0, 1]
vals = [np.linalg.norm(X[0] - Y[0]), np.linalg.norm(X[1] - Y[1])]
M_sparse = coo_matrix((vals, (rows, cols)), shape=(2, 2))


##############################################################################
# Solve sparse OT problem
# ------------------------
#
# When passing a sparse cost matrix to ot.emd with log=True, the solution
# is returned in the log dictionary with fields 'flow_sources', 'flow_targets',
# and 'flow_values' containing the edge information.

# %%

G, log = ot.emd(a, b, M_sparse, log=True)

print("Sparse OT cost:", log["cost"])
print("Edges:")
for i, j, v in zip(log["flow_sources"], log["flow_targets"], log["flow_values"]):
    print(f"  source {i} -> target {j}, flow={v:.3f}")


##############################################################################
# Visualize allowed edges
# ---------------------------------
#
# The sparse cost matrix only allows transport along specific edges.

# %%


plt.figure(figsize=(8, 4))

# Sparse OT: allowed edges only
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c="r", marker="o", s=100, zorder=3)
plt.scatter(Y[:, 0], Y[:, 1], c="b", marker="x", s=100, zorder=3)
for i, j in zip(rows, cols):
    plt.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], "b-", linewidth=2, alpha=0.6)
plt.title("Sparse OT: Allowed Edges Only")

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xticks([0, 1])
plt.yticks([0, 1])

# Dense OT: all possible edges
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c="r", marker="o", s=100, zorder=3)
plt.scatter(Y[:, 0], Y[:, 1], c="b", marker="x", s=100, zorder=3)
for i in range(2):
    for j in range(2):
        plt.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], "b-", linewidth=2, alpha=0.3)
plt.title("Dense OT: All Possible Edges")
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xticks([0, 1])
plt.yticks([0, 1])

plt.tight_layout()


##############################################################################
# Larger example with clusters
# --------------------------------------
#
# Now we create a more realistic example with multiple clusters of sources
# and targets, where transport is only allowed within each cluster.

# %%

grid_size = 4
n_clusters = grid_size * grid_size
points_per_cluster = 2
cluster_spacing = 15.0
intra_cluster_spacing = 1.5
cluster_centers = (
    np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])
    * cluster_spacing
)

X_large = []
Y_large = []
a_large = []
b_large = []

for idx, (cx, cy) in enumerate(cluster_centers):
    for i in range(points_per_cluster):
        X_large.append(
            [cx + intra_cluster_spacing * (i - 1), cy - intra_cluster_spacing]
        )
        a_large.append(1.0 / (n_clusters * points_per_cluster))

    for i in range(points_per_cluster):
        Y_large.append(
            [cx + intra_cluster_spacing * (i - 1), cy + intra_cluster_spacing]
        )
        b_large.append(1.0 / (n_clusters * points_per_cluster))

X_large = np.array(X_large)
Y_large = np.array(Y_large)
a_large = np.array(a_large)
b_large = np.array(b_large)

nA = nB = n_clusters * points_per_cluster
source_labels = np.repeat(np.arange(n_clusters), points_per_cluster)
sink_labels = np.repeat(np.arange(n_clusters), points_per_cluster)


##############################################################################
# Build sparse cost matrix (intra-cluster only)
# ----------------------------------------------
#
# We construct a sparse cost matrix that only includes edges within each cluster.

# %%

M_full = ot.dist(X_large, Y_large, metric="euclidean")

rows = []
cols = []
vals = []
for k in range(n_clusters):
    src_idx = np.where(source_labels == k)[0]
    sink_idx = np.where(sink_labels == k)[0]
    for i in src_idx:
        for j in sink_idx:
            rows.append(i)
            cols.append(j)
            vals.append(M_full[i, j])
M_sparse_large = coo_matrix((vals, (rows, cols)), shape=(nA, nB))


##############################################################################
# Visualize allowed edges structure
# ----------------------------------
#
# Dense OT allows all connections, while sparse OT restricts to intra-cluster edges.

# %%

plt.figure(figsize=(16, 6))

# Dense OT: all possible edges
plt.subplot(1, 2, 1)
for i in range(nA):
    for j in range(nB):
        plt.plot(
            [X_large[i, 0], Y_large[j, 0]],
            [X_large[i, 1], Y_large[j, 1]],
            color="blue",
            alpha=0.1,
            linewidth=0.7,
        )
plt.scatter(X_large[:, 0], X_large[:, 1], c="r", marker="o", s=20)
plt.scatter(Y_large[:, 0], Y_large[:, 1], c="b", marker="x", s=20)
plt.axis("equal")
plt.title("Dense OT: All Possible Edges")

# Sparse OT: only intra-cluster edges
plt.subplot(1, 2, 2)
for k in range(n_clusters):
    src_idx = np.where(source_labels == k)[0]
    sink_idx = np.where(sink_labels == k)[0]
    for i in src_idx:
        for j in sink_idx:
            plt.plot(
                [X_large[i, 0], Y_large[j, 0]],
                [X_large[i, 1], Y_large[j, 1]],
                color="blue",
                alpha=0.7,
                linewidth=1.5,
            )
plt.scatter(X_large[:, 0], X_large[:, 1], c="r", marker="o", s=20)
plt.scatter(Y_large[:, 0], Y_large[:, 1], c="b", marker="x", s=20)
plt.axis("equal")
plt.title("Sparse OT: Only Intra-Cluster Edges")

plt.tight_layout()


##############################################################################
# Solve and compare sparse vs dense OT
# -------------------------------------
#
# We solve both dense and sparse OT problems and verify that they produce
# the same optimal solution when the sparse edges include the optimal paths.

# %%

# Solve dense OT (full cost matrix)
G_dense = ot.emd(a_large, b_large, M_full)
cost_dense = np.sum(G_dense * M_full)
print(f"Dense OT cost: {cost_dense:.6f}")

# Solve sparse OT (intra-cluster only)
G_sparse, log_sparse = ot.emd(a_large, b_large, M_sparse_large, log=True)
cost_sparse = log_sparse["cost"]
print(f"Sparse OT cost: {cost_sparse:.6f}")


##############################################################################
# Visualize optimal transport plans
# ----------------------------------
#
# Plot the edges that carry flow in the optimal solutions.

# %%

plt.figure(figsize=(16, 6))

# Dense OT
plt.subplot(1, 2, 1)
for i in range(nA):
    for j in range(nB):
        if G_dense[i, j] > 1e-10:
            plt.plot(
                [X_large[i, 0], Y_large[j, 0]],
                [X_large[i, 1], Y_large[j, 1]],
                color="blue",
                alpha=0.7,
                linewidth=1.5,
            )
plt.scatter(X_large[:, 0], X_large[:, 1], c="r", marker="o", s=20)
plt.scatter(Y_large[:, 0], Y_large[:, 1], c="b", marker="x", s=20)
plt.axis("equal")
plt.title("Dense OT: Optimal Transport Plan")

# Sparse OT
plt.subplot(1, 2, 2)
if log_sparse["flow_sources"] is not None:
    for i, j, v in zip(
        log_sparse["flow_sources"],
        log_sparse["flow_targets"],
        log_sparse["flow_values"],
    ):
        if v > 1e-10:
            plt.plot(
                [X_large[i, 0], Y_large[j, 0]],
                [X_large[i, 1], Y_large[j, 1]],
                color="blue",
                alpha=0.7,
                linewidth=1.5,
            )
plt.scatter(X_large[:, 0], X_large[:, 1], c="r", marker="o", s=20)
plt.scatter(Y_large[:, 0], Y_large[:, 1], c="b", marker="x", s=20)
plt.axis("equal")
plt.title("Sparse OT: Optimal Transport Plan")

plt.tight_layout()
plt.show()
