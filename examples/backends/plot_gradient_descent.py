# -*- coding: utf-8 -*-
r"""
===============================================================================
Solve Fused Unbalanced Gromov Wasserstein with Adam
===============================================================================

Since the FUGW loss is differentiable, it can be minimized with first-order optimization.
We show how to do this with the `loss_fugw_batch` function and compare the results with
the dedicated FUGW solver `fused_unbalanced_gromov_wasserstein`.
"""

# Author: Rémi Flamary <remi.flamary@polytechnique.edu>
#         Sonia Mazelet <sonia.mazelet@polytechnique.edu>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 2

import numpy as np
import matplotlib.pylab as pl
import torch
import ot
from ot.batch._quadratic import loss_fugw_batch, tensor_batch
from ot.gromov import fused_unbalanced_gromov_wasserstein
from sklearn.manifold import MDS


# %%
# Generation of source and target graphs
# ----------------

rng = np.random.RandomState(42)


def get_sbm(n, nc, ratio, P):
    nbpc = np.round(n * ratio).astype(int)
    n = np.sum(nbpc)
    C = np.zeros((n, n))
    for c1 in range(nc):
        for c2 in range(c1 + 1):
            if c1 == c2:
                for i in range(np.sum(nbpc[:c1]), np.sum(nbpc[: c1 + 1])):
                    for j in range(np.sum(nbpc[:c2]), i):
                        if rng.rand() <= P[c1, c2]:
                            C[i, j] = 1
            else:
                for i in range(np.sum(nbpc[:c1]), np.sum(nbpc[: c1 + 1])):
                    for j in range(np.sum(nbpc[:c2]), np.sum(nbpc[: c2 + 1])):
                        if rng.rand() <= P[c1, c2]:
                            C[i, j] = 1

    return C + C.T


def plot_graph(x, C, color="C0", s=100):
    for j in range(C.shape[0]):
        for i in range(j):
            if C[i, j] > 0:
                pl.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], alpha=0.2, color="k")
    pl.scatter(x[:, 0], x[:, 1], c=color, s=s, zorder=10, edgecolors="k")


def get_sbm_labels(n, ratio):
    nbpc = np.round(n * ratio).astype(int)
    return np.concatenate(
        [np.full(count, label, dtype=int) for label, count in enumerate(nbpc)]
    )


def get_noisy_one_hot(labels, n_classes, noise_level=0.1):
    x = np.eye(n_classes)[labels]
    x += noise_level * rng.randn(*x.shape)
    return x


n1 = 30
n2 = 20
nc1 = 3
nc2 = 2
ratio1 = np.array([0.33, 0.33, 0.33])
ratio2 = np.array([0.5, 0.5])

P1 = np.array([[0.8, 0.08, 0.0], [0.08, 0.8, 0.08], [0.0, 0.08, 0.8]])
P2 = np.array(0.6 * np.eye(2) + 0.05 * np.ones((2, 2)))
C1 = get_sbm(n1, nc1, ratio1, P1)
C2 = get_sbm(n2, nc2, ratio2, P2)
labels1 = get_sbm_labels(n1, ratio1)
labels2 = get_sbm_labels(n2, ratio2)

# Use noisy one-hot encodings of the SBM classes as node features.
feature_dim = max(nc1, nc2)
x1 = get_noisy_one_hot(labels1, feature_dim)
x2 = get_noisy_one_hot(labels2, feature_dim)
all_features = np.vstack([x1, x2])
feature_min = all_features[:, :3].min(axis=0, keepdims=True)
feature_max = all_features[:, :3].max(axis=0, keepdims=True)

# get 2d positions for visualization
pos1 = MDS(dissimilarity="precomputed", random_state=0, n_init=1).fit_transform(1 - C1)
pos2 = MDS(dissimilarity="precomputed", random_state=0, n_init=1).fit_transform(1 - C2)

colors1 = np.clip(
    (x1 - feature_min) / np.maximum(feature_max - feature_min, 1e-15), 0.0, 1.0
)
colors2 = np.clip(
    (x2 - feature_min) / np.maximum(feature_max - feature_min, 1e-15), 0.0, 1.0
)


pl.figure(1, (10, 5))
pl.clf()
pl.subplot(1, 2, 1)
plot_graph(pos1, C1, color=colors1)
pl.title("SBM source graph")
pl.axis("off")
pl.subplot(1, 2, 2)
plot_graph(pos2, C2, color=colors2)
pl.title("SBM target graph")
_ = pl.axis("off")


# %%
# Solve FUGW with Adam
# ----------------

# Even though `loss_fugw_batch` supports batches of problems, we use a
# batch of size 1 here for clarity.

a = ot.unif(C1.shape[0])
b = ot.unif(C2.shape[0])
M = ot.dist(x1, x2)
M /= M.max()

a_torch = torch.tensor(a[None, :])
b_torch = torch.tensor(b[None, :])
C1_torch = torch.tensor(C1[None, :, :])
C2_torch = torch.tensor(C2[None, :, :])
M_torch = torch.tensor(M[None, :, :])
L = tensor_batch(a_torch, b_torch, C1_torch, C2_torch, loss="sqeuclidean")

alpha = 0.5
reg_marginals = 1.0
lr = 1e-2
nb_iter_max = 1000

T0_torch = torch.tensor(
    rng.rand(a_torch.shape[0], a_torch.shape[1], b_torch.shape[1]),
    dtype=a_torch.dtype,
)
T0_torch /= T0_torch.sum(dim=(1, 2), keepdim=True)
T_torch = torch.log(torch.expm1(T0_torch)).clone().requires_grad_(True)
optimizer = torch.optim.Adam([T_torch], lr=lr)
loss_iter = []
mass_iter = []

for i in range(nb_iter_max):
    optimizer.zero_grad()
    # Positive transport plan parameterized as log(1 + exp(T)).
    plan_torch = torch.nn.functional.softplus(T_torch)
    loss = loss_fugw_batch(
        a_torch,
        b_torch,
        L,
        M_torch,
        plan_torch,
        alpha=alpha,
        reg_marginals=reg_marginals,
        divergence="kl",
        recompute_const=True,
    )[0]

    loss_iter.append(float(loss.detach()))
    mass_iter.append(float(plan_torch.detach().sum()))
    loss.backward()
    optimizer.step()

T_adam = torch.nn.functional.softplus(T_torch).detach().cpu().numpy()[0]

pl.figure(2, (10, 4))
pl.clf()
pl.subplot(1, 2, 1)
pl.plot(loss_iter)
pl.grid()
pl.title("FUGW loss along iterations")
pl.xlabel("Iterations")
pl.subplot(1, 2, 2)
pl.plot(mass_iter)
pl.grid()
pl.title("Transport mass")
_ = pl.xlabel("Iterations")


# %%
# Compare with the dedicated FUGW solver
# -------------------------------------
#
# The dedicated solver uses a block coordinate descent scheme. We compare the
# coupling it returns with the coupling obtained by direct Adam minimization on
# `loss_fugw_batch`. The FUGW loss is non convex so minimizing it directly with Adam does not
# necessarily give the same solution as the dedicated solver. By comparing the FUGW costs obtained by both methods,
# we find that the BCD solver gives a better solution than direct minimization on this example.

T_bcd, _, log = fused_unbalanced_gromov_wasserstein(
    C1,
    C2,
    wx=a,
    wy=b,
    reg_marginals=reg_marginals,
    divergence="kl",
    unbalanced_solver="mm",
    alpha=alpha,
    M=M,
    init_pi=np.outer(a, b),
    max_iter=100,
    tol=1e-7,
    max_iter_ot=200,
    tol_ot=1e-7,
    log=True,
)


print("Final batch FUGW loss:", loss_iter[-1])
print("FUGW cost reported by the dedicated solver:", log["fugw_cost"])


# %%
# Visualize the learned couplings
# -------------------------------
# We visualize the couplings obtained by both methods to compare them.
# The BCD solver gives sharper plans but both
# methods find couplings that match the structures of the graphs.

pl.figure(3, (10, 4))
pl.clf()
pl.subplot(1, 2, 1)
pl.imshow(T_adam, interpolation="nearest")
pl.title("Coupling from direct minimization")
pl.xlabel("Target nodes")
pl.ylabel("Source nodes")
pl.colorbar()
pl.subplot(1, 2, 2)
pl.imshow(T_bcd, interpolation="nearest")
pl.title("Coupling from BCD solver")
pl.xlabel("Target nodes")
pl.ylabel("Source nodes")
_ = pl.colorbar()
