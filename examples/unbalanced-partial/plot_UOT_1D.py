# -*- coding: utf-8 -*-
"""
===============================
1D Unbalanced optimal transport
===============================

This example illustrates the computation of Unbalanced Optimal transport
using a Kullback-Leibler relaxation.
"""

# Author: Hicham Janati <hicham.janati@inria.fr>
#         Clément Bonet <clemebt.bonet.mapp@polytechnique.edu>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 4

import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import torch
import cvxpy as cp

##############################################################################
# Generate data
# -------------


# %% parameters

n = 100  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
a = gauss(n, m=20, s=5)  # m= mean, s= std
b = gauss(n, m=60, s=10)

# make distributions unbalanced
b *= 5.0

# loss matrix
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))


##############################################################################
# Plot distributions and loss matrix
# ----------------------------------

# %% plot the distributions

pl.figure(1, figsize=(6.4, 3))
pl.plot(x, a, "b", label="Source distribution")
pl.plot(x, b, "r", label="Target distribution")
pl.legend()

# plot distributions and loss matrix

pl.figure(2, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, M, "Cost matrix M")


##############################################################################
# Solve Unbalanced OT with MM Unbalanced
# -----------------------------------

# %% MM Unbalanced

alpha = 1.0  # Unbalanced KL relaxation parameter

Gs = ot.unbalanced.mm_unbalanced(a, b, M / M.max(), alpha, verbose=False)

pl.figure(4, figsize=(6.4, 3))
pl.plot(x, a, "b", label="Source distribution")
pl.plot(x, b, "r", label="Target distribution")
pl.fill(x, Gs.sum(1), "b", alpha=0.5, label="Transported source")
pl.fill(x, Gs.sum(0), "r", alpha=0.5, label="Transported target")
pl.legend(loc="upper right")
pl.title("Distributions and transported mass for UOT")
pl.show()

print("Mass of reweighted marginals:", Gs.sum())


##############################################################################
# Solve 1D UOT with Frank-Wolfe
# -----------------------------

# %% 1D UOT with FW


alpha = M.max()  # Unbalanced KL relaxation parameter

a_reweighted, b_reweighted, loss = ot.unbalanced.uot_1d(
    x, x, alpha, u_weights=a, v_weights=b, p=2
)

pl.figure(4, figsize=(6.4, 3))
pl.plot(x, a, "b", label="Source distribution")
pl.plot(x, b, "r", label="Target distribution")
pl.fill(x, a_reweighted, "b", alpha=0.5, label="Transported source")
pl.fill(x, b_reweighted, "r", alpha=0.5, label="Transported target")
pl.legend(loc="upper right")
pl.title("Distributions and transported mass for UOT")
pl.show()

print("Mass of reweighted marginals:", a_reweighted.sum())


##############################################################################
# Solve 1D UOT with Frank-Wolfe (backprop mode)
# -----------------------------


# %% 1D UOT with FW (backprop mode)


alpha = M.max()  # Unbalanced KL relaxation parameter

a_reweighted, b_reweighted, loss = ot.unbalanced.uot_1d(
    torch.tensor(x, dtype=torch.float64),
    torch.tensor(x, dtype=torch.float64),
    alpha,
    u_weights=torch.tensor(a, dtype=torch.float64),
    v_weights=torch.tensor(b, dtype=torch.float64),
    p=2,
    mode="backprop",
)

pl.figure(4, figsize=(6.4, 3))
pl.plot(x, a, "b", label="Source distribution")
pl.plot(x, b, "r", label="Target distribution")
pl.fill(x, a_reweighted, "b", alpha=0.5, label="Transported source")
pl.fill(x, b_reweighted, "r", alpha=0.5, label="Transported target")
pl.legend(loc="upper right")
pl.title("Distributions and transported mass for UOT")
pl.show()

print("Mass of reweighted marginals:", a_reweighted.sum())


##############################################################################
# Solve 1D USOT with Frank-Wolfe with UOT (TO CHECK)
# -----------------------------

# %% TEST USOT


alpha = M.max()  # Unbalanced KL relaxation parameter

a_reweighted, b_reweighted, loss = ot.unbalanced.unbalanced_sliced_ot(
    torch.tensor(x.reshape((n, 1)), dtype=torch.float64),
    torch.tensor(x.reshape((n, 1)), dtype=torch.float64),
    alpha,
    torch.tensor(a, dtype=torch.float64),
    torch.tensor(b, dtype=torch.float64),
    mode="backprop",
    p=2,
)


# plot the transported mass
# -------------------------

pl.figure(4, figsize=(6.4, 3))
pl.plot(x, a, "b", label="Source distribution")
pl.plot(x, b, "r", label="Target distribution")
pl.fill(x, a_reweighted.numpy(), "b", alpha=0.5, label="Transported source")
pl.fill(x, b_reweighted.numpy(), "r", alpha=0.5, label="Transported target")
pl.legend(loc="upper right")
pl.title("Distributions and transported mass for UOT")
pl.show()

print("Mass of reweighted marginals:", a_reweighted.sum())


##############################################################################
# Solve Unbalanced OT with cvxpy
# ------------------------------

# %% UOT with cvxpy


# (https://colab.research.google.com/github/gpeyre/ot4ml/blob/main/python/5-unbalanced.ipynb)

alpha = M.max()  # Unbalanced KL relaxation parameter
n, m = a.shape[0], b.shape[0]

P = cp.Variable((n, m))

u = np.ones((n, 1))
v = np.ones((m, 1))
q = cp.sum(cp.kl_div(cp.matmul(P, v), a[:, None]))
r = cp.sum(cp.kl_div(cp.matmul(P.T, u), b[:, None]))

constr = [0 <= P]
# uncomment to perform balanced OT
# constr = [0 <= P, cp.matmul(P,u)==a[:,None], cp.matmul(P.T,v)==b[:,None]]

objective = cp.Minimize(cp.sum(cp.multiply(P, M)) + alpha * q + alpha * r)

prob = cp.Problem(objective, constr)
result = prob.solve()

G = P.value

pl.figure(4, figsize=(6.4, 3))
pl.plot(x, a, "b", label="Source distribution")
pl.plot(x, b, "r", label="Target distribution")
pl.fill(x, G.sum(1), "b", alpha=0.5, label="Transported source")
pl.fill(x, G.sum(0), "r", alpha=0.5, label="Transported target")
pl.legend(loc="upper right")
pl.title("Distributions and transported mass for UOT")
pl.show()

print("Mass of reweighted marginals:", Gs.sum())


##############################################################################
# Solve Unbalanced Sinkhorn
# -------------------------

# %% Sinkhorn UOT

# Sinkhorn

epsilon = 0.1  # entropy parameter
alpha = 1.0  # Unbalanced KL relaxation parameter
Gs = ot.unbalanced.sinkhorn_unbalanced(a, b, M / M.max(), epsilon, alpha, verbose=True)

pl.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gs, "UOT matrix Sinkhorn")
pl.show()

pl.figure(4, figsize=(6.4, 3))
pl.plot(x, a, "b", label="Source distribution")
pl.plot(x, b, "r", label="Target distribution")
pl.fill(x, Gs.sum(1), "b", alpha=0.5, label="Transported source")
pl.fill(x, Gs.sum(0), "r", alpha=0.5, label="Transported target")
pl.legend(loc="upper right")
pl.title("Distributions and transported mass for UOT")
pl.show()

print("Mass of reweighted marginals:", Gs.sum())
