# -*- coding: utf-8 -*-
"""
===============================
1D Unbalanced optimal transport
===============================

This example illustrates the computation of Unbalanced Optimal transport
using a Kullback-Leibler relaxation.
"""

# Author: Hicham Janati <hicham.janati@inria.fr>
#         Clément Bonet <clement.bonet.mapp@polytechnique.edu>
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

Gs, log = ot.unbalanced.mm_unbalanced(a, b, M / M.max(), alpha, verbose=False, log=True)

pl.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gs, "UOT plan")
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
print("Unbalanced OT loss:", log["total_cost"] * M.max())


##############################################################################
# Solve 1D UOT with Frank-Wolfe
# -----------------------------


# %% 1D UOT with FW


alpha = M.max()  # Unbalanced KL relaxation parameter

a_reweighted, b_reweighted, loss = ot.unbalanced.uot_1d(
    torch.tensor(x, dtype=torch.float64),
    torch.tensor(x, dtype=torch.float64),
    alpha,
    u_weights=torch.tensor(a, dtype=torch.float64),
    v_weights=torch.tensor(b, dtype=torch.float64),
    p=2,
    returnCost="total",
)

pl.figure(4, figsize=(6.4, 3))
pl.plot(x, a, "b", label="Source distribution")
pl.plot(x, b, "r", label="Target distribution")
pl.fill(x, a_reweighted, "b", alpha=0.5, label="Transported source")
pl.fill(x, b_reweighted, "r", alpha=0.5, label="Transported target")
pl.legend(loc="upper right")
pl.title("Distributions and transported mass for UOT")
pl.show()

print("Mass of reweighted marginals:", a_reweighted.sum().item())
print("Unbalanced OT loss:", loss.item())


##############################################################################
# Solve Unbalanced Sinkhorn
# -------------------------

# %% Sinkhorn UOT

# Sinkhorn

epsilon = 0.1  # entropy parameter
alpha = 1.0  # Unbalanced KL relaxation parameter
Gs = ot.unbalanced.sinkhorn_unbalanced(a, b, M / M.max(), epsilon, alpha, verbose=True)

pl.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gs, "Entropic UOT plan")
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
