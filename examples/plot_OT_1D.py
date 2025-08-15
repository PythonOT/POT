# %%
# -*- coding: utf-8 -*-
"""
====================================
Optimal Transport for fixed support
====================================

This example illustrates the computation of EMD and Sinkhorn transport plans
and their visualization.

"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License
# sphinx_gallery_thumbnail_number = 3

import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

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

# loss matrix
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
M /= M.max()


##############################################################################
# Plot distributions and loss matrix
# ----------------------------------

# %% plot the distributions

pl.figure(1, figsize=(6.4, 3))
pl.plot(x, a, "b", label="Source distribution")
pl.plot(x, b, "r", label="Target distribution")
pl.legend()

# %% plot distributions and loss matrix

pl.figure(2, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, M, "Cost matrix M")

##############################################################################
# Solve Exact OT
# ---------


# %% EMD

# use fast 1D solver
G0 = ot.emd_1d(x, x, a, b)

# Equivalent to
# G0 = ot.emd(a, b, M)

pl.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, G0, "OT matrix G0")

##############################################################################
# Solve Sinkhorn
# --------------


# %% Sinkhorn

lambd = 1e-3
Gs = ot.sinkhorn(a, b, M, lambd, verbose=True)

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gs, "OT matrix Sinkhorn")

pl.show()


##############################################################################
# Solve Smooth OT
# ---------------
# We illustrate below Smooth and Sparse (KL an L2 reg.) OT and
# sparsity-constrained OT, together with their visualizations.
#
# Reference:
#
# Blondel, M., Seguy, V., & Rolet, A. (2018). Smooth and Sparse Optimal
# Transport. Proceedings of the # Twenty-First International Conference on
# Artificial Intelligence and # Statistics (AISTATS).


# %% Smooth OT with KL regularization

lambd = 2e-3
Gsm = ot.smooth.smooth_ot_dual(a, b, M, lambd, reg_type="kl")

pl.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gsm, "OT matrix Smooth OT KL reg.")

pl.show()


# %% Smooth OT with squared l2 regularization

lambd = 1e-1
Gsm = ot.smooth.smooth_ot_dual(a, b, M, lambd, reg_type="l2")

pl.figure(5, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gsm, "OT matrix Smooth OT l2 reg.")

pl.show()

# %% Sparsity-constrained OT

lambd = 1e-1

max_nz = 2  # two non-zero entries are permitted per column of the OT plan
Gsc = ot.smooth.smooth_ot_dual(
    a, b, M, lambd, reg_type="sparsity_constrained", max_nz=max_nz
)
pl.figure(6, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gsc, "Sparsity constrained OT matrix; k=2.")

pl.show()
