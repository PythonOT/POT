# -*- coding: utf-8 -*-
"""
================================
Sparsity-constrained optimal transport example
================================

This example illustrates EMD, squared l2 regularized OT, and sparsity-constrained OT plans.
The sparsity-constrained OT can be considered as a middle ground between EMD and squared l2 regularized OT.

"""

# Author: Tianlin Liu <t.liu@unibas.ch>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 5

import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

##############################################################################
# Generate data
# -------------


#%% parameters

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

#%% plot the distributions

pl.figure(1, figsize=(6.4, 3))
pl.plot(x, a, 'b', label='Source distribution')
pl.plot(x, b, 'r', label='Target distribution')
pl.legend()

#%% plot distributions and loss matrix

pl.figure(2, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, M, 'Cost matrix M')


#%% EMD

# use fast 1D solver
G0 = ot.emd_1d(x, x, a, b)

# Equivalent to
# G0 = ot.emd(a, b, M)

pl.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, G0, 'OT matrix G0')


#%% Smooth OT with squared l2 regularization

lambd = 1e-1
Gsm = ot.smooth.smooth_ot_dual(a, b, M, lambd, reg_type='l2')

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gsm, 'OT matrix Smooth OT l2 reg.')

pl.show()


#%% Smooth OT with squared l2 regularization

lambd = 1e-1
Gsc = ot.sparse.sparsity_constrained_ot_dual(a, b, M, lambd, max_nz=2)
pl.figure(5, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gsc, 'Sparsity contrained OT matrix; k=2.')

pl.show()

# %%
