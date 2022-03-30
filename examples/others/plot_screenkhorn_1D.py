# -*- coding: utf-8 -*-
"""
===============================
1D Screened optimal transport
===============================

This example illustrates the computation of Screenkhorn [26].

[26] Alaya M. Z., Bérar M., Gasso G., Rakotomamonjy A. (2019).
Screening Sinkhorn Algorithm for Regularized Optimal Transport,
Advances in Neural Information Processing Systems 33 (NeurIPS).
"""

# Author: Mokhtar Z. Alaya <mokhtarzahdi.alaya@gmail.com>
#
# License: MIT License

import numpy as np
import matplotlib.pylab as pl
import ot.plot
from ot.datasets import make_1D_gauss as gauss
from ot.bregman import screenkhorn

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

# plot distributions and loss matrix

pl.figure(2, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, M, 'Cost matrix M')

##############################################################################
# Solve Screenkhorn
# -----------------------

# Screenkhorn
lambd = 2e-03  # entropy parameter
ns_budget = 30  # budget number of points to be keeped in the source distribution
nt_budget = 30  # budget number of points to be keeped in the target distribution

G_screen = screenkhorn(a, b, M, lambd, ns_budget, nt_budget, uniform=False, restricted=True, verbose=True)
pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, G_screen, 'OT matrix Screenkhorn')
pl.show()
