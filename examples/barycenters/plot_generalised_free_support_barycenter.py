# -*- coding: utf-8 -*-
"""
=======================================
Generalised Wasserstein Barycenter Demo
=======================================

This example illustrates the computation of Generalised Wasserstein Barycenter
as proposed in [42].


[42] DELON, Julie, GOZLAN, Nathael, et SAINT-DIZIER, Alexandre.
Generalized Wasserstein barycenters between probability measures living on different subspaces.
arXiv preprint arXiv:2105.09755, 2021.

"""

# Author: Eloi Tanguy <eloi.tanguy@polytechnique.edu>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 1

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import ot
# necessary for 3d plot even if not used
from mpl_toolkits.mplot3d import Axes3D

##############################################################################
# Generate data
# -------------

# Input measures
I1 = pl.imread('../../data/redcross.png').astype(np.float64)[::4, ::4, 2]
I2 = pl.imread('../../data/tooth.png').astype(np.float64)[::4, ::4, 2]
I3 = pl.imread('../../data/heart.png').astype(np.float64)[::4, ::4, 2]

sz = I1.shape[0]
UU, VV = np.meshgrid(np.arange(sz), np.arange(sz))

# Input measure locations in their respective 2D spaces
X = [np.stack((UU[I == 0], VV[I == 0]), 1) * 1.0 for I in [I1, I2, I3]]

# Input measure weights
a = [ot.unif(x.shape[0]) for x in X]

# Projections 3D -> 2D
P1 = np.array([[1,0,0],[0,1,0]])
P2 = np.array([[0,1,0],[0,0,1]])
P3 = np.array([[1,0,0],[0,0,1]])
P = [P1,P2,P3]

# Barycenter weights
weights = np.array([1/3, 1/3, 1/3])

# Number of barycenter points to compute
L = 500

##############################################################################
# Barycenter computation and plot
# ----------------------
bar = ot.lp.generalized_free_support_barycenter(X, a, P, L)

X_visu = [x @ Pi for (x, Pi) in zip(X, P)]  # send measures to the global space for visu
fig = plt.figure(figsize=(7, 7))
axis = fig.add_subplot(1, 1, 1, projection="3d")
for x in X_visu:
    axis.scatter(x[:, 0], x[:, 1], x[:, 2], marker='.', alpha=.6)
axis.scatter(bar[:, 0], bar[:, 1], bar[:, 2], marker='.', alpha=.6)
plt.show()
