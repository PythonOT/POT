# -*- coding: utf-8 -*-
"""
==============================
Debiased Sinkhorn barycenter demo
==============================

This example illustrates the computation of the debiased Sinkhorn barycenter
as proposed in [28].


[28] Janati, H., Cuturi, M., Gramfort, A. Proceedings of the 37th
 International Conference on Machine Learning, PMLR 119:4692-4701, 2020

"""

# Author: Hicham Janati <hicham.janati100@gmail.com>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 4

import numpy as np
import matplotlib.pylab as plt
import ot
from ot.bregman import barycenter, convolutional_barycenter2d
# necessary for 3d plot even if not used
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.collections import PolyCollection

##############################################################################
# Debiased barycenter of 1D Gaussians
# ------------------------------------

#%% parameters

n = 100  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)

# creating matrix A containing all distributions
A = np.vstack((a1, a2)).T
n_distributions = A.shape[1]

# loss matrix + normalization
M = ot.utils.dist0(n)
M /= M.max()

#%% barycenter computation

alpha = 0.2  # 0<=alpha<=1
weights = np.array([1 - alpha, alpha])

epsilons = [5e-3, 1e-2, 5e-2]


bars = [barycenter(A, M, reg, weights, method="sinkhorn") for reg in epsilons]
bars_debiased = [barycenter(A, M, reg, weights, method="debiased") for reg in epsilons]

labels = ["Sinkhorn barycenter", "Debiased barycenter"]
colors = ["indianred", "gold"]

f, axes = plt.subplots(1, len(epsilons), tight_layout=True, sharey=True, figsize=(12, 4))
for ax, eps, bar, bar_debiased in zip(axes, epsilons, bars, bars_debiased):
    ax.plot(A[:, 0], color="k", ls="--", label="Input data", alpha=0.3)
    ax.plot(A[:, 1], color="k", ls="--", alpha=0.3)
    for data, label, color in zip([bar, bar_debiased], labels, colors):
        ax.plot(data, color=color, label=label, lw=2)
    ax.set_title(r"$\varepsilon = %.3f$" % eps)
plt.legend()
plt.show()


##############################################################################
# Debiased barycenter of 2D images
# ---------------------------------



f1 = 1 - plt.imread('../../data/redcross.png')[:, :, 2]
f2 = 1 - plt.imread('../../data/tooth.png')[:, :, 2]
f3 = 1 - plt.imread('../../data/duck.png')[:, :, 2]

A = []
f1 = f1 / np.sum(f1)
f2 = f2 / np.sum(f2)
f3 = f3 / np.sum(f3)

A.append(f1)
A.append(f2)
A.append(f3)

A = np.array(A)

##############################################################################
# Display the input images

f, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img in zip(axes, A):
    ax.imshow(img, cmap="Greys")
plt.show()


##############################################################################
# Barycenter computation and visualization
# ----------------------------------------
#

bars_sinkhorn, bars_debiased = [], []
epsilons = [5e-3, 7e-3, 1e-2]
for eps in epsilons:
    bar = convolutional_barycenter2d(A, eps, method="sinkhorn")
    bar_debiased = convolutional_barycenter2d(A, eps, method="debiased")
    bars_sinkhorn.append(bar)
    bars_debiased.append(bar_debiased)

titles = ["Sinkhorn", "Debiased"]
all_bars = [bars_sinkhorn, bars_debiased]
f, axes = plt.subplots(2, 3, figsize=(12, 8))
for jj, (method, ax_row, bars) in enumerate(zip(titles, axes, all_bars)):
    for ii, (ax, img, eps) in enumerate(zip(ax_row, bars, epsilons)):
        ax.imshow(img, cmap="Greys")
        if jj == 0:
            ax.set_title(r"$\varepsilon = %.3f$" % eps, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if ii == 0:
            ax.set_ylabel(method, fontsize=15)
plt.show()