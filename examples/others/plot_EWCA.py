# -*- coding: utf-8 -*-
"""
====================================================
Optimal Transport for dimensionality reduction
using EWCA (Entropic Wasserstein Component Analysis)
====================================================

This example illustrates the computation of EWCA
for dimensionality reduction and its visualization.

"""

# Author: Antoine Collas <antoine.collas@inria.fr>
#
# License: MIT License
# sphinx_gallery_thumbnail_number = 3

import numpy as np
import matplotlib.pylab as pl
from ot.dr import ewca
from sklearn.datasets import make_blobs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker as mticker
import matplotlib.patches as patches
import matplotlib

##############################################################################
# Generate data
# -------------

n_samples = 20
esp = 0.8
centers = np.array([[esp, esp], [-esp, -esp]])
cluster_std = 0.4

rng = np.random.RandomState(42)
X, y = make_blobs(
    n_samples=n_samples,
    n_features=2,
    centers=centers,
    cluster_std=cluster_std,
    shuffle=False,
    random_state=rng,
)
X = X - X.mean(0)

##############################################################################
# Compute EWCA
# -------------
pi, U = ewca(X, k=2, reg=0.5)


##############################################################################
# Plot
# -------------
fs = 15
scale = 3
origin = np.array([0, 0])
cmap = matplotlib.colormaps.get_cmap("tab10")

fig, ax = pl.subplots(1, 2, figsize=(10, 5))
u = U[:, 0]
ax[0].plot(
    [origin[0], scale * u[0] + origin[0]],
    [origin[1], scale * u[1] + origin[1]],
    color="grey",
    linestyle="--",
    lw=3,
    alpha=0.3,
)
label_ = r"$\mathbf{U}$"
ax[0].plot(
    [origin[0], -scale * u[0] + origin[0]],
    [origin[1], -scale * u[1] + origin[1]],
    color="grey",
    linestyle="--",
    lw=3,
    alpha=0.3,
    label=label_,
)
X1 = X @ u[:, None] @ u[:, None].T + origin

ax[0].axis("scaled")
thresh = 0.15
mm = 1
for i in range(n_samples):
    for j in range(n_samples):
        v = pi[i, j] / pi.max()
        if v >= thresh or (i, j) == (n_samples - 1, n_samples - 1):
            ax[0].plot(
                [X[i, 0], X1[j, 0]],
                [X[i, 1], X1[j, 1]],
                alpha=mm * v,
                linestyle="-",
                c="C0",
                label=r"$\pi_{ij}$"
                if (i, j) == (n_samples - 1, n_samples - 1)
                else None,
            )
ax[0].scatter(
    X[:, 0],
    X[:, 1],
    color=[cmap(y[i] + 1) for i in range(n_samples)],
    alpha=0.4,
    label=r"$\mathbf{x}_i$",
    zorder=30,
    s=50,
)
ax[0].scatter(
    X1[:, 0],
    X1[:, 1],
    color=[cmap(y[i] + 1) for i in range(n_samples)],
    alpha=0.9,
    s=50,
    marker="+",
    label=r"$\mathbf{U}\mathbf{U}^{\top}\mathbf{x}_i$",
    zorder=30,
)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].legend(fontsize=fs, loc="upper left")

divider = make_axes_locatable(ax[1])
norm = matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=100)
im = ax[1].imshow(n_samples * pi * 100, cmap=pl.cm.Blues, norm=norm, aspect="auto")
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(im, cax=cax, orientation="vertical")
ticks_loc = cb.ax.get_yticks().tolist()
cb.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
cb.ax.set_yticklabels([f"{int(i)}%" for i in cb.get_ticks()])
cb.ax.tick_params(labelsize=fs)
for i, class_ in enumerate(np.sort(np.unique(y))):
    indices = y == class_
    idx_min = np.min(np.arange(len(y))[indices])
    idx_max = np.max(np.arange(len(y))[indices])
    width = idx_max - idx_min + 1
    rect = patches.Rectangle(
        (idx_min - 0.5, idx_min - 0.5),
        width,
        width,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax[1].add_patch(rect)

ax[1].set_title("OT plan", fontsize=fs)
ax[1].set_ylabel(r"($\mathbf{x}_1, \cdots, \mathbf{x}_n$)")
x_label = r"($\mathbf{U}\mathbf{U}^{\top}\mathbf{x}_1, \cdots,"
x_label += r"\mathbf{U}\mathbf{U}^{\top}\mathbf{x}_n$)"
ax[1].set_xlabel(x_label)

pl.tight_layout()
pl.show()
