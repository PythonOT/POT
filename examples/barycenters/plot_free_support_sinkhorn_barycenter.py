# -*- coding: utf-8 -*-
"""
========================================================
2D free support Sinkhorn barycenters of distributions
========================================================

Illustration of Sinkhorn barycenter calculation between empirical distributions understood as point clouds

"""

# Authors: Eduardo Fernandes Montesuma <eduardo.fernandes-montesuma@universite-paris-saclay.fr>
#
# License: MIT License

import ot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# %%
# General Parameters
# ------------------
reg = 1e-2 # Entropic Regularization
numItermax = 20 # Maximum number of iterations for the Barycenter algorithm
numInnerItermax = 50 # Maximum number of sinkhorn iterations
n_samples = 200

# %%
# Generate Data
# -------------

get_rotation = lambda angle: np.array([
        [np.cos((angle / 180) * np.pi), -np.sin((angle / 180) * np.pi)],
        [np.sin((angle / 180) * np.pi), np.cos((angle / 180) * np.pi)]
    ])

R2, R3, R4 = get_rotation(15), get_rotation(30), get_rotation(45)

X1, _ = make_moons(n_samples=300, noise=1e-1)
a1 = ot.utils.unif(X1.shape[0], type_as=X1)
X2 = np.dot(X1, R2)
a2 = ot.utils.unif(X1.shape[0], type_as=X2)
X3 = np.dot(X1, R3)
a3 = ot.utils.unif(X1.shape[0], type_as=X3)
X4 = np.dot(X1, R4)
a4 = ot.utils.unif(X1.shape[0], type_as=X4)

# %%
# Inspect generated distributions
# -------------------------------

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].scatter(x=X1[:, 0], y=X1[:, 1], c='steelblue', edgecolor='k')
axes[1].scatter(x=X2[:, 0], y=X2[:, 1], c='steelblue', edgecolor='k')
axes[2].scatter(x=X3[:, 0], y=X3[:, 1], c='steelblue', edgecolor='k')
axes[3].scatter(x=X4[:, 0], y=X4[:, 1], c='steelblue', edgecolor='k')

axes[0].set_xlim([-3, 3])
axes[0].set_ylim([-3, 3])
axes[0].set_title('Rotation: 0')

axes[1].set_xlim([-3, 3])
axes[1].set_ylim([-3, 3])
axes[0].set_title('Rotation: 15')

axes[2].set_xlim([-3, 3])
axes[2].set_ylim([-3, 3])
axes[0].set_title('Rotation: 30')

axes[3].set_xlim([-3, 3])
axes[3].set_ylim([-3, 3])
axes[0].set_title('Rotation: 45')

plt.tight_layout()
plt.show()

# %%
# Interpolating Empirical Distributions
# -------------------------------------

fig = plt.figure(figsize=(10, 10))

weights = np.array([
    [3/3, 0/3],
    [2/3, 1/3],
    [1/3, 2/3],
    [0/3, 3/3],
]).astype(np.float32)

for k in range(4):
    XB_init = np.random.randn(n_samples, 2)
    XB = ot.bregman.free_support_sinkhorn_barycenter(
        measures_locations=[X1, X2],
        measures_weights=[a1, a2],
        weights=weights[k],
        X_init=XB_init,
        reg=reg,
        numItermax=numItermax,
        numInnerItermax=numInnerItermax
    )
    ax = plt.subplot2grid((4, 4), (0, k))
    ax.scatter(XB[:, 0], XB[:, 1], color='steelblue', edgecolor='k')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

for k in range(1, 4, 1):
    XB_init = np.random.randn(n_samples, 2)
    XB = ot.bregman.free_support_sinkhorn_barycenter(
        measures_locations=[X1, X3],
        measures_weights=[a1, a2],
        weights=weights[k],
        X_init=XB_init,
        reg=reg,
        numItermax=numItermax,
        numInnerItermax=numInnerItermax
    )
    ax = plt.subplot2grid((4, 4), (k, 0))
    ax.scatter(XB[:, 0], XB[:, 1], color='steelblue', edgecolor='k')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

for k in range(1, 4, 1):
    XB_init = np.random.randn(n_samples, 2)
    XB = ot.bregman.free_support_sinkhorn_barycenter(
        measures_locations=[X3, X4],
        measures_weights=[a1, a2],
        weights=weights[k],
        X_init=XB_init,
        reg=reg,
        numItermax=numItermax,
        numInnerItermax=numInnerItermax
    )
    ax = plt.subplot2grid((4, 4), (3, k))
    ax.scatter(XB[:, 0], XB[:, 1], color='steelblue', edgecolor='k')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

for k in range(1, 3, 1):
    XB_init = np.random.randn(n_samples, 2)
    XB = ot.bregman.free_support_sinkhorn_barycenter(
        measures_locations=[X2, X4],
        measures_weights=[a1, a2],
        weights=weights[k],
        X_init=XB_init,
        reg=reg,
        numItermax=numItermax,
        numInnerItermax=numInnerItermax
    )
    ax = plt.subplot2grid((4, 4), (k, 3))
    ax.scatter(XB[:, 0], XB[:, 1], color='steelblue', edgecolor='k')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

plt.show()