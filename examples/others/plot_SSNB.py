# -*- coding: utf-8 -*-
r"""
=====================================================
Smooth and Strongly Convex Nearest Brenier Potentials
=====================================================

This example is designed to show how to use SSNB [58] in POT.
SSNB computes an l-strongly convex potential :math:`\varphi` with an L-Lipschitz gradient such that
:math:`\nabla \varphi \# \mu \approx \nu`. This regularity can be enforced only on the components of a partition
of the ambient space, which is a relaxation compared to imposing global regularity.

In this example, we consider a source measure :math:`\mu_s` which is the uniform measure on the unit square in
:math:`\mathbb{R}^2`, and the target measure :math:`\mu_t` which is the image of :math:`\mu_x` by
:math:`T(x_1, x_2) = (x_1 + 2\mathrm{sign}(x_2), 2 * x_2)`. The map :math:`T` is non-smooth, and we wish to approximate
it using a "Brenier-style" map :math:`\nabla \varphi` which is regular on the partition
:math:`\lbrace x_1 <=0, x_1>0\rbrace`, which is well adapted to this particular dataset.

We represent the gradients of the "bounding potentials" :math:`\varphi_l, \varphi_u` (from [59], Theorem 3.14),
which bound any SSNB potential which is optimal in the sense of [58], Definition 1:

.. math::
    \varphi \in \mathrm{argmin}_{\varphi \in \mathcal{F}}\ \mathrm{W}_2(\nabla \varphi \#\mu_s, \mu_t),

where :math:`\mathcal{F}` is the space functions that are on every set :math:`E_k` l-strongly convex
with an L-Lipschitz gradient, given :math:`(E_k)_{k \in [K]}` a partition of the ambient source space.

We perform the optimisation on a low amount of fitting samples and with few iterations,
since solving the SSNB problem is quite computationally expensive.

THIS EXAMPLE REQUIRES CVXPY

.. [58] François-Pierre Paty, Alexandre d’Aspremont, and Marco Cuturi. Regularity as regularization:
        Smooth and strongly convex brenier potentials in optimal transport. In International Conference
        on Artificial Intelligence and Statistics, pages 1222–1232. PMLR, 2020.

.. [59] Adrien B Taylor. Convex interpolation and performance estimation of first-order methods for
        convex optimization. PhD thesis, Catholic University of Louvain, Louvain-la-Neuve, Belgium,
        2017.
"""

# Author: Eloi Tanguy <eloi.tanguy@u-paris.fr>
# License: MIT License

# sphinx_gallery_thumbnail_number = 4

import matplotlib.pyplot as plt
import numpy as np
import ot

# %%
# Generating the fitting data
n_fitting_samples = 30
rng = np.random.RandomState(seed=0)
Xs = rng.uniform(-1, 1, size=(n_fitting_samples, 2))
Xs_classes = (Xs[:, 0] < 0).astype(int)
Xt = np.stack([Xs[:, 0] + 2 * np.sign(Xs[:, 0]), 2 * Xs[:, 1]], axis=-1)

plt.scatter(Xs[Xs_classes == 0, 0], Xs[Xs_classes == 0, 1], c='blue', label='source class 0')
plt.scatter(Xs[Xs_classes == 1, 0], Xs[Xs_classes == 1, 1], c='dodgerblue', label='source class 1')
plt.scatter(Xt[:, 0], Xt[:, 1], c='red', label='target')
plt.axis('equal')
plt.title('Splitting sphere dataset')
plt.legend(loc='upper right')
plt.show()

# %%
# Plotting image of barycentric projection (SSNB initialisation values)
plt.clf()
pi = ot.emd(ot.unif(n_fitting_samples), ot.unif(n_fitting_samples), ot.dist(Xs, Xt))
plt.scatter(Xs[:, 0], Xs[:, 1], c='dodgerblue', label='source')
plt.scatter(Xt[:, 0], Xt[:, 1], c='red', label='target')
bar_img = pi @ Xt
for i in range(n_fitting_samples):
    plt.plot([Xs[i, 0], bar_img[i, 0]], [Xs[i, 1], bar_img[i, 1]], color='black', alpha=.5)
plt.title('Images of in-data source samples by the barycentric map')
plt.legend(loc='upper right')
plt.axis('equal')
plt.show()

# %%
# Fitting the Nearest Brenier Potential
L = 3  # need L > 2 to allow the 2*y term, default is 1.4
phi, G = ot.mapping.nearest_brenier_potential_fit(Xs, Xt, Xs_classes, its=10, init_method='barycentric',
                                                  gradient_lipschitz_constant=L)

# %%
# Plotting the images of the source data
plt.clf()
plt.scatter(Xs[:, 0], Xs[:, 1], c='dodgerblue', label='source')
plt.scatter(Xt[:, 0], Xt[:, 1], c='red', label='target')
for i in range(n_fitting_samples):
    plt.plot([Xs[i, 0], G[i, 0]], [Xs[i, 1], G[i, 1]], color='black', alpha=.5)
plt.title('Images of in-data source samples by the fitted SSNB')
plt.legend(loc='upper right')
plt.axis('equal')
plt.show()

# %%
# Computing the predictions (images by nabla phi) for random samples of the source distribution
n_predict_samples = 50
Ys = rng.uniform(-1, 1, size=(n_predict_samples, 2))
Ys_classes = (Ys[:, 0] < 0).astype(int)
phi_lu, G_lu = ot.mapping.nearest_brenier_potential_predict_bounds(Xs, phi, G, Ys, Xs_classes, Ys_classes,
                                                                   gradient_lipschitz_constant=L)

# %%
# Plot predictions for the gradient of the lower-bounding potential
plt.clf()
plt.scatter(Xs[:, 0], Xs[:, 1], c='dodgerblue', label='source')
plt.scatter(Xt[:, 0], Xt[:, 1], c='red', label='target')
for i in range(n_predict_samples):
    plt.plot([Ys[i, 0], G_lu[0, i, 0]], [Ys[i, 1], G_lu[0, i, 1]], color='black', alpha=.5)
plt.title('Images of new source samples by $\\nabla \\varphi_l$')
plt.legend(loc='upper right')
plt.axis('equal')
plt.show()

# %%
# Plot predictions for the gradient of the upper-bounding potential
plt.clf()
plt.scatter(Xs[:, 0], Xs[:, 1], c='dodgerblue', label='source')
plt.scatter(Xt[:, 0], Xt[:, 1], c='red', label='target')
for i in range(n_predict_samples):
    plt.plot([Ys[i, 0], G_lu[1, i, 0]], [Ys[i, 1], G_lu[1, i, 1]], color='black', alpha=.5)
plt.title('Images of new source samples by $\\nabla \\varphi_u$')
plt.legend(loc='upper right')
plt.axis('equal')
plt.show()
