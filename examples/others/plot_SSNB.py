# -*- coding: utf-8 -*-
# sphinx_gallery_thumbnail_number = 2
r"""
=====================================================
Smooth and Strongly Convex Nearest Brenier Potentials
=====================================================

This example is designed to show how to use SSNB [58] in POT.
SSNB computes an l-strongly convex potential :math:`\varphi` with an L-Lipschitz gradient such that
:math:`\nabla \varphi \# \mu \approx \nu`. This regularity can be enforced only on the components of a partition
of the ambient space, which is a relaxation compared to imposing global regularity.

In this example, we consider a source measure :math:`\mu_s` which is the uniform measure on the unit sphere in
:math:`\mathbb{R}^2`, and the target measure :math:`\mu_t` which is the image of :math:`\mu_x` by
:math:`T(x_1, x_2) = (x_1 + 2\mathrm{sign}(x_2), x_2)`. The map :math:`T` is non-smooth, and we wish to approximate it
using a "Brenier-style" map :math:`\nabla \varphi` which is regular on the partition
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

import matplotlib.pyplot as plt
import numpy as np
import ot

# %%
# Generating the fitting data
n_fitting_samples = 16
t = np.linspace(0, 2 * np.pi, n_fitting_samples)
r = 1
Xs = np.stack([r * np.cos(t), r * np.sin(t)], axis=-1)
Xs_classes = (Xs[:, 0] < 0).astype(int)
Xt = np.stack([Xs[:, 0] + 2 * np.sign(Xs[:, 0]), Xs[:, 1]], axis=-1)

plt.scatter(Xs[Xs_classes == 0, 0], Xs[Xs_classes == 0, 1], c='blue', label='source class 0')
plt.scatter(Xs[Xs_classes == 1, 0], Xs[Xs_classes == 1, 1], c='dodgerblue', label='source class 1')
plt.scatter(Xt[:, 0], Xt[:, 1], c='red', label='target')
plt.title('Splitting sphere dataset')
plt.legend(loc='upper right')
plt.show()

# %%
# Fitting the Nearest Brenier Potential
phi, G = ot.nearest_brenier_potential_fit(Xs, Xt, Xs_classes, its=10, seed=0)

# %%
# Computing the predictions (images by nabla phi) for random samples of the source distribution
rng = np.random.RandomState(seed=0)
n_predict_samples = 100
t = rng.uniform(0, 2 * np.pi, size=n_predict_samples)
r = rng.uniform(size=n_predict_samples)
Ys = np.stack([r * np.cos(t), r * np.sin(t)], axis=-1)
Ys_classes = (Ys[:, 0] < 0).astype(int)
phi_lu, G_lu = ot.nearest_brenier_potential_predict_bounds(Xs, phi, G, Ys, Xs_classes, Ys_classes)

# %%
# Plot predictions for the gradient of the lower-bounding potential
plt.clf()
plt.scatter(Xs[:, 0], Xs[:, 1], c='dodgerblue', label='source')
plt.scatter(Xt[:, 0], Xt[:, 1], c='red', label='target')
for i in range(n_predict_samples):
    plt.plot([Ys[i, 0], G_lu[0, i, 0]], [Ys[i, 1], G_lu[0, i, 1]], color='black', alpha=.5)
plt.title('Images of new source samples by $\\nabla \\varphi_l$')
plt.legend(loc='upper right')
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
plt.show()
