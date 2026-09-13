# -*- coding: utf-8 -*-
"""
=========================================================
UnifOrtho Sliced Wasserstein in high dimension
=========================================================

This example illustrates the UnifOrtho sampling scheme for Sliced
Wasserstein directions, introduced in [97] and recommended for large
dimensions by a recent numerical and theoretical study [98], and compares
it to the default uniform (Monte Carlo) sampling of slicing directions.

Sliced Wasserstein (SWD) approximates the Wasserstein distance by averaging
1D Wasserstein distances over projections onto random directions
:math:`\\theta` drawn uniformly on the sphere. By default these directions
are sampled purely at random (Monte Carlo), which introduces some variance
in the estimate for a given number of projections.

UnifOrtho takes a different route that works in *any* dimension:
instead of drawing directions independently, it draws them in blocks of ``dim`` directions,
each block is a random orthonormal basis (a draw from the Haar measure on the
orthogonal group :math:`O(\\mathrm{dim})`). Directions within a block are
therefore exactly mutually orthogonal, spreading them out much more evenly
than independent draws would.

We first visualize this block structure on the ordinary 3D sphere, purely
for intuition -- dimension 3 is precisely where QSW/RQSW should be
preferred in practice, not UnifOrtho. We then measure convergence to the
true Sliced Wasserstein distance in a genuinely high dimension, where
UnifOrtho is the recommended choice.

.. [97] Rowland, M., Hron, J., Tang, Y., Choromanski, K., Sarlos, T., &
    Weller, A. (2019). Orthogonal Estimation of Wasserstein Distances.
    Proceedings of the 22nd International Conference on Artificial
    Intelligence and Statistics (AISTATS), PMLR 89.
.. [98] Petrovic, V., Bardenet, R., & Desolneux, A. (2025). Repulsive
    Monte Carlo on the sphere for the sliced Wasserstein distance.
    arXiv:2509.10166.
"""

# Author: Samuel Vangu <samuelvangu0@gmail.com>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 1

import numpy as np
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3D projection)

import ot
from ot.sliced import get_random_projections, get_projections_uniortho

##############################################################################
# Visualize the block structure on the sphere (d=3, for intuition only)
# -----------------------------------------------------------------------
# We draw 15 directions on :math:`S^2` with each scheme:
#
# - ``uniform``: directions are Gaussian vectors normalized to unit norm
#   (standard Monte Carlo sampling of the sphere) -- no structure between
#   the points.
# - ``unif_ortho``: 5 independent blocks of 3 mutually orthogonal
#   directions each. Each block is colored separately below, so that
#   same-colored points are exactly orthogonal to one another -- this is
#   the structure that is not visible in the uniform sample.
#
# Dimension 3 is used here only because it is the one human beings can
# actually look at. It is *not* the dimension UnifOrtho is recommended
# for -- see the convergence experiment below.

d = 3
n_blocks = 15
n_projections = n_blocks * d
seed = 42

theta_uniform = get_random_projections(d, n_projections, seed=seed)
theta_uniortho = get_projections_uniortho(d, n_projections, seed=seed)

fig = pl.figure(1, figsize=(10, 5))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.scatter(
    theta_uniform[0], theta_uniform[1], theta_uniform[2], c="gray", s=25, alpha=0.8
)
ax1.set_title("Uniform (Monte Carlo)")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
block_ids = np.repeat(np.arange(n_blocks), d)
ax2.scatter(
    theta_uniortho[0],
    theta_uniortho[1],
    theta_uniortho[2],
    c=block_ids,
    cmap="tab10",
    s=25,
    alpha=0.9,
)
ax2.set_title("UnifOrtho (one color per orthogonal block)")

for ax in (ax1, ax2):
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

pl.tight_layout()
pl.show()

# Every triple of same-colored points on the right is an exact orthogonal
# basis of R^3 -- three mutually perpendicular directions. The uniform
# sample on the left has no such guarantee: any two of its points can end
# up arbitrarily close to each other.

##############################################################################
# Convergence to the true Sliced Wasserstein distance, in high dimension
# --------------------------------------------------------------------------
# As with the QSW/RQSW example, we build ``Xt`` as a pure translation of
# ``Xs`` by a fixed vector :math:`\delta`, which makes the true Sliced
# Wasserstein distance known exactly, with zero approximation error left
# except from the number of projections:
#
# .. math::
#     \mathcal{SWD}_2(\mu, \nu) = \frac{\|\delta\|}{\sqrt{d}}
#
# This time we work in dimension 30 -- the regime UnifOrtho is recommended
# for . The list of projection counts below is deliberately chosen to never be a multiple of ``d``:
# when ``n_projections`` is an exact multiple of 30, UnifOrtho draws a
# whole number of *complete* orthogonal bases, which for this particular
# translation experiment happens to recover the exact answer up to
# floating-point precision (an identity, not an approximation) -- an
# interesting fact in its own right, but not representative of the
# typical case we want to illustrate here.

d = 30
rng = np.random.RandomState(0)

n_samples = 200
delta = rng.normal(size=d) * 1.2
Xs = rng.uniform(-2, 2, (n_samples, d))
Xt = Xs + delta

# Exact reference: no approximation at all, at any cost.
sw_true = np.linalg.norm(delta) / np.sqrt(d)

n_proj_list = [35, 65, 95, 190, 380, 760]  # all >= d, none a multiple of d
n_trials = 10

errors_uniform = np.zeros((n_trials, len(n_proj_list)))
errors_uniortho = np.zeros((n_trials, len(n_proj_list)))

for j, n_proj in enumerate(n_proj_list):
    for t in range(n_trials):
        sw_uniform = ot.sliced_wasserstein_distance(
            Xs, Xt, n_projections=n_proj, sampling_slices="uniform", seed=t
        )
        sw_uniortho = ot.sliced_wasserstein_distance(
            Xs, Xt, n_projections=n_proj, sampling_slices="unif_ortho", seed=t
        )
        errors_uniform[t, j] = np.abs(sw_uniform - sw_true)
        errors_uniortho[t, j] = np.abs(sw_uniortho - sw_true)

mean_err_uniform = errors_uniform.mean(axis=0)
std_err_uniform = errors_uniform.std(axis=0)
mean_err_uniortho = errors_uniortho.mean(axis=0)
std_err_uniortho = errors_uniortho.std(axis=0)

pl.figure(2, figsize=(6, 5))
pl.plot(n_proj_list, mean_err_uniform, "o-", label="Uniform (MC)")
pl.fill_between(
    n_proj_list,
    mean_err_uniform - std_err_uniform,
    mean_err_uniform + std_err_uniform,
    alpha=0.3,
)
pl.plot(n_proj_list, mean_err_uniortho, "s-", label="UnifOrtho")
pl.fill_between(
    n_proj_list,
    mean_err_uniortho - std_err_uniortho,
    mean_err_uniortho + std_err_uniortho,
    alpha=0.3,
)
pl.xscale("log")
pl.yscale("log")
pl.xlabel("Number of projections")
pl.ylabel("Absolute error to the true SWD")
pl.title(f"Convergence of the Sliced Wasserstein estimate (d={d})")
pl.legend()
pl.show()

# UnifOrtho consistently reaches a given accuracy with markedly fewer
# projections than uniform sampling in this high-dimensional setting --
# the opposite of the low-dimensional case, where QSW/RQSW are the better choice.
# As a rule of thumb from the literature [98]: prefer RQSW in low dimension, UnifOrtho in high dimension, and
# either may do in between. Since UnifOrtho remains an unbiased,
# stochastic estimator (like RQSW), it is also a drop-in replacement for
# uniform sampling in stochastic optimization settings.

# %%
