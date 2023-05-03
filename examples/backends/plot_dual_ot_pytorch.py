# -*- coding: utf-8 -*-
r"""
======================================================================
Dual OT solvers for entropic and quadratic regularized OT with Pytorch
======================================================================


"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 3

import numpy as np
import matplotlib.pyplot as pl
import torch
import ot
import ot.plot

# %%
# Data generation
# ---------------

torch.manual_seed(1)

n_source_samples = 100
n_target_samples = 100
theta = 2 * np.pi / 20
noise_level = 0.1

Xs, ys = ot.datasets.make_data_classif(
    'gaussrot', n_source_samples, nz=noise_level)
Xt, yt = ot.datasets.make_data_classif(
    'gaussrot', n_target_samples, theta=theta, nz=noise_level)

# one of the target mode changes its variance (no linear mapping)
Xt[yt == 2] *= 3
Xt = Xt + 4


# %%
# Plot data
# ---------

pl.figure(1, (10, 5))
pl.clf()
pl.scatter(Xs[:, 0], Xs[:, 1], marker='+', label='Source samples')
pl.scatter(Xt[:, 0], Xt[:, 1], marker='o', label='Target samples')
pl.legend(loc=0)
pl.title('Source and target distributions')

# %%
# Convert data to torch tensors
# -----------------------------

xs = torch.tensor(Xs)
xt = torch.tensor(Xt)

# %%
# Estimating dual variables for entropic OT
# -----------------------------------------

u = torch.randn(n_source_samples, requires_grad=True)
v = torch.randn(n_source_samples, requires_grad=True)

reg = 0.5

optimizer = torch.optim.Adam([u, v], lr=1)

# number of iteration
n_iter = 200


losses = []

for i in range(n_iter):

    # generate noise samples

    # minus because we maximize te dual loss
    loss = -ot.stochastic.loss_dual_entropic(u, v, xs, xt, reg=reg)
    losses.append(float(loss.detach()))

    if i % 10 == 0:
        print("Iter: {:3d}, loss={}".format(i, losses[-1]))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


pl.figure(2)
pl.plot(losses)
pl.grid()
pl.title('Dual objective (negative)')
pl.xlabel("Iterations")

Ge = ot.stochastic.plan_dual_entropic(u, v, xs, xt, reg=reg)

# %%
# Plot the estimated entropic OT plan
# -----------------------------------

pl.figure(3, (10, 5))
pl.clf()
ot.plot.plot2D_samples_mat(Xs, Xt, Ge.detach().numpy(), alpha=0.1)
pl.scatter(Xs[:, 0], Xs[:, 1], marker='+', label='Source samples', zorder=2)
pl.scatter(Xt[:, 0], Xt[:, 1], marker='o', label='Target samples', zorder=2)
pl.legend(loc=0)
pl.title('Source and target distributions')


# %%
# Estimating dual variables for quadratic OT
# ------------------------------------------

u = torch.randn(n_source_samples, requires_grad=True)
v = torch.randn(n_source_samples, requires_grad=True)

reg = 0.01

optimizer = torch.optim.Adam([u, v], lr=1)

# number of iteration
n_iter = 200


losses = []


for i in range(n_iter):

    # generate noise samples

    # minus because we maximize te dual loss
    loss = -ot.stochastic.loss_dual_quadratic(u, v, xs, xt, reg=reg)
    losses.append(float(loss.detach()))

    if i % 10 == 0:
        print("Iter: {:3d}, loss={}".format(i, losses[-1]))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


pl.figure(4)
pl.plot(losses)
pl.grid()
pl.title('Dual objective (negative)')
pl.xlabel("Iterations")

Gq = ot.stochastic.plan_dual_quadratic(u, v, xs, xt, reg=reg)


# %%
# Plot the estimated quadratic OT plan
# ------------------------------------

pl.figure(5, (10, 5))
pl.clf()
ot.plot.plot2D_samples_mat(Xs, Xt, Gq.detach().numpy(), alpha=0.1)
pl.scatter(Xs[:, 0], Xs[:, 1], marker='+', label='Source samples', zorder=2)
pl.scatter(Xt[:, 0], Xt[:, 1], marker='o', label='Target samples', zorder=2)
pl.legend(loc=0)
pl.title('OT plan with quadratic regularization')
