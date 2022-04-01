# -*- coding: utf-8 -*-
r"""
======================================================================
Continuous OT plan estimation with Pytorch
======================================================================


"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 3

import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as animation
import torch
from torch import nn
import ot
import ot.plot

# %%
# Data generation
# ---------------

torch.manual_seed(42)
np.random.seed(42)

n_source_samples = 300
n_target_samples = 300
theta = 2 * np.pi / 20
noise_level = 0.1

Xs = np.random.randn(n_source_samples, 2) * 0.5
Xt = np.random.randn(n_target_samples, 2) * 2

# one of the target mode changes its variance (no linear mapping)
Xt = Xt + 4


# %%
# Plot data
# ---------

pl.figure(1, (5, 5))
pl.clf()
pl.scatter(Xs[:, 0], Xs[:, 1], marker='+', label='Source samples')
pl.scatter(Xt[:, 0], Xt[:, 1], marker='o', label='Target samples')
pl.legend(loc=0)
ax_bounds = pl.axis()
pl.title('Source and target distributions')

# %%
# Convert data to torch tensors
# -----------------------------

xs = torch.tensor(Xs)
xt = torch.tensor(Xt)

# %%
# Estimating dual variables for entropic OT
# -----------------------------------------

torch.manual_seed(42)

# define the MLP model


class Potential(torch.nn.Module):
    def __init__(self):
        super(Potential, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 1)
        self.relu = torch.nn.ReLU()  # instead of Heaviside step fn

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)  # instead of Heaviside step fn
        output = self.fc2(output)
        return output.ravel()


u = Potential().double()
v = Potential().double()

reg = 1

optimizer = torch.optim.Adam(list(u.parameters()) + list(v.parameters()), lr=.005)

# number of iteration
n_iter = 2000


losses = []

for i in range(n_iter):

    # generate noise samples

    # minus because we maximize te dual loss
    loss = -ot.stochastic.loss_dual_entropic(u(xs), v(xt), xs, xt, reg=reg)
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

Ge = ot.stochastic.plan_dual_entropic(u(xs), v(xt), xs, xt, reg=reg)

# %%
# Plot teh estimated entropic OT plan
# -----------------------------------

# pl.figure(3, (5, 5))
# pl.clf()
# ot.plot.plot2D_samples_mat(Xs, Xt, Ge.detach().numpy(), alpha=0.1)
# pl.scatter(Xs[:, 0], Xs[:, 1], marker='+', label='Source samples', zorder=2)
# pl.scatter(Xt[:, 0], Xt[:, 1], marker='o', label='Target samples', zorder=2)
# pl.legend(loc=0)
# ax_bounds = pl.axis()
# pl.title('Entropic OT plan')


#%%
nv = 100
xl = np.linspace(ax_bounds[0], ax_bounds[1], nv)
yl = np.linspace(ax_bounds[2], ax_bounds[3], nv)

XX, YY = np.meshgrid(xl, yl)

xg = np.concatenate((XX.ravel()[:, None], YY.ravel()[:, None]), axis=1)

wxg = np.exp(-((xg[:, 0] - 4)**2 + (xg[:, 1] - 4)**2) / (2 * 2))
wxg = wxg / np.sum(wxg)

xg = torch.tensor(xg)
wxg = torch.tensor(wxg)


pl.figure(4, (15, 5))
pl.clf()
pl.subplot(1, 3, 1)

iv = 0
Gg = ot.stochastic.plan_dual_entropic(u(xs[iv:iv + 1, :]), v(xg), xs[iv:iv + 1, :], xg, reg=reg, wt=wxg)
Gg = Gg.reshape((nv, nv)).detach().numpy()

pl.scatter(Xs[:, 0], Xs[:, 1], marker='+', label='Source samples', zorder=2, alpha=0.1)
pl.scatter(Xt[:, 0], Xt[:, 1], marker='o', label='Target samples', zorder=2, alpha=0.1)
pl.scatter(Xs[iv:iv + 1, 0], Xs[iv:iv + 1, 1], marker='+', label='Source sample', zorder=2, alpha=1, color='C0')
pl.pcolormesh(XX, YY, Gg, cmap='Blues', label='Density of transported sourec sample')
pl.legend(loc=0)
ax_bounds = pl.axis()
pl.title('Density of transported source sample')

pl.subplot(1, 3, 2)

iv = 1
Gg = ot.stochastic.plan_dual_entropic(u(xs[iv:iv + 1, :]), v(xg), xs[iv:iv + 1, :], xg, reg=reg, wt=wxg)
Gg = Gg.reshape((nv, nv)).detach().numpy()

pl.scatter(Xs[:, 0], Xs[:, 1], marker='+', label='Source samples', zorder=2, alpha=0.1)
pl.scatter(Xt[:, 0], Xt[:, 1], marker='o', label='Target samples', zorder=2, alpha=0.1)
pl.scatter(Xs[iv:iv + 1, 0], Xs[iv:iv + 1, 1], marker='+', label='Source sample', zorder=2, alpha=1, color='C0')
pl.pcolormesh(XX, YY, Gg, cmap='Blues', label='Density of transported sourec sample')
pl.legend(loc=0)
ax_bounds = pl.axis()
pl.title('Density of transported source sample')

pl.subplot(1, 3, 3)

iv = 6
Gg = ot.stochastic.plan_dual_entropic(u(xs[iv:iv + 1, :]), v(xg), xs[iv:iv + 1, :], xg, reg=reg, wt=wxg)
Gg = Gg.reshape((nv, nv)).detach().numpy()

pl.scatter(Xs[:, 0], Xs[:, 1], marker='+', label='Source samples', zorder=2, alpha=0.1)
pl.scatter(Xt[:, 0], Xt[:, 1], marker='o', label='Target samples', zorder=2, alpha=0.1)
pl.scatter(Xs[iv:iv + 1, 0], Xs[iv:iv + 1, 1], marker='+', label='Source sample', zorder=2, alpha=1, color='C0')
pl.pcolormesh(XX, YY, Gg, cmap='Blues', label='Density of transported sourec sample')
pl.legend(loc=0)
ax_bounds = pl.axis()
pl.title('Density of transported source sample')
