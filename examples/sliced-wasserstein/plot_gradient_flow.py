# -*- coding: utf-8 -*-
"""
==========================
2D Gradient Flow using SWD
==========================

This example illustrates using the  sliced Wasserstein Distance proposed in [31] for points registration.

[31] Bonneel, Nicolas, et al. "Sliced and radon wasserstein barycenters of measures." Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45

"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#
# License: MIT License

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import torch

import ot.torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = "cpu"

##############################################################################
# Generate data
# -------------

# %% parameters and data generation

n = 500  # nb samples

mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

mu_t = np.array([4, 4])
cov_t = np.array([[1, -.8], [-.8, 1]])

xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)
##################
# Compute the flow
# ----------------

# %% Parameters
n_steps = 100
lr = 1e-1
n_projections = 10
seed = 0
p = 2
random_generator = torch.Generator(device).manual_seed(seed)

all_xs = np.empty((n_steps, *xs.shape))
all_grads = np.empty((n_steps, *xs.shape))


# %% Compute the flow

curr_x = torch.tensor(xs, device=device, requires_grad=True)
device_xt = torch.tensor(xt, device=device)
for i in range(n_steps):
    all_xs[i] = curr_x.detach().cpu().numpy()
    loss = ot.torch.ot_loss_sliced(curr_x, device_xt, p=2, n_projections=n_projections, seed=random_generator)
    grad = torch.autograd.grad(loss, curr_x)[0]
    all_grads[i] = grad.cpu().numpy()
    curr_x.data -= lr * len(xs) * grad


###########
# Plot data
# ---------
fig, ax = plt.subplots()
source_scatter, = ax.plot(*xs.T, '+b', label='Source samples')
target_scatter, = ax.plot(*xt.T, 'xr', label='Target samples')
text = ax.text(0.75, 0.95,  '', transform=ax.transAxes)
ax.legend(loc="upper left")

# %% Initialize the animation
def init():
    text.set_text("")
    source_scatter.set_data(*xs.T)
    return source_scatter, text


# %% Utility function to display the flow
def update_plot(num):
    text.set_text(f"epoch {num + 1}/{n_steps}")
    source_scatter.set_data(*all_xs[num].T)
    return source_scatter, text


# %% compute the the flow

ani = anim.FuncAnimation(fig, update_plot, frames=n_steps,
                         init_func=init, blit=True, interval=150, repeat=False)

