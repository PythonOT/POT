# -*- coding: utf-8 -*-
"""
=====================================
OT Barycenter with Generic Costs Demo
=====================================

This example illustrates the computation of an Optimal Transport Barycenter for
a ground cost that is not a power of a norm. We take the example of ground costs
:math:`c_k(x, y) = \|P_k(x)-y\|_2^2`, where :math:`P_k` is the (non-linear)
projection onto a circle k. This is an example of the fixed-point barycenter
solver introduced in [76] which generalises [20] and [43].

The ground barycenter function :math:`B(y_1, ..., y_K) = \mathrm{argmin}_{x \in
\mathbb{R}^2} \sum_k \lambda_k c_k(x, y_k)` is computed by gradient descent over
:math:`x` with Pytorch.

[76] Tanguy, Eloi and Delon, Julie and Gozlan, Nathaël (2024). Computing
Barycentres of Measures for Generic Transport Costs. arXiv preprint 2501.04016
(2024)

[20] Cuturi, M. and Doucet, A. (2014) Fast Computation of Wasserstein
Barycenters. InternationalConference in Machine Learning

[43] Álvarez-Esteban, Pedro C., et al. A fixed-point approach to barycenters in
Wasserstein space. Journal of Mathematical Analysis and Applications 441.2
(2016): 744-762.

"""

# Author: Eloi Tanguy <eloi.tanguy@math.cnrs.fr>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 1

# %%
# Generate data
import torch
from torch.optim import Adam
from ot.utils import dist
import numpy as np
from ot.lp import free_support_barycenter_generic_costs
import matplotlib.pyplot as plt


torch.manual_seed(42)

n = 200  # number of points of the of the barycentre
d = 2  # dimensions of the original measure
K = 4  # number of measures to barycentre
m = 50  # number of points of the measures
b_list = [torch.ones(m) / m] * K  # weights of the 4 measures
weights = torch.ones(K) / K  # weights for the barycentre
stop_threshold = 1e-20  # stop threshold for B and for fixed-point algo


# map R^2 -> R^2 projection onto circle
def proj_circle(X, origin, radius):
    diffs = X - origin[None, :]
    norms = torch.norm(diffs, dim=1)
    return origin[None, :] + radius * diffs / norms[:, None]


# circles on which to project
origin1 = torch.tensor([-1.0, -1.0])
origin2 = torch.tensor([-1.0, 2.0])
origin3 = torch.tensor([2.0, 2.0])
origin4 = torch.tensor([2.0, -1.0])
r = np.sqrt(2)
P_list = [
    lambda X: proj_circle(X, origin1, r),
    lambda X: proj_circle(X, origin2, r),
    lambda X: proj_circle(X, origin3, r),
    lambda X: proj_circle(X, origin4, r),
]

# measures to barycentre are projections of different random circles
# onto the K circles
Y_list = []
for k in range(K):
    t = torch.rand(m) * 2 * np.pi
    X_temp = 0.5 * torch.stack([torch.cos(t), torch.sin(t)], axis=1)
    X_temp = X_temp + torch.tensor([0.5, 0.5])[None, :]
    Y_list.append(P_list[k](X_temp))


# %%
# Define costs and ground barycenter function
# cost_list[k] is a function taking x (n, d) and y (n_k, d_k) and returning a
# (n, n_k) matrix of costs
def c1(x, y):
    return dist(P_list[0](x), y)


def c2(x, y):
    return dist(P_list[1](x), y)


def c3(x, y):
    return dist(P_list[2](x), y)


def c4(x, y):
    return dist(P_list[3](x), y)


cost_list = [c1, c2, c3, c4]


# batched total ground cost function for candidate points x (n, d)
# for computation of the ground barycenter B with gradient descent
def C(x, y):
    """
    Computes the barycenter cost for candidate points x (n, d) and
    measure supports y: List(n, d_k).
    """
    n = x.shape[0]
    K = len(y)
    out = torch.zeros(n)
    for k in range(K):
        out += (1 / K) * torch.sum((P_list[k](x) - y[k]) ** 2, axis=1)
    return out


# ground barycenter function
def B(y, its=150, lr=1, stop_threshold=stop_threshold):
    """
    Computes the ground barycenter for measure supports y: List(n, d_k).
    Output: (n, d) array
    """
    x = torch.randn(n, d)
    x.requires_grad_(True)
    opt = Adam([x], lr=lr)
    for _ in range(its):
        x_prev = x.data.clone()
        opt.zero_grad()
        loss = torch.sum(C(x, y))
        loss.backward()
        opt.step()
        diff = torch.sum((x.data - x_prev) ** 2)
        if diff < stop_threshold:
            break
    return x


# %%
# Compute the barycenter measure
fixed_point_its = 3
X_init = torch.rand(n, d)
X_bar = free_support_barycenter_generic_costs(
    Y_list,
    b_list,
    X_init,
    cost_list,
    B,
    numItermax=fixed_point_its,
    stopThr=stop_threshold,
)

# %%
# Plot Barycenter (Iteration 3)
alpha = 0.4
s = 80
labels = ["circle 1", "circle 2", "circle 3", "circle 4"]
for Y, label in zip(Y_list, labels):
    plt.scatter(*(Y.numpy()).T, alpha=alpha, label=label, s=s)
plt.scatter(
    *(X_bar.detach().numpy()).T, label="Barycenter", c="black", alpha=alpha, s=s
)
plt.axis("equal")
plt.xlim(-0.3, 1.3)
plt.ylim(-0.3, 1.3)
plt.axis("off")
plt.legend()
plt.tight_layout()

# %%
