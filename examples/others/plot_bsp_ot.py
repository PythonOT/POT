"""
===================
Fast and accurate bijections using BSP-OT example
===================

This example shows how to use the BSP-OT solver to compute a bijection
between two large point clouds, and animate the morphing between them.

[?] Genest, B., Bonneel, N., Nivoliers, V., Coeurjolly, D.
BSP-OT: Sparse transport plans between discrete measures in log-linear time
ACM Transactions on Graphics, Siggraph Asia (2025).

"""

# Author: Baptiste Genest <baptistegenest@gmail.com>
#
# License: MIT License

import ot.bsp
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch


##############################################################################
# Data generation
# ----------------------------------


def sample_ball(n, radius=1.0, center=(0.0, 0.0)):
    theta = 2 * np.pi * np.random.rand(n)
    r = radius * np.sqrt(np.random.rand(n))

    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]

    return np.stack((x, y), axis=1)


def sample_two_balls(n, radius=1.0, centers=((-1, 0), (1, 1))):
    assert n % 2 == 0, "n must be even"

    n_half = n // 2
    X1 = sample_ball(n_half, radius, centers[0])
    X2 = sample_ball(n_half, radius, centers[1])

    return np.vstack((X1, X2))


# ----------------------------
# Load point clouds
# ----------------------------
N = 100000
A = sample_ball(N)
B = sample_two_balls(N, 0.5)

N = A.shape[0]


##############################################################################
# Bijection computation
# ----------------------------------

start = time.time()

# %% call BSP-OT solver
# The solver returns the transport cost, the final bijection and the
# intermediary ones used to compute the final one (here we set k = 64).
# Here we only use the final bijection.
cost, perm, _ = ot.bsp.bsp_solve(A, B, 64, 2)
print(
    "Bijection computed between {} points, with cost {} in {}s".format(
        N, cost, time.time() - start
    )
)

# %% Reorder B according to bijection
# such that the points are in correspondence along the morphing animation
# using simply A*(1-t) + B*t
B_perm = B[perm]

##############################################################################
# Setup animation
# ----------------------------

# Animation parameters
FRAMES = 100
INTERVAL = 20

# Plot setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")
ax.set_title("Bijective Point Cloud Morphing")

scat = ax.scatter(A[:, 0], A[:, 1], s=0.05, c="tab:blue")

all_pts = np.vstack([A, B_perm])
pad = 0.5
ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)


# ----------------------------
# Animation update
# ----------------------------
def update(frame):
    t = frame / (FRAMES - 1)
    t = t * t * (3 - 2 * t)
    P = (1 - t) * A + t * B_perm
    scat.set_offsets(P)
    return (scat,)


# ----------------------------
# Run animation
# ----------------------------
ani = FuncAnimation(fig, update, frames=FRAMES, interval=INTERVAL, blit=True)

plt.show()
