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

# ----------------------------
# Parameters
# ----------------------------

FRAMES = 100
INTERVAL = 20


def sample_ball(n, radius=1.0, center=(0.0, 0.0)):
    theta = 2 * np.pi * np.random.rand(n)
    r = radius * np.sqrt(np.random.rand(n))

    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]

    return np.stack((x, y), axis=1)


def sample_two_balls(n, radius=1.0, centers=((-1, 0), (1, 0))):
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

# ----------------------------
# Bijection
# ----------------------------
start = time.time()
cost, perm, _ = ot.bsp.bsp_wrap.bsp_solve(A, B, 64)
print(
    "Bijection computed between {} points, with cost {} in {}s".format(
        N, cost, time.time() - start
    )
)

# Reorder B according to bijection
B_perm = B[perm]

# ----------------------------
# Setup figure
# ----------------------------
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
