# -*- coding: utf-8 -*-
"""
==================
Batch parallel linear OT
==================

Shows how to use the batch parallel linear OT solvers.

"""

# Author: Paul Krzakala <paul.krzakala@gmail.com>
# License: MIT License

from ot.batch._linear import *
from ot.batch._quadratic import *
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
b = 4
n = 20
d = 4
noise = 0.1
max_iter = 10000
tol = 1e-5

# Generate random data
X = np.random.randn(b, n, d).astype("float32")
Y = np.random.randn(b, n, d).astype("float32")
M = cost_matrix_l2_batch(X, Y)

# Define grid search
epsilons = np.logspace(-2, 1, num=10)

plt.figure(figsize=(10, 6))

cost_list = []
n_iter_list = []
for epsilon in epsilons:
    res = linear_solver_batch(
        epsilon=epsilon, M=M, max_iter=max_iter, tol=tol, log_dual=True
    )
    n_iter = res.log["n_iter"]
    cost = res.value_linear.mean()
    cost_list.append(cost)
    n_iter_list.append(n_iter)

ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(epsilons, n_iter_list, "r-o", label="Iterations")
ax2.plot(epsilons, cost_list, "b-s", label="Cost")

ax1.set_xlabel("Epsilon")
ax1.set_ylabel("Iterations", color="r")
ax2.set_ylabel("Cost", color="b")
ax1.set_xscale("log")

ax1.tick_params(axis="y", labelcolor="r")
ax2.tick_params(axis="y", labelcolor="b")

plt.title("Epsilon vs Iters and Cost")
plt.show()
