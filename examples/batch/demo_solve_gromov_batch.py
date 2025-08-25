# -*- coding: utf-8 -*-
"""
==================
Batch parallel linear OT
==================

Shows how to use the batch parallel quadratic OT solvers.

"""

from ot.batch._linear import *
from ot.batch._quadratic import *
import numpy as np
import matplotlib.pyplot as plt

b = 32
n = 20
d_nodes = 4
d_edges = 2
epsilon = 0.1
max_iter = 10
tol = 1e-5
max_iter_inner = 10
tol_inner = 1e-5

X1 = np.random.randn(b, n, d_nodes).astype("float32")
C1 = np.random.randn(b, n, n, d_edges).astype("float32")

X2 = X1[:, np.random.permutation(n), :]
permutation = np.random.permutation(n)
C2 = C1[:, permutation, :, :][:, :, permutation, :]

M = cost_matrix_l2_batch(X1, X2)

cost_linear_list = []
cost_quadratic_list = []
alpha_list = np.linspace(0.001, 0.999, 20)

for alpha in alpha_list:
    res = solve_gromov_batch(
        alpha=alpha,
        epsilon=epsilon,
        M=M,
        C1=C1,
        C2=C2,
        max_iter=max_iter,
        tol=tol,
        max_iter_inner=max_iter_inner,
        tol_inner=tol_inner,
    )

    cost_linear = res.value_linear.mean() / (1 - alpha)
    cost_quadratic = res.value_quad.mean() / (alpha)
    cost_linear_list.append(cost_linear)
    cost_quadratic_list.append(cost_quadratic)

plt.figure(figsize=(10, 6))
plt.plot(alpha_list, cost_linear_list, marker="o", label="Cost Linear")
plt.plot(alpha_list, cost_quadratic_list, marker="o", label="Cost Quadratic")
plt.xlabel("Alpha")
plt.ylabel("Average Cost")
plt.legend()
plt.title("Cost vs Alpha")
plt.show()
