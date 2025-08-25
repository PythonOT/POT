# -*- coding: utf-8 -*-
"""
==================
Batch parallel OT
==================

Shows the efficiency of using parallel OT solvers for optimal transport.

"""

# Author: Paul Krzakala <paul.krzakala@gmail.com>
# License: MIT License

from ot.batch._linear import *
from ot.batch._quadratic import *
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import torch

b = 256
n = 40
d_nodes = 16
d_edges = 16
epsilon = 0.1
max_iter = 10
tol = 1e-5
max_iter_inner = 10
tol_inner = 1e-5
alpha = 0.5

X1 = torch.randn(b, n, d_nodes, dtype=torch.float32)
C1 = torch.randn(b, n, n, d_edges, dtype=torch.float32)

X2 = X1[:, torch.randperm(n), :]
perm = torch.randperm(n)
C2 = C1[:, perm, :, :][:, :, perm, :]

M = cost_matrix_l2_batch(X1, X2)

# Compute using for loops and batches of size 1
start = perf_counter()
average_cost = 0
for i in range(b):
    res = solve_gromov_batch(
        alpha=alpha,
        epsilon=epsilon,
        M=M[i : i + 1],
        C1=C1[i : i + 1],
        C2=C2[i : i + 1],
        max_iter=max_iter,
        tol=tol,
        max_iter_inner=max_iter_inner,
        tol_inner=tol_inner,
    )
    average_cost += res.value.item()
average_cost /= b

end = perf_counter()
print(f"Quadratic solver naive (CPU): {end - start:.4f} seconds")
print(f"Average cost (CPU): {average_cost:.4f}")
print("")

start = perf_counter()
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
average_cost = res.value.mean().item()
end = perf_counter()
print(f"Quadratic solver batch (CPU): {end - start:.4f} seconds")
print(f"Average cost (CPU): {average_cost:.4f}")
print("")

if torch.cuda.is_available():
    M, C1, C2 = M.cuda(), C1.cuda(), C2.cuda()
    # GPU Warmup
    print("Warming up GPU...")
    for _ in range(100):
        solve_gromov_batch(
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
    print("Done.")
    start = perf_counter()
    out = solve_gromov_batch(
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
    average_cost = out["cost"].mean().item()
    end = perf_counter()
    print(f"Quadratic solver batch (GPU): {end - start:.4f} seconds")
    print(f"Average cost (GPU): {average_cost:.4f}")
