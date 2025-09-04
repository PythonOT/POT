# -*- coding: utf-8 -*-
"""
==================
Batch parallel linear OT
==================

Showcase the importance of correctly setting the "symmetric" boolean.

"""

from ot.batch._linear import *
from ot.batch._quadratic import *
import numpy as np
import torch
from time import perf_counter

np.random.seed(0)


def demo_symmetric():
    print("")
    print("Generate random data with C1 and C2 NOT symmetric...")
    b = 256
    n = 40
    epsilon = 0.1
    max_iter = 10
    tol = 1e-5
    max_iter_inner = 10
    tol_inner = 1e-5

    C1 = torch.randn(b, n, n, dtype=torch.float32)

    perm = torch.randperm(n)
    C2 = C1[:, perm, :][:, :, perm]
    C2 = torch.randn(b, n, n, dtype=torch.float32)

    print("")

    print("Solving gromov with symmetric=True...")
    start = perf_counter()
    res = solve_gromov_batch(
        epsilon=epsilon,
        C1=C1,
        C2=C2,
        max_iter=max_iter,
        tol=tol,
        max_iter_inner=max_iter_inner,
        tol_inner=tol_inner,
        symmetric=True,
    )
    end = perf_counter()
    n_iter = res.log["n_iter"]
    print(f"... solver took: {n_iter} iterations.")
    print(f"... solver took: {end - start:.4f} seconds")
    print(f"... final value: {res.value.mean().item():.4f}")

    print("")

    print("Solving gromov with symmetric=False...")
    start = perf_counter()
    res = solve_gromov_batch(
        epsilon=epsilon,
        C1=C1,
        C2=C2,
        max_iter=max_iter,
        tol=tol,
        max_iter_inner=max_iter_inner,
        tol_inner=tol_inner,
        symmetric=False,
    )
    end = perf_counter()
    n_iter = res.log["n_iter"]
    print(f"... solver took: {n_iter} iterations.")
    print(f"... solver took: {end - start:.4f} seconds")
    print(f"... final value: {res.value.mean().item():.4f}")

    print("")
    print(
        "Conclusion: symmetric=False is slightly faster but leads to suboptimal values because some incorrect approximations are made."
    )


def demo_recompute_const():
    print("")
    print("Generate random data...")
    b = 64
    n = 64
    d_nodes = 4
    d_edges = 2

    C1 = np.random.randn(b, n, n, d_edges).astype("float32")
    C2 = np.random.randn(b, n, n, d_edges).astype("float32")

    p = np.ones((b, n), dtype=np.float32) / n
    q = np.ones((b, n), dtype=np.float32) / n

    def compute_loss(T, recompute_const=False):
        metric = QuadraticEuclidean()
        L = metric.cost_tensor(p, q, C1, C2, symmetric=True)
        LT = tensor_product(L, T, recompute_const=recompute_const, symmetric=True)
        loss = (LT * T).sum((1, 2))
        return loss

    print("")
    print("Generate random coupling matrix T that does not satisfy marginals...")
    T = np.random.rand(b, n, n).astype("float32")

    print("")
    print("Compute the gromov loss - recompute_const=False")
    loss = compute_loss(T, recompute_const=False)
    is_pos = (loss >= -1e-6).all()
    print("All losses are positive: ", is_pos)

    print("")
    print("Compute the gromov loss - recompute_const=True")
    T = np.random.rand(b, n, n).astype("float32")
    loss = compute_loss(T, recompute_const=True)
    is_pos = (loss >= -1e-6).all()
    print("All losses are positive: ", is_pos)

    print("")
    print(
        "Conclusion: when T does not satisfy marginals, it is important to set recompute_const=True to ensure that the loss computation is correct."
    )


if __name__ == "__main__":
    print(
        "----------------- Setting symmetric Flag correctly in ot.solve_gromov  -----------------"
    )
    demo_symmetric()
    print("")
    print(
        "----------------- Testing recompute_const option in tensor_product  -----------------"
    )
    demo_recompute_const()
