# -*- coding: utf-8 -*-
"""
==================
Batch parallel OT
==================

Shows how to use the batch parallel OT solvers for optimal transport.

"""

# Author: Paul Krzakala <paul.krzakala@gmail.com>
# License: MIT License

from ot.batch._linear import *
from ot.batch._quadratic import *
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import torch


def demo_solver_linear():
    r"""Demo for the linear solver batch OT."""

    b = 128
    n = 20
    d = 4

    # Generate random data
    X = np.random.randn(b, n, d).astype("float32")
    Y = X[:, np.random.permutation(n), :]

    plt.figure(figsize=(10, 6))

    for epsilon in [0.1, 0.2, 0.5]:
        noise_levels = np.linspace(0.0, 0.1, 10)
        avg_cost_list = []
        for noise in noise_levels:
            Y += noise * np.random.randn(b, n, d).astype("float32")
            M = cost_matrix_l2_batch(X, Y)
            res = linear_solver_batch(
                epsilon=epsilon, M=M, max_iter=1000, tol=1e-5, log_dual=False
            )
            cost = res.value_linear.mean()
            avg_cost_list.append(cost)
        plt.plot(
            noise_levels, avg_cost_list, marker="o", label=r"epsilon={}".format(epsilon)
        )

    plt.xlabel("Noise Level")
    plt.ylabel("Average Cost")
    plt.legend()
    plt.show()


def demo_solver_quad():
    b = 512
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
    alpha_list = np.linspace(0.0, 1, 20)

    for alpha in alpha_list:
        res = quadratic_solver_batch(
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

        cost_linear = res.value_linear.mean()
        cost_quadratic = res.value_quad.mean()
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


def demo_gpu():
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
        res = quadratic_solver_batch(
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
    res = quadratic_solver_batch(
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
            quadratic_solver_batch(
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
        out = quadratic_solver_batch(
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


def demo_symmetric():
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

    start = perf_counter()
    res = quadratic_solver_batch(
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
    print(f"Iterations symmetric=True: {n_iter}")
    print(f"Time symmetric=True: {end - start:.4f} seconds")
    print(f"Value symmetric=True: {res.value.mean().item():.4f}")

    print("")

    start = perf_counter()
    res = quadratic_solver_batch(
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
    print(f"Iterations symmetric=False: {n_iter}")
    print(f"Time symmetric=False: {end - start:.4f} seconds")
    print(f"Value symmetric=False: {res.value.mean().item():.4f}")


if __name__ == "__main__":
    # demo_solver_linear()
    # demo_solver_quad()
    # demo_gpu()
    demo_symmetric()
