from ot.batch._linear import *
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def demo_solver_linear():
    b = 128
    n = 20
    d = 4

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


if __name__ == "__main__":
    demo_solver_linear()
