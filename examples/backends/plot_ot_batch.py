"""
=================================================
Solving Many Optimal Transport Problems in Parallel
=================================================

In some situations, one may want to solve many OT problems with the same
structure (same number of samples, same cost function, etc.) at the same time.

In that case using a for loop to solve the problems sequentially is inefficient.
This example shows how to use the batch solvers implemented in POT to solve
many problems in parallel on CPU or GPU (even more efficient on GPU).

"""

# Author: Paul Krzakala <paul.krzakala@gmail.com>
# License: MIT License

#############################################################################
#
# Computing the cost matrices
# ---------------------------------------------
# Let's create a batch of optimal transport problems with n samples in d dimensions.
# First we need to compute the cost matrices for each problem. We could do that with a for loop and ot.dist but instead we can directly use ot.batch.dist_batch.

import ot
import numpy as np

n_problems = 4  # nb problems/batch size
n_samples = 8  # nb samples
dim = 2  # nb dimensions

np.random.seed(0)
samples_source = np.random.randn(n_problems, n_samples, dim)
samples_target = samples_source + 0.1 * np.random.randn(n_problems, n_samples, dim)

# Naive approach
M_list = []
for i in range(n_problems):
    M_list.append(
        ot.dist(samples_source[i], samples_target[i])
    )  # List of cost matrices n_samples x n_samples
# Batched approach
M_batch = ot.batch.dist_batch(
    samples_source, samples_target
)  # Array of cost matrices n_problems x n_samples x n_samples

for i in range(n_problems):
    assert np.allclose(M_list[i], M_batch[i])

#############################################################################
#
# Solving the problems
# ---------------------------------------------
#
# Now we can either solve the problems sequentially with a for loop (ot.solve) or in parallel (ot.batch.solve_batch).

reg = 1.0
max_iter = 100
tol = 1e-3

# Naive approach
results_values_list = []
for i in range(n_problems):
    res = ot.solve(M_list[i], reg=reg, max_iter=max_iter, tol=tol, reg_type="entropy")
    results_values_list.append(res.value_linear)

# Batched approach
results_batch = ot.batch.solve_batch(
    M=M_batch, reg=reg, max_iter=max_iter, tol=tol, reg_type="entropy"
)
results_values_batch = results_batch.value_linear

assert np.allclose(np.array(results_values_list), results_values_batch, atol=tol * 10)

#############################################################################
#
# Compare the computation time
# ---------------------------------------------
#
# Lets compare the computation time of the two approaches on larger problems.
# Note that the results can be even more impressive on GPU.

from time import perf_counter

n_problems = 128
n_samples = 8
dim = 2
reg = 10.0
max_iter = 1000
tol = 1e-3

samples_source = np.random.randn(n_problems, n_samples, dim)
samples_target = samples_source + 0.1 * np.random.randn(n_problems, n_samples, dim)


def benchmark_naive(samples_source, samples_target):
    start = perf_counter()
    for i in range(n_problems):
        M = ot.dist(samples_source[i], samples_target[i])
        res = ot.solve(M, reg=reg, max_iter=max_iter, tol=tol, reg_type="entropy")
    end = perf_counter()
    return end - start


def benchmark_batch(samples_source, samples_target):
    start = perf_counter()
    M_batch = ot.batch.dist_batch(samples_source, samples_target)
    res_batch = ot.batch.solve_batch(
        M=M_batch, reg=reg, max_iter=max_iter, tol=tol, reg_type="entropy"
    )
    end = perf_counter()
    return end - start


time_naive = benchmark_naive(samples_source, samples_target)
time_batch = benchmark_batch(samples_source, samples_target)

print(f"Naive approach time: {time_naive:.4f} seconds")
print(f"Batched approach time: {time_batch:.4f} seconds")

#############################################################################
#
# Gromov-Wasserstein
# ---------------------------------------------
#
# ot.batch also implements a batched Gromov-Wasserstein solver.
#
# But this solver is NOT the same as calling ot.solve_gromov in a for loop.
#
# ot.solve_gromov uses the conditional gradient algorithm, each inner loop uses exact emd solver.
#
# ot.batch.solve_gromov_batch uses a proximal variant where each inner loop uses entropic regularization.
#
# Both methods have a different value/time trade-off. In this example, solve_gromov_batch is slower but gives a better value.
#
# If your data lives on a GPU, then solve_gromov_batch will always be much faster.

from ot import solve_gromov
from ot.batch import solve_gromov_batch


def benchmark_naive_gw(samples_source, samples_target):
    start = perf_counter()
    avg_value = 0
    for i in range(n_problems):
        C1 = ot.dist(samples_source[i], samples_source[i])
        C2 = ot.dist(samples_target[i], samples_target[i])
        res = solve_gromov(C1, C2, max_iter=1000, tol=tol)
        avg_value += res.value
    avg_value /= n_problems
    end = perf_counter()
    return end - start, avg_value


def benchmark_batch_gw(samples_source, samples_target):
    start = perf_counter()
    C1_batch = ot.batch.dist_batch(samples_source, samples_source)
    C2_batch = ot.batch.dist_batch(samples_target, samples_target)
    res_batch = solve_gromov_batch(
        C1_batch, C2_batch, reg=1, max_iter=100, max_iter_inner=50, tol=tol
    )
    avg_value = np.mean(res_batch.value)
    end = perf_counter()
    return end - start, avg_value


time_naive_gw, avg_value_naive_gw = benchmark_naive_gw(samples_source, samples_target)
time_batch_gw, avg_value_batch_gw = benchmark_batch_gw(samples_source, samples_target)

print(f"{'Method':<20}{'Time (s)':<15}{'Avg Value':<15}")
print(f"{'Naive GW':<20}{time_naive_gw:<15.4f}{avg_value_naive_gw:<15.4f}")
print(f"{'Batched GW':<20}{time_batch_gw:<15.4f}{avg_value_batch_gw:<15.4f}")
