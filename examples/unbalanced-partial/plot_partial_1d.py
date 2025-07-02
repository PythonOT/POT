"""
=========================
Partial Wasserstein in 1D
=========================

This script demonstrates how to compute and visualize the Partial Wasserstein distance between two 1D discrete distributions using `ot.partial.partial_wasserstein_1d`.

We illustrate the intermediate transport plans for all `k = 1...n`, where `n = min(len(x_a), len(x_b))`.
"""

import numpy as np
import matplotlib.pyplot as plt
from ot.partial import partial_wasserstein_1d

# Simulate two 1D discrete distributions
np.random.seed(42)
n = 6
x_a = np.sort(np.random.uniform(0, 10, size=n))
x_b = np.sort(np.random.uniform(0, 10, size=n))

# Plot original distributions
plt.figure(figsize=(10, 2))
plt.eventplot([x_a, x_b], lineoffsets=[1, -1], colors=["C0", "C1"], linelengths=0.6)
plt.yticks([1, -1], ["x_a", "x_b"])
plt.title("Original 1D Discrete Distributions")
plt.grid(True)
plt.show()

# %%
indices_a, indices_b, marginal_costs = partial_wasserstein_1d(x_a, x_b)

# Compute cumulative cost
cumulative_costs = np.cumsum(marginal_costs)

# Visualize all partial transport plans
fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True)

for k, ax in enumerate(axes):
    ax.eventplot([x_a, x_b], lineoffsets=[1, -1], colors=["C0", "C1"], linelengths=0.6)
    ax.set_yticks([1, -1])
    ax.set_yticklabels(["x_a", "x_b"])
    ax.set_title(
        f"Partial Transport - k = {k+1}, Cumulative Cost = {cumulative_costs[k]:.2f}"
    )
    ax.grid(True)

    subset_a = np.sort(x_a[indices_a[: k + 1]])
    subset_b = np.sort(x_b[indices_b[: k + 1]])

    for x_a_i, x_b_j in zip(subset_a, subset_b):
        ax.plot([x_a_i, x_b_j], [1, -1], "k--", alpha=0.7)

plt.tight_layout()
plt.show()
