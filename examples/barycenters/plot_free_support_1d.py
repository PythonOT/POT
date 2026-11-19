"""
Exact barycenter of empirical distributions on the line
======================================================

Three instruments report distributions of a peak position on different
grids. Averaging probabilities bin by bin would require choosing a common
grid. The quadratic Wasserstein barycenter instead averages their quantiles
and returns the locations and masses of the resulting peaks.
"""

import matplotlib.pyplot as plt
import numpy as np
from ot.lp import free_support_barycenter_1d

positions = [[1.0, 4.0], [2.0, 3.5, 5.0], [1.5, 4.5]]
masses = [[0.25, 0.75], [0.25, 0.25, 0.5], [0.5, 0.5]]
x, a = free_support_barycenter_1d(positions, masses)

fig, ax = plt.subplots()
for i, (xi, ai) in enumerate(zip(positions, masses)):
    ax.stairs(xi, np.r_[0, np.cumsum(ai)], baseline=None, label=f"Instrument {i + 1}")
ax.stairs(x, np.r_[0, np.cumsum(a)], baseline=None, color="k", label="Barycenter")
ax.set(xlabel="Quantile", ylabel="Peak position")
ax.legend()
plt.show()
