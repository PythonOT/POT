"""
Compute a barycenter on a coarser reporting grid
================================================

Fine-grained measurements may need to be summarized on a common set of
reporting bins. The columns of the cost matrix specify which positions the
barycenter may occupy. Each input histogram retains its original bins.
"""

import matplotlib.pyplot as plt
import numpy as np
import ot

source = np.arange(40)
target = np.arange(2, 40, 5)
a = np.exp(-((source - 9) / 4) ** 2)
b = np.exp(-((source - 26) / 6) ** 2)
histograms = np.column_stack((a / a.sum(), b / b.sum()))
cost = (source[:, None] - target) ** 2
mass = ot.lp.barycenter(histograms, cost)

plt.plot(source, histograms, alpha=0.7)
plt.bar(target, mass / 5, width=5, alpha=0.4, label="Barycenter")
plt.xlabel("Position")
plt.ylabel("Mass per unit width")
plt.legend()
plt.show()
