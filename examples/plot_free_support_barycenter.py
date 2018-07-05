# -*- coding: utf-8 -*-
"""
====================================================
2D Wasserstein barycenters of distributions
====================================================

Illustration of 2D Wasserstein barycenters if discributions that are weighted
sum of diracs.

"""

# Author: Vivien Seguy <vivien.seguy@iip.ist.i.kyoto-u.ac.jp>
#
# License: MIT License

import numpy as np
import matplotlib.pylab as pl
import ot.plot

##############################################################################
# Generate data
# -------------
#%% parameters and data generation
N = 3
d = 2
measures_locations = []
measures_weights = []

for i in range(N):

    n = np.random.randint(low=1, high=20)  # nb samples

    mu = np.random.normal(0., 4., (d,))

    A = np.random.rand(d, d)
    cov = np.dot(A, A.transpose())

    x_i = ot.datasets.make_2D_samples_gauss(n, mu, cov)
    b_i = np.random.uniform(0., 1., (n,))
    b_i = b_i / np.sum(b_i)

    measures_locations.append(x_i)
    measures_weights.append(b_i)


##############################################################################
# Compute free support barycenter
# -------------

k = 10
X_init = np.random.normal(0., 1., (k, d))
b = np.ones((k,)) / k

X = ot.lp.cvx.free_support_barycenter(measures_locations, measures_weights, X_init, b)


##############################################################################
# Plot data
# ---------

#%% plot samples

pl.figure(1)
for (x_i, b_i) in zip(measures_locations, measures_weights):
    color = np.random.randint(low=1, high=10 * N)
    pl.scatter(x_i[:, 0], x_i[:, 1], s=b * 1000, label='input measure')
pl.scatter(X[:, 0], X[:, 1], s=b * 1000, c='black', marker='^', label='2-Wasserstein barycenter')
pl.title('Data measures and their barycenter')
pl.legend(loc=0)
pl.show()
