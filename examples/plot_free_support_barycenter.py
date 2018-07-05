# -*- coding: utf-8 -*-
"""
====================================================
2D Wasserstein barycenters between empirical distributions
====================================================

Illustration of 2D Wasserstein barycenters between discributions that are weighted
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
N = 6
d = 2
measures_locations = []
measures_weights = []

for i in range(N):

    n = np.random.randint(low=1, high=20)  # nb samples

    mu = np.random.normal(0., 4., (d,))

    A = np.random.rand(d, d)
    cov = np.dot(A,A.transpose())

    xs = ot.datasets.make_2D_samples_gauss(n, mu, cov)
    b = np.random.uniform(0., 1., (n,))
    b = b/np.sum(b)

    measures_locations.append(xs)
    measures_weights.append(b)

k = 10
X_init = np.random.normal(0., 1., (k,d))
b_init = np.ones((k,)) / k


##############################################################################
# Compute free support barycenter
# -------------
X = ot.lp.cvx.free_support_barycenter(measures_locations, measures_weights, X_init, b_init)


##############################################################################
# Plot data
# ---------

#%% plot samples

pl.figure(1)
for (xs, b) in zip(measures_locations, measures_weights):
    color = np.random.randint(low=1, high=10*N)
    pl.scatter(xs[:, 0], xs[:, 1], s=b*1000, label='input measure')
pl.scatter(X[:, 0], X[:, 1], s=b_init*1000, c='black' , marker='^', label='2-Wasserstein barycenter')
pl.title('Data measures and their barycenter')
pl.legend(loc=0)
pl.show()