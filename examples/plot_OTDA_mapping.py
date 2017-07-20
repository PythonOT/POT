# -*- coding: utf-8 -*-
"""
===============================================
OT mapping estimation for domain adaptation [8]
===============================================

[8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
    "Mapping estimation for discrete optimal transport",
    Neural Information Processing Systems (NIPS), 2016.
"""

import numpy as np
import matplotlib.pylab as pl
import ot


#%% dataset generation

np.random.seed(0)  # makes example reproducible

n = 100  # nb samples in source and target datasets
theta = 2 * np.pi / 20
nz = 0.1
xs, ys = ot.datasets.get_data_classif('gaussrot', n, nz=nz)
xt, yt = ot.datasets.get_data_classif('gaussrot', n, theta=theta, nz=nz)

# one of the target mode changes its variance (no linear mapping)
xt[yt == 2] *= 3
xt = xt + 4


#%% plot samples

pl.figure(1, (6.4, 3))
pl.clf()
pl.scatter(xs[:, 0], xs[:, 1], c=ys, marker='+', label='Source samples')
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o', label='Target samples')
pl.legend(loc=0)
pl.title('Source and target distributions')


#%% OT linear mapping estimation

eta = 1e-8   # quadratic regularization for regression
mu = 1e0     # weight of the OT linear term
bias = True  # estimate a bias

ot_mapping = ot.da.OTDA_mapping_linear()
ot_mapping.fit(xs, xt, mu=mu, eta=eta, bias=bias, numItermax=20, verbose=True)

xst = ot_mapping.predict(xs)  # use the estimated mapping
xst0 = ot_mapping.interp()   # use barycentric mapping


pl.figure(2)
pl.clf()
pl.subplot(2, 2, 1)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=.3)
pl.scatter(xst0[:, 0], xst0[:, 1], c=ys,
           marker='+', label='barycentric mapping')
pl.title("barycentric mapping")

pl.subplot(2, 2, 2)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=.3)
pl.scatter(xst[:, 0], xst[:, 1], c=ys, marker='+', label='Learned mapping')
pl.title("Learned mapping")
pl.tight_layout()

#%% Kernel mapping estimation

eta = 1e-5   # quadratic regularization for regression
mu = 1e-1     # weight of the OT linear term
bias = True  # estimate a bias
sigma = 1    # sigma bandwidth fot gaussian kernel


ot_mapping_kernel = ot.da.OTDA_mapping_kernel()
ot_mapping_kernel.fit(
    xs, xt, mu=mu, eta=eta, sigma=sigma, bias=bias, numItermax=10, verbose=True)

xst_kernel = ot_mapping_kernel.predict(xs)  # use the estimated mapping
xst0_kernel = ot_mapping_kernel.interp()   # use barycentric mapping


#%% Plotting the mapped samples

pl.figure(2)
pl.clf()
pl.subplot(2, 2, 1)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=.2)
pl.scatter(xst0[:, 0], xst0[:, 1], c=ys, marker='+',
           label='Mapped source samples')
pl.title("Bary. mapping (linear)")
pl.legend(loc=0)

pl.subplot(2, 2, 2)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=.2)
pl.scatter(xst[:, 0], xst[:, 1], c=ys, marker='+', label='Learned mapping')
pl.title("Estim. mapping (linear)")

pl.subplot(2, 2, 3)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=.2)
pl.scatter(xst0_kernel[:, 0], xst0_kernel[:, 1], c=ys,
           marker='+', label='barycentric mapping')
pl.title("Bary. mapping (kernel)")

pl.subplot(2, 2, 4)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=.2)
pl.scatter(xst_kernel[:, 0], xst_kernel[:, 1], c=ys,
           marker='+', label='Learned mapping')
pl.title("Estim. mapping (kernel)")
pl.tight_layout()

pl.show()
