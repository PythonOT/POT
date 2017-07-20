# -*- coding: utf-8 -*-
"""
==============================
OT for empirical distributions
==============================

"""

import numpy as np
import matplotlib.pylab as pl
import ot


#%% parameters

n = 150  # nb bins

xs, ys = ot.datasets.get_data_classif('3gauss', n)
xt, yt = ot.datasets.get_data_classif('3gauss2', n)

a, b = ot.unif(n), ot.unif(n)
# loss matrix
M = ot.dist(xs, xt)
# M/=M.max()

#%% plot samples

pl.figure(1)
pl.subplot(2, 2, 1)
pl.scatter(xs[:, 0], xs[:, 1], c=ys, marker='+', label='Source samples')
pl.legend(loc=0)
pl.title('Source  distributions')

pl.subplot(2, 2, 2)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o', label='Target samples')
pl.legend(loc=0)
pl.title('target  distributions')

pl.figure(2)
pl.imshow(M, interpolation='nearest')
pl.title('Cost matrix M')


#%% OT estimation

# EMD
G0 = ot.emd(a, b, M)

# sinkhorn
lambd = 1e-1
Gs = ot.sinkhorn(a, b, M, lambd)


# Group lasso regularization
reg = 1e-1
eta = 1e0
Gg = ot.da.sinkhorn_lpl1_mm(a, ys.astype(np.int), b, M, reg, eta)


#%% visu matrices

pl.figure(3)

pl.subplot(2, 3, 1)
pl.imshow(G0, interpolation='nearest')
pl.title('OT matrix ')

pl.subplot(2, 3, 2)
pl.imshow(Gs, interpolation='nearest')
pl.title('OT matrix Sinkhorn')

pl.subplot(2, 3, 3)
pl.imshow(Gg, interpolation='nearest')
pl.title('OT matrix Group lasso')

pl.subplot(2, 3, 4)
ot.plot.plot2D_samples_mat(xs, xt, G0, c=[.5, .5, 1])
pl.scatter(xs[:, 0], xs[:, 1], c=ys, marker='+', label='Source samples')
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o', label='Target samples')


pl.subplot(2, 3, 5)
ot.plot.plot2D_samples_mat(xs, xt, Gs, c=[.5, .5, 1])
pl.scatter(xs[:, 0], xs[:, 1], c=ys, marker='+', label='Source samples')
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o', label='Target samples')

pl.subplot(2, 3, 6)
ot.plot.plot2D_samples_mat(xs, xt, Gg, c=[.5, .5, 1])
pl.scatter(xs[:, 0], xs[:, 1], c=ys, marker='+', label='Source samples')
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o', label='Target samples')
pl.tight_layout()

#%% sample interpolation

xst0 = n * G0.dot(xt)
xsts = n * Gs.dot(xt)
xstg = n * Gg.dot(xt)

pl.figure(4, figsize=(8, 3))
pl.subplot(1, 3, 1)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=0.5)
pl.scatter(xst0[:, 0], xst0[:, 1], c=ys,
           marker='+', label='Transp samples', s=30)
pl.title('Interp samples')
pl.legend(loc=0)

pl.subplot(1, 3, 2)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=0.5)
pl.scatter(xsts[:, 0], xsts[:, 1], c=ys,
           marker='+', label='Transp samples', s=30)
pl.title('Interp samples Sinkhorn')

pl.subplot(1, 3, 3)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=0.5)
pl.scatter(xstg[:, 0], xstg[:, 1], c=ys,
           marker='+', label='Transp samples', s=30)
pl.title('Interp samples Grouplasso')
pl.tight_layout()
pl.show()
