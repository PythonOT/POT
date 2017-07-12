# -*- coding: utf-8 -*-
"""
========================
OT for domain adaptation
========================

"""

import matplotlib.pylab as pl
import ot


#%% parameters

n = 150  # nb samples in source and target datasets

xs, ys = ot.datasets.get_data_classif('3gauss', n)
xt, yt = ot.datasets.get_data_classif('3gauss2', n)


#%% plot samples

pl.figure(1, figsize=(6.4, 3))

pl.subplot(1, 2, 1)
pl.scatter(xs[:, 0], xs[:, 1], c=ys, marker='+', label='Source samples')
pl.legend(loc=0)
pl.title('Source  distributions')

pl.subplot(1, 2, 2)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o', label='Target samples')
pl.legend(loc=0)
pl.title('target  distributions')


#%% OT estimation

# LP problem
da_emd = ot.da.OTDA()     # init class
da_emd.fit(xs, xt)       # fit distributions
xst0 = da_emd.interp()    # interpolation of source samples

# sinkhorn regularization
lambd = 1e-1
da_entrop = ot.da.OTDA_sinkhorn()
da_entrop.fit(xs, xt, reg=lambd)
xsts = da_entrop.interp()

# non-convex Group lasso regularization
reg = 1e-1
eta = 1e0
da_lpl1 = ot.da.OTDA_lpl1()
da_lpl1.fit(xs, ys, xt, reg=reg, eta=eta)
xstg = da_lpl1.interp()

# True Group lasso regularization
reg = 1e-1
eta = 2e0
da_l1l2 = ot.da.OTDA_l1l2()
da_l1l2.fit(xs, ys, xt, reg=reg, eta=eta, numItermax=20, verbose=True)
xstgl = da_l1l2.interp()

#%% plot interpolated source samples

param_img = {'interpolation': 'nearest', 'cmap': 'spectral'}

pl.figure(2, figsize=(8, 4.5))
pl.subplot(2, 4, 1)
pl.imshow(da_emd.G, **param_img)
pl.title('OT matrix')

pl.subplot(2, 4, 2)
pl.imshow(da_entrop.G, **param_img)
pl.title('OT matrix\nsinkhorn')

pl.subplot(2, 4, 3)
pl.imshow(da_lpl1.G, **param_img)
pl.title('OT matrix\nnon-convex Group Lasso')

pl.subplot(2, 4, 4)
pl.imshow(da_l1l2.G, **param_img)
pl.title('OT matrix\nGroup Lasso')

pl.subplot(2, 4, 5)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=0.3)
pl.scatter(xst0[:, 0], xst0[:, 1], c=ys,
           marker='+', label='Transp samples', s=30)
pl.title('Interp samples')
pl.legend(loc=0)

pl.subplot(2, 4, 6)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=0.3)
pl.scatter(xsts[:, 0], xsts[:, 1], c=ys,
           marker='+', label='Transp samples', s=30)
pl.title('Interp samples\nSinkhorn')

pl.subplot(2, 4, 7)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=0.3)
pl.scatter(xstg[:, 0], xstg[:, 1], c=ys,
           marker='+', label='Transp samples', s=30)
pl.title('Interp samples\nnon-convex Group Lasso')

pl.subplot(2, 4, 8)
pl.scatter(xt[:, 0], xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=0.3)
pl.scatter(xstgl[:, 0], xstgl[:, 1], c=ys,
           marker='+', label='Transp samples', s=30)
pl.title('Interp samples\nGroup Lasso')
pl.tight_layout()
pl.show()
