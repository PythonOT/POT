# -*- coding: utf-8 -*-
"""
==========================================
2D Optimal transport for different metrics
==========================================

Stole the figure idea from Fig. 1 and 2 in
https://arxiv.org/pdf/1706.07650.pdf


@author: rflamary
"""

import numpy as np
import matplotlib.pylab as plt
import ot

#%% parameters and data generation

for data in range(2):

    if data:
        n = 20  # nb samples
        xs = np.zeros((n, 2))
        xs[:, 0] = np.arange(n) + 1
        xs[:, 1] = (np.arange(n) + 1) * -0.001  # to make it strictly convex...

        xt = np.zeros((n, 2))
        xt[:, 1] = np.arange(n) + 1
    else:

        n = 50  # nb samples
        xtot = np.zeros((n + 1, 2))
        xtot[:, 0] = np.cos(
            (np.arange(n + 1) + 1.0) * 0.9 / (n + 2) * 2 * np.pi)
        xtot[:, 1] = np.sin(
            (np.arange(n + 1) + 1.0) * 0.9 / (n + 2) * 2 * np.pi)

        xs = xtot[:n, :]
        xt = xtot[1:, :]

    a, b = ot.unif(n), ot.unif(n)  # uniform distribution on samples

    # loss matrix
    M1 = ot.dist(xs, xt, metric='euclidean')
    M1 /= M1.max()

    # loss matrix
    M2 = ot.dist(xs, xt, metric='sqeuclidean')
    M2 /= M2.max()

    # loss matrix
    Mp = np.sqrt(ot.dist(xs, xt, metric='euclidean'))
    Mp /= Mp.max()

    #%% plot samples

    plt.figure(1 + 3 * data, figsize=(7, 3))
    plt.clf()
    plt.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    plt.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    plt.axis('equal')
    plt.title('Source and traget distributions')

    plt.figure(2 + 3 * data, figsize=(7, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(M1, interpolation='nearest')
    plt.title('Euclidean cost')

    plt.subplot(1, 3, 2)
    plt.imshow(M2, interpolation='nearest')
    plt.title('Squared Euclidean cost')

    plt.subplot(1, 3, 3)
    plt.imshow(Mp, interpolation='nearest')
    plt.title('Sqrt Euclidean cost')
    plt.tight_layout()

    #%% EMD
    G1 = ot.emd(a, b, M1)
    G2 = ot.emd(a, b, M2)
    Gp = ot.emd(a, b, Mp)

    plt.figure(3 + 3 * data, figsize=(7, 3))

    plt.subplot(1, 3, 1)
    ot.plot.plot2D_samples_mat(xs, xt, G1, c=[.5, .5, 1])
    plt.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    plt.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    plt.axis('equal')
    # plt.legend(loc=0)
    plt.title('OT Euclidean')

    plt.subplot(1, 3, 2)
    ot.plot.plot2D_samples_mat(xs, xt, G2, c=[.5, .5, 1])
    plt.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    plt.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    plt.axis('equal')
    # plt.legend(loc=0)
    plt.title('OT squared Euclidean')

    plt.subplot(1, 3, 3)
    ot.plot.plot2D_samples_mat(xs, xt, Gp, c=[.5, .5, 1])
    plt.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    plt.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    plt.axis('equal')
    # plt.legend(loc=0)
    plt.title('OT sqrt Euclidean')
    plt.tight_layout()

plt.show()
