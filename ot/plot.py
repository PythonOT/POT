"""
Functions for plotting OT matrices
"""


import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec


def plot1D_mat(a, b, M, title=''):
    """ Plot matrix M  with the source and target 1D distribution

    Creates a subplot with the source distribution a on the left and
    target distribution b on the tot. The matrix M is shown in between.


    Parameters
    ----------
    a : np.array, shape (na,)
        Source distribution
    b : np.array, shape (nb,)
        Target distribution
    M : np.array, shape (na,nb)
        Matrix to plot
    """
    na, nb = M.shape

    gs = gridspec.GridSpec(3, 3)

    xa = np.arange(na)
    xb = np.arange(nb)

    ax1 = plt.subplot(gs[0, 1:])
    plt.plot(xb, b, 'r', label='Target distribution')
    plt.yticks(())
    plt.title(title)

    ax2 = plt.subplot(gs[1:, 0])
    plt.plot(a, xa, 'b', label='Source distribution')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xticks(())

    plt.subplot(gs[1:, 1:], sharex=ax1, sharey=ax2)
    plt.imshow(M, interpolation='nearest')
    plt.axis('off')

    plt.xlim((0, nb))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0., hspace=0.2)


def plot2D_samples_mat(xs, xt, G, thr=1e-8, **kwargs):
    """ Plot matrix M  in 2D with  lines using alpha values

    Plot lines between source and target 2D samples with a color
    proportional to the value of the matrix G between samples.


    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    b : ndarray, shape (nt,2)
        Target samples positions
    G : ndarray, shape (na,nb)
        OT matrix
    thr : float, optional
        threshold above which the line is drawn
    **kwargs : dict
        paameters given to the plot functions (default color is black if
        nothing given)
    """
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                plt.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                         alpha=G[i, j] / mx, **kwargs)
