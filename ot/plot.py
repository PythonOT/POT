"""
Functions for plotting OT matrices

.. warning::
    Note that by default the module is not import in :mod:`ot`. In order to
    use it you need to explicitly import :mod:`ot.plot`


"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import matplotlib.pylab as pl
from matplotlib import gridspec


def plot1D_mat(a, b, M, title=''):
    r""" Plot matrix :math:`\mathbf{M}`  with the source and target 1D distribution

    Creates a subplot with the source distribution :math:`\mathbf{a}` on the left and
    target distribution :math:`\mathbf{b}` on the top. The matrix :math:`\mathbf{M}` is shown in between.


    Parameters
    ----------
    a : ndarray, shape (na,)
        Source distribution
    b : ndarray, shape (nb,)
        Target distribution
    M : ndarray, shape (na, nb)
        Matrix to plot
    """
    na, nb = M.shape

    gs = gridspec.GridSpec(3, 3)

    xa = np.arange(na)
    xb = np.arange(nb)

    ax1 = pl.subplot(gs[0, 1:])
    pl.plot(xb, b, 'r', label='Target distribution')
    pl.yticks(())
    pl.title(title)

    ax2 = pl.subplot(gs[1:, 0])
    pl.plot(a, xa, 'b', label='Source distribution')
    pl.gca().invert_xaxis()
    pl.gca().invert_yaxis()
    pl.xticks(())

    pl.subplot(gs[1:, 1:], sharex=ax1, sharey=ax2)
    pl.imshow(M, interpolation='nearest')
    pl.axis('off')

    pl.xlim((0, nb))
    pl.tight_layout()
    pl.subplots_adjust(wspace=0., hspace=0.2)


def plot2D_samples_mat(xs, xt, G, thr=1e-8, **kwargs):
    r""" Plot matrix :math:`\mathbf{G}` in 2D with lines using alpha values

    Plot lines between source and target 2D samples with a color
    proportional to the value of the matrix :math:`\mathbf{G}` between samples.


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
        parameters given to the plot functions (default color is black if
        nothing given)
    """

    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()
    if 'alpha' in kwargs:
        scale = kwargs['alpha']
        del kwargs['alpha']
    else:
        scale = 1
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                pl.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                        alpha=G[i, j] / mx * scale, **kwargs)
