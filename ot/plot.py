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


def plot1D_mat(a, b, M, title='', a_label='Source distribution',
               b_label='Target distribution', color_source='#7ED321',
               color_target='#4A90E2', coupling_cmap='gray'):
    r""" Plot matrix :math:`\mathbf{M}` with the source and target 1D distributions.

    Creates a subplot with the source distribution :math:`\mathbf{a}` on the
    bottom and target distribution :math:`\mathbf{b}` on the left.
    The matrix :math:`\mathbf{M}` is shown in between.


    Parameters
    ----------
    a : ndarray, shape (na,)
        Source distribution
    b : ndarray, shape (nb,)
        Target distribution
    M : ndarray, shape (na, nb)
        Matrix to plot
    a_label: str, optional
        Label for source distribution
    b_label: str, optional
        Label for target distribution
    title: str, optional
        Title of the plot
    color_source: str, optional
        Color of the source distribution
    color_target: str, optional
        Color of the target distribution
    coupling_cmap: str, optional
        Colormap for the coupling matrix

    Returns
    -------
    ax1: source plot ax
    ax2: target plot ax
    ax3: coupling plot ax

    .. seealso::
        :func:`rescale_for_imshow_plot`
    """
    na, nb = M.shape

    fig = pl.figure(figsize=(8, 8))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1],
                          width_ratios=[1, 1, 1],
                          hspace=0, wspace=0)

    xa = np.arange(na)
    xb = np.arange(nb)

    # horizontal source on the bottom, flipped vertically
    ax1 = fig.add_subplot(gs[2, 1:])
    ax1.plot(xa, np.max(a) - a, color=color_source, label=a_label, linewidth=2)
    ax1.fill_between(xa, np.max(a) - a, np.max(a) * np.ones_like(a),
                     color=color_source, alpha=.5)
    ax1.set_xticks(())
    ax1.set_yticks(())
    ax1.set_title(a_label, y=-.15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # vertical target on the left
    ax2 = fig.add_subplot(gs[0:2, 0])
    ax2.plot(b, xb, color=color_target, label=b_label, linewidth=2)
    ax2.fill_between(b, xb, color=color_target, alpha=.5)
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    ax2.set_xticks(())
    ax2.set_yticks(())
    ax2.set_title(b_label)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # plan image, transposed since imshow is in "yx" coords
    ax3 = fig.add_subplot(gs[0:2, 1:], sharey=ax2, sharex=ax1)
    ax3.imshow(1 - M.T, interpolation='nearest', origin='lower',
               cmap=coupling_cmap)
    ax3.set_aspect('equal')
    ax3.set_title(title)

    # Set spines visibility to True and customize if desired
    ax3.spines['top'].set_visible(True)
    ax3.spines['right'].set_visible(True)
    ax3.spines['bottom'].set_visible(True)
    ax3.spines['left'].set_visible(True)

    pl.subplots_adjust(hspace=0, wspace=0)
    return ax1, ax2, ax3


def rescale_for_imshow_plot(x, y, n, a_y=None, b_y=None):
    r"""
    Gives arrays xr, yr that can be plotted over an (n, n)
    imshow plot (in 'xy' coordinates). If `a_y` or `b_y` is provided,
    y is sliced over its indices such that y stays in [ay, by].

    Parameters
    ----------
    x : ndarray, shape (nx,)
    y : ndarray, shape (ny,)
    n : int
        Size of the imshow plot on which to plot (x, y)
    a_y : float, optional
        Lower bound for y
    b_y : float, optional
        Upper bound for y

    Returns
    -------
    xr : ndarray, shape (nx,)
        Rescaled x values
    yr : ndarray, shape (ny,)
        Rescaled y values (due to slicing, may have less elements than y)

    .. seealso::
        :func:`plot1D_mat`

    """
    # slice over the y values that are in the y range
    a_x, b_x = np.min(x), np.max(x)
    if a_y is None:
        a_y = np.min(y)
    if b_y is None:
        b_y = np.max(y)
    idx = (y >= a_y) & (y <= b_y)
    x_rescaled = (x[idx] - a_x) * (n - 1) / (b_x - a_x)
    y_rescaled = (y[idx] - a_y) * (n - 1) / (b_y - a_y)
    return x_rescaled, y_rescaled


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
