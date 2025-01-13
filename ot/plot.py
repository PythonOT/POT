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


def plot1D_mat(
    a,
    b,
    M,
    title="",
    plot_style="yx",
    a_label="",
    b_label="",
    color_source="b",
    color_target="r",
    coupling_cmap="gray_r",
):
    r"""Plot matrix :math:`\mathbf{M}` with the source and target 1D distributions.

    Creates a subplot with the source distribution :math:`\mathbf{a}` and target
    distribution :math:`\mathbf{b}`.
    In 'yx' mode (default), the source is on the left and
    the target on the top, and in 'xy' mode, source on the bottom (upside
    down) and the target on the left.
    The matrix :math:`\mathbf{M}` is shown in between.

    Parameters
    ----------
    a : ndarray, shape (na,)
        Source distribution
    b : ndarray, shape (nb,)
        Target distribution
    M : ndarray, shape (na, nb)
        Matrix to plot
    title : str, optional
        Title of the plot
    plot_style : str, optional
        Style of the plot, 'yx' or 'xy'. 'yx' places the source on the left and
        the target on the top, 'xy' places the source on the bottom (upside
        down) and the target on the left.
    a_label : str, optional
        Label for source distribution
    b_label : str, optional
        Label for target distribution
    color_source : str, optional
        Color of the source distribution
    color_target : str, optional
        Color of the target distribution
    coupling_cmap : str, optional
        Colormap for the coupling matrix

    Returns
    -------
    ax1 : source plot ax
    ax2 : target plot ax
    ax3 : coupling plot ax

    See Also
    --------
    :func:`rescale_for_imshow_plot`
    """
    assert plot_style in ["yx", "xy"], "plot_style should be 'yx' or 'xy'"
    na, nb = M.shape

    gs = gridspec.GridSpec(
        3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1], hspace=0, wspace=0
    )

    xa = np.arange(na)
    xb = np.arange(nb)

    # helper function for code factorisation
    def _set_ticks_and_spines(ax, empty_ticks=True, visible_spines=False):
        if empty_ticks:
            ax.set_xticks(())
            ax.set_yticks(())

        ax.spines["top"].set_visible(visible_spines)
        ax.spines["right"].set_visible(visible_spines)
        ax.spines["bottom"].set_visible(visible_spines)
        ax.spines["left"].set_visible(visible_spines)

    if plot_style == "xy":
        # horizontal source on the bottom, flipped vertically
        ax1 = pl.subplot(gs[2, 1:])
        ax1.plot(xa, np.max(a) - a, color=color_source, linewidth=2)
        ax1.fill(
            xa,
            np.max(a) - a,
            np.max(a) * np.ones_like(a),
            color=color_source,
            alpha=0.5,
        )
        ax1.set_title(a_label, y=-0.15)

        # vertical target on the left
        ax2 = pl.subplot(gs[0:2, 0])
        ax2.plot(b, xb, color=color_target, linewidth=2)
        ax2.fill(b, xb, color=color_target, alpha=0.5)
        ax2.invert_xaxis()
        ax2.invert_yaxis()
        ax2.set_title(b_label)

        _set_ticks_and_spines(ax1, empty_ticks=True, visible_spines=False)
        _set_ticks_and_spines(ax2, empty_ticks=True, visible_spines=False)

        # coupling matrix in the middle
        ax3 = pl.subplot(gs[0:2, 1:], sharey=ax2, sharex=ax1)
        ax3.imshow(M.T, interpolation="nearest", origin="lower", cmap=coupling_cmap)
        ax3.set_title(title)
        _set_ticks_and_spines(ax3, empty_ticks=False, visible_spines=True)

        pl.subplots_adjust(hspace=0, wspace=0)
        return ax1, ax2, ax3

    else:  # plot_style == 'yx'
        # vertical source on the left
        ax1 = pl.subplot(gs[1:, 0])
        ax1.plot(a, xa, color=color_source, linewidth=2)
        ax1.fill(a, xa, color=color_source, alpha=0.5)
        ax1.invert_xaxis()
        ax1.set_title(a_label)

        # horizontal target on the top
        ax2 = pl.subplot(gs[0, 1:])
        ax2.plot(xb, b, color=color_target, linewidth=2)
        ax2.fill(xb, b, color=color_target, alpha=0.5)
        ax2.set_title(b_label)

        _set_ticks_and_spines(ax1, empty_ticks=True, visible_spines=False)
        _set_ticks_and_spines(ax2, empty_ticks=True, visible_spines=False)

        # coupling matrix in the middle
        ax3 = pl.subplot(gs[1:, 1:], sharey=ax1, sharex=ax2)
        ax3.imshow(M, interpolation="nearest", cmap=coupling_cmap)
        # Set title below matrix plot
        ax3.text(
            0.5,
            -0.025,
            title,
            ha="center",
            va="top",
            transform=ax3.transAxes,
            fontsize="large",
        )
        _set_ticks_and_spines(ax3, empty_ticks=False, visible_spines=True)

        pl.subplots_adjust(hspace=0, wspace=0)
        return ax1, ax2, ax3


def rescale_for_imshow_plot(x, y, n, m=None, a_y=None, b_y=None):
    r"""
    Gives arrays xr, yr that can be plotted over an (n, m)
    imshow plot (in 'xy' coordinates). If `a_y` or `b_y` is provided,
    y is sliced over its indices such that y stays in [ay, by].

    Parameters
    ----------
    x : ndarray, shape (nx,)
    y : ndarray, shape (nx,)
    n : int
        x-axis size of the imshow plot on which to plot (x, y)
    m : int, optional
        y-axis size of the imshow plot, defaults to n
    a_y : float, optional
        Lower bound for y
    b_y : float, optional
        Upper bound for y

    Returns
    -------
    xr : ndarray, shape (nx,)
        Rescaled x values (due to slicing, may have less elements than x)
    yr : ndarray, shape (nx,)
        Rescaled y values (due to slicing, may have less elements than y)

    See Also
    --------
    :func:`plot1D_mat`

    """
    # slice over the y values that are in the y range
    a_x, b_x = np.min(x), np.max(x)
    assert x.shape[0] == y.shape[0], "x and y arrays should have the same size"
    if a_y is None:
        a_y = np.min(y)
    if b_y is None:
        b_y = np.max(y)
    if m is None:
        m = n
    idx = (y >= a_y) & (y <= b_y)
    x_rescaled = (x[idx] - a_x) * (n - 1) / (b_x - a_x)
    y_rescaled = (y[idx] - a_y) * (m - 1) / (b_y - a_y)
    return x_rescaled, y_rescaled


def plot2D_samples_mat(xs, xt, G, thr=1e-8, **kwargs):
    r"""Plot matrix :math:`\mathbf{G}` in 2D with lines using alpha values

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

    if ("color" not in kwargs) and ("c" not in kwargs):
        kwargs["color"] = "k"
    mx = G.max()
    if "alpha" in kwargs:
        scale = kwargs["alpha"]
        del kwargs["alpha"]
    else:
        scale = 1
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                pl.plot(
                    [xs[i, 0], xt[j, 0]],
                    [xs[i, 1], xt[j, 1]],
                    alpha=G[i, j] / mx * scale,
                    **kwargs,
                )
