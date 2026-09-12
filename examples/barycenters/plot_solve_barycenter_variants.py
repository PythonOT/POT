# -*- coding: utf-8 -*-
"""
======================================
Optimal Transport Barycenter solvers comparison
======================================

This example illustrates solutions returned for different variants of exact,
regularized and unbalanced OT barycenter problems with free support using our wrapper `ot.solve_bary_sample`.
"""

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License
# sphinx_gallery_thumbnail_number = 2

# %%

import numpy as np
import matplotlib.pylab as pl
import ot
from ot.plot import plot2D_samples_mat

# %%
# 2D data example
# ---------------
#
# We first generate two sets of samples in 2D of 8 and 16
# points uniformly separated on circles. The weights of the samples are uniform.

# Problem size
n1, n2 = 8, 16
nbary = 12

# Generate random data
np.random.seed(0)

r1, r2 = 1, 3
x1 = r1 * np.array(
    [(np.cos(2 * i * np.pi / n1), np.sin(2 * i * np.pi / n1)) for i in range(n1)]
)

x2 = r2 * np.array(
    [(np.cos(2 * i * np.pi / n2), np.sin(2 * i * np.pi / n2)) for i in range(n2)]
)

style = {"markeredgecolor": "k"}

pl.figure(1, (4, 4))
pl.plot(x1[:, 0], x1[:, 1], "ob", **style)
pl.plot(x2[:, 0], x2[:, 1], "or", **style)
pl.title("Source distributions")
pl.show()


# %%
# Set up parameters for balanced OT barycenter solvers and solve
# ---------------------------------------

# balanced OT
lst_balanced_solvers = [  # name, param for ot.solve function
    ("Exact OT", dict()),
    ("Entropic Reg. OT", dict(reg=1.0)),
]

lst_balanced_res = []
for name, param in lst_balanced_solvers:
    print(f"-- name = {name} / param = {param}")
    res = ot.solve_bary_sample(X_a_list=[x1, x2], n=nbary, **param)
    lst_balanced_res.append(res)
    list_P = [res.list_res[k].plan for k in range(2)]
    print("X:", res.X)
    print("loss:", res.value)
    print("loss:", res.log)
    print(
        "marginals OT 1:",
        res.list_res[0].plan.sum(axis=1),
        res.list_res[0].plan.sum(axis=0),
    )
    print(
        "marginals OT 2:",
        res.list_res[1].plan.sum(axis=1),
        res.list_res[1].plan.sum(axis=0),
    )

##############################################################################
# Plot distributions and plans for balanced OT barycenter solvers
# ----------


def plot_list_res(
    lst_res,
    lst_solvers,
    fig_num=1,
    n_cols=2,
    show_masses=False,
    show_legend=True,
    s=100,
    fig_width=None,
    fig_height=None,
):
    n_plots = len(lst_res)
    n_rows = int(np.ceil(n_plots / n_cols))

    if fig_width is None:
        fig_width = 8 * n_cols
    if fig_height is None:
        fig_height = 4 * n_rows

    fig, axes = pl.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        num=fig_num,
    )
    axes = axes.ravel()

    legend_handles = None
    for i, (name, param) in enumerate(lst_solvers):
        ax = axes[i]
        pl.sca(ax)

        X = lst_res[i].X
        list_P = [lst_res[i].list_res[k].plan for k in range(2)]
        loss = lst_res[i].value

        plot2D_samples_mat(x1, X, list_P[0])
        plot2D_samples_mat(x2, X, list_P[1])

        if show_masses:
            # Marginals induced by transport plans

            a1 = list_P[0].sum(axis=1) * list_P[0].shape[0]
            a2 = list_P[1].sum(axis=1) * list_P[1].shape[0]

            # weighted average barycenter masses
            b = (
                0.5
                * (list_P[0].sum(axis=0) + list_P[1].sum(axis=0))
                * list_P[0].shape[1]
            )

            # background uniform distribution
            ax.scatter(x1[:, 0], x1[:, 1], s=s, color="blue", marker="o", alpha=0.25)
            ax.scatter(x2[:, 0], x2[:, 1], s=s, color="red", marker="o", alpha=0.25)
            ax.scatter(X[:, 0], X[:, 1], s=s, color="green", marker="o", alpha=0.25)

            list_size_1 = s * a1
            list_size_2 = s * a2
            list_size_b = s * b
        else:
            list_size_1 = s
            list_size_2 = s
            list_size_b = s

        if i == 0:  # add labels
            h1 = ax.scatter(
                x1[:, 0],
                x1[:, 1],
                s=list_size_1,
                color="blue",
                marker="o",
                alpha=1,
                label="Source distribution 1",
            )
            h2 = ax.scatter(
                x2[:, 0],
                x2[:, 1],
                s=list_size_2,
                color="red",
                marker="o",
                alpha=1,
                label="Source distribution 2",
            )
            h3 = ax.scatter(
                X[:, 0],
                X[:, 1],
                s=list_size_b,
                color="green",
                marker="o",
                alpha=1,
                label="Barycenter distribution",
            )

        else:
            h1 = ax.scatter(
                x1[:, 0], x1[:, 1], s=list_size_1, color="blue", marker="o", alpha=1
            )
            h2 = ax.scatter(
                x2[:, 0], x2[:, 1], s=list_size_2, color="red", marker="o", alpha=1
            )
            h3 = ax.scatter(
                X[:, 0], X[:, 1], s=list_size_b, color="green", marker="o", alpha=1
            )

        if legend_handles is None:
            legend_handles = [h1, h2, h3]

        ax.set_title(name)

    ############################################################
    # remove unused axes

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    ############################################################
    # Single legend above all subplots

    if show_legend:
        labels = [h.get_label() for h in legend_handles]

        fig.legend(
            legend_handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
        )
    fig.tight_layout()
    pl.show()


plot_list_res(
    lst_balanced_res,
    lst_balanced_solvers,
    fig_num=2,
    n_cols=2,
    show_masses=False,
    fig_width=8,
    fig_height=4,
    show_legend=True,
)


# %%
# Set up parameters for unbalanced OT barycenter solvers and solve
# ---------------------------------------

lambda_unbalanced_vals = [1, 2.5, 10]

# unbalanced OT KL
lst_unbalanced_solvers = [
    (
        "Unbalanced KL No Reg \n" + r"$\lambda_u$=%s" % lambda_val,
        dict(unbalanced=lambda_val),
    )
    for lambda_val in lambda_unbalanced_vals
] + [
    (
        "Unbalanced KL with KL Reg \n"
        + r"$\lambda_u$=%s, $\lambda_{ent}$=%s" % (lambda_val, 0.1),
        dict(reg=0.1, unbalanced=lambda_val, unbalanced_type="kl", reg_type="kl"),
    )
    for lambda_val in lambda_unbalanced_vals
]

lst_unbalanced_res = []
for name, param in lst_unbalanced_solvers:
    print(f"-- name = {name} / param = {param}")
    res = ot.solve_bary_sample(X_a_list=[x1, x2], n=nbary, **param)
    lst_unbalanced_res.append(res)
    list_P = [res.list_res[k].plan for k in range(2)]
    print("X:", res.X)
    print("loss:", res.value)
    print("loss:", res.log)
    print(
        "marginals OT 1:",
        res.list_res[0].plan.sum(axis=1),
        res.list_res[0].plan.sum(axis=0),
    )
    print(
        "marginals OT 2:",
        res.list_res[1].plan.sum(axis=1),
        res.list_res[1].plan.sum(axis=0),
    )

##############################################################################
# Plot distributions and plans for unbalanced OT barycenter solvers
# ----------

plot_list_res(
    lst_unbalanced_res,
    lst_unbalanced_solvers,
    fig_num=3,
    n_cols=3,
    show_masses=True,
    fig_width=12,
    fig_height=8.5,
    show_legend=False,
)
