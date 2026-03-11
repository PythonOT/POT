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
# We first generate two sets of samples in 2D that 25 and 50
# samples respectively located on circles. The weights of the samples are
# uniform.

# Problem size
n1 = 25
n2 = 50

# Generate random data
np.random.seed(0)

x1 = np.random.randn(n1, 2)
x1 /= np.sqrt(np.sum(x1**2, 1, keepdims=True)) / 2

x2 = np.random.randn(n2, 2)
x2 /= np.sqrt(np.sum(x2**2, 1, keepdims=True)) / 4

style = {"markeredgecolor": "k"}

pl.figure(1, (4, 4))
pl.plot(x1[:, 0], x1[:, 1], "ob", **style)
pl.plot(x2[:, 0], x2[:, 1], "or", **style)
pl.title("Source distributions")
pl.show()

# sphinx_gallery_end_ignore


# %%
# Set up parameters for barycenter solvers and solve
# ---------------------------------------

lst_regs = [
    "No Reg.",
    "Entropic",
]  # support e.g ["No Reg.", "Entropic", "L2", "Group Lasso + L2"]
lst_unbalanced = [
    "Balanced",
    "Unbalanced KL",
]  # ["Balanced", "Unb. KL", "Unb. L2", "Unb L1 (partial)"]

lst_solvers = [  # name, param for ot.solve function
    # balanced OT
    ("Exact OT", dict()),
    ("Entropic Reg. OT", dict(reg=0.1)),
    # unbalanced OT KL
    ("Unbalanced KL No Reg.", dict(unbalanced=0.05)),
    (
        "Unbalanced KL with KL Reg.",
        dict(reg=0.1, unbalanced=0.05, unbalanced_type="kl", reg_type="kl"),
    ),
]

lst_res = []
for name, param in lst_solvers:
    print(f"-- name = {name} / param = {param}")
    res = ot.solve_bary_sample(X_a_list=[x1, x2], n=35, **param)
    lst_res.append(res)
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
# Plot distributions and plans
# ----------

pl.figure(2, figsize=(16, 16))

for i, bname in enumerate(lst_unbalanced):
    for j, rname in enumerate(lst_regs):
        pl.subplot(len(lst_unbalanced), len(lst_regs), i * len(lst_regs) + j + 1)

        X = lst_res[i * len(lst_regs) + j].X
        list_P = [lst_res[i * len(lst_regs) + j].list_res[k].plan for k in range(2)]
        loss = lst_res[i * len(lst_regs) + j].value

        plot2D_samples_mat(x1, X, list_P[0])
        plot2D_samples_mat(x2, X, list_P[1])

        if i == 0 and j == 0:  # add labels
            pl.plot(x1[:, 0], x1[:, 1], "ob", label="Source distribution 1", **style)
            pl.plot(x2[:, 0], x2[:, 1], "or", label="Source distribution 2", **style)
            pl.plot(X[:, 0], X[:, 1], "og", label="Barycenter distribution", **style)
            pl.legend(loc="best")
        else:
            pl.plot(x1[:, 0], x1[:, 1], "ob", **style)
            pl.plot(x2[:, 0], x2[:, 1], "or", **style)
            pl.plot(X[:, 0], X[:, 1], "og", **style)

        if i == 0:
            pl.title(rname)
        if j == 0:
            pl.ylabel(bname, fontsize=14)
