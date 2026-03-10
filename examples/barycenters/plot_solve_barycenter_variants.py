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
# sphinx_gallery_thumbnail_number = 3

# %%

import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

##############################################################################
# Generate data
# -------------


# %% parameters

n = 50  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)[:, None]

# Gaussian distributions
a = 0.6 * gauss(n, m=15, s=5) + 0.4 * gauss(n, m=35, s=5)  # m= mean, s= std
b = gauss(n, m=25, s=5)


##############################################################################
# Plot distributions and loss matrix
# ----------------------------------

# %% plot the distributions

pl.figure(1, figsize=(6.4, 3))
pl.plot(x[:, 0], a, "b", label="Source distribution 1")
pl.plot(x[:, 0], b, "r", label="Source distribution 2")
pl.legend()


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
    ("Entropic Reg. OT", dict(reg=0.005)),
    # unbalanced OT KL
    ("Unbalanced KL No Reg.", dict(unbalanced=0.005)),
    (
        "Unbalanced KL with KL Reg.",
        dict(reg=0.0005, unbalanced=0.005, unbalanced_type="kl", reg_type="kl"),
    ),
]

lst_res = []
for name, param in lst_solvers:
    res = ot.solve_bary_sample(X_a_list=[x, x], n=50, a_list=[a, b], **param)
    lst_res.append(res)


##############################################################################
# Plot distributions and plans
# ----------

pl.figure(3, figsize=(9, 9))

for i, bname in enumerate(lst_unbalanced):
    for j, rname in enumerate(lst_regs):
        pl.subplot(len(lst_unbalanced), len(lst_regs), i * len(lst_regs) + j + 1)

        bary_bins = np.histogram(lst_res[i * len(lst_regs) + j], bins=x)[0]
        if i == 0 and j == 0:  # add labels
            pl.plot(x[:, 0], a, "b", label="Source distribution 1")
            pl.plot(x[:, 0], b, "r", label="Source distribution 2")
            pl.plot(x[:, 0], bary_bins, "g", label="Barycenter")
        else:
            pl.plot(x[:, 0], a, "b")
            pl.plot(x[:, 0], b, "r")
            pl.plot(x[:, 0], bary_bins, "g")

        for i, local_res in enumerate(lst_res[i * len(lst_regs) + j].list_res):
            plan = local_res.plan
            m2 = plan.sum(0)
            m1 = plan.sum(1)
            if i == 0:
                m1, m2 = m1 / a.max(), m2 * n
            else:
                m1, m2 = m1 / b.max(), m2 * n
            pl.imshow(plan, cmap="Greys")
            pl.plot(x[:, 0], m2 * 10, "g")
            pl.plot(m1 * 10, x, "b" if i == 0 else "r")

        pl.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        if i == 0:
            pl.title(rname)
        if j == 0:
            pl.ylabel(bname, fontsize=14)
