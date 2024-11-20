# -*- coding: utf-8 -*-
"""
================================================
Different gradient computations for regularized optimal transport
================================================

This example illustrates the differences in terms of computation time between the gradient options for the Sinkhorn solver.

"""

# Author: Sonia Mazelet <sonia.mazelet@polytechnique.edu>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 4

import numpy as np
import matplotlib.pylab as pl
import ot
from ot.datasets import make_1D_gauss as gauss
from ot.backend import torch


##############################################################################
# Time comparison of the Sinkhorn solver for different gradient options
# -------------


# %% parameters

n = 100  # nb bins
n_trials = 500
times_autodiff = torch.zeros(n_trials)
times_envelope = torch.zeros(n_trials)
times_last_step = torch.zeros(n_trials)

# bin positions
x = np.arange(n, dtype=np.float64)

# Time required for the Sinkhorn solver and gradient computations, for different gradient options over multiple Gaussian distributions
for i in range(n_trials):
    # Gaussian distributions with random parameters
    ma = np.random.randint(10, 40, 2)
    sa = np.random.randint(5, 10, 2)
    mb = np.random.randint(10, 40)
    sb = np.random.randint(5, 10)

    a = 0.6 * gauss(n, m=ma[0], s=sa[0]) + 0.4 * gauss(
        n, m=ma[1], s=sa[1]
    )  # m= mean, s= std
    b = gauss(n, m=mb, s=sb)

    # loss matrix
    M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
    M /= M.max()

    a = torch.tensor(a, requires_grad=True)
    b = torch.tensor(b, requires_grad=True)
    M = torch.tensor(M, requires_grad=True)

    # autodiff provides the gradient for all the outputs (plan, value, value_linear)
    ot.tic()
    res_autodiff = ot.solve(M, a, b, reg=10, grad="autodiff")
    res_autodiff.value.backward()
    times_autodiff[i] = ot.toq()

    # envelope provides the gradient for value
    ot.tic()
    res_envelope = ot.solve(M, a, b, reg=10, grad="envelope")
    res_envelope.value.backward()
    times_envelope[i] = ot.toq()

    # last_step provides the gradient for all the outputs, but only for the last iteration of the Sinkhorn algorithm
    ot.tic()
    res_last_step = ot.solve(M, a, b, reg=10, grad="last_step")
    res_last_step.value.backward()
    times_last_step[i] = ot.toq()

pl.figure(1, figsize=(4, 3))
pl.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
pl.boxplot(
    ([times_autodiff, times_envelope, times_last_step]),
    tick_labels=["autodiff", "envelope", "last_step"],
    showfliers=False,
)
pl.ylabel("Time (s)")
pl.show()
