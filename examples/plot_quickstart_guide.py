# coding: utf-8
"""
=============================================
Quickstart Guide
=============================================


This is a quickstart guide to the Python Optimal Transport (POT) toolbox. We use
here the new API of POT which is more flexible and allows to solve a wider range
of problems with just a few functions. The old API is still available (the new
one is a convenient wrapper around the old one) and we provide pointers to the
old API when needed.

"""

# Author: Remi Flamary
#
# License: MIT License
# sphinx_gallery_thumbnail_number = 1

# Import necessary libraries

import numpy as np
import pylab as pl

import ot


# %%
# Example data
# --------------
#
# Data generation
# ~~~~~~~~~~~~~~~

# Problem size
n1 = 25
n2 = 50

# Generate random data
np.random.seed(0)
a = ot.utils.unif(n1)  # weights of points in the source domain
b = ot.utils.unif(n2)  # weights of points in the target domain

x1 = np.random.randn(n1, 2)
x1 /= (
    np.sqrt(np.sum(x1**2, 1, keepdims=True)) / 2
)  # project on the unit circle and scale
x2 = np.random.randn(n2, 2)
x2 /= (
    np.sqrt(np.sum(x2**2, 1, keepdims=True)) / 4
)  # project on the unit circle and scale

# %%
# Plot data
# ~~~~~~~~~

style = {"markeredgecolor": "k"}

pl.figure(1, (4, 4))
pl.plot(x1[:, 0], x1[:, 1], "ob", label="Source samples", **style)
pl.plot(x2[:, 0], x2[:, 1], "or", label="Target samples", **style)
pl.legend(loc=0)
pl.title("Source and target distributions")
pl.show()

# %%
# Solving exact Optimal Transport
# -------------------------------
# Solve the Optimal Transport problem between the samples
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The :func:`ot.solve_sample` function can be used to solve the Optimal Transport problem
# between two sets of samples. The function takes as its two first arguments the
# positions of the source and target samples, and returns an :class:`ot.utils.OTResult` object.

# Solve the OT problem
sol = ot.solve_sample(x1, x2, a, b)

# get the OT plan
P = sol.plan

# get the OT loss
loss = sol.value

# get the dual potentials
alpha, beta = sol.potentials

print(f"OT loss = {loss:1.3f}")

# %%
# We provide
# the weights of the samples in the source and target domains :code:`a` and
# :code:`b`. If not provided, the weights are assumed to be uniform.
#
# The :class:`ot.utils.OTResult` object contains the following attributes:
#
# - :code:`value`: the value of the OT problem
# - :code:`plan`: the OT matrix
# - :code:`potentials`: Dual potentials of the OT problem
# - :code:`log`: log dictionary of the solver
#
# The OT matrix :math:`P` is a matrix of size :code:`(n1, n2)` where
# :code:`P[i,j]` is the amount of mass
# transported from :code:`x1[i]` to :code:`x2[j]`.
#
# The OT loss is the sum of the element-wise product of the OT matrix and the
# cost matrix taken by default as the Squared Euclidean distance.
#

# %%
# Plot the OT plan and dual potentials
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from ot.plot import plot2D_samples_mat

pl.figure(1, (8, 4))

pl.subplot(1, 2, 1)
plot2D_samples_mat(x1, x2, P)
pl.plot(x1[:, 0], x1[:, 1], "ob", label="Source samples", **style)
pl.plot(x2[:, 0], x2[:, 1], "or", label="Target samples", **style)
pl.title("OT plan P loss={:.3f}".format(loss))

pl.subplot(1, 2, 2)
pl.scatter(x1[:, 0], x1[:, 1], c=alpha, cmap="viridis", edgecolors="k")
pl.scatter(x2[:, 0], x2[:, 1], c=beta, cmap="plasma", edgecolors="k")
pl.title("Dual potentials")
pl.show()


pl.figure(2, (3, 1.7))
pl.imshow(P, cmap="Greys")
pl.title("OT plan")
pl.show()

# %%
# Solve the Optimal Transport problem with a custom cost matrix
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The cost matrix can be customized by passing it to the more general
# :func:`ot.solve` function. The cost matrix should be a matrix of size
# :code:`(n1, n2)` where :code:`C[i,j]` is the cost of transporting mass from
# :code:`x1[i]` to :code:`x2[j]`.
#
# In this example, we use the Citybloc distance as the cost matrix.

# Compute the cost matrix
C = ot.dist(x1, x2, metric="cityblock")

# Solve the OT problem with the custom cost matrix
P_city = ot.solve(C).plan

# Compute the OT loss (equivalent to ot.solve(C).value)
loss_city = np.sum(P_city * C)

# %%
# Note that we show here how to sole the OT problem with a custom cost matrix
# with the more general :func:`ot.solve` function.
# But the same can be done with the :func:`ot.solve_sample` function by passing
# :code:`metric='cityblock'` as argument.
#
# .. note::
#    The examples above use the new API of POT. The old API is still available
#    and and OT plan and loss can be computed with the :func:`ot.emd`  and
#    the :func:`ot.emd2` functions as below:
#
#    .. code-block:: python
#
#       P = ot.emd(a, b, C)
#       loss = ot.emd2(a, b, C) # same as np.sum(P*C) but differentiable wrt a/b
#
# Plot the OT plan and dual potentials for other loss
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

pl.figure(1, (3, 3))
plot2D_samples_mat(x1, x2, P)
pl.plot(x1[:, 0], x1[:, 1], "ob", label="Source samples", **style)
pl.plot(x2[:, 0], x2[:, 1], "or", label="Target samples", **style)
pl.title("OT plan (Citybloc) loss={:.3f}".format(loss_city))

pl.figure(2, (3, 1.7))
pl.imshow(P_city, cmap="Greys")
pl.title("OT plan (Citybloc)")
pl.show()

# %%
# Sinkhorn and Regularized OT
# ---------------------------
#
# Solve Entropic Regularized OT with Sinkhorn algorithm
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Solve the Sinkhorn problem (just add reg parameter value)
sol = ot.solve_sample(x1, x2, a, b, reg=1e-1)

# get the OT plan and loss
P_sink = sol.plan
loss_sink = sol.value  # objective value of the Sinkhorn problem (incl. entropy)
loss_sink_linear = sol.value_linear  # np.sum(P_sink * C) linear part of loss

# %%
# The Sinkhorn algorithm solves the Entropic Regularized OT problem. The
# regularization strength can be controlled with the :code:`reg` parameter.
# The Sinkhorn algorithm can be faster than the exact OT solver for large
# regularization strength but the solution is only an approximation of the
# exact OT problem and the OT plan is not sparse.
#
# Plot the OT plan and dual potentials for Sinkhorn
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

pl.figure(1, (3, 3))
plot2D_samples_mat(x1, x2, P_sink)
pl.plot(x1[:, 0], x1[:, 1], "ob", label="Source samples", **style)
pl.plot(x2[:, 0], x2[:, 1], "or", label="Target samples", **style)
pl.title("Sinkhorn OT plan loss={:.3f}".format(loss_sink))
pl.show()

pl.figure(2, (3, 1.7))
pl.imshow(P_sink, cmap="Greys")
pl.title("Sinkhorn OT plan")
pl.show()
