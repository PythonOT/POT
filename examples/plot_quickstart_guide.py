# coding: utf-8
"""
=============================================
Quickstart Guide
=============================================

.. note::
    Example added in release: 0.9.6

Quickstart guide to the POT toolbox.

For better readability, only the use of POT is provided and the plotting code
with matplotlib is hidden (but is available in the source file of the example).

.. note::
    We use here the unified API of POT which is more flexible and allows to solve a wider range of problems with just a few functions. The classical API is still available (the unified API
    one is a convenient wrapper around the classical one) and we provide pointers to the
    classical API when needed.

"""

# Author: Remi Flamary
#
# License: MIT License
# sphinx_gallery_thumbnail_number = 4

# Import necessary libraries

import numpy as np
import pylab as pl

import ot


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
a = ot.utils.unif(n1)  # weights of points in the source domain
b = ot.utils.unif(n2)  # weights of points in the target domain

x1 = np.random.randn(n1, 2)
x1 /= np.sqrt(np.sum(x1**2, 1, keepdims=True)) / 2

x2 = np.random.randn(n2, 2)
x2 /= np.sqrt(np.sum(x2**2, 1, keepdims=True)) / 4

# Compute the cost matrix
C = ot.dist(x1, x2)  # Squared Euclidean cost matrix by default

# sphinx_gallery_start_ignore
style = {"markeredgecolor": "k"}


def plot_plan(P=None, title="", axis=True):
    if P is not None:
        plot2D_samples_mat(x1, x2, P)
    pl.plot(x1[:, 0], x1[:, 1], "ob", label="Source samples", **style)
    pl.plot(x2[:, 0], x2[:, 1], "or", label="Target samples", **style)
    if not axis:
        pl.axis("off")
    pl.title(title)


pl.figure(1, (4, 4))
plot_plan(title="Source and target distributions")
pl.legend(loc=0)
pl.show()

pl.figure(2, (3.5, 1.7))
pl.imshow(C)
pl.colorbar()
pl.title("Cost matrix C")

# sphinx_gallery_end_ignore

# %%
#
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

# sphinx_gallery_start_ignore
from ot.plot import plot2D_samples_mat

pl.figure(1, (8, 4))

pl.subplot(1, 2, 1)
plot_plan(P, "OT plan P loss={:.3f}".format(loss))

pl.subplot(1, 2, 2)
pl.scatter(x1[:, 0], x1[:, 1], c=alpha, cmap="viridis", edgecolors="k")
pl.scatter(x2[:, 0], x2[:, 1], c=beta, cmap="plasma", edgecolors="k")
pl.title("Dual potentials")
pl.show()


pl.figure(2, (3, 1.7))
pl.imshow(P, cmap="Greys")
pl.title("OT plan")
pl.show()
# sphinx_gallery_end_ignore

# %%
# The figure above shows the Optimal Transport plan between the source and target
# samples. The color intensity represents the amount of mass transported
# between the samples. The dual potentials of the OT problem are also shown.
#
# The weights of the samples in the source and target domains :code:`a` and
# :code:`b` are given to the function. If not provided, the weights are assumed
# to be uniform See :func:`ot.solve_sample` for more details.
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
# Optimal Transport problem with a custom cost matrix
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The cost matrix can be customized by passing it to the more general
# :func:`ot.solve` function. The cost matrix should be a matrix of size
# :code:`(n1, n2)` where :code:`C[i,j]` is the cost of transporting mass from
# :code:`x1[i]` to :code:`x2[j]`.
#
# In this example, we use the Citybloc distance as the cost matrix.

# Compute the cost matrix
C_city = ot.dist(x1, x2, metric="cityblock")

# Solve the OT problem with the custom cost matrix
sol = ot.solve(C_city)
# the parameters a and b are not provided so uniform weights are assumed
P_city = sol.plan
# on empirical data the same can be done with ot.solve_sample :
# sol = ot.solve_sample(x1, x2, metric='cityblock')

# Compute the OT loss (equivalent to ot.solve(C).value)
loss_city = sol.value  # same as np.sum(P_city * C)

# sphinx_gallery_start_ignore
pl.figure(1, (3, 3))
plot_plan(P_city, "OT plan (Citybloc) loss={:.3f}".format(loss_city))

pl.figure(2, (3, 1.7))
pl.imshow(P_city, cmap="Greys")
pl.title("OT plan (Citybloc)")
pl.show()
# sphinx_gallery_end_ignore

# %%
# Note that we show here how to solve the OT problem with a custom cost matrix
# with the more general :func:`ot.solve` function.
# But the same can be done with the :func:`ot.solve_sample` function by passing
# :code:`metric='cityblock'` as argument.
#
# The cost matrix can be computed with the :func:`ot.dist` function which
# computes the pairwise distance between two sets of samples or can be provided
# directly as a matrix by the user when no samples are available.
#
# .. note::
#    The examples above use the unified API of POT. The classic API is still available
#    and and OT plan and loss can be computed with the :func:`ot.emd`  and
#    the :func:`ot.emd2` functions as below:
#
#    .. code-block:: python
#
#       P = ot.emd(a, b, C)
#       loss = ot.emd2(a, b, C) # same as np.sum(P*C) but differentiable wrt a/b
#
#
# Sinkhorn and Regularized OT
# ---------------------------
#
# Entropic OT with Sinkhorn algorithm
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Solve the Sinkhorn problem (just add reg parameter value)
sol = ot.solve_sample(x1, x2, a, b, reg=1e-1)

# get the OT plan and loss
P_sink = sol.plan
loss_sink = sol.value  # objective value of the Sinkhorn problem (incl. entropy)
loss_sink_linear = sol.value_linear  # np.sum(P_sink * C) linear part of loss

# sphinx_gallery_start_ignore
pl.figure(1, (3, 3))
plot_plan(P_sink, "Sinkhorn OT plan loss={:.3f}".format(loss_sink))
pl.show()

pl.figure(2, (3, 1.7))
pl.imshow(P_sink, cmap="Greys")
pl.title("Sinkhorn OT plan")
pl.show()
# sphinx_gallery_end_ignore
# %%
# The Sinkhorn algorithm solves the Entropic Regularized OT problem. The
# regularization strength can be controlled with the :code:`reg` parameter.
# The Sinkhorn algorithm can be faster than the exact OT solver for large
# regularization strength but the solution is only an approximation of the
# exact OT problem and the OT plan is not sparse.
#
# Quadratic Regularized OT
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Use quadratic regularization
P_quad = ot.solve_sample(x1, x2, a, b, reg=3, reg_type="L2").plan

loss_quad = ot.solve_sample(x1, x2, a, b, reg=3, reg_type="L2").value

# sphinx_gallery_start_ignore
pl.figure(1, (9, 3))

pl.subplot(1, 3, 1)
plot_plan(P, "OT plan loss={:.3f}".format(loss))

pl.subplot(1, 3, 2)
plot_plan(P_sink, "Sinkhorn plan loss={:.3f}".format(loss_sink))

pl.subplot(1, 3, 3)
plot_plan(P_quad, "Quadratic reg plan loss={:.3f}".format(loss_quad))
pl.show()
# sphinx_gallery_end_ignore
# %%
# We plot above the OT plans obtained with different regularizations. The
# quadratic regularization is another common choice for regularized OT and
# preserves the sparsity of the OT plan.
#
# Solve the Regularized OT problem with user-defined regularization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


# Define a custom regularization function
def f(G):
    return 0.5 * np.sum(G**2)


def df(G):
    return G


P_reg = ot.solve_sample(x1, x2, a, b, reg=3, reg_type=(f, df)).plan

# sphinx_gallery_start_ignore
pl.figure(1, (3, 3))
plot_plan(P_reg, "User-defined reg plan")
pl.show()
# sphinx_gallery_end_ignore
# %%
#
# .. note::
#    The examples above use the unified API of POT. The classic API is still available
#    and and the entropic OT plan and loss can be computed with the
#    :func:`ot.sinkhorn` # and :func:`ot.sinkhorn2` functions as below:
#
#    .. code-block:: python
#
#      Gs = ot.sinkhorn(a, b, C, reg=1e-1)
#      loss_sink = ot.sinkhorn2(a, b, C, reg=1e-1)
#
#    For quadratic regularization, the :func:`ot.smooth.smooth_ot_dual` function
#    can be used to compute the solution of the regularized OT problem. For
#    user-defined regularization, the :func:`ot.optim.cg` function can be used
#    to solve the regularized OT problem with Conditional Gradient algorithm.
#
# Unbalanced and Partial OT
# ----------------------------
#
# Unbalanced Optimal Transport
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Unbalanced OT relaxes the marginal constraints and allows for the source and
# target total weights to be different. The :func:`ot.solve_sample` function can be
# used to solve the unbalanced OT problem by setting the marginal penalization
# :code:`unbalanced` parameter to a positive value.
#

# Solve the unbalanced OT problem with KL penalization
P_unb_kl = ot.solve_sample(x1, x2, a, b, unbalanced=5e-2).plan

# Unbalanced with KL penalization ad KL regularization
P_unb_kl_reg = ot.solve_sample(
    x1, x2, a, b, unbalanced=5e-2, reg=1e-1
).plan  # also regularized

# Unbalanced with L2 penalization
P_unb_l2 = ot.solve_sample(x1, x2, a, b, unbalanced=7e1, unbalanced_type="L2").plan

# sphinx_gallery_start_ignore
pl.figure(1, (9, 3))

pl.subplot(1, 3, 1)
plot_plan(P_unb_kl, "Unbalanced KL plan")

pl.subplot(1, 3, 2)
plot_plan(P_unb_kl_reg, "Unbalanced KL + reg plan")

pl.subplot(1, 3, 3)
plot_plan(P_unb_l2, "Unbalanced L2 plan")
pl.show()
# sphinx_gallery_end_ignore
# %%
# .. note::
#    Solving the unbalanced OT problem with the classic API can be done with the
#    :func:`ot.unbalanced.sinkhorn_unbalanced` function as below:
#
#    .. code-block:: python
#
#      G_unb_kl = ot.unbalanced.sinkhorn_unbalanced(a, b, C, eps=reg, alpha=unbalanced)
#
# Partial Optimal Transport
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Solve the Unbalanced OT problem with TV penalization (equivalent)
P_part_pen = ot.solve_sample(x1, x2, a, b, unbalanced=3, unbalanced_type="TV").plan

# Solve the Partial OT problem with mass constraints (only classic API)
P_part_const = ot.partial.partial_wasserstein(a, b, C, m=0.5)  # 50% mass transported

# sphinx_gallery_start_ignore
pl.figure(1, (6, 3))

pl.subplot(1, 2, 1)
plot_plan(P_part_pen, "Partial TV plan")

pl.subplot(1, 2, 2)
plot_plan(P_part_const, "Partial 50% mass plan")
pl.show()

# sphinx_gallery_end_ignore
# %%
#
# Gromov-Wasserstein and Fused Gromov-Wasserstein
# -------------------------------------
#
# Gromov-Wasserstein and Entropic GW
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The Gromov-Wasserstein distance is a similarity measure between metric
# measure spaces. So it does not require the samples to be in the same space.
#

# Define the metric cost matrices in each spaces

C1 = ot.dist(x1, x1, metric="sqeuclidean")
C2 = ot.dist(x2, x2, metric="sqeuclidean")

C1 /= C1.max()
C2 /= C2.max()

# Solve the Gromov-Wasserstein problem
sol_gw = ot.solve_gromov(C1, C2, a=a, b=b)
P_gw = sol_gw.plan
loss_gw = sol_gw.value  # quadratic + reg if reg>0
loss_gw_quad = sol_gw.value_quad  # quadratic part of loss

# Solve the Entropic Gromov-Wasserstein problem
P_egw = ot.solve_gromov(C1, C2, a=a, b=b, reg=1e-2).plan

# sphinx_gallery_start_ignore
pl.figure(1, (6, 3))

pl.subplot(1, 2, 1)
plot_plan(P_gw, "GW plan")

pl.subplot(1, 2, 2)
plot_plan(P_egw, "Entropic GW plan")
pl.show()
# sphinx_gallery_end_ignore
# %%
# .. note::
#    The Gromov-Wasserstein problem can be solved with the classic API using the
#    :func:`ot.gromov.gromov_wasserstein` function and the Entropic
#    Gromov-Wasserstein problem can be solved with the
#    :func:`ot.gromov.entropic_gromov_wasserstein` function.
#
#    .. code-block:: python
#
#      P_gw = ot.gromov.gromov_wasserstein(C1, C2, a, b)
#      P_egw = ot.gromov.entropic_gromov_wasserstein(C1, C2, a, b, epsilon=reg)
#
#      loss_gw = ot.gromov.gromov_wasserstein2(C1, C2, a, b)
#      loss_egw = ot.gromov.entropic_gromov_wasserstein2(C1, C2, a, b, epsilon=reg)
#
# Fused Gromov-Wasserstein
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Cost matrix
M = C / np.max(C)

# Solve FGW problem with alpha=0.1
sol = ot.solve_gromov(C1, C2, M, a=a, b=b, alpha=0.1)
P_fgw = sol.plan
loss_fgw = sol.value
loss_fgw_linear = sol.value_linear  # linear part of loss (wrt M)
loss_fgw_quad = sol.value_quad  # quadratic part of loss (wrt C1 and C2)

# Solve entropic FGW problem with alpha=0.1
P_efgw = ot.solve_gromov(C1, C2, M, a=a, b=b, alpha=0.1, reg=1e-3).plan

# sphinx_gallery_start_ignore
pl.figure(1, (6, 3))

pl.subplot(1, 2, 1)
plot_plan(P_fgw, "FGW plan")

pl.subplot(1, 2, 2)
plot_plan(P_efgw, "Entropic FGW plan")
pl.show()

# sphinx_gallery_end_ignore
# %%
# .. note::
#    The Fused Gromov-Wasserstein problem can be solved with the classic API using
#    the :func:`ot.gromov.fused_gromov_wasserstein` function and the Entropic
#    Fused Gromov-Wasserstein problem can be solved with the
#    :func:`ot.gromov.entropic_fused_gromov_wasserstein` function.
#
#    .. code-block:: python
#
#      P_fgw = ot.gromov.fused_gromov_wasserstein(C1, C2, M, a, b, alpha=0.1)
#      P_efgw = ot.gromov.entropic_fused_gromov_wasserstein(C1, C2, M, a, b, alpha=0.1, epsilon=reg)
#
#      loss_fgw = ot.gromov.fused_gromov_wasserstein2(C1, C2, M, a, b, alpha=0.1)
#      loss_efgw = ot.gromov.entropic_fused_gromov_wasserstein2(C1, C2, M, a, b, alpha=0.1, epsilon=reg)
#
# Large scale OT
# --------------
#
# We discuss here strategies to solve large scale OT problems using approximations
# of the exact OT problem.
#
# Large scale Sinkhorn
# ~~~~~~~~~~~~~~~~~~~~
#
# When having samples with a large number of points, the Sinkhorn algorithm can
# be implemented in a Lazy version which is more memory efficient and avoids
# the computation of the :math:`n \times m` cost matrix.
#
# POT provides two implementation of the lazy Sinkhorn algorithm that return their
# results in a lazy form of type :class:`ot.utils.LazyTensor`. This object can be
# used to compute the loss or the OT plan in a lazy way or to recover its values
# in a dense form.
#

# Solve the Sinkhorn problem in a lazy way
sol = ot.solve_sample(x1, x2, a, b, reg=1e-1, lazy=True)

# Solve the sinkhoorn in a lazy way with geomloss
sol_geo = ot.solve_sample(x1, x2, a, b, reg=1e-1, method="geomloss", lazy=True)

# get the OT lazy plan and loss
P_sink_lazy = sol.lazy_plan

# recover values for Lazy plan
P12 = P_sink_lazy[1, 2]
P1dots = P_sink_lazy[1, :]
# convert to dense matrix !!warning this can be memory consuming
P_sink_lazy_dense = P_sink_lazy[:]

# sphinx_gallery_start_ignore
pl.figure(1, (3, 3))
plot_plan(P_sink_lazy_dense, "Lazy Sinkhorn OT plan")
pl.show()

pl.figure(2, (3, 1.7))
pl.imshow(P_sink_lazy_dense, cmap="Greys")
pl.title("Lazy Sinkhorn OT plan")
pl.show()

# sphinx_gallery_end_ignore
# %%
# .. note::
#    The lazy Sinkhorn algorithm can be found in the classic API with the
#    :func:`ot.bregman.empirical_sinkhorn` function with parameter
#    :code:`lazy=True`. Similarly the geoloss implementation is available
#    with the :func:`ot.bregman.empirical_sinkhorn2_geomloss`.
#
#
# the first example shows how to solve the Sinkhorn problem in a lazy way with
# the default POT implementation. The second example shows how to solve the
# Sinkhorn problem in a lazy way with the PyKeops/Geomloss implementation that provides
# a very efficient way to solve large scale problems on low dimensionality
# samples.
#
# Factored and Low rank OT
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# The Sinkhorn algorithm can be implemented in a low rank version that
# approximates the OT plan with a low rank matrix. This can be useful to
# accelerate the computation of the OT plan for large scale problems.
# A similar non-regularized version of low rank factorization is also available.
#

# Solve the Factored OT problem (use lazy=True for large scale)
P_fact = ot.solve_sample(x1, x2, a, b, method="factored", rank=15).plan

P_lowrank = ot.solve_sample(x1, x2, a, b, reg=0.1, method="lowrank", rank=10).plan

# sphinx_gallery_start_ignore
pl.figure(1, (6, 3))

pl.subplot(1, 2, 1)
plot_plan(P_fact, "Factored OT plan")

pl.subplot(1, 2, 2)
plot_plan(P_lowrank, "Low rank OT plan")
pl.show()

pl.figure(2, (6, 1.7))

pl.subplot(1, 2, 1)
pl.imshow(P_fact, cmap="Greys")
pl.title("Factored OT plan")

pl.subplot(1, 2, 2)
pl.imshow(P_lowrank, cmap="Greys")
pl.title("Low rank OT plan")
pl.show()

# sphinx_gallery_end_ignore
# %%
# .. note::
#    The factored OT problem can be solved with the classic API using the
#    :func:`ot.factored.factored_optimal_transport` function and the low rank
#    OT problem can be solved with the :func:`ot.lowrank.lowrank_sinkhorn` function.
#
# Gaussian OT with Bures-Wasserstein
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The Gaussian Wasserstein  or Bures-Wasserstein distance is the Wasserstein distance
# between Gaussian distributions. It can be used as an approximation of the
# Wasserstein distance between empirical distributions by estimating the
# covariance matrices of the samples.
#

# Compute the Bures-Wasserstein distance
bw_value = ot.solve_sample(x1, x2, a, b, method="gaussian").value

print(f"Exact OT loss = {loss:1.3f}")
print(f"Bures-Wasserstein distance = {bw_value:1.3f}")

# %%
# .. note::
#    The Gaussian Wasserstein problem can be solved with the classic API using the
#    :func:`ot.gaussian.empirical_bures_wasserstein_distance` function.
#
# Comparing all OT plans
# ----------------------
#
# The figure below shows all the OT plans computed in this example.
# The color intensity represents the amount of mass transported
# between the samples.
#

# plot all plans

# sphinx_gallery_start_ignore
pl.figure(1, (9, 13))

pl.subplot(4, 3, 1)
plot_plan(P, "OT plan", axis=False)

pl.subplot(4, 3, 2)
plot_plan(P_sink, "Sinkhorn plan", axis=False)

pl.subplot(4, 3, 3)
plot_plan(P_quad, "Quadratic reg. plan", axis=False)

pl.subplot(4, 3, 4)
plot_plan(P_unb_kl, "Unbalanced KL plan", axis=False)

pl.subplot(4, 3, 5)
plot_plan(P_unb_kl_reg, "Unbalanced KL + reg plan", axis=False)

pl.subplot(4, 3, 6)
plot_plan(P_unb_l2, "Unbalanced L2 plan", axis=False)

pl.subplot(4, 3, 7)
plot_plan(P_part_const, "Partial 50% mass plan", axis=False)

pl.subplot(4, 3, 8)
plot_plan(P_fact, "Factored OT plan", axis=False)

pl.subplot(4, 3, 9)
plot_plan(P_lowrank, "Low rank OT plan", axis=False)

pl.subplot(4, 3, 10)
plot_plan(P_gw, "GW plan", axis=False)

pl.subplot(4, 3, 11)
plot_plan(P_egw, "Entropic GW plan", axis=False)

pl.subplot(4, 3, 12)
plot_plan(P_fgw, "Fused GW plan", axis=False)
pl.show()

# sphinx_gallery_end_ignore
# %%
#
