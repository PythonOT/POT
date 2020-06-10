# coding: utf-8
"""
=============================================
Introduction to Optimal Transport with Python
=============================================

This example gives an introduction on how to use Optimal Transport in Python.

"""

# Author: Remi Flamary, Nicolas Courty, Aurelie Boisbunon
#
# License: MIT License
# sphinx_gallery_thumbnail_number = 1

##############################################################################
# POT Python Optimal Transport Toolbox
# ------------------------------------
#
# POT installation
# ```````````````````
#
# * Install with pip::
#
#     pip install pot
# * Install with conda::
#
#     conda install -c conda-forge pot
#
# Import the toolbox
# ```````````````````
#

import numpy as np  # always need it
import pylab as pl  # do the plots

import ot  # ot

##############################################################################
# Getting help
# `````````````
#
# Online  documentation : `<https://pythonot.github.io/all.html>`_
#
# Or inline help:
#

help(ot.dist)


##############################################################################
# First OT Problem
# ----------------
#
# We will solve the Bakery/Cafés problem of transporting croissants from a
# number of Bakeries to Cafés in a City (In this case Manhattan). We did a
# quick google map search in Manhattan for bakeries and Cafés:
#
# .. image:: images/bak.png
#     :align: center
#     :alt: bakery-cafe-manhattan
#     :width: 600px
#     :height: 280px
#
# We extracted from this search their positions and generated fictional
# production and sale number (that both sum to the same value).
#
# We have acess to the position of Bakeries ``bakery_pos`` and their
# respective production ``bakery_prod`` which describe the source
# distribution. The Cafés where the croissants are sold are defiend also by
# their position ``cafe_pos`` and ``cafe_prod``. For fun we also provide a
# map ``Imap`` that will illustrate the position of these shops in the city.
#
#
# Now we load the data
#
#

data = np.load('../data/manhattan.npz')

bakery_pos = data['bakery_pos']
bakery_prod = data['bakery_prod']
cafe_pos = data['cafe_pos']
cafe_prod = data['cafe_prod']
Imap = data['Imap']

print('Bakery production: {}'.format(bakery_prod))
print('Cafe sale: {}'.format(cafe_prod))
print('Total croissants : {}'.format(cafe_prod.sum()))


##############################################################################
# Plotting bakeries in the city
# -----------------------------
#
# Next we plot the position of the bakeries and cafés on the map. The size of
# the circle is proportional to their production.
#

pl.figure(1, (8, 7))
pl.clf()
pl.imshow(Imap, interpolation='bilinear')  # plot the map
pl.scatter(bakery_pos[:, 0], bakery_pos[:, 1], s=bakery_prod, c='r', ec='k', label='Bakeries')
pl.scatter(cafe_pos[:, 0], cafe_pos[:, 1], s=cafe_prod, c='b', ec='k', label='Cafés')
pl.legend()
pl.title('Manhattan Bakeries and Cafés')


##############################################################################
# Cost matrix
# -----------
#
#
# We compute the cost matrix between the bakeries and the cafés, this will be
# the transport cost matrix. This can be done using the
# `ot.dist <https://pythonot.github.io/all.html#ot.dist>`_ that defaults to
# squared euclidean distance but can return other things such as cityblock
# (or manhattan distance).
#
#

C = ot.dist(bakery_pos, cafe_pos)

labels = [str(i) for i in range(len(bakery_prod))]
f = pl.figure(2, (13, 6), constrained_layout=True)
pl.clf()
pl.subplot(121)
pl.imshow(Imap, interpolation='bilinear')  # plot the map
for i in range(len(cafe_pos)):
    pl.text(cafe_pos[i, 0], cafe_pos[i, 1], labels[i], color='b',
            fontsize=14, fontweight='bold', ha='center', va='center')
for i in range(len(bakery_pos)):
    pl.text(bakery_pos[i, 0], bakery_pos[i, 1], labels[i], color='r',
            fontsize=14, fontweight='bold', ha='center', va='center')
pl.title('Manhattan Bakeries and Cafés')

ax = pl.subplot(122)
im = pl.imshow(C)
pl.title('Cost matrix')
cbar = pl.colorbar(im, ax=ax, shrink=0.5, use_gridspec=True)
cbar.ax.set_ylabel("cost", rotation=-90, va="bottom")

pl.xlabel('Cafés')
pl.ylabel('Bakeries')
pl.show()


##############################################################################
# Solving the OT problem with `ot.emd <https://pythonot.github.io/all.html#ot.emd>`_
# -----------------------------------------------------------------------------------

ot_emd = ot.emd(bakery_prod, cafe_prod, C)

# Transportation plan vizualization
# `````````````````````````````````
#
# A good vizualization of the OT matrix in the 2D plane is to denote the
# transportation of mass between a Bakery and a Café by a line. This can easily
# be done with a double ``for`` loop.
#
# In order to make it more interpretable one can also use the ``alpha``
# parameter of plot and set it to ``alpha=G[i,j]/G.max()``.

# Plot the matrix and the map
f = pl.figure(3, (13, 6), constrained_layout=True)
pl.clf()
pl.subplot(121)
pl.imshow(Imap, interpolation='bilinear')  # plot the map
for i in range(len(bakery_pos)):
    for j in range(len(cafe_pos)):
        pl.plot([bakery_pos[i, 0], cafe_pos[j, 0]], [bakery_pos[i, 1], cafe_pos[j, 1]],
                '-k', lw=3. * ot_emd[i, j] / ot_emd.max())
for i in range(len(cafe_pos)):
    pl.text(cafe_pos[i, 0], cafe_pos[i, 1], labels[i], color='b', fontsize=14,
            fontweight='bold', ha='center', va='center')
for i in range(len(bakery_pos)):
    pl.text(bakery_pos[i, 0], bakery_pos[i, 1], labels[i], color='r', fontsize=14,
            fontweight='bold', ha='center', va='center')
pl.title('Manhattan Bakeries and Cafés')

ax = pl.subplot(122)
im = pl.imshow(ot_emd)
pl.title('Transport matrix')
cbar = f.colorbar(im, ax=ax, shrink=0.5, use_gridspec=True)
cbar.ax.set_ylabel("transport", rotation=-90, va="bottom")

pl.xlabel('Cafés')
pl.ylabel('Bakeries')
pl.show()


##############################################################################
# OT loss and dual variables
# --------------------------
#
# The resulting wasserstein loss loss is of the form:
#
# .. math::
#     W=\sum_{i,j}\gamma_{i,j}C_{i,j}
#
# where :math:`\gamma` is the optimal transport matrix.
#

W = np.sum(ot_emd * C)
print('Wasserstein loss = {0:.3f}'.format(W))

##############################################################################
# Regularized OT with Sinkhorn
# ----------------------------
#
# The Sinkhorn algorithm is very simple to code. You can implement it directly
# using the following pseudo-code
#
# .. image:: images/sinkhorn.png
#     :align: center
#     :alt: Sinkhorn algorithm
#     :width: 440px
#     :height: 240px
#
# An alternative is to use the POT toolbox with
# `ot.sinkhorn <https://pythonot.github.io/all.html#ot.sinkhorn>`_
#
# Be carefull to numerical problems. A good pre-processing for Sinkhorn is to
# divide the cost matrix ``C`` by its maximum value.

# Compute Sinkhorn transport matrix
ot_sinkhorn = ot.sinkhorn(bakery_prod, cafe_prod, reg=0.1, M=C / C.max())

# Plot the matrix and the map
f = pl.figure(4, (13, 6), constrained_layout=True)
pl.clf()
pl.subplot(121)
pl.imshow(Imap, interpolation='bilinear')  # plot the map
for i in range(len(bakery_pos)):
    for j in range(len(cafe_pos)):
        pl.plot([bakery_pos[i, 0], cafe_pos[j, 0]],
                [bakery_pos[i, 1], cafe_pos[j, 1]],
                '-k', lw=3. * ot_sinkhorn[i, j] / ot_sinkhorn.max())
for i in range(len(cafe_pos)):
    pl.text(cafe_pos[i, 0], cafe_pos[i, 1], labels[i], color='b',
            fontsize=14, fontweight='bold', ha='center', va='center')
for i in range(len(bakery_pos)):
    pl.text(bakery_pos[i, 0], bakery_pos[i, 1], labels[i], color='r',
            fontsize=14, fontweight='bold', ha='center', va='center')
pl.title('Manhattan Bakeries and Cafés')

ax = pl.subplot(122)
im = pl.imshow(ot_sinkhorn)
pl.title('Transport matrix')
cbar = f.colorbar(im, ax=ax, shrink=0.5, use_gridspec=True)
cbar.ax.set_ylabel("transport", rotation=-90, va="bottom")

pl.xlabel('Cafés')
pl.ylabel('Bakeries')
pl.show()

# Compute the Wasserstein loss for Sinkhorn, and compare with EMD
W_sinkhorn = np.sum(ot_sinkhorn * C)
print('Wasserstein loss (EMD) = {0:.3f}'.format(W))
print('Wasserstein loss (Sink) = {0:.3f}'.format(W_sinkhorn))
