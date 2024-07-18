# %%
# -*- coding: utf-8 -*-
r"""
====================================================
GMM Plan 1D
====================================================

Illustration of the GMM plan for 
the Mixture Wasserstein between two GMM in 1D.

"""

# Author: Eloi Tanguy <eloi.tanguy@u-paris>
#         Remi Flamary <remi.flamary@polytehnique.edu>
#         Julie Delon <julie.delon@math.cnrs.fr>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 4

import numpy as np
import matplotlib.pylab as pl
from matplotlib import colormaps as cm 
import ot
import ot.plot
from ot.utils import proj_SDP, proj_simplex
from ot.gmm import gmm_ot_loss, gmm_ot_plan_density, gmm_ot_plan, gmm_pdf

# %%
##############################################################################
# Generate data and plot it
# -------------------------
np.random.seed(3)
ks = 3
kt = 2
d = 1
eps = 0.1
m_s = np.random.rand(ks, d) 
m_t = np.random.rand(kt, d) 
C_s = np.random.randn(ks, d, d)*0.1
C_s = np.matmul(C_s, np.transpose(C_s, (0, 2, 1)))
C_t = np.random.randn(kt, d, d)*0.1
C_t = np.matmul(C_t, np.transpose(C_t, (0, 2, 1))) 
w_s = ot.unif(ks)
w_t = ot.unif(kt)

axis = [-3, 3, -3, 3]
pl.figure(1, (20, 10))
pl.clf()


# %%
##############################################################################
# Compute plan 
# ------------

n = 100
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
xx, yy = np.meshgrid(x, y)
xx = xx.reshape((n**2, 1))
yy = yy.reshape((n**2, 1))
plan = gmm_ot_plan_density(xx, yy, m_s, m_t, C_s, C_t, w_s, w_t, plan=None, atol=0.1)

a = gmm_pdf(x[:,None], m_s, C_s, w_s)
b = gmm_pdf(y[:,None], m_t, C_t, w_t)
plan = plan.reshape((n,n))
ot.plot.plot1D_mat(a, b, plan, title='Plan between two GMM')

# %%
