#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import ot

from ot.datasets import get_1D_gauss as gauss
reload(ot.lp)

#%% parameters

n=5000 # nb bins
m=6000 # nb bins

mean1 = 1000
mean2 = 1100

# bin positions
x=np.arange(n,dtype=np.float64)
y=np.arange(m,dtype=np.float64)

# Gaussian distributions
a=gauss(n,m=mean1,s=5) # m= mean, s= std

b=gauss(m,m=mean2,s=10)

# loss matrix
M=ot.dist(x.reshape((-1,1)), y.reshape((-1,1))) ** (1./2)
#M/=M.max()

#%%

print('Computing {} EMD '.format(1))

# emd loss 1 proc
ot.tic()
G, alpha, beta = ot.emd(a,b,M, dual_variables=True)
ot.toc('1 proc : {} s')

cost1 = (G * M).sum()
cost_dual = np.vdot(a, alpha) + np.vdot(b, beta)

# emd loss 1 proc
ot.tic()
cost_emd2 = ot.emd2(a,b,M)
ot.toc('1 proc : {} s')

ot.tic()
G2 = ot.emd(b, a, np.ascontiguousarray(M.T))
ot.toc('1 proc : {} s')

cost2 = (G2 * M.T).sum()

M_reduced = M - alpha.reshape(-1,1) - beta.reshape(1, -1)

# Check that both cost computations are equivalent
np.testing.assert_almost_equal(cost1, cost_emd2)
# Check that dual and primal cost are equal
np.testing.assert_almost_equal(cost1, cost_dual)
# Check symmetry
np.testing.assert_almost_equal(cost1, cost2)
# Check with closed-form solution for gaussians
np.testing.assert_almost_equal(cost1, np.abs(mean1-mean2))

[ind1, ind2] = np.nonzero(G)

# Check that reduced cost is zero on transport arcs
np.testing.assert_array_almost_equal((M - alpha.reshape(-1, 1) - beta.reshape(1, -1))[ind1, ind2], np.zeros(ind1.size))