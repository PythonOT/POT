# -*- coding: utf-8 -*-
r"""
=================================================================================
DEMD vs LP Gradient Decent without Pytorch
=================================================================================


"""

# Author: Ronak Mehta <ronakrm@cs.wisc.edu> 
#         Xizheng Yu <xyu354@wisc.edu>
#
# License: MIT License

import io
import sys
import numpy as np
import matplotlib.pyplot as pl
import ot
from demd import demd, demd_minimize

# %%
# Define function to get random (n, d) data
# -------------------------------------------
def getData(n, d, dist='skewedGauss'):
	print(f'Data: {d} Random Dists with {n} Bins ***')

	x = np.arange(n, dtype=np.float64).reshape((n, 1))
	M = ot.utils.dist(x, metric='minkowski')

	data = []
	for i in range(d):
		# m = 100*np.random.rand(1)
		m = n*(0.5*np.random.rand(1))*float(np.random.randint(2)+1)
		if dist == 'skewedGauss':
			a = ot.datasets.make_1D_gauss(n, m=m, s=5)
		elif dist == 'uniform':
			a = np.random.rand(n)
			a = a / sum(a)
		else:
			print('unknown dist')
		data.append(a)

	return data, M

# %%
# Gradient Decent
# ---------------

# %% parameters and data
n = 50  # nb bins
d = 7

vecsize = n*d

# data, M = getData(n, d, 'uniform')
data, M = getData(n, d, 'skewedGauss')
data = np.vstack(data)

# %% demd
# Redirect the standard output to a string buffer
old_stdout = sys.stdout
sys.stdout = output_buffer = io.StringIO()

x = demd_minimize(demd, data, d, n, vecsize, niters=3000, lr=0.00001)

# after minimization, any distribution can be used as a estimate of barycenter
bary = x[0]

sys.stdout = old_stdout
output = output_buffer.getvalue()

rows = output.strip().split("\n")
demd_loss = [float(row.split()[-3]) for row in rows[1:]]

print(output)

# %% lp barycenter
def lp_1d_bary(data, M, n, d):

	A = np.vstack(data).T

	alpha = 1.0 #/d  # 0<=alpha<=1
	weights = np.array(d*[alpha]) 

	bary, bary_log = ot.lp.barycenter(A, M, weights, solver='interior-point', verbose=True, log=True)
	
	return bary_log['fun'], bary

# Redirect the standard output to a string buffer
old_stdout = sys.stdout
sys.stdout = output_buffer = io.StringIO()

obj, lp_bary = lp_1d_bary(data, M, n, d) 

# Restore the standard output and get value
sys.stdout = old_stdout
output = output_buffer.getvalue()

rows = output.strip().split("\n")
lp_loss = [float(row.split()[-1]) for row in rows[1:-3]]

print(output)


#%%
# Compare the loss between DEMD and LP Barycenter
# ---------
# The barycenter approach does not minize the distance between
# the distributions, while our DEMD does.
index = [*range(0, len(demd_loss))]

pl.plot(index, demd_loss, label = "DEMD")
pl.plot(index, lp_loss[:len(demd_loss)], label="LP")
pl.yscale('log')
pl.ylabel('Loss')
pl.xlabel('Epochs')
pl.legend()