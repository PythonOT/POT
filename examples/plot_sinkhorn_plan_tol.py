"""
Full-plan stopping for dense NumPy Sinkhorn
===========================================

A marginal residual and the error in the whole transport matrix are
different quantities. This example uses an independent, planted entropic
optimum to compare POT's standard stopping rule and the opt-in full-plan
tolerance.
"""

import numpy as np

import ot

x = np.linspace(0, 1, 64)
y = np.linspace(0, 1, 128)
M = 0.5 * (x[:, None] - y[None, :]) ** 2
reg = 0.02

# An exact positive diagonal scaling of the Gibbs kernel is the entropic
# optimum for its induced marginals. Use it as a numerical reference.
reference = np.exp(-M / reg) * (0.2 + x[:, None]) * (1.3 - y)
reference /= reference.sum()
a, b = reference.sum(axis=1), reference.sum(axis=0)

target = 2.5e-8
stock, stock_log = ot.sinkhorn(a, b, M, reg, stopThr=1e-7, log=True)
plan, plan_log = ot.sinkhorn(a, b, M, reg, stopThr=1e-7, plan_tol=target, log=True)

print(f"Requested whole-plan TV tolerance: {target:.3e}")
print(f"Observed TV (ordinary): {0.5 * np.abs(stock - reference).sum():.3e}")
print(f"Observed TV (opt-in):   {0.5 * np.abs(plan - reference).sum():.3e}")
print("Float64 TV estimate:    {:.3e}".format(plan_log["plan_tv_estimate"]))
print("Updates (ordinary / opt-in):", stock_log["niter"] + 1, plan_log["niter"] + 1)
