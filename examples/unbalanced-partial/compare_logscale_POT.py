# -*- coding: utf-8 -*-
"""
===============================================================
Log-scale entropic partial wasserstein
===============================================================

This example shows that the log-scale computation can be used to stabilize the computation of the entropic partial wasserstein.
The idea was used in Sinkhorn algorithm (Sec. 4.4 in [15]), and is now implemented for entropic partial wasserstein in the entropic_partial_wasserstein_logscale function.

The function entropic_partial_wasserstein_logscale is generally slower than the original entropic_partial_wasserstein, but is more stable.

[15] Peyré, G., & Cuturi, M. (2018). Computational Optimal Transport .

"""

import ot
import torch
import numpy as np
import pdb
import time


def compute_OT(M, alpha, beta, epsilon):
    s1, s2 = M.shape[0], M.shape[1]
    assert s1 == s2
    unif_vec = ot.unif(s1)

    a, b = unif_vec / beta, unif_vec

    time_start = time.time()
    pi_1_np = ot.partial.entropic_partial_wasserstein(a, b, M, m=alpha, reg=epsilon)
    time_end1 = time.time()

    pi_2_np = ot.partial.entropic_partial_wasserstein_logscale(
        a, b, M, m=alpha, reg=epsilon
    )
    time_end2 = time.time()

    print(
        f"(Numpy) Original: any_nan = {np.any(np.isnan(pi_1_np))}, time = {time_end1 - time_start:.4f}, reg = {epsilon:.4f}"
    )
    print(
        f"(Numpy) Log scale: any_nan = {np.any(np.isnan(pi_2_np))}, time = {time_end2 - time_end1:.4f}, reg = {epsilon:.4f}"
    )
    if not np.any(np.isnan(pi_1_np)):
        print(f"Difference = {np.linalg.norm(pi_1_np - pi_2_np):.4f}")

    # compute the same thing using torch
    a = torch.tensor(a, dtype=torch.float)
    b = torch.tensor(b, dtype=torch.float)
    M = torch.tensor(M, dtype=torch.float)
    m = torch.tensor(alpha, dtype=torch.float)

    time_start = time.time()
    pi_1 = ot.partial.entropic_partial_wasserstein(a, b, M, m=m, reg=epsilon)
    time_end1 = time.time()
    pi_2 = ot.partial.entropic_partial_wasserstein_logscale(a, b, M, m=m, reg=epsilon)
    time_end2 = time.time()

    print(
        f"(Torch) Original: any_nan = {torch.any(torch.isnan(pi_1))}, time = {time_end1 - time_start:.4f}, reg = {epsilon:.4f}"
    )
    print(
        f"(Torch) Log scale: any_nan = {torch.any(torch.isnan(pi_2))}, time = {time_end2 - time_end1:.4f}, reg = {epsilon:.4f}"
    )
    if not torch.any(torch.isnan(pi_1)):
        print(f"Difference = {torch.norm(pi_1 - pi_2):.4f}")


beta = 0.35
alpha = 0.01

M_1 = np.loadtxt("../../data/entropic_partial_OT_cost.txt")

# both methods should give the same result when eps is large
epsilon = 10.0
compute_OT(M_1, alpha, beta, epsilon)

# Log scale method should give correct results even when eps is small, but original method will fail
epsilon = 0.1
compute_OT(M_1, alpha, beta, epsilon)


"""
Output:

(Numpy) Original: any_nan = False, time = 0.0050, reg = 10.0000
(Numpy) Log scale: any_nan = False, time = 0.0060, reg = 10.0000
Difference = 0.0000
(Torch) Original: any_nan = False, time = 1.0847, reg = 10.0000
(Torch) Log scale: any_nan = False, time = 0.0090, reg = 10.0000
Difference = 0.0000
G:\Mycode\Pull_tmp\POT\ot\partial.py:715: RuntimeWarning: invalid value encountered in divide
  q1 = q1 * Kprev / K1
G:\Mycode\Pull_tmp\POT\ot\partial.py:719: RuntimeWarning: invalid value encountered in divide
  q2 = q2 * K1prev / K2
G:\Mycode\Pull_tmp\POT\ot\partial.py:723: RuntimeWarning: invalid value encountered in divide
  q3 = q3 * K2prev / K
Warning: numerical errors at iteration 1
(Numpy) Original: any_nan = True, time = 0.0033, reg = 0.1000
(Numpy) Log scale: any_nan = False, time = 0.0060, reg = 0.1000
Warning: numerical errors at iteration 0
(Torch) Original: any_nan = True, time = 0.0010, reg = 0.1000  
(Torch) Log scale: any_nan = False, time = 0.0160, reg = 0.1000
"""
