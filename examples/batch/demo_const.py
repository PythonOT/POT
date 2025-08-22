# -*- coding: utf-8 -*-
"""
==================
Batch parallel linear OT
==================

Showcase the importance of correctly setting the "symmetric" boolean.

"""

from ot.batch._linear import *
from ot.batch._quadratic import *
import numpy as np
from time import perf_counter

np.random.seed(0)

b = 64
n = 64
d_nodes = 4
d_edges = 2

C1 = np.random.randn(b, n, n, d_edges).astype("float32")
C2 = np.random.randn(b, n, n, d_edges).astype("float32")

p = np.ones((b, n), dtype=np.float32) / n
q = np.ones((b, n), dtype=np.float32) / n


def compute_loss(T, recompute_const=False):
    metric = QuadraticEuclidean()
    L = metric.cost_tensor(p, q, C1, C2, symmetric=True)
    LT = tensor_product(L, T, recompute_const=recompute_const, symmetric=True)
    loss = (LT * T).sum((1, 2))
    return loss


# Not setting symmetric=True leads to invalid loss computations

print("Check loss at random coupling - recompute_const=False")
T = np.random.rand(b, n, n).astype("float32")
loss = compute_loss(T, recompute_const=False)
is_pos = (loss >= -1e-6).all()
print("All losses are positive: ", is_pos)

print("Check loss at random coupling - recompute_const=True")
T = np.random.rand(b, n, n).astype("float32")
loss = compute_loss(T, recompute_const=True)
is_pos = (loss >= -1e-6).all()
print("All losses are positive: ", is_pos)
