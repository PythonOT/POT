# -*- coding: utf-8 -*-
"""
Domain adaptation with optimal transport and GPU
"""

import numpy as np
from ..utils import unif
from ..da import OTDA
from .bregman import sinkhornGPU


def pairwiseEuclideanGPU(a, b, returnAsGPU=False, squared=False, cudamat=None):
    # a is shape (n, f) and b shape (m, f). Return matrix c of shape (n, m).
    # First compute in c_GPU the squared euclidean distance. And return its
    # square root. At each cell [i,j] of c, we want to have
    # sum{k in range(f)} ( (a[i,k] - b[j,k])^2 ). We know that
    # (a-b)^2 = a^2 -2ab +b^2. Thus we want to have in each cell of c:
    # sum{k in range(f)} ( a[i,k]^2 -2a[i,k]b[j,k] +b[j,k]^2).

    a_GPU = cudamat.CUDAMatrix(a)
    b_GPU = cudamat.CUDAMatrix(b)

    # Multiply a by b transpose to obtain in each cell [i,j] of c the
    # value sum{k in range(f)} ( a[i,k]b[j,k] )
    c_GPU = cudamat.dot(a_GPU, b_GPU.transpose())
    # multiply by -2 to have sum{k in range(f)} ( -2a[i,k]b[j,k] )
    c_GPU.mult(-2)

    # Compute the vectors of the sum of squared elements.
    a_GPU = cudamat.pow(a_GPU, 2).sum(axis=1)
    b_GPU = cudamat.pow(b_GPU, 2).sum(axis=1)

    # Add the vectors in each columns (respectivly rows) of c.
    # sum{k in range(f)} ( a[i,k]^2 -2a[i,k]b[j,k] )
    c_GPU.add_col_vec(a_GPU)
    # sum{k in range(f)} ( a[i,k]^2 -2a[i,k]b[j,k] +b[j,k]^2)
    c_GPU.add_row_vec(b_GPU.transpose())

    if not squared:
        c_GPU = cudamat.sqrt(c_GPU)

    if returnAsGPU:
        return c_GPU
    else:
        return c_GPU.asarray()


class OTDA_sinkhorn_GPU(OTDA):
    def fit(self, xs, xt, reg=1, ws=None, wt=None, norm=None):
        import cudamat
        cudamat.init()
        xs = np.asarray(xs, dtype=np.float64)
        xt = np.asarray(xt, dtype=np.float64)

        self.xs = xs
        self.xt = xt

        if wt is None:
            wt = unif(xt.shape[0])
        if ws is None:
            ws = unif(xs.shape[0])

        self.ws = ws
        self.wt = wt

        self.M_GPU = pairwiseEuclideanGPU(xs, xt, returnAsGPU=True,
                                          squared=True, cudamat=cudamat)

        if norm == "median":
            self.M_GPU.divide(float(np.median(self.M_GPU.asarray())))
        elif norm == "max":
            self.M_GPU.divide(float(np.max(self.M_GPU.asarray())))
        elif norm == "log":
            M = np.log(1 + self.M_GPU.asarray())
            self.M_GPU = cudamat.CUDAMatrix(M)
        elif norm == "loglog":
            M = np.log(1 + np.log(1 + self.M_GPU.asarray()))
            self.M_GPU = cudamat.CUDAMatrix(M)

        self.G = sinkhornGPU(ws, wt, self.M_GPU, reg, cudamat=cudamat)
        self.computed = True
