# -*- coding: utf-8 -*-

import time

import numpy as np
import cupy as cp
from scipy.spatial.distance import cdist
import ot


def benchDistance(a, b):
    # First compare computation time for computing pairwise euclidean matrix
    time1 = time.time()
    M1 = cdist(a, b, metric="sqeuclidean")
    time2 = time.time()
    M2 = ot.utils.pairwiseEuclidean(a, b, gpu=False, squared=True)
    time3 = time.time()
    M3 = ot.utils.pairwiseEuclidean(a, b, gpu=True, squared=True)
    time4 = time.time()

    np.testing.assert_allclose(M1, M2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(M2, cp.asnumpy(M3), rtol=1e-5, atol=1e-5)
    print("        scipy's cdist, time: {:6.2f} sec ".format(time2 - time1))
    print("pairwiseEuclidean CPU, time: {:6.2f} sec ".format(time3 - time2))
    print("pairwiseEuclidean GPU, time: {:6.2f} sec ".format(time4 - time3))


def benchSinkhorn(a, b):
    # Then compare computation time for computing optimal sinkhorn coupling
    ot1 = ot.da.SinkhornTransport(gpu=False)
    ot2 = ot.da.SinkhornTransport(gpu=True)

    time1 = time.time()
    ot1.fit(Xs=a, Xt=b)
    g1 = ot1.coupling_
    time2 = time.time()
    ot2.fit(Xs=a, Xt=b)
    g2 = ot2.coupling_
    time3 = time.time()

    print("Sinkhorn CPU, time: {:6.2f} sec ".format(time2 - time1))
    print("Sinkhorn GPU, time: {:6.2f} sec ".format(time3 - time2))
    np.testing.assert_allclose(g1, cp.asnumpy(g2), rtol=1e-5, atol=1e-5)


for tp in [np.float32, np.float64]:
    print("Using " + str(tp))
    a = np.random.rand(10000, 100).astype(tp)
    b = np.random.rand(10000, 100).astype(tp)
    benchDistance(a, b)
    benchSinkhorn(a, b)
