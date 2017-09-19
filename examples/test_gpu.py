# -*- coding: utf-8 -*-

import time
import numpy as np
import ot
import cupy as cp

a = np.random.rand(10000, 100)
b = np.random.rand(10000, 100)

ot1 = ot.da.SinkhornTransport(gpu=False)
ot2 = ot.da.SinkhornTransport(gpu=True)

time1 = time.time()
ot1.fit(Xs=a, Xt=b)
g1 = ot1.coupling_
time2 = time.time()
ot2.fit(Xs=a, Xt=b)
g2 = ot2.coupling_
time3 = time.time()


print("Normal, time: {:6.2f} sec ".format(time2 - time1))
print("   GPU, time: {:6.2f} sec ".format(time3 - time2))
np.testing.assert_allclose(g1, cp.asnumpy(g2), rtol=1e-5, atol=1e-5)
