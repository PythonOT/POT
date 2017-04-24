import ot
import numpy as np
import time
import ot.gpu


def describeRes(r):
    print("min:{:.3E}, max:{:.3E}, mean:{:.3E}, std:{:.3E}"
          .format(np.min(r), np.max(r), np.mean(r), np.std(r)))

for n in [5000, 10000, 15000, 20000]:
    print(n)
    a = np.random.rand(n // 4, 100)
    labels_a = np.random.randint(10, size=(n // 4))
    b = np.random.rand(n, 100)
    time1 = time.time()
    transport = ot.da.OTDA_lpl1()
    transport.fit(a, labels_a, b)
    G1 = transport.G
    time2 = time.time()
    transport = ot.gpu.da.OTDA_lpl1()
    transport.fit(a, labels_a, b)
    G2 = transport.G
    time3 = time.time()
    print("Normal sinkhorn lpl1, time: {:6.2f} sec ".format(time2 - time1))
    describeRes(G1)
    print("   GPU sinkhorn lpl1, time: {:6.2f} sec ".format(time3 - time2))
    describeRes(G2)
