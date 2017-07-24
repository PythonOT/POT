import ot
import numpy as np
import time
import pytest

try:  # test if cudamat installed
    import ot.gpu
    nogpu = False
except ImportError:
    nogpu = True


@pytest.mark.skipif(nogpu, reason="No GPU available")
def test_gpu_sinkhorn():

    np.random.seed(0)

    def describeRes(r):
        print("min:{:.3E}, max::{:.3E}, mean::{:.3E}, std::{:.3E}".format(
            np.min(r), np.max(r), np.mean(r), np.std(r)))

    for n in [50, 100, 500, 1000]:
        print(n)
        a = np.random.rand(n // 4, 100)
        b = np.random.rand(n, 100)
        time1 = time.time()
        transport = ot.da.OTDA_sinkhorn()
        transport.fit(a, b)
        G1 = transport.G
        time2 = time.time()
        transport = ot.gpu.da.OTDA_sinkhorn()
        transport.fit(a, b)
        G2 = transport.G
        time3 = time.time()
        print("Normal sinkhorn, time: {:6.2f} sec ".format(time2 - time1))
        describeRes(G1)
        print("   GPU sinkhorn, time: {:6.2f} sec ".format(time3 - time2))
        describeRes(G2)

        assert np.allclose(G1, G2, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(nogpu, reason="No GPU available")
def test_gpu_sinkhorn_lpl1():
    np.random.seed(0)

    def describeRes(r):
        print("min:{:.3E}, max:{:.3E}, mean:{:.3E}, std:{:.3E}"
              .format(np.min(r), np.max(r), np.mean(r), np.std(r)))

    for n in [50, 100, 500, 1000]:
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
        print("Normal sinkhorn lpl1, time: {:6.2f} sec ".format(
            time2 - time1))
        describeRes(G1)
        print("   GPU sinkhorn lpl1, time: {:6.2f} sec ".format(
            time3 - time2))
        describeRes(G2)

        assert np.allclose(G1, G2, rtol=1e-5, atol=1e-5)
