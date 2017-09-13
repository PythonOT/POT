"""Tests for module gpu for gpu acceleration """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import time

import numpy as np
import pytest
import ot

try:  # test if cupy is installed
    import ot.gpu
    nogpu = False
except ImportError:
    nogpu = True


@pytest.mark.skipif(nogpu, reason="No GPU available")
def test_gpu_sinkhorn():

    rng = np.random.RandomState(0)

    def describe_res(r):
        print("min:{:.3E}, max::{:.3E}, mean::{:.3E}, std::{:.3E}".format(
            np.min(r), np.max(r), np.mean(r), np.std(r)))

    for n_samples in [50, 100, 500, 1000]:
        print(n_samples)
        a = rng.rand(n_samples // 4, 100)
        b = rng.rand(n_samples, 100)
        time1 = time.time()
        transport = ot.da.SinkhornTransport()
        transport.fit(Xs=a, Xt=b)
        G1 = transport.coupling_
        time2 = time.time()
        transport = ot.gpu.da.SinkhornTransport()
        transport.fit(Xs=a, Xt=b)
        G2 = transport.coupling_
        time3 = time.time()
        print("Normal sinkhorn, time: {:6.2f} sec ".format(time2 - time1))
        describe_res(G1)
        print("   GPU sinkhorn, time: {:6.2f} sec ".format(time3 - time2))
        describe_res(G2)

        np.testing.assert_allclose(G1, G2, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(nogpu, reason="No GPU available")
def test_gpu_sinkhorn_lpl1():

    rng = np.random.RandomState(0)

    def describe_res(r):
        print("min:{:.3E}, max:{:.3E}, mean:{:.3E}, std:{:.3E}"
              .format(np.min(r), np.max(r), np.mean(r), np.std(r)))

    for n_samples in [50, 100, 500]:
        print(n_samples)
        a = rng.rand(n_samples // 4, 100)
        labels_a = np.random.randint(10, size=(n_samples // 4))
        b = rng.rand(n_samples, 100)
        time1 = time.time()
        transport = ot.da.SinkhornLpl1Transport()
        transport.fit(Xs=a, ys=labels_a, Xt=b)
        G1 = transport.coupling_
        time2 = time.time()
        transport = ot.gpu.da.SinkhornLpl1Transport()
        transport.fit(Xs=a, ys=labels_a, Xt=b)
        G2 = transport.coupling_
        time3 = time.time()
        print("Normal sinkhorn lpl1, time: {:6.2f} sec ".format(
            time2 - time1))
        describe_res(G1)
        print("   GPU sinkhorn lpl1, time: {:6.2f} sec ".format(
            time3 - time2))
        describe_res(G2)

        np.testing.assert_allclose(G1, G2, rtol=1e-5, atol=1e-5)
