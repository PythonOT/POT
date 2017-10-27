"""Tests for module gpu for gpu acceleration """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import time
import pytest

from ot.da import OTDA_sinkhorn, OTDA_lpl1

try:  # test if cudamat installed
    from ot.gpu.da import OTDA_sinkhorn as OTDA_sinkhorn_gpu
    from ot.gpu.da import OTDA_lpl1 as OTDA_lpl1_gpu
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
        transport = OTDA_sinkhorn()
        transport.fit(a, b)
        G1 = transport.G
        time2 = time.time()
        transport = OTDA_sinkhorn_gpu()
        transport.fit(a, b)
        G2 = transport.G
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
        transport = OTDA_lpl1()
        transport.fit(a, labels_a, b)
        G1 = transport.G
        time2 = time.time()
        transport = OTDA_lpl1_gpu()
        transport.fit(a, labels_a, b)
        G2 = transport.G
        time3 = time.time()
        print("Normal sinkhorn lpl1, time: {:6.2f} sec ".format(
            time2 - time1))
        describe_res(G1)
        print("   GPU sinkhorn lpl1, time: {:6.2f} sec ".format(
            time3 - time2))
        describe_res(G2)

        np.testing.assert_allclose(G1, G2, rtol=1e-5, atol=1e-5)
