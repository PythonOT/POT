import ot
import numpy as np


def test_omp_emd2():
    # test emd2 and emd2 with openmp for simple identity
    n = 1000
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    w = ot.emd2(u, u, M)
    w2 = ot.emd2(u, u, M, numThreads=8)


test_omp_emd2()
