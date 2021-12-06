# /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import ot
from .benchmark import (
    setup_backends,
    exec_bench,
    convert_to_html_table
)


def setup(n_samples):
    rng = np.random.RandomState(123456789)
    a = rng.rand(n_samples // 4, 100)
    b = rng.rand(n_samples, 100)

    wa = ot.unif(n_samples // 4)
    wb = ot.unif(n_samples)

    M = ot.dist(a.copy(), b.copy())
    return wa, wb, M


if __name__ == "__main__":
    setup_backends()
    results = exec_bench(
        setup=setup,
        tested_function=lambda *args: ot.bregman.sinkhorn(*args, reg=1, stopThr=1e-7),
        param_list=[50, 100, 500, 1000], #, 2000, 5000, 10000],
        n_runs=10
    )

    print(convert_to_html_table(results, param_name="Sample size"))
