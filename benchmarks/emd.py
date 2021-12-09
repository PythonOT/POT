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
    rng = np.random.RandomState(789465132)
    x = rng.randn(n_samples, 2)
    y = rng.randn(n_samples, 2)

    a = ot.utils.unif(n_samples)
    M = ot.dist(x, y)
    return a, M


if __name__ == "__main__":
    n_runs = 100
    warmup_runs = 10
    param_list = [50, 100, 500, 1000, 2000, 5000]

    setup_backends()
    results = exec_bench(
        setup=setup,
        tested_function=lambda a, M: ot.emd(a, a, M),
        param_list=param_list,
        n_runs=n_runs,
        warmup_runs=warmup_runs
    )
    print(convert_to_html_table(
        results, 
        param_name="Sample size",
        main_title=f"EMD - Averaged on {n_runs} runs"
    ))
