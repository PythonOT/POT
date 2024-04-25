"""Tests for gromov._quantized.py """

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

import numpy as np
import pytest
import warnings

import ot
from ot.backend import NumpyBackend
from ot.backend import torch, tf

from ot.gromov._quantized import (
    networkx_import, sklearn_import)


def test_quantized_gromov(nx):
    n_samples = 30  # nb samples

    rng = np.random.RandomState(0)
    C1 = rng.uniform(low=0., high=10, size=(n_samples, n_samples))
    C1 = (C1 + C1.T) / 2.

    C2 = rng.uniform(low=10., high=20., size=(n_samples, n_samples))
    C2 = (C2 + C2.T) / 2.

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    npart2 = 3

    C1b, C2b, pb, qb = nx.from_numpy(C1, C2, p, q)

    for npart1 in [1, n_samples + 1, 2]:
        log_tests = [True, False, False, True, True, False]

        pairs_part_rep = [('random', 'random')]
        if networkx_import:
            pairs_part_rep += [('louvain', 'random'), ('fluid', 'pagerank'),
                               ('spectral', 'random')]
        if sklearn_import:
            pairs_part_rep += [('kmeans', 'kmeans')]

        count_mode = 0

        for part_method, rep_method in pairs_part_rep:
            print(part_method, rep_method)
            log_ = log_tests[count_mode]
            count_mode += 1

            res = ot.gromov.quantized_gromov_wasserstein(
                C1, C2, npart1, npart2, C1, None, p=p, q=None, part_method=part_method,
                rep_method=rep_method, log=log_)

            resb = ot.gromov.quantized_gromov_wasserstein(
                C1b, C2b, npart1, npart2, None, C2b, p=None, q=qb, part_method=part_method,
                rep_method=rep_method, log=log_)

            if log_:
                T_global, Ts_local, T, log = res
                T_globalb, Ts_localb, Tb, logb = resb
            else:
                T_global, Ts_local, T = res
                T_globalb, Ts_localb, Tb = resb

            Tb = nx.to_numpy(Tb)
            # check constraints
            np.testing.assert_allclose(T, Tb, atol=1e-06)
            np.testing.assert_allclose(
                p, Tb.sum(1), atol=1e-06)  # cf convergence gromov
            np.testing.assert_allclose(
                q, Tb.sum(0), atol=1e-06)  # cf convergence gromov

            if log_:
                for key in log.keys():
                    # The inner test T_global[i, j] != 0. can lead to different
                    # computation of 1D OT computations between partition depending
                    # on the different float errors across backend
                    if key in logb.keys():
                        np.testing.assert_allclose(log[key], logb[key], atol=1e-06)

    # complementary tests for utils functions
    part1b = ot.gromov.get_graph_partition(
        C1b, npart1, part_method=pairs_part_rep[-1][0], random_state=0)
    part2b = ot.gromov._quantized.get_graph_partition(
        C2b, npart2, part_method=pairs_part_rep[-1][0], random_state=0)

    rep_indices1b = ot.gromov.get_graph_representants(
        C1b, part1b, rep_method=pairs_part_rep[-1][1], random_state=0)
    rep_indices2b = ot.gromov.get_graph_representants(
        C2b, part2b, rep_method=pairs_part_rep[-1][1], random_state=0)

    CR1b, list_R1b, list_p1b = ot.gromov.format_partitioned_graph(
        C1b, pb, part1b, rep_indices1b)
    CR2b, list_R2b, list_p2b = ot.gromov.format_partitioned_graph(
        C2b, qb, part2b, rep_indices2b)

    T_globalb, Ts_localb, _ = ot.gromov.quantized_gromov_wasserstein_partitioned(
        CR1b, CR2b, list_R1b, list_R2b, list_p1b, list_p2b, build_OT=False)

    T_globalb = nx.to_numpy(T_globalb)
    np.testing.assert_allclose(T_global, T_globalb, atol=1e-06)

    for key in Ts_localb.keys():
        T_localb = nx.to_numpy(Ts_localb[key])
        np.testing.assert_allclose(Ts_local[key], T_localb, atol=1e-06)

    # tests for edge cases
    method = 'unknown_method'
    with pytest.raises(ValueError):
        ot.gromov.get_graph_partition(
            C1b, npart1, part_method=method, random_state=0)
    with pytest.raises(ValueError):
        ot.gromov.get_graph_representants(
            C1b, part1b, rep_method=method, random_state=0)
