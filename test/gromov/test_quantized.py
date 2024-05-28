"""Tests for gromov._quantized.py """

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

import numpy as np
import pytest

import ot

from ot.gromov._quantized import (
    networkx_import, sklearn_import)


def test_quantized_gw(nx):
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
            pairs_part_rep += [('louvain', 'random'), ('fluid', 'pagerank')]
        if sklearn_import:
            pairs_part_rep += [('spectral', 'random')]

        count_mode = 0

        for part_method, rep_method in pairs_part_rep:
            print(part_method, rep_method)
            log_ = log_tests[count_mode]
            count_mode += 1

            res = ot.gromov.quantized_fused_gromov_wasserstein(
                C1, C2, npart1, npart2, p, None, C1, None, part_method=part_method,
                rep_method=rep_method, log=log_)

            resb = ot.gromov.quantized_fused_gromov_wasserstein(
                C1b, C2b, npart1, npart2, None, qb, None, C2b, part_method=part_method,
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


def test_quantized_fgw(nx):
    n_samples = 30  # nb samples

    rng = np.random.RandomState(0)
    C1 = rng.uniform(low=0., high=10, size=(n_samples, n_samples))
    C1 = (C1 + C1.T) / 2.

    F1 = rng.uniform(low=0., high=10, size=(n_samples, 1))

    C2 = rng.uniform(low=10., high=20., size=(n_samples, n_samples))
    C2 = (C2 + C2.T) / 2.

    F2 = rng.uniform(low=0., high=10, size=(n_samples, 1))

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    npart1 = 2
    npart2 = 3

    C1b, C2b, F1b, F2b, pb, qb = nx.from_numpy(C1, C2, F1, F2, p, q)

    log_tests = [True, False, False, True, True, False]

    pairs_part_rep = []
    if networkx_import:
        pairs_part_rep += [('louvain_fused', 'pagerank'),
                           ('louvain', 'pagerank_fused'),
                           ('fluid_fused', 'pagerank_fused')]
    if sklearn_import:
        pairs_part_rep += [('spectral_fused', 'random')]

    pairs_part_rep += [('random', 'random')]
    count_mode = 0

    alpha = 0.5

    for part_method, rep_method in pairs_part_rep:
        log_ = log_tests[count_mode]
        count_mode += 1

        res = ot.gromov.quantized_fused_gromov_wasserstein(
            C1, C2, npart1, npart2, p, None, C1, None, F1, F2, alpha,
            part_method, rep_method, log_)

        resb = ot.gromov.quantized_fused_gromov_wasserstein(
            C1b, C2b, npart1, npart2, None, qb, None, C2b, F1b, F2b, alpha,
            part_method, rep_method, log_)

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
    DF1b = ot.dist(F1b, F1b)
    DF2b = ot.dist(F2b, F2b)
    C1b_new = alpha * C1b + (1 - alpha) * DF1b
    C2b_new = alpha * C2b + (1 - alpha) * DF2b

    part1b = ot.gromov.get_graph_partition(
        C1b_new, npart1, part_method=pairs_part_rep[-1][0], random_state=0)
    part2b = ot.gromov._quantized.get_graph_partition(
        C2b_new, npart2, part_method=pairs_part_rep[-1][0], random_state=0)

    rep_indices1b = ot.gromov.get_graph_representants(
        C1b, part1b, rep_method=pairs_part_rep[-1][1], random_state=0)
    rep_indices2b = ot.gromov.get_graph_representants(
        C2b, part2b, rep_method=pairs_part_rep[-1][1], random_state=0)

    CR1b, list_R1b, list_p1b, FR1b = ot.gromov.format_partitioned_graph(
        C1b, pb, part1b, rep_indices1b, F1b, DF1b, alpha)
    CR2b, list_R2b, list_p2b, FR2b = ot.gromov.format_partitioned_graph(
        C2b, qb, part2b, rep_indices2b, F2b, DF2b, alpha)

    MRb = ot.dist(FR1b, FR2b)

    T_globalb, Ts_localb, _ = ot.gromov.quantized_fused_gromov_wasserstein_partitioned(
        CR1b, CR2b, list_R1b, list_R2b, list_p1b, list_p2b, MRb, alpha, build_OT=False)

    T_globalb = nx.to_numpy(T_globalb)
    np.testing.assert_allclose(T_global, T_globalb, atol=1e-06)

    for key in Ts_localb.keys():
        T_localb = nx.to_numpy(Ts_localb[key])
        np.testing.assert_allclose(Ts_local[key], T_localb, atol=1e-06)

    # tests for edge cases of the graph partitioning
    for method in ['unknown_method', 'GW', 'FGW']:
        with pytest.raises(ValueError):
            ot.gromov.get_graph_partition(
                C1b, npart1, part_method=method, random_state=0)

    # tests for edge cases of the representant selection
    with pytest.raises(ValueError):
        ot.gromov.get_graph_representants(
            C1b, part1b, rep_method='unknown_method', random_state=0)

    # tests for edge cases of the format_partitioned_graph function
    with pytest.raises(ValueError):
        CR1b, list_R1b, list_p1b, FR1b = ot.gromov.format_partitioned_graph(
            C1b, pb, part1b, rep_indices1b, F1b, None, alpha)

    # Tests in qFGW solvers
    # for non admissible values of alpha
    with pytest.raises(ValueError):
        ot.gromov.quantized_fused_gromov_wasserstein_partitioned(
            CR1b, CR2b, list_R1b, list_R2b, list_p1b, list_p2b, MRb, 0, build_OT=False)

    # for non-consistent feature information provided
    with pytest.raises(ValueError):
        ot.gromov.quantized_fused_gromov_wasserstein(
            C1, C2, npart1, npart2, p, q, None, None, F1, None, 0.5,
            'spectral_fused', 'random', log_)


@pytest.skip_backend("jax", reason="test very slow with jax backend")
def test_quantized_gw_samples(nx):
    n_samples_1 = 20  # nb samples
    n_samples_2 = 30  # nb samples

    rng = np.random.RandomState(0)
    X1 = rng.uniform(low=0., high=10, size=(n_samples_1, 2))
    X2 = rng.uniform(low=0., high=10, size=(n_samples_2, 4))

    p = ot.unif(n_samples_1)
    q = ot.unif(n_samples_2)

    npart1 = 2
    npart2 = 3

    X1b, X2b, pb, qb = nx.from_numpy(X1, X2, p, q)

    log_tests = [True, False, True]
    methods = ['random']
    if sklearn_import:
        methods += ['kmeans']

    count_mode = 0
    alpha = 1.

    for method in methods:
        log_ = log_tests[count_mode]
        count_mode += 1

        res = ot.gromov.quantized_fused_gromov_wasserstein_samples(
            X1, X2, npart1, npart2, p, None, None, None, alpha, method, log_)

        resb = ot.gromov.quantized_fused_gromov_wasserstein_samples(
            X1b, X2b, npart1, npart2, None, qb, None, None, alpha, method, log_)

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


@pytest.skip_backend("jax", reason="test very slow with jax backend")
def test_quantized_fgw_samples(nx):
    n_samples_1 = 20  # nb samples
    n_samples_2 = 30  # nb samples

    rng = np.random.RandomState(0)
    X1 = rng.uniform(low=0., high=10, size=(n_samples_1, 2))
    X2 = rng.uniform(low=0., high=10, size=(n_samples_2, 4))

    F1 = rng.uniform(low=0., high=10, size=(n_samples_1, 3))
    F2 = rng.uniform(low=0., high=10, size=(n_samples_2, 3))

    p = ot.unif(n_samples_1)
    q = ot.unif(n_samples_2)

    npart1 = 2
    npart2 = 3

    X1b, X2b, F1b, F2b, pb, qb = nx.from_numpy(X1, X2, F1, F2, p, q)

    methods = []
    if sklearn_import:
        methods += ['kmeans', 'kmeans_fused']
    methods += ['random']

    alpha = 0.5

    for npart1 in [1, n_samples_1 + 1, 2]:
        log_tests = [True, False, True]
        count_mode = 0

        for method in methods:
            log_ = log_tests[count_mode]
            count_mode += 1

            res = ot.gromov.quantized_fused_gromov_wasserstein_samples(
                X1, X2, npart1, npart2, p, None, F1, F2, alpha, method, log_)

            resb = ot.gromov.quantized_fused_gromov_wasserstein_samples(
                X1b, X2b, npart1, npart2, None, qb, F1b, F2b, alpha, method, log_)

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
    part1b, rep_indices1 = ot.gromov.get_partition_and_representants_samples(
        X1b, npart1, method=method, random_state=0)
    part2b, rep_indices2 = ot.gromov.get_partition_and_representants_samples(
        X2b, npart2, method=method, random_state=0)

    CR1b, list_R1b, list_p1b, FR1b = ot.gromov.format_partitioned_samples(
        X1b, pb, part1b, rep_indices1, F1b, alpha)
    CR2b, list_R2b, list_p2b, FR2b = ot.gromov.format_partitioned_samples(
        X2b, qb, part2b, rep_indices2, F2b, alpha)

    MRb = ot.dist(FR1b, FR2b)

    T_globalb, Ts_localb, _ = ot.gromov.quantized_fused_gromov_wasserstein_partitioned(
        CR1b, CR2b, list_R1b, list_R2b, list_p1b, list_p2b, MRb, alpha, build_OT=False)

    T_globalb = nx.to_numpy(T_globalb)
    np.testing.assert_allclose(T_global, T_globalb, atol=1e-06)

    for key in Ts_localb.keys():
        T_localb = nx.to_numpy(Ts_localb[key])
        np.testing.assert_allclose(Ts_local[key], T_localb, atol=1e-06)
