"""Tests for main module ot """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import pytest
import numpy as np
import sys
import ot

try:  # test if torch is installed
    if not sys.platform.endswith('win32'):  # and not windows
        import ot.torch
        import torch
        nogo = False

        lst_types = [torch.float32, torch.float64]

        lst_devices = ['cpu']
        if torch.cuda.is_available():
            lst_devices.append('cuda')
    else:
        nogo = True

except BaseException:
    nogo = True


@pytest.mark.skipif(nogo, reason="Missing pytorch")
def test_dist():

    n = 200

    lst_metrics = ['sqeuclidean', 'euclidean', 'cityblock', 0, 0.5, 1, 2, 5]

    for dtype in lst_types:
        for device in lst_devices:

            x = torch.randn(n, 2, dtype=dtype, device=device)
            y = torch.randn(n, 2, dtype=dtype, device=device)

            for metric in lst_metrics:

                M = ot.torch.dist(x, y, metric)

                assert M.shape[0] == n
                assert M.shape[1] == n


@pytest.mark.skipif(nogo, reason="Missing pytorch")
def test_ot_loss():

    n = 10

    lst_metrics = ['sqeuclidean', 'euclidean', 'cityblock', 0, 0.5, 1, 2, 5]

    for dtype in lst_types:
        for device in lst_devices:

            x = torch.randn(n, 2, dtype=dtype, device=device)
            y = torch.randn(n, 2, dtype=dtype, device=device)

            a = ot.torch.unif(n, dtype=dtype, device=device)
            b = ot.torch.unif(n, dtype=dtype, device=device)

            for metric in lst_metrics:

                M = ot.torch.dist(x, y, metric)
                loss = ot.torch.ot_loss(a, b, M)

                assert float(loss) >= 0


@pytest.mark.skipif(nogo, reason="Missing pytorch")
def test_ot_solve():

    n = 10

    lst_metrics = ['sqeuclidean', 'euclidean', 'cityblock', 0, 0.5, 1, 2, 5]

    for dtype in lst_types:
        for device in lst_devices:

            x = torch.randn(n, 2, dtype=dtype, device=device)
            y = torch.randn(n, 2, dtype=dtype, device=device)

            a = ot.torch.unif(n, dtype=dtype, device=device)
            b = ot.torch.unif(n, dtype=dtype, device=device)

            for metric in lst_metrics:

                M = ot.torch.dist(x, y, metric)
                G = ot.torch.ot_solve(a, b, M)

                np.testing.assert_allclose(ot.unif(n), G.sum(1).cpu().numpy())
                np.testing.assert_allclose(ot.unif(n), G.sum(0).cpu().numpy())  # cf convergence sinkhorn
