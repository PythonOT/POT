"""Tests for main module ot """

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import pytest
import numpy as np
import ot

try:  # test if torch is installed

    import ot.torch
    import torch

    nogo = False

    lst_types = [torch.float32, torch.float64]

    lst_devices = ['cpu']
    if torch.cuda.is_available():
        lst_devices.append('cuda')


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
def test_proj_simplex():
    n = 10

    for dtype in lst_types:
        for device in lst_devices:
            x = torch.randn(n, dtype=dtype, device=device)

            xp = ot.torch.proj_simplex(x)

            assert torch.all(xp >= 0)
            assert torch.allclose(xp.sum(), torch.tensor(1.0, dtype=dtype, device=device))

            x = torch.randn(n, 3, dtype=dtype, device=device)

            xp = ot.torch.proj_simplex(x)

            assert torch.all(xp >= 0)
            assert torch.allclose(xp.sum(0), torch.ones(3, dtype=dtype, device=device))


@pytest.mark.skipif(nogo, reason="Missing pytorch")
def test_ot_loss_grad():
    n = 10

    lst_metrics = ['sqeuclidean', 'euclidean', 'cityblock', 0, 0.5, 1, 2, 5]

    for dtype in lst_types:
        for device in lst_devices:

            for metric in lst_metrics:
                x = torch.randn(n, 2, dtype=dtype, device=device, requires_grad=True)
                y = torch.randn(n, 2, dtype=dtype, device=device, requires_grad=True)

                a = ot.torch.unif(n, dtype=dtype, device=device, requires_grad=True)
                b = ot.torch.unif(n, dtype=dtype, device=device, requires_grad=True)

                M = ot.torch.dist(x, y, metric)
                loss = ot.torch.ot_loss(a, b, M)

                loss.backward()

                assert x.grad is not None
                assert y.grad is not None
                assert a.grad is not None
                assert b.grad is not None

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


@pytest.mark.skipif(nogo, reason="Missing pytorch")
def test_emd1d():
    torch.random.manual_seed(42)
    n = 10
    k = 5
    ps = [1, 2, 3]

    for dtype in lst_types:
        for device in lst_devices:

            x = torch.randn(k, n, dtype=dtype, device=device)
            y = torch.randn(k, n, dtype=dtype, device=device)

            a = ot.torch.unif(n, dtype=dtype, device=device)
            b = ot.torch.unif(n, dtype=dtype, device=device)

            for p in ps:
                cpp_cost = np.zeros(k)
                torch_cost = ot.torch.lp.emd_1d(a, b, x, y, p)
                for i in range(k):
                    cpp_cost[i] = ot.lp.emd2_1d(x[i].cpu().numpy(), y[i].cpu().numpy(), (a / a.sum()).cpu().numpy(),
                                                (b / b.sum()).cpu().numpy(), "minkowski", p=p)

                np.testing.assert_allclose(cpp_cost, torch_cost.cpu().numpy(), atol=1e-7)
