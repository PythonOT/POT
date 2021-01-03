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

    lst_types = [torch.float32, torch.float64]

    lst_devices = ['cpu']
    if torch.cuda.is_available():
        lst_devices.append('cuda')


except BaseException:
    pytest.skip("Missing pytorch", allow_module_level=True)


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


@pytest.mark.parametrize("random_weights", [True, False])
@pytest.mark.parametrize("batch_size", [0, 2, 10])
def test_ot_loss_1d(random_weights, batch_size):
    torch.random.manual_seed(42)
    n = 300
    m = 200
    k = 5
    ps = [1, 2, 3]

    for dtype in lst_types:
        for device in lst_devices:
            if batch_size:
                x = torch.randn(n, batch_size, k, dtype=dtype, device=device)
                y = torch.randn(m, batch_size, k, dtype=dtype, device=device)
            else:
                x = torch.randn(n, k, dtype=dtype, device=device)
                y = torch.randn(m, k, dtype=dtype, device=device)
            if random_weights:
                if batch_size:
                    a = torch.rand(n, batch_size, dtype=dtype, device=device)
                    b = torch.rand(m, batch_size, dtype=dtype, device=device)
                else:
                    a = torch.rand(n, dtype=dtype, device=device)
                    b = torch.rand(m, dtype=dtype, device=device)
                a = a / torch.sum(a, 0, keepdim=True)
                b = b / torch.sum(b, 0, keepdim=True)
                np_a = a.cpu().numpy()
                np_b = b.cpu().numpy()
            else:
                a = b = np_a = np_b = None

            for p in ps:
                same_dist_cost = ot.torch.lp.ot_loss_1d(x, x, a, a, p)
                assert np.allclose(same_dist_cost.cpu().numpy(), 0., atol=1e-5)
                torch_cost = ot.torch.lp.ot_loss_1d(x, y, a, b, p)

                if batch_size:
                    cpu_cost = np.zeros((batch_size, k))
                else:
                    cpu_cost = np.zeros(k)

                for i in range(k):
                    if batch_size:
                        for batch_num in range(batch_size):
                            cpu_cost[batch_num, i] = ot.lp.emd2_1d(x[:, batch_num, i].cpu().numpy(),
                                                                   y[:, batch_num, i].cpu().numpy(),
                                                                   np_a if np_a is None else np_a[:, batch_num],
                                                                   np_b if np_b is None else np_b[:, batch_num],
                                                                   "minkowski", p=p)
                    else:
                        cpu_cost[i] = ot.lp.emd2_1d(x[:, i].cpu().numpy(), y[:, i].cpu().numpy(), np_a, np_b,
                                                    "minkowski", p=p)

                np.testing.assert_allclose(cpu_cost, torch_cost.cpu().numpy(), atol=1e-5)


def test_ot_loss_1d_grad():
    torch.random.manual_seed(42)
    n = 10
    m = 5
    k = 5
    ps = [1, 2, 3]

    for dtype in lst_types:
        for device in lst_devices:
            x = torch.randn(n, k, dtype=dtype, device=device, requires_grad=True)
            y = torch.randn(m, k, dtype=dtype, device=device, requires_grad=True)

            a = torch.rand(n, dtype=dtype, device=device, requires_grad=True)
            b = torch.rand(m, dtype=dtype, device=device, requires_grad=True)

            for p in ps:
                torch.autograd.gradcheck(lambda *inp: ot.torch.lp.ot_loss_1d(*inp, p=p), (x, y, a, b), eps=1e-3,
                                         atol=1e-2, raise_exception=True)

                res_equal = ot.torch.lp.ot_loss_1d(x, x, a, a, p=p).sum()
                print(torch.autograd.grad(res_equal, (x, a)))


@pytest.mark.filterwarnings("error")
def test_quantile():
    torch.random.manual_seed(42)
    dims = (100, 5, 3)
    cws = torch.rand(*dims)
    cws = cws / cws.sum(0, keepdim=True)
    cws = torch.cumsum(cws, 0)
    qs, _ = torch.sort(torch.rand(*dims), dim=0)
    xs = torch.randn(*dims)
    res = ot.torch.utils.quantile_function(qs, cws, xs)
    assert np.all(res.cpu().numpy() <= xs.max(0, keepdim=True)[0].cpu().numpy())
    assert np.all(res.cpu().numpy() >= xs.min(0, keepdim=True)[0].cpu().numpy())
