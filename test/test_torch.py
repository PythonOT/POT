"""Tests for main module ot """

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Adrien Corenflos <adrien.corenflos@gmail.com>
#
# License: MIT License

import numpy as np
import pytest

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

    dtype = torch.float64
    for device in lst_devices:
        x = torch.randn(n, k, dtype=dtype, device=device, requires_grad=True)
        y = torch.randn(m, k, dtype=dtype, device=device, requires_grad=True)

        a = torch.rand(n, dtype=dtype, device=device, requires_grad=True)
        a = a / a.sum()
        b = torch.rand(m, dtype=dtype, device=device, requires_grad=True)
        b = b / b.sum()

        for p in ps:
            torch.autograd.gradcheck(lambda *inp: ot.torch.lp.ot_loss_1d(*inp, p=p), (x, y, a, b), eps=1e-3,
                                     atol=1e-2, raise_exception=True)


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("k", [3, 5, 20])
def test_quantile(seed, k):
    torch.random.manual_seed(seed)
    dims = (100, 5, k)
    cws = torch.rand(*dims)
    cws = cws / cws.sum(0, keepdim=True)
    cws = torch.cumsum(cws, 0)
    qs, _ = torch.sort(torch.rand(*dims), dim=0)
    xs = torch.randn(*dims)
    res = ot.torch.utils.quantile_function(qs, cws, xs)
    assert np.all(res.cpu().numpy() <= xs.max(0, keepdim=True)[0].cpu().numpy())
    assert np.all(res.cpu().numpy() >= xs.min(0, keepdim=True)[0].cpu().numpy())


def test_quantile_duplicates():
    seed = 31415
    torch.random.manual_seed(seed)
    dims = (100, 5, 3)
    cws = torch.rand(*dims)
    cws[6:12] = cws[5:6]
    cws = cws / cws.sum(0, keepdim=True)
    cws = torch.cumsum(cws, 0)
    qs, _ = torch.sort(torch.rand(*dims), dim=0)
    xs = torch.randn(*dims)
    res = ot.torch.utils.quantile_function(qs, cws, xs)
    assert np.all(res.cpu().numpy() <= xs.max(0, keepdim=True)[0].cpu().numpy())
    assert np.all(res.cpu().numpy() >= xs.min(0, keepdim=True)[0].cpu().numpy())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_get_random_projections(seed):
    for device in lst_devices:
        torch_device = torch.device(device)
        for dtype in lst_types:
            gen = torch.Generator(torch_device)
            gen = gen.manual_seed(seed)
            projections_gen = ot.torch.sliced.get_random_projections(n_projections=50, d=15, seed=gen, dtype=dtype)
            assert projections_gen.dtype == dtype
            assert projections_gen.device.type == device

            projections_seed = ot.torch.sliced.get_random_projections(n_projections=50, d=15, seed=seed,
                                                                      device=torch_device, dtype=dtype)
            assert projections_seed.dtype == dtype
            assert projections_seed.device.type == device

            torch.manual_seed(seed)
            projections_global = ot.torch.sliced.get_random_projections(n_projections=50, d=15, dtype=dtype,
                                                                        device=torch_device)
            assert projections_global.dtype == dtype
            assert projections_global.device.type == device

            np.testing.assert_almost_equal(projections_gen.cpu().numpy(), projections_seed.cpu().numpy())
            np.testing.assert_almost_equal(projections_global.cpu().numpy(), projections_seed.cpu().numpy())


@pytest.mark.parametrize("np_seed", [0, 1, 2])
@pytest.mark.parametrize("torch_seed", [42, 66])
def test_sliced_different_dists(np_seed, torch_seed):
    n_projs = 100
    n = 100
    m = 50
    rng = np.random.RandomState(np_seed)

    x = rng.randn(n, 2)
    u = rng.uniform(0, 1, n)
    u /= u.sum()
    y = rng.randn(m, 2)
    v = rng.uniform(0, 1, m)
    v /= v.sum()

    ot_res = ot.sliced_wasserstein_distance(x, y, u, v, n_projections=n_projs, seed=torch_seed)
    torch_res = ot.torch.sliced.ot_loss_sliced(x, y, u, v, p=2, n_projections=n_projs, seed=torch_seed)
    np.testing.assert_almost_equal(torch_res, ot_res, decimal=5)


@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("data_seed", [42, 66])
@pytest.mark.parametrize("op_seed", [123, 1234])
def test_sliced_grad(p, data_seed, op_seed):
    device = "cpu"
    # scatter does not have a deterministic implementation for GPU,
    # so the test fails on GPU for lack of gradient determinism.
    n_projs = 100
    n = 30
    m = 50
    k = 3
    rng = np.random.RandomState(data_seed)
    np_x = rng.normal(size=(n, k))
    np_y = rng.normal(size=(m, k))
    np_a = rng.uniform(size=(n,))
    np_b = rng.normal(size=(m,))

    torch.random.manual_seed(data_seed)
    dtype = torch.float64
    x = torch.tensor(np_x, dtype=dtype, device=device, requires_grad=True)
    y = torch.tensor(np_y, dtype=dtype, device=device, requires_grad=True)

    a = torch.tensor(np_a, dtype=dtype, device=device, requires_grad=True)
    b = torch.tensor(np_b, dtype=dtype, device=device, requires_grad=True)
    gen = torch.Generator(device=device)
    torch.autograd.gradcheck(
        lambda X, Y, u, v: ot.torch.ot_loss_sliced(X, Y, u / u.sum(), v / v.sum(), p=p, n_projections=n_projs,
                                                   seed=gen.manual_seed(op_seed)),
        (x, y, a, b), eps=1e-6,
        atol=1e-2, raise_exception=True)
