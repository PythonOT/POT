"""Tests that solvers allocate their internal arrays on the input device"""

# License: MIT License

import numpy as np
import pytest

import ot
from ot.backend import torch


def _inputs():
    # torch.from_numpy always returns CPU tensors, whatever the default device
    rng = np.random.RandomState(0)
    n, d = 6, 2
    xs = torch.from_numpy(rng.randn(n, d))
    xt = torch.from_numpy(rng.randn(n + 1, d))
    a = ot.unif(n, type_as=xs)
    b = ot.unif(n + 1, type_as=xs)
    M = ot.dist(xs, xt)
    M = M / M.max()
    xs3 = torch.from_numpy(rng.randn(n, 3))
    xt3 = torch.from_numpy(rng.randn(n + 1, 3))
    return dict(
        xs=xs,
        xt=xt,
        a=a,
        b=b,
        M=M,
        C1=ot.dist(xs, xs),
        C2=ot.dist(xt, xt),
        u=torch.from_numpy(rng.rand(n)),
        v=torch.from_numpy(rng.rand(n + 1)),
        xs3=xs3 / xs3.norm(dim=1, keepdim=True),
        xt3=xt3 / xt3.norm(dim=1, keepdim=True),
        m=xs[:2],
        mt=xt[:2],
        C=torch.from_numpy(np.stack([np.eye(d), 2 * np.eye(d)])),
        w=torch.from_numpy(np.array([0.3, 0.7])),
        V=torch.from_numpy(rng.randn(3, 4)),
    )


CASES = {
    "emd": lambda i: ot.emd(i["a"], i["b"], i["M"]),
    "sinkhorn": lambda i: ot.sinkhorn(i["a"], i["b"], i["M"], 1.0),
    "sinkhorn_log": lambda i: ot.sinkhorn(
        i["a"], i["b"], i["M"], 1.0, method="sinkhorn_log"
    ),
    "sinkhorn_unbalanced": lambda i: ot.sinkhorn_unbalanced(
        i["a"], i["b"], i["M"], 1.0, 1.0
    ),
    "solve": lambda i: ot.solve(i["M"], i["a"], i["b"], reg=1.0).plan,
    "solve_sample": lambda i: ot.solve_sample(i["xs"], i["xt"], i["a"], i["b"]).plan,
    "solve_gromov": lambda i: ot.solve_gromov(i["C1"], i["C2"]).plan,
    "partial_wasserstein": lambda i: ot.partial.partial_wasserstein(
        i["a"], i["b"], i["M"], m=0.5
    ),
    "wasserstein_1d": lambda i: ot.wasserstein_1d(i["u"], i["v"]),
    "wasserstein_circle": lambda i: ot.wasserstein_circle(i["u"], i["v"]),
    "sliced_wasserstein_sphere": lambda i: ot.sliced_wasserstein_sphere(
        i["xs3"], i["xt3"], n_projections=5, seed=0
    ),
    "empirical_bures_wasserstein_distance_hd": lambda i: (
        ot.gaussian.empirical_bures_wasserstein_distance_hd(i["xs"], i["xt"], 1)
    ),
    "empirical_bures_wasserstein_mapping_hd": lambda i: (
        ot.gaussian.empirical_bures_wasserstein_mapping_hd(i["xs"], i["xt"], 1)
    ),
    "gmm_pdf": lambda i: ot.gmm.gmm_pdf(i["xs"], i["m"], i["C"], i["w"]),
    "gmm_ot_apply_map_bary": lambda i: ot.gmm.gmm_ot_apply_map(
        i["xs"], i["m"], i["mt"], i["C"], i["C"], i["w"], i["w"], method="bary"
    ),
    "gmm_ot_apply_map_rand": lambda i: ot.gmm.gmm_ot_apply_map(
        i["xs"], i["m"], i["mt"], i["C"], i["C"], i["w"], i["w"], method="rand", seed=0
    ),
    "gmm_ot_plan_density": lambda i: ot.gmm.gmm_ot_plan_density(
        i["xs"], i["xt"], i["m"], i["mt"], i["C"], i["C"], i["w"], i["w"]
    ),
    "lowrank_sinkhorn": lambda i: ot.lowrank_sinkhorn(
        i["xs"], i["xt"], rank=2, init="deterministic"
    ),
    "semirelaxed_gromov_barycenters": lambda i: (
        ot.gromov.semirelaxed_gromov_barycenters(
            3, [i["C1"], i["C2"]], max_iter=2, random_state=0
        )
    ),
    "projection_sparse_simplex": lambda i: ot.utils.projection_sparse_simplex(
        i["V"], 2
    ),
}


@pytest.mark.skipif(not torch, reason="torch not installed")
@pytest.mark.parametrize("name", list(CASES))
def test_no_default_device_allocation(name, meta_default_device):
    res = CASES[name](_inputs())
    res = res if isinstance(res, (tuple, list)) else [res]
    for r in res:
        if torch.is_tensor(r):
            assert r.device.type == "cpu"
