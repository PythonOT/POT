import ot
import numpy as np
import warnings
import time
import torch

from ot.bregman._geomloss import empirical_sinkhorn2_geomloss

warnings.filterwarnings("ignore")

STEP_NAMES = ["setup", "MST", "adj_list", "artif+BFS", "thread", "flow", "potentials"]
SIZES = [4000]
NUM_ITER_MAX = 100000

for n in SIZES:
    np.random.seed(42)
    a, b = ot.utils.unif(n), ot.utils.unif(n)
    X = np.random.randn(n, 2)
    Y = np.random.randn(n, 2)
    M = ot.dist(X, Y)

    print(f"\n{'='*80}")
    print(f"  n = {n}  (numItermax={NUM_ITER_MAX})")
    print(f"{'='*80}")

    # --- 1. Cold start EMD ---
    t0 = time.time()
    G_cold, log_cold = ot.emd(a, b, M, numItermax=NUM_ITER_MAX, log=True)
    t_cold = time.time() - t0
    cost_cold = log_cold["cost"]
    niter_cold = log_cold.get("niter", -1)
    alpha_e, beta_e = log_cold["u"], log_cold["v"]

    # --- 2. Exact warmstart (use cold-start duals) ---
    t0 = time.time()
    G_exact, log_exact = ot.emd(
        a, b, M, numItermax=NUM_ITER_MAX, log=True, warmstart_dual=(alpha_e, beta_e)
    )
    t_exact = time.time() - t0
    cost_exact = log_exact["cost"]
    niter_exact = log_exact.get("niter", -1)
    init_exact_ms = log_exact.get("warmstart_init_ms", 0.0)
    steps_exact = log_exact.get("warmstart_step_times_ms", [0.0] * 7)

    # --- 3. GeomLoss Sinkhorn ---
    x_t = torch.from_numpy(np.ascontiguousarray(X)).float()
    y_t = torch.from_numpy(np.ascontiguousarray(Y)).float()
    a_t = torch.from_numpy(a.flatten()).float().contiguous()
    b_t = torch.from_numpy(b.flatten()).float().contiguous()

    t0 = time.time()
    _, log_geo = empirical_sinkhorn2_geomloss(
        x_t,
        y_t,
        0.00001,
        a_t,
        b_t,
        metric="sqeuclidean",
        backend="tensorized",
        scaling=0.9,
        log=True,
    )
    t_geo_sink = time.time() - t0

    f_np = log_geo["f"].cpu().detach().numpy().flatten()
    g_np = log_geo["g"].cpu().detach().numpy().flatten()
    alpha_geo = 2.0 * f_np
    beta_geo = 2.0 * g_np
    rc_min = (M - alpha_geo[:, None] - beta_geo[None, :]).min()
    if rc_min < 0:
        alpha_geo += rc_min / 2
        beta_geo += rc_min / 2

    # --- 4. EMD warmstart with GeomLoss potentials ---
    t0 = time.time()
    G_gw, log_gw = ot.emd(
        a, b, M, numItermax=NUM_ITER_MAX, log=True, warmstart_dual=(alpha_geo, beta_geo)
    )
    t_geo_warm = time.time() - t0
    cost_geo = log_gw["cost"]
    niter_geo = log_gw.get("niter", -1)
    init_geo_ms = log_gw.get("warmstart_init_ms", 0.0)
    steps_geo = log_gw.get("warmstart_step_times_ms", [0.0] * 7)

    cost_ok = np.allclose(cost_cold, cost_geo, rtol=1e-5)
    cost_diff = cost_geo - cost_cold

    # --- Print summary ---
    print(f"\n  Timing (seconds):")
    print(f"    cold EMD        : {t_cold:8.3f}s   ({niter_cold} iters)")
    print(f"    exact warmstart : {t_exact:8.3f}s   ({niter_exact} iters)")
    print(f"    GeomLoss sink   : {t_geo_sink:8.3f}s")
    print(f"    geo warmstart   : {t_geo_warm:8.3f}s   ({niter_geo} iters)")

    print(
        f"\n  Cost check: cold={cost_cold:.6f}  geo_ws={cost_geo:.6f}  diff={cost_diff:.2e}  {'OK' if cost_ok else 'MISMATCH'}"
    )

    print(f"\n  Warmstart init breakdown (ms):")
    print(f"    {'step':<14} {'exact_ws':>10} {'geo_ws':>10}")
    print(f"    {'-'*14} {'-'*10} {'-'*10}")
    total_e, total_g = 0.0, 0.0
    for i, name in enumerate(STEP_NAMES):
        se = steps_exact[i] if i < len(steps_exact) else 0.0
        sg = steps_geo[i] if i < len(steps_geo) else 0.0
        total_e += se
        total_g += sg
        print(f"    {name:<14} {se:>10.1f} {sg:>10.1f}")
    print(f"    {'TOTAL':<14} {total_e:>10.1f} {total_g:>10.1f}")
    print(f"    (C++ total)     {init_exact_ms:>10.1f} {init_geo_ms:>10.1f}")
