# %%
import numpy as np
from ot.backend import get_backend
import torch
import ot
from torch.optim import Adam


# %%
rng = np.random.RandomState(0)
n = 10
d = 2
X = rng.randn(n, d)
Y = rng.randn(n, d) + np.array([5.0, 0.0])[None, :]
n_proj = 20
P = ot.sliced.get_random_projections(d, n_proj)
a = rng.uniform(0, 1, n)
a /= a.sum()
b = rng.uniform(0, 1, n)
b /= b.sum()
sw2 = ot.sliced.sliced_wasserstein_distance(X, Y, a=a, b=b, projections=P)

# %%
nx = get_backend(torch.tensor([0.0]))
X_t = nx.from_numpy(X)
Y_t = nx.from_numpy(Y)
a_t = nx.from_numpy(a)
b_t = nx.from_numpy(b)
P_t = nx.from_numpy(P)
sw2_t = ot.sliced.sliced_wasserstein_distance(X_t, Y_t, a=a_t, b=b_t, projections=P_t)

# %%
