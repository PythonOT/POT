# %% Test nystrom
import matplotlib.pyplot as plt
import numpy as np
from ot.bregman import empirical_sinkhorn_nystroem
from ot.bregman import empirical_sinkhorn
from sklearn.datasets import make_blobs


def plot2D_samples_mat(xs, xt, G, ax, thr=1e-8, **kwargs):
    if ("color" not in kwargs) and ("c" not in kwargs):
        kwargs["color"] = "k"
    mx = G.max()
    if "alpha" in kwargs:
        scale = kwargs["alpha"]
        del kwargs["alpha"]
    else:
        scale = 1
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                ax.plot(
                    [xs[i, 0], xt[j, 0]],
                    [xs[i, 1], xt[j, 1]],
                    alpha=G[i, j] / mx * scale,
                    **kwargs,
                )


# %%
offset = 2
seed = 42
centers = np.array(
    [
        [0, 0],
        [offset, 0],
        [0, 0],
        [-offset, 0],
    ]
)
X, y = make_blobs(
    n_samples=100,
    centers=centers,
    n_features=2,
    random_state=seed,
    cluster_std=0.1,
    shuffle=False,
)
y[y == 1] = 0
y[y == 2] = 1
y[y == 3] = 1

Xs = X[y == 0]
Xt = X[y == 1]

# rank = 50
# reg = 1e-3
# K_true = np.exp(- ot.dist(Xs, Xt) / reg)
# error = []
# all_rank = [5, 10, 50, 100, 200, 300, 500, 800]
# for r in all_rank:
#     U_nys, V_nys = kernel_nystroem(Xs, Xt, rank=r, reg=reg)
#     # print(f'{U_nys.shape=}')
#     error.append(np.linalg.norm(K_true - U_nys @ V_nys.T, ord=2))

# fig, ax = plt.subplots(figsize=(5, 5))
# ax.plot(all_rank, error, '-o', lw=2, label='error')
# ax.legend()
# ax.set_xlabel('rank')
# ax.set_yscale('log')
# # ax.set_xscale('log')
# ax.grid()

# %%
reg = 2
rank = 100
_, _, dict_log = empirical_sinkhorn_nystroem(
    Xs, Xt, rank=rank, reg=reg, numItermax=3000, log=True
)
G_nys = dict_log["lazy_plan"][:]
# %%
G_sinkh = empirical_sinkhorn(Xs, Xt, reg=reg, numIterMax=3000)
# %%
cmap = plt.cm.get_cmap("tab10")
fs = 12
fig, ax = plt.subplots(2, 1, figsize=(7, 7), constrained_layout=True)
for i, (T, name) in enumerate(zip([G_sinkh, G_nys], ["T_sinkhorn", "T_nystroem"])):
    ax[i].scatter(
        X[:, 0],
        X[:, 1],
        c=[cmap(i) for i in y],
        alpha=0.9,
        edgecolor="k",
        zorder=1000,
        s=50,
    )
    # ax[i].set_aspect('equal', adjustable='box')
    plot2D_samples_mat(Xs, Xt, T, ax[i], alpha=0.3, thr=1e-7)
    ax[i].set_title(name, fontsize=fs)
    ax[i].grid(alpha=0.5)
# plt.tight_layout()
# fig.subplots_adjust(wspace=0, hspace=0)
# %%
# Set up the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
vmin = min(G_sinkh.min(), G_nys.min())
vmax = max(G_sinkh.max(), G_nys.max())
im1 = ax1.imshow(G_sinkh, cmap="coolwarm", vmin=vmin, vmax=vmax)
ax1.set_title("T_sinkhorn")
im2 = ax2.imshow(G_nys, cmap="coolwarm", vmin=vmin, vmax=vmax)
ax2.set_title("T_nystroem")
cbar = fig.colorbar(
    im1, ax=[ax1, ax2], orientation="vertical", fraction=0.046, pad=0.04, shrink=0.8
)

# plt.tight_layout()
plt.show()
# %%
