# -*- coding: utf-8 -*-
"""
==================================================
Partial Wasserstein and Gromov-Wasserstein example
==================================================

.. note::
    Example added in release: 0.7.0.

This example is designed to show how to use the Partial (Gromov-)Wasserstein
distance computation in POT [29].

[29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
Transport with Applications on Positive-Unlabeled Learning". NeurIPS.
"""

# Author: Laetitia Chapel <laetitia.chapel@irisa.fr>
# Mass-interpretation extension: Jialin Tang <jtang3898@sdsu.edu>
# License: MIT License

# sphinx_gallery_thumbnail_number = 2

# necessary for 3d plot even if not used
from mpl_toolkits.mplot3d import Axes3D  # noqa
import scipy as sp
import numpy as np
import matplotlib.pylab as pl
import ot


#############################################################################
#
# Sample two 2D Gaussian distributions and plot them
# --------------------------------------------------
#
# For demonstration purpose, we sample two Gaussian distributions in 2-d
# spaces and add some random noise.


n_samples = 20  # nb samples (gaussian)
n_noise = 20  # nb of samples (noise)
random_state = np.random.RandomState(42)

mu = np.array([0, 0])
cov = np.array([[1, 0], [0, 2]])

xs = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=random_state)
xs = np.append(xs, (random_state.rand(n_noise, 2) + 1) * 4).reshape((-1, 2))
xt = ot.datasets.make_2D_samples_gauss(n_samples, mu, cov, random_state=random_state)
xt = np.append(xt, (random_state.rand(n_noise, 2) + 1) * -3).reshape((-1, 2))

M = sp.spatial.distance.cdist(xs, xt)

fig = pl.figure()
ax1 = fig.add_subplot(131)
ax1.plot(xs[:, 0], xs[:, 1], "+b", label="Source samples")
ax2 = fig.add_subplot(132)
ax2.scatter(xt[:, 0], xt[:, 1], color="r")
ax3 = fig.add_subplot(133)
ax3.imshow(M)
pl.show()

#############################################################################
#
# Compute partial Wasserstein plans and distance
# ----------------------------------------------

p = ot.unif(n_samples + n_noise)
q = ot.unif(n_samples + n_noise)

w0, log0 = ot.partial.partial_wasserstein(p, q, M, m=0.5, log=True)
w, log = ot.partial.entropic_partial_wasserstein(p, q, M, reg=0.1, m=0.5, log=True)

print("Partial Wasserstein distance (m = 0.5): " + str(log0["partial_w_dist"]))
print("Entropic partial Wasserstein distance (m = 0.5): " + str(log["partial_w_dist"]))

pl.figure(figsize=(10, 5))
pl.subplot(1, 2, 1)
pl.imshow(w0, cmap="gray_r")
pl.title("Partial Wasserstein")
pl.subplot(1, 2, 2)
pl.imshow(w, cmap="gray_r")
pl.title("Entropic partial Wasserstein")
pl.show()


#############################################################################
#
# Interpret the transported mass on the same 2D samples
# -----------------------------------------------------
#
# Both histograms have total mass one, so ``m`` is also the transported
# fraction here. With 40 uniformly weighted samples, each sample can supply
# or receive at most 1/40 of the mass. The parameter constrains total mass,
# rather than selecting a fixed number of points or one-to-one matches.
#
# We vary ``m`` using the exact partial Wasserstein solver. The first 20
# samples in each distribution are Gaussian samples (called inliers below),
# and the last 20 are remote samples (called outliers). These synthetic
# labels are used only to summarize the plans; they are not solver inputs.
# A small mass can omit genuine overlap, while a large mass can require
# transporting remote samples. Partial OT does not guarantee outlier removal.

mass_study = {
    "masses": np.array([0.25, 0.5, 0.75, 1.0]),
    "plans": [],
    "source_weights": p.copy(),
    "target_weights": q.copy(),
    "cost_matrix": M.copy(),
    "inlier_count": n_samples,
    "balanced_plan": ot.emd(p, q, M),
    "records": [],
}
tolerance = 1e-8

for mass in mass_study["masses"]:
    plan = ot.partial.partial_wasserstein(p, q, M, m=mass)
    assert np.isfinite(plan).all() and (plan >= 0).all()
    assert (plan.sum(axis=1) <= p + tolerance).all()
    assert (plan.sum(axis=0) <= q + tolerance).all()
    assert np.isclose(plan.sum(), mass, atol=tolerance, rtol=0)
    mass_study["plans"].append(plan)
    mass_study["records"].append(
        {
            "mass": float(mass),
            "cost": float(np.sum(plan * M)),
            "source_inlier_mass": float(plan[:n_samples, :].sum()),
            "source_outlier_mass": float(plan[n_samples:, :].sum()),
            "target_inlier_mass": float(plan[:, :n_samples].sum()),
            "target_outlier_mass": float(plan[:, n_samples:].sum()),
            # Count a pair involving either outlier group only once.
            "any_outlier_mass": float(plan.sum() - plan[:n_samples, :n_samples].sum()),
        }
    )

# At full mass, these normalized histograms recover balanced OT. Optimal
# plans need not be unique, so compare the marginals and objective values.
full_mass_plan = mass_study["plans"][-1]
assert np.allclose(full_mass_plan.sum(axis=1), p, atol=tolerance, rtol=0)
assert np.allclose(full_mass_plan.sum(axis=0), q, atol=tolerance, rtol=0)
assert np.isclose(
    np.sum(full_mass_plan * M),
    np.sum(mass_study["balanced_plan"] * M),
    atol=tolerance,
    rtol=0,
)

print("\nExact partial Wasserstein mass study (raw transport costs):")
print("   m      cost  source in source out  target in target out    any out")
for record in mass_study["records"]:
    print(
        f'{record["mass"]:4.2f} {record["cost"]:9.4f}'
        f' {record["source_inlier_mass"]:10.4f}'
        f' {record["source_outlier_mass"]:10.4f}'
        f' {record["target_inlier_mass"]:10.4f}'
        f' {record["target_outlier_mass"]:10.4f}'
        f' {record["any_outlier_mass"]:10.4f}'
    )

#############################################################################
#
# The plan matrices show which sample groups are coupled. Dashed lines
# separate the Gaussian and remote samples on each axis. All four panels
# use the same color scale, in units of mass per source-target pair.

fig_plans, plan_axes = pl.subplots(
    1, 4, figsize=(12, 3.6), sharex=True, sharey=True, constrained_layout=True
)
for ax, mass, plan in zip(plan_axes, mass_study["masses"], mass_study["plans"]):
    image = ax.imshow(plan, cmap="Blues", vmin=0, vmax=p.max())
    ax.axhline(n_samples - 0.5, color="gray", linestyle="--", linewidth=1)
    ax.axvline(n_samples - 0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"m = {mass:g}")
    ax.set_xlabel("Target sample index")
    ax.set_xticks([0, n_samples, n_samples + n_noise - 1])
plan_axes[0].set_ylabel("Source sample index")
fig_plans.suptitle("Exact partial OT: indices 0-19 are inliers, 20-39 are outliers")
fig_plans.colorbar(image, ax=plan_axes, label="Mass per pair", shrink=0.75)
pl.show()

#############################################################################
#
# Summarize the mass used from each distribution separately. Each stacked
# bar has height ``m``. Mass involving any outlier is reported in the table
# as total mass minus the inlier-inlier block, since adding the source and
# target outlier masses would count outlier-outlier pairs twice.
#
# These summaries describe participation of the synthetic groups, not
# point-correspondence accuracy. Raw costs at different masses are not a
# direct comparison of matching quality, because the amount transported
# changes. The solver also does not choose ``m`` automatically.

fig_mass, mass_axes = pl.subplots(
    1, 2, figsize=(9, 4), sharey=True, constrained_layout=True
)
for ax, side in zip(mass_axes, ("source", "target")):
    inlier_masses = [record[f"{side}_inlier_mass"] for record in mass_study["records"]]
    outlier_masses = [
        record[f"{side}_outlier_mass"] for record in mass_study["records"]
    ]
    ax.bar(mass_study["masses"], inlier_masses, width=0.15, label="Gaussian (inlier)")
    ax.bar(
        mass_study["masses"],
        outlier_masses,
        bottom=inlier_masses,
        width=0.15,
        label="Remote (outlier)",
    )
    ax.set_title(f"{side.capitalize()} matched mass")
    ax.set_xlabel("Requested transported mass m")
    ax.set_xticks(mass_study["masses"])
    ax.set_ylim(0, 1.05)
mass_axes[0].set_ylabel("Matched mass")
mass_axes[1].legend(loc="upper left")
mass_study["figures"] = {"plans": fig_plans, "mass_accounting": fig_mass}
pl.show()


#############################################################################
#
# Sample one 2D and 3D Gaussian distributions and plot them
# ---------------------------------------------------------
#
# The Gromov-Wasserstein distance allows to compute distances with samples that
# do not belong to the same metric space. For demonstration purpose, we sample
# two Gaussian distributions in 2- and 3-dimensional spaces.

n_samples = 20  # nb samples
n_noise = 10  # nb of samples (noise)

p = ot.unif(n_samples + n_noise)
q = ot.unif(n_samples + n_noise)

mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

mu_t = np.array([0, 0, 0])
cov_t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


xs = ot.datasets.make_2D_samples_gauss(
    n_samples, mu_s, cov_s, random_state=random_state
)
xs = np.concatenate((xs, ((random_state.rand(n_noise, 2) + 1) * 4)), axis=0)
P = sp.linalg.sqrtm(cov_t)
xt = random_state.randn(n_samples, 3).dot(P) + mu_t
xt = np.concatenate((xt, ((random_state.rand(n_noise, 3) + 1) * 10)), axis=0)

fig = pl.figure()
ax1 = fig.add_subplot(121)
ax1.plot(xs[:, 0], xs[:, 1], "+b", label="Source samples")
ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(xt[:, 0], xt[:, 1], xt[:, 2], color="r")
pl.show()


#############################################################################
#
# Compute partial Gromov-Wasserstein plans and distance
# -----------------------------------------------------

C1 = sp.spatial.distance.cdist(xs, xs)
C2 = sp.spatial.distance.cdist(xt, xt)

# transport 100% of the mass
print("------m = 1")
m = 1
res0, log0 = ot.gromov.partial_gromov_wasserstein(C1, C2, p, q, m=m, log=True)
res, log = ot.gromov.entropic_partial_gromov_wasserstein(
    C1, C2, p, q, 10, m=m, log=True, verbose=True
)

print("Wasserstein distance (m = 1): " + str(log0["partial_gw_dist"]))
print("Entropic Wasserstein distance (m = 1): " + str(log["partial_gw_dist"]))

pl.figure(figsize=(10, 5))
pl.title("mass to be transported m = 1")
pl.axis("off")
pl.subplot(1, 2, 1)
pl.imshow(res0, cmap="gray_r")
pl.title("Gromov-Wasserstein")
pl.subplot(1, 2, 2)
pl.imshow(res, cmap="gray_r")
pl.title("Entropic Gromov-Wasserstein")
pl.show()

# transport 2/3 of the mass
print("------m = 2/3")
m = 2 / 3
res0, log0 = ot.gromov.partial_gromov_wasserstein(
    C1, C2, p, q, m=m, log=True, verbose=True
)
res, log = ot.gromov.entropic_partial_gromov_wasserstein(
    C1, C2, p, q, 10, m=m, log=True, verbose=True
)

print("Partial Wasserstein distance (m = 2/3): " + str(log0["partial_gw_dist"]))
print("Entropic partial Wasserstein distance (m = 2/3): " + str(log["partial_gw_dist"]))

pl.figure(figsize=(10, 5))
pl.title("mass to be transported m = 2/3")
pl.axis("off")
pl.subplot(1, 2, 1)
pl.imshow(res0, cmap="gray_r")
pl.title("Partial Gromov-Wasserstein")
pl.subplot(1, 2, 2)
pl.imshow(res, cmap="gray_r")
pl.title("Entropic partial Gromov-Wasserstein")
pl.show()
