Releases
========

0.8.0
-----

*November 2021*

This new stable release introduces several important features.

First we now have an OpenMP compatible exact ot solver in ``ot.emd``.
The OpenMP version is used when the parameter ``numThreads`` is greater
than one and can lead to nice speedups on multi-core machines.

| Second we have introduced a backend mechanism that allows to use
  standard POT function seamlessly on Numpy, Pytorch and Jax arrays.
  Other backends are coming but right now POT can be used seamlessly for
  training neural networks in Pytorch. Notably we propose the first
  differentiable computation of the exact OT loss with ``ot.emd2`` (can
  be differentiated w.r.t. both cost matrix and sample weights), but
  also for the classical Sinkhorn loss with ``ot.sinkhorn2``, the
  Wasserstein distance in 1D with ``ot.wasserstein_1d``, sliced
  Wasserstein with ``ot.sliced_wasserstein_distance`` and
  Gromov-Wasserstein with ``ot.gromov_wasserstein2``. Examples of how
  this new feature can be used are now available in the documentation
  where the Pytorch backend is used to estimate a `minimal Wasserstein
  estimator <https://PythonOT.github.io/auto_examples/backends/plot_unmix_optim_torch.html>`__,
  a `Generative Network
  (GAN) <https://PythonOT.github.io/auto_examples/backends/plot_wass2_gan_torch.html>`__,
  for a `sliced Wasserstein gradient
  flow <https://PythonOT.github.io/auto_examples/backends/plot_sliced_wass_grad_flow_pytorch.html>`__
  and `optimizing the Gromov-Wassersein
  distance <https://PythonOT.github.io/auto_examples/backends/plot_optim_gromov_pytorch.html>`__.
  Note that the Jax backend is still in early development and quite slow
  at the moment, we strongly recommend for Jax users to use the `OTT
  toolbox <https://github.com/google-research/ott>`__ when possible.
| As a result of this new feature, the old ``ot.gpu`` submodule is now
  deprecated since GPU implementations can be done using GPU arrays on
  the torch backends.

Other novel features include implementation for `Sampled Gromov
Wasserstein and Pointwise Gromov
Wasserstein <https://PythonOT.github.io/auto_examples/gromov/plot_gromov.html#compute-gw-with-a-scalable-stochastic-method-with-any-loss-function>`__,
Sinkhorn in log space with ``method='sinkhorn_log'``, `Projection Robust
Wasserstein <https://PythonOT.github.io/gen_modules/ot.dr.html?highlight=robust#ot.dr.projection_robust_wasserstein>`__,
ans `deviased Sinkorn
barycenters <https://PythonOT.github.ioauto_examples/barycenters/plot_debiased_barycenter.html>`__.

This release will also simplify the installation process. We have now a
``pyproject.toml`` that defines the build dependency and POT should now
build even when cython is not installed yet. Also we now provide
pe-compiled wheels for linux ``aarch64`` that is used on Raspberry PI
and android phones and for MacOS on ARM processors.

Finally POT was accepted for publication in the Journal of Machine
Learning Research (JMLR) open source software track and we ask the POT
users to cite `this
paper <https://www.jmlr.org/papers/v22/20-451.html>`__ from now on. The
documentation has been improved in particular by adding a "Why OT?"
section to the quick start guide and several new examples illustrating
the new features. The documentation now has two version : the stable
version https://pythonot.github.io/ corresponding to the last release
and the master version https://pythonot.github.io/master that
corresponds to the current master branch on GitHub.

As usual, we want to thank all the POT contributors (now 37 people have
contributed to the toolbox). But for this release we thank in particular
Nathan Cassereau and Kamel Guerda from the AI support team at
`IDRIS <http://www.idris.fr/>`__ for their support to the development of
the backend and OpenMP implementations.

New features
^^^^^^^^^^^^

-  OpenMP support for exact OT solvers (PR #260)
-  Backend for running POT in numpy/torch + exact solver (PR #249)
-  Backend implementation of most functions in ``ot.bregman`` (PR #280)
-  Backend implementation of most functions in ``ot.optim`` (PR #282)
-  Backend implementation of most functions in ``ot.gromov`` (PR #294,
   PR #302)
-  Test for arrays of different type and device (CPU/GPU) (PR #304,
   #303)
-  Implementation of Sinkhorn in log space with
   ``method='sinkhorn_log'`` (PR #290)
-  Implementation of regularization path for L2 Unbalanced OT (PR #274)
-  Implementation of Projection Robust Wasserstein (PR #267)
-  Implementation of Debiased Sinkhorn Barycenters (PR #291)
-  Implementation of Sampled Gromov Wasserstein and Pointwise Gromov
   Wasserstein (PR #275)
-  Add ``pyproject.toml`` and build POT without installing cython first
   (PR #293)
-  Lazy implementation in log space for sinkhorn on samples (PR #259)
-  Documentation cleanup (PR #298)
-  Two up-to-date documentations `for stable
   release <https://PythonOT.github.io/>`__ and for `master
   branch <https://pythonot.github.io/master/>`__.
-  Building wheels on ARM for Raspberry PI and smartphones (PR #238)
-  Update build wheels to new version and new pythons (PR #236, #253)
-  Implementation of sliced Wasserstein distance (Issue #202, PR #203)
-  Add minimal build to CI and perform pep8 test separately (PR #210)
-  Speedup of tests and return run time (PR #262)
-  Add "Why OT" discussion to the documentation (PR #220)
-  New introductory example to discrete OT in the documentation (PR
   #191)
-  Add templates for Issues/PR on Github (PR#181)

Closed issues
^^^^^^^^^^^^^

-  Debug Memory leak in GAN example (#254)
-  DEbug GPU bug (Issue #284, #287, PR #288)
-  set\_gradients method for JAX backend (PR #278)
-  Quicker GAN example for CircleCI build (PR #258)
-  Better formatting in Readme (PR #234)
-  Debug CI tests (PR #240, #241, #242)
-  Bug in Partial OT solver dummy points (PR #215)
-  Bug when Armijo linesearch (Issue #184, #198, #281, PR #189, #199,
   #286)
-  Bug Barycenter Sinkhorn (Issue 134, PR #195)
-  Infeasible solution in exact OT (Issues #126,#93, PR #217)
-  Doc for SUpport Barycenters (Issue #200, PR #201)
-  Fix labels transport in BaseTransport (Issue #207, PR #208)
-  Bug in ``emd_1d``, non respected bounds (Issue #169, PR #170)
-  Removed Python 2.7 support and update codecov file (PR #178)
-  Add normalization for WDA and test it (PR #172, #296)
-  Cleanup code for new version of ``flake8`` (PR #176)
-  Fixed requirements in ``setup.py`` (PR #174)
-  Removed specific MacOS flags (PR #175)

0.7.0
-----

*May 2020*

This is the new stable release for POT. We made a lot of changes in the
documentation and added several new features such as Partial OT,
Unbalanced and Multi Sources OT Domain Adaptation and several bug fixes.
One important change is that we have created the GitHub organization
`PythonOT <https://github.com/PythonOT>`__ that now owns the main POT
repository https://github.com/PythonOT/POT and the repository for the
new documentation is now hosted at https://PythonOT.github.io/.

This is the first release where the Python 2.7 tests have been removed.
Most of the toolbox should still work but we do not offer support for
Python 2.7 and will close related Issues.

A lot of changes have been done to the documentation that is now hosted
on https://PythonOT.github.io/ instead of readthedocs. It was a hard
choice but readthedocs did not allow us to run sphinx-gallery to update
our beautiful examples and it was a huge amount of work to maintain. The
documentation is now automatically compiled and updated on merge. We
also removed the notebooks from the repository for space reason and also
because they are all available in the `example
gallery <auto_examples/index.html>`__. Note
that now the output of the documentation build for each commit in the PR
is available to check that the doc builds correctly before merging which
was not possible with readthedocs.

The CI framework has also been changed with a move from Travis to Github
Action which allows to get faster tests on Windows, MacOS and Linux. We
also now report our coverage on
`Codecov.io <https://codecov.io/gh/PythonOT/POT>`__ and we have a
reasonable 92% coverage. We also now generate wheels for a number of OS
and Python versions at each merge in the master branch. They are
available as outputs of this
`action <https://github.com/PythonOT/POT/actions?query=workflow%3A%22Build+dist+and+wheels%22>`__.
This will allow simpler multi-platform releases from now on.

In terms of new features we now have `OTDA Classes for unbalanced
OT <https://pythonot.github.io/gen_modules/ot.da.html#ot.da.UnbalancedSinkhornTransport>`__,
a new Domain adaptation class form `multi domain problems
(JCPOT) <auto_examples/domain-adaptation/plot_otda_jcpot.html#sphx-glr-auto-examples-domain-adaptation-plot-otda-jcpot-py>`__,
and several solvers to solve the `Partial Optimal
Transport <auto_examples/unbalanced-partial/plot_partial_wass_and_gromov.html#sphx-glr-auto-examples-unbalanced-partial-plot-partial-wass-and-gromov-py>`__
problems.

This release is also the moment to thank all the POT contributors (old
and new) for helping making POT such a nice toolbox. A lot of changes
(also in the API) are coming for the next versions.

Features
^^^^^^^^

-  New documentation on https://PythonOT.github.io/ (PR #160, PR #143,
   PR #144)
-  Documentation build on CircleCI with sphinx-gallery (PR #145,PR #146,
   #155)
-  Run sphinx gallery in CI (PR #146)
-  Remove notebooks from repo because available in doc (PR #156)
-  Build wheels in CI (#157)
-  Move from travis to GitHub Action for Windows, MacOS and Linux (PR
   #148, PR #150)
-  Partial Optimal Transport (PR#141 and PR #142)
-  Laplace regularized OTDA (PR #140)
-  Multi source DA with target shift (PR #137)
-  Screenkhorn algorithm (PR #121)

Closed issues
^^^^^^^^^^^^^

-  Add JMLR paper to teh readme ad Mathieu Blondel to the Acknoledgments
   (PR #231, #232)
-  Bug in Unbalanced OT example (Issue #127)
-  Clean Cython output when calling setup.py clean (Issue #122)
-  Various Macosx compilation problems (Issue #113, Issue #118, PR#130)
-  EMD dimension mismatch (Issue #114, Fixed in PR #116)
-  2D barycenter bug for non square images (Issue #124, fixed in PR
   #132)
-  Bad value in EMD 1D (Issue #138, fixed in PR #139)
-  Log bugs for Gromov-Wassertein solver (Issue #107, fixed in PR #108)
-  Weight issues in barycenter function (PR #106)

0.6.0
-----

*July 2019*

This is the first official stable release of POT and this means a jump
to 0.6! The library has been used in the wild for a while now and we
have reached a state where a lot of fundamental OT solvers are available
and tested. It has been quite stable in the last months but kept the
beta flag in its Pypi classifiers until now.

Note that this release will be the last one supporting officially Python
2.7 (See https://python3statement.org/ for more reasons). For next
release we will keep the travis tests for Python 2 but will make them
non necessary for merge in 2020.

The features are never complete in a toolbox designed for solving
mathematical problems and research but with the new contributions we now
implement algorithms and solvers from 24 scientific papers (listed in
the README.md file). New features include a direct implementation of the
`empirical Sinkhorn
divergence <all.html#ot.bregman.empirical_sinkhorn_divergence>`__,
a new efficient (Cython implementation) solver for `EMD in
1D <all.html#ot.lp.emd_1d>`__ and
corresponding `Wasserstein
1D <all.html#ot.lp.wasserstein_1d>`__.
We now also have implementations for `Unbalanced
OT <auto_examples/plot_UOT_1D.html>`__
and a solver for `Unbalanced OT
barycenters <auto_examples/plot_UOT_barycenter_1D.html>`__.
A new variant of Gromov-Wasserstein divergence called `Fused
Gromov-Wasserstein <all.html?highlight=fused_#ot.gromov.fused_gromov_wasserstein>`__
has been also contributed with exemples of use on `structured
data <auto_examples/plot_fgw.html>`__
and computing `barycenters of labeld
graphs <auto_examples/plot_barycenter_fgw.html>`__.

A lot of work has been done on the documentation with several new
examples corresponding to the new features and a lot of corrections for
the docstrings. But the most visible change is a new `quick start
guide <quickstart.html>`__ for POT
that gives several pointers about which function or classes allow to
solve which specific OT problem. When possible a link is provided to
relevant examples.

We will also provide with this release some pre-compiled Python wheels
for Linux 64bit on github and pip. This will simplify the install
process that before required a C compiler and numpy/cython already
installed.

Finally we would like to acknowledge and thank the numerous contributors
of POT that has helped in the past build the foundation and are still
contributing to bring new features and solvers to the library.

Features
^^^^^^^^

-  Add compiled manylinux 64bits wheels to pip releases (PR #91)
-  Add quick start guide (PR #88)
-  Make doctest work on travis (PR #90)
-  Update documentation (PR #79, PR #84)
-  Solver for EMD in 1D (PR #89)
-  Solvers for regularized unbalanced OT (PR #87, PR#99)
-  Solver for Fused Gromov-Wasserstein (PR #86)
-  Add empirical Sinkhorn and empirical Sinkhorn divergences (PR #80)

Closed issues
^^^^^^^^^^^^^

-  Issue #59 fail when using "pip install POT" (new details in doc+
   hopefully wheels)
-  Issue #85 Cannot run gpu modules
-  Issue #75 Greenkhorn do not return log (solved in PR #76)
-  Issue #82 Gromov-Wasserstein fails when the cost matrices are
   slightly different
-  Issue #72 Macosx build problem

0.5.0
-----

*Sep 2018*

POT is 2 years old! This release brings numerous new features to the
toolbox as listed below but also several bug correction.

| Among the new features, we can highlight a `non-regularized
  Gromov-Wasserstein
  solver <auto_examples/plot_gromov.html>`__,
  a new `greedy variant of
  sinkhorn <all.html#ot.bregman.greenkhorn>`__,
| `non-regularized <all.html#ot.lp.barycenter>`__,
  `convolutional
  (2D) <auto_examples/plot_convolutional_barycenter.html>`__
  and `free
  support <auto_examples/plot_free_support_barycenter.html>`__
  Wasserstein barycenters and
  `smooth <https://github.com/rflamary/POT/blob/prV0.5/notebooks/plot_OT_1D_smooth.html>`__
  and
  `stochastic <all.html#ot.stochastic.sgd_entropic_regularization>`__
  implementation of entropic OT.

POT 0.5 also comes with a rewriting of ot.gpu using the cupy framework
instead of the unmaintained cudamat. Note that while we tried to keed
changes to the minimum, the OTDA classes were deprecated. If you are
happy with the cudamat implementation, we recommend you stay with stable
release 0.4 for now.

The code quality has also improved with 92% code coverage in tests that
is now printed to the log in the Travis builds. The documentation has
also been greatly improved with new modules and examples/notebooks.

This new release is so full of new stuff and corrections thanks to the
old and new POT contributors (you can see the list in the
`readme <https://github.com/rflamary/POT/blob/master/README.md>`__).

Features
^^^^^^^^

-  Add non regularized Gromov-Wasserstein solver (PR #41)
-  Linear OT mapping between empirical distributions and 90% test
   coverage (PR #42)
-  Add log parameter in class EMDTransport and SinkhornLpL1Transport (PR
   #44)
-  Add Markdown format for Pipy (PR #45)
-  Test for Python 3.5 and 3.6 on Travis (PR #46)
-  Non regularized Wasserstein barycenter with scipy linear solver
   and/or cvxopt (PR #47)
-  Rename dataset functions to be more sklearn compliant (PR #49)
-  Smooth and sparse Optimal transport implementation with entropic and
   quadratic regularization (PR #50)
-  Stochastic OT in the dual and semi-dual (PR #52 and PR #62)
-  Free support barycenters (PR #56)
-  Speed-up Sinkhorn function (PR #57 and PR #58)
-  Add convolutional Wassersein barycenters for 2D images (PR #64)
-  Add Greedy Sinkhorn variant (Greenkhorn) (PR #66)
-  Big ot.gpu update with cupy implementation (instead of un-maintained
   cudamat) (PR #67)

Deprecation
^^^^^^^^^^^

Deprecated OTDA Classes were removed from ot.da and ot.gpu for version
0.5 (PR #48 and PR #67). The deprecation message has been for a year
here since 0.4 and it is time to pull the plug.

Closed issues
^^^^^^^^^^^^^

-  Issue #35 : remove import plot from ot/\ **init**.py (See PR #41)
-  Issue #43 : Unusable parameter log for EMDTransport (See PR #44)
-  Issue #55 : UnicodeDecodeError: 'ascii' while installing with pip

0.4
---

*15 Sep 2017*

This release contains a lot of contribution from new contributors.

Features
^^^^^^^^

-  Automatic notebooks and doc update (PR #27)
-  Add gromov Wasserstein solver and Gromov Barycenters (PR #23)
-  emd and emd2 can now return dual variables and have max\_iter (PR #29
   and PR #25)
-  New domain adaptation classes compatible with scikit-learn (PR #22)
-  Proper tests with pytest on travis (PR #19)
-  PEP 8 tests (PR #13)

Closed issues
^^^^^^^^^^^^^

-  emd convergence problem du to fixed max iterations (#24)
-  Semi supervised DA error (#26)

0.3.1
-----

*11 Jul 2017*

-  Correct bug in emd on windows

0.3
---

*7 Jul 2017*

-  emd\* and sinkhorn\* are now performed in parallel for multiple
   target distributions
-  emd and sinkhorn are for OT matrix computation
-  emd2 and sinkhorn2 are for OT loss computation
-  new notebooks for emd computation and Wasserstein Discriminant
   Analysis
-  relocate notebooks
-  update documentation
-  clean\_zeros(a,b,M) for removimg zeros in sparse distributions
-  GPU implementations for sinkhorn and group lasso regularization

V0.2
----

*7 Apr 2017*

-  New dimensionality reduction method (WDA)
-  Efficient method emd2 returns only tarnsport (in paralell if several
   histograms given)

0.1.11
------

*5 Jan 2017*

-  Add sphinx gallery for better documentation
-  Small efficiency tweak in sinkhorn
-  Add simple tic() toc() functions for timing

0.1.10
------

*7 Nov 2016* \* numerical stabilization for sinkhorn (log domain and
epsilon scaling)

0.1.9
-----

*4 Nov 2016*

-  Update classes and examples for domain adaptation
-  Joint OT matrix and mapping estimation

0.1.7
-----

*31 Oct 2016*

-  Original Domain adaptation classes

0.1.3
-----

-  pipy works

First pre-release
-----------------

*28 Oct 2016*

It provides the following solvers: \* OT solver for the linear program/
Earth Movers Distance. \* Entropic regularization OT solver with
Sinkhorn Knopp Algorithm. \* Bregman projections for Wasserstein
barycenter [3] and unmixing. \* Optimal transport for domain adaptation
with group lasso regularization \* Conditional gradient and Generalized
conditional gradient for regularized OT.

Some demonstrations (both in Python and Jupyter Notebook format) are
available in the examples folder.
