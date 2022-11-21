# Releases

## 0.8.3dev

#### New features

- Added Generalized Wasserstein Barycenter solver + example (PR #372), fixed graphical details on the example (PR #376)
- Added Free Support Sinkhorn Barycenter + example (PR #387)

#### Closed issues


- Fixed an issue with the documentation gallery sections (PR #395)
- Fixed an issue where sinkhorn divergence did not have a gradients (Issue #393, PR #394)
- Fixed an issue where we could not ask TorchBackend to place a random tensor on GPU
  (Issue #371, PR #373)
- Fixed an issue where Sinkhorn solver assumed a symmetric cost matrix (Issue #374, PR #375)
- Fixed an issue where hitting iteration limits would be reported to stderr by std::cerr regardless of Python's stderr stream status (PR #377)
- Fixed an issue where the metric argument in ot.dist did not allow a callable parameter (Issue #378, PR #379)
- Fixed an issue where the max number of iterations in ot.emd was not allowed to go beyond 2^31 (PR #380)
- Fixed an issue where pointers would overflow in the EMD solver, returning an
incomplete transport plan above a certain size (slightly above 46k, its square being
roughly 2^31) (PR #381)
- Error raised when mass mismatch in emd2 (PR #386)
- Fixed an issue where a pytorch example would throw an error if executed on a GPU (Issue #389, PR #391)
- Added a work-around for scipy's bug, where you cannot compute the Hamming distance with a "None" weight attribute. (Issue #400, PR #402)
- Fixed an issue where the doc could not be built due to some changes in matplotlib's API (Issue #403, PR #402)
- Replaced Numpy C Compiler with Setuptools C Compiler due to deprecation issues (Issue #408, PR #409)


## 0.8.2

This releases introduces several new notable features. The less important
but most exiting one being that we now have a logo for the toolbox (color
and dark background) :

![](https://pythonot.github.io/master/_images/logo.svg)![](https://pythonot.github.io/master/_static/logo_dark.svg)

This logo is generated using with matplotlib and using the solution of an OT
problem provided by POT (with `ot.emd`). Generating the logo can be done with a
simple python script also provided in the [documentation gallery](https://pythonot.github.io/auto_examples/others/plot_logo.html#sphx-glr-auto-examples-others-plot-logo-py).

New OT solvers include [Weak
OT](https://pythonot.github.io/gen_modules/ot.weak.html#ot.weak.weak_optimal_transport)
 and [OT with factored
coupling](https://pythonot.github.io/gen_modules/ot.factored.html#ot.factored.factored_optimal_transport)
that can be used on large datasets. The [Majorization Minimization](https://pythonot.github.io/gen_modules/ot.unbalanced.html?highlight=mm_#ot.unbalanced.mm_unbalanced) solvers for
non-regularized Unbalanced OT are now also available. We also now provide an
implementation of [GW and FGW unmixing](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.gromov_wasserstein_linear_unmixing) and [dictionary learning](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.gromov_wasserstein_dictionary_learning). It is now
possible to use autodiff to solve entropic an quadratic regularized OT in the
dual for full or stochastic optimization thanks to the new functions to compute
the dual loss for [entropic](https://pythonot.github.io/gen_modules/ot.stochastic.html#ot.stochastic.loss_dual_entropic) and [quadratic](https://pythonot.github.io/gen_modules/ot.stochastic.html#ot.stochastic.loss_dual_quadratic) regularized OT and reconstruct the [OT
plan](https://pythonot.github.io/gen_modules/ot.stochastic.html#ot.stochastic.plan_dual_entropic) on part or all of the data. They can be used for instance to solve OT
problems with stochastic gradient or for estimating the [dual potentials as
neural networks](https://pythonot.github.io/auto_examples/backends/plot_stoch_continuous_ot_pytorch.html#sphx-glr-auto-examples-backends-plot-stoch-continuous-ot-pytorch-py).

On the backend front, we now have backend compatible functions and classes in
the domain adaptation [`ot.da`](https://pythonot.github.io/gen_modules/ot.da.html#module-ot.da) and unbalanced OT [`ot.unbalanced`](https://pythonot.github.io/gen_modules/ot.unbalanced.html) modules. This
means that the DA classes can be used on tensors from all compatible backends.
The [free support Wasserstein barycenter](https://pythonot.github.io/gen_modules/ot.lp.html?highlight=free%20support#ot.lp.free_support_barycenter) solver is now also backend compatible.

Finally we have worked on the documentation to provide an update of existing
examples in the gallery and and several new examples including [GW dictionary
learning](https://pythonot.github.io/auto_examples/gromov/plot_gromov_wasserstein_dictionary_learning.html#sphx-glr-auto-examples-gromov-plot-gromov-wasserstein-dictionary-learning-py)
[weak Optimal
Transport](https://pythonot.github.io/auto_examples/others/plot_WeakOT_VS_OT.html#sphx-glr-auto-examples-others-plot-weakot-vs-ot-py),
[NN based dual potentials
estimation](https://pythonot.github.io/auto_examples/backends/plot_stoch_continuous_ot_pytorch.html#sphx-glr-auto-examples-backends-plot-stoch-continuous-ot-pytorch-py)
and [Factored coupling OT](https://pythonot.github.io/auto_examples/others/plot_factored_coupling.html#sphx-glr-auto-examples-others-plot-factored-coupling-py).
.

#### New features

- Remove deprecated `ot.gpu` submodule (PR #361)
- Update examples in the gallery (PR #359)
- Add stochastic loss and OT plan computation for regularized OT and 
  backend examples(PR #360)
- Implementation of factored OT with emd and sinkhorn (PR #358)
- A brand new logo for POT (PR #357)
- Better list of related examples in quick start guide with `minigallery` (PR #334)
- Add optional log-domain Sinkhorn implementation in WDA to support smaller values
  of the regularization parameter (PR #336)
- Backend implementation for `ot.lp.free_support_barycenter` (PR #340)
- Add weak OT solver + example  (PR #341)
- Add backend support for Domain Adaptation and Unbalanced solvers (PR #343)
- Add (F)GW linear dictionary learning solvers + example  (PR #319)
- Add links to related PR and Issues in the doc release page (PR #350)
- Add new minimization-maximization algorithms for solving exact Unbalanced OT + example (PR #362)

#### Closed issues

- Fix mass gradient of `ot.emd2` and `ot.gromov_wasserstein2` so that they are 
  centered (Issue #364, PR #363)
- Fix bug in instantiating an `autograd` function `ValFunction` (Issue #337, 
  PR #338)
- Fix POT ABI compatibility with old and new numpy (Issue #346, PR #349)
- Warning when feeding integer cost matrix to EMD solver resulting in an integer transport plan (Issue #345, PR #343)
- Fix bug where gromov_wasserstein2 does not perform backpropagation with CUDA
  tensors (Issue #351, PR #352)


## 0.8.1.0
*December 2021*

This is a bug fix release that will remove the `benchmarks` module form the
installation and correct the documentation generation.

#### Closed issues

- Bug in documentation generation (tag VS master push, PR #332)
- Remove installation of the benchmarks in global namespace (Issue #331, PR #333)


## 0.8.1
*December 2021*

This release fixes several bugs and introduces two new backends: Cupy
and Tensorflow. Note that the tensorflow backend will work only when tensorflow
has enabled the Numpy behavior (for transpose that is not by default in
tensorflow). We also introduce a simple benchmark on CPU GPU for the sinkhorn
solver that will be provided in the
[backend](https://pythonot.github.io/gen_modules/ot.backend.html) documentation.

This release also brings a few changes in dependencies and compatibility. First
we removed tests for Python 3.6 that will not be updated in the future.
Also note that POT now depends on Numpy (>= 1.20) because a recent change in ABI is making the
wheels non-compatible with older numpy versions. If you really need an older
numpy POT will work with no problems but you will need to build it from source.

As always we want to that the contributors who helped make POT better (and bug free).

#### New features

- New benchmark for sinkhorn solver on CPU/GPU and between backends (PR #316)
- New tensorflow backend (PR #316)
- New Cupy backend (PR #315)
- Documentation always up-to-date with README, RELEASES, CONTRIBUTING and
  CODE_OF_CONDUCT files (PR #316, PR #322).

#### Closed issues

- Fix bug in older Numpy ABI (<1.20) (Issue #308, PR #326)
- Fix bug  in `ot.dist` function when non euclidean distance (Issue #305, PR #306)
- Fix gradient scaling for functions using `nx.set_gradients` (Issue #309, 
  PR #310)
- Fix bug in generalized Conditional gradient solver and SinkhornL1L2 
  (Issue #311, PR #313)
- Fix log error in `gromov_barycenters` (Issue #317, PR #3018)

## 0.8.0
*November 2021*

This new stable release introduces several important features. 

First we now have
an OpenMP compatible exact ot solver in `ot.emd`. The OpenMP version is used
when the parameter `numThreads` is greater than one and can lead to nice
speedups on multi-core machines. 

Second we have introduced a backend mechanism that allows to use standard POT
function seamlessly on Numpy, Pytorch and Jax arrays. Other backends are coming
but right now POT can be used seamlessly for training neural networks in
Pytorch. Notably we propose the first differentiable computation of the exact OT
loss with `ot.emd2` (can be differentiated w.r.t. both cost matrix and sample
weights), but also for the classical Sinkhorn loss with `ot.sinkhorn2`, the
Wasserstein distance in 1D with `ot.wasserstein_1d`, sliced Wasserstein with
`ot.sliced_wasserstein_distance` and Gromov-Wasserstein with `ot.gromov_wasserstein2`. Examples of how
this new feature can be used are now available in the documentation where the
Pytorch backend is used to estimate a [minimal Wasserstein
estimator](https://PythonOT.github.io/auto_examples/backends/plot_unmix_optim_torch.html),
a [Generative Network
(GAN)](https://PythonOT.github.io/auto_examples/backends/plot_wass2_gan_torch.html),
for a  [sliced Wasserstein gradient
flow](https://PythonOT.github.io/auto_examples/backends/plot_sliced_wass_grad_flow_pytorch.html)
and [optimizing the Gromov-Wassersein distance](https://PythonOT.github.io/auto_examples/backends/plot_optim_gromov_pytorch.html). Note that the Jax backend is still in early development and quite
slow at the moment, we strongly recommend for Jax users to use the [OTT
toolbox](https://github.com/google-research/ott)  when possible.  
 As a result of this new feature,
 the old `ot.gpu` submodule is now deprecated since GPU
implementations can be done using GPU arrays on the torch backends.

Other novel features include implementation for [Sampled Gromov Wasserstein and
Pointwise Gromov
Wasserstein](https://PythonOT.github.io/auto_examples/gromov/plot_gromov.html#compute-gw-with-a-scalable-stochastic-method-with-any-loss-function),
Sinkhorn in log space with `method='sinkhorn_log'`, [Projection Robust
Wasserstein](https://PythonOT.github.io/gen_modules/ot.dr.html?highlight=robust#ot.dr.projection_robust_wasserstein),
ans [deviased Sinkorn barycenters](https://PythonOT.github.ioauto_examples/barycenters/plot_debiased_barycenter.html).

This release will also simplify the installation process. We have now a
`pyproject.toml` that defines the build dependency and POT should now build even
when cython is not installed yet. Also we now provide pe-compiled wheels for
linux `aarch64` that is used on Raspberry PI and android phones and for MacOS on
ARM processors.


Finally POT was accepted for publication in the Journal of Machine Learning
Research (JMLR) open source software track and we ask the POT users to cite [this
paper](https://www.jmlr.org/papers/v22/20-451.html) from now on. The documentation has been improved in particular by adding a
"Why OT?" section to the quick start guide and several new examples illustrating
the new features. The documentation now has two version : the stable version 
[https://pythonot.github.io/](https://pythonot.github.io/)
corresponding to the last release and the master version [https://pythonot.github.io/master](https://pythonot.github.io/master) that corresponds to the
current master branch on GitHub.


As usual, we want to thank all the POT contributors (now 37 people have
contributed to the toolbox). But for this release we thank in particular Nathan
Cassereau and Kamel Guerda from the AI support team at
[IDRIS](http://www.idris.fr/) for their support to the development of the
backend and OpenMP implementations. 


#### New features

- OpenMP support for exact OT solvers (PR #260)
- Backend for running POT in numpy/torch + exact solver (PR #249)
- Backend implementation of most functions in `ot.bregman` (PR #280)
- Backend implementation of most functions in `ot.optim` (PR #282)
- Backend implementation of most functions in `ot.gromov` (PR #294, PR #302)
- Test for arrays of different type and device (CPU/GPU) (PR #304, #303)
- Implementation of Sinkhorn in log space with `method='sinkhorn_log'` (PR #290)
- Implementation of regularization path for L2 Unbalanced OT (PR #274)
- Implementation of Projection Robust Wasserstein (PR #267)
- Implementation of Debiased Sinkhorn Barycenters (PR #291)
- Implementation of Sampled Gromov Wasserstein and Pointwise Gromov Wasserstein
  (PR #275)
- Add `pyproject.toml` and build POT without installing cython first (PR #293)
- Lazy implementation in log space for sinkhorn on samples (PR #259)
- Documentation cleanup (PR #298)
- Two up-to-date documentations [for stable
  release](https://PythonOT.github.io/) and for [master branch](https://pythonot.github.io/master/).
- Building wheels on ARM for Raspberry PI and smartphones (PR #238)
- Update build wheels to new version and new pythons (PR #236, #253)
- Implementation of sliced Wasserstein distance (Issue #202, PR #203)
- Add minimal build to CI and perform pep8 test separately (PR #210)
- Speedup of tests and return run time (PR #262)
- Add "Why OT" discussion to the documentation (PR #220)
- New introductory example to discrete OT in the documentation (PR #191)
- Add templates for Issues/PR on Github (PR#181)

#### Closed issues

- Debug Memory leak in GAN example (#254)
- DEbug GPU bug (Issue #284, #287, PR #288)
- set_gradients method for JAX backend (PR #278)
- Quicker GAN example for CircleCI build (PR #258)
- Better formatting in Readme (PR #234)
- Debug CI tests (PR #240, #241, #242)
- Bug in Partial OT solver dummy points (PR #215)
- Bug when Armijo linesearch  (Issue #184, #198, #281, PR #189, #199, #286)
- Bug Barycenter Sinkhorn (Issue 134, PR #195)
- Infeasible solution in exact OT (Issues #126,#93, PR #217)
- Doc for SUpport Barycenters (Issue #200, PR #201)
- Fix labels transport in BaseTransport (Issue #207, PR #208)
- Bug in `emd_1d`, non respected bounds (Issue #169, PR #170)
- Removed Python 2.7 support and update codecov file (PR #178)
- Add normalization for WDA and test it (PR #172, #296)
- Cleanup code for new version of `flake8` (PR #176)
- Fixed requirements in `setup.py` (PR #174)
- Removed specific MacOS flags (PR #175)


## 0.7.0
*May 2020*

This is the new stable release for POT. We made a lot of changes in the
documentation and added several new features such as Partial OT, Unbalanced and
Multi Sources OT Domain Adaptation and several bug fixes. One important change
is that we have created the GitHub organization
[PythonOT](https://github.com/PythonOT) that now owns the main POT repository
[https://github.com/PythonOT/POT](https://github.com/PythonOT/POT) and the
repository for the new documentation is now hosted at
[https://PythonOT.github.io/](https://PythonOT.github.io/).

This is the first release where the Python 2.7 tests have been removed. Most of
the toolbox should still work but we do not offer support for Python 2.7 and
will close related Issues. 

A lot of changes have been done to the documentation that is now hosted on
[https://PythonOT.github.io/](https://PythonOT.github.io/) instead of
readthedocs. It was a hard choice but readthedocs did not allow us to run
sphinx-gallery to update our beautiful examples and it was a huge amount of work
to maintain. The documentation is now automatically compiled and updated on
merge. We also removed the notebooks from the repository for space reason and
also because they are all available in the [example
gallery](https://pythonot.github.io/auto_examples/index.html). Note that now the
output of the documentation build for each commit in the PR is available to
check that the doc builds correctly before merging which was not possible with
readthedocs.

The CI framework has also been changed with a move from Travis to Github Action
which allows to get faster tests on Windows, MacOS and Linux. We also now report
our coverage on [Codecov.io](https://codecov.io/gh/PythonOT/POT) and we have a
reasonable 92% coverage. We also now generate wheels for a number of OS and
Python versions at each merge in the master branch. They are available as
outputs of this
[action](https://github.com/PythonOT/POT/actions?query=workflow%3A%22Build+dist+and+wheels%22).
This will allow simpler multi-platform releases from now on.

In terms of new features we now have [OTDA Classes for unbalanced
OT](https://pythonot.github.io/gen_modules/ot.da.html#ot.da.UnbalancedSinkhornTransport),
a new Domain adaptation class form [multi domain problems
(JCPOT)](https://pythonot.github.io/auto_examples/domain-adaptation/plot_otda_jcpot.html#sphx-glr-auto-examples-domain-adaptation-plot-otda-jcpot-py),
and several solvers to solve the [Partial Optimal
Transport](https://pythonot.github.io/auto_examples/unbalanced-partial/plot_partial_wass_and_gromov.html#sphx-glr-auto-examples-unbalanced-partial-plot-partial-wass-and-gromov-py)
problems.

This release is also the moment to thank all the POT contributors (old and new)
for helping making POT such a nice toolbox. A lot of changes (also in the API)
are coming for the next versions. 


#### Features

- New documentation on [https://PythonOT.github.io/](https://PythonOT.github.io/) (PR #160, PR #143, PR #144)
- Documentation build on CircleCI with sphinx-gallery (PR #145,PR #146, #155)
- Run sphinx gallery in CI (PR #146)
- Remove notebooks from repo because available in doc (PR #156)
- Build wheels in CI (#157)
- Move from travis to GitHub Action for Windows, MacOS and Linux (PR #148, PR #150)
- Partial Optimal Transport (PR#141 and PR #142)
- Laplace regularized OTDA (PR #140)
- Multi source DA with target shift (PR #137)
- Screenkhorn algorithm (PR #121)

#### Closed issues

- Add JMLR paper to the readme and Mathieu Blondel to the Acknoledgments (PR
  #231, #232)
- Bug in Unbalanced OT example (Issue #127)
- Clean Cython output when calling setup.py clean (Issue #122)
- Various Macosx compilation problems (Issue #113, Issue #118, PR#130)
- EMD dimension mismatch (Issue #114, Fixed in PR #116)
- 2D barycenter bug for non square images (Issue #124, fixed in PR #132)
- Bad value in EMD 1D (Issue #138, fixed in PR #139)
- Log bugs for Gromov-Wassertein solver (Issue #107, fixed in PR #108)
- Weight issues in barycenter function (PR #106)

## 0.6.0 
*July 2019*

This is the first official stable release of POT and this means a jump to 0.6! 
The library has been used in
the wild for a while now and we have reached a state where a lot of fundamental
OT solvers are available and tested. It has been quite stable in the last months
but kept the beta flag in its Pypi classifiers until now. 

Note that this release will be the last one supporting officially Python 2.7 (See
https://python3statement.org/ for more reasons). For next release we will keep
the travis tests for Python 2 but will make them non necessary for merge in 2020.

The features are never complete in a toolbox designed for solving mathematical
problems and research but with the new contributions we now implement algorithms
and solvers from 24 scientific papers (listed in the README.md file). New
features include a direct implementation of the [empirical Sinkhorn
divergence](https://pot.readthedocs.io/en/latest/all.html#ot.bregman.empirical_sinkhorn_divergence),
a new efficient (Cython implementation) solver for [EMD in
1D](https://pot.readthedocs.io/en/latest/all.html#ot.lp.emd_1d) and
corresponding [Wasserstein
1D](https://pot.readthedocs.io/en/latest/all.html#ot.lp.wasserstein_1d). We now
also have implementations for [Unbalanced
OT](https://github.com/rflamary/POT/blob/master/notebooks/plot_UOT_1D.ipynb) and
a solver for [Unbalanced OT
barycenters](https://github.com/rflamary/POT/blob/master/notebooks/plot_UOT_barycenter_1D.ipynb).
A new variant of Gromov-Wasserstein divergence called [Fused
Gromov-Wasserstein](https://pot.readthedocs.io/en/latest/all.html?highlight=fused_#ot.gromov.fused_gromov_wasserstein)
has been also contributed with exemples of use on [structured
data](https://github.com/rflamary/POT/blob/master/notebooks/plot_fgw.ipynb) and
computing [barycenters of labeld
graphs](https://github.com/rflamary/POT/blob/master/notebooks/plot_barycenter_fgw.ipynb).


A lot of work has been done on the documentation with several new
examples corresponding to the new features and a lot of corrections for the
docstrings. But the most visible change is a new 
[quick start guide](https://pot.readthedocs.io/en/latest/quickstart.html) for
POT that gives several pointers about which function or classes allow to solve which
specific OT problem. When possible a link is provided to relevant examples.

We will also provide with this release some pre-compiled Python wheels for Linux
64bit on
github and pip. This will simplify the install process that before required a C
compiler and numpy/cython already installed.

Finally we would like to acknowledge and thank the numerous contributors of POT
that has helped in the past build the foundation and are still contributing to
bring new features and solvers to the library.



#### Features

* Add compiled manylinux 64bits wheels to pip releases (PR #91)
* Add quick start guide (PR #88)
* Make doctest work on travis (PR #90)
* Update documentation (PR #79, PR #84)
* Solver for EMD in 1D (PR #89)
* Solvers for regularized unbalanced OT (PR #87, PR#99)
* Solver for Fused Gromov-Wasserstein (PR #86)
* Add empirical Sinkhorn and empirical Sinkhorn divergences (PR #80)


#### Closed issues

- Issue #59 fail when using "pip install POT" (new details in doc+ hopefully
  wheels)
- Issue #85 Cannot run gpu modules
- Issue #75 Greenkhorn do not return log (solved in PR #76)
- Issue #82 Gromov-Wasserstein fails when the cost matrices are slightly different
- Issue #72 Macosx build problem


## 0.5.0 
*Sep 2018*

POT is 2 years old! This release brings numerous new features to the 
toolbox as listed below but also several bug correction.

Among the new features, we can highlight a [non-regularized Gromov-Wasserstein 
solver](https://github.com/rflamary/POT/blob/master/notebooks/plot_gromov.ipynb), 
a new [greedy variant of sinkhorn](https://pot.readthedocs.io/en/latest/all.html#ot.bregman.greenkhorn),  
[non-regularized](https://pot.readthedocs.io/en/latest/all.html#ot.lp.barycenter), 
[convolutional (2D)](https://github.com/rflamary/POT/blob/master/notebooks/plot_convolutional_barycenter.ipynb)
and [free support](https://github.com/rflamary/POT/blob/master/notebooks/plot_free_support_barycenter.ipynb)
 Wasserstein barycenters and [smooth](https://github.com/rflamary/POT/blob/prV0.5/notebooks/plot_OT_1D_smooth.ipynb) 
 and [stochastic](https://pot.readthedocs.io/en/latest/all.html#ot.stochastic.sgd_entropic_regularization) 
implementation of entropic OT.

POT 0.5 also comes with a rewriting of ot.gpu using the cupy framework instead of 
the unmaintained cudamat. Note that while we tried to keed changes to the 
minimum, the OTDA classes were deprecated. If you are happy with the cudamat 
implementation, we recommend you stay with stable release 0.4 for now.

The code quality has also improved with 92% code coverage in tests that is now 
printed to the log in the Travis builds. The documentation has also been 
greatly improved with new modules and examples/notebooks.

This new release is so full of new stuff and corrections thanks to the old
and new POT contributors (you can see the list in the [readme](https://github.com/rflamary/POT/blob/master/README.md)).

#### Features

* Add non regularized Gromov-Wasserstein solver  (PR #41)
* Linear OT mapping between empirical distributions and 90\% test coverage (PR #42)
* Add log parameter in class EMDTransport and SinkhornLpL1Transport (PR #44)
* Add Markdown format for Pipy (PR #45)
* Test for Python 3.5 and 3.6 on Travis (PR #46)
* Non regularized Wasserstein barycenter with scipy linear solver and/or cvxopt (PR #47)
* Rename dataset functions to be more sklearn compliant (PR #49)
* Smooth and sparse Optimal transport implementation with entropic and quadratic regularization (PR #50)
* Stochastic OT in the dual and semi-dual (PR #52 and PR #62)
* Free support barycenters (PR #56)
* Speed-up Sinkhorn function (PR #57 and PR #58)
* Add convolutional Wassersein barycenters for 2D images (PR #64) 
* Add Greedy Sinkhorn variant (Greenkhorn) (PR #66)
* Big ot.gpu update with cupy implementation (instead of un-maintained cudamat) (PR #67)

#### Deprecation

Deprecated OTDA Classes were removed from ot.da and ot.gpu for version 0.5 
(PR #48 and PR #67). The deprecation message has been for a year here since 
0.4 and it is time to pull the plug.

#### Closed issues

* Issue #35 : remove import plot from ot/__init__.py (See PR #41)
* Issue #43 : Unusable parameter log for EMDTransport (See PR #44)
* Issue #55 : UnicodeDecodeError: 'ascii' while installing with pip 


## 0.4 
*15 Sep 2017*

This release contains a lot of contribution from new contributors.


#### Features

* Automatic notebooks and doc update (PR #27)
* Add gromov Wasserstein solver and Gromov Barycenters (PR #23)
* emd and emd2 can now return dual variables and have max_iter (PR #29 and PR #25) 
* New domain adaptation classes compatible with scikit-learn (PR #22)
* Proper tests with pytest on travis (PR #19)
* PEP 8 tests (PR #13)

#### Closed issues

* emd convergence problem du to fixed max iterations (#24) 
* Semi supervised DA error (#26)

## 0.3.1
*11 Jul 2017*

* Correct bug in emd on windows

## 0.3 
*7 Jul 2017*

* emd* and sinkhorn* are now performed in parallel for multiple target distributions
* emd and sinkhorn are for OT matrix computation
* emd2 and sinkhorn2 are for OT loss computation
* new notebooks for emd computation and Wasserstein Discriminant Analysis
* relocate notebooks
* update documentation
* clean_zeros(a,b,M) for removimg zeros in sparse distributions
* GPU implementations for sinkhorn and group lasso regularization


## V0.2 
*7 Apr 2017*

* New dimensionality reduction method (WDA)
* Efficient method emd2 returns only tarnsport (in paralell if several histograms given)



## 0.1.11 
*5 Jan 2017*

* Add sphinx gallery for better documentation
* Small efficiency tweak in sinkhorn
* Add simple tic() toc() functions for timing


## 0.1.10 
*7 Nov 2016*
* numerical stabilization for sinkhorn (log domain and epsilon scaling)

## 0.1.9
*4 Nov 2016*

* Update classes and examples for domain adaptation
* Joint OT matrix and mapping estimation

## 0.1.7
*31 Oct 2016*

* Original Domain adaptation classes

## 0.1.3

* pipy works

## First pre-release
*28 Oct 2016*

It provides the following solvers:
* OT solver for the linear program/ Earth Movers Distance.
* Entropic regularization OT solver  with Sinkhorn Knopp Algorithm.
* Bregman projections for Wasserstein barycenter [3] and unmixing.
* Optimal transport for domain adaptation with group lasso regularization
* Conditional gradient and Generalized conditional gradient for regularized OT.

Some demonstrations (both in Python and Jupyter Notebook format) are available in the examples folder.
