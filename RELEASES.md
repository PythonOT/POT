# Releases

## 0.7.0
*May 2020*

This is the new stable release for POT. We made a lot of changes in the documentation and added several new features such as Partial OT, Unbalanced and Multi Sources OT Domain Adaptation and several bug fixes. One important change is that we have created the GitHub organization [PythonOT](https://github.com/PythonOT) that now owns the main POT repository [https://github.com/PythonOT/POT](https://github.com/PythonOT/POT) and the repository for the new documentation is now hosted at [https://PythonOT.github.io/](https://PythonOT.github.io/).

This is the first release where the Python 2.7 tests have been removed. Most of the toolbox should still work but we do not offer support for Python 2.7 and will close related Issues. 

A lot of changes have been done to the documentation that is now hosted on [https://PythonOT.github.io/](https://PythonOT.github.io/) instead of readthedocs. It was a hard choice but readthedocs did not allow us to run sphinx-gallery to update our beautiful examples and it was a huge amount of work to maintain. The documentation is now automatically compiled and updated on merge. We also removed the notebooks from the repository for space reason and also because they are all available in the [example gallery](https://pythonot.github.io/auto_examples/index.html). Note that now the output of the documentation build for each commit in the PR is available to check that the doc builds correctly before merging which was not possible with readthedocs.

The CI framework has also been changed with a move from Travis to Github Action which allows to get faster tests on Windows, MacOS and Linux. We also now report our coverage on [Codecov.io](https://codecov.io/gh/PythonOT/POT) and we have a reasonable 92% coverage. We also now generate wheels for a number of OS and Python versions at each merge in the master branch. They are available as outputs of this [action](https://github.com/PythonOT/POT/actions?query=workflow%3A%22Build+dist+and+wheels%22). This will allow simpler multi-platform releases from now on.

In terms of new features we now have [OTDA Classes for unbalanced OT](https://pythonot.github.io/gen_modules/ot.da.html#ot.da.UnbalancedSinkhornTransport), a new Domain adaptation class form [multi domain problems (JCPOT)](https://pythonot.github.io/auto_examples/domain-adaptation/plot_otda_jcpot.html#sphx-glr-auto-examples-domain-adaptation-plot-otda-jcpot-py), and several solvers to solve the [Partial Optimal Transport](https://pythonot.github.io/auto_examples/unbalanced-partial/plot_partial_wass_and_gromov.html#sphx-glr-auto-examples-unbalanced-partial-plot-partial-wass-and-gromov-py) problems.

This release is also the moment to thank all the POT contributors (old and new) for helping making POT such a nice toolbox. A lot of changes (also in the API) are comming for the next versions. 


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
problems and research but with the new contributions we now implement algorithms and solvers 
from 24 scientific papers (listed in the README.md file). New features include a
direct implementation of the [empirical Sinkhorn divergence](https://pot.readthedocs.io/en/latest/all.html#ot.bregman.empirical_sinkhorn_divergence)
, a new efficient (Cython implementation) solver for [EMD in 1D](https://pot.readthedocs.io/en/latest/all.html#ot.lp.emd_1d)
and corresponding [Wasserstein
1D](https://pot.readthedocs.io/en/latest/all.html#ot.lp.wasserstein_1d). We now also
have implementations for [Unbalanced OT](https://github.com/rflamary/POT/blob/master/notebooks/plot_UOT_1D.ipynb) 
and a solver for [Unbalanced OT barycenters](https://github.com/rflamary/POT/blob/master/notebooks/plot_UOT_barycenter_1D.ipynb).
A new variant of Gromov-Wasserstein divergence called [Fused
Gromov-Wasserstein](https://pot.readthedocs.io/en/latest/all.html?highlight=fused_#ot.gromov.fused_gromov_wasserstein)
 has been also contributed with exemples of use on [structured data](https://github.com/rflamary/POT/blob/master/notebooks/plot_fgw.ipynb) 
and computing [barycenters of labeld graphs](https://github.com/rflamary/POT/blob/master/notebooks/plot_barycenter_fgw.ipynb).


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
