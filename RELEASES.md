# POT Releases


## 0.5.0 Year 2
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


## 0.4 Community edition
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

## 0.3 Summer release
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



## V0.1.11 New years resolution
*5 Jan 2017*

* Add sphinx gallery for better documentation
* Small efficiency tweak in sinkhorn
* Add simple tic() toc() functions for timing


## V0.1.10 
*7 Nov 2016*
* numerical stabilization for sinkhorn (log domain and epsilon scaling)

## V0.1.9 DA classes and mapping
*4 Nov 2016*

* Update classes and examples for domain adaptation
* Joint OT matrix and mapping estimation

## V0.1.7
*31 Oct 2016*

* Original Domain adaptation classes



## PyPI version 0.1.3

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
