# POT Releases

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
