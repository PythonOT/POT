# POT: Python Optimal Transport

This open source Python library provide several solvers for optimization problems related to Optimal Transport for signal, image processing and machine learning.

It provides the following solvers:
* OT solver for the linear program/ Earth Movers Distance [1].
* Entropic regularization OT solver  with Sinkhorn Knopp Algorithm [2].
* Bregman projections for Wasserstein barycenter [3] and unmixing [4].
* Optimal transport for domain adaptation with group lasso regularization [5]
* Conditional gradient [6] and Generalized conditional gradient for regularized OT [7].

Some demonstrations (both in Python and Jupyter Notebook format) are available in the examples folder.

## Installation

The Library has been tested on Linux and MacOSX. It requires a C++ compiler for using the EMD solver and rely on the following Python modules:

- Numpy (>=1.11)
- Scipy (>=0.17)

To install the library, you can install it locally (after downloading it) on you machine using
```
python setup.py install --user
```

The toolbox is also available on PyPI with a possibly slightly older version. You can install it with:
```
pip install POT
```

After a correct installation, you should be able to import the module without errors:
```python
import ot
```

Note that for easier access the module is name ot instead of pot.

## Examples

The examples folder contain several examples and use case for the library.

 Here is a list of the Python notebook if you want a quick look:

* [1D optimal transport](examples/Demo_1D_OT.ipynb)
* [2D optimal transport on empirical distributions](examples/Demo_2D_OT_samples.ipynb)
* [1D Wasserstein barycenter](examples/Demo_1D_barycenter.ipynb)
* [OT with user provided regularization](examples/Demo_Optim_OTreg.ipynb)


## Acknowledgements

The contributors to this library are:
* [Rémi Flamary](http://remi.flamary.com/)
* [Nicolas Courty](http://people.irisa.fr/Nicolas.Courty/)
* [Laetitia Chapel](http://people.irisa.fr/Laetitia.Chapel/)

This toolbox benefit a lot from open source research and we would like to thank the following persons for providing some code (in various languages):

* [Gabriel Peyré](http://gpeyre.github.io/) (Wasserstein Barycenters in Matlab)
* [Nicolas Bonneel](http://liris.cnrs.fr/~nbonneel/) ( C++ code for EMD)
* [Antoine Rolet](https://arolet.github.io/) ( Mex file for EMD )
* [Marco Cuturi](http://marcocuturi.net/) (Sinkhorn Knopp in Matlab/Cuda)

## References

[1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W. (2011, December). Displacement interpolation using Lagrangian mass transport. In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p. 158). ACM.

[2] Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in Neural Information Processing Systems (pp. 2292-2300).

[3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015). Iterative Bregman projections for regularized transportation problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

[4] S. Nakhostin, N. Courty, R. Flamary, D. Tuia, T. Corpetti, Supervised planetary unmixing with optimal transport, Whorkshop on Hyperspectral Image and Signal Processing : Evolution in Remote Sensing (WHISPERS), 2016.

[5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport for Domain Adaptation," in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

[6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

[7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized conditional gradient: analysis of convergence and applications. arXiv preprint arXiv:1510.06567.
