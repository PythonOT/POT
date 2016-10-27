# POT: Python Optimal Transport library
Python Optimal Transport library

This Python library is an open source implementation of several functions that allow to solve optimal transport problems in Python.

It provides the following solvers:
* Linear program (LP) OT solver/ Earth Movers Distance (using code from Antoine Rolet and Nicolas Bonneel [1]).
* Entropic regularization OT solver  with Sinkhorn Knopp Algorithm [2].
* Bregman projections for Wasserstein barycenter [3] and unmixing [4].
* Optimal transport for domain adaptation with group lasso regularization [5]
* Conditional gradient and Generalized conditional gradient for regularized OT [5].

Some demonstrations (both in Python and Jupyter Notebook Format) are available in the examples folder.


## Installation


## Examples

## Acknowledgements

The main developers of this library are:
* Rémi Flamary
* Nicolas Courty

This toolbox benefit a lot from Open Source research and we would like to thank the Following persons for providing some code (in various languages):

* Gabriel Peyré (Wasserstein Barycenters in Matlab)
* Nicolas Bonneel ( C++ code for EMD)
* Antoine Rolet ( Mex file fro EMD )
* Marco Cuturi (Sinkhorn Knopp in Matlab/Cuda)

## References

[1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W. (2011, December). Displacement interpolation using Lagrangian mass transport. In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p. 158). ACM.

[2] Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in Neural Information Processing Systems (pp. 2292-2300).

[3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015). Iterative Bregman projections for regularized transportation problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

[4] S. Nakhostin, N. Courty, R. Flamary, D. Tuia, T. Corpetti, Supervised planetary unmixing with optimal transport, Whorkshop on Hyperspectral Image and Signal Processing : Evolution in Remote Sensing (WHISPERS), 2016.

[5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport for Domain Adaptation," in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

[6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.
