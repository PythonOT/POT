# -*- coding: utf-8 -*-
"""
Solvers related to Bregman projections for entropic regularized OT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#         Kilian Fatras <kilian.fatras@irisa.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Hicham Janati <hicham.janati100@gmail.com>
#         Mokhtar Z. Alaya <mokhtarzahdi.alaya@gmail.com>
#         Alexander Tong <alexander.tong@yale.edu>
#         Ievgen Redko <ievgen.redko@univ-st-etienne.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

from ._sinkhorn import (sinkhorn,
                        sinkhorn2,
                        sinkhorn_knopp,
                        sinkhorn_log,
                        greenkhorn,
                        sinkhorn_stabilized,
                        sinkhorn_epsilon_scaling)
                       
from ._barycenter import (barycenter,
                          barycenter_sinkhorn,
                          free_support_sinkhorn_barycenter,
                          barycenter_stabilized,
                          barycenter_debiased,
                          jcpot_barycenter)

from ._convolutional import (convolutional_barycenter2d,
                             convolutional_barycenter2d_debiased)

from ._empirical import (empirical_sinkhorn,
                         empirical_sinkhorn2,
                         empirical_sinkhorn_divergence)

from ._screenkhorn import (screenkhorn)


__all__ = ['sinkhorn', 'sinkhorn2', 'sinkhorn_knopp', 'sinkhorn_log',
           'greenkhorn', 'sinkhorn_stabilized', 'sinkhorn_epsilon_scaling',
           'barycenter', 'barycenter_sinkhorn','free_support_sinkhorn_barycenter',
           'barycenter_stabilized', 'barycenter_debiased', 'jcpot_barycenter',
           'convolutional_barycenter2d', 'convolutional_barycenter2d_debiased',
           'empirical_sinkhorn', 'empirical_sinkhorn2',
           'empirical_sinkhorn_divergence',
           'screenkhorn'
           ]
