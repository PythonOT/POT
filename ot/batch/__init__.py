# -*- coding: utf-8 -*-
"""
Batch operations for optimal transport.
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Paul Krzakala <paul.krzakala@gmail.com>
#         Thibaut Germain <thibaut.germain.pro@gmail.com>
#
# License: MIT License

from ._linear import (
    solve_batch,
    solve_sample_batch,
    dist_batch,
    loss_linear_batch,
    loss_linear_samples_batch,
)
from ._quadratic import (
    solve_gromov_batch,
    loss_quadratic_batch,
    loss_quadratic_samples_batch,
    tensor_batch,
)
from ._utils import (
    bregman_log_projection_batch,
    bregman_projection_batch,
    entropy_batch,
    proximal_bregman_log_plan_batch,
)

__all__ = [
    "solve_batch",
    "solve_sample_batch",
    "solve_gromov_batch",
    "dist_batch",
    "loss_linear_batch",
    "loss_linear_samples_batch",
    "bregman_log_projection_batch",
    "bregman_projection_batch",
    "entropy_batch",
    "loss_quadratic_batch",
    "loss_quadratic_samples_batch",
    "tensor_batch",
    "proximal_bregman_log_plan_batch",
]
