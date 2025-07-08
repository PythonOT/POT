# -*- coding: utf-8 -*-
"""
Solvers related to Gromov-Wasserstein problems.

"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Cedric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

# All submodules and packages
from ._utils import (
    init_matrix,
    tensor_product,
    gwloss,
    gwggrad,
    init_matrix_semirelaxed,
    semirelaxed_init_plan,
    update_barycenter_structure,
    update_barycenter_feature,
    div_between_product,
    div_to_product,
    fused_unbalanced_across_spaces_cost,
    uot_cost_matrix,
    uot_parameters_and_measures,
)

from ._gw import (
    gromov_wasserstein,
    gromov_wasserstein2,
    fused_gromov_wasserstein,
    fused_gromov_wasserstein2,
    solve_gromov_linesearch,
    gromov_barycenters,
    fgw_barycenters,
)

from ._bregman import (
    entropic_gromov_wasserstein,
    entropic_gromov_wasserstein2,
    BAPG_gromov_wasserstein,
    BAPG_gromov_wasserstein2,
    entropic_gromov_barycenters,
    entropic_fused_gromov_wasserstein,
    entropic_fused_gromov_wasserstein2,
    BAPG_fused_gromov_wasserstein,
    BAPG_fused_gromov_wasserstein2,
    entropic_fused_gromov_barycenters,
)

from ._estimators import (
    GW_distance_estimation,
    pointwise_gromov_wasserstein,
    sampled_gromov_wasserstein,
)

from ._semirelaxed import (
    semirelaxed_gromov_wasserstein,
    semirelaxed_gromov_wasserstein2,
    semirelaxed_fused_gromov_wasserstein,
    semirelaxed_fused_gromov_wasserstein2,
    solve_semirelaxed_gromov_linesearch,
    entropic_semirelaxed_gromov_wasserstein,
    entropic_semirelaxed_gromov_wasserstein2,
    entropic_semirelaxed_fused_gromov_wasserstein,
    entropic_semirelaxed_fused_gromov_wasserstein2,
    semirelaxed_gromov_barycenters,
    semirelaxed_fgw_barycenters,
)

from ._dictionary import (
    gromov_wasserstein_dictionary_learning,
    gromov_wasserstein_linear_unmixing,
    fused_gromov_wasserstein_dictionary_learning,
    fused_gromov_wasserstein_linear_unmixing,
)

from ._lowrank import lowrank_gromov_wasserstein_samples


from ._quantized import (
    quantized_fused_gromov_wasserstein_partitioned,
    get_graph_partition,
    get_graph_representants,
    format_partitioned_graph,
    quantized_fused_gromov_wasserstein,
    get_partition_and_representants_samples,
    format_partitioned_samples,
    quantized_fused_gromov_wasserstein_samples,
)

from ._unbalanced import (
    fused_unbalanced_gromov_wasserstein,
    fused_unbalanced_gromov_wasserstein2,
    unbalanced_co_optimal_transport,
    unbalanced_co_optimal_transport2,
    fused_unbalanced_across_spaces_divergence,
)

from ._partial import (
    partial_gromov_wasserstein,
    partial_gromov_wasserstein2,
    partial_fused_gromov_wasserstein,
    partial_fused_gromov_wasserstein2,
    solve_partial_gromov_linesearch,
    entropic_partial_gromov_wasserstein,
    entropic_partial_gromov_wasserstein2,
    entropic_partial_fused_gromov_wasserstein,
    entropic_partial_fused_gromov_wasserstein2,
)


__all__ = [
    "init_matrix",
    "tensor_product",
    "gwloss",
    "gwggrad",
    "init_matrix_semirelaxed",
    "semirelaxed_init_plan",
    "update_barycenter_structure",
    "update_barycenter_feature",
    "div_between_product",
    "div_to_product",
    "fused_unbalanced_across_spaces_cost",
    "uot_cost_matrix",
    "uot_parameters_and_measures",
    "gromov_wasserstein",
    "gromov_wasserstein2",
    "fused_gromov_wasserstein",
    "fused_gromov_wasserstein2",
    "solve_gromov_linesearch",
    "gromov_barycenters",
    "fgw_barycenters",
    "entropic_gromov_wasserstein",
    "entropic_gromov_wasserstein2",
    "BAPG_gromov_wasserstein",
    "BAPG_gromov_wasserstein2",
    "entropic_gromov_barycenters",
    "entropic_fused_gromov_wasserstein",
    "entropic_fused_gromov_wasserstein2",
    "BAPG_fused_gromov_wasserstein",
    "BAPG_fused_gromov_wasserstein2",
    "entropic_fused_gromov_barycenters",
    "GW_distance_estimation",
    "pointwise_gromov_wasserstein",
    "sampled_gromov_wasserstein",
    "semirelaxed_gromov_wasserstein",
    "semirelaxed_gromov_wasserstein2",
    "semirelaxed_fused_gromov_wasserstein",
    "semirelaxed_fused_gromov_wasserstein2",
    "solve_semirelaxed_gromov_linesearch",
    "entropic_semirelaxed_gromov_wasserstein",
    "entropic_semirelaxed_gromov_wasserstein2",
    "entropic_semirelaxed_fused_gromov_wasserstein",
    "entropic_semirelaxed_fused_gromov_wasserstein2",
    "semirelaxed_fgw_barycenters",
    "semirelaxed_gromov_barycenters",
    "gromov_wasserstein_dictionary_learning",
    "gromov_wasserstein_linear_unmixing",
    "fused_gromov_wasserstein_dictionary_learning",
    "fused_gromov_wasserstein_linear_unmixing",
    "lowrank_gromov_wasserstein_samples",
    "quantized_fused_gromov_wasserstein_partitioned",
    "get_graph_partition",
    "get_graph_representants",
    "format_partitioned_graph",
    "quantized_fused_gromov_wasserstein",
    "get_partition_and_representants_samples",
    "format_partitioned_samples",
    "quantized_fused_gromov_wasserstein_samples",
    "fused_unbalanced_gromov_wasserstein",
    "fused_unbalanced_gromov_wasserstein2",
    "unbalanced_co_optimal_transport",
    "unbalanced_co_optimal_transport2",
    "fused_unbalanced_across_spaces_divergence",
    "partial_gromov_wasserstein",
    "partial_gromov_wasserstein2",
    "partial_fused_gromov_wasserstein",
    "partial_fused_gromov_wasserstein2",
    "solve_partial_gromov_linesearch",
    "entropic_partial_gromov_wasserstein",
    "entropic_partial_gromov_wasserstein2",
    "entropic_partial_fused_gromov_wasserstein",
    "entropic_partial_fused_gromov_wasserstein2",
]
