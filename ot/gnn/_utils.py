# -*- coding: utf-8 -*-
"""
GNN layers utils
"""

# Author: Sonia Mazelet <sonia.mazelet@ens-paris-saclay.fr>
#         Rémi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import torch
from ..utils import dist
from ..gromov import fused_gromov_wasserstein2
from ..lp import emd2
from torch_geometric.utils import subgraph


def TFGW_template_initialization(
    n_tplt, n_tplt_nodes, n_features, feature_init_mean=0.0, feature_init_std=1.0
):
    """
    Initializes templates for the Template Fused Gromov Wasserstein layer.
    Returns the adjacency matrices and the features of the nodes of the templates.
    Adjacency matrices are initialized uniformly with values in :math:`[0,1]`.
    Node features are initialized following a normal distribution.

    Parameters
    ----------

      n_tplt: int
        Number of templates.
      n_tplt_nodes: int
        Number of nodes per template.
      n_features: int
        Number of features for the nodes.
      feature_init_mean: float, optional
        Mean of the random normal law to initialize the template features.
      feature_init_std: float, optional
        Standard deviation of the random normal law to initialize the template features.

    Returns
    ----------
      tplt_adjacencies: torch.Tensor, shape (n_templates, n_template_nodes, n_template_nodes)
           Adjacency matrices for the templates.
      tplt_features: torch.Tensor, shape (n_templates, n_template_nodes, n_features)
           Node features for each template.
      q: torch.Tensor, shape (n_templates, n_template_nodes)
           weight on the template nodes.
    """

    tplt_adjacencies = torch.rand((n_tplt, n_tplt_nodes, n_tplt_nodes))
    tplt_features = torch.Tensor(n_tplt, n_tplt_nodes, n_features)

    torch.nn.init.normal_(tplt_features, mean=feature_init_mean, std=feature_init_std)

    q = torch.zeros(n_tplt, n_tplt_nodes)

    tplt_adjacencies = 0.5 * (
        tplt_adjacencies + torch.transpose(tplt_adjacencies, 1, 2)
    )

    return tplt_adjacencies, tplt_features, q


def FGW_distance_to_templates(
    G_edges,
    tplt_adjacencies,
    G_features,
    tplt_features,
    tplt_weights,
    alpha=0.5,
    multi_alpha=False,
    batch=None,
):
    """
    Computes the FGW distances between a graph and templates.

    Parameters
    ----------
    G_edges : torch.Tensor, shape (n_edges, 2)
        Edge indices of the graph in the Pytorch Geometric format.
    tplt_adjacencies : list of torch.Tensor, shape (n_templates, n_template_nodes, n_templates_nodes)
        List of the adjacency matrices of the templates.
    G_features : torch.Tensor, shape (n_nodes, n_features)
        Graph node features.
    tplt_features : list of torch.Tensor, shape (n_templates, n_template_nodes, n_features)
        List of the node features of the templates.
    weights : torch.Tensor, shape (n_templates, n_template_nodes)
        Weights on the nodes of the templates.
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1).
        Weights features (alpha=0) and structure (alpha=1).
    multi_alpha: bool, optional
        If True, the alpha parameter is a vector of size n_templates.
    batch: torch.Tensor, optional
        Batch vector which assigns each node to its graph.

    Returns
    -------
    distances : torch.Tensor, shape (n_templates) if batch=None, else shape (n_graphs, n_templates).
        Vector of fused Gromov-Wasserstein distances between the graph and the templates.
    """

    if batch is None:
        n, n_feat = G_features.shape
        n_T, _, n_feat_T = tplt_features.shape

        weights_G = torch.ones(n) / n

        C = torch.sparse_coo_tensor(
            G_edges, torch.ones(len(G_edges[0])), size=(n, n)
        ).type(torch.float)
        C = C.to_dense()

        if not n_feat == n_feat_T:
            raise ValueError(
                "The templates and the graphs must have the same feature dimension."
            )

        distances = torch.zeros(n_T)

        for j in range(n_T):
            template_features = tplt_features[j].reshape(
                len(tplt_features[j]), n_feat_T
            )
            M = dist(G_features, template_features).type(torch.float)

            # if alpha is zero the emd distance is used
            if multi_alpha and torch.any(alpha > 0):
                embedding = fused_gromov_wasserstein2(
                    M,
                    C,
                    tplt_adjacencies[j],
                    weights_G,
                    tplt_weights[j],
                    alpha=alpha[j],
                    symmetric=True,
                    max_iter=50,
                )
            elif not multi_alpha and torch.all(alpha == 0):
                embedding = emd2(weights_G, tplt_weights[j], M, numItermax=50)
            elif not multi_alpha and alpha > 0:
                embedding = fused_gromov_wasserstein2(
                    M,
                    C,
                    tplt_adjacencies[j],
                    weights_G,
                    tplt_weights[j],
                    alpha=alpha,
                    symmetric=True,
                    max_iter=50,
                )
            else:
                embedding = emd2(weights_G, tplt_weights[j], M, numItermax=50)

            distances[j] = embedding

    else:
        n_T, _, n_feat_T = tplt_features.shape

        num_graphs = torch.max(batch) + 1
        distances = torch.zeros(num_graphs, n_T)

        # iterate over the graphs in the batch
        for i in range(num_graphs):
            nodes = torch.where(batch == i)[0]

            G_edges_i, _ = subgraph(nodes, edge_index=G_edges, relabel_nodes=True)
            G_features_i = G_features[nodes]

            n, n_feat = G_features_i.shape

            weights_G = torch.ones(n) / n

            n_edges = len(G_edges_i[0])

            C = torch.sparse_coo_tensor(
                G_edges_i, torch.ones(n_edges), size=(n, n)
            ).type(torch.float)
            C = C.to_dense()

            if not n_feat == n_feat_T:
                raise ValueError(
                    "The templates and the graphs must have the same feature dimension."
                )

            for j in range(n_T):
                template_features = tplt_features[j].reshape(
                    len(tplt_features[j]), n_feat_T
                )
                M = dist(G_features_i, template_features).type(torch.float)

                # if alpha is zero the emd distance is used
                if multi_alpha and torch.any(alpha > 0):
                    embedding = fused_gromov_wasserstein2(
                        M,
                        C,
                        tplt_adjacencies[j],
                        weights_G,
                        tplt_weights[j],
                        alpha=alpha[j],
                        symmetric=True,
                        max_iter=50,
                    )
                elif not multi_alpha and torch.all(alpha == 0):
                    embedding = emd2(weights_G, tplt_weights[j], M, numItermax=50)
                elif not multi_alpha and alpha > 0:
                    embedding = fused_gromov_wasserstein2(
                        M,
                        C,
                        tplt_adjacencies[j],
                        weights_G,
                        tplt_weights[j],
                        alpha=alpha,
                        symmetric=True,
                        max_iter=50,
                    )
                else:
                    embedding = emd2(weights_G, tplt_weights[j], M, numItermax=50)

                distances[i, j] = embedding

    return distances


def wasserstein_distance_to_templates(
    G_features, tplt_features, tplt_weights, batch=None
):
    """
    Computes the Wasserstein distances between a graph and graph templates.

    Parameters
    ----------
    G_features : torch.Tensor, shape (n_nodes, n_features)
        Node features of the graph.
    tplt_features : list of torch.Tensor, shape (n_templates, n_template_nodes, n_features)
        List of the node features of the templates.
    weights : torch.Tensor, shape (n_templates, n_template_nodes)
        Weights on the nodes of the templates.
    batch: torch.Tensor, optional
        Batch vector which assigns each node to its graph.

    Returns
    -------
    distances : torch.Tensor, shape (n_templates) if batch=None, else shape (n_graphs, n_templates)
        Vector of Wasserstein distances between the graph and the templates.
    """

    if batch is None:
        n, n_feat = G_features.shape
        n_T, _, n_feat_T = tplt_features.shape

        weights_G = torch.ones(n) / n

        if not n_feat == n_feat_T:
            raise ValueError(
                "The templates and the graphs must have the same feature dimension."
            )

        distances = torch.zeros(n_T)

        for j in range(n_T):
            template_features = tplt_features[j].reshape(
                len(tplt_features[j]), n_feat_T
            )
            M = dist(G_features, template_features).type(torch.float)

            distances[j] = emd2(weights_G, tplt_weights[j], M, numItermax=50)

    else:
        n_T, _, n_feat_T = tplt_features.shape

        num_graphs = torch.max(batch) + 1
        distances = torch.zeros(num_graphs, n_T)

        # iterate over the graphs in the batch
        for i in range(num_graphs):
            nodes = torch.where(batch == i)[0]

            G_features_i = G_features[nodes]

            n, n_feat = G_features_i.shape

            weights_G = torch.ones(n) / n

            if not n_feat == n_feat_T:
                raise ValueError(
                    "The templates and the graphs must have the same feature dimension."
                )

            for j in range(n_T):
                template_features = tplt_features[j].reshape(
                    len(tplt_features[j]), n_feat_T
                )
                M = dist(G_features_i, template_features).type(torch.float)

                distances[i, j] = emd2(weights_G, tplt_weights[j], M, numItermax=50)

    return distances
