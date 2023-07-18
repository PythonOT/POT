# -*- coding: utf-8 -*-
"""
GNN layers utils
"""

import torch
from ..utils import dist
from ..gromov import fused_gromov_wasserstein2, gromov_wasserstein2
from torch_geometric.utils import subgraph


def TFGW_template_initialisation(n_tplt, n_tplt_nodes, n_features, feature_init_mean=0., feature_init_std=1.):
    """"
    Initialises templates for the Template Fused Gromov Wasserstein layer.
    Returns the adjacency matrices and the features of the nodes of the templates.
    Adjacency matrics are intialised uniformly with values in :math:[0,1]
    Features of the nodes are intialised following a normal distribution.

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
      tplt_adjacencies: torch tensor, shape (n_templates, n_template_nodes, n_template_nodes)
           Adjancency matrices for the templates.
      tplt_features: torch tensor, shape (n_templates, n_template_nodes, n_features)
           Node features for each template.
      q0: weight on the template nodes.
    """

    tplt_adjacencies = torch.rand((n_tplt, n_tplt_nodes, n_tplt_nodes))
    tplt_features = torch.Tensor(n_tplt, n_tplt_nodes, n_features)

    torch.nn.init.normal_(tplt_features, mean=feature_init_mean, std=feature_init_std)

    q0 = torch.zeros(n_tplt, n_tplt_nodes)

    tplt_adjacencies = 0.5 * (tplt_adjacencies + torch.transpose(tplt_adjacencies, 1, 2))

    return tplt_adjacencies, tplt_features, q0


def distance_to_templates(G_edges, tplt_adjacencies, G_features, tplt_features, tplt_weights, alpha=0.5, multi_alpha=False, batch=None, fused=True):
    """
    Computes the FGW distances between a graph and graph templates.

    Parameters
    ----------
    G_edges : torch tensor, shape(n_edges, 2)
        Edge indexes of the graph in the Pytorch Geometric format.
    tplt_adjacencies : list of torch tensors, shape (n_templates, n_template_nodes, n_templates_nodes)
        List of the adjacency matrices of the templates.
    G_features : torch tensor, shape (n_nodes, n_features)
        Node features of the graph.
    tplt_features : list of torch tensors, shape (n_templates, n_template_nodes, n_features)
        List of the node features of the templates.
    weights : torch tensor, shape (n_templates, n_template_nodes)
        Weights on the nodes of the templates.
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1).
        Weights features (alpha=0) and structure (alpha=1).
    multi_alpha: bool, optional
        If True, the alpha parameter is a vector of size n_templates.
    batch: torch tensor
        Node level batch vector.
    fused: bool, optional
        If True, the fused Gromov-Wasserstein distance is computed.
        Else, the Wasserstein distance is computed.

    Returns
    -------
    distances : torch tensor, shape (n_templates)
        Vector of fused Gromov-Wasserstein distances between the graph and the templates.
    """

    if not batch is None:
        n_T, _, n_feat_T = tplt_features.shape

        num_graphs = torch.max(batch) + 1
        distances = torch.zeros(num_graphs, n_T)

        #iterate over the graphs in the batch
        for i in range(num_graphs):

            nodes = torch.where(batch == i)[0]

            G_edges_i, _ = subgraph(nodes, edge_index=G_edges, relabel_nodes=True)
            G_features_i = G_features[nodes]

            n, n_feat = G_features_i.shape

            weights_G = torch.ones(n) / n

            n_edges = len(G_edges_i[0])

            C = torch.sparse_coo_tensor(G_edges_i, torch.ones(n_edges), size=(n, n)).type(torch.float)
            C = C.to_dense()

            if not n_feat == n_feat_T:
                raise ValueError('The templates and the graphs must have the same feature dimension.')

            for j in range(n_T):

                if fused:

                    template_features = tplt_features[j].reshape(len(tplt_features[j]), n_feat_T)
                    M = dist(G_features_i, template_features).type(torch.float)

                    if multi_alpha:
                        embedding = fused_gromov_wasserstein2(M, C, tplt_adjacencies[j], weights_G, tplt_weights[j], alpha=alpha[j], symmetric=True, max_iter=50)
                    else:
                        embedding = fused_gromov_wasserstein2(M, C, tplt_adjacencies[j], weights_G, tplt_weights[j], alpha=alpha, symmetric=True, max_iter=50)

                else:

                    embedding = gromov_wasserstein2(C, tplt_adjacencies[j], weights_G, tplt_weights[j], max_iter=50)

                distances[i, j] = embedding

    else:

        n, n_feat = G_features.shape
        n_T, _, n_feat_T = tplt_features.shape

        weights_G = torch.ones(n) / n

        C = torch.sparse_coo_tensor(G_edges, torch.ones(len(G_edges[0])), size=(n, n)).type(torch.float)
        C = C.to_dense()

        if not n_feat == n_feat_T:
            raise ValueError('The templates and the graphs must have the same feature dimension.')

        distances = torch.zeros(n_T)

        for j in range(n_T):

            if fused:

                template_features = tplt_features[j].reshape(len(tplt_features[j]), n_feat_T)
                M = dist(G_features, template_features).type(torch.float)

                if multi_alpha:
                    embedding = fused_gromov_wasserstein2(M, C, tplt_adjacencies[j], weights_G, tplt_weights[j], alpha=alpha[j], symmetric=True, max_iter=100)
                else:
                    embedding = fused_gromov_wasserstein2(M, C, tplt_adjacencies[j], weights_G, tplt_weights[j], alpha=alpha, symmetric=True, max_iter=100)

            else:

                embedding = gromov_wasserstein2(C, tplt_adjacencies[j], weights_G, tplt_weights[j], max_iter=50)

            distances[j] = embedding

    return distances
