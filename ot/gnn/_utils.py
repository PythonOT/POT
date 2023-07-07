# -*- coding: utf-8 -*-
"""
GNN layers utils
"""

import torch
import torch.nn.functional as F
import numpy as np
from ..utils import dist
from ..gromov import fused_gromov_wasserstein2


def template_initialisation(n_templates, n_template_nodes, n_features, feature_init_mean=0., feature_init_std=1.):
    """"
    Initialises templates for the Template Fused Gromov Wasserstein layer.
    Returns the adjacency matrices and the features of the nodes of the templates.
    Adjacency matrics are intialised uniformly with values in :math:{0,1} with an additive noise following a normal distribution.
    Features of the nodes are intialised following a normal distribution.

    Parameters
    ----------

      n_templates: int
        Number of templates.
      n_template_nodes: int
        Number of nodes per template.
      n_features: int
        Number of features for the nodes.
      feature_init_mean: float, optional
        Mean of the random normal law to initialize the template features.
      feature_init_std: float, optional
        Standard deviation of the random normal law to initialize the template features.

    Returns
    ----------
      Templates: torch tensor, shape (n_templates, n_template_nodes, n_template_nodes)
           Adjancency matrices for the templates.
      Templates_features: torch tensor, shape (n_templates, n_template_nodes, n_features)
           Node features for each template.
      q0: weight on the template nodes.
    """

    templates_adjacency = torch.rand((n_templates, n_template_nodes, n_template_nodes))
    templates_features = torch.Tensor(n_templates, n_template_nodes, n_features)

    torch.nn.init.normal_(templates_features, mean=feature_init_mean, std=feature_init_std)

    templates_adjacency = templates_adjacency

    q0 = torch.zeros(n_templates, n_template_nodes)

    return 0.5 * (templates_adjacency + torch.transpose(templates_adjacency, 1, 2)), templates_features, q0


def FGW_pooling(x, edge_index, x_T, C_T, alpha, q):
    """
    Computes the FGW distances between a graph and graph templates.

    Parameters
    ----------
    x : torch tensor, shape (n_nodes, n_features)
        Node features of the graph.
    edge_index : torch tensor, shape(n_edges, 2)
        Edge indexes of the graph in the Pytorch Geometric format.
    x_T : list of torch tensors, shape (n_templates, n_templates_nodes, n_features)
        List of the node features of the templates.
    C_T : list of torch tensors, shape (n_templates, n_templates_nodes, n_templates_nodes)
        List of the adjacency matrices of the templates.
    alpha : float
        Trade-off parameter (0 < alpha < 1).
    q : torch tensor, shape (n_templates)

    Returns
    -------
    distances : torch tensor, shape (n_templates)
        Vector of fused Gromov-Wasserstein distances between the graph and the templates.
    """

    n, n_feat = x.shape
    n_T, _, n_feat_T = x_T.shape

    p = torch.ones(n) / n

    C = torch.sparse_coo_tensor(edge_index, torch.ones(len(edge_index[0])), size=(n, n)).type(torch.float)
    C = C.to_dense()

    if not n_feat == n_feat_T:
        raise ValueError('The templates and the graphs must have the same feature dimension.')

    distances = torch.zeros(n_T)

    for j in range(n_T):

        template_features = x_T[j].reshape(len(x_T[j]), n_feat_T)
        M = dist(x, template_features).type(torch.float)

        embedding = fused_gromov_wasserstein2(M, C, C_T[j], p, q[j], alpha=alpha, symmetric=True, max_iter=100)
        distances[j] = embedding

    return distances
