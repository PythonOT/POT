"""
Template Fused Gromov Wasserstein
"""

import torch
import torch.nn as nn
from ._utils import template_initialisation, FGW_pooling


class TFGWLayer(nn.Module):
    """
    Template Fused Gromov-Wasserstein (TFGW) layer. This layer acts as a pooling layer for graph neural networks.
        It computes the fused Gromov-Wasserstein distances between the graph and a set of templates.

    Parameters
    ----------
    n_features : int
        Feature dimension of the nodes.
    n_templates : int
         Number of graph templates.
    n_templates_nodes : int
        Number of nodes in each template.
    alpha0 : float, optional
        Trade-off parameter (0 < alpha < 1). If None alpha is trained, else it is fixed at the given value.
    train_node_weights : bool, optional
        If True, the templates node weights are learned.
        Else, they are uniform.


    References
    ----------
    .. [52]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Template based graph neural network with optimal transport distances"
    """

    def __init__(self, n_features, n_templates=2, n_template_nodes=2, alpha0=None, train_node_weights=True):
        """
        Template Fused Gromov-Wasserstein (TFGW) layer. This layer acts as a pooling layer for graph neural networks.
            It computes the fused Gromov-Wasserstein distances between the graph and a set of templates.

        Parameters
        ----------
        n_features : int
                Feature dimension of the nodes.        
        n_templates : int
                Number of graph templates.
        n_templates_nodes : int
                Number of nodes in each template.
        alpha0 : float, optional
                Trade-off parameter (0 < alpha < 1). If None alpha is trained, else it is fixed at the given value.
        train_node_weights : bool, optional
                If True, the templates node weights are learned.
                Else, they are uniform.

        References
        ----------
        .. [51]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
              "Template based graph neural network with optimal transport distances"

        """
        super().__init__()

        self.n_templates = n_templates
        self.n_templates_nodes = n_template_nodes
        self.n_features = n_features

        templates, templates_features, self.q0 = template_initialisation(self.n_templates_nodes, self.n_templates, self.n_features)
        self.templates = nn.Parameter(templates)
        self.templates_features = nn.Parameter(templates_features)

        self.softmax = nn.Softmax(dim=1)

        if train_node_weights:
            self.q0 = nn.Parameter(self.q0)

        if alpha0 is None:
            alpha0 = torch.Tensor([0])
            self.alpha0 = nn.Parameter(alpha0)
        else:
            alpha0 = torch.Tensor([alpha0])
            self.alpha0 = torch.logit(alpha0)

    def forward(self, x, edge_index):
        alpha = torch.sigmoid(self.alpha0)
        q = self.softmax(self.q0)
        x = FGW_pooling(x, edge_index, self.templates_features, self.templates, alpha, q)
        return x
