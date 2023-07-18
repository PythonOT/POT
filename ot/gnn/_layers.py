"""
Template Fused Gromov Wasserstein
"""

import torch
import torch.nn as nn
from ._utils import TFGW_template_initialisation, distance_to_templates


class TFGWPooling(nn.Module):
    """
    Template Fused Gromov-Wasserstein (TFGW) layer. This layer is a pooling layer for graph neural networks.
        It computes the fused Gromov-Wasserstein distances between the graph and a set of templates.

    Parameters
    ----------
    n_features : int
        Feature dimension of the nodes.
    n_tplt : int
         Number of graph templates.
    n_tplt_nodes : int
        Number of nodes in each template.
    alpha : float, optional
        FGW trade-off parameter (0 < alpha < 1). If None alpha is trained, else it is fixed at the given value.
        Weights features (alpha=0) and structure (alpha=1).
    train_node_weights : bool, optional
        If True, the templates node weights are learned.
        Else, they are uniform.
    multi_alpha: bool, optional
        If True, the alpha parameter is a vector of size n_tplt.
    feature_init_mean: float, optional
        Mean of the random normal law to initialize the template features.
    feature_init_std: float, optional
        Standard deviation of the random normal law to initialize the template features.



    References
    ----------
    .. [52]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Template based graph neural network with optimal transport distances"
    """

    def __init__(self, n_features, n_tplt=2, n_tplt_nodes=2, alpha=None, train_node_weights=True, multi_alpha=False, feature_init_mean=0., feature_init_std=1.):
        """
        Template Fused Gromov-Wasserstein (TFGW) layer. This layer is a pooling layer for graph neural networks.
            It computes the fused Gromov-Wasserstein distances between the graph and a set of templates.

        Parameters
        ----------
        n_features : int
                Feature dimension of the nodes.
        n_tplt : int
                Number of graph templates.
        n_tplt_nodes : int
                Number of nodes in each template.
        alpha : float, optional
                Trade-off parameter (0 < alpha < 1). If None alpha is trained, else it is fixed at the given value.
                Weights features (alpha=0) and structure (alpha=1).
        train_node_weights : bool, optional
                If True, the templates node weights are learned.
                Else, they are uniform.
        multi_alpha: bool, optional
                If True, the alpha parameter is a vector of size n_tplt.
        feature_init_mean: float, optional
                Mean of the random normal law to initialize the template features.
        feature_init_std: float, optional
                Standard deviation of the random normal law to initialize the template features.

        References
        ----------
        .. [52]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
              "Template based graph neural network with optimal transport distances"

        """
        super().__init__()

        self.n_tplt = n_tplt
        self.n_tplt_nodes = n_tplt_nodes
        self.n_features = n_features
        self.multi_alpha = multi_alpha
        self.feature_init_mean = feature_init_mean
        self.feature_init_std = feature_init_std

        tplt_adjacencies, tplt_features, self.q0 = TFGW_template_initialisation(self.n_tplt, self.n_tplt_nodes, self.n_features, self.feature_init_mean, self.feature_init_std)
        self.tplt_adjacencies = nn.Parameter(tplt_adjacencies)
        self.tplt_features = nn.Parameter(tplt_features)

        self.softmax = nn.Softmax(dim=1)

        if train_node_weights:
            self.q0 = nn.Parameter(self.q0)

        if alpha is None:
            if multi_alpha:
                self.alpha0 = torch.Tensor([0] * self.n_tplt)
            else:
                alpha0 = torch.Tensor([0])
            self.alpha0 = nn.Parameter(alpha0)
        else:
            if multi_alpha:
                self.alpha0 = torch.Tensor([alpha] * self.n_tplt)
            else:
                self.alpha0 = torch.Tensor([alpha])
            self.alpha0 = torch.logit(alpha0)

    def forward(self, x, edge_index, batch=None):
        alpha = torch.sigmoid(self.alpha0)
        q = self.softmax(self.q0)
        x = distance_to_templates(edge_index, self.tplt_adjacencies, x, self.tplt_features, q, alpha, self.multi_alpha, batch)
        return x


class TGWPooling(nn.Module):
    """
    Template Gromov-Wasserstein (TGW) layer. This layer is a pooling layer for graph neural networks.
        It computes the Gromov-Wasserstein distances between the graph and a set of templates.

    Parameters
    ----------
    n_features : int
        Feature dimension of the nodes.
    n_tplt : int
         Number of graph templates.
    n_tplt_nodes : int
        Number of nodes in each template.
    train_node_weights : bool, optional
        If True, the templates node weights are learned.
        Else, they are uniform.
    feature_init_mean: float, optional
        Mean of the random normal law to initialize the template features.
    feature_init_std: float, optional
        Standard deviation of the random normal law to initialize the template features.


    References
    ----------
    .. [52]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Template based graph neural network with optimal transport distances"
    """

    def __init__(self, n_features, n_tplt=2, n_tplt_nodes=2, train_node_weights=True, feature_init_mean=0., feature_init_std=1.):
        """
        Template Fused Gromov-Wasserstein (TFGW) layer. This layer is a pooling layer for graph neural networks.
            It computes the fused Gromov-Wasserstein distances between the graph and a set of templates.

        Parameters
        ----------
        n_features : int
                Feature dimension of the nodes.
        n_tplt : int
                Number of graph templates.
        n_tplt_nodes : int
                Number of nodes in each template.
        train_node_weights : bool, optional
                If True, the templates node weights are learned.
                Else, they are uniform.
        feature_init_mean: float, optional
                Mean of the random normal law to initialize the template features.
        feature_init_std: float, optional
                Standard deviation of the random normal law to initialize the template features.

        References
        ----------
        .. [52]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
              "Template based graph neural network with optimal transport distances"

        """
        super().__init__()

        self.n_tplt = n_tplt
        self.n_tplt_nodes = n_tplt_nodes
        self.n_features = n_features
        self.feature_init_mean = feature_init_mean
        self.feature_init_std = feature_init_std

        tplt_adjacencies, tplt_features, self.q0 = TFGW_template_initialisation(self.n_tplt, self.n_tplt_nodes, self.n_features, self.feature_init_mean, self.feature_init_std)
        self.tplt_adjacencies = nn.Parameter(tplt_adjacencies)
        self.tplt_features = nn.Parameter(tplt_features)

        self.softmax = nn.Softmax(dim=1)

        if train_node_weights:
            self.q0 = nn.Parameter(self.q0)

    def forward(self, x, edge_index, batch=None):
        q = self.softmax(self.q0)
        x = distance_to_templates(edge_index, self.tplt_adjacencies, x, self.tplt_features, q, batch=batch, fused=False)
        return x
