"""Tests for module gnn"""

# Author: Sonia Mazelet <sonia.mazelet@ens-paris-saclay.fr>
#
# License: MIT License


import pytest

try:  # test if pytorch_geometric is installed
    import torch_geometric
except ImportError:
    torch_geometric = False


@pytest.mark.skipif(not torch_geometric, reason="pytorch_geometric not installed")
def test_TFGW():
    # Test the TFGW layer by passing two graphs through the layer and doing backpropagation.

    import torch
    from torch_geometric.nn import Linear
    from torch_geometric.data import Data as GraphData
    from torch_geometric.loader import DataLoader
    import torch.nn as nn
    from ot.gnn import TFGWPooling, TGWPooling

    class pooling_TFGW(nn.Module):
        """
        Pooling architecture using the LTFGW layer.
        """

        def __init__(self, n_features, n_templates, n_template_nodes):
            """
            Pooling architecture using the LTFGW layer.
            """
            super().__init__()

            self.n_features = n_features
            self.n_templates = n_templates
            self.n_template_nodes = n_template_nodes

            self.TFGW = TFGWPooling(self.n_templates, self.n_template_nodes, self.n_features)

            self.linear = Linear(self.n_templates, 1)

        def forward(self, x, edge_index):

            x = self.TFGW(x, edge_index)

            x = self.linear(x)

            return x


    class pooling_TGW(nn.Module):
        """
        Pooling architecture using the LTFGW layer.
        """

        def __init__(self, n_features, n_templates, n_template_nodes):
            """
            Pooling architecture using the LTFGW layer.
            """
            super().__init__()

            self.n_features = n_features
            self.n_templates = n_templates
            self.n_template_nodes = n_template_nodes

            self.TFGW = TGWPooling(self.n_templates, self.n_template_nodes, self.n_features)

            self.linear = Linear(self.n_templates, 1)

        def forward(self, x, edge_index):

            x = self.TFGW(x, edge_index)

            x = self.linear(x)

            return x            

    n_templates = 3
    n_template_nodes = 3
    n_nodes = 10
    n_features = 3
    n_epochs = 3

    C1 = torch.randint(0, 2, size=(n_nodes, n_nodes))
    C2 = torch.randint(0, 2, size=(n_nodes, n_nodes))

    edge_index1 = torch.stack(torch.where(C1 == 1))
    edge_index2 = torch.stack(torch.where(C2 == 1))

    x1 = torch.rand(n_nodes, n_features)
    x2 = torch.rand(n_nodes, n_features)

    graph1 = GraphData(x=x1, edge_index=edge_index1, y=torch.tensor([0.]))
    graph2 = GraphData(x=x2, edge_index=edge_index2, y=torch.tensor([1.]))

    dataset = DataLoader([graph1, graph2], batch_size=1)

    model_FGW = pooling_TFGW(n_features, n_templates, n_template_nodes)

    optimizer = torch.optim.Adam(model_FGW.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model_FGW.train()

    for i in range(n_epochs):
        for data in dataset:

            out = model_FGW(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

    model_GW = pooling_TGW(n_features, n_templates, n_template_nodes)

    optimizer = torch.optim.Adam(model_GW.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model_GW.train()

    for i in range(n_epochs):
        for data in dataset:

            out = model_GW(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
