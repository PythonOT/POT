"""Tests for module gnn"""

# Author: Sonia Mazelet <sonia.mazelet@ens-paris-saclay.fr>
#         Rémi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import pytest

try:  # test if pytorch_geometric is installed
    import torch_geometric
    import torch
    from torch_geometric.nn import Linear
    from torch_geometric.data import Data as GraphData
    from torch_geometric.loader import DataLoader
    import torch.nn as nn
    from ot.gnn import TFGWPooling, TWPooling

except ImportError:
    torch_geometric = False


@pytest.mark.skipif(not torch_geometric, reason="pytorch_geometric not installed")
def test_TFGW_optim():
    # Test the TFGW layer by passing two graphs through the layer and doing backpropagation.

    class pooling_TFGW(nn.Module):
        """
        Pooling architecture using the TFGW layer.
        """

        def __init__(self, n_features, n_templates, n_template_nodes):
            """
            Pooling architecture using the TFGW layer.
            """
            super().__init__()

            self.n_features = n_features
            self.n_templates = n_templates
            self.n_template_nodes = n_template_nodes

            self.TFGW = TFGWPooling(
                self.n_templates, self.n_template_nodes, self.n_features
            )

            self.linear = Linear(self.n_templates, 1)

        def forward(self, x, edge_index):
            x = self.TFGW(x, edge_index)

            x = self.linear(x)

            return x

    torch.manual_seed(0)

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

    graph1 = GraphData(x=x1, edge_index=edge_index1, y=torch.tensor([0.0]))
    graph2 = GraphData(x=x2, edge_index=edge_index2, y=torch.tensor([1.0]))

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
            optimizer.zero_grad()


@pytest.mark.skipif(not torch_geometric, reason="pytorch_geometric not installed")
def test_TFGW_variants():
    # Test the TFGW layer by passing two graphs through the layer and doing backpropagation.

    class GNN_pooling(nn.Module):
        """
        Pooling architecture using the TW layer.
        """

        def __init__(self, n_features, n_templates, n_template_nodes, pooling_layer):
            """
            Pooling architecture using the TW layer.
            """
            super().__init__()

            self.n_features = n_features
            self.n_templates = n_templates
            self.n_template_nodes = n_template_nodes

            self.TFGW = pooling_layer

            self.linear = Linear(self.n_templates, 1)

        def forward(self, x, edge_index, batch=None):
            x = self.TFGW(x, edge_index, batch=batch)

            x = self.linear(x)

            return x

    n_templates = 3
    n_template_nodes = 3
    n_nodes = 10
    n_features = 3

    torch.manual_seed(0)

    C1 = torch.randint(0, 2, size=(n_nodes, n_nodes))
    edge_index1 = torch.stack(torch.where(C1 == 1))
    x1 = torch.rand(n_nodes, n_features)
    graph1 = GraphData(x=x1, edge_index=edge_index1, y=torch.tensor([0.0]))
    batch1 = torch.tensor([1] * n_nodes)
    batch1[: n_nodes // 2] = 0

    criterion = torch.nn.CrossEntropyLoss()

    for train_node_weights in [True, False]:
        for alpha in [None, 0, 0.5]:
            for multi_alpha in [True, False]:
                model = GNN_pooling(
                    n_features,
                    n_templates,
                    n_template_nodes,
                    pooling_layer=TFGWPooling(
                        n_templates,
                        n_template_nodes,
                        n_features,
                        alpha=alpha,
                        multi_alpha=multi_alpha,
                        train_node_weights=train_node_weights,
                    ),
                )

                # predict
                out1 = model(graph1.x, graph1.edge_index)
                loss = criterion(out1, graph1.y)
                loss.backward()

                # predict on batch
                out1 = model(graph1.x, graph1.edge_index, batch1)


@pytest.mark.skipif(not torch_geometric, reason="pytorch_geometric not installed")
def test_TW_variants():
    # Test the TFGW layer by passing two graphs through the layer and doing backpropagation.

    class GNN_pooling(nn.Module):
        """
        Pooling architecture using the TW layer.
        """

        def __init__(self, n_features, n_templates, n_template_nodes, pooling_layer):
            """
            Pooling architecture using the TW layer.
            """
            super().__init__()

            self.n_features = n_features
            self.n_templates = n_templates
            self.n_template_nodes = n_template_nodes

            self.TFGW = pooling_layer

            self.linear = Linear(self.n_templates, 1)

        def forward(self, x, edge_index, batch=None):
            x = self.TFGW(x, edge_index, batch=batch)

            x = self.linear(x)

            return x

    n_templates = 3
    n_template_nodes = 3
    n_nodes = 10
    n_features = 3

    torch.manual_seed(0)

    C1 = torch.randint(0, 2, size=(n_nodes, n_nodes))
    edge_index1 = torch.stack(torch.where(C1 == 1))
    x1 = torch.rand(n_nodes, n_features)
    graph1 = GraphData(x=x1, edge_index=edge_index1, y=torch.tensor([0.0]))
    batch1 = torch.tensor([1] * n_nodes)
    batch1[: n_nodes // 2] = 0

    criterion = torch.nn.CrossEntropyLoss()

    for train_node_weights in [True, False]:
        model = GNN_pooling(
            n_features,
            n_templates,
            n_template_nodes,
            pooling_layer=TWPooling(
                n_templates,
                n_template_nodes,
                n_features,
                train_node_weights=train_node_weights,
            ),
        )

        out1 = model(graph1.x, graph1.edge_index)
        loss = criterion(out1, graph1.y)
        loss.backward()

        # predict on batch
        out1 = model(graph1.x, graph1.edge_index, batch1)


@pytest.mark.skipif(not torch_geometric, reason="pytorch_geometric not installed")
def test_TW():
    # Test the TW layer by passing two graphs through the layer and doing backpropagation.

    class pooling_TW(nn.Module):
        """
        Pooling architecture using the TW layer.
        """

        def __init__(self, n_features, n_templates, n_template_nodes):
            """
            Pooling architecture using the TW layer.
            """
            super().__init__()

            self.n_features = n_features
            self.n_templates = n_templates
            self.n_template_nodes = n_template_nodes

            self.TFGW = TWPooling(
                self.n_templates, self.n_template_nodes, self.n_features
            )

            self.linear = Linear(self.n_templates, 1)

        def forward(self, x, edge_index):
            x = self.TFGW(x, edge_index)

            x = self.linear(x)

            return x

    torch.manual_seed(0)

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

    graph1 = GraphData(x=x1, edge_index=edge_index1, y=torch.tensor([0.0]))
    graph2 = GraphData(x=x2, edge_index=edge_index2, y=torch.tensor([1.0]))

    dataset = DataLoader([graph1, graph2], batch_size=1)

    model_W = pooling_TW(n_features, n_templates, n_template_nodes)

    optimizer = torch.optim.Adam(model_W.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model_W.train()

    for i in range(n_epochs):
        for data in dataset:
            out = model_W(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
