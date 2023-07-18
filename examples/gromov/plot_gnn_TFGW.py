# -*- coding: utf-8 -*-
"""
==============================
Graph classification with Tempate Based Fused Gromov Wasserstein
==============================

This example first illustrates how to train a graph classification gnn based on the Template Fused Gromov Wasserstein layer .

[52] C. Vincent-Cuaz, R. Flamary, M. Corneli, T. Vayer, N. Courty (2022).
"Template based graph neural network with optimal transport distances"
(https://papers.nips.cc/paper_files/paper/2022/file/4d3525bc60ba1adc72336c0392d3d902-Paper-Conference.pdf). Advances in Neural Information Processing Systems, 35.

"""

# Author: Sonia Mazelet <sonia.mazelet@ens-paris-saclay.fr>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 1

#%%

import matplotlib.pyplot as pl
import torch
import networkx as nx
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, one_hot
from torch_geometric.utils import stochastic_blockmodel_graph as sbm
from torch_geometric.data import Data as GraphData
import torch.nn as nn
from torch_geometric.nn import Linear, GCNConv
from ot.gnn import TFGWPooling


##############################################################################
# Generate data
# -------------

# parameters

# We create two 20 graphs: 10 stochastic block model graphs and 10 noisy circular graphs.
torch.manual_seed(0)

n_graphs = 50
n_nodes = 10
n_node_classes = 2

P1 = [[1]]
P2 = [[0.9, 0.1], [0.1, 0.9]]

block_sizes1 = [n_nodes]
block_sizes2 = [n_nodes // 2, n_nodes // 2]

x1 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
x1 = one_hot(x1, num_classes=n_node_classes)
x1 = torch.reshape(x1, (n_nodes, n_node_classes))

x2 = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
x2 = one_hot(x2, num_classes=n_node_classes)
x2 = torch.reshape(x2, (n_nodes, n_node_classes))

graphs1 = [GraphData(x=x1, edge_index=sbm(block_sizes1, P1), y=torch.tensor([0])) for i in range(n_graphs)]
graphs2 = [GraphData(x=x2, edge_index=sbm(block_sizes2, P2), y=torch.tensor([1])) for i in range(n_graphs)]

graphs = graphs1 + graphs2

train_graphs, test_graphs = random_split(graphs, [n_graphs, n_graphs])

train_loader = DataLoader(train_graphs, batch_size=10, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=10, shuffle=False)


#%%

##############################################################################
# Plot data
# ---------

# plot one graph of each class

fontsize = 10

pl.figure(1, figsize=(8, 2.5))
pl.clf()
pl.subplot(121)
pl.axis('off')
pl.axis
pl.title('Graph of class 1', fontsize=fontsize)
G = to_networkx(graphs1[0], to_undirected=True)
pos = nx.spring_layout(G, seed=0)
nx.draw_networkx(G, pos, with_labels=False, node_color="tab:blue")

pl.subplot(122)
pl.axis('off')
pl.title('Graph of class 2', fontsize=fontsize)
G = to_networkx(graphs2[0], to_undirected=True)
pos = nx.spring_layout(G, seed=0)
nx.draw_networkx(G, pos, with_labels=False, nodelist=[0, 1, 2, 3, 4], node_color="tab:blue")
nx.draw_networkx(G, pos, with_labels=False, nodelist=[5, 6, 7, 8, 9], node_color="tab:red")

pl.tight_layout()
pl.show()

#%%

##############################################################################
# Pooling architecture using the TFGW layer
# ---------


class pooling_TFGW(nn.Module):
    """
    Pooling architecture using the TFGW layer.
    """

    def __init__(self, n_features, n_templates, n_template_nodes, n_classes, n_hidden_layers, feature_init_mean=0., feature_init_std=1.):
        """
        Pooling architecture using the TFGW layer.
        """
        super().__init__()

        self.n_templates = n_templates
        self.n_template_nodes = n_template_nodes
        self.n_hidden_layers = n_hidden_layers
        self.n_features = n_features

        self.conv = GCNConv(self.n_features, self.n_hidden_layers)

        self.TFGW = TFGWPooling(self.n_hidden_layers, self.n_templates, self.n_template_nodes, feature_init_mean=feature_init_mean, feature_init_std=feature_init_std)

        self.linear = Linear(self.n_templates, n_classes)

    def forward(self, x, edge_index, batch=None):
        x = self.conv(x, edge_index)

        x = self.TFGW(x, edge_index, batch)

        x = self.linear(x)

        return x


n_epochs = 100

model = pooling_TFGW(n_features=2, n_templates=2, n_template_nodes=2, n_classes=2, n_hidden_layers=2, feature_init_mean=0.5, feature_init_std=0.5)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
criterion = torch.nn.CrossEntropyLoss()

accuracy = []
loss_all = []
for epoch in range(n_epochs):
    losses = []
    accs = []
    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pred = out.argmax(dim=1)
        train_correct = pred == data.y
        train_acc = int(train_correct.sum()) / len(data)
        accs.append(train_acc)
    print(f'Epoch: {epoch:03d}, Loss: {torch.mean(torch.tensor(losses)):.4f},Train Accuracy: {torch.mean(torch.tensor(accs)):.4f}')
    accuracy.append(torch.mean(torch.tensor(accs)))
    loss_all.append(torch.mean(torch.tensor(losses)))


pl.figure(1, figsize=(8, 2.5))
pl.clf()
pl.subplot(121)
pl.plot(loss_all)
pl.xlabel('epochs')
pl.title('Loss')

pl.subplot(122)
pl.plot(accuracy)
pl.xlabel('epochs')
pl.title('Accuracy')

pl.tight_layout()
pl.show()

test_accs = []
for data in test_loader:
    out = model(data.x, data.edge_index, data.batch)
    pred = out.argmax(dim=1)
    test_correct = pred == data.y
    test_acc = int(test_correct.sum()) / len(data)
    test_accs.append(test_acc)

print(f'Test Accuracy: {torch.mean(torch.tensor(test_acc)):.4f}')


# %%
