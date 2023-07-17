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

# sphinx_gallery_thumbnail_number = 3

#%%

import matplotlib.pyplot as pl
import numpy as np
import torch
import networkx as nx
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from networkx.generators.community import stochastic_block_model as sbm
from torch_geometric.utils import to_networkx, to_undirected, one_hot
from torch_geometric.utils import stochastic_blockmodel_graph as sbm
from torch_geometric.data import Data as GraphData
import torch.nn as nn
from torch_geometric.nn import Linear
from ot.gnn import TFGWPooling 
from tqdm import tqdm




def circular_graph(N=20, structure_noise=False,p=None):
    """ Create a noisy circular graph
    """
    edges1=[]
    edges2=[]
    for i in range(N):
        edges1.append(i)
        edges2.append(i+1)
        if structure_noise:
            randomint = torch.randint(p,(1,))
            if randomint == 0:
                if i <= N - 3:
                    edges1.append(i)
                    edges2.append(i+2)
                if i == N - 2:
                    edges1.append(i)
                    edges2.append(0)
                if i == N - 1:
                    edges1.append(i)
                    edges2.append(1)
    edges1.append(N)
    edges2.append(0)
    return to_undirected(torch.vstack([torch.tensor(edges1),torch.tensor(edges2)]))

##############################################################################
# Generate data
# -------------

# parameters

# We create two 20 graphs: 10 stochastic block model graphs and 10 noisy circular graphs.
torch.manual_seed(0)

n_graphs=100
n_nodes=10
P=[[0.9,0.2],[0.2,0.9]]
block_sizes=[5,5]

x1=torch.tensor([0,0,0,0,0,1,1,1,1,1])
x1=one_hot(x1)
x1=torch.reshape(x1,(10,2))

graphs1=[GraphData(x=x1,edge_index=sbm(block_sizes,P),y=torch.tensor([0])) for i in range(n_graphs)]


edges=[circular_graph(n_nodes, True,7) for i in range(n_graphs)]
xs=[one_hot(torch.tensor([i % 2 for i in range(max(edges[i][0])+1)])) for i in range(n_graphs)]
xs=[torch.reshape(xs[i],(len(xs[i]),2)) for i in range(n_graphs)]


graphs2=[GraphData(x=xs[i],edge_index=edges[i],y=torch.tensor([1])) for i in range(n_graphs)]


graphs=graphs1+graphs2

train_graphs,test_graphs=random_split(graphs, [n_graphs,n_graphs])


train_loader=DataLoader(train_graphs, batch_size=10, shuffle=True)
test_loader=DataLoader(test_graphs, batch_size=10, shuffle=True)


#%%

##############################################################################
# Plot data
# ---------

# plot one graph of each class

fontsize=10

pl.figure(1, figsize=(8, 2.5))
pl.clf()
pl.subplot(121)
pl.axis('off')
pl.axis
pl.title('Graph of class 1', fontsize=fontsize)
G=to_networkx(graphs1[0],to_undirected=True)
pos = nx.spring_layout(G, seed=0)
nx.draw_networkx(G,pos ,with_labels=False,nodelist=[0,1,2,3,4] ,node_color="tab:red")
nx.draw_networkx(G,pos, with_labels=False,nodelist=[5,6,7,8,9] ,node_color="tab:blue")

pl.subplot(122)
pl.axis('off')
pl.title('Graph of class 2', fontsize=fontsize)
G=to_networkx(graphs2[0],to_undirected=True)
pos = nx.spring_layout(G, seed=0)
nx.draw_networkx(G, pos,with_labels=False,nodelist=[0,2,4,6,8,10] ,node_color="tab:red")
nx.draw_networkx(G, pos,with_labels=False,nodelist=[1,3,5,7,9] ,node_color="tab:blue")


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

    def __init__(self, n_features, n_templates, n_template_nodes, n_classes,feature_init_mean=0., feature_init_std=1.):
        """
        Pooling architecture using the TFGW layer.
        """
        super().__init__()

        self.n_features = n_features
        self.n_templates = n_templates
        self.n_template_nodes = n_template_nodes


        self.TFGW = TFGWPooling(self.n_features,self.n_templates, self.n_template_nodes,feature_init_mean=feature_init_mean,feature_init_std=feature_init_std)

        self.linear = Linear(self.n_templates, n_classes)

    def forward(self, x, edge_index, batch=None):

        x = self.TFGW(x, edge_index, batch)


        x = self.linear(x)

        return x

n_epochs=150

model = pooling_TFGW(2, 4, 4, 2,feature_init_mean=0.5, feature_init_std=0.5)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
criterion = torch.nn.CrossEntropyLoss()


accuracy=[]
loss_all=[]
for epoch in range(n_epochs):
    losses=[]
    accs=[]
    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss=criterion(out, data.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pred = out.argmax(dim=1)
        train_correct = pred==data.y
        train_acc = int(train_correct.sum()) / len(data)
        accs.append(train_acc)
    print(f'Epoch: {epoch:03d}, Loss: {torch.mean(torch.tensor(losses)):.4f},Train Accuracy: {torch.mean(torch.tensor(accs)):.4f}')
    accuracy.append(torch.mean(torch.tensor(accs)))
    loss_all.append(torch.mean(torch.tensor(losses)))


pl.figure(1, figsize=(8, 2.5))
pl.clf()
pl.subplot(121)
pl.plot(loss_all)
pl.title('Loss')

pl.subplot(122)
pl.plot(accuracy)
pl.title('Accuracy')

pl.tight_layout()
pl.show()

test_accs=[]
for data in test_loader:
    out = model(data.x, data.edge_index, data.batch)
    pred = out.argmax(dim=1)
    test_correct = pred==data.y
    test_acc = int(test_correct.sum()) / len(data)
    test_accs.append(test_acc)

print(f'Test Accuracy: {torch.mean(torch.tensor(test_acc)):.4f}') 



# %%
