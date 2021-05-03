# Import libs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import os

os.environ['DGLBACKEND'] = 'pytorch'  # tell DGL what backend to use
import dgl
from dgl import DGLGraph
from dgl.data import MiniGCDataset

import time

import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

# %%
#
# matplotlib.rcParams['figure.figsize'] = (3, 3)
#
#
# def draw(g, title):
#     plt.figure()
#     nx.draw(g.to_networkx(), with_labels=True, node_color='skyblue', edge_color='white')
#     plt.gcf().set_facecolor('k')
#     plt.title(title)
#
# # The datset contains 8 different types of graphs:
# graph_type = (
#     'cycle',
#     'star',
#     'wheel',
#     'lollipop',
#     'hypercube',
#     'grid',
#     'clique',
#     'circular ladder',
# )
#
# # %%
#
# # visualise the 8 classes of graphs
# for graph, label in MiniGCDataset(8, 10, 20):
#     draw(graph, f'Class: {label}, {graph_type[label]} graph')

# %%
#
# # create artifical data feature (= in degree) for each node
# def create_artificial_features(dataset):
#     for (graph, _) in dataset:
#         graph.ndata['feat'] = graph.in_degrees().view(-1, 1).float()
#         graph.edata['feat'] = torch.ones(graph.number_of_edges(), 1)
#     return dataset
#

# %%

# # Generate artifical graph dataset with DGL
# trainset = MiniGCDataset(350, 10, 20)
# testset = MiniGCDataset(100, 10, 20)
#
# trainset = create_artificial_features(trainset)
# testset = create_artificial_features(testset)
#
# print(trainset[0])

# %%

class GatedGCN_layer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.C = nn.Linear(input_dim, output_dim)
        self.D = nn.Linear(input_dim, output_dim)
        self.E = nn.Linear(input_dim, output_dim)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bx_j = edges.src['BX']
        # e_j = Ce_j + Dxj + Ex
        CE = edges.data['CE']
        DX = edges.src['DX']
        EX = edges.dst['EX']
        e_j = CE + DX + EX
        edges.data['E'] = e_j
        return {'Bx_j': Bx_j, 'e_j': e_j}

    def reduce_func(self, nodes):
        Ax = nodes.data['AX']
        Bx_j = nodes.mailbox['Bx_j']
        e_j = nodes.mailbox['e_j']
        # sigma_j = σ(e_j)
        σ_j = torch.sigmoid(e_j)
        # h = Ax + Σ_j η_j * Bxj
        h = Ax + torch.sum(σ_j * Bx_j, dim=1) / torch.sum(σ_j, dim=1)
        # h = Ax + torch.sum(Bx_j, dim=1)
        return {'H': h}

    def forward(self, g, X, E_X, snorm_n, snorm_e):
        g.ndata['H'] = X
        g.ndata['AX'] = self.A(X)
        g.ndata['BX'] = self.B(X)
        g.ndata['DX'] = self.D(X)
        g.ndata['EX'] = self.E(X)
        g.edata['E'] = E_X
        g.edata['CE'] = self.C(E_X)

        g.update_all(self.message_func, self.reduce_func)

        H = g.ndata['H']  # result of graph convolution
        E = g.edata['E']  # result of graph convolution

        H *= snorm_n  # normalize activation w.r.t. graph node size
        E *= snorm_e  # normalize activation w.r.t. graph edge size

        H = self.bn_node_h(H)  # batch normalization
        E = self.bn_node_e(E)  # batch normalization

        H = torch.relu(H)  # non-linear activation
        E = torch.relu(E)  # non-linear activation

        H = X + H  # residual connection
        E = E_X + E  # residual connection

        return H, E


class NormalGCN_layer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bx_j = edges.src['BX']
        return {'Bx_j': Bx_j}

    def reduce_func(self, nodes):
        Ax = nodes.data['AX']
        Bx_j = nodes.mailbox['Bx_j']
        # h = Ax + Σ_j η_j * Bxj
        h = Ax + torch.sum(Bx_j, dim=1)
        return {'H': h}

    def forward(self, g, X, E_X, snorm_n, snorm_e):
        g.ndata['H'] = X
        g.ndata['AX'] = self.A(X)
        g.ndata['BX'] = self.B(X)

        g.edata['E'] = E_X

        g.update_all(self.message_func, self.reduce_func)

        H = g.ndata['H']  # result of graph convolution

        H *= snorm_n  # normalize activation w.r.t. graph node size

        H = self.bn_node_h(H)  # batch normalization

        H = torch.relu(H)  # non-linear activation

        H = X + H  # residual connection

        return H, E_X


# %%

class MLP_layer(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L = nb of hidden layers
        super().__init__()
        list_FC_layers = [
            nn.Linear(input_dim, input_dim) for l in range(L)
        ]
        list_FC_layers.append(nn.Linear(input_dim, output_dim))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = torch.relu(y)
        y = self.FC_layers[self.L](y)
        return y


# %%

class GatedGCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, L):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)
        self.GatedGCN_layers = nn.ModuleList([
            GatedGCN_layer(hidden_dim, hidden_dim) for _ in range(L)
        ])
        self.MLP_layer = MLP_layer(hidden_dim, output_dim)

    def forward(self, g, X, E, snorm_n, snorm_e):
        # input embedding
        H = self.embedding_h(X)
        E = self.embedding_e(E)

        # graph convnet layers
        for GGCN_layer in self.GatedGCN_layers:
            H, E = GGCN_layer(g, H, E, snorm_n, snorm_e)

        # MLP classifier
        g.ndata['H'] = H
        y = dgl.mean_nodes(g, 'H')
        y = self.MLP_layer(y)

        return y



class NormalGCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, L):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)
        self.NormalGCN_layers = nn.ModuleList([
            NormalGCN_layer(hidden_dim, hidden_dim) for _ in range(L)
        ])
        self.MLP_layer = MLP_layer(hidden_dim, output_dim)

    def forward(self, g, X, E, snorm_n, snorm_e):
        # input embedding
        H = self.embedding_h(X)
        E = self.embedding_e(E)

        # graph convnet layers
        for GGCN_layer in self.NormalGCN_layers:
            H, E = GGCN_layer(g, H, E, snorm_n, snorm_e)

        # MLP classifier
        g.ndata['H'] = H
        y = dgl.mean_nodes(g, 'H')
        y = self.MLP_layer(y)

        return y


# %%

# # instantiate network
# model = NormalGCN(input_dim=1, hidden_dim=150, output_dim=8, L=2)
# print(model)


# %% md

## Define a few helper functions

# %%

# Collate function to prepare graphs
#
# def collate(samples):
#     graphs, labels = map(list, zip(*samples))  # samples is a list of pairs (graph, label)
#     labels = torch.tensor(labels)
#     sizes_n = [graph.number_of_nodes() for graph in graphs]  # graph sizes
#     snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
#     snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization
#     sizes_e = [graph.number_of_edges() for graph in graphs]  # nb of edges
#     snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
#     snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
#     batched_graph = dgl.batch(graphs)  # batch graphs
#     return batched_graph, labels, snorm_n, snorm_e
#

# %%

# Compute accuracy
#
# def accuracy(logits, targets):
#     preds = logits.detach().argmax(dim=1)
#     acc = (preds == targets).sum().item()
#     return acc


# %% md

## Test forward pass

# %%
#
# # Define DataLoader and get first graph batch
#
# train_loader = DataLoader(trainset, batch_size=10, shuffle=True, collate_fn=collate)
# batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e = next(iter(train_loader))
# batch_X = batch_graphs.ndata['feat']
# batch_E = batch_graphs.edata['feat']
#
# # %%
#
# # Checking some sizes
#
# print(f'batch_graphs:', batch_graphs)
# print(f'batch_labels:', batch_labels)
# print('batch_X size:', batch_X.size())
# print('batch_E size:', batch_E.size())
#
# # %%
#
# batch_scores = model(batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e)
# print(batch_scores.size())
#
# batch_labels = batch_labels
# print(f'accuracy: {accuracy(batch_scores, batch_labels)}')

# %% md

## Test backward pass

# %%
#
# # Loss
# J = nn.CrossEntropyLoss()(batch_scores, batch_labels.long())
#
# # Backward pass
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer.zero_grad()
# J.backward()
# optimizer.step()
#

# %% md

## Train one epoch

# %%
#
# def train(model, data_loader, loss, optimizer):
#     model.train()
#     epoch_loss = 0
#     epoch_train_acc = 0
#     nb_data = 0
#     gpu_mem = 0
#
#     for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
#         batch_X = batch_graphs.ndata['feat']
#         batch_E = batch_graphs.edata['feat']
#
#         batch_scores = model(batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e)
#         J = loss(batch_scores, batch_labels.long())
#         optimizer.zero_grad()
#         J.backward()
#         optimizer.step()
#
#         epoch_loss += J.detach().item()
#         epoch_train_acc += accuracy(batch_scores, batch_labels)
#         nb_data += batch_labels.size(0)
#
#     epoch_loss /= (iter + 1)
#     epoch_train_acc /= nb_data
#
#     return epoch_loss, epoch_train_acc
#

# %% md

## Evaluation

# %%
#
# def evaluate(model, data_loader, loss):
#     model.eval()
#     epoch_test_loss = 0
#     epoch_test_acc = 0
#     nb_data = 0
#
#     with torch.no_grad():
#         for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
#             batch_X = batch_graphs.ndata['feat']
#             batch_E = batch_graphs.edata['feat']
#
#             batch_scores = model(batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e)
#             J = loss(batch_scores, batch_labels.long())
#
#             epoch_test_loss += J.detach().item()
#             epoch_test_acc += accuracy(batch_scores, batch_labels)
#             nb_data += batch_labels.size(0)
#
#         epoch_test_loss /= (iter + 1)
#         epoch_test_acc /= nb_data
#
#     return epoch_test_loss, epoch_test_acc


# %% md

# Train GNN
#
# sweep_config = {
#     'method': 'grid', #grid, random
#     'metric': {
#       'name': 'loss',
#       'goal': 'minimize'
#     },
#     'parameters': {
#         'epochs': {
#             'values': [20, 50]
#         },
#         'learning_rate': {
#             'values': [3e-4, 3e-5, 1e-5]
#         },
#         'hidden_dim':{
#             'values':[128,256]
#         },
#     }
# }

# sweep_id = wandb.sweep(sweep_config, project="test-sweep")
# %%
#
#
#
# def train_wandb():
#     config_defaults = {
#         'epochs': 25,
#         'learning_rate': 1e-3,
#         'hidden_dim': 128
#     }
#
#     wandb.init(config=config_defaults)
#
#     config = wandb.config
#
#     # datasets
#     train_loader = DataLoader(trainset, batch_size=50, shuffle=True, collate_fn=collate)
#     test_loader = DataLoader(testset, batch_size=50, shuffle=False, collate_fn=collate)
#
#     # Create model
#     model = GatedGCN(input_dim=1, hidden_dim=config.hidden_dim, output_dim=8, L=4)
#     loss = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
#
#     for epoch in range(100):
#         start = time.time()
#         train_loss, train_acc = train(model, train_loader, loss, optimizer)
#
#         test_loss, test_acc = evaluate(model, test_loader, loss)
#
#         wandb.log({
#             "train_loss": train_loss,
#             "test_loss": test_loss,
#             "train_acc": train_acc,
#             "test_acc": test_acc,
#         })
#
#         print(f'Epoch {epoch}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}')
#         print(f'train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}')
#%%
# wandb.agent(sweep_id, train_wandb)
# %%
#
# plt.plot(epoch_train_losses, label="train_loss")
# plt.plot(epoch_test_losses, label="test_loss")
# plt.legend()
# plt.figure()
# plt.plot(epoch_train_accs, label="train_acc")
# plt.plot(epoch_test_accs, label="test_acc")
# plt.legend()

# %%