from random import random

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import gated_gcn
import excitation_gcn
import numpy as np

os.environ['DGLBACKEND'] = 'pytorch'  # tell DGL what backend to use
import dgl
from dgl import DGLGraph
from dgl.data import MiniGCDataset


from torch.utils.data import random_split

import time


#%%
# def accuracy(logits, targets):
#     preds = logits.detach().argmax(dim=1)
#     acc = (preds == targets).sum().item()
#     return acc


# Collate function to prepare graphs

def collate(samples):
    graphs, labels = map(list, zip(*samples))  # samples is a list of pairs (graph, label)
    # print("samples")
    # print(samples)
    # print("labels")
    # print(labels)
    # labels = torch.cat(labels).view(-1, labels[0].shape[0])
    labels = torch.tensor(labels)
    sizes_n = [graph.number_of_nodes() for graph in graphs]  # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization
    sizes_e = [graph.number_of_edges() for graph in graphs]  # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, labels, snorm_n, snorm_e



def train(model, data_loader, loss, optimizer, scheduler):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        if iter % 20 == 0:
            print('starting iter', iter)
        batch_X = batch_graphs.ndata['feat']
        batch_E = batch_graphs.edata['feat']

        batch_scores = model(batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e)
        J = loss(batch_scores, batch_labels.long())
        optimizer.zero_grad()
        J.backward()
        optimizer.step()

        epoch_loss += J.detach().item()
        # epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)

    scheduler.step()

    epoch_loss /= (iter + 1)
    # epoch_train_acc /= nb_data

    # return epoch_loss, epoch_train_acc
    return epoch_loss


def evaluate(model, data_loader, loss):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0

    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_X = batch_graphs.ndata['feat']
            batch_E = batch_graphs.edata['feat']

            batch_scores = model(batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e)
            J = loss(batch_scores, batch_labels.long())

            epoch_test_loss += J.detach().item()
            # epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)
        # epoch_test_acc /= nb_data

    # return epoch_test_loss, epoch_test_acc
    return epoch_test_loss

#%%
# Generate artifical graph dataset with DGL
data = MiniGCDataset(500, 10, 20)
test_ratio = 0.2
n = len(data)
train_size = int(n * (1 - test_ratio))
test_size = n - train_size
testset, trainset = random_split(data, (test_size, train_size), generator=torch.Generator().manual_seed(42))

#%%
next(iter(trainset))

#%%

def get_model(name,input_dim, hidden_dim, output_dim, L):
    if name == "normal":
        return gated_gcn.NormalGCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, L=L)
    elif name == "excitation":
        return excitation_gcn.ExcitationGCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, L=L)
    else:
        return gated_gcn.GatedGCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, L=L)


#%%
config = {
    'epochs': 150,
    'learning_rate': 1e-4,
    'hidden_dim': 128,
    'model': 'excitation',
    'decay_steps': 20,
    'L':4
}

def create_artificial_features(dataset):
    for (graph, _) in dataset:
        graph.ndata['feat'] = graph.in_degrees().view(-1, 1).float()
        graph.edata['feat'] = torch.ones(graph.number_of_edges(), 1)
    return dataset
# trainset_GCL = create_artificial_features(MiniGCDataset(100, 10, 20))
# train_GCL_loader = DataLoader(trainset_GCL, batch_size=3, shuffle=True, collate_fn=collate)
# datasets

trainset = create_artificial_features(trainset)
testset = create_artificial_features(testset)

train_loader = DataLoader(trainset, batch_size=50, shuffle=True, collate_fn=collate)
test_loader = DataLoader(testset, batch_size=50, shuffle=False, collate_fn=collate)

print('size of train set:', len(train_loader), 'batches')
print('size of test set:', len(test_loader), 'batches')
#%%
#
# batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e = next(iter(train_loader))
# batch_X = batch_graphs.ndata['feat']
# batch_E = batch_graphs.edata['feat']
# #%%
# model = get_model("excitation", 1, 150, 14, L=4)
#
# batch_scores = model(batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e)
# print(batch_scores.size())
#

# %% md

## Test backward pass

# %%

# # Loss
# J = nn.MSELoss()(batch_scores, batch_labels)
#
# # Backward pass
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer.zero_grad()
# J.backward()
# optimizer.step()
#
#


#%%
# Create model
# model = GatedGCN(input_dim=1, hidden_dim=config.hidden_dim, output_dim=8, L=4)
model = get_model(name=config['model'], input_dim=1, hidden_dim=config['hidden_dim'], output_dim=8, L=config['L'])

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['decay_steps'], gamma=0.8)


for epoch in range(50):
    start = time.time()
    print('starting epoch', epoch)
    train_loss = train(model, train_loader, loss, optimizer, scheduler)

    test_loss = evaluate(model, test_loader, loss)

    print(f'Epoch {epoch}, lr:{scheduler.get_last_lr()[0]}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}')
    # print(f'train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}')
    # if np.isnan([train_loss, test_loss]).any() or (test_acc>0.99 and epoch > 30):
    #     break

