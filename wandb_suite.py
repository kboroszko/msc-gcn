from random import random

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import os
import gated_gcn
import numpy as np

os.environ['DGLBACKEND'] = 'pytorch'  # tell DGL what backend to use
import dgl
from dgl import DGLGraph
from dgl.data import MiniGCDataset


import time


#%%

class satCheck():

    def __init__(self, length, steps_to_stop):
        self.vlog = []
        self.length = length
        self.mean_log = []
        self.sts = steps_to_stop

    def log(self, v):
        if len(self.vlog) > self.length:
            self.vlog = self.vlog[1:] + [v]
        else:
            self.vlog.append(v)
        self.mean_log.append(np.mean(self.vlog))
        return (np.diff(self.mean_log[-self.sts:]) > 0).any() or len(self.mean_log) < self.length



#%%
def accuracy(logits, targets):
    preds = logits.detach().argmax(dim=1)
    acc = (preds == targets).sum().item()
    return acc


# Collate function to prepare graphs

def collate(samples):
    graphs, labels = map(list, zip(*samples))  # samples is a list of pairs (graph, label)
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
        batch_X = batch_graphs.ndata['feat']
        batch_E = batch_graphs.edata['feat']

        batch_scores = model(batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e)
        J = loss(batch_scores, batch_labels.long())
        optimizer.zero_grad()
        J.backward()
        optimizer.step()

        epoch_loss += J.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)

    scheduler.step()

    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc


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
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc


#%%

sweep_config = {
    'method': 'grid', #grid, random
    'metric': {
      'name': 'loss',
      'goal': 'minimize'
    },
    'parameters': {
        'decay_steps': {
            'values': [20,40, 60]
        },
        'learning_rate': {
            'values': [3e-4, 1e-5]
        },
        'hidden_dim':{
            'values':[64,128,256]
        },
        'model':{
            'values':['gated', 'normal']
        },
        'L':{
            'values':[2,4,8]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="scheduler")

#%%

# create artifical data feature (= in degree) for each node
def create_artificial_features(dataset):
    for (graph, _) in dataset:
        graph.ndata['feat'] = graph.in_degrees().view(-1, 1).float()
        graph.edata['feat'] = torch.ones(graph.number_of_edges(), 1)
    return dataset


# Generate artifical graph dataset with DGL
trainset = MiniGCDataset(350, 10, 20)
testset = MiniGCDataset(150, 10, 20)

trainset = create_artificial_features(trainset)
testset = create_artificial_features(testset)
#%%

def get_model(name,input_dim, hidden_dim, output_dim, L):
    if name == "normal":
        return gated_gcn.NormalGCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, L=L)
    else:
        return gated_gcn.GatedGCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, L=L)


#%%
def train_wandb():
    config_defaults = {
        'epochs': 150,
        'learning_rate': 1e-5,
        'hidden_dim': 128,
        'model': 'normal',
        'decay_steps': 20,
        'L':4
    }

    saturation = satCheck(10, 7)

    wandb.init(config=config_defaults)


    config = wandb.config


    wandb.config.name = f"{config.model}-hd{config.hidden_dim}-ds{config.decay_steps}-L{config.L}-{random.rand() : .4f}"

    # datasets
    train_loader = DataLoader(trainset, batch_size=50, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(testset, batch_size=50, shuffle=False, collate_fn=collate)

    # Create model
    # model = GatedGCN(input_dim=1, hidden_dim=config.hidden_dim, output_dim=8, L=4)
    model = get_model(name=config.model, input_dim=1, hidden_dim=config.hidden_dim, output_dim=8, L=config.L)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_steps, gamma=0.8)

    test_acc_was_1 = False

    for epoch in range(150):
        start = time.time()
        train_loss, train_acc = train(model, train_loader, loss, optimizer, scheduler)

        test_loss, test_acc = evaluate(model, test_loader, loss)
        test_acc_was_1 = test_acc_was_1 or test_acc > 0.999
        if test_acc_was_1:
            wandb.log({
                "test_acc": 1,
            })
        else:
            wandb.log({
                "test_acc": test_acc,
            })

        wandb.log({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        print(f'Epoch {epoch}, lr:{scheduler.get_last_lr()[0]}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}')
        print(f'train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}')
        if np.isnan([train_loss, test_loss]).any() or (test_acc>0.99 and epoch > 30):
            break
        wandb.config.epochs = epoch
#%%
wandb.agent(sweep_id, train_wandb)

#%%
# optimizer = optim.SGD(get_model("normal", 1, 100, 8, 4).parameters(), lr=0.1)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
#
# for i in range(15):
#     lr = scheduler.get_last_lr()[0] # latest pytorch 1.5+ uses get_last_lr,  previously it was get_lr iirc;
#     lr1 = optimizer.param_groups[0]['lr'] # either the above line or this, both should do the same thing
#     print(i, lr, lr1)
#     scheduler.step()
