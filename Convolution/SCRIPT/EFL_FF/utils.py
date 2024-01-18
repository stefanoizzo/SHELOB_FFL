#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Anonymous
"""
# %%
# IMPORT SECTION
import sys
import os
import torch
import numpy as np
import seaborn as sns
from torch import Tensor
from typing import Tuple
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from matplotlib.collections import PatchCollection

# %%
_interactive_mode = 'ipykernel_launcher' in sys.argv[0] or \
                    (len(sys.argv) == 1 and sys.argv[0] == '')

if _interactive_mode:
    from tqdm.auto import tqdm, trange
else:
    from tqdm import tqdm, trange

def is_interactive():
    return _interactive_mode

def myshow(plot_show=True):
    if _interactive_mode and plot_show:
        plt.show()
    else:
        plt.close()

# %%

def separate_data(data, num_clients, num_classes, niid=False, 
                    least_samples=None, partition=None, beta=0.1, balance=False, 
                    class_per_client=2, save_fig=None):
    X = {}
    y = {}
    statistic = {}

    dataset_content, dataset_label = data

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                selected_clients = selected_clients[:int(num_clients/num_classes*class_per_client)]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        statistic.setdefault(client, [])
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs].copy()
        y[client] = dataset_label[idxs].copy()

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
        
    del data

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)
    print("Separate data finished!\n")

    ylabels = np.arange(num_clients)
    xlabels = np.arange(num_classes)

    x_mesh, y_mesh = np.meshgrid(np.arange(num_classes), np.arange(num_clients))
    s = np.zeros((num_clients,num_classes), dtype=np.uint16)
    for k_stat, v_stat in statistic.items():
        for elem in v_stat:
            s[k_stat, elem[0]] = elem[1]
    # c gestisce il colore dei cerchi
    if not niid:
        c = np.ones((num_clients,num_classes), dtype=np.uint16)
        R = s/s.max()/3
        viridis_cmap = cm.get_cmap('viridis')
        new_start = 0.5  
        new_end = 1.0    
        cm_color = cm.colors.LinearSegmentedColormap.from_list(
            'viridis_cut', viridis_cmap(np.linspace(new_start, new_end, 256))
        )
    else: 
        c = s
        R = s/s.max()/1.5
        cm_color = 'viridis'

    fig, ax = plt.subplots(figsize=(0.4*num_classes, 0.4*num_clients))
    circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flat, x_mesh.flat, y_mesh.flat)]
    col = PatchCollection(circles, array=c.flatten(), cmap=cm_color, zorder=10) 
    ax.add_collection(col)

    ax.set(title='Number of samples per class per client', xlabel='Classes', ylabel='Client ID')
    ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_clients),
        xticklabels=xlabels, yticklabels=ylabels)
    ax.set_xticks(np.arange(num_classes+1)-0.5, minor=True)
    ax.set_yticks(np.arange(num_clients+1)-0.5, minor=True)
    ax.grid(which='minor', zorder=0, alpha=0.5, color='white')
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_facecolor("#E8EAED")
    if save_fig is not None:
        plt.savefig(save_fig / 'samples_per_class_per_client.png', dpi=300)
    myshow()

    fig, ax = plt.subplots(figsize=(0.4*num_clients, 0.4*num_classes)) 
    circles = [plt.Circle((i,j), radius=r) for r, j, i in zip(R.flat, x_mesh.flat, y_mesh.flat)]
    col = PatchCollection(circles, array=c.flatten(), cmap=cm_color, zorder=10) 
    ax.add_collection(col)

    ax.set(xlabel='Client ID', ylabel='Classes')
    ax.set(xticks=np.arange(num_clients), yticks=np.arange(num_classes),
        xticklabels=ylabels, yticklabels=xlabels)
    ax.set_xticks(np.arange(num_clients+1)-0.5, minor=True)
    ax.set_yticks(np.arange(num_classes+1)-0.5, minor=True)
    # plt.xticks(rotation=70)
    ax.grid(which='minor', zorder=0, alpha=0.5, color='white')
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_facecolor("#E8EAED")
    if save_fig is not None:
        plt.savefig(save_fig / 'samples_per_class_per_client_inverted.png', dpi=300)
    myshow()

    return X, y, statistic


# %%
def plot_block_assignment(pd_block_assignment, output_path, num_blocks=None, name=''):
    # Define the colors
    if num_blocks is None:
        num_blocks = pd_block_assignment['n_blocks'].max()
    else:
        assert num_blocks >= pd_block_assignment['n_blocks'].max(), \
            'num_blocks must be greater than the maximum number of blocks assigned to a client'
    viridis_cmap = cm.get_cmap('viridis')
    new_start = 0.3  
    new_end = 0.8  
    new_cmap = cm.colors.LinearSegmentedColormap.from_list(
        'viridis_cut', viridis_cmap(np.linspace(new_start, new_end, 256))
    )
    new_palette = [new_cmap(i) for i in np.linspace(0, 1, num_blocks+1)]
    colors = [new_palette[i] for i in pd_block_assignment['n_blocks'].values]

    plt.figure(figsize=(6,3))
    sns.barplot(x=pd_block_assignment.index, y=pd_block_assignment['n_blocks'],
                palette=colors)
    plt.xlabel('Client ID')
    plt.ylabel('Number of blocks')
    plt.ylim(0, num_blocks)
    plt.savefig(os.path.join(output_path, f'block_assignment_{name}.png'), 
                bbox_inches='tight', dpi=300)
    myshow()

# %%
#create function to calculate Mahalanobis distance
def mahalanobis(x=None, data=None, cov=None):

    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    
    Returns the Mahalanobis distance of each observation vector in x.
    Note: if x and data have more than two dimensions, then they are reshaped to form 2D arrays preserving the last dimension.
    """
    
    if len(data.shape)>2:
        assert np.all(data.shape == x.shape), "x and data must have the same shape"
        x = np.reshape(x, (-1, x.shape[-1]))
        data = np.reshape(data, (-1, data.shape[-1]))
    
    x_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()

# %%
class My_TensorDataset():
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    def __init__(self, data: Tensor, target: Tensor, transform=None, 
                target_transform=None)-> None: 
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target)
        assert data.size(0) == target.size(0), "Size mismatch between tensors"
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        
        y = self.target[index]
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def __len__(self):
        return self.data.size(0)
