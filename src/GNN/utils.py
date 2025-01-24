import numpy as np
import scipy.sparse as sp
import torch
import dgl

import os
from dgl.data import *
from sklearn.metrics import roc_auc_score


DATASET_LIST = [
    'Cornell', 'Texas', 'Wisconsin', 'Amazon', 'Actor'
]


def preprocess_data(
    dataset, 
    train_ratio,
    remove_zero_in_degree_nodes: bool = False,
    result_file_path: str = None,
    dataset_file_path: str = None
):

    if dataset in DATASET_LIST:
        npz_data = np.load(dataset_file_path + dataset + ".npz")
        edge = npz_data['edges']
        labels = npz_data['node_labels']
        features = npz_data['node_features']

        train_mask = npz_data['train_masks']
        val_mask = npz_data['val_masks']
        test_mask = npz_data['test_masks']

        g = dgl.graph((edge[0, :], edge[1, :]))
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)
        edges_trans = g.edges()
        src_nodes = edges_trans[0].numpy()
        dst_nodes = edges_trans[1].numpy()
        def edge_feature_calculation(edges):
            result = np.load(result_file_path + dataset + "/result.npy")
            matrix_yes_no = result[src_nodes, dst_nodes]
            matrix_yes_no_tensor = torch.tensor(matrix_yes_no, dtype=torch.float32)
            return {'yes_no': matrix_yes_no_tensor}

        g.apply_edges(edge_feature_calculation)

        train = np.flatnonzero(train_mask)
        val = np.flatnonzero(val_mask)
        test = np.flatnonzero(test_mask)

        nclass = len(np.unique(labels))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        train = torch.LongTensor(train)
        val = torch.LongTensor(val)
        test = torch.LongTensor(test)

        return g, nclass, features, labels, train, val, test
    else:
        raise ValueError(f'dataset {dataset} not supported in dataloader')


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


@torch.no_grad()
def accuracy(pr_logits, gt_labels):
    return (pr_logits.argmax(dim=-1) == gt_labels).float().mean().item()

@torch.no_grad()
def roc_auc(pr_logits, gt_labels):
    return roc_auc_score(gt_labels.cpu().numpy(), pr_logits[:, 1].cpu().numpy())   
