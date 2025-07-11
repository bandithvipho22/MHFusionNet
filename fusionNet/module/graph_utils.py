from __future__ import absolute_import

import torch
import numpy as np
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)

    # adj_mx = adj_mx * (1-torch.eye(adj_mx.shape[0])) + torch.eye(adj_mx.shape[0])

    return adj_mx


"Original"

# def adj_mx_from_skeleton(skeleton):
#     num_joints = skeleton.num_joints()
#     edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), skeleton.parents())))
#
#     adj = adj_mx_from_edges(num_joints, edges, sparse=False)
#
#     return adj


"Edited function"


def adj_mx_from_skeleton():
    num_joints = 18
    parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27,
               28, 27, 30]
    # parents = (0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), parents)))
    # edges = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6), (0, 7), (7, 17), (17, 8), (8, 9), (9, 10), (8, 14),
    #          (14, 15), (15, 16), (8, 11), (11, 12), (12, 13)]

    adj = adj_mx_from_edges(num_joints, edges, sparse=False)

    return adj
