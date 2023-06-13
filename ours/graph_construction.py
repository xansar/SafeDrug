# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: graph_construction.py
@time: 2023/6/8 14:58
@e-mail: xansar@ruc.edu.cn
"""
# -*- coding: utf-8 -*-
import os

import dgl
import numpy as np
import torch
from tqdm import tqdm


def factorize_and_recover(H, niter=10, k=10):
    """
    H: torch sparse 稀疏矩阵
    """
    m, n = H.shape
    q = min(6, m, n)
    U, S, V = torch.svd_lowrank(H, q=q, niter=niter)
    S = torch.diag(S)
    H_hat = U @ S @ V.T
    v_row, i_row = torch.topk(H_hat, k=k, dim=0)
    v_col, i_col = torch.topk(H_hat, k=k, dim=1)
    row_indices = torch.vstack([i_row.flatten(), torch.arange(n).repeat(k)])
    col_indices = torch.vstack([torch.arange(m).reshape(-1, 1).repeat(1, k).flatten(), i_col.flatten()])
    indices = torch.hstack([row_indices, col_indices])
    values = torch.ones_like(indices).float()[0]
    H_filter = torch.sparse_coo_tensor(indices, values, size=H.shape).coalesce()
    indices = H_filter.indices()
    values = torch.ones_like(H_filter.values())

    H_filter = torch.sparse_coo_tensor(indices, values, size=H.shape).coalesce()
    return H_filter

def construct_graphs(cache_pth, data_train, nums_dict, k=5):
    name_lst = ['diag', 'proc', 'med']

    if os.path.exists(cache_pth):
        coo_dict = dgl.data.utils.load_info(cache_pth)
    else:
        # 这里要注意，因为模型的embedding有pad token end token在，这里要注意n_diag的实际值
        # 构建诊断和手术联合超图

        coo_dict = {
            name: [] for name in name_lst
        }
        visit_num = 0
        for u, visits in enumerate(data_train):
            for t, v in enumerate(visits):
                # 这里v是[诊断，手术，药物]
                # 1961, 1433, 134

                set_dict = {
                    name_lst[i]: v[i]
                    for i in range(3)
                }
                hyper_edge = {
                    name: sorted(set_dict[name]) for name in name_lst
                }

                for n in name_lst:
                    for item in hyper_edge[n]:
                        coo_dict[n].append([item, visit_num])
                    # coo_dict[n].append(hyper_edge[n])

                visit_num += 1


        # 将稀疏邻接矩阵转换成numpy array
        # 超图
        coo_dict = {
            n: np.array(coo_dict[n], dtype=np.int64).T
            for n in name_lst
        }

        dgl.data.utils.save_info(
            cache_pth,
            coo_dict
        )

    H_dict = {}

    for n in name_lst:
        H_i = torch.from_numpy(coo_dict[n])
        H_v = torch.ones(H_i.shape[1])
        H = torch.sparse_coo_tensor(
            indices=H_i,
            values=H_v,
            size=(nums_dict[n], int(coo_dict[n].max(1)[1]) + 1)
        )
        H = factorize_and_recover(H, k=k)
        H_dict[n] = H

    return H_dict

def graph2hypergraph(adj):
    n_nodes = adj.shape[0]
    n_edges = adj.sum()

    idx = torch.nonzero(adj).T  # 这里需要转置一下
    H_v = adj[idx[0], idx[1]].repeat(2)

    node_idx = idx.reshape(-1)
    edge_idx = torch.arange(n_edges, device=adj.device).repeat(2).long()
    H_i = torch.vstack([node_idx, edge_idx])

    H = torch.sparse_coo_tensor(
        indices=H_i,
        values=H_v,
        size=(n_nodes, int(n_edges))
    )
    return H

if __name__ == '__main__':
    a = torch.randint(0, 2, (10, 15)).float()
    # mask = torch.rand_like(a) > 0.8
    # a = mask * a
    factorize_and_recover(a.to_sparse_coo())