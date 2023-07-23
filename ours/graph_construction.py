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
import torch.sparse as tsp
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt


def tf_idf_compute(H, k=5, filter=True):
    """

    Args:
        H: 邻接矩阵, 0是item,1是超边,值是顺序

    Returns:

    """
    n_entities, n_edges = H.shape
    mask_H = torch.sparse_coo_tensor(H.indices(), torch.ones_like(H.values()), size=H.shape)
    # 先统计每个entity出现的次数
    entity_cnt = tsp.sum(mask_H, dim=1).values()  # (n_entities, )
    # 用边数除以cnt,取log,得到逆频率
    inverse_entity_freq = torch.log(n_edges / entity_cnt + 1)
    # 用ndcg的算法计算超边内的顺序权重,用来替换tf
    assert H.values().min() == 1
    H_v = 1 / torch.log2(H.values() + 1)
    entity_idf = inverse_entity_freq[H.indices()[0]]
    # 将每一个超边的ndcg的值进行归一化处理,使得这些值加起来等于1
    H_ = torch.sparse_coo_tensor(H.indices(), H_v, size=H.shape).coalesce()
    # 首先计算每条超边的权重和
    edge_sum = tsp.sum(H_, dim=0).values()
    edge_sum = edge_sum[H.indices()[1]]
    # 计算tf-idf
    H_v = H_v / edge_sum * entity_idf
    H_res = torch.sparse_coo_tensor(H.indices(), H_v, size=H.shape).coalesce().to_dense()
    if filter:
        v_row, i_row = torch.topk(H_res, k=k, dim=0)
        v_col, i_col = torch.topk(H_res, k=k, dim=1)
        row_indices = torch.vstack([i_row.flatten(), torch.arange(n_edges).repeat(k)])
        col_indices = torch.vstack([torch.arange(n_entities).reshape(-1, 1).repeat(1, k).flatten(), i_col.flatten()])
        indices = torch.hstack([row_indices, col_indices])
        values = torch.hstack([v_row.flatten(), v_col.flatten()])
        key = values != 0  # 只取非0元素
        indices = indices[:, key]
        values = values[key]
        # row_values = v_row.flatten()
        H_res = torch.sparse_coo_tensor(indices, values, size=H.shape).coalesce()
    else:
        H_res = H_res.to_sparse_coo()
    return H_res


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

def construct_graphs_v2(cache_pth, data_train, nums_dict):
    """
    问题: 太稠密了
    Args:
        cache_pth:
        data_train:
        nums_dict:

    Returns:

    """
    name_lst = ['diag', 'proc', 'med']

    graphs_pth = os.path.join(cache_pth, f'cocur_graphs.pkl')
    # if os.path.exists(graphs_pth):
    if False:
        graphs = torch.load(graphs_pth)
        H_dict = graphs['H_dict']
        cluster_dict = graphs['cluster_dict']
    else:
        # 这里要注意，因为模型的embedding有pad token end token在，这里要注意n_diag的实际值
        # 构建诊断和手术联合超图

        coo_dict = {
            name: [] for name in name_lst
        }
        visit_num = 0
        Med_Diag_matrix = torch.zeros(nums_dict['med'], nums_dict['diag'])
        Med_Proc_matrix = torch.zeros(nums_dict['med'], nums_dict['proc'])
        Med_Med_matrix = torch.zeros(nums_dict['med'], nums_dict['med'])
        for u, visits in enumerate(data_train):
            for t, v in enumerate(visits):
                # 这里v是[诊断，手术，药物]
                # 分解构建md,mp,mm共现矩阵
                set_dict = {
                    name_lst[i]: v[i]
                    for i in range(3)
                }
                # 填充共现矩阵
                for m in set_dict['med']:
                    ## med diag
                    for d in set_dict['diag']:
                        Med_Diag_matrix[m, d] += 1
                    ## med proc
                    for p in set_dict['proc']:
                        Med_Proc_matrix[m, p] += 1
                    ## med med
                    for m in set_dict['med']:
                        Med_Med_matrix[m, m] += 1

        def generate_H(matrix):
            matrix /= torch.sum(matrix, 1, keepdim=True)
            matrix = matrix.to_sparse_coo()
            return matrix

        Med_Diag_matrix = generate_H(Med_Diag_matrix)
        Med_Proc_matrix = generate_H(Med_Proc_matrix)
        Med_Med_matrix = generate_H(Med_Med_matrix)

        H_dict = {
            'md': Med_Diag_matrix,
            'mp': Med_Proc_matrix,
            'mm': Med_Med_matrix
        }

        torch.save(
            {
                'H_dict': H_dict,
            },
            graphs_pth
        )

    H_dict = {k: v.coalesce() for k, v in H_dict.items()}
    # cluster_dict = {k: v.long() for k, v in cluster_dict.items()}
    return H_dict


def construct_graphs_v3(cache_pth, data_train, nums_dict):
    name_lst = ['diag', 'proc', 'med']

    graphs_pth = os.path.join(cache_pth, f'fusion_graph.pkl')
    # if os.path.exists(graphs_pth):
    if False:
        graphs = torch.load(graphs_pth)
        H_dict = graphs['H_dict']
        cluster_dict = graphs['cluster_dict']
    else:
        # 这里要注意，因为模型的embedding有pad token end token在，这里要注意n_diag的实际值
        # 构建诊断和手术联合超图

        coo = []
        visit_num = 0
        for u, visits in enumerate(data_train):
            for t, v in enumerate(visits):
                # 这里v是[诊断，手术，药物]
                diag, proc, med = v
                # 将三条边拼接在一起,顺序是diag,proc,med,所以proc和med的idx分别要做平移
                diag = diag
                proc = [p + nums_dict['diag'] for p in proc]
                med = [m + nums_dict['diag'] + nums_dict['med'] for m in med]
                hyper_edge = diag + proc + med

                for idx, item in enumerate(hyper_edge):
                    coo.append([item, visit_num])

                visit_num += 1

        # 将稀疏邻接矩阵转换成numpy array
        # 超图
        coo = np.array(coo, dtype=np.int64).T
        H_i = torch.from_numpy(coo)
        H_v = torch.ones(H_i.shape[1])
        H = torch.sparse_coo_tensor(
            indices=H_i,
            values=H_v,
            size=(nums_dict['diag'] + nums_dict['proc'] + nums_dict['med'], visit_num)
        ).coalesce().float()

        # torch.save(
        #     {
        #         'H_dict': H_dict,
        #     },
        #     graphs_pth
        # )
    return H

def construct_graphs(cache_pth, data_train, nums_dict, k=5, n_clusters=1000):
    name_lst = ['diag', 'proc', 'med']

    graphs_pth = os.path.join(cache_pth, f'{n_clusters}_graphs.pkl')
    # if os.path.exists(graphs_pth):
    if False:
        graphs = torch.load(graphs_pth)
        H_dict = graphs['H_dict']
        cluster_dict = graphs['cluster_dict']
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

                set_dict = {
                    name_lst[i]: v[i]
                    for i in range(3)
                }

                hyper_edge = {
                    # name: sorted(set_dict[name]) for name in name_lst
                    name: set_dict[name] for name in name_lst
                }

                for n in name_lst:
                    for idx, item in enumerate(hyper_edge[n]):
                        coo_dict[n].append([item, visit_num])
                        # coo_dict[n].append([item, visit_num, idx + 1])   # 这里把顺序加进来
                    # coo_dict[n].append(hyper_edge[n])

                visit_num += 1

        # 将稀疏邻接矩阵转换成numpy array
        # 超图
        coo_dict = {
            n: np.array(coo_dict[n], dtype=np.int64).T
            for n in name_lst
        }

        # dgl.data.utils.save_info(
        #     cache_pth,
        #     coo_dict
        # )

        H_dict = {}
        cluster_dict = {}

        for n in name_lst:
            H_i = torch.from_numpy(coo_dict[n])
            H_v = torch.ones(H_i.shape[1])
            # H_i = torch.from_numpy(coo_dict[n])[:2, :]
            # H_v = torch.from_numpy(coo_dict[n])[2, :]
            H = torch.sparse_coo_tensor(
                indices=H_i,
                values=H_v,
                size=(nums_dict[n], int(coo_dict[n].max(1)[1]) + 1)
            ).coalesce()
            # H = factorize_and_recover(H, k=k)
            # H = tf_idf_compute(H, filter=False)
            # 考虑对超图进行过滤,缩小边的数量
            # labels = cluster_hyperedges(H, n_clusters=n_clusters, dim=64)
            # cluster_dict[n] = torch.from_numpy(labels)
            H_dict[n] = H.float()
            # H = cluster_hyperedges(H, 256, 64)

            # tsne_visual(H.to_dense().T)

        torch.save(
            {
                'H_dict': H_dict,
                'cluster_dict': cluster_dict
            },
            graphs_pth
        )

    H_dict = {k: v.coalesce() for k, v in H_dict.items()}
    # cluster_dict = {k: v.long() for k, v in cluster_dict.items()}
    return H_dict, cluster_dict


def cluster_hyperedges(H, n_clusters, dim):
    assert H.is_sparse
    # X = torch.sparse.mm(H.T, H).coalesce().to_dense().numpy()
    # X = X[:100, :100]
    X = H.T.coalesce().to_dense().numpy()
    # X = X[:100, :100]
    low_dim_X = pca(X, dim)
    model = SpectralClustering(n_clusters=n_clusters,
                               assign_labels='cluster_qr',
                               random_state=0,
                               )
    labels = model.fit_predict(low_dim_X)
    label_lst = np.unique(labels)
    # 需要做成一个n_cluster,max_cluster_size的矩阵,空的地方补0
    max_size = -1
    clusters = []
    sample_idx = np.arange(len(labels))
    for id, label in enumerate(label_lst):
        index = sample_idx[labels == label]
        clusters.append(index)
        if len(index) > max_size:
            max_size = len(index)
    clusters_matrix = np.zeros((len(label_lst), max_size))
    for i, clu in enumerate(clusters):
        length = len(clu)
        clusters_matrix[i][:length] = clu
    # visual_X = tsne(low_dim_X, 2)
    # label_lst = np.unique(labels)
    # for id, label in enumerate(label_lst):
    #     index = labels == label
    #     plt.scatter(visual_X[index, :][:, 0], visual_X[index, :][:, 1], marker='o', s=4)
    # plt.show()

    return clusters_matrix


def pca(x, dim):
    model = PCA(n_components=dim)
    low_x = model.fit_transform(x)
    return low_x


def tsne(x, dim):
    # t-SNE的降维与可视化
    assert dim < 4
    ts = TSNE(n_components=dim, init='pca', random_state=0)
    # 训练模型
    y = ts.fit_transform(x)
    return y

def visualization(X, title=None):
    X = X.detach().cpu().numpy()
    visual_X = tsne(X, 2)
    plt.scatter(visual_X[:, 0], visual_X[:, 1], marker='o', s=4)
    plt.title(title)
    plt.show()


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
    ).coalesce()
    return H


def desc_hypergraph_construction(desc2idx_dict, entity_num):
    indices = []
    edge_num = 0
    for desc, idx_lst in desc2idx_dict.items():
        # 这里需要构建indices,行是实体(idx)列是超边(这里是desc)
        for idx in idx_lst:
            indices.append([idx, desc])
        edge_num += 1
    indices = np.array(indices, dtype=np.int64).T
    H_i = torch.from_numpy(indices)
    H_v = torch.ones(H_i.shape[1])
    H = torch.sparse_coo_tensor(
        indices=H_i,
        values=H_v,
        size=(entity_num, edge_num)
    )
    return H


if __name__ == '__main__':
    a = torch.randint(0, 2, (10, 15)).float()
    # mask = torch.rand_like(a) > 0.8
    # a = mask * a
    factorize_and_recover(a.to_sparse_coo())
