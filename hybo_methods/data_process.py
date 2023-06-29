# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: data_process.py
@time: 2023/4/24 18:57
@e-mail: xansar@ruc.edu.cn
"""
import os
import dill
import numpy as np
from tqdm import tqdm


def read_data(pth):
    with open(os.path.join(pth, 'records_final.pkl'), 'rb') as fr:
        data = dill.load(fr)
    with open(os.path.join(pth, 'voc_final.pkl'), 'rb') as fr:
        voc = dill.load(fr)
    return data, voc


def data_split(data):
    # 这里数据划分好像有点问题，不是按照每个病人划分的，也没有shuffle
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point: split_point + eval_len]
    data_eval = data[split_point + eval_len:]
    return data_train, data_eval, data_test


def construct_graphs(data_train, n_diag):
    def jaccard_similarity(l1, l2):
        set1 = set(l1)
        set2 = set(l2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union

    # 构建诊断和手术联合超图
    hyper_edges_lst = []
    hyper_graph_coo = []  # [(item_idx, hyper_edge_idx)]
    train_data_with_h_edge_idx = [] # 原始train_data的每一个visit中有[diag, pro, drug]，这里变成[diag, pro, drug, h_edge_idx]
    for u, visits in enumerate(data_train):
        hyper_edges_in_cur_ehr = []
        visits_with_idx = []
        for t, v in enumerate(visits):
            # 这里v是[诊断，手术，药物]
            diag, pro, drug = v  # drug暂时用不到
            pro = [p + n_diag for p in pro]  # 将pro的编码与diag的编码接到一起，用来构建异构超图
            hyper_edge = sorted(diag + pro)  # 异构超边，包含diag和pro
            if hyper_edge not in hyper_edges_lst:
                hyper_edges_lst.append(hyper_edge)  # 加入超边集合
                hyper_edge_idx = len(hyper_edges_lst) - 1  # 此时新超边的序号就是超边列表的最后一个位置的序号
            else:
                hyper_edge_idx = hyper_edges_lst.index(hyper_edge)
            for item in hyper_edge:
                hyper_graph_coo.append([item, hyper_edge_idx])
            v.append(hyper_edge_idx)
            visits_with_idx.append(v)
        # 记录当前病人的visits包含了哪些超边
        train_data_with_h_edge_idx.append(visits_with_idx)


    # 构建线图
    # 基于hyper_edges_lst，计算任意一对超对之间的相似度
    line_graph_coo = []  # [(hyper_edge_idx_i, hyper_edge_idx_j, weight)]
    num = (len(hyper_edges_lst) - 1) * len(hyper_edges_lst) / 2
    bar = tqdm(total=num, desc='line graph')
    for i in range(len(hyper_edges_lst)):
        for j in range(i + 1, len(hyper_edges_lst)):
            edge_i = hyper_edges_lst[i]
            edge_j = hyper_edges_lst[j]
            sim = jaccard_similarity(edge_i, edge_j)
            line_graph_coo.append([i, j, sim])
            bar.update(1)

    # 将稀疏邻接矩阵转换成numpy array
    # 超图
    hyper_graph_coo = np.array(hyper_graph_coo, dtype=np.int32).T
    line_graph_coo = np.array(line_graph_coo, dtype=np.float32).T
    line_graph_weight = line_graph_coo[2, :]
    line_graph_coo = line_graph_coo[:2, :].astype(np.int32)
    return hyper_graph_coo, line_graph_coo, line_graph_weight, train_data_with_h_edge_idx

def save_data(pth, train_data_with_h_edge_idx, data_eval, data_test):
    with open(os.path.join(pth, 'hg_records_final.pkl'), 'wb') as fw:
        dill.dump({'train': train_data_with_h_edge_idx, 'eval': data_eval, 'test': data_test}, fw)


if __name__ == '__main__':
    data_dir = '../old_data/output'
    data, voc = read_data(data_dir)
    # 获取diag和pro实体的个数，将pro实体的编码放在diag后面
    n_diag = len(voc['diag_voc'].idx2word)
    n_pro = len(voc['pro_voc'].idx2word)
    # print(old_data)
    data_train, data_eval, data_test = data_split(data)
    # print(data_train)
    # print(data_eval)
    # print(data_test)
    hyper_graph_coo, line_graph_coo, line_graph_weight, train_data_with_h_edge_idx = construct_graphs(data_train, n_diag)
    # with open(os.path.join(data_dir, 'train_data_with_h_edge_idx.pkl'), 'wb') as fw:
    #     dill.dump(train_data_with_h_edge_idx, fw)
    np.savez(os.path.join(data_dir, 'graphs.npz'),
             hyper_graph_coo=hyper_graph_coo,
             line_graph_coo=line_graph_coo,
             line_graph_weight=line_graph_weight)
    save_data(data_dir, train_data_with_h_edge_idx, data_eval, data_test)
    graphs = np.load(os.path.join(data_dir, 'graphs.npz'))
    print(graphs)
