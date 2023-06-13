# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: dataloader.py
@time: 2023/4/25 16:30
@e-mail: xansar@ruc.edu.cn
"""
from torch.utils.data import Dataset
import torch
import os
import dill


class MIMICDataset(Dataset):
    def __init__(self, data):
        super(MIMICDataset, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def collate_fn(X):
    pad_token = -1
    length = [len(x) for x in X]
    max_T = max(length)
    batch_size = len(X)
    # 需要diags，pros，drugs还有hyper_edges
    # diags: (batch_size, max_T, max_len_diags),使用pad
    diags = []
    pros = []
    drugs = []
    h_edges = []    # batch_size, max_T
    max_len_diags = -1
    max_len_pros = -1
    max_len_drugs = -1
    for x in X:
        cur_diag = []
        cur_pro = []
        cur_drug = []
        cur_h_edge =[]
        for v in x:  # 遍历每个visit
            diag, pro, drug, h_edge = v
            cur_diag.append(diag)
            cur_pro.append(pro)
            cur_drug.append(drug)
            cur_h_edge.append(h_edge)
            if len(diag) > max_len_diags:
                max_len_diags = len(diag)
            if len(pro) > max_len_pros:
                max_len_pros = len(pro)
            if len(drug) > max_len_drugs:
                max_len_drugs = len(drug)
        diags.append(cur_diag)
        pros.append(cur_pro)
        drugs.append(cur_drug)
        h_edges.append(cur_h_edge)

    # pad
    def pad_seq(seqs, max_len):
        for i in range(batch_size):
            seq = seqs[i]
            for j in range(len(seq)):
                seq[j] = seq[j] + (max_len - len(seq[j])) * [pad_token]
            seqs[i] = seq + [[pad_token] * max_len] * (max_T - len(seq))
        return seqs

    diags = pad_seq(diags, max_len_diags)
    pros = pad_seq(pros, max_len_pros)
    drugs = pad_seq(drugs, max_len_drugs)
    h_edges = [h_edge_seq + [pad_token] * (max_T - len(h_edge_seq)) for h_edge_seq in h_edges]

    return {
        'diags': torch.tensor(diags, dtype=torch.long),
        'pros': torch.tensor(pros, dtype=torch.long),
        'drugs': torch.tensor(drugs, dtype=torch.long),
        'h_edges': torch.tensor(h_edges, dtype=torch.long),
    }


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data_process import read_data

    data_dir = '../old_data/output'
    data, voc = read_data(data_dir)
    with open(os.path.join(data_dir, 'train_data_with_h_edge_idx.pkl'), 'rb') as fr:
        train_data_with_h_edge_idx = dill.load(fr)

    dataset = MIMICDataset(train_data_with_h_edge_idx)
    data_loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
    for item in data_loader:
        print(item)
