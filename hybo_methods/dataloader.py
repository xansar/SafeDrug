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
from config import parse_args
# from torch.nn.utils.rnn import pad_sequence
ARGS = parse_args()

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
    neg_drugs = []
    max_len_diags = -1
    max_len_pros = -1
    max_len_drugs = -1
    max_len_neg_drugs = -1

    loss_bce_target = torch.zeros(batch_size, max_T, ARGS.med_sz)
    loss_multi_target = torch.full((batch_size, max_T, ARGS.med_sz), -1)

    key_padding_mask = torch.zeros(batch_size, max_T).bool()
    attn_mask = torch.triu(torch.ones(batch_size, max_T, max_T), diagonal=1).bool()
    attn_mask += torch.tril(torch.ones(batch_size, max_T, max_T), diagonal=-ARGS.win_sz).bool()

    for i, x in enumerate(X):
        cur_diag = []
        cur_pro = []
        cur_drug = []
        cur_neg_drug = []

        key_padding_mask[i, length[i]:] = True
        attn_mask[i, length[i]:, :] = True
        attn_mask[i, :, length[i]:] = True
        attn_mask[i, torch.arange(max_T), torch.arange(max_T)] = False


        for j, v in enumerate(x):  # 遍历每个visit
            diag, pro, drug = v

            loss_bce_target[i, j, drug] = 1
            loss_multi_target[i, j, :len(drug)] = torch.tensor(drug)

            cur_diag.append(diag)
            cur_pro.append(pro)
            cur_drug.append(drug)
            neg_drug = [i not in drug for i in range(ARGS.med_sz)]
            cur_neg_drug.append(neg_drug)
            if len(diag) > max_len_diags:
                max_len_diags = len(diag)
            if len(pro) > max_len_pros:
                max_len_pros = len(pro)
            if len(drug) > max_len_drugs:
                max_len_drugs = len(drug)
            if len(neg_drug) > max_len_neg_drugs:
                max_len_neg_drugs = len(neg_drug)
        diags.append(cur_diag)
        pros.append(cur_pro)
        drugs.append(cur_drug)
        neg_drugs.append(cur_neg_drug)

    # pad
    def pad_seq(seqs, max_len):
        for i in range(batch_size):
            seq = seqs[i]
            for j in range(len(seq)):
                seq[j] = seq[j] + (max_len - len(seq[j])) * [pad_token]
            seqs[i] = seq + [[pad_token] * max_len] * (max_T - len(seq))
        return seqs



    diags = torch.tensor(pad_seq(diags, max_len_diags), dtype=torch.long)
    pros = torch.tensor(pad_seq(pros, max_len_pros), dtype=torch.long)
    drugs = torch.tensor(pad_seq(drugs, max_len_drugs), dtype=torch.long)
    neg_drugs = torch.tensor(pad_seq(neg_drugs, max_len_neg_drugs), dtype=torch.long)


    records = {
        'diag': diags,
        'proc': pros,
        'med': drugs,
        'neg_med': neg_drugs
    }
    masks = {
        'key_padding_mask': key_padding_mask,
        'attn_mask': attn_mask
    }
    target = {
        'loss_bce_target': loss_bce_target,
        'loss_multi_target': loss_multi_target
    }
    return records, masks, target


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
