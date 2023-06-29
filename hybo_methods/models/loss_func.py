# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: loss_func.py
@time: 2023/6/16 14:33
@e-mail: xansar@ruc.edu.cn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class H2LossFunc(nn.Module):
    def __init__(self, margin, tensor_ddi_adj, target_ddi, kp, multihot2idx, ddi_rate_score):
        super(H2LossFunc, self).__init__()
        self.margin = margin
        self.tensor_ddi_adj = tensor_ddi_adj
        self.target_ddi = target_ddi
        self.kp = kp
        self.multihot2idx = multihot2idx
        self.ddi_rate_score = ddi_rate_score

    def forward(self, output, loss_mask):
        pos_score = output['pos']
        neg_score = output['neg']
        margin_loss = self.compute_margin_loss(pos_score, neg_score, loss_mask)
        ddi_loss, current_ddi_rate = self.compute_ddi_loss(pos_score, loss_mask)

        if current_ddi_rate <= self.target_ddi:
            loss = margin_loss
        else:
            beta = min(0, 1 + (self.target_ddi - current_ddi_rate) / self.kp)
            loss = (
                    beta * margin_loss
                    + (1 - beta) * ddi_loss
            )

        return {
            'total': loss,
            'margin': margin_loss,
            'ddi': ddi_loss
        }

    def compute_ddi_loss(self, result, loss_mask):
        # 计算ddi-loss
        # 这里需要想一个方法,怎么让这个值变小一点
        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.unsqueeze(-1)
        neg_pred_prob = neg_pred_prob.transpose(-1, -2) * neg_pred_prob  # (bsz, max_visit, voc_size, voc_size)

        loss_mask = (loss_mask == False).unsqueeze(-1).unsqueeze(-1)
        if self.tensor_ddi_adj.device != neg_pred_prob.device:
            self.tensor_ddi_adj = self.tensor_ddi_adj.to(neg_pred_prob.device)
        ddi_loss = 0.0005 * (neg_pred_prob.mul(self.tensor_ddi_adj) * loss_mask).sum()

        result = F.sigmoid(result).detach().cpu().numpy()
        result[result >= 0.5] = 1
        result[result < 0.5] = 0
        y_label = self.multihot2idx(result)
        current_ddi_rate = self.ddi_rate_score(
            [y_label], path="../data/ddi_A_final.pkl"
        )
        visit_num = loss_mask.sum()
        return ddi_loss.sum() / visit_num, current_ddi_rate


    def compute_margin_loss(self, pos_score, neg_score, loss_mask):
        margin_loss = pos_score - neg_score + self.margin
        margin_loss[margin_loss < 0] = 0  # 这里score是距离,越小越好
        loss_mask = (loss_mask).unsqueeze(-1)  # bsz, maxvisit, 1
        margin_loss = margin_loss * loss_mask
        visit_num = loss_mask.sum()
        return margin_loss.sum() / visit_num
