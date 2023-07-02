# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: ehr_memory_attn.py
@time: 2023/6/25 14:17
@e-mail: xansar@ruc.edu.cn
"""
import torch.nn as nn

class EHRMemoryAttention(nn.Module):
    """
    这里做一个简易的attention+ffn,用的transformer架构
    """
    def __init__(self, embedding_dim, n_heads, dropout):
        super(EHRMemoryAttention, self).__init__()
        self.visit_mem_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim * 3,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Implementation of Feedforward model
        d_model = embedding_dim * 3
        dim_feedforward = embedding_dim * 3
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU()

    def forward(self, visit_rep, E_mem):
        """

        Args:
            visit_rep: 当前的visit的表示
            E_mem: ehr中的超边经过聚类得到的聚类中心,代表典型病例

        Returns:

        """
        x = visit_rep
        k = E_mem
        v = E_mem
        x = self.norm1(x + self._att_block(x, k, v))
        x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _att_block(self, x, k, v):
        x, attn = self.visit_mem_attn(x, k, v,
                           need_weights=True)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)