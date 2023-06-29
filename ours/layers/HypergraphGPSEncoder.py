# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: HypergraphGPSEncoder.py
@time: 2023/6/22 15:51
@e-mail: xansar@ruc.edu.cn
"""
import os.path

import torch
import torch.sparse as tsp
import torch.nn as nn
from .HypergraphConv import HypergraphConv
from .position_encoding import FeatureEncoder


class MPNN(nn.Module):
    """
    local MPNN part
    """

    def __init__(self, embed_dim, n_heads, dropout):
        super(MPNN, self).__init__()
        self.conv = HypergraphConv(
            in_channels=embed_dim,
            out_channels=embed_dim,
            use_attention=True,
            heads=n_heads, dropout=dropout,
            concat=False
        )
        # 超边特征计算,对所有节点做线性变换然后求和,加上对超边特征变换求和
        # 原始GatedGCN中分别对头尾节点使用不同的变换,这里统一成相同的
        self.node_ffn = nn.Linear(embed_dim, embed_dim)
        self.edge_ffn = nn.Linear(embed_dim, embed_dim)

    def compute_edge_feat(self, X, E, A):
        """
        获取超边的表示，通过聚合当前超边下所有item的embedding
        实际上就是乘以H(n_edges, n_items)
        Args:
            X: 节点特征
            E: 边特征
            A:

        Returns:

        """

        # embed: n_items, dim
        n_items, n_edges = A.shape
        norm_factor = (tsp.sum(A, dim=0) ** -1).to_dense().reshape(n_edges, -1)

        assert norm_factor.shape == (n_edges, 1)
        X_trans = self.node_ffn(X)
        # E计算:变换后的节点聚合,原始E变换,原始E
        E_res = norm_factor * tsp.mm(A.T, X_trans) + self.edge_ffn(E) + E
        return E_res

    def forward(self, X, E, A):
        adj_index = A.indices()
        hyperedge_weight = A.values()
        E_res = self.compute_edge_feat(X, E, A)
        X_res = self.conv(X, adj_index, hyperedge_attr=E_res, hyperedge_weight=hyperedge_weight)
        return X_res, E_res


class GlobalAttention(nn.Module):
    """
    global attention part
    """

    def __init__(self, embed_dim, n_heads, dropout):
        super(GlobalAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, X, ke_bias=None):
        X_attn = self._sa_block(X, ke_bias, None)  # 这里attn_mask如果是float类型,可以直接加到attn_weights上面
        return X_attn

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        # Requires PyTorch v1.11+ to support `average_attn_weights=False`
        # option to return attention weights of individual heads.
        x, A = self.self_attn(x, x, x,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=True,
                              average_attn_weights=False)
        # self.attn_weights = A.detach().cpu()
        return x


class FeedForwardLayer(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(FeedForwardLayer, self).__init__()
        # Feed Forward block.
        self.ff_linear1 = nn.Linear(embed_dim, embed_dim * 2)
        self.ff_linear2 = nn.Linear(embed_dim * 2, embed_dim)
        self.act_fn_ff = nn.LeakyReLU()
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))


class HypergraphGPSEncoderLayer(nn.Module):
    """
    参考GraphGPS,变成超图
    """

    def __init__(self, embed_dim, n_heads, dropout):
        super(HypergraphGPSEncoderLayer, self).__init__()
        self.MPNN_layer = MPNN(embed_dim, n_heads, dropout)
        self.global_att = GlobalAttention(embed_dim, n_heads, dropout)

        self.local_bn = nn.BatchNorm1d(embed_dim)
        self.local_dropout = nn.Dropout(dropout)
        self.global_bn = nn.BatchNorm1d(embed_dim)
        self.global_dropout = nn.Dropout(dropout)

        self.node_ff_block = FeedForwardLayer(embed_dim, dropout)
        self.node_norm = nn.BatchNorm1d(embed_dim)

    def forward(self, input_params, ke_bias):
        """

        Args:
            input_params: X, E, A
            X: 节点特征
            E: 边特征
            A: 邻接矩阵
        Returns:

        """
        X, E, A = input_params
        X_M_hat, E_res = self.MPNN_layer(X, E, A)
        X_M = self.local_bn(self.local_dropout(X_M_hat) + X)

        X_T_hat = self.global_att(X, ke_bias)
        X_T = self.global_bn(self.global_dropout(X_T_hat) + X)

        X_res = self.node_norm(self.node_ff_block(X_M + X_T) + X)
        output = (X_res, E_res, A)
        return output


class HypergraphGPSEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, n_layers, H, idx2word, cache_dir, device, name):
        super(HypergraphGPSEncoder, self).__init__()
        self.n_layers = n_layers
        self.name = name
        cache_dir = os.path.join(cache_dir, name)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.feature_encoder = FeatureEncoder(
            H=H,
            idx2word=idx2word,
            se_dim=embed_dim,
            pe_dim=embed_dim,
            ke_dim=embed_dim,
            cache_dir=cache_dir,
            device=device,
            name=name,
        )

        self.encoders = nn.ModuleList()
        for i in range(n_layers):
            self.encoders.append(
                HypergraphGPSEncoderLayer(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    dropout=dropout
                )
            )

    def forward(self, X, E, H):
        """

        Args:
            X: 节点特征
            E: 边特征
            H: 邻接矩阵
            egivecs: 超图拉普拉斯矩阵特征向量, [num_nodes, max_freqs]

        Returns:

        """
        side_encodings, ke_bias = self.feature_encoder()
        pe_encoding, se_encoding, ke_encoding = side_encodings['pe'], side_encodings['se'], side_encodings['ke']
        X = X + pe_encoding + se_encoding + ke_encoding
        X_lst = [X]
        E_lst = [E]
        for i in range(self.n_layers):
            input_params = X, E, H
            layer = self.encoders[i]
            X, E, H = layer(input_params, ke_bias)
            X_lst.append(X)
            E_lst.append(E)
        X = sum(X_lst) / (self.n_layers + 1)
        E = sum(E_lst) / (self.n_layers + 1)

        return X, E
