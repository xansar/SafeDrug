# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: H2DrugRec.py
@time: 2023/6/14 14:30
@e-mail: xansar@ruc.edu.cn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as tsp

from torch_geometric.nn.dense.linear import Linear as pyg_Linear
from torch_geometric.nn.inits import glorot

from .conv import HypergraphConv



class HyperDrugRec(nn.Module):
    def __init__(self, voc_size_dict, adj_dict, padding_dict, embedding_dim, n_heads, dropout, device):
        super(HyperDrugRec, self).__init__()
        self.name_lst = ['diag', 'proc', 'med']
        self.voc_size_dict = voc_size_dict
        self.num_dict = voc_size_dict
        self.device = device

        self.tensor_ddi_adj = adj_dict['ddi_adj'].to(device)  # 普通图
        self.adj_dict = {k: v.to(device) for k, v in adj_dict.items()}

        self.padding_dict = padding_dict


        self.n_heads = n_heads

        self.embedding_dim = embedding_dim
        self.embedding = nn.ModuleDict(
            {
                n: nn.Embedding(
                    num_embeddings=self.num_dict[n] + 1,
                    embedding_dim=embedding_dim,
                    padding_idx=self.padding_dict[n])
                for n in self.name_lst
            }
        )
        self.graph_encoder = nn.ModuleDict(
            {
                n: HypergraphConv(
                    in_channels=embedding_dim,
                    out_channels=embedding_dim,
                    use_attention=True,
                    heads=n_heads, dropout=dropout,
                    concat=False)
                for n in self.name_lst
            }
        )

        self.layernorm_1 = nn.ModuleDict(
            {
                n: nn.LayerNorm(embedding_dim)
                for n in self.name_lst
            }
        )
        self.layernorm_2 = nn.ModuleDict(
            {
                n: nn.LayerNorm(embedding_dim)
                for n in self.name_lst
            }
        )

        self.ffn = nn.ModuleDict(
            {
                n: nn.Linear(embedding_dim, embedding_dim)
                for n in self.name_lst
            }
        )

        self.node2edge_agg = nn.ModuleDict(
            {
                n: Node2EdgeAggregator(embedding_dim, embedding_dim, n_heads, dropout)
                for n in self.name_lst
            }
        )

        # 算序列依赖
        self.d_p_fusion = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(embedding_dim),

            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(embedding_dim),
        )

        self.d_p_seq_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.med_seq_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.patient_rep_layer = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(embedding_dim),

            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(embedding_dim),
        )

    def get_embedding(self):
        """
        获取embedding
        Returns:

        """
        x = {}
        for n in self.name_lst:
            idx = torch.arange(self.num_dict[n], dtype=torch.long, device=self.device)
            x[n] = self.embedding[n](idx)
        return x

    def get_hyperedge_representation(self, embed, adj):
        """
        获取超边的表示，通过聚合当前超边下所有item的embedding
        实际上就是乘以H(n_edges, n_items)
        Args:
            embed:
            adj:

        Returns:

        """

        # embed: n_items, dim
        n_items, n_edges = adj.shape
        norm_factor = (tsp.sum(adj, dim=0) ** -1).to_dense().reshape(n_edges, -1)

        assert norm_factor.shape == (n_edges, 1)
        omega = norm_factor * tsp.mm(adj.T, embed)
        return omega

    # 先写graph encode的部分
    def graph_encode(self, x, adj):
        """
        进行超图编码
        Args:
            x: dict
            adj: dict

        Returns:
            x: dict
        """
        for n in self.name_lst:
            adj_index = adj[n].indices()
            hyperedge_attr = self.get_hyperedge_representation(x[n], adj[n])
            x[n] = self.layernorm_1[n](x[n] + self.graph_encoder[n](x[n], adj_index, hyperedge_attr=hyperedge_attr))
            x[n] = self.layernorm_2[n](x[n] + self.ffn[n](x[n]))

        return x

    def node2edge(self, entity_seq_embed):
        """

        Args:
            entity_seq_embed: (bsz, max_vist, max_size, dim)
            records: (bsz, max_vist, max_size)

        Returns:

        """
        visit_seq_embed = {}
        for n in self.name_lst:
            # 还是先把数据展平, bsz, max_vist, max_size, dim
            seq_embed = entity_seq_embed[n]
            bsz, max_vist, max_size, dim = seq_embed.shape
            seq_embed = seq_embed.reshape(bsz * max_vist, max_size, dim)
            visit_seq_embed[n] = self.node2edge_agg[n](seq_embed).reshape(bsz, max_vist, dim)
        return visit_seq_embed

    def forward(self, records, masks):
        """

        Args:
            records: dict(diag, proc, med),其中每个元素shape为(bsz, max_vist, max_name_size)
                    对齐方式上,diag^t,proc^t和med^(t-1)对齐,第一次的med使用pad补
            masks: 多头注意力用

        Returns:

        """
        # 首先是embedding层
        raw_embedding = self.get_embedding()  # dict
        # 最开始是超图编码层
        graph_embedding = self.graph_encode(raw_embedding, self.adj_dict)
        # 现在有两组embedding,融合一下
        tmp_embedding = graph_embedding

        # 解析序列数据
        entity_seq_embed = {}  # (bsz, max_vist, max_size, dim)
        for n in self.name_lst:
            # 这里需要注意,因为records中含有pad,需要给embedding补一行
            padding_row = self.embedding[n](
                torch.tensor(self.num_dict[n], dtype=torch.long, device=self.device)).reshape(1, self.embedding_dim)
            tmp_embedding[n] = torch.vstack([tmp_embedding[n], padding_row])
            entity_seq_embed[n] = tmp_embedding[n][records[n]]

        # 首先是visit-level的数据表示
        # 使用多头注意力
        visit_seq_embed = self.node2edge(entity_seq_embed)  # bsz, max_visit, dim
        # 这里med单独拿出来作为用药史
        med_history = visit_seq_embed['med']  # 想做一个纯药物史解码器
        # 这里要注意最后一个时刻的药物实际上是看不到的,只能作为监督信息
        batch_size, _, dim = med_history.shape
        pad_head_med_history = torch.zeros(batch_size, 1, dim, dtype=med_history.dtype, device=med_history.device)
        med_history = torch.cat([pad_head_med_history, med_history], dim=1)[:, :-1, :]  # 这里就shift过了

        # 解码还是使用多头注意力,当前时刻的diag和proc作为query, 过去时刻的diag和proc作为key,过去时刻的med作为value
        diag_and_proc = torch.cat([visit_seq_embed['diag'], visit_seq_embed['proc']], dim=-1)
        # diag_and_proc = torch.cat([diag_and_proc, self.d_p_fusion(diag_and_proc)], dim=-1) # bsz, max_visit, 3 * dim
        diag_and_proc = visit_seq_embed['diag'] + visit_seq_embed['proc'] + self.d_p_fusion(
            diag_and_proc)  # bsz, max_visit, dim

        # 设计mask
        # 下面两步计算出来的是每个visit的上下文表示
        attn_mask = masks['attn_mask'].repeat(self.n_heads, 1, 1)
        d_p_rep = self.d_p_seq_attn(
            query=diag_and_proc, key=diag_and_proc, value=diag_and_proc, need_weights=False,
            # todo: 这里注意,med history的第一个是空的
            # key_padding_mask=masks['key_padding_mask'],   # 两个mask用一个就行
            # 用来遮挡key中的padding,和key一样的shape,这里应该需要把补足用的visitmask上,bsz_max_visit
            attn_mask=attn_mask,  # 加上会输出nan,因为有些地方的是补0的,整行都是True,相当于不计算
        )[0]
        med_rep = self.med_seq_attn(
            query=med_history, key=med_history, value=med_history, need_weights=False,
            # key_padding_mask=masks['key_padding_mask'],
            # 用来遮挡key中的padding,和key一样的shape,这里应该需要把补足用的visitmask上,bsz_max_visit
            attn_mask=attn_mask,  # 加上会输出nan,因为有些地方的是补0的,整行都是True,相当于不计算
        )[0]
        patient_rep = self.patient_rep_layer(torch.cat([d_p_rep, med_rep], dim=-1))  # 构建一个融合表示, bsz, max_visit, dim

        # 下面计算logits,分为两部分,一部分是patient_rep计算,一部分是直接用med计算
        med_embedding = tmp_embedding['med']
        patient_output = torch.matmul(patient_rep, med_embedding.T)
        med_output = torch.matmul(med_rep, med_embedding.T)

        # 两个logits相加
        output = patient_output + med_output
        # 这里要注意,最后一列是padding,需要去掉
        output = output[:, :, :-1]
        # 计算ddi-loss
        neg_pred_prob = F.sigmoid(output)
        neg_pred_prob = neg_pred_prob.unsqueeze(-1)
        neg_pred_prob = neg_pred_prob.transpose(-1, -2) * neg_pred_prob  # (bsz, max_visit, voc_size, voc_size)

        loss_mask = (masks['key_padding_mask'] == False).unsqueeze(-1).unsqueeze(-1)
        batch_neg = 0.0005 * (neg_pred_prob.mul(self.tensor_ddi_adj) * loss_mask).sum()

        return output, batch_neg


class Node2EdgeAggregator(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, dropout):
        super(Node2EdgeAggregator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.out_dim = out_channels // n_heads  # 分头后的空间
        self.dropout = dropout

        self.lin = pyg_Linear(self.out_dim, self.out_dim, bias=False,
                              weight_initializer='glorot')
        self.att = nn.Parameter(torch.Tensor(1, n_heads, 2 * self.out_dim))
        glorot(self.att)

    def forward(self, x):
        """

        Args:
            x: bsz, max_size, dim

        Returns:

        """
        # 首先使用平均计算超边表示,然后将节点表示和超边表示拼接起来计算注意力
        bsz, max_size, dim = x.shape
        x = x.reshape(-1, max_size, self.n_heads, self.out_channels // self.n_heads)  # bsz, max_size, head, out_dim
        x = self.lin(x)  # bsz, max_size, dim

        hyperedge_attr = x.mean(1)  # bsz, dim
        hyperedge_attr = self.lin(hyperedge_attr)
        hyperedge_attr = hyperedge_attr.reshape(-1, self.n_heads, self.out_channels // self.n_heads)
        hyperedge_attr = hyperedge_attr.unsqueeze(1).repeat(1, max_size, 1, 1)  # bsz, max_size, head, out_dim

        # 跟x拼接起来
        alpha = (torch.cat([x, hyperedge_attr], dim=-1) * self.att).sum(dim=-1)  # bsz, max_size, head
        alpha = F.leaky_relu(alpha)  # bsz, max_size, head
        alpha = torch.softmax(alpha, dim=1)  # bsz, max_size, head
        alpha = F.dropout(alpha, p=self.dropout, training=self.training).unsqueeze(-1)  # bsz, max_size, head, 1
        out = (alpha * x).sum(1)  # bsz, head, out_dim
        out = out.reshape(-1, self.out_channels)  # bsz, dim
        return out