# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: hgt_decoder.py
@time: 2023/7/12 14:02
@e-mail: xansar@ruc.edu.cn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .HypergraphGPSEncoder import Node2EdgeAggregator
from info_nce import InfoNCE
from .ehr_memory_attn import EHRMemoryAttention, HistoryAttention
from .moe import MoEPredictor

class HGTDecoder(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout, n_experts, experts_k, n_ehr_edges, voc_size_dict, padding_dict, device, X_hat, E_mem, ddi_adj):
        super(HGTDecoder, self).__init__()
        self.name_lst = ['diag', 'proc', 'med']
        self.num_dict = voc_size_dict
        self.voc_size_dict = voc_size_dict
        self.n_ehr_edges = n_ehr_edges
        self.embedding_dim = embedding_dim
        self.padding_dict = padding_dict
        self.device = device
        # self.X_hat = {k: v.to(device) for k, v in X_hat.items()}
        for n in self.name_lst:
            if X_hat[n].shape[0] != self.num_dict[n] + 1:
                padding_row = torch.zeros(1, self.embedding_dim).to(self.device)
                X_hat[n] = torch.vstack([X_hat[n].to(device), padding_row])

        self.X_hat = nn.ModuleDict({
            n: nn.Embedding(voc_size_dict[n] + 1, embedding_dim).from_pretrained(X_hat[n].to(device), freeze=False)
            for n in self.name_lst
        })
        self.E_mem = nn.ModuleDict({
            n: nn.Embedding(E_mem[n].shape[0], embedding_dim).from_pretrained(E_mem[n].to(device), freeze=False)
            for n in ['dp', 'm']
        })
        # self.X_hat = nn.ParameterDict({
        #     k: nn.Parameter(v).to(device)
        #     # k: nn.Parameter(torch.randn_like(v)).to(device)
        #     for k, v in X_hat.items()
        # })
        # self.E_mem = {k: v.to(device) for k, v in E_mem.items()}
        # self.E_mem = nn.ParameterDict({
        #     k: nn.Parameter(v).to(device)
        #     # k: nn.Parameter(torch.randn_like(v)).to(device)
        #     for k, v in E_mem.items()
        # })

        self.tensor_ddi_adj = ddi_adj.to(device)
        self.embedding_norm = nn.ModuleDict(
            {
                n: nn.Sequential(
                    nn.LayerNorm(embedding_dim),
                    nn.Dropout(dropout)
                )
                # n: nn.LayerNorm(embedding_dim)
                for n in self.name_lst
            }
        )

        self.node2edge_agg = nn.ModuleDict(
            {
                n: Node2EdgeAggregator(embedding_dim, n_heads, dropout)
                for n in self.name_lst
            }
        )
        # self.node2edge_agg = node2edge_agg

        self.visit_mem_attn = EHRMemoryAttention(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            dropout=dropout
        )

        self.med_context_attn = HistoryAttention(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        # self.med_context_attn = nn.MultiheadAttention(
        #     # embed_dim=embedding_dim * 3,
        #     embed_dim=embedding_dim,
        #     num_heads=n_heads,
        #     dropout=dropout,
        #     batch_first=True,
        # )

        self.dp_context_attn = HistoryAttention(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        # self.dp_context_attn = nn.MultiheadAttention(
        #     # embed_dim=embedding_dim * 3,
        #     embed_dim=embedding_dim,
        #     num_heads=n_heads,
        #     dropout=dropout,
        #     batch_first=True,
        # )

        self.mem_context_attn = HistoryAttention(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        # self.mem_context_attn = nn.MultiheadAttention(
        #     # embed_dim=embedding_dim * 3,
        #     embed_dim=embedding_dim,
        #     num_heads=n_heads,
        #     dropout=dropout,
        #     batch_first=True,
        # )

        # self.dp_output_norm = nn.LayerNorm(embedding_dim)
        # self.mem_output_norm = nn.LayerNorm(embedding_dim)
        # self.med_output_norm = nn.LayerNorm(embedding_dim)
        #
        # self.dp_moe_preditor = MoEPredictor(
        #     embed_dim=embedding_dim,
        #     output_size=voc_size_dict['med'],
        #     n_experts=n_experts,
        #     k=experts_k,
        #     dropout=dropout
        # )
        #
        # self.mem_moe_preditor = MoEPredictor(
        #     embed_dim=embedding_dim,
        #     output_size=voc_size_dict['med'],
        #     n_experts=n_experts,
        #     k=experts_k,
        #     dropout=dropout,
        # )
        #
        # self.med_moe_preditor = MoEPredictor(
        #     embed_dim=embedding_dim,
        #     output_size=voc_size_dict['med'],
        #     n_experts=n_experts,
        #     k=experts_k,
        #     dropout=dropout,
        # )
        #
        # self.dp_pred_norm = nn.LayerNorm(self.voc_size_dict['med'])
        # self.mem_pred_norm = nn.LayerNorm(self.voc_size_dict['med'])
        # self.med_pred_norm = nn.LayerNorm(self.voc_size_dict['med'])

        # self.gate_control = nn.Sequential(
        #     nn.Linear(3 * self.voc_size_dict['med'], 3 * self.voc_size_dict['med']),
        #     nn.Dropout(dropout),
        #     nn.ReLU(),
        # )

        self.fusion_moe_preditor = MoEPredictor(
            embed_dim=embedding_dim,
            output_size=voc_size_dict['med'],
            n_experts=n_experts,
            k=experts_k,
            dropout=dropout,
        )

        self.fusion_pred_norm = nn.LayerNorm(self.voc_size_dict['med'])

        self.gate_control = nn.Sequential(
            nn.Linear(3 * embedding_dim, 3 * embedding_dim),
            nn.Dropout(dropout),
            nn.Tanh()
        )

        self.info_nce_loss = InfoNCE(reduction='mean')

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

    def forward(self, records, masks, true_visit_idx, visit2edge_idx):
        assert len(visit2edge_idx) == true_visit_idx.sum().item()
        X_hat = self.X_hat
        E_mem = {
            'dp': self.E_mem['dp'](torch.arange(self.n_ehr_edges).to(self.device)),
            'm': self.E_mem['m'](torch.arange(self.n_ehr_edges).to(self.device))
        }
        # 解析序列数据
        entity_seq_embed = {}  # (bsz, max_vist, max_size, dim)
        for n in self.name_lst:
            # if X_hat[n].shape[0] != self.num_dict[n] + 1:
            #     padding_row = torch.zeros(1, self.embedding_dim).to(self.device)
            #     X_hat[n] = torch.vstack([X_hat[n], padding_row])
            entity_seq_embed[n] = self.embedding_norm[n](X_hat[n](records[n]))


        # 首先是visit-level的数据表示
        # 使用多头注意力
        visit_seq_embed = self.node2edge(entity_seq_embed)  # bsz, max_visit, dim
        # 先上下文再记忆
        # 这里将diag和proc还有历史用药拼起来
        visit_rep = visit_seq_embed['diag'] + visit_seq_embed['proc']
        batch_size, max_visit, dim = visit_rep.shape
        visit_rep = visit_rep.reshape(batch_size * max_visit, dim)
        visit_rep = visit_rep[true_visit_idx]  # 只保留非空的visit

        visit_rep_mem = self.visit_mem_attn(
             visit_rep, E_mem
        )

        med_rep = visit_seq_embed['med'].reshape(batch_size * max_visit, dim)[true_visit_idx]

        if self.training:
            # 因为训练的时候就在这里对齐,所以这里需要对齐
            dp_edge_in_batch = E_mem['dp'][visit2edge_idx]
            m_edge_in_batch = E_mem['m'][visit2edge_idx]
            dp_ssl_loss = self.info_nce_loss(visit_rep, dp_edge_in_batch)
            m_ssl_loss = self.info_nce_loss(med_rep, m_edge_in_batch)
            ssl_loss = dp_ssl_loss + m_ssl_loss
        else:
            ssl_loss = med_rep.mean()

        # 这里med单独拿出来作为用药史
        med_history = visit_seq_embed['med']  # 想做一个纯药物史解码器
        # 这里要注意最后一个时刻的药物实际上是看不到的,只能作为监督信息
        batch_size, max_visit, dim = med_history.shape
        pad_head_med_history = torch.zeros(batch_size, 1, dim, dtype=med_history.dtype, device=med_history.device)
        med_history = torch.cat([pad_head_med_history, med_history], dim=1)[:, :-1, :]  # 这里就shift过了
        med_history = med_history.reshape(batch_size * max_visit, dim)
        med_history = med_history[true_visit_idx]  # 只保留非空的visit


        # 计算包含上下文信息的表示
        # attn_mask = masks['attn_mask'].repeat(self.n_heads, 1, 1)
        attn_mask = masks['attn_mask']
        attn_mask = attn_mask[true_visit_idx][:, true_visit_idx]
        assert attn_mask.shape == (visit_rep.shape[0], visit_rep.shape[0])
        # visit_rep_mem = visit_rep_mem.reshape(batch_size, max_visit, dim * 3)

        # dp_context_rep, dp_context_attn = self.dp_context_attn(
        #     query=visit_rep, key=visit_rep, value=visit_rep, need_weights=True,
        #     # key_padding_mask=masks['key_padding_mask'],
        #     # 用来遮挡key中的padding,和key一样的shape,这里应该需要把补足用的visitmask上,bsz_max_visit
        #     attn_mask=attn_mask,  # 加上会输出nan,因为有些地方的是补0的,整行都是True,相当于不计算
        # )
        # dp_context_rep += visit_rep
        dp_context_rep = self.dp_context_attn(visit_rep, attn_mask)


        # mem_context_rep, mem_context_attn = self.mem_context_attn(
        #     query=visit_rep_mem, key=visit_rep_mem, value=visit_rep_mem, need_weights=True,
        #     # key_padding_mask=masks['key_padding_mask'],
        #     # 用来遮挡key中的padding,和key一样的shape,这里应该需要把补足用的visitmask上,bsz_max_visit
        #     attn_mask=attn_mask,  # 加上会输出nan,因为有些地方的是补0的,整行都是True,相当于不计算
        # )
        # mem_context_rep += visit_rep_mem
        mem_context_rep = self.mem_context_attn(visit_rep_mem, attn_mask)

        # med_context_rep, med_context_attn = self.med_context_attn(
        #     query=med_history, key=med_history, value=med_history, need_weights=True,
        #     # key_padding_mask=masks['key_padding_mask'],
        #     # 用来遮挡key中的padding,和key一样的shape,这里应该需要把补足用的visitmask上,bsz_max_visit
        #     attn_mask=attn_mask,  # 加上会输出nan,因为有些地方的是补0的,整行都是True,相当于不计算
        # )
        # med_context_rep += med_history
        med_context_rep = self.med_context_attn(med_history, attn_mask)

        # 先融合再预测
        cat_rep = torch.cat([
            dp_context_rep.unsqueeze(-1),
            mem_context_rep.unsqueeze(-1),
            med_context_rep.unsqueeze(-1)],
            -1)
        gate = self.gate_control(cat_rep.reshape(-1, 3 * mem_context_rep.shape[-1])).reshape(-1, mem_context_rep.shape[-1], 3)
        assert len(gate.shape) == 3 and gate.shape[-1] == 3
        fusion_rep = (gate * cat_rep).sum(-1)
        fusion_output, fusion_moe_loss = self.fusion_moe_preditor(fusion_rep)
        output = self.fusion_pred_norm(fusion_output)
        moe_loss = fusion_moe_loss

        # # 这里直接接MoE输出预测
        # med_embedding = X_hat['med']
        # # 这里要注意,最后一列是padding,需要去掉
        # # dp_dot_output = torch.matmul(dp_context_rep, med_embedding.T)[:, :-1]
        # # dot_output = dot_output.reshape(batch_size * max_visit, -1)
        # # dp_dot_output = self.dp_output_norm(dp_context_rep)
        # dp_output, dp_moe_loss = self.mem_moe_preditor(dp_context_rep)
        # dp_output = self.dp_pred_norm(dp_output)
        #
        # # 这里要注意,最后一列是padding,需要去掉
        # # mem_dot_output = torch.matmul(mem_context_rep, med_embedding.T)[:, :-1]
        # # dot_output = dot_output.reshape(batch_size * max_visit, -1)
        # # mem_dot_output = self.mem_output_norm(mem_context_rep)
        # mem_output, mem_moe_loss = self.mem_moe_preditor(mem_context_rep)
        # mem_output = self.mem_pred_norm(mem_output)
        #
        # # 纯药物output
        # # med_dot_output = torch.matmul(med_context_rep, med_embedding.T)[:, :-1]
        # # med_dot_output = self.med_output_norm(med_context_rep)
        # med_output, med_moe_loss = self.med_moe_preditor(med_context_rep)
        # med_output = self.med_pred_norm(med_output)
        # # 门控,输入用拼接,加,减,乘,一共5 * med_size
        # # lin+relu+norm+sig
        # # 输出是med_size
        # gate_input = torch.cat([
        #     dp_output.unsqueeze(-1),
        #     mem_output.unsqueeze(-1),
        #     med_output.unsqueeze(-1),
        # ], dim=-1)
        # gate = self.gate_control(gate_input.reshape(-1, 3 * med_output.shape[-1]))
        # gate = gate.reshape(-1, med_output.shape[-1], 3)
        # assert len(gate.shape) == 3 and gate.shape[-1] == 3
        # gate = torch.softmax(gate, -1)
        #
        # # 整合
        # assert gate.shape == gate_input.shape
        # output = (gate * gate_input).sum(-1)
        # moe_loss = dp_moe_loss + mem_moe_loss + med_moe_loss

        # 计算ddi-loss
        neg_pred_prob = F.sigmoid(output)
        neg_pred_prob = neg_pred_prob.unsqueeze(-1)
        neg_pred_prob = neg_pred_prob.transpose(-1, -2) * neg_pred_prob  # (true visit num, voc_size, voc_size)

        # loss_mask = (masks['key_padding_mask'] == False).unsqueeze(-1).unsqueeze(-1)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        # 计算ssl
        side_loss = {
            'ddi': batch_neg,
            'ssl': ssl_loss,
            'moe': moe_loss
        }

        return output, side_loss