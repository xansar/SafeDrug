# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: hybo_model.py
@time: 2023/6/14 16:43
@e-mail: xansar@ruc.edu.cn
"""
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as tsp

from torch_geometric.nn.dense.linear import Linear as pyg_Linear
from torch_geometric.nn.inits import glorot

import hybo_methods.manifolds as manifolds
from hybo_methods.layers.hyp_layers import H2GraphConvolution


class H2DrugRec(nn.Module):
    def __init__(self, c, voc_size_dict, adj_dict, padding_dict, embedding_dim, n_heads, n_layers, dropout, scale,
                 device, use_att=True):
        super(H2DrugRec, self).__init__()
        self.c = torch.tensor([c])
        self.manifold = getattr(manifolds, "Hyperboloid")()

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

        # todo:这里需要加一个转换,就是从欧式空间到双曲空间
        for n in self.name_lst:
            self.embedding.state_dict()[n + '.weight'].uniform_(-scale, scale)
            self.embedding[n].weight = nn.Parameter(
                self.manifold.expmap0(self.embedding.state_dict()[n + '.weight'], self.c))

            self.embedding[n].weight = manifolds.ManifoldParameter(self.embedding[n].weight, True, self.manifold,
                                                                   self.c)

        self.c = self.c.to(device)

        self.graph_encoder = nn.ModuleDict(
            {
                n: H2GraphConvolution(
                    manifold=self.manifold,
                    c=self.c,
                    in_features=embedding_dim,
                    out_features=embedding_dim,
                    num_layers=n_layers,
                    dropout=dropout,
                    n_heads=n_heads,
                    use_att=use_att,
                )
                for n in self.name_lst
            }
        )

        # todo: 明天开始改这里

        self.node2edge_agg = nn.ModuleDict(
            {
                n: H2Node2EdgeAggregator(self.c, self.manifold, embedding_dim, embedding_dim, n_heads, dropout)
                for n in self.name_lst
            }
        )

        self.node2edge_agg['neg_med'] = H2Node2EdgeAggregator(self.c, self.manifold, embedding_dim, embedding_dim,
                                                              n_heads, dropout)

        # 算序列依赖
        self.d_p_fusion = H2FusionLayer(self.c, self.manifold, embedding_dim)

        self.d_p_seq_attn = H2MultiHeadAttention(
            c=self.c,
            manifold=self.manifold,
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.med_seq_attn = H2MultiHeadAttention(
            c=self.c,
            manifold=self.manifold,
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.patient_rep_layer = H2FusionLayer(self.c, self.manifold, embedding_dim)

    def get_embedding(self):
        """
        获取embedding
        Returns:

        """
        x = {}
        for n in self.name_lst:
            idx = torch.arange(self.num_dict[n], dtype=torch.long, device=self.device)
            x_n = self.embedding[n](idx)
            x[n] = self.manifold.proj(x_n, c=self.c)
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
            x[n] = self.graph_encoder[n](x[n], adj[n])

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
            bsz, max_visit, max_size, dim = seq_embed.shape
            seq_embed = seq_embed.reshape(bsz * max_visit, max_size, dim)
            visit_seq_embed[n] = self.node2edge_agg[n](seq_embed).reshape(bsz, max_visit, dim)
        return visit_seq_embed

    def decode(self, embed_v, embed_m):
        if embed_v.shape != embed_m.shape:
            # todo: 明天把bpr解码写完
            # 这里embed_v的shape是bsz, max_visit, dim
            bsz, max_visit, dim = embed_v.shape
            med_size = embed_m.shape[0]
            embed_v = embed_v.reshape(bsz * max_visit, dim)
            # 展开后需要repeat_interleave,也就是每一行要重复
            embed_v = embed_v.repeat_interleave(med_size, dim=0)  # bsz * max_visit * med_size, dim
            # 接下来修改med的size,需要整块重复
            embed_m = embed_m.repeat(bsz * max_visit, 1)
            assert embed_v.shape == embed_m.shape
            sqdist = self.manifold.sqdist(embed_v, embed_m, self.c)  # 这里需要一一对应,shape是bsz * max_visit * med_size, 1
            # 修改成bsz, max_visit, med_size
            sqdist = sqdist.reshape(bsz, max_visit, med_size)
        else:
            # 这里是用来算neg med的
            # 首先展平
            bsz, max_visit, dim = embed_v.shape
            embed_v = embed_v.reshape(bsz * max_visit, dim)
            embed_m = embed_m.reshape(bsz * max_visit, dim)
            assert embed_m.shape == embed_v.shape
            sqdist = self.manifold.sqdist(embed_v, embed_m, self.c)  # shape是bsz * max_visit, 1
            # 修改shape,变成bsz, max_visit, 1,然后在最后一个维度上扩展
            sqdist = sqdist.reshape(bsz, max_visit, 1).repeat(1, 1, self.voc_size_dict[
                'med'])  # shape是bsz * max_visit, med_size
        return sqdist

    def forward(self, records, masks):
        """

        Args:
            records: dict(diag, proc, med),其中每个元素shape为(bsz, max_vist, max_name_size)
                    对齐方式上,diag^t,proc^t和med^(t-1)对齐,第一次的med使用pad补
            masks: 多头注意力用

        Returns:

        """
        # 首先是embedding层
        # 这里要注意加个这个
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
        # diag_and_proc = torch.cat([visit_seq_embed['diag'], visit_seq_embed['proc']], dim=-1)
        # diag_and_proc = torch.cat([diag_and_proc, self.d_p_fusion(diag_and_proc)], dim=-1) # bsz, max_visit, 3 * dim

        ## 这里有个加法,需要先把三项映射到切空间
        # diag_and_proc = visit_seq_embed['diag'] + visit_seq_embed['proc'] + self.d_p_fusion(
        #     diag_and_proc)  # bsz, max_visit, dim
        hybo_diag_visit_seq_embed = self.manifold.logmap0(visit_seq_embed['diag'], c=self.c)
        hybo_proc_visit_seq_embed = self.manifold.logmap0(visit_seq_embed['proc'], c=self.c)
        hybo_fusion_visit_seq_embed = self.manifold.logmap0(
            self.d_p_fusion(visit_seq_embed['diag'], visit_seq_embed['proc']), c=self.c)

        diag_and_proc = hybo_diag_visit_seq_embed + hybo_proc_visit_seq_embed + hybo_fusion_visit_seq_embed  # bsz, max_visit, dim
        ## 映射回来
        diag_and_proc = self.manifold.proj(self.manifold.expmap0(diag_and_proc, c=self.c), c=self.c)

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
        patient_rep = self.patient_rep_layer(d_p_rep, med_rep)  # 构建一个融合表示, bsz, max_visit, dim

        # 下面计算logits,分为两部分,一部分是patient_rep计算,一部分是直接用med计算
        med_embedding = tmp_embedding['med']
        # 这里需要将预测变成正负样本对的形式,每个visit对应的负样本药物不一样,要么采样,要么算均值

        # 方法1:构建正样本药物集合表示和负样本药物集合表示,那么首先要获取对应的id
        # visit表示是现成的,shape是bsz, max_visit, dim
        # 那么负样本的size是bsz, max_visit, dim
        # 需要构建neg_med, bsz, max_visit, max_neg_length
        neg_med_embed = med_embedding[records['neg_med']]  # bsz, max_visit, max_neg_length, dim

        # 简单的均值聚合
        bsz, max_visit, max_length, dim = neg_med_embed.shape
        neg_med_embed = neg_med_embed.reshape(bsz * max_visit, max_length, dim)
        neg_med_embed = self.node2edge_agg['neg_med'](neg_med_embed).reshape(bsz, max_visit, dim)  # bsz, max_visit, dim

        alpha = 0.5
        # pos
        pos_dp_score = self.decode(patient_rep, med_embedding[:-1, ...])  # bsz, max_visit, med_size
        pos_score = pos_dp_score
        # pos_med_score = self.decode(med_rep, med_embedding[:-1, ...])  # bsz, max_visit, med_size
        # pos_score = alpha * pos_dp_score + (1 - alpha) * pos_med_score

        # neg
        neg_dp_score = self.decode(patient_rep, neg_med_embed)  # bsz, max_visit, med_size
        neg_score = neg_dp_score
        # neg_med_score = self.decode(med_rep, neg_med_embed)  # bsz, max_visit, med_size
        # neg_score = alpha * neg_dp_score + (1 - alpha) * neg_med_score

        output = {
            'pos': pos_score,
            'neg': neg_score
        }

        # margin_loss = pos_score - neg_score + self.margin
        # margin_loss[margin_loss < 0] = 0    # 这里score是距离,越小越好
        # loss_mask = (masks['key_padding_mask'] == False).unsqueeze(-1)  # bsz, maxvisit, 1
        # margin_loss = margin_loss * loss_mask
        # # 计算ddi-loss
        # # 这里需要想一个方法,怎么让这个值变小一点
        # neg_pred_prob = F.sigmoid(pos_score)
        # neg_pred_prob = neg_pred_prob.unsqueeze(-1)
        # neg_pred_prob = neg_pred_prob.transpose(-1, -2) * neg_pred_prob  # (bsz, max_visit, voc_size, voc_size)
        #
        # loss_mask = (masks['key_padding_mask'] == False).unsqueeze(-1).unsqueeze(-1)
        # batch_neg = 0.0005 * (neg_pred_prob.mul(self.tensor_ddi_adj) * loss_mask).sum()

        # patient_output = self.decode(patient_rep, med_embedding)    # bsz, max_visit, dim
        # med_output = self.decode(med_rep, med_embedding)    # bsz, max_visit, dim
        #
        # # 两个logits相加
        # output = patient_output + med_output
        # # 这里要注意,最后一列是padding,需要去掉
        # output = output[:, :, :-1]
        #
        # # 计算ddi-loss
        # neg_pred_prob = F.sigmoid(output)
        # neg_pred_prob = neg_pred_prob.unsqueeze(-1)
        # neg_pred_prob = neg_pred_prob.transpose(-1, -2) * neg_pred_prob  # (bsz, max_visit, voc_size, voc_size)
        #
        # loss_mask = (masks['key_padding_mask'] == False).unsqueeze(-1).unsqueeze(-1)
        # batch_neg = 0.0005 * (neg_pred_prob.mul(self.tensor_ddi_adj) * loss_mask).sum()

        return output


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


class H2FusionLayer(nn.Module):
    def __init__(self, c, manifold, embedding_dim):
        super(H2FusionLayer, self).__init__()
        self.c = c
        self.manifold = manifold

        self.ln_1 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.act_1 = nn.LeakyReLU()
        self.layernorm_1 = nn.LayerNorm(embedding_dim)

        self.ln_2 = nn.Linear(embedding_dim, embedding_dim)
        self.act_2 = nn.LeakyReLU()
        self.layernorm_2 = nn.LayerNorm(embedding_dim)

    def forward(self, x1, x2):
        x1 = self.manifold.logmap0(x1, c=self.c)
        x2 = self.manifold.logmap0(x2, c=self.c)
        x_tangent = torch.cat([x1, x2], dim=-1)

        x_tangent = self.layernorm_1(self.act_1(self.ln_1(x_tangent)))
        out = self.layernorm_2(self.act_2(self.ln_2(x_tangent)))

        out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)
        return out


class H2Node2EdgeAggregator(Node2EdgeAggregator):
    def __init__(self, c, manifold, in_channels, out_channels, n_heads, dropout):
        super(H2Node2EdgeAggregator, self).__init__(in_channels, out_channels, n_heads, dropout)
        self.c = c
        self.manifold = manifold

    def h2_forward(self, x):
        x_tanget = self.manifold.logmap0(x, c=self.c)
        out = super(H2Node2EdgeAggregator, self).forward(x_tanget)
        out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)
        return out


class H2MultiHeadAttention(nn.MultiheadAttention):
    def __init__(self, c, manifold, **kwargs):
        self.c = c
        self.manifold = manifold
        super(H2MultiHeadAttention, self).__init__(**kwargs)

    def h2_forward(self, query, key, value, need_weights=False):
        query = self.manifold.logmap0(query, c=self.c)
        key = self.manifold.logmap0(key, c=self.c)
        value = self.manifold.logmap0(value, c=self.c)

        out = super(H2MultiHeadAttention, self).forward(query, key, value, need_weights=need_weights)[0]

        out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)
        return out
