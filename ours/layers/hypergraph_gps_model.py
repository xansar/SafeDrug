# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: hypergraph_gps_model.py
@time: 2023/6/8 14:41
@e-mail: xansar@ruc.edu.cn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as tsp
from info_nce import InfoNCE
from torch_geometric.nn.dense.linear import Linear as pyg_Linear
from torch_geometric.nn.inits import glorot

# from .conv import HyperGraphEncoder
from .hgt_encoder import HGTEncoder, Node2EdgeAggregator
from .moe import MoEPredictor
from .ehr_memory_attn import EHRMemoryAttention

import faiss

# from ours.util import tsne_visual


class HGTDrugRec(nn.Module):
    def __init__(self, voc_size_dict, adj_dict, padding_dict, voc_dict, cluster_matrix, n_ehr_edges, embedding_dim, n_heads, n_layers,
                 n_protos, n_experts, n_select, n_clusters,
                 dropout, device, cache_dir):
        super(HGTDrugRec, self).__init__()
        self.graph_embed_cache = None

        self.name_lst = ['diag', 'proc', 'med']
        self.voc_size_dict = voc_size_dict
        self.num_dict = voc_size_dict
        self.n_ehr_edges = n_ehr_edges
        self.n_protos = n_protos
        self.n_clusters = n_clusters

        self.device = device

        self.tensor_ddi_adj = adj_dict['ddi_adj'].to(device)  # 普通图
        self.adj_dict = {k: v.to(device) for k, v in adj_dict.items()}
        # 为每个超边赋予一个可学习的权重
        self.edge_weight_dict = nn.ParameterDict()
        for n in self.name_lst:
            adj = self.adj_dict[n]
            n_edge = adj.shape[1]
            weight = torch.rand(n_edge)
            weight = torch.nn.Parameter(weight, requires_grad=True).to(device)
            self.edge_weight_dict[n] = weight

        self.padding_dict = padding_dict

        self.cluster_matrix = cluster_matrix
        self.n_select = n_select

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
        self.embedding_norm = nn.ModuleDict(
            {
                n: nn.BatchNorm1d(embedding_dim)
                for n in self.name_lst
            }
        )
        # self.graph_encoder = nn.ModuleDict(
        #     {
        #         n: HyperGraphEncoder(
        #             embedding_dim=embedding_dim,
        #             n_heads=n_heads,
        #             dropout=dropout,
        #             n_layers=n_layers
        #         )
        #         for n in self.name_lst
        #     }
        # )

        self.graph_encoder = nn.ModuleDict(
            {
                n: HGTEncoder(
                    embed_dim=embedding_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    n_layers=n_layers,
                    H=adj_dict[n],
                    idx2word=voc_dict[n].idx2word,
                    cache_dir=cache_dir,
                    device=device,
                    name=n,
                )
                for n in self.name_lst
            }
        )

        self.ssl_temp = 0.1
        self.ssl_reg = 1e-3

        self.node2edge_agg = nn.ModuleDict(
            {
                n: Node2EdgeAggregator(embedding_dim, n_heads, dropout)
                for n in self.name_lst
            }
        )

        # self.edge2proto_mha = nn.ModuleDict(
        #     {
        #         n: nn.MultiheadAttention(
        #             embed_dim=embedding_dim,
        #             num_heads=n_heads,
        #             dropout=dropout,
        #             batch_first=True,
        #         )
        #         for n in self.name_lst
        #     }
        # )

        # # 算序列依赖
        # self.d_p_fusion = nn.Sequential(
        #     nn.Linear(2 * embedding_dim, embedding_dim),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(embedding_dim),
        #
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(embedding_dim),
        # )

        # self.visit_mem_attn = nn.MultiheadAttention(
        #     embed_dim=embedding_dim * 3,
        #     num_heads=n_heads,
        #     dropout=dropout,
        #     batch_first=True,
        # )

        self.visit_mem_attn = EHRMemoryAttention(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            dropout=dropout
        )

        self.med_context_attn = nn.MultiheadAttention(
            # embed_dim=embedding_dim * 3,
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.context_attn = nn.MultiheadAttention(
            # embed_dim=embedding_dim * 3,
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # self.output_mlp = nn.Sequential(
        #     nn.Linear(3 * embedding_dim, embedding_dim),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(embedding_dim),    # 这里不能用batch,因为有空的visit
        #
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(embedding_dim),
        # )
        # self.output_mlp = nn.Sequential(
        #     nn.Linear(3 * embedding_dim, embedding_dim),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm1d(embedding_dim),  # 这里不能用batch,因为有空的visit
        # )

        self.output_norm = nn.BatchNorm1d(self.voc_size_dict['med'])

        self.moe_preditor = MoEPredictor(
            embed_dim=voc_size_dict['med'],
            output_size=voc_size_dict['med'],
            n_experts=n_experts
        )

        self.med_moe_preditor = MoEPredictor(
            embed_dim=embedding_dim,
            output_size=voc_size_dict['med'],
            n_experts=n_experts
        )

        self.pred_norm = nn.BatchNorm1d(self.voc_size_dict['med'])
        self.med_pred_norm = nn.BatchNorm1d(self.voc_size_dict['med'])

        self.gate_control = nn.Sequential(
            nn.Linear(5 * self.voc_size_dict['med'], self.voc_size_dict['med']),
            nn.ReLU(),
            nn.BatchNorm1d(self.voc_size_dict['med']),
            nn.Sigmoid()
        )
        # self.d_p_seq_attn = nn.MultiheadAttention(
        #     embed_dim=embedding_dim,
        #     num_heads=n_heads,
        #     dropout=dropout,
        #     batch_first=True,
        # )
        # self.med_seq_attn = nn.MultiheadAttention(
        #     embed_dim=embedding_dim,
        #     num_heads=n_heads,
        #     dropout=dropout,
        #     batch_first=True,
        # )
        #
        # self.patient_rep_layer = nn.Sequential(
        #     nn.Linear(2 * embedding_dim, embedding_dim),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(embedding_dim),
        #
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(embedding_dim),
        # )

        self.info_nce_loss = InfoNCE(reduction='none')

    def get_embedding(self):
        """
        获取embedding
        Returns:

        """
        x = {}
        for n in self.name_lst:
            idx = torch.arange(self.num_dict[n], dtype=torch.long, device=self.device)
            x[n] = self.embedding_norm[n](self.embedding[n](idx))
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
        if adj.is_sparse:
            norm_factor = (tsp.sum(adj, dim=0) ** -1).to_dense().reshape(n_edges, -1)
            assert norm_factor.shape == (n_edges, 1)
            E = norm_factor * tsp.mm(adj.T, embed)
        else:
            norm_factor = (torch.sum(adj, dim=0) ** -1).reshape(n_edges, -1)
            assert norm_factor.shape == (n_edges, 1)
            E = norm_factor * torch.mm(adj.T, embed)

        return E

    def hyperedge_select(self, adj, cluster_matrix, edge_weight, k=1):
        mask = (cluster_matrix != 0).float()
        idx = torch.multinomial(mask, k, replacement=False).flatten()
        idx = cluster_matrix.gather(1, idx.reshape(-1, 1))
        assert idx.shape == (cluster_matrix.shape[0], 1)
        idx = idx.reshape(-1)
        selected_weight = edge_weight[idx]
        selected_adj = adj.to_dense()[:, idx].to_sparse_coo().coalesce()
        if adj.shape[1] > self.n_ehr_edges:
            side_edge = adj.to_dense()[:, self.n_ehr_edges + 1:].to_sparse_coo().coalesce()
            side_edge_weight = edge_weight[self.n_ehr_edges + 1:]
            selected_adj = torch.cat([selected_adj, side_edge], dim=1).coalesce()
            selected_weight = torch.cat([selected_weight, side_edge_weight], dim=0)
        return selected_adj, selected_weight

    # 先写graph encode的部分
    def graph_encode(self, x, adj, edge_weight):
        """
        进行超图编码
        Args:
            x: dict
            adj: dict

        Returns:
            x: dict
        """
        X = {}
        E = {}
        for n in self.name_lst:
            # 这里需要随机从簇里抽出一部分超边
            if self.training:
                # selected_adj, selected_edge_weight = self.hyperedge_select(adj[n], self.cluster_matrix[n], edge_weight[n], self.n_select)
                selected_adj, selected_edge_weight = adj[n], edge_weight[n]
            else:
                selected_adj, selected_edge_weight = adj[n], edge_weight[n]

            edges = self.get_hyperedge_representation(x[n], selected_adj)
            X[n], E[n] = self.graph_encoder[n](x[n], edges, selected_adj, selected_edge_weight)
            # adj_index = adj[n].indices()
            # hyperedge_attr = self.get_hyperedge_representation(x[n], adj[n])
            # x[n] = self.layernorm_1[n](x[n] + self.graph_encoder[n](x[n], adj_index, hyperedge_attr=hyperedge_attr))
            # x[n] = self.layernorm_2[n](x[n] + self.ffn[n](x[n]))
        # 将三个图上的超边拼接起来,这里要注意,因为有辅助超边信息,很可能没办法直接拼起来,需要将前面ehr的部分取出来
        # 假设有一个变量self.n_ehr_edges,记录了所有的visit数量
        if self.training:
            # ehr_size = min(self.n_ehr_edges, self.n_clusters)
            ehr_size = self.n_ehr_edges
        else:
            ehr_size = self.n_ehr_edges
        # idx = 265
        # E_ehr = torch.cat([E['diag'][:ehr_size], E['proc'][:ehr_size], E['med'][:ehr_size]], dim=-1)
        E_ehr = {
            'dp': E['diag'][:ehr_size] + E['proc'][:ehr_size],
            'm': E['med'][:ehr_size]
        }
        # 这里执行聚类
        if self.training:
            E_mem = E_ehr
            # E_mem, mem2ehr = self.edges_cluster(E_ehr)
        else:
            E_mem = E_ehr
            # E_mem, mem2ehr = self.edges_cluster(E_ehr)

        # cluster_ssl = self.cluster_SSL(E_ehr, E_mem, mem2ehr)
        # cluster_ssl = self.ssl_reg * cluster_ssl
        # assert E_mem.shape == (self.n_protos, self.embedding_dim * 3)
        return X, E, E_mem

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

    def DPM_SSL(self, E):
        if self.training:
            # ehr_size = min(self.n_ehr_edges, self.n_clusters)
            ehr_size = self.n_ehr_edges
        else:
            ehr_size = self.n_ehr_edges
        D, P, M = E['diag'][:ehr_size], E['proc'][:ehr_size], E['med'][:ehr_size]
        assert D.shape == P.shape == M.shape
        dp_ssl_loss = self.info_nce_loss(D, P)
        dm_ssl_loss = self.info_nce_loss(D, M)
        pm_ssl_loss = self.info_nce_loss(P, M)
        loss = dp_ssl_loss + dm_ssl_loss + pm_ssl_loss
        return loss.mean()

    # def cluster_SSL(self, edges, centroids, edge2cluster):
    #     norm_edges = F.normalize(edges)
    #     edge2centroids = centroids[edge2cluster]
    #
    #     pos_score = torch.mul(norm_edges, edge2centroids).sum(dim=1)
    #     pos_score = torch.exp(pos_score / self.ssl_temp)
    #     ttl_score = torch.matmul(norm_edges, centroids.transpose(0, 1))
    #     ttl_score = torch.exp(ttl_score / self.ssl_temp).sum(dim=1)
    #
    #     proto_nce_loss = -torch.log(pos_score / ttl_score).sum()
    #     return proto_nce_loss

    def edges_cluster(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        https://github.com/RUCAIBox/NCL/blob/master/ncl.py
        """
        # k = min(self.n_protos, x.shape[0])
        kmeans = faiss.Kmeans(d=self.embedding_dim * 3, k=self.n_protos, gpu=True)
        x = x.detach().cpu()
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def cache_in_eval(self):
        # 首先是embedding层
        raw_embedding = self.get_embedding()  # dict
        # 最开始是超图编码层
        X_hat, E_hat, E_mem = self.graph_encode(raw_embedding, self.adj_dict, self.edge_weight_dict)
        for n in self.name_lst:
            # 这里需要注意,因为records中含有pad,需要给embedding补一行
            padding_row = self.embedding[n](
                torch.tensor(self.num_dict[n], dtype=torch.long, device=self.device)).reshape(1, self.embedding_dim)
            X_hat[n] = torch.vstack([X_hat[n], padding_row])
        self.graph_embed_cache = X_hat, E_hat, E_mem

    def forward(self, records, masks, true_visit_idx):
        """

        Args:
            records: dict(diag, proc, med),其中每个元素shape为(bsz, max_vist, max_name_size)
                    对齐方式上,diag^t,proc^t和med^(t-1)对齐,第一次的med使用pad补
            masks: 多头注意力用
            true_visit_idx: 用来过滤空的visit

        Returns:

        """
        if self.training:
            # 首先是embedding层
            raw_embedding = self.get_embedding()  # dict
            # 最开始是超图编码层
            self.graph_embed_cache = None
            X_hat, E_hat, E_mem = self.graph_encode(raw_embedding, self.adj_dict, self.edge_weight_dict)
            for n in self.name_lst:
                # 这里需要注意,因为records中含有pad,需要给embedding补一行
                padding_row = self.embedding[n](
                    torch.tensor(self.num_dict[n], dtype=torch.long, device=self.device)).reshape(1, self.embedding_dim)
                X_hat[n] = torch.vstack([X_hat[n], padding_row])
        else:
            X_hat, E_hat, E_mem = self.graph_embed_cache

        # 自监督损失
        ## 三个领域相互做ssl
        dpm_ssl = self.DPM_SSL(E_hat)
        ssl_loss = dpm_ssl

        # 解析序列数据
        entity_seq_embed = {}  # (bsz, max_vist, max_size, dim)
        for n in self.name_lst:
            entity_seq_embed[n] = X_hat[n][records[n]]

        # 首先是visit-level的数据表示
        # 使用多头注意力
        visit_seq_embed = self.node2edge(entity_seq_embed)  # bsz, max_visit, dim

        # 这里med单独拿出来作为用药史
        med_history = visit_seq_embed['med']  # 想做一个纯药物史解码器
        # 这里要注意最后一个时刻的药物实际上是看不到的,只能作为监督信息
        batch_size, max_visit, dim = med_history.shape
        pad_head_med_history = torch.zeros(batch_size, 1, dim, dtype=med_history.dtype, device=med_history.device)
        med_history = torch.cat([pad_head_med_history, med_history], dim=1)[:, :-1, :]  # 这里就shift过了
        med_history = med_history.reshape(batch_size * max_visit, dim)
        med_history = med_history[true_visit_idx]   # 只保留非空的visit

        # # 这里将diag和proc还有历史用药拼起来
        # visit_rep = torch.cat([visit_seq_embed['diag'], visit_seq_embed['proc'], med_history], dim=-1)
        # assert visit_rep.shape == (batch_size, max_visit, dim * 3)
        #
        # # 用多头注意力,当前为q,图上的超边为k和v
        # # 这里感觉用dp作为q,历史dp作为k,历史作为v比较好
        # visit_rep = visit_rep.reshape(batch_size * max_visit, dim * 3)
        # visit_rep = visit_rep[true_visit_idx]   # 只保留非空的visit
        # visit_rep_mem = self.visit_mem_attn(
        #     visit_rep, E_mem
        # )

        # 先上下文再记忆
        # 这里将diag和proc还有历史用药拼起来
        visit_rep = visit_seq_embed['diag'] + visit_seq_embed['proc']
        assert visit_rep.shape == (batch_size, max_visit, dim)
        visit_rep = visit_rep.reshape(batch_size * max_visit, dim)
        visit_rep = visit_rep[true_visit_idx]   # 只保留非空的visit

        # 计算包含上下文信息的表示
        # attn_mask = masks['attn_mask'].repeat(self.n_heads, 1, 1)
        attn_mask = masks['attn_mask']
        attn_mask = attn_mask[true_visit_idx][:, true_visit_idx]
        assert attn_mask.shape == (visit_rep.shape[0], visit_rep.shape[0])
        # visit_rep_mem = visit_rep_mem.reshape(batch_size, max_visit, dim * 3)

        context_rep, context_attn = self.context_attn(
            query=visit_rep, key=visit_rep, value=visit_rep, need_weights=True,
            # key_padding_mask=masks['key_padding_mask'],
            # 用来遮挡key中的padding,和key一样的shape,这里应该需要把补足用的visitmask上,bsz_max_visit
            attn_mask=attn_mask,  # 加上会输出nan,因为有些地方的是补0的,整行都是True,相当于不计算
        )

        med_context_rep, med_context_attn = self.med_context_attn(
            query=med_history, key=med_history, value=med_history, need_weights=True,
            # key_padding_mask=masks['key_padding_mask'],
            # 用来遮挡key中的padding,和key一样的shape,这里应该需要把补足用的visitmask上,bsz_max_visit
            attn_mask=attn_mask,  # 加上会输出nan,因为有些地方的是补0的,整行都是True,相当于不计算
        )

        # 用多头注意力,当前为q,图上的超边为k和v
        # 这里感觉用dp作为q,历史dp作为k,历史作为v比较好

        visit_rep_mem = self.visit_mem_attn(
            context_rep, E_mem
        )

        # context_rep = self.output_mlp(context_rep)
        # 这里直接接MoE输出预测
        med_embedding = X_hat['med']
        # 这里要注意,最后一列是padding,需要去掉
        dp_dot_output = torch.matmul(visit_rep_mem, med_embedding.T)[:, :-1]
        # dot_output = dot_output.reshape(batch_size * max_visit, -1)
        dp_dot_output = self.output_norm(dp_dot_output)
        dp_output, dp_moe_loss = self.moe_preditor(dp_dot_output)
        dp_output = self.pred_norm(dp_output)

        # 纯药物output
        med_output, med_moe_loss = self.med_moe_preditor(med_context_rep)
        med_output = self.med_pred_norm(med_output)
        # 门控,输入用拼接,加,减,乘,一共5 * med_size
        # lin+relu+norm+sig
        # 输出是med_size
        gate_input = torch.cat([
            dp_output,
            med_output,
            dp_output + med_output,
            dp_output - med_output,
            dp_output * med_output
        ], dim=-1)
        gate = self.gate_control(gate_input)

        # 整合
        output = gate * dp_output + (1 - gate) * med_output
        moe_loss = dp_moe_loss + med_moe_loss

        # output = dot_output
        # moe_loss = dot_output.mean()

        # output = torch.matmul(context_rep, med_embedding.T)[:, :, :-1]
        # output = output.reshape(batch_size, max_visit, -1)
        #
        #
        # # 解码还是使用多头注意力,当前时刻的diag和proc作为query, 过去时刻的diag和proc作为key,过去时刻的med作为value
        # diag_and_proc = torch.cat([visit_seq_embed['diag'], visit_seq_embed['proc']], dim=-1)
        # # diag_and_proc = torch.cat([diag_and_proc, self.d_p_fusion(diag_and_proc)], dim=-1) # bsz, max_visit, 3 * dim
        # diag_and_proc = visit_seq_embed['diag'] + visit_seq_embed['proc'] + self.d_p_fusion(
        #     diag_and_proc)  # bsz, max_visit, dim
        #
        # # 设计mask
        # # 下面两步计算出来的是每个visit的上下文表示
        # attn_mask = masks['attn_mask'].repeat(self.n_heads, 1, 1)
        # d_p_rep = self.d_p_seq_attn(
        #     query=diag_and_proc, key=diag_and_proc, value=diag_and_proc, need_weights=False,
        #     # key_padding_mask=masks['key_padding_mask'],   # 两个mask用一个就行
        #     # 用来遮挡key中的padding,和key一样的shape,这里应该需要把补足用的visitmask上,bsz_max_visit
        #     attn_mask=attn_mask,  # 加上会输出nan,因为有些地方的是补0的,整行都是True,相当于不计算
        # )[0]
        # med_rep = self.med_seq_attn(
        #     query=med_history, key=med_history, value=med_history, need_weights=False,
        #     # key_padding_mask=masks['key_padding_mask'],
        #     # 用来遮挡key中的padding,和key一样的shape,这里应该需要把补足用的visitmask上,bsz_max_visit
        #     attn_mask=attn_mask,  # 加上会输出nan,因为有些地方的是补0的,整行都是True,相当于不计算
        # )[0]
        # patient_rep = self.patient_rep_layer(torch.cat([d_p_rep, med_rep], dim=-1))  # 构建一个融合表示, bsz, max_visit, dim
        #
        # # 下面计算logits,分为两部分,一部分是patient_rep计算,一部分是直接用med计算
        # med_embedding = X_hat['med']
        #
        # patient_output = torch.matmul(patient_rep, med_embedding.T)
        # med_output = torch.matmul(med_rep, med_embedding.T)
        #
        # # 两个logits相加
        # output = patient_output + med_output

        # 计算ddi-loss
        neg_pred_prob = F.sigmoid(output)
        neg_pred_prob = neg_pred_prob.unsqueeze(-1)
        neg_pred_prob = neg_pred_prob.transpose(-1, -2) * neg_pred_prob  # (true visit num, voc_size, voc_size)

        # loss_mask = (masks['key_padding_mask'] == False).unsqueeze(-1).unsqueeze(-1)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        # 计算ssl
        # ssl_loss = (self.SSL(patient_rep, med_rep) * loss_mask.squeeze(-1)).sum() / loss_mask.sum()
        # side_loss = {
        #     'ddi': batch_neg,
        #     'ssl': ssl_loss
        # }
        # 计算ssl
        side_loss = {
            'ddi': batch_neg,
            'ssl': ssl_loss,
            'moe': moe_loss
        }

        return output, side_loss


# class Node2EdgeAggregator(nn.Module):
#     def __init__(self, in_channels, out_channels, n_heads, dropout):
#         super(Node2EdgeAggregator, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.n_heads = n_heads
#         self.out_dim = out_channels // n_heads  # 分头后的空间
#         self.dropout = dropout
#
#         self.lin = pyg_Linear(self.out_dim, self.out_dim, bias=False,
#                               weight_initializer='glorot')
#         self.att = nn.Parameter(torch.Tensor(1, n_heads, 2 * self.out_dim))
#         glorot(self.att)
#
#     def forward(self, x):
#         """
#
#         Args:
#             x: bsz, max_size, dim
#
#         Returns:
#
#         """
#         # 首先使用平均计算超边表示,然后将节点表示和超边表示拼接起来计算注意力
#         bsz, max_size, dim = x.shape
#         x = x.reshape(-1, max_size, self.n_heads, self.out_channels // self.n_heads)  # bsz, max_size, head, out_dim
#         x = self.lin(x)  # bsz, max_size, dim
#
#         hyperedge_attr = x.mean(1)  # bsz, dim
#         hyperedge_attr = self.lin(hyperedge_attr)
#         hyperedge_attr = hyperedge_attr.reshape(-1, self.n_heads, self.out_channels // self.n_heads)
#         hyperedge_attr = hyperedge_attr.unsqueeze(1).repeat(1, max_size, 1, 1)  # bsz, max_size, head, out_dim
#
#         # 跟x拼接起来
#         alpha = (torch.cat([x, hyperedge_attr], dim=-1) * self.att).sum(dim=-1)  # bsz, max_size, head
#         alpha = F.leaky_relu(alpha)  # bsz, max_size, head
#         alpha = torch.softmax(alpha, dim=1)  # bsz, max_size, head
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training).unsqueeze(-1)  # bsz, max_size, head, 1
#         out = (alpha * x).sum(1)  # bsz, head, out_dim
#         out = out.reshape(-1, self.out_channels)  # bsz, dim
#         return out

