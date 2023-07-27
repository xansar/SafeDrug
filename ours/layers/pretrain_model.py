# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: pretrain_model.py
@time: 2023/7/11 16:33
@e-mail: xansar@ruc.edu.cn
"""
"""
负责基于图进行预训练,用来做embedding初始化
考虑将diag和proc合并成一张图
"""

import torch
import torch.nn as nn
import torch.sparse as tsp
import torch.nn.functional as F

from .hypergraph_gps_model import HGTEncoder, Node2EdgeAggregator
from info_nce import InfoNCE
from torch import Tensor
from typing import Optional, Callable

class HGTCL(nn.Module):
    def __init__(self, adj_dict, voc_dict, cache_dir, device, voc_size_dict,
                 n_ehr_edges, args):
        super(HGTCL, self).__init__()
        self.name_lst = ['diag', 'proc', 'med']
        self.num_dict = voc_size_dict
        self.n_ehr_edges = n_ehr_edges
        # self.padding_dict = padding_dict
        self.device = device

        # self.adj_dict = {k: v.to(device) for k, v in adj_dict.items()}
        # # 为每个超边赋予一个可学习的权重
        # self.edge_weight_dict = nn.ParameterDict()
        # for n in self.name_lst:
        #     adj = self.adj_dict[n]
        #     n_edge = adj.shape[1]
        #     weight = torch.rand(n_edge)
        #     weight = torch.nn.Parameter(weight, requires_grad=True).to(device)
        #     self.edge_weight_dict[n] = weight

        # self.info_nce_loss = InfoNCE(reduction='mean')
        # self.node_infoNCE_loss = InfoNCE(reduction='mean', temperature=args.tau_n)
        # self.s_edge_infoNCE_loss = InfoNCE(reduction='mean', temperature=args.tau_se)
        # self.c_edge_infoNCE_loss = InfoNCE(reduction='mean', temperature=args.tau_ce)
        self.tau_m = args.tau_m
        self.tau_ce = args.tau_ce
        self.tau_se = args.tau_se
        self.tau_n = args.tau_n
        # self.member_infoNCE_loss = InfoNCE(reduction='mean', temperature=args.tau_m)

        self.node_embedding = nn.ModuleDict(
            {
                n: nn.Embedding(
                    num_embeddings=self.num_dict[n],
                    embedding_dim=args.dim)
                for n in self.name_lst
            }
        )

        self.edge_embedding = nn.ModuleDict(
            {
                n: nn.Embedding(
                    num_embeddings=self.n_ehr_edges + self.num_dict[n],
                    embedding_dim=args.dim)
                for n in self.name_lst
            }
        )

        self.nodes_embed_norm = nn.ModuleDict(
            {
                n: nn.LayerNorm(args.dim)
                for n in self.name_lst
            }
        )

        self.edges_embed_norm = nn.ModuleDict(
            {
                n: nn.LayerNorm(args.dim)
                for n in self.name_lst
            }
        )

        self.graph_encoder = nn.ModuleDict(
            {
                n: HGTEncoder(
                    embed_dim=args.dim,
                    n_heads=args.n_heads,
                    dropout=args.dropout,
                    n_layers=args.n_layers,
                    H=adj_dict[n],
                    idx2word=voc_dict[n].idx2word,
                    cache_dir=cache_dir,
                    device=device,
                    name=n,
                )
                for n in self.name_lst
            }
        )

        self.proj_mlp_n = nn.ModuleDict({
            n: nn.Sequential(
                nn.Linear(args.dim, args.dim),
                nn.ELU(),
                nn.Linear(args.dim, args.dim),
            )
            for n in self.name_lst
        })

        self.proj_mlp_e = nn.ModuleDict({
            n: nn.Sequential(
                nn.Linear(args.dim, args.dim),
                nn.ELU(),
                nn.Linear(args.dim, args.dim),
            )
            for n in ['dp', 'm']
        })
        # self.fc1_n = nn.Linear(args.dim, args.dim)
        # self.fc2_n = nn.Linear(args.dim, args.dim)
        # self.fc1_e = nn.Linear(args.dim, args.dim)
        # self.fc2_e = nn.Linear(args.dim, args.dim)
        self.disc = nn.Bilinear(args.dim, args.dim, 1)
        # self.node2edge_agg = nn.ModuleDict(
        #     {
        #         n: Node2EdgeAggregator(embedding_dim, n_heads, dropout)
        #         for n in self.name_lst
        #     }
        # )

        # self.node_in_edge = self.compute_node_id_in_edges()

        # self.ssl_temp = 0.1
        # self.ssl_reg = 1e-3
        # self.embedding_dim = embedding_dim
        # self.n_protos = 128
        # self.domain_ssl_weight = 0.1
        # self.cluster_ssl_weight = 1.
        # self.node2edge_ssl_weight = 0.1
        # self.dp2med_ssl_weight = 0.1

    # def compute_node_id_in_edges(self):
    #     node_in_edge = {}
    #     for n in self.name_lst:
    #         adj = self.adj_dict[n].to_dense()[:, :self.n_ehr_edges]
    #         assert adj.shape == (self.num_dict[n], self.n_ehr_edges)
    #
    #         max_size = torch.sum(adj, 0).max().int().item()
    #         pos_matrix = torch.empty((self.n_ehr_edges, max_size))
    #         pos_matrix = torch.fill(pos_matrix, -1)
    #         pos_idx = torch.argwhere(adj.T == 1).cpu()
    #         for i in range(self.n_ehr_edges):
    #             indices = pos_idx[pos_idx[:, 0] == i][:, 1]  # 找到第i条边包含哪几个节点
    #             length = len(indices)
    #             pos_matrix[i, :length] = indices
    #         pos_matrix = torch.where(pos_matrix == -1, self.padding_dict[n], pos_matrix)
    #         node_in_edge[n] = pos_matrix.long()
    #     return node_in_edge

    # def get_embedding(self):
    #     """
    #     获取embedding
    #     Returns:
    #
    #     """
    #     x = {}
    #     for n in self.name_lst:
    #         idx = torch.arange(self.num_dict[n], dtype=torch.long, device=self.device)
    #         x[n] = self.embedding_norm[n](self.embedding[n](idx))
    #     return x
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.node_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        self.fc1_n.reset_parameters()
        self.fc2_n.reset_parameters()
        self.fc1_e.reset_parameters()
        self.fc2_e.reset_parameters()
        self.disc.reset_parameters()

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

    # def graph_encode(self, x, adj, edge_weight):
    #     """
    #     进行超图编码
    #     Args:
    #         x: dict
    #         adj: dict
    #
    #     Returns:
    #         x: dict
    #     """
    #     X = {}
    #     E = {}
    #     for n in self.name_lst:  # dp和m
    #         # 这里需要随机从簇里抽出一部分超边
    #         if self.training:
    #             # selected_adj, selected_edge_weight = self.hyperedge_select(adj[n], self.cluster_matrix[n], edge_weight[n], self.n_select)
    #             selected_adj, selected_edge_weight = adj[n], edge_weight[n]
    #         else:
    #             selected_adj, selected_edge_weight = adj[n], edge_weight[n]
    #
    #         edges = self.get_hyperedge_representation(x[n], selected_adj)
    #         X[n], E[n] = self.graph_encoder[n](x[n], edges, selected_adj, selected_edge_weight)
    #         # adj_index = adj[n].indices()
    #         # hyperedge_attr = self.get_hyperedge_representation(x[n], adj[n])
    #         # x[n] = self.layernorm_1[n](x[n] + self.graph_encoder[n](x[n], adj_index, hyperedge_attr=hyperedge_attr))
    #         # x[n] = self.layernorm_2[n](x[n] + self.ffn[n](x[n]))
    #     # 将三个图上的超边拼接起来,这里要注意,因为有辅助超边信息,很可能没办法直接拼起来,需要将前面ehr的部分取出来
    #     # 假设有一个变量self.n_ehr_edges,记录了所有的visit数量
    #     if self.training:
    #         # ehr_size = min(self.n_ehr_edges, self.n_clusters)
    #         ehr_size = self.n_ehr_edges
    #     else:
    #         ehr_size = self.n_ehr_edges
    #     # idx = 265
    #     # E_ehr = torch.cat([E['diag'][:ehr_size], E['proc'][:ehr_size], E['med'][:ehr_size]], dim=-1)
    #     E_ehr = {
    #         'dp': E['diag'][:ehr_size] + E['proc'][:ehr_size],
    #         'm': E['med'][:ehr_size]
    #     }
    #     # 这里执行聚类
    #     if self.training:
    #         E_mem = E_ehr
    #         # E_mem, mem2ehr = self.edges_cluster(E_ehr)
    #     else:
    #         E_mem = E_ehr
    #         # E_mem, mem2ehr = self.edges_cluster(E_ehr)
    #
    #     # cluster_ssl = self.cluster_SSL(E_ehr, E_mem, mem2ehr)
    #     # cluster_ssl = self.ssl_reg * cluster_ssl
    #     # assert E_mem.shape == (self.n_protos, self.embedding_dim * 3)
    #     return X, E, E_mem
    def get_features(self):
        # 首先是embedding层
        X = {}
        E = {}
        for n in self.name_lst:
            node_idx = torch.arange(self.num_dict[n], dtype=torch.long, device=self.device)
            X[n] = self.nodes_embed_norm[n](self.node_embedding[n](node_idx))
            edge_idx = torch.arange(self.n_ehr_edges + self.num_dict[n], dtype=torch.long, device=self.device)  # 自连边
            E[n] = self.edges_embed_norm[n](self.edge_embedding[n](edge_idx))
        return X, E

    def node_projection(self, z):
        return {
            k: self.proj_mlp_n[k](v)
            for k, v in z.items()
        }
        # return self.fc2_n(F.elu(self.fc1_n(z)))

    def edge_projection(self, z):
        return {
            k: self.proj_mlp_e[k](v)
            for k, v in z.items()
        }
        # return self.fc2_e(F.elu(self.fc1_e(z)))

    def disc_similarity(self, z1, z2):
        return torch.sigmoid(self.disc(z1, z2)).squeeze()

    def cosine_similarity(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def f(self, x, tau):
        return torch.exp(x / tau)

    def __semi_loss(self, h1, h2, tau):
        between_sim = self.f(self.cosine_similarity(h1, h2), tau)
        return -torch.log(between_sim.diag() / between_sim.sum(1))

    def __loss(self, z1, z2, tau):
        l1 = self.__semi_loss(z1, z2, tau)
        l2 = self.__semi_loss(z2, z1, tau)

        loss = (l1 + l2) * 0.5
        loss = loss.mean()
        return loss

    def forward(self, hyperedge_index_dict, X, E):
        # 最开始是超图编码层
        X_hat = {}
        E_hat = {}
        for n in self.name_lst:  # dp和m
            x = X[n]

            hyperedge_index = hyperedge_index_dict[n]
            num_nodes, num_edges = self.num_dict[n], self.n_ehr_edges
            node_idx = torch.arange(0, num_nodes, device=x.device)
            edge_idx = torch.arange(num_edges, num_edges + num_nodes, device=x.device)
            self_loop = torch.stack([node_idx, edge_idx])
            self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)

            H = torch.sparse_coo_tensor(
                indices=self_loop_hyperedge_index,
                values=torch.ones_like(self_loop_hyperedge_index[0, :]),
                size=(num_nodes, num_edges + num_nodes)
            ).coalesce().float()

            e = self.get_hyperedge_representation(x, H) + E[n]

            node, edge = self.graph_encoder[n](x, e, H)
            X_hat[n] = node
            E_hat[n] = edge[:num_edges]
        # 将三个图上的超边拼接起来,这里要注意,因为有辅助超边信息,很可能没办法直接拼起来,需要将前面ehr的部分取出来
        E_mem = {
            'dp': E_hat['diag'] + E_hat['proc'],
            'm': E_hat['med']
        }
        return X_hat, E_mem

    # def compute_loss(self, X_hat_1, E_mem_1, X_hat_2, E_mem_2):
    #     node_ssl_loss = self.node_ssl_loss(X_hat_1, X_hat_2)
    #     edge_ssl_loss = self.edge_ssl_loss(E_mem_1, E_mem_2)
    #
    #     # E_mem['m']与包含的药物之间应该也做InfoNCE
    #     # 这里的问题是如何将解码阶段多头注意力的权重引入进来
    #     dp2med_ssl_loss = self.dp2med_ssl_loss(E_mem, X_hat)
    #
    #     # 超边表示与节点之间做InfoNCE,将edge表示与node表示统一起来
    #     ## 邻接矩阵左乘X,然后乘度矩阵
    #     node2edge_ssl = self.node2edge_ssl(X_hat, E_mem)
    #     ssl_loss = edge_ssl_loss * self.domain_ssl_weight \
    #                + node2edge_ssl * self.node2edge_ssl_weight \
    #                + dp2med_ssl_loss * self.dp2med_ssl_weight
    #     return ssl_loss

    # def compute_loss(self):
    #     # dp与m之间做InfoNCE
    #     X_hat, E_hat, E_mem = self.encode()
    #     edge_ssl_loss = self.edge_ssl_loss(E_mem)
    #
    #     # E_mem['m']与包含的药物之间应该也做InfoNCE
    #     # 这里的问题是如何将解码阶段多头注意力的权重引入进来
    #     dp2med_ssl_loss = self.dp2med_ssl_loss(E_mem, X_hat)
    #
    #     # 节点与聚类中心之间做InfoNCE,能让节点更靠近
    #     dp_centroids, dp_edge2cluster = self.edges_cluster(E_mem['dp'])
    #     m_centroids, m_edge2cluster = self.edges_cluster(E_mem['m'])
    #     dp_cluster_ssl = self.cluster_ssl(E_mem['dp'], dp_centroids, dp_edge2cluster)
    #     m_cluster_ssl = self.cluster_ssl(E_mem['m'], m_centroids, m_edge2cluster)
    #
    #     # 超边表示与节点之间做InfoNCE,将edge表示与node表示统一起来
    #     ## 邻接矩阵左乘X,然后乘度矩阵
    #     node2edge_ssl = self.node2edge_ssl(X_hat, E_mem)
    #
    #     ssl_loss = edge_ssl_loss * self.domain_ssl_weight \
    #                + (dp_cluster_ssl + m_cluster_ssl) * self.cluster_ssl_weight \
    #                + node2edge_ssl * self.node2edge_ssl_weight\
    #                + dp2med_ssl_loss * self.dp2med_ssl_weight
    #     return ssl_loss

    # def node2edge_ssl(self, X_hat, E_mem):
    #     edge_embed = {}
    #     for n in self.name_lst:
    #         node_in_edge = self.node_in_edge[n]
    #         padding_row = self.embedding[n](
    #             torch.tensor(self.num_dict[n], dtype=torch.long, device=self.device)).reshape(1, self.embedding_dim)
    #         X_hat[n] = torch.vstack([X_hat[n], padding_row])
    #         edge_embed_with_node = X_hat[n][node_in_edge]
    #         edge_embed_agg = self.node2edge_agg[n](edge_embed_with_node).squeeze(1)
    #         edge_embed[n] = edge_embed_agg
    #
    #     # 计算dp表示
    #     dp_edge_embedding = edge_embed['diag'] + edge_embed['proc']
    #     m_edge_embedding = edge_embed['med']
    #     dp_ssl_loss = self.info_nce_loss(dp_edge_embedding, E_mem['dp'])
    #     m_ssl_loss = self.info_nce_loss(m_edge_embedding, E_mem['m'])
    #     ssl_loss = dp_ssl_loss + m_ssl_loss
    #     return ssl_loss

    # def cluster_ssl(self, edges, centroids, edge2cluster):
    #     norm_edges = F.normalize(edges)
    #     edge2centroids = centroids[edge2cluster]
    #
    #     pos_score = torch.mul(norm_edges, edge2centroids).sum(dim=1)
    #     pos_score = torch.exp(pos_score / self.ssl_temp)
    #     ttl_score = torch.matmul(norm_edges, centroids.transpose(0, 1))
    #     ttl_score = torch.exp(ttl_score / self.ssl_temp).sum(dim=1)
    #
    #     proto_nce_loss = -torch.log(pos_score / ttl_score).mean()
    #     return proto_nce_loss
    #
    # def edges_cluster(self, x):
    #     """Run K-means algorithm to get k clusters of the input tensor x
    #     https://github.com/RUCAIBox/NCL/blob/master/ncl.py
    #     """
    #     # k = min(self.n_protos, x.shape[0])
    #     kmeans = faiss.Kmeans(d=self.embedding_dim, k=self.n_protos, gpu=True)
    #     x = x.detach().cpu()
    #     kmeans.train(x)
    #     cluster_cents = kmeans.centroids
    #
    #     _, I = kmeans.index.search(x, 1)
    #
    #     # convert to cuda Tensors for broadcast
    #     centroids = torch.Tensor(cluster_cents).to(self.device)
    #     centroids = F.normalize(centroids, p=2, dim=1)
    #
    #     node2cluster = torch.LongTensor(I).squeeze().to(self.device)
    #     return centroids, node2cluster
    def member_ssl_loss(self, X_hat, E_mem, masked_hyperedge_index):
        def single_loss(n, e, hyperedge_index, tau):
            e_perm = e[torch.randperm(e.size(0))]
            n_perm = n[torch.randperm(n.size(0))]
            pos = self.f(self.disc_similarity(n[hyperedge_index[0]], e[hyperedge_index[1]]), tau)
            neg_n = self.f(self.disc_similarity(n[hyperedge_index[0]], e_perm[hyperedge_index[1]]), tau)
            neg_e = self.f(self.disc_similarity(n_perm[hyperedge_index[0]], e[hyperedge_index[1]]), tau)

            loss_n = -torch.log(pos / (pos + neg_n))
            loss_e = -torch.log(pos / (pos + neg_e))
            loss_n = loss_n[~torch.isnan(loss_n)]
            loss_e = loss_e[~torch.isnan(loss_e)]
            loss = loss_n + loss_e
            loss = loss.mean()
            return loss

        d_adj_indices = masked_hyperedge_index['diag']
        p_adj_indices = masked_hyperedge_index['proc']
        m_adj_indices = masked_hyperedge_index['med']

        d_member_loss = single_loss(X_hat['diag'], E_mem['diag'], d_adj_indices, self.tau_m)
        p_member_loss = single_loss(X_hat['proc'], E_mem['proc'], p_adj_indices, self.tau_m)
        m_member_loss = single_loss(X_hat['med'], E_mem['med'], m_adj_indices, self.tau_m)


        # n_d = F.normalize(X_hat['diag'][d_adj_indices[0]])
        # n_p = F.normalize(X_hat['proc'][p_adj_indices[0]])
        #
        # n_m = F.normalize(X_hat['med'][m_adj_indices[0]])
        # e_d = F.normalize(E_mem['diag'][d_adj_indices[1]])
        #
        # e_p = F.normalize(E_mem['proc'][p_adj_indices[1]])
        # e_m = F.normalize(E_mem['med'][m_adj_indices[1]])
        #
        # d_member_loss = self.member_infoNCE_loss(n_d, e_d, positive_key=e_d, negative_keys=None)
        # d_member_loss = d_member_loss[~torch.isnan(d_member_loss)]
        #
        # p_member_loss = self.member_infoNCE_loss(n_p, e_p)
        # p_member_loss = p_member_loss[~torch.isnan(p_member_loss)]
        #
        # m_member_loss = self.member_infoNCE_loss(n_m, e_m)
        # m_member_loss = m_member_loss[~torch.isnan(m_member_loss)]

        loss = d_member_loss + p_member_loss + m_member_loss

        return loss

    # def member_ssl_loss(self, X_hat_1, X_hat_2, E_mem_1, E_mem_2, masked_hyperedge_index_1, massk_hyperedge_index_2):
    #     d_adj_indices_1 = masked_hyperedge_index['diag']
    #     p_adj_indices_1 = masked_hyperedge_index['med']
    #     m_adj_indices_1 = masked_hyperedge_index['proc']
    #
    #     loss = 0.
    #     n_d_1 = F.normalize(X_hat_1['diag'][d_adj_indices[0]])
    #     n_d_2 = F.normalize(X_hat_2['diag'][d_adj_indices[0]])
    #     e_d_1 = F.normalize(E_mem_1['dp'][d_adj_indices[1]])
    #     e_d_2 = F.normalize(E_mem_2['dp'][d_adj_indices[1]])
    #     d_member_loss = (self.member_infoNCE_loss(n_d_1, e_d_2) + self.member_infoNCE_loss(n_d_2, e_d_1)) / 2
    #     d_member_loss = d_member_loss[~torch.isnan(d_member_loss)]
    #     loss += d_member_loss
    #
    #     n_p_1 = F.normalize(X_hat_1['proc'][p_adj_indices[0]])
    #     n_p_2 = F.normalize(X_hat_2['proc'][p_adj_indices[0]])
    #     e_p_1 = F.normalize(E_mem_1['dp'][p_adj_indices[1]])
    #     e_p_2 = F.normalize(E_mem_2['dp'][p_adj_indices[1]])
    #     p_member_loss = (self.member_infoNCE_loss(n_p_1, e_p_2) + self.member_infoNCE_loss(n_p_2, e_p_1)) / 2
    #     p_member_loss = p_member_loss[~torch.isnan(p_member_loss)]
    #     loss += p_member_loss
    #
    #     n_m_1 = F.normalize(X_hat_1['med'][m_adj_indices[0]])
    #     n_m_2 = F.normalize(X_hat_2['med'][m_adj_indices[0]])
    #     e_m_1 = F.normalize(E_mem_1['m'][m_adj_indices[1]])
    #     e_m_2 = F.normalize(E_mem_2['m'][m_adj_indices[1]])
    #     m_member_loss = (self.member_infoNCE_loss(n_m_1, e_m_2) + self.member_infoNCE_loss(n_m_2, e_m_1)) / 2
    #     m_member_loss = m_member_loss[~torch.isnan(m_member_loss)]
    #     loss += m_member_loss
    #
    #     return loss

    def edge_ssl_loss(self, E_mem_1, E_mem_2, edge_mask):
        s_loss = 0.
        for n in self.name_lst:
            if n == 'diag' or n == 'proc':
                e1 = E_mem_1['dp'][edge_mask[n]]
                e2 = E_mem_2['dp'][edge_mask[n]]
            else:
                e1 = E_mem_1['m'][edge_mask[n]]
                e2 = E_mem_2['m'][edge_mask[n]]
            s_loss += self.__loss(e1, e2, self.tau_se)

        # cross_domain_mask = edge_mask['diag'] & edge_mask['proc'] & edge_mask['med']
        # c_dp_1 = E_mem_1['dp'][cross_domain_mask]
        # c_dp_2 = E_mem_2['dp'][cross_domain_mask]
        # c_m_1 = E_mem_1['m'][cross_domain_mask]
        # c_m_2 = E_mem_2['m'][cross_domain_mask]
        # c_loss_dp2m = (self.__loss(c_dp_1, c_m_2, self.tau_ce) + self.__loss(c_dp_2, c_m_1, self.tau_ce)) / 2
        # c_loss_m2dp = (self.__loss(c_m_2, c_dp_1, self.tau_ce) + self.__loss(c_m_1, c_dp_2, self.tau_ce)) / 2
        # c_loss = c_loss_dp2m + c_loss_m2dp

        # d_1 = F.normalize(E_mem_1['dp'][edge_mask['diag']])
        # d_2 = F.normalize(E_mem_2['dp'][edge_mask['diag']])
        # p_1 = F.normalize(E_mem_1['dp'][edge_mask['proc']])
        # p_2 = F.normalize(E_mem_2['dp'][edge_mask['proc']])
        # m_1 = F.normalize(E_mem_1['m'][edge_mask['med']])
        # m_2 = F.normalize(E_mem_2['m'][edge_mask['med']])
        # # structure
        # s_loss_d = (self.s_edge_infoNCE_loss(d_1, d_2) + self.s_edge_infoNCE_loss(d_2, d_1)) / 2
        # s_loss_p = (self.s_edge_infoNCE_loss(p_1, p_2) + self.s_edge_infoNCE_loss(p_2, p_1)) / 2
        # s_loss_m = (self.s_edge_infoNCE_loss(m_1, m_2) + self.s_edge_infoNCE_loss(m_2, m_1)) / 2
        # s_loss = s_loss_d + s_loss_p + s_loss_m
        #
        # # cross-domain
        # cross_domain_mask = edge_mask['diag'] & edge_mask['proc'] & edge_mask['med']
        # c_dp_1 = F.normalize(E_mem_1['dp'][cross_domain_mask])
        # c_dp_2 = F.normalize(E_mem_2['dp'][cross_domain_mask])
        # c_m_1 = F.normalize(E_mem_1['m'][cross_domain_mask])
        # c_m_2 = F.normalize(E_mem_2['m'][cross_domain_mask])
        # c_loss_dp2m = (self.c_edge_infoNCE_loss(c_dp_1, c_m_2) + self.c_edge_infoNCE_loss(c_dp_2, c_m_1)) / 2
        # c_loss_m2dp = (self.c_edge_infoNCE_loss(c_m_2, c_dp_1) + self.c_edge_infoNCE_loss(c_m_1, c_dp_2)) / 2
        # c_loss = c_loss_dp2m + c_loss_m2dp

        loss = s_loss
        # loss = s_loss + c_loss
        loss = loss[~torch.isnan(loss)]
        return loss

    def node_ssl_loss(self, X_hat_1, X_hat_2):
        loss = 0.
        for n in self.name_lst:
            n1 = X_hat_1[n]
            n2 = X_hat_2[n]
            loss += self.__loss(n1, n2, self.tau_n)
            # x_1 = F.normalize(X_hat_1[n])
            # x_2 = F.normalize(X_hat_2[n])
            # loss_1 = self.node_infoNCE_loss(x_1, x_2)
            # loss_2 = self.node_infoNCE_loss(x_2, x_1)
            # cur_loss = (loss_1 + loss_2) / 2
            # cur_loss = cur_loss[~torch.isnan(cur_loss)]
            # loss += cur_loss
        return loss

    # def dp2med_ssl_loss(self, E_mem, X_hat):
    #     dp_edge = E_mem['dp']
    #     meds = self.get_hyperedge_representation(X_hat['med'], self.adj_dict['med'])
    #     meds = meds[:self.n_ehr_edges, :]
    #     loss = self.info_nce_loss(dp_edge, meds)
    #     return loss


class TriCL(nn.Module):
    def __init__(self, encoder, embedding_dim, proj_dim: int, num_nodes, num_edges, device):
        super(TriCL, self).__init__()
        self.device = device
        self.encoder = encoder

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_dim = embedding_dim
        self.edge_dim = embedding_dim

        self.node_embedding = nn.Embedding(self.num_nodes, self.node_dim)
        self.edge_embedding = nn.Embedding(self.num_edges + self.num_nodes, self.edge_dim)  # 加上自连边

        self.fc1_n = nn.Linear(self.node_dim, proj_dim)
        self.fc2_n = nn.Linear(proj_dim, self.node_dim)
        self.fc1_e = nn.Linear(self.edge_dim, proj_dim)
        self.fc2_e = nn.Linear(proj_dim, self.edge_dim)

        self.disc = nn.Bilinear(self.node_dim, self.edge_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.node_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        self.fc1_n.reset_parameters()
        self.fc2_n.reset_parameters()
        self.fc1_e.reset_parameters()
        self.fc2_e.reset_parameters()
        self.disc.reset_parameters()

    def get_features(self, adj):
        node_idx = torch.arange(self.num_nodes, device=self.device)
        node_features = self.node_embedding(node_idx)

        edge_idx = torch.arange(self.num_edges + self.num_nodes, device=self.device)
        agg_edge_feat = self.get_hyperedge_representation(node_features, adj)

        edge_features = self.edge_embedding(edge_idx) + torch.cat([agg_edge_feat, node_features], dim=0)
        return node_features, edge_features

    @staticmethod
    def get_hyperedge_representation(embed, adj):
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

    def forward(self, x: Tensor, y: Tensor, hyperedge_index: Tensor):
        """

        Args:
            x: 节点特征
            y: 边特征
            hyperedge_index:

        Returns:

        """
        num_nodes, num_edges = self.num_nodes, self.num_edges

        # if num_nodes is None:
        #     num_nodes = int(hyperedge_index[0].max()) + 1
        # if num_edges is None:
        #     num_edges = int(hyperedge_index[1].max()) + 1

        node_idx = torch.arange(0, num_nodes, device=x.device)
        edge_idx = torch.arange(num_edges, num_edges + num_nodes, device=x.device)
        self_loop = torch.stack([node_idx, edge_idx])
        self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
        H = torch.sparse_coo_tensor(
            indices=self_loop_hyperedge_index,
            values=torch.ones_like(self_loop_hyperedge_index[0, :]),
            size=(num_nodes, num_edges + num_nodes)
        ).coalesce().float()
        n, e = self.encoder(x, y, H)
        return n, e[:num_edges]

    def without_selfloop(self, x: Tensor, hyperedge_index: Tensor, node_mask: Optional[Tensor] = None,
                         num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_nodes is None:
            num_nodes = int(hyperedge_index[0].max()) + 1
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1

        if node_mask is not None:
            node_idx = torch.where(~node_mask)[0]
            edge_idx = torch.arange(num_edges, num_edges + len(node_idx), device=x.device)
            self_loop = torch.stack([node_idx, edge_idx])
            self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
            n, e = self.encoder(x, self_loop_hyperedge_index, num_nodes, num_edges + len(node_idx))
            return n, e[:num_edges]
        else:
            return self.encoder(x, hyperedge_index, num_nodes, num_edges)

    def f(self, x, tau):
        return torch.exp(x / tau)

    def node_projection(self, z: Tensor):
        return self.fc2_n(F.elu(self.fc1_n(z)))

    def edge_projection(self, z: Tensor):
        return self.fc2_e(F.elu(self.fc1_e(z)))

    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def disc_similarity(self, z1: Tensor, z2: Tensor):
        return torch.sigmoid(self.disc(z1, z2)).squeeze()

    def __semi_loss(self, h1: Tensor, h2: Tensor, tau: float, num_negs: Optional[int]):
        if num_negs is None:
            between_sim = self.f(self.cosine_similarity(h1, h2), tau)
            return -torch.log(between_sim.diag() / between_sim.sum(1))
        else:
            pos_sim = self.f(F.cosine_similarity(h1, h2), tau)
            negs = []
            for _ in range(num_negs):
                negs.append(h2[torch.randperm(h2.size(0))])
            negs = torch.stack(negs, dim=-1)
            neg_sim = self.f(F.cosine_similarity(h1.unsqueeze(-1).tile(num_negs), negs), tau)
            return -torch.log(pos_sim / (pos_sim + neg_sim.sum(1)))

    def __semi_loss_batch(self, h1: Tensor, h2: Tensor, tau: float, batch_size: int):
        device = h1.device
        num_samples = h1.size(0)
        num_batches = (num_samples - 1) // batch_size + 1
        indices = torch.arange(0, num_samples, device=device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]
            between_sim = self.f(self.cosine_similarity(h1[mask], h2), tau)

            loss = -torch.log(between_sim[:, i * batch_size: (i + 1) * batch_size].diag() / between_sim.sum(1))
            losses.append(loss)
        return torch.cat(losses)

    def __loss(self, z1: Tensor, z2: Tensor, tau: float, batch_size: Optional[int],
               num_negs: Optional[int], mean: bool):
        if batch_size is None or num_negs is not None:
            l1 = self.__semi_loss(z1, z2, tau, num_negs)
            l2 = self.__semi_loss(z2, z1, tau, num_negs)
        else:
            l1 = self.__semi_loss_batch(z1, z2, tau, batch_size)
            l2 = self.__semi_loss_batch(z2, z1, tau, batch_size)

        loss = (l1 + l2) * 0.5
        loss = loss.mean() if mean else loss.sum()
        return loss

    def node_level_loss(self, n1: Tensor, n2: Tensor, node_tau: float,
                        batch_size: Optional[int] = None, num_negs: Optional[int] = None,
                        mean: bool = True):
        loss = self.__loss(n1, n2, node_tau, batch_size, num_negs, mean)
        return loss

    def group_level_loss(self, e1: Tensor, e2: Tensor, edge_tau: float,
                         batch_size: Optional[int] = None, num_negs: Optional[int] = None,
                         mean: bool = True):
        loss = self.__loss(e1, e2, edge_tau, batch_size, num_negs, mean)
        return loss

    def membership_level_loss(self, n: Tensor, e: Tensor, hyperedge_index: Tensor, tau: float,
                              batch_size: Optional[int] = None, mean: bool = True):
        e_perm = e[torch.randperm(e.size(0))]
        n_perm = n[torch.randperm(n.size(0))]
        if batch_size is None:
            pos = self.f(self.disc_similarity(n[hyperedge_index[0]], e[hyperedge_index[1]]), tau)
            neg_n = self.f(self.disc_similarity(n[hyperedge_index[0]], e_perm[hyperedge_index[1]]), tau)
            neg_e = self.f(self.disc_similarity(n_perm[hyperedge_index[0]], e[hyperedge_index[1]]), tau)

            loss_n = -torch.log(pos / (pos + neg_n))
            loss_e = -torch.log(pos / (pos + neg_e))
        else:
            num_samples = hyperedge_index.shape[1]
            num_batches = (num_samples - 1) // batch_size + 1
            indices = torch.arange(0, num_samples, device=n.device)

            aggr_pos = []
            aggr_neg_n = []
            aggr_neg_e = []
            for i in range(num_batches):
                mask = indices[i * batch_size: (i + 1) * batch_size]

                pos = self.f(self.disc_similarity(n[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]), tau)
                neg_n = self.f(
                    self.disc_similarity(n[hyperedge_index[:, mask][0]], e_perm[hyperedge_index[:, mask][1]]), tau)
                neg_e = self.f(
                    self.disc_similarity(n_perm[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]), tau)

                aggr_pos.append(pos)
                aggr_neg_n.append(neg_n)
                aggr_neg_e.append(neg_e)
            aggr_pos = torch.concat(aggr_pos)
            aggr_neg_n = torch.concat(aggr_neg_n)
            aggr_neg_e = torch.concat(aggr_neg_e)

            loss_n = -torch.log(aggr_pos / (aggr_pos + aggr_neg_n))
            loss_e = -torch.log(aggr_pos / (aggr_pos + aggr_neg_e))

        loss_n = loss_n[~torch.isnan(loss_n)]
        loss_e = loss_e[~torch.isnan(loss_e)]
        loss = loss_n + loss_e
        loss = loss.mean() if mean else loss.sum()
        return loss
