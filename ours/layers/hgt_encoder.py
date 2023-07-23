# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: hgt_encoder.py
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
import faiss

from .hypergraph_gps_model import HypergraphGPSEncoder, Node2EdgeAggregator
from info_nce import InfoNCE


class HGTEncoder(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout, n_layers, adj_dict, voc_dict, cache_dir, device, voc_size_dict,
                 n_ehr_edges, padding_dict):
        super(HGTEncoder, self).__init__()
        self.name_lst = ['diag', 'proc', 'med']
        self.num_dict = voc_size_dict
        self.n_ehr_edges = n_ehr_edges
        self.padding_dict = padding_dict
        self.device = device

        self.adj_dict = {k: v.to(device) for k, v in adj_dict.items()}
        # 为每个超边赋予一个可学习的权重
        self.edge_weight_dict = nn.ParameterDict()
        for n in self.name_lst:
            adj = self.adj_dict[n]
            n_edge = adj.shape[1]
            weight = torch.rand(n_edge)
            weight = torch.nn.Parameter(weight, requires_grad=True).to(device)
            self.edge_weight_dict[n] = weight

        self.info_nce_loss = InfoNCE(reduction='mean')

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

        self.graph_encoder = nn.ModuleDict(
            {
                n: HypergraphGPSEncoder(
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

        self.node2edge_agg = nn.ModuleDict(
            {
                n: Node2EdgeAggregator(embedding_dim, n_heads, dropout)
                for n in self.name_lst
            }
        )

        self.node_in_edge = self.compute_node_id_in_edges()

        self.ssl_temp = 0.1
        self.ssl_reg = 1e-3
        self.embedding_dim = embedding_dim
        self.n_protos = 128
        self.domain_ssl_weight = 0.1
        self.cluster_ssl_weight = 1.
        self.node2edge_ssl_weight = 0.1
        self.dp2med_ssl_weight = 0.1

    def compute_node_id_in_edges(self):
        node_in_edge = {}
        for n in self.name_lst:
            adj = self.adj_dict[n].to_dense()[:, :self.n_ehr_edges]
            assert adj.shape == (self.num_dict[n], self.n_ehr_edges)

            max_size = torch.sum(adj, 0).max().int().item()
            pos_matrix = torch.empty((self.n_ehr_edges, max_size))
            pos_matrix = torch.fill(pos_matrix, -1)
            pos_idx = torch.argwhere(adj.T == 1).cpu()
            for i in range(self.n_ehr_edges):
                indices = pos_idx[pos_idx[:, 0] == i][:, 1]  # 找到第i条边包含哪几个节点
                length = len(indices)
                pos_matrix[i, :length] = indices
            pos_matrix = torch.where(pos_matrix == -1, self.padding_dict[n], pos_matrix)
            node_in_edge[n] = pos_matrix.long()
        return node_in_edge

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
        for n in self.name_lst:  # dp和m
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

    def encode(self):
        # 首先是embedding层
        raw_embedding = self.get_embedding()  # dict
        # 最开始是超图编码层
        X_hat, E_hat, E_mem = self.graph_encode(raw_embedding, self.adj_dict, self.edge_weight_dict)
        # 将X_hat表示成visit-level
        return X_hat, E_hat, E_mem

    def compute_loss(self):
        # dp与m之间做InfoNCE
        X_hat, E_hat, E_mem = self.encode()
        edge_ssl_loss = self.edge_ssl_loss(E_mem)

        # E_mem['m']与包含的药物之间应该也做InfoNCE
        # 这里的问题是如何将解码阶段多头注意力的权重引入进来
        dp2med_ssl_loss = self.dp2med_ssl_loss(E_mem, X_hat)

        # 节点与聚类中心之间做InfoNCE,能让节点更靠近
        dp_centroids, dp_edge2cluster = self.edges_cluster(E_mem['dp'])
        m_centroids, m_edge2cluster = self.edges_cluster(E_mem['m'])
        dp_cluster_ssl = self.cluster_ssl(E_mem['dp'], dp_centroids, dp_edge2cluster)
        m_cluster_ssl = self.cluster_ssl(E_mem['m'], m_centroids, m_edge2cluster)

        # 超边表示与节点之间做InfoNCE,将edge表示与node表示统一起来
        ## 邻接矩阵左乘X,然后乘度矩阵
        node2edge_ssl = self.node2edge_ssl(X_hat, E_mem)

        ssl_loss = edge_ssl_loss * self.domain_ssl_weight \
                   + (dp_cluster_ssl + m_cluster_ssl) * self.cluster_ssl_weight \
                   + node2edge_ssl * self.node2edge_ssl_weight\
                   + dp2med_ssl_loss * self.dp2med_ssl_weight
        return ssl_loss

    def node2edge_ssl(self, X_hat, E_mem):
        edge_embed = {}
        for n in self.name_lst:
            node_in_edge = self.node_in_edge[n]
            padding_row = self.embedding[n](
                torch.tensor(self.num_dict[n], dtype=torch.long, device=self.device)).reshape(1, self.embedding_dim)
            X_hat[n] = torch.vstack([X_hat[n], padding_row])
            edge_embed_with_node = X_hat[n][node_in_edge]
            edge_embed_agg = self.node2edge_agg[n](edge_embed_with_node).squeeze(1)
            edge_embed[n] = edge_embed_agg

        # 计算dp表示
        dp_edge_embedding = edge_embed['diag'] + edge_embed['proc']
        m_edge_embedding = edge_embed['med']
        dp_ssl_loss = self.info_nce_loss(dp_edge_embedding, E_mem['dp'])
        m_ssl_loss = self.info_nce_loss(m_edge_embedding, E_mem['m'])
        ssl_loss = dp_ssl_loss + m_ssl_loss
        return ssl_loss

    def cluster_ssl(self, edges, centroids, edge2cluster):
        norm_edges = F.normalize(edges)
        edge2centroids = centroids[edge2cluster]

        pos_score = torch.mul(norm_edges, edge2centroids).sum(dim=1)
        pos_score = torch.exp(pos_score / self.ssl_temp)
        ttl_score = torch.matmul(norm_edges, centroids.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.ssl_temp).sum(dim=1)

        proto_nce_loss = -torch.log(pos_score / ttl_score).mean()
        return proto_nce_loss

    def edges_cluster(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        https://github.com/RUCAIBox/NCL/blob/master/ncl.py
        """
        # k = min(self.n_protos, x.shape[0])
        kmeans = faiss.Kmeans(d=self.embedding_dim, k=self.n_protos, gpu=True)
        x = x.detach().cpu()
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def edge_ssl_loss(self, E_mem):
        dp = E_mem['dp']
        m = E_mem['m']
        loss = self.info_nce_loss(dp, m)
        return loss.mean()

    def dp2med_ssl_loss(self, E_mem, X_hat):
        dp_edge = E_mem['dp']
        meds = self.get_hyperedge_representation(X_hat['med'], self.adj_dict['med'])
        meds = meds[:self.n_ehr_edges, :]
        loss = self.info_nce_loss(dp_edge, meds)
        return loss

