# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: pretrainer.py
@time: 2023/7/11 18:47
@e-mail: xansar@ruc.edu.cn
"""
import sys
sys.path.append("..")
from layers import TriCL, HGTEncoder
from torch.optim import Adam
from tqdm import tqdm, trange
import torch
from .utils import drop_incidence, drop_features, valid_node_edge_mask, hyperedge_index_masking


class BasePretrainer:
    def __init__(self):
        super(BasePretrainer, self).__init__()

    def build_encoder(self, *args, **kwargs):
        raise NotImplementedError

    def pretrain(self, *args, **kwargs):
        raise NotImplementedError

    def get_encoded_embedding(self, *args, **kwargs):
        raise NotImplementedError


# class HGTPretrainer(BasePretrainer):
#     def __init__(
#             self,
#             model,
#             adj_dict,
#             voc_size_dict,
#             n_ehr_edges,
#             device,
#             args,
#     ):
#         super(HGTPretrainer, self).__init__()
#         self.name_lst = ['diag', 'proc', 'med']
#         pretrain_epoch = args.pretrain_epoch
#         pretrain_lr = args.pretrain_lr
#         pretrain_weight_decay = args.pretrain_weight_decay
#         self.model = model
#         self.device = device
#         self.n_ehr_edges = n_ehr_edges
#         self.pretrain_epoch = pretrain_epoch
#         self.num_dict = voc_size_dict
#         self.adj_dict = {n: adj_dict[n].to(device) for n in self.name_lst}
#         self.w_edge = args.w_edge
#         self.w_member = args.w_member
#         self.drop_feature_ratio = args.drop_feature_rate
#         self.drop_edge_ratio = args.drop_edge_rate
#         # self.model = self.build_encoder(embedding_dim, n_heads, n_layers, dropout, n_ehr_edges, voc_size_dict,
#         #                                 adj_dict, padding_dict, voc_dict, device, cache_dir)
#         self.optimizer = self.build_optimizer(self.model, lr=pretrain_lr, weight_decay=pretrain_weight_decay)
#
#     @staticmethod
#     def build_optimizer(encoder, **kwargs):
#         lr = kwargs['lr']
#         weight_decay = kwargs['weight_decay']
#         optimizer = Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
#         return optimizer
#
#     # @staticmethod
#     # def build_encoder(embedding_dim, n_heads, n_layers, dropout, n_ehr_edges, voc_size_dict, adj_dict,
#     #                   padding_dict, voc_dict, device, cache_dir):
#     #     encoder = HGTCL(
#     #         voc_size_dict=voc_size_dict,
#     #         adj_dict=adj_dict,
#     #         voc_dict=voc_dict,
#     #         n_ehr_edges=n_ehr_edges,
#     #         embedding_dim=embedding_dim,
#     #         n_heads=n_heads,
#     #         n_layers=n_layers,
#     #         dropout=dropout,
#     #         device=device,
#     #         cache_dir=cache_dir
#     #     )
#     #     return encoder.to(device)
#     @staticmethod
#     def filter_incidence(row, col, hyperedge_attr, mask):
#         return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]
#     def drop_edge(self, adj, p: float = 0.2):
#         hyperedge_index = adj.indices()
#         if p == 0.0:
#             return hyperedge_index
#
#         row, col = hyperedge_index
#         mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p
#
#         row, col, _ = self.filter_incidence(row, col, None, mask)
#         hyperedge_index = torch.stack([row, col], dim=0)
#         return hyperedge_index
#
#     @staticmethod
#     def drop_feat(x, p: float):
#         drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
#         x = x.clone()
#         x[:, drop_mask] = 0
#         return x
#
#     @staticmethod
#     def valid_node_edge_mask(hyperedge_index, num_nodes: int, num_edges: int):
#         ones = hyperedge_index.new_ones(hyperedge_index.shape[1])
#         Dn = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)
#         De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_edges)
#         node_mask = Dn != 0
#         edge_mask = De != 0
#         return edge_mask
#
#     @staticmethod
#     def hyperedge_index_masking(adj, num_nodes, num_edges, node_mask, edge_mask):
#         hyperedge_index = adj.indices()
#         if node_mask is None and edge_mask is None:
#             return hyperedge_index
#
#         H = torch.sparse_coo_tensor(hyperedge_index, \
#                                     hyperedge_index.new_ones((hyperedge_index.shape[1],)),
#                                     (num_nodes, num_edges)).to_dense()
#         if node_mask is not None and edge_mask is not None:
#             masked_hyperedge_index = H[node_mask][:, edge_mask].to_sparse().indices()
#         elif node_mask is None and edge_mask is not None:
#             masked_hyperedge_index = H[:, edge_mask].to_sparse().indices()
#         elif node_mask is not None and edge_mask is None:
#             masked_hyperedge_index = H[node_mask].to_sparse().indices()
#         return masked_hyperedge_index
#
#     def pretrain(self):
#
#         for i in trange(self.pretrain_epoch):
#             self.optimizer.zero_grad()
#             X, E = self.model.get_features()
#             X_1 = {n: self.drop_feat(X[n], self.drop_feature_ratio)for n in self.name_lst}
#             X_2 = {n: self.drop_feat(X[n], self.drop_feature_ratio)for n in self.name_lst}
#             E_1 = {n: self.drop_feat(E[n], self.drop_feature_ratio)for n in self.name_lst}
#             E_2 = {n: self.drop_feat(E[n], self.drop_feature_ratio)for n in self.name_lst}
#             adj_dict_1 = {n: self.drop_edge(self.adj_dict[n], self.drop_edge_ratio) for n in self.name_lst}
#             adj_dict_2 = {n: self.drop_edge(self.adj_dict[n], self.drop_edge_ratio) for n in self.name_lst}
#             X_hat_1, E_mem_1 = self.model(adj_dict_1, X_1, E_1)
#             X_hat_2, E_mem_2 = self.model(adj_dict_2, X_2, E_2)
#
#             # Projection Head
#             X_hat_1, X_hat_2 = self.model.node_projection(X_hat_1), self.model.node_projection(X_hat_2)
#             E_mem_1, E_mem_2 = self.model.edge_projection(E_mem_1), self.model.edge_projection(E_mem_2)
#
#             node_ssl_loss = self.model.node_ssl_loss(X_hat_1, X_hat_2)
#
#             n_edges = self.n_ehr_edges
#             edge_masks_1 = {n: self.valid_node_edge_mask(adj_dict_1[n], self.num_dict[n], n_edges) for n in self.name_lst}
#             edge_masks_2 = {n: self.valid_node_edge_mask(adj_dict_2[n], self.num_dict[n], n_edges) for n in self.name_lst}
#             edge_mask = {n: edge_masks_1[n] & edge_masks_2[n] for n in self.name_lst}
#
#             edge_ssl_loss = self.model.edge_ssl_loss(E_mem_1, E_mem_2, edge_mask)
#
#
#             # 把drop后度数为0的边去掉
#             masked_E_mem_1 = {
#                 'diag': E_mem_1['dp'][edge_masks_1['diag']],
#                 'proc': E_mem_1['dp'][edge_masks_1['proc']],
#                 'med': E_mem_1['m'][edge_masks_1['med']],
#             }
#             masked_E_mem_2 = {
#                 'diag': E_mem_2['dp'][edge_masks_2['diag']],
#                 'proc': E_mem_2['dp'][edge_masks_2['proc']],
#                 'med': E_mem_2['m'][edge_masks_2['med']],
#             }
#             masked_hyperedge_index_1 = {
#                 n: self.hyperedge_index_masking(self.adj_dict[n], self.num_dict[n], n_edges, None, edge_masks_1[n])
#                 for n in self.name_lst
#             }
#             masked_hyperedge_index_2 = {
#                 n: self.hyperedge_index_masking(self.adj_dict[n], self.num_dict[n], n_edges, None, edge_masks_2[n])
#                 for n in self.name_lst
#             }
#             member_ssl_loss_1 = self.model.member_ssl_loss(X_hat_1, masked_E_mem_2, masked_hyperedge_index_2)
#             member_ssl_loss_2 = self.model.member_ssl_loss(X_hat_2, masked_E_mem_1, masked_hyperedge_index_1)
#             member_ssl_loss = (member_ssl_loss_1 + member_ssl_loss_2) / 2
#
#             loss = node_ssl_loss + self.w_edge * edge_ssl_loss + self.w_member * member_ssl_loss
#             loss.backward()
#             self.optimizer.step()
#             print(f'epoch {i}: {loss.item()}')
#
#     def get_encoded_features(self):
#         X, E = self.model.get_features()
#         hyperedge_index_dict = {n: self.adj_dict[n].indices() for n in self.name_lst}
#         X_hat, E_mem = self.model(hyperedge_index_dict, X, E)
#         return X_hat, E_mem
#
#     # def get_encoded_embedding(self):
#     #     X_hat, E_hat, E_mem = self.model.encode()
#     #     dp_centroids, dp_edge2cluster = self.model.edges_cluster(E_mem['dp'])
#     #     m_centroids, m_edge2cluster = self.model.edges_cluster(E_mem['m'])
#     #     res = {
#     #         'X': X_hat,
#     #         'E': E_hat,
#     #         'E_mem': E_mem,
#     #         'cluster': {
#     #             'dp': {
#     #                 'centroids': dp_centroids,
#     #                 'edge2cluster': dp_edge2cluster
#     #             },
#     #             'm': {
#     #                 'centroids': m_centroids,
#     #                 'edge2cluster': m_edge2cluster
#     #             }
#     #         }
#     #     }
#     #     return res
class HGTPretrainer(BasePretrainer):
    def __init__(
            self,
            args,
            num_dict,
            num_edges,
            adj_dict,
            device,
            voc_dict,
            cache_dir,

    ):
        super(HGTPretrainer, self).__init__()
        pretrain_epoch = args.pretrain_epoch
        pretrain_lr = args.pretrain_lr
        pretrain_weight_decay = args.pretrain_weight_decay
        self.name_lst = ['diag', 'proc', 'med']
        self.num_negs = None
        self.params = {
            'drop_incidence_rate': args.drop_incidence_rate,
            'drop_feature_rate': args.drop_feature_rate,
            'tau_n': args.tau_n,
            'tau_g': args.tau_g,
            'tau_m': args.tau_m,
            'batch_size_1': args.batch_size_1,
            'batch_size_2': args.batch_size_2,
            'w_g': args.w_g,
            'w_m': args.w_m,
        }

        self.num_dict = num_dict
        self.num_edges = num_edges

        self.adj_dict = {n: adj_dict[n].to(device) for n in self.name_lst}

        self.device = device
        self.pretrain_epoch = pretrain_epoch
        self.model_dict = {
            n: self.build_model(
                args, num_dict[n], self.num_edges,
                adj=adj_dict[n],
                idx2word=voc_dict[n].idx2word,
                cache_dir=cache_dir,
                device=device,
                name=n,
            )
            for n in self.name_lst
        }

        self.optimizer_dict = {
            n: self.build_optimizer(self.model_dict[n], lr=pretrain_lr, weight_decay=pretrain_weight_decay)
            for n in self.name_lst
        }

    @staticmethod
    def build_model(args, num_nodes, num_edges, adj, idx2word, cache_dir, device, name):
        encoder = HGTEncoder(
            embed_dim=args.dim,
            n_heads=args.n_heads,
            dropout=args.dropout,
            n_layers=args.n_layers,
            H=adj,
            idx2word=idx2word,
            cache_dir=cache_dir,
            device=device,
            name=name,
        ).to(device)

        model = TriCL(
            encoder,
            embedding_dim=args.dim,
            proj_dim=args.proj_dim,
            num_nodes=num_nodes,
            num_edges=num_edges,
            device=device
        ).to(device)
        return model

    @staticmethod
    def build_optimizer(encoder, **kwargs):
        lr = kwargs['lr']
        weight_decay = kwargs['weight_decay']
        optimizer = Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

    @staticmethod
    def get_raw_node_edge_representation(model, adj):
        node_features, edge_features = model.get_features(adj)
        return node_features, edge_features

    def step(self, model, optimizer, adj, num_nodes):
        hyperedge_index = adj.indices()
        num_edges = self.num_edges

        params = self.params
        num_negs = self.num_negs
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Hypergraph Augmentation
        hyperedge_index1 = drop_incidence(hyperedge_index, params['drop_incidence_rate'])
        hyperedge_index2 = drop_incidence(hyperedge_index, params['drop_incidence_rate'])
        node_features, edge_features = model.get_features(adj)
        x1 = drop_features(node_features, params['drop_feature_rate'])
        x2 = drop_features(node_features, params['drop_feature_rate'])
        y1 = drop_features(edge_features, params['drop_feature_rate'])
        y2 = drop_features(edge_features, params['drop_feature_rate'])

        node_mask1, edge_mask1 = valid_node_edge_mask(hyperedge_index1, num_nodes, num_edges)
        node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)
        node_mask = node_mask1 & node_mask2
        edge_mask = edge_mask1 & edge_mask2

        # Encoder
        n1, e1 = model(x1, y1, hyperedge_index1)
        n2, e2 = model(x2, y2, hyperedge_index2)

        # Projection Head
        n1, n2 = model.node_projection(n1), model.node_projection(n2)
        e1, e2 = model.edge_projection(e1), model.edge_projection(e2)

        loss_n = model.node_level_loss(n1, n2, params['tau_n'], batch_size=params['batch_size_1'], num_negs=num_negs)
        loss_g = model.group_level_loss(e1[edge_mask], e2[edge_mask], params['tau_g'],
                                        batch_size=params['batch_size_1'], num_negs=num_negs)

        masked_index1 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, edge_mask1)
        masked_index2 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, edge_mask2)
        loss_m1 = model.membership_level_loss(n1, e2[edge_mask2], masked_index2, params['tau_m'],
                                              batch_size=params['batch_size_2'])
        loss_m2 = model.membership_level_loss(n2, e1[edge_mask1], masked_index1, params['tau_m'],
                                              batch_size=params['batch_size_2'])
        loss_m = (loss_m1 + loss_m2) * 0.5

        loss = loss_n + params['w_g'] * loss_g + params['w_m'] * loss_m
        loss.backward()
        optimizer.step()
        return loss.item()

    def pretrain(self):
        for i in trange(self.pretrain_epoch):
            loss_dict = {}
            for n in self.name_lst:
                loss_dict[n] = self.step(self.model_dict[n], self.optimizer_dict[n], self.adj_dict[n], self.num_dict[n])
            print(f'epoch_{i}: diag-{loss_dict["diag"]}\tproc-{loss_dict["proc"]}\tmed-{loss_dict["med"]}\n')

    def get_encoded_embedding(self, model, adj):
        model = model
        node_features, edge_features = model.get_features(adj)
        hyperedge_index = adj.indices()
        X_hat, E_hat = model(node_features, edge_features, hyperedge_index)
        res = {
            'X': X_hat,
            'E': E_hat,
        }
        return res
