# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: build_pretrainer.py
@time: 2023/7/11 18:47
@e-mail: xansar@ruc.edu.cn
"""
from layers import HGTEncoder
from torch.optim import Adam
from tqdm import tqdm, trange


class BasePretrainer:
    def __init__(self):
        super(BasePretrainer, self).__init__()

    def build_encoder(self, *args, **kwargs):
        raise NotImplementedError

    def pretrain(self, *args, **kwargs):
        raise NotImplementedError

    def get_encoded_embedding(self):
        raise NotImplementedError


class HGTPretrainer(BasePretrainer):
    def __init__(
            self,
            embedding_dim,
            n_heads,
            n_layers,
            dropout,
            n_ehr_edges,
            voc_size_dict,
            adj_dict,
            padding_dict,
            voc_dict,
            device,
            cache_dir,
            pretrain_args,

    ):
        super(HGTPretrainer, self).__init__()
        pretrain_epoch = pretrain_args['pretrain_epoch']
        pretrain_lr = pretrain_args['pretrain_lr']
        pretrain_weight_decay = pretrain_args['pretrain_weight_decay']

        self.device = device
        self.pretrain_epoch = pretrain_epoch
        self.encoder = self.build_encoder(embedding_dim, n_heads, n_layers, dropout, n_ehr_edges, voc_size_dict,
                                          adj_dict, padding_dict, voc_dict, device, cache_dir)
        self.optimizer = self.build_optimizer(self.encoder, lr=pretrain_lr, weight_decay=pretrain_weight_decay)

    @staticmethod
    def build_optimizer(encoder, **kwargs):
        lr = kwargs['lr']
        weight_decay = kwargs['weight_decay']
        optimizer = Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

    @staticmethod
    def build_encoder(embedding_dim, n_heads, n_layers, dropout, n_ehr_edges, voc_size_dict, adj_dict,
                      padding_dict, voc_dict, device, cache_dir):
        encoder = HGTEncoder(
            voc_size_dict=voc_size_dict,
            adj_dict=adj_dict,
            padding_dict=padding_dict,
            voc_dict=voc_dict,
            n_ehr_edges=n_ehr_edges,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            device=device,
            cache_dir=cache_dir
        )
        return encoder.to(device)

    def pretrain(self):
        self.encoder = self.encoder.to(self.device)
        for i in trange(self.pretrain_epoch):
            self.optimizer.zero_grad()
            loss = self.encoder.compute_loss()
            loss.backward()
            self.optimizer.step()

    def get_encoded_embedding(self):
        X_hat, E_hat, E_mem = self.encoder.encode()
        dp_centroids, dp_edge2cluster = self.encoder.edges_cluster(E_mem['dp'])
        m_centroids, m_edge2cluster = self.encoder.edges_cluster(E_mem['m'])
        res = {
            'X': X_hat,
            'E': E_hat,
            'E_mem': E_mem,
            'cluster': {
                'dp': {
                    'centroids': dp_centroids,
                    'edge2cluster': dp_edge2cluster
                },
                'm': {
                    'centroids': m_centroids,
                    'edge2cluster': m_edge2cluster
                }
            }
        }
        return res

