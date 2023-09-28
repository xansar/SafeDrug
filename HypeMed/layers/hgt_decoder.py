
import torch
import torch.nn as nn
import torch.nn.functional as F
from info_nce import InfoNCE

from .ehr_memory_attn import EHRMemoryAttention, HistoryAttention
from .hgt_encoder import Node2EdgeAggregator


class HGTDecoder(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout, n_ehr_edges, voc_size_dict, padding_dict, device, X_hat, E_mem, ddi_adj, channel_ablation=None, embed_ablation=None):
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
        E_mem = {
            'd': E_mem['diag'],
            'p': E_mem['proc'],
            'm': E_mem['med']
        }
        self.embed_ablation = embed_ablation
        if self.embed_ablation is None:
            self.X_hat = nn.ModuleDict({
                # n: nn.Embedding(voc_size_dict[n] + 1, embedding_dim).from_pretrained(torch.randn_like(X_hat[n]).to(device), freeze=False, padding_idx=padding_dict[n])
                n: nn.Embedding(voc_size_dict[n] + 1, embedding_dim).from_pretrained(X_hat[n].to(device), freeze=False, padding_idx=padding_dict[n])
                for n in self.name_lst
            })
            self.E_mem = nn.ModuleDict({
                # n: nn.Embedding(E_mem[n].shape[0], embedding_dim).from_pretrained(torch.randn_like(E_mem[n]).to(device), freeze=False)
                n: nn.Embedding(E_mem[n].shape[0], embedding_dim).from_pretrained(E_mem[n].to(device), freeze=False)
                for n in ['d', 'p', 'm']
            })
        else:
            print(f'embed_ablation: {self.embed_ablation}')
            if self.embed_ablation == 'random':
                self.X_hat = nn.ModuleDict({
                    n: nn.Embedding(voc_size_dict[n] + 1, embedding_dim).from_pretrained(torch.randn_like(X_hat[n]).to(device), freeze=False, padding_idx=padding_dict[n])
                    # n: nn.Embedding(voc_size_dict[n] + 1, embedding_dim).from_pretrained(X_hat[n].to(device),
                    #                                                                      freeze=False,
                    #                                                                      padding_idx=padding_dict[n])
                    for n in self.name_lst
                })
                self.E_mem = nn.ModuleDict({
                    n: nn.Embedding(E_mem[n].shape[0], embedding_dim).from_pretrained(torch.randn_like(E_mem[n]).to(device), freeze=False)
                    # n: nn.Embedding(E_mem[n].shape[0], embedding_dim).from_pretrained(E_mem[n].to(device), freeze=False)
                    for n in ['d', 'p', 'm']
                })
            elif self.embed_ablation == 'fix':
                self.X_hat = nn.ModuleDict({
                    # n: nn.Embedding(voc_size_dict[n] + 1, embedding_dim).from_pretrained(torch.randn_like(X_hat[n]).to(device), freeze=False, padding_idx=padding_dict[n])
                    n: nn.Embedding(voc_size_dict[n] + 1, embedding_dim).from_pretrained(X_hat[n].to(device),
                                                                                         freeze=True,
                                                                                         padding_idx=padding_dict[n])
                    for n in self.name_lst
                })
                self.E_mem = nn.ModuleDict({
                    # n: nn.Embedding(E_mem[n].shape[0], embedding_dim).from_pretrained(torch.randn_like(E_mem[n]).to(device), freeze=False)
                    n: nn.Embedding(E_mem[n].shape[0], embedding_dim).from_pretrained(E_mem[n].to(device), freeze=True)
                    for n in ['d', 'p', 'm']
                })
            else:
                raise ValueError

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
        self.from_diag = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, embedding_dim),
                nn.Dropout(dropout),
                # nn.Tanh()
            )
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

        self.dp_context_attn = HistoryAttention(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.mem_context_attn = HistoryAttention(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.fusion_pred_norm = nn.LayerNorm(self.voc_size_dict['med'])
        self.pred_bias = torch.nn.Parameter(torch.zeros(self.voc_size_dict['med']), requires_grad=True)

        self.channel_ablation = channel_ablation
        if channel_ablation is None:
            self.gate_control = nn.Sequential(
                nn.Linear(3 * embedding_dim, 3),
                nn.Dropout(dropout),
                # nn.Tanh()
            )
        else:
            print(f'channel_ablation: {self.channel_ablation}')
            self.gate_control = nn.Sequential(
                nn.Linear(2 * embedding_dim, 2),
                nn.Dropout(dropout),
                # nn.Tanh()
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

    def domain_interact(self, v_d, v_p, v_hm):
        i_v_d = self.from_history_med(v_hm) + v_d
        i_v_p = self.from_diag(v_hm) + v_p
        patient_rep = self.from_dp(torch.cat([i_v_d, i_v_p], dim=1)) + i_v_d, i_v_p
        return patient_rep


    def forward(self, records, masks, true_visit_idx, visit2edge_idx):
        assert len(visit2edge_idx) == true_visit_idx.sum().item()
        X_hat = self.X_hat
        E_mem = {
            'd': self.E_mem['d'](torch.arange(self.n_ehr_edges).to(self.device)),
            'p': self.E_mem['p'](torch.arange(self.n_ehr_edges).to(self.device)),
            'm': self.E_mem['m'](torch.arange(self.n_ehr_edges).to(self.device))
        }

        # 解析序列数据
        entity_seq_embed = {}  # (bsz, max_vist, max_size, dim)
        for n in self.name_lst:
            entity_seq_embed[n] = self.embedding_norm[n](X_hat[n](records[n]))

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
        med_history = med_history[true_visit_idx]  # 只保留非空的visit

        # 先上下文再记忆
        # 这里将diag和proc还有历史用药拼起来
        # 增加因果关系进来,
        # diag ---> proc     last_med
        #  |          |          |
        #  |--->med<--|----------|
        # visit_rep = visit_seq_embed['diag'] + visit_seq_embed['proc']
        diag_rep = visit_seq_embed['diag']
        proc_rep = visit_seq_embed['proc'] + self.from_diag(diag_rep)
        visit_rep = diag_rep + proc_rep
        batch_size, max_visit, dim = visit_rep.shape
        visit_rep = visit_rep.reshape(batch_size * max_visit, dim)
        visit_rep = visit_rep[true_visit_idx]  # 只保留非空的visit

        E_mem_patient_rep = E_mem['d'] + self.from_diag(E_mem['d']) + E_mem['p']
        E_mem_med_rep = E_mem['m']
        visit_rep_mem = self.visit_mem_attn(
             visit_rep, E_mem_patient_rep, E_mem_med_rep
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




        # 计算包含上下文信息的表示
        # attn_mask = masks['attn_mask'].repeat(self.n_heads, 1, 1)
        attn_mask = masks['attn_mask']
        attn_mask = attn_mask[true_visit_idx][:, true_visit_idx]
        assert attn_mask.shape == (visit_rep.shape[0], visit_rep.shape[0])
        dp_context_rep = self.dp_context_attn(visit_rep, attn_mask)
        med_context_rep = self.med_context_attn(med_history, attn_mask)

        mem_context_rep = self.mem_context_attn(visit_rep_mem, attn_mask)

        if self.channel_ablation is None:
            # 先融合再预测
            cat_rep = torch.cat([
                dp_context_rep.unsqueeze(-1),
                mem_context_rep.unsqueeze(-1),
                med_context_rep.unsqueeze(-1)],
                -1)
            # gate = self.gate_control(cat_rep.reshape(-1, 3 * mem_context_rep.shape[-1])).reshape(-1, mem_context_rep.shape[-1], 3)
            gate = self.gate_control(cat_rep.reshape(-1, 3 * mem_context_rep.shape[-1])).reshape(-1, 1, 3)
            # assert len(gate.shape) == 3 and gate.shape[-1] == 3
            gate = torch.softmax(gate, -1)
            fusion_rep = (gate * cat_rep).sum(-1)
        else:
            if self.channel_ablation == 'dp':
                cat_rep = torch.cat([
                    mem_context_rep.unsqueeze(-1),
                    med_context_rep.unsqueeze(-1)],
                    -1)
            elif self.channel_ablation == 'med':
                cat_rep = torch.cat([
                    dp_context_rep.unsqueeze(-1),
                    mem_context_rep.unsqueeze(-1)],
                    -1)
            elif self.channel_ablation == 'mem':
                cat_rep = torch.cat([
                    dp_context_rep.unsqueeze(-1),
                    med_context_rep.unsqueeze(-1)],
                    -1)
            else:
                raise ValueError
            # gate = self.gate_control(cat_rep.reshape(-1, 2 * mem_context_rep.shape[-1])).reshape(-1, mem_context_rep.shape[-1], 2)
            gate = self.gate_control(cat_rep.reshape(-1, 2 * mem_context_rep.shape[-1])).reshape(-1, 1, 2)
            assert len(gate.shape) == 3 and gate.shape[-1] == 2
            gate = torch.softmax(gate, -1)
            fusion_rep = (gate * cat_rep).sum(-1)


        med_rep = X_hat['med'](torch.arange(self.num_dict['med'], dtype=torch.long, device=self.device))

        fusion_output = torch.matmul(fusion_rep, med_rep.T) + self.pred_bias
        output = self.fusion_pred_norm(fusion_output)

        neg_pred_prob = F.sigmoid(output)
        neg_pred_prob = neg_pred_prob.unsqueeze(-1)
        neg_pred_prob = neg_pred_prob.transpose(-1, -2) * neg_pred_prob  # (true visit num, voc_size, voc_size)

        # loss_mask = (masks['key_padding_mask'] == False).unsqueeze(-1).unsqueeze(-1)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        # 计算ssl
        side_loss = {
            'ddi': batch_neg,
            'ssl': ssl_loss,
            # 'moe': moe_loss
        }

        return output, side_loss