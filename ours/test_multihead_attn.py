# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: test_multihead_attn.py
@time: 2023/6/11 20:22
@e-mail: xansar@ruc.edu.cn
"""
import torch
from torch.nn import MultiheadAttention, TransformerEncoderLayer
if __name__ == '__main__':
    bsz = 16
    max_visit = 4
    # seq = torch.tensor(
    #     [[0, 1, 2, 3]]
    # )

    embedding_dim = 2
    n_heads = 2
    dropout = 0.

    embed = torch.randn(bsz, max_visit, embedding_dim)

    key_padding_mask = torch.tensor([
        [False, False, True, True]
    ])

    attn_mask = torch.tensor([
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [True, False, False, False],
        ]
    ]).repeat(n_heads, 1, 1)


    q = torch.randn(bsz, embedding_dim * 2)
    k = torch.randn(bsz, embedding_dim * 2)
    v = torch.randn(bsz, embedding_dim)
    mla = MultiheadAttention(
        embed_dim=embedding_dim * 2,
        kdim=embedding_dim * 2,
        vdim=embedding_dim,
        num_heads=n_heads,
        dropout=dropout,
        batch_first=True,
    )
    out = mla(
        query=q, key=k, value=v, need_weights=False,
        key_padding_mask=None,
        # 用来遮挡key中的padding,和key一样的shape,这里应该需要把补足用的visitmask上,bsz_max_visit
        attn_mask=None,  # 加上会输出nan,因为有些地方的是补0的,整行都是True,相当于不计算
    )[0]
    print(out)
