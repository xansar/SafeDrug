# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: test_multihead_attn.py
@time: 2023/6/11 20:22
@e-mail: xansar@ruc.edu.cn
"""
import torch
from torch.nn import MultiheadAttention
if __name__ == '__main__':
    bsz = 1
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

    mla = MultiheadAttention(
        embed_dim=embedding_dim,
        num_heads=n_heads,
        dropout=dropout,
        batch_first=True,
    )

    out = mla(
        query=embed, key=embed, value=embed, need_weights=False,
        # todo: 这里注意,med history的第一个是空的
        key_padding_mask=key_padding_mask,
        # 用来遮挡key中的padding,和key一样的shape,这里应该需要把补足用的visitmask上,bsz_max_visit
        attn_mask=attn_mask,  # 加上会输出nan,因为有些地方的是补0的,整行都是True,相当于不计算
    )[0]
    print(out)
