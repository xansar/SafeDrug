# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: config.py
@time: 2023/6/11 15:43
@e-mail: xansar@ruc.edu.cn
"""
import argparse

# setting
model_name = "HGTDrugRec"
# resume_path = 'saved/{}/Epoch_49_TARGET_0.06_JA_0.5183_DDI_0.05854.model'.format(model_name)
# resume_path = "Epoch_49_TARGET_0.06_JA_0.5183_DDI_0.05854.model"
# resume_path = f'saved/{model_name}/Epoch_49_TARGET_0.06_JA_0.5155_DDI_0.0611.model'
resume_path = f'saved/{model_name}/'

"""HyperDrugRec"""
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--Test", action="store_true", default=False, help="test mode")
#     parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
#     parser.add_argument("--wandb", action="store_true", default=False, help="use wandb to log")
#     parser.add_argument("--model_name", type=str, default=model_name, help="model name")
#     parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
#     parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
#     parser.add_argument("--bsz", type=int, default=16, help="batch size")
#     parser.add_argument("--eval_bsz", type=int, default=1, help="eval batch size")
#     parser.add_argument("--win_sz", type=int, default=2, help="seq window size")
#     parser.add_argument("--n_layers", type=int, default=1, help="num of gnn layers")
#     parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
#     parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")
#     parser.add_argument("--dim", type=int, default=64, help="dimension")
#     parser.add_argument("--med_sz", type=int, default=131, help="med size")
#     parser.add_argument("--n_heads", type=int, default=4, help="num of heads")
#     parser.add_argument("--H_k", type=int, default=5, help="H factorize recover")
#     parser.add_argument("--dropout", type=int, default=0.3, help="dropout ratio")
#     parser.add_argument("--cuda", type=int, default=0, help="which cuda")
#     return parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Test", action="store_true", default=False, help="test mode")
    parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
    parser.add_argument("--wandb", action="store_true", default=False, help="use wandb to log")
    parser.add_argument("--model_name", type=str, default=model_name, help="model name")
    parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
    parser.add_argument("--multi_weight", type=float, default=0.05, help="multi loss weight")
    parser.add_argument("--ddi_weight", type=float, default=0.5, help="ddi loss weight")
    parser.add_argument("--moe_weight", type=float, default=1, help="moe loss weight")
    parser.add_argument("--ssl_weight", type=float, default=0.01, help="ssl loss weight")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--bsz", type=int, default=16, help="batch size")
    parser.add_argument("--epoch", type=int, default=50, help="epoch")
    parser.add_argument("--eval_bsz", type=int, default=1, help="eval batch size")
    parser.add_argument("--win_sz", type=int, default=3, help="seq window size")
    parser.add_argument("--n_layers", type=int, default=2, help="num of gnn layers")
    parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
    parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")
    parser.add_argument("--dim", type=int, default=64, help="dimension")
    parser.add_argument("--med_sz", type=int, default=131, help="med size")
    parser.add_argument("--n_heads", type=int, default=4, help="num of heads")
    parser.add_argument("--n_protos", type=int, default=256, help="num of edge protos")
    parser.add_argument("--n_clusters", type=int, default=512, help="num of cluster")
    parser.add_argument("--n_experts", type=int, default=32, help="num of experts")
    parser.add_argument("--experts_k", type=int, default=4, help="num of experts used")
    parser.add_argument("--H_k", type=int, default=5, help="H factorize recover")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout ratio")
    parser.add_argument("--cuda", type=int, default=2, help="which cuda")
    parser.add_argument('--ddp', action='store_true', default=False, help='whether to use ddp')
    parser.add_argument('--name', type=str, default='', help='name to save model')
    return parser.parse_args()
