# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: HyperDrugRec.py
@time: 2023/6/8 14:07
@e-mail: xansar@ruc.edu.cn
"""
import dill
import numpy as np
from collections import defaultdict
from torch.optim import Adam
import os
import torch
import time
import wandb


from layers.model import HyperDrugRec
from graph_construction import construct_graphs, graph2hypergraph
from util import llprint, multi_label_metric, ddi_rate_score, buildMPNN, replace_with_padding_woken, multihot2idx
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import collate_fn
from config import parse_args


torch.manual_seed(1203)
np.random.seed(2048)
# torch.set_num_threads(30)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# wandb
def init_wandb(args):
    # start a new wandb run to track this script
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="MLHC",
            group=args.model_name,

            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "architecture": args.model_name,
                "dataset": "MIMIC-III",
                "epochs": 50,
                "batch_size": args.bsz,
                "emb_dim": args.dim,
                "seed": 1203,
            },

            name=f'{args.model_name}_lr_{args.lr}',

            # dir
            dir='./saved'
        )


# evaluate
def eval(model, eval_data_loader, voc_size, epoch, padding_dict):
    model.eval()

    # smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1, avg_ddi = [[] for _ in range(6)]
    med_cnt, visit_cnt = 0, 0

    for step, (records, masks, targets) in enumerate(eval_data_loader):
        records = {k: replace_with_padding_woken(v, -1, padding_dict[k]).to(model.device) for k, v in records.items()}
        masks = {k: v.to(model.device) for k, v in masks.items()}
        targets = {k: v.to(model.device) for k, v in targets.items()}


        # y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        bsz, max_visit, med_size = targets['loss_bce_target'].shape

        padding_mask = (masks['key_padding_mask'] == False).reshape(bsz * max_visit).detach().cpu().numpy()
        true_visit_idx = np.where(padding_mask == True)[0]
        target_output, _ = model(records, masks)

        y_gt = targets['loss_bce_target'].detach().cpu().numpy()

        y_pred_prob = F.sigmoid(target_output).detach().cpu().numpy()

        y_pred = y_pred_prob.copy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        y_pred_label = multihot2idx(y_pred)

        y_gt = y_gt.reshape(bsz * max_visit, -1)[true_visit_idx]
        y_pred_prob = y_pred_prob.reshape(bsz * max_visit, -1)[true_visit_idx]
        y_pred = y_pred.reshape(bsz * max_visit, -1)[true_visit_idx]
        y_pred_label = [y_pred_label[n] for n in true_visit_idx.tolist()]
        # for adm_idx, adm in enumerate(input):
        #     # adm: diag, pro, med
        #     target_output, _ = model(input[: adm_idx + 1])
        #
        #     y_gt_tmp = np.zeros(voc_size[2])
        #     y_gt_tmp[adm[2]] = 1
        #     y_gt.append(y_gt_tmp)
        #
        #     # prediction prod
        #     target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
        #     y_pred_prob.append(target_output)
        #
        #     # prediction med set
        #     y_pred_tmp = target_output.copy()
        #     y_pred_tmp[y_pred_tmp >= 0.5] = 1
        #     y_pred_tmp[y_pred_tmp < 0.5] = 0
        #     y_pred.append(y_pred_tmp)
        #
        #     # prediction label
        #     y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
        #     y_pred_label.append(sorted(y_pred_label_tmp))
        #     visit_cnt += 1
        #     med_cnt += len(y_pred_label_tmp)

        # smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        # ddi rate
        ddi_rate = ddi_rate_score([y_pred_label], path="../old_data/output/ddi_A_final.pkl")
        avg_ddi.append(ddi_rate)

        med_cnt += y_pred.sum()
        visit_cnt += len(y_pred_label)

        llprint("\rtest step: {} / {}".format(step, len(eval_data_loader)))



    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            np.mean(avg_ddi),
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )
    )

    return (
        np.mean(avg_ddi),
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )


def main():
    args = parse_args()

    if not args.Test:
        init_wandb(args)

    if not os.path.exists(os.path.join("saved", args.model_name)):
        os.makedirs(os.path.join("saved", args.model_name))
    # load old_data
    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'

    ehr_adj_path = '../data/ehr_adj_final.pkl'
    ddi_adj_path = '../data/ddi_A_final.pkl'
    ddi_mask_path = '../data/ddi_mask_H.pkl'
    molecule_path = '../data/idx2drug.pkl'
    cache_path = "../data/graphs.pkl"

    device = torch.device("cuda:{}".format(args.cuda))

    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    molecule = dill.load(open(molecule_path, "rb"))

    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    # 这里数据划分好像有点问题，不是按照每个病人划分的，也没有shuffle
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]

    if args.debug:
        data_train = data_train[:100]
        data_eval = data_eval[:100]

    train_dataloader = DataLoader(data_train, batch_size=args.bsz, collate_fn=collate_fn,
                                  shuffle=True, pin_memory=True)
    eval_dataloader = DataLoader(data_eval, batch_size=args.bsz, collate_fn=collate_fn, shuffle=False,
                                 pin_memory=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=False,
    #                              pin_memory=True)

    # MPNNSet, N_fingerprint, average_projection = buildMPNN(
    #     molecule, med_voc.idx2word, 2, device
    # )
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    voc_size_dict = {
        'diag': voc_size[0],
        'proc': voc_size[1],
        'med': voc_size[2]
    }
    adj_dict = construct_graphs(cache_path, data_train, nums_dict=voc_size_dict, k=args.H_k)

    adj_dict['ddi_adj'] = torch.FloatTensor(ddi_adj)

    h_ddi_adj = graph2hypergraph(adj_dict['ddi_adj'])  # 超图
    # 将ddi超图和ehr超图结合起来
    adj_dict['med'] = torch.cat([adj_dict['med'], h_ddi_adj], dim=-1).coalesce()

    padding_dict = {
        'diag': voc_size_dict['diag'],
        'proc': voc_size_dict['proc'],
        'med': voc_size_dict['med'],
    }
    # model = SafeDrugModel(
    #     voc_size,
    #     ddi_adj,
    #     ddi_mask_H,
    #     MPNNSet,
    #     N_fingerprint,
    #     average_projection,
    #     emb_dim=args.dim,
    #     device=device,
    # )
    model = HyperDrugRec(
        voc_size_dict=voc_size_dict,
        adj_dict=adj_dict,
        padding_dict=padding_dict,
        embedding_dim=args.dim,
        n_heads=args.n_heads,
        dropout=args.dropout,
        device=device
    )
    # model.load_state_dict(torch.load(open(args.resume_path, 'rb')))

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, "rb")))
        model.to(device=device)
        tic = time.time()

        ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
        # ###
        # for threshold in np.linspace(0.00, 0.20, 30):
        #     print ('threshold = {}'.format(threshold))
        #     ddi, ja, prauc, _, _, f1, avg_med = eval(model, data_test, voc_size, 0, threshold)
        #     ddi_list.append(ddi)
        #     ja_list.append(ja)
        #     prauc_list.append(prauc)
        #     f1_list.append(f1)
        #     med_list.append(avg_med)
        # total = [ddi_list, ja_list, prauc_list, f1_list, med_list]
        # with open('ablation_ddi.pkl', 'wb') as infile:
        #     dill.dump(total, infile)
        # ###
        result = []
        for _ in range(10):
            test_sample = np.random.choice(
                data_test, round(len(data_test) * 0.8), replace=True
            )
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
                model, test_sample, voc_size, 0
            )
            result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print(outstring)

        print("test time: {}".format(time.time() - tic))
        return

    model.to(device=device)
    # print('parameters', get_n_params(model))
    # exit()
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 50
    for epoch in range(EPOCH):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch + 1))

        model.train()
        bce_loss_lst, multi_loss_lst, ddi_loss_lst, total_loss_lst = [[] for _ in range(4)]
        for step, (records, masks, targets) in enumerate(train_dataloader):
            # 做点yu
            records = {k: replace_with_padding_woken(v, -1, padding_dict[k]).to(device) for k, v in records.items()}
            masks = {k: v.to(device) for k, v in masks.items()}
            targets = {k: v.to(device) for k, v in targets.items()}

            result, loss_ddi = model(records, masks)
            bce_loss_mask = (masks['key_padding_mask'] == False).unsqueeze(-1).repeat(1, 1, voc_size_dict['med'])

            loss_bce = (F.binary_cross_entropy_with_logits(
                result, targets['loss_bce_target'], reduction='none'
            ) * bce_loss_mask).mean()

            bsz, max_visit, _ = result.shape
            multi_loss_mask = (masks['key_padding_mask'] == False).reshape(bsz * max_visit, -1)
            loss_multi = (F.multilabel_margin_loss(
                F.sigmoid(result).reshape(bsz * max_visit, -1), targets['loss_multi_target'].reshape(bsz * max_visit, -1), reduction='none'
            ) * multi_loss_mask).mean()

            result = F.sigmoid(result).detach().cpu().numpy()
            result[result >= 0.5] = 1
            result[result < 0.5] = 0
            y_label = multihot2idx(result)
            current_ddi_rate = ddi_rate_score(
                [y_label], path="../old_data/output/ddi_A_final.pkl"
            )



            if current_ddi_rate <= args.target_ddi:
                loss = 0.95 * loss_bce + 0.05 * loss_multi
            else:
                beta = min(0, 1 + (args.target_ddi - current_ddi_rate) / args.kp)
                loss = (
                    beta * (0.95 * loss_bce + 0.05 * loss_multi)
                    + (1 - beta) * loss_ddi
                )

            bce_loss_lst.append(loss_bce.item())
            multi_loss_lst.append(loss_multi.item())
            ddi_loss_lst.append(loss_ddi.item())
            total_loss_lst.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            llprint("\rtraining step: {} / {}".format(step, len(train_dataloader)))

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
            model, eval_dataloader, voc_size, epoch, padding_dict
        )
        print(
            "training time: {}, test time: {}".format(
                time.time() - tic, time.time() - tic2
            )
        )


        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)

        if args.wandb:
            wandb.log({
                'ja': ja,
                'ddi_rate': ddi_rate,
                'avg_p': avg_p,
                'avg_r': avg_r,
                'avg_f1': avg_f1,
                'prauc': prauc,
                'med': avg_med,
                'bce_loss': np.mean(bce_loss_lst),
                'multi_loss': np.mean(multi_loss_lst),
                'ddi_loss': np.mean(ddi_loss_lst),
                'total_loss': np.mean(total_loss_lst),
            })

        if epoch >= 5:
            print(
                "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                    np.mean(history["ddi_rate"][-5:]),
                    np.mean(history["med"][-5:]),
                    np.mean(history["ja"][-5:]),
                    np.mean(history["avg_f1"][-5:]),
                    np.mean(history["prauc"][-5:]),
                )
            )

        torch.save(
            model.state_dict(),
            open(
                os.path.join(
                    "saved",
                    args.model_name,
                    "Epoch_{}_TARGET_{:.2}_JA_{:.4}_DDI_{:.4}.model".format(
                        epoch, args.target_ddi, ja, ddi_rate
                    ),
                ),
                "wb",
            ),
        )

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print("best_epoch: {}".format(best_epoch))

    if args.wandb:
        wandb.finish()

    dill.dump(
        history,
        open(
            os.path.join(
                "saved", args.model_name, "history_{}.pkl".format(args.model_name)
            ),
            "wb",
        ),
    )


if __name__ == "__main__":
    main()




