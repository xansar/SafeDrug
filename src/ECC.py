import dill
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from collections import defaultdict
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import jaccard_score
from sklearn import tree
import os
import time

import sys
sys.path.append('..')
from util import multi_label_metric

model_name = 'ECC'

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

def augment(y_pred, appear_idx):
    m, n = y_pred.shape
    y_pred_aug = np.zeros((m, output_len))
    y_pred_aug[:, appear_idx] = y_pred

    return y_pred_aug
def create_dataset(data, diag_voc, pro_voc, med_voc):
    i1_len = len(diag_voc.idx2word)
    i2_len = len(pro_voc.idx2word)
    global output_len
    output_len = len(med_voc.idx2word)
    input_len = i1_len + i2_len
    X = []
    y = []
    for patient in data:
        for visit in patient:
            i1 = visit[0]
            i2 = visit[1]
            o = visit[2]

            multi_hot_input = np.zeros(input_len)
            multi_hot_input[i1] = 1
            multi_hot_input[np.array(i2) + i1_len] = 1

            multi_hot_output = np.zeros(output_len)
            multi_hot_output[o] = 1

            X.append(multi_hot_input)
            y.append(multi_hot_output)

    return np.array(X), np.array(y)
# def create_dataset(data, diag_voc, pro_voc, med_voc):
#     i1_len = len(diag_voc.idx2word)
#     i2_len = len(pro_voc.idx2word)
#     output_len = len(med_voc.idx2word)
#     input_len = i1_len + i2_len
#     X = []
#     y = []
#     for patient in data:
#         for visit in patient:
#             i1 = visit[0]
#             i2 = visit[1]
#             o = visit[2]
#
#             multi_hot_input = np.zeros(input_len)
#             multi_hot_input[i1] = 1
#             multi_hot_input[np.array(i2) + i1_len] = 1
#
#             multi_hot_output = np.zeros(output_len)
#             multi_hot_output[o] = 1
#
#             X.append(multi_hot_input)
#             y.append(multi_hot_output)
#
#     return np.array(X), np.array(y)


def main():
    # grid_search = False
    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    epoch = 100

    np.random.seed(epoch)
    np.random.shuffle(data)
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point+eval_len:]
    data_test = data[split_point:split_point + eval_len]

    train_X, train_y = create_dataset(data_train, diag_voc, pro_voc, med_voc)
    train_X = train_X
    train_y = train_y
    test_X, test_y = create_dataset(data_test, diag_voc, pro_voc, med_voc)
    eval_X, eval_y = create_dataset(data_eval, diag_voc, pro_voc, med_voc)

    """
    some drugs do not appear in the train set (their index is non_appear_idx)
    we omit them during training ECC (resulting in appear_idx)
    and directly not recommend these for test and eval
    """
    # non_appear_idx = np.where(train_y.sum(axis=0) == 0)[0]
    appear_idx = np.where(train_y.sum(axis=0) > 0)[0]
    train_y = train_y[:, appear_idx]

    # confirmed_index = np.where(train_y.sum(axis=0) == 0)[0]
    # predicted_index = np.where(train_y.sum(axis=0) > 0)[0]
    # train_X, train_y = train_X[predicted_index], train_y[predicted_index]

    base_dt = LogisticRegression()

    tic_total_fit = time.time()
    global chains
    chains = [ClassifierChain(base_dt, order='random', random_state=i) for i in range(10)]
    for i, chain in enumerate(chains):
        tic = time.time()
        chain.fit(train_X, train_y)
        fittime = time.time() - tic
        print ('id {}, fitting time: {}'.format(i, fittime))
    print ('total fitting time: {}'.format(time.time() - tic_total_fit))

    # exit()
    result = []
    for _ in range(10):
        index = np.random.choice(np.arange(len(test_X)), round(len(test_X) * 0.8), replace=True)
        test_sample = test_X[index]
        y_sample = test_y[index]

        tic = time.time()
        y_pred_chains = np.array([augment(chain.predict(test_sample), appear_idx) for chain in chains])
        y_prob_chains = np.array([augment(chain.predict_proba(test_sample), appear_idx) for chain in chains])
        pretime = time.time() - tic
        print('inference time: {}'.format(pretime))

        y_pred = y_pred_chains.mean(axis=0)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_prob = y_prob_chains.mean(axis=0)

        ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_sample, y_pred, y_prob)

        # ddi rate
        ddi_A = dill.load(open('../data/ddi_A_final.pkl', 'rb'))
        all_cnt = 0
        dd_cnt = 0
        med_cnt = 0
        visit_cnt = 0
        for adm in y_pred:
            med_code_set = np.where(adm == 1)[0]
            visit_cnt += 1
            med_cnt += len(med_code_set)
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
        ddi_rate = dd_cnt / all_cnt
        result.append([ddi_rate, ja, avg_f1, prauc, med_cnt / visit_cnt])
    result = np.array(result)
    mean = result.mean(axis=0)
    std = result.std(axis=0)

    outstring = ""
    for m, s in zip(mean, std):
        outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

    print(outstring)
    #
    # tic = time.time()
    # print(
    #     'Epoch: {}, DDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
    #         epoch, ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, med_cnt / visit_cnt
    #     ))
    #
    # history = defaultdict(list)
    # history['fittime'].append(fittime)
    # history['pretime'].append(pretime)
    # history['jaccard'].append(ja)
    # history['ddi_rate'].append(ddi_rate)
    # history['avg_p'].append(avg_p)
    # history['avg_r'].append(avg_r)
    # history['avg_f1'].append(avg_f1)
    # history['prauc'].append(prauc)
    #
    # dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))
    # print('Avg_Fittime: {:.8}, Avg_Pretime: {:.8}, Avg_Jaccard: {:.4}, Avg_DDI: {:.4}, Avg_p: {:.4}, Avg_r: {:.4}, \
    #         Avg_f1: {:.4}, AVG_PRC: {:.4}\n'.format(
    #     np.mean(history['fittime']),
    #     np.mean(history['pretime']),
    #     np.mean(history['jaccard']),
    #     np.mean(history['ddi_rate']),
    #     np.mean(history['avg_p']),
    #     np.mean(history['avg_r']),
    #     np.mean(history['avg_f1']),
    #     np.mean(history['prauc'])
    #     ))

    # tic = time.time()
    # y_pred_chains = np.array([augment(chain.predict(test_X), appear_idx) for chain in chains])
    # y_prob_chains = np.array([augment(chain.predict_proba(test_X), appear_idx) for chain in chains])
    # pretime = time.time() - tic
    # print ('inference time: {}'.format(pretime))
    #
    # y_pred = y_pred_chains.mean(axis=0)
    # y_pred[y_pred >= 0.5] = 1
    # y_pred[y_pred < 0.5] = 0
    # y_prob = y_prob_chains.mean(axis=0)
    #
    # ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(test_y, y_pred, y_prob)
    #
    # # ddi rate
    # ddi_A = dill.load(open('../data/ddi_A_final.pkl', 'rb'))
    # all_cnt = 0
    # dd_cnt = 0
    # med_cnt = 0
    # visit_cnt = 0
    # for adm in y_pred:
    #     med_code_set = np.where(adm==1)[0]
    #     visit_cnt += 1
    #     med_cnt += len(med_code_set)
    #     for i, med_i in enumerate(med_code_set):
    #         for j, med_j in enumerate(med_code_set):
    #             if j <= i:
    #                 continue
    #             all_cnt += 1
    #             if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
    #                 dd_cnt += 1
    # ddi_rate = dd_cnt / all_cnt
    # print('Epoch: {}, DDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
    #     epoch, ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, med_cnt / visit_cnt
    #     ))

if __name__ == '__main__':
    main()   
