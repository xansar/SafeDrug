# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: tmp.py
@time: 2023/6/13 14:36
@e-mail: xansar@ruc.edu.cn
"""
import dill
import pandas as pd

from nltk import word_tokenize,pos_tag   #分词、词性标注
from nltk.corpus import stopwords    #停用词
from nltk.stem import PorterStemmer    #词干提取
from nltk.stem import WordNetLemmatizer    #词性还原

interpunctuations = [',', ' ', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']   #定义符号列表

def tokenize_and_clean(text):
    text = text.lower()
    cutwords1 = word_tokenize(text)   #分词

    cutwords2 = [word for word in cutwords1 if word not in interpunctuations]   #去除标点符号

    stops = set(stopwords.words("english"))
    cutwords3 = [word for word in cutwords2 if word not in stops]  #判断分词在不在停用词列表内

    cutwords4 = []
    for cutword1 in cutwords3:
        cutwords4.append(PorterStemmer().stem(cutword1))    #词干提取

    cutwords5 = []
    for cutword2 in cutwords4:
        cutwords5.append(WordNetLemmatizer().lemmatize(cutword2))   #指定还原词性为名词

    return cutwords5

def tokenize(idx2desc_dict):
    res = {}
    word2id = {}
    id2word = {}
    word_cnt = 0
    for idx, desc in idx2desc_dict.items():
        res[idx] = tokenize_and_clean(desc)
        for word in res[idx]:
            if word not in word2id.keys():
                word2id[word] = word_cnt
                id2word[word_cnt] = word
                word_cnt += 1
    assert word_cnt == len(word2id.keys()) == len(id2word.keys())
    word_voc = {
        'word2id': word2id,
        'id2word': id2word,
        'cnt': word_cnt
    }
    return res, word_voc

def idx2icd_desc(icd_pth, voc):
    dtype = {'ICD9_CODE': str}
    icd2desc_df = pd.read_csv(icd_pth, dtype=dtype)
    icd2desc_df = icd2desc_df.drop(['ROW_ID'], axis=1)    # 删除row id,没用
    icd2desc_dict = {}
    for index, row in icd2desc_df.iterrows():
        icd_code = row['ICD9_CODE']
        short_title = row['SHORT_TITLE']
        long_title = row['LONG_TITLE']
        desc = short_title + '. ' + long_title
        if icd_code in voc.word2idx.keys():
            icd2desc_dict[icd_code] = desc
    # 存在漏的,也就是voc里面有的在这个df里找不到
    idx2desc_dict = {}
    for icd, idx in voc.word2idx.items():
        if icd not in icd2desc_dict.keys():
            continue
        idx2desc_dict[idx] = icd2desc_dict[icd]
    idx2desc_dict, word_voc = tokenize(idx2desc_dict)
    idx2word_id = {}
    for idx, desc in idx2desc_dict.items():
        idx2word_id[idx] = [word_voc['word2id'][word] for word in desc]
    return idx2word_id, word_voc

def generate_reverse_table(idx2desc_dict):
    desc2idx_dict = {}
    for idx, desc_lst in idx2desc_dict.items():
        for word in desc_lst:
            if word in desc2idx_dict.keys():
                desc2idx_dict[word].append(idx)
            else:
                desc2idx_dict[word] = [idx]
    return desc2idx_dict


def load_voc(pth):
    with open(pth, 'rb') as rf:
        vocs = dill.load(rf)
    diag_voc = vocs['diag_voc']
    proc_voc = vocs['pro_voc']
    med_voc = vocs['med_voc']
    return diag_voc, proc_voc, med_voc

if __name__ == '__main__':
    voc_pth = 'voc_final.pkl'
    diag_voc, proc_voc, med_voc = load_voc(voc_pth)

    diag_icd_pth = '~/data/physionet.org/files/mimiciii/1.4/D_ICD_DIAGNOSES.csv'
    proc_icd_pth = '~/data/physionet.org/files/mimiciii/1.4/D_ICD_PROCEDURES.csv'
    diag_idx2desc_dict, diag_word_voc = idx2icd_desc(diag_icd_pth, diag_voc)
    proc_idx2desc_dict, proc_word_voc = idx2icd_desc(proc_icd_pth, proc_voc)
    diag_desc2idx_dict = generate_reverse_table(diag_idx2desc_dict)
    proc_desc2idx_dict = generate_reverse_table(proc_idx2desc_dict)

    with open('desc_dict.pkl', 'wb') as wf:
        data = {
            'diag': {
                'id2desc': diag_idx2desc_dict,
                'desc2id': diag_desc2idx_dict,
                'voc': diag_word_voc
            },
            'proc': {
                'id2desc': proc_idx2desc_dict,
                'desc2id': proc_desc2idx_dict,
                'voc': proc_word_voc
            }
        }
        dill.dump(obj=data, file=wf)



