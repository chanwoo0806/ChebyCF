import numpy as np

def recall(label_inter, relevance, k):
    right_pred = relevance[:, :k].sum(axis=1)
    recall_n = np.array([len(items) for items in label_inter])
    recall_n[recall_n == 0.] = 1.
    recall = (right_pred / recall_n).sum()
    return recall

def precision(relevance, k):
    right_pred = relevance[:, :k].sum(axis=1)
    precis_n = k
    precision = (right_pred / precis_n).sum()
    return precision

def mrr(relevance, k):
    pred_data = relevance[:, :k]
    scores = 1. / np.arange(1, k + 1)
    mrr = (pred_data * scores).sum()
    return mrr

def ndcg(label_inter, relevance, k):
    rel = relevance[:, :k]
    max_rel = np.zeros_like(rel)
    for i, items in enumerate(label_inter):
        length = min(len(items), k)
        max_rel[i, :length] = 1
    dcg =  (rel     * 1. / np.log2(np.arange(2, k + 2))).sum(axis=1)
    idcg = (max_rel * 1. / np.log2(np.arange(2, k + 2))).sum(axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = (dcg / idcg).sum()
    return ndcg