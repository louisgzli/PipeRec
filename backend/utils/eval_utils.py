import csv
import re
from typing import List

import scipy.stats
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_auc_score, mean_squared_error


def split_by_type(text):
    regex = r"[\u4e00-\u9fa5]+|[0-9]+|[a-zA-Z]+|[ ]+|[.#]+"
    return re.findall(regex, text, re.UNICODE)


def clean_text(text):
    return ''.join(split_by_type(' '.join(text.lower().split())))


def load_id_name_map(file_name):
    df = pd.read_csv(file_name, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8', on_bad_lines='skip',
                     converters={i: str for i in range(100)})
    column_id, column_name = df.columns[0:2]
    df[column_name] = df[column_name].apply(clean_text)
    return df.set_index(column_id)[column_name].to_dict()


def calculate_auc(label, predict):
    fpr, tpr, threshold = metrics.roc_curve(label, predict)
    return metrics.auc(fpr, tpr)


def calculate_aupr(label, predict):
    precision, recall, threshold = metrics.precision_recall_curve(label, predict)
    return metrics.auc(recall, precision)


def calculate_pr_at_precision(label, predict, min_precision):
    precision, recall, threshold = metrics.precision_recall_curve(label, predict)
    satisfied_metric = []
    for th, p, r in zip(threshold, precision, recall):
        if p >= min_precision:
            satisfied_metric.append([th, p, r])
    if len(satisfied_metric) == 0:
        # the last precision and the recall value should be ignored.
        # The last precision and recall values are always 1. and 0. respectively
        # and do not have a corresponding threshold.
        max_precision_index = precision.tolist().index(max(precision.tolist()[:-1]))
        return threshold[max_precision_index], precision[max_precision_index], 'no_recall'
    else:
        return sorted(satisfied_metric, key=(lambda x: x[2]), reverse=True)[0]


def calculate_metrics(eval_df, min_precision):
    label = eval_df['label'].tolist()
    score = eval_df['score'].tolist()
    auc_score = calculate_auc(label, score)
    aupr = calculate_aupr(label, score)
    threshold, precision, recall = calculate_pr_at_precision(label, score, min_precision)
    return auc_score, aupr, threshold, precision, recall


def evaluate_recall_at_precision(eval_df, sample_items, min_precision=0.6):
    total_auc, total_aupr, total_threshold, total_precision, total_recall = calculate_metrics(eval_df, min_precision)
    items_auc, items_aupr, items_threshold, items_precision, items_recall = calculate_metrics(
        eval_df[eval_df['item'].isin(sample_items)], min_precision
    )
    return (
        total_auc, total_aupr, total_threshold, total_precision, total_recall,
        items_auc, items_aupr, items_threshold, items_precision, items_recall
    )


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def compute_hit_score(rec_id: ndarray, label_id: ndarray, topk_list: List[int], weights: ndarray) -> List[float]:
    """命中率
    :param rec_id: 推荐矩阵，一般维度是 [n, max(topk_list)]
    :param label_id: 真实点击array，一维数组，维度是 (n, )
    :param topk_list:
    :param weights 每条样本的加权数组
    """
    assert rec_id.shape[0] == label_id.shape[0], "Sample nums does not match!"
    res = []
    for k in topk_list:
        click_metrics = np.array(rec_id[:, :k] == label_id[:, None]).astype(int)
        res.append((click_metrics.sum(1) * weights).sum() / (weights.sum() + 1e-12))
    return res


def compute_mrr_score(rec_id: ndarray, label_id: ndarray, topk_list: List[int], weights: ndarray) -> List[float]:
    """mrr
    :param rec_id: 推荐矩阵，一般维度是 [n, max(topk_list)]
    :param label_id: 真实点击array，一维数组，维度是 (n, )
    :param topk_list:
    :param weights 每条样本的加权数组
    """
    assert rec_id.shape[0] == label_id.shape[0], "Sample nums does not match!"
    res = []
    for k in topk_list:
        click_metrics = np.array(rec_id[:, :k] == label_id[:, None]).astype(int) / np.arange(1, k+1)
        res.append((click_metrics.sum(1) * weights).sum() / (weights.sum() + 1e-12))
    return res


def compute_ndcg_score(rec_id: ndarray, label_id: ndarray, topk_list: List[int], weights: ndarray) -> List[float]:
    """ndcg
    :param rec_id: 推荐矩阵，一般维度是 [n, max(topk_list)]
    :param label_id: 真实点击array，一维数组，维度是 (n, )
    :param topk_list:
    :param weights 每条样本的加权数组
    """
    assert rec_id.shape[0] == label_id.shape[0], "Sample nums does not match!"
    res = []
    for k in topk_list:
        i_dcgs = np.log2(np.arange(k) + 2)
        dcg = (np.array(rec_id[:, :k] == label_id[:, None]).astype(int) / i_dcgs).sum(1)
        idcg = (-np.sort(-np.array(rec_id[:, :k] == label_id[:, None]).astype(int)) / i_dcgs).sum(1)
        res.append(((dcg / (idcg + 1e-12)) * weights).sum() / (weights.sum() + 1e-12))
    return res


def recall_at_precision(y_true, y_score, min_precision=0.65):
    """固定 精度 求最大召回
    """
    p, r, th = precision_recall_curve(y_true, y_score)

    satisfied_metric = []
    # precision_recall_curve返回的threshold比pr少一维，这里填上对齐
    th = np.insert(th, 0, min(y_score))
    for th, p, r in zip(th, p, r):
        if p >= min_precision:
            satisfied_metric.append([th, p, r])

    if satisfied_metric:  # 线上要求: p>=0.8 and max(r)
        th_best, p_best, r_best = sorted(satisfied_metric, key=(lambda x: x[2]), reverse=True)[0]
    else:  # 如果没有满足线上要求的，最取precision最大值
        p_best = max(p)
        p_best_idx = p.tolist().index(p_best)
        r_best = r[p_best_idx]
        if p_best_idx >= len(th):
            th_best = th[-1]
        else:
            th_best = 0

    return r_best, p_best, th_best


def auc(y_true, y_pred):
    return roc_auc_score(y_true=y_true, y_score=y_pred)


def mse(y_true, y_pred):
    return mean_squared_error(y_true=y_true, y_pred=y_pred)
