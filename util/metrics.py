import numpy as np
import torch
from sklearn import metrics
import math

def compute_auc(cls_scores, cls_labels):
    """Compute AUC of given prediction scores and ground truth."""

    cls_aucs = []
    for i in range(cls_scores.shape[1]):
        scores_per_class = cls_scores[:, i]
        labels_per_class = cls_labels[:, i]
        auc_per_class = metrics.roc_auc_score(labels_per_class,
                                              scores_per_class)
        # print('class {} auc = {:.2f}'.format(i + 1, auc_per_class * 100))

        cls_aucs.append(auc_per_class * 100)

    return cls_aucs

def compute_auprc(cls_scores, cls_labels):
    cls_aucs = []
    for i in range(cls_scores.shape[1]):
        scores_per_class = cls_scores[:, i]
        labels_per_class = cls_labels[:, i]
        # precision,  recall , _ = metrics.precision_recall_curve(labels_per_class, scores_per_class)
        # auc_per_class = metrics.auc(recall, precision)
        auc_per_class = metrics.average_precision_score(labels_per_class,
                                              scores_per_class)
        # print('class {} auc = {:.2f}'.format(i + 1, auc_per_class * 100))
        cls_aucs.append(auc_per_class * 100)
    return cls_aucs

def cal_metrics(cls_scores, gt_labels):
    cls_auc = compute_auc(cls_scores, gt_labels)
    mean_auc = np.mean(cls_auc)
    cls_aupr = compute_auprc(cls_scores, gt_labels)
    mean_aupr = np.mean(cls_aupr)
    return dict(cls_auc=cls_auc, mauc=mean_auc, cls_aupr=cls_aupr, maupr=mean_aupr)

