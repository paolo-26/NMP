#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation metrics."""
import numpy as np
from sklearn import metrics


def compute_auc(x, y):
    """Compute ROC - AUC for every timestep and return the list of AUCs."""
    # print(x.shape)
    # print(y.shape)
    iterations = int(x.shape[1]/88)
    start = 0
    end = start + 88
    aucs = []
    f1s = []
    # cnt = 0
    # cnt2 = 0
    for o in range(iterations):
        L = len(x)
        auc = []
        f1 = []
        for i in range(L):

            # if all(v == 0 for v in x[i, start:end]):
            #     cnt += 1
            #
            # if all(v == 0 for v in y[i, start:end]):
            #     cnt2 += 1
            fpr, tpr, thresholds = metrics.roc_curve(x[i, start:end],
                                                     y[i, start:end],
                                                     pos_label=1)
            auc.append(metrics.auc(fpr, tpr))

            f1.append(metrics.f1_score(y[i, start:end],
                                       [int(x) for x in x[i, start:end]],
                                       pos_label=1))

        start += 88
        end += 88
        # print(f1)
        aucs.append(np.nanmean(auc))
        f1s.append(np.nanmean(f1))

    # print(cnt)
    # print(cnt2)
    return (aucs, f1s)
