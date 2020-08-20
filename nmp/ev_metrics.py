#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation metrics."""
# import numpy as np
import pandas as pd
# from sklearn import metrics
from sklearn.metrics import roc_auc_score
from math import nan


def compute_auc(x, y):
    """Compute AUC-ROC for every timestep and return the list of AUCs.

    Averaged over pitches.

    x: true values
    y: score values

    Returns a dataframe of shape (88, n_ft)
    """
    n_ft = int(x.shape[1]/88)  # Number of future timesteps
    start = 0
    end = start + 88
    auc = []
    aucs = []

    for i in range(n_ft):
        auc = []
        for p in range(87, -1, -1):

            try:
                auc.append(roc_auc_score(x[:, start+p],  y[:, start+p]))

            except Exception:
                # There are no played notes.
                auc.append(nan)

            # auc.append(score)

            # fpr, tpr, thresholds = metrics.roc_curve(x[:, start+p],
            #                                          y[:, start+p],
            #                                          pos_label=1)

            # auc.append(metrics.auc(fpr, tpr))

        aucs.append(auc)
        start += 88  # Next predicted timestep
        end += 88  # Next predicted timestep

    auc_df = pd.DataFrame(aucs).transpose()
    auc_df.index = list(range(88, 0, -1))
    return auc_df
