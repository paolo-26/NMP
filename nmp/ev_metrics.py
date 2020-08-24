#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation metrics."""
# import numpy as np
import pandas as pd
# from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score
from math import nan
from nmp.dataset import threshold
import numpy as np


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

        aucs.append(auc)
        start += 88  # Next predicted timestep
        end += 88  # Next predicted timestep

    auc_df = pd.DataFrame(aucs).transpose()
    auc_df.index = list(range(88, 0, -1))
    return auc_df


def compute_f1(x, y):
    """Compute F1-score for every timestep and return the list of F1 scores.

    Averaged over pitches.

    x: true values
    y: score values

    Returns a dataframe of shape (88, n_ft)
    """
    n_ft = int(x.shape[1]/88)  # Number of future timesteps
    start = 0
    end = start + 88
    f1 = []
    f1s = []

    for i in range(n_ft):
        f1 = []
        for p in range(87, -1, -1):

            try:
                f1.append(f1_score(x[:, start+p],  y[:, start+p]))

            except Exception:
                # There are no played notes.
                f1.append(nan)

        f1s.append(f1)
        start += 88  # Next predicted timestep
        end += 88  # Next predicted timestep

    f1_df = pd.DataFrame(f1s).transpose()
    f1_df.index = list(range(88, 0, -1))
    return f1_df


def compute_best_thresh(x, y):
    """Compute best threshold."""

    best_thresh = None
    best_res = 0
    for thresh in np.arange(0.01, 0.3, 0.05):
        pred_f1 = compute_f1(x, threshold(y, thresh))
        res = np.nanmean(pred_f1)
        if res > best_res:
            best_thresh = thresh
            best_res = res

    return (best_thresh, best_res)
