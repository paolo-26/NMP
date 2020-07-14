#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation metrics."""
import numpy as np
from sklearn import metrics


def compute_auc(x, y):
    """Compute AUC-ROC for every timestep and return the list of AUCs."""
    n_ft = int(x.shape[1]/88)  # Number of future timesteps
    start = 0
    end = start + 88
    aucs = []  # List of AUC values; one value per timestep

    for o in range(n_ft):  # For every predicted (future) timestep
        L = len(x)  # Number of instances
        auc = []  # List of AUCs values for all instances.
        for i in range(L):

            fpr, tpr, thresholds = metrics.roc_curve(x[i, start:end],
                                                     y[i, start:end],
                                                     pos_label=1)

            # Append AUC value for specific instance.
            auc.append(metrics.auc(fpr, tpr))

        aucs.append(np.nanmean(auc))  # Average AUC for current timestep

        start += 88  # Next predicted timestep
        end += 88  # Next predicted timestep

    return aucs
