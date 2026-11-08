# -*- coding: utf-8 -*-
"""Evaluation metrics shared by training and final evaluation.

Dev metrics always use only explicit 0/1 labels, consistently across the three
strategies. Final evaluation aggregates per-study (mean probability over the
frontal images of a study) before computing study-level metrics.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve, confusion_matrix)


def explicit_mask(raw_labels):
    """Rows whose raw label is an explicit 0 or 1 (used for dev metrics)."""
    return np.isin(raw_labels, ('0', '1'))


def threshold_by_youden(y_true, probs):
    """Decision threshold maximising Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    best = int(np.argmax(j))
    return float(thresholds[best]), float(j[best])


def binary_metrics(y_true, probs, threshold=0.5):
    """ROC-AUC, AP, and sensitivity / specificity / confusion matrix at a threshold."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        'auroc': float(roc_auc_score(y_true, probs)),
        'ap': float(average_precision_score(y_true, probs)),
        'sensitivity': float(tp / (tp + fn)) if tp + fn else float('nan'),
        'specificity': float(tn / (tn + fp)) if tn + fp else float('nan'),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'n': int(len(y_true)),
    }


def aggregate_studies(pred_df):
    """Average probabilities over the frontal images of each patient+study."""
    return (pred_df.groupby(['patient_id', 'study_id'])
                   .agg(prob=('prob', 'mean'),
                        label=('pleural_effusion_raw', 'first'))
                   .reset_index())


def curves(y_true, probs):
    """ROC and PR curve coordinates for figures."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    return {'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
            'precision': prec.tolist(), 'recall': rec.tolist()}


def metrics_table(metrics_dict):
    """Pretty text table of one strategy's metrics."""
    keys = [('auroc', 'AUROC'), ('ap', 'AP'), ('sensitivity', 'Sensitivity'),
            ('specificity', 'Specificity')]
    lines = []
    for key, name in keys:
        lines.append(f'{name:<14}{metrics_dict[key]:.4f}')
    cm = metrics_dict
    lines.append(f'Confusion     TN={cm["tn"]} FP={cm["fp"]} FN={cm["fn"]} TP={cm["tp"]}')
    return '\n'.join(lines)
