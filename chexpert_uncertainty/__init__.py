# -*- coding: utf-8 -*-
"""Shared code for the CheXpert pleural effusion uncertainty-label experiments.

Modules
-------
config   : load and validate the experiment YAML configuration.
data     : manifest loading, label strategies, dataset and transforms.
model    : DenseNet-121 with a single sigmoid output.
train    : deterministic training loop with logging, checkpoints, predictions.
metrics  : AUROC / AP / sensitivity / specificity, Youden's J threshold,
           study-level aggregation and curve data for figures.
"""
