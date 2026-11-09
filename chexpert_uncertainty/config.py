# -*- coding: utf-8 -*-
"""Experiment configuration: single source of truth for every run.

The three label strategies share one training budget, one seed and one split;
only the strategy flag differs. All hyperparameters live in a YAML file (see
configs/experiment.yaml), which makes the runs reproducible and comparable.
"""
import yaml


DEFAULTS = {
    'data': {
        'image_root': None,            # folder containing the image files
        'train_manifest': None,        # CSV with training rows (image_path, patient_id, study_id, label)
        'dev_manifest': None,          # CSV with dev rows, used for checkpoint and threshold selection
        'final_eval_manifest': None,   # official expert-labelled set, used only for final evaluation
        'image_size': 224,             # square input resolution after aspect-preserving resize + padding
        'num_workers': 2,              # DataLoader workers
    },
    'strategy': 'uzeroes',             # uzeroes | uones | uignore
    'model': 'densenet121',            # ImageNet-pretrained DenseNet-121, single sigmoid output
    'training': {
        'seed': 6018,                  # fixed seed shared by all three strategies (patient split + RNG)
        'epochs': 10,                  # maximum epochs; early stopping may end the run sooner
        'batch_size': 32,
        'lr': 1e-4,                    # Adam learning rate
        'weight_decay': 1e-5,
        'early_stopping_patience': 4,  # stop when dev AUROC fails to improve for this many epochs
        'augment': True,               # random horizontal flip during training only
    },
    'outputs': 'outputs',              # artifacts land in <outputs>/<strategy>/
}

STRATEGIES = ('uzeroes', 'uones', 'uignore')


def load(path):
    """Load a YAML config and fill missing keys with DEFAULTS.

    Raises ValueError if the strategy is unknown or a required data key
    (image_root, train_manifest) is missing.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for section, values in DEFAULTS.items():
        cfg.setdefault(section, values)
    for key, value in DEFAULTS['data'].items():
        cfg['data'].setdefault(key, value)
    for key, value in DEFAULTS['training'].items():
        cfg['training'].setdefault(key, value)
    if cfg['strategy'] not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {cfg['strategy']!r}")
    missing = [k for k in ('image_root', 'train_manifest') if not cfg['data'].get(k)]
    if missing:
        raise ValueError(f'config missing required data keys: {missing}')
    return cfg
