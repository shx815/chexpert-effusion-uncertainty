# -*- coding: utf-8 -*-
"""Experiment configuration: single source of truth for every run.

All hyperparameters live in a YAML file (see configs/experiment.yaml) so that
the three label strategies share one training budget, one seed and one split.
"""
import yaml


DEFAULTS = {
    'data': {
        'image_root': None,
        'train_manifest': None,
        'dev_manifest': None,
        'final_eval_manifest': None,
        'image_size': 224,
        'num_workers': 2,
    },
    'strategy': 'uzeroes',          # uzeroes | uones | uignore
    'model': 'densenet121',
    'training': {
        'seed': 6018,
        'epochs': 10,
        'batch_size': 32,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'early_stopping_patience': 4,
        'augment': True,
    },
    'outputs': 'outputs',
}

STRATEGIES = ('uzeroes', 'uones', 'uignore')


def load(path):
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
