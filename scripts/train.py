# -*- coding: utf-8 -*-
"""Train one label strategy.

Usage (from the repository root):
  python scripts/train.py --config configs/experiment.yaml --strategy uzeroes
  python scripts/train.py --config configs/experiment.yaml --strategy uones
  python scripts/train.py --config configs/experiment.yaml --strategy uignore

All three runs share the same config (seed, budget, splits); only the strategy
flag differs. Artifacts land in outputs/<strategy>/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chexpert_uncertainty.config import load
from chexpert_uncertainty.train import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='path to experiment YAML')
    parser.add_argument('--strategy', choices=('uzeroes', 'uones', 'uignore'),
                        help='override the strategy in the config')
    args = parser.parse_args()

    cfg = load(args.config)
    if args.strategy:
        cfg['strategy'] = args.strategy
    run(cfg)


if __name__ == '__main__':
    main()
