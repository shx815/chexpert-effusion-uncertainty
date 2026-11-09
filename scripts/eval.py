# -*- coding: utf-8 -*-
"""Final evaluation after the three strategy runs have finished.

For every strategy in outputs/<strategy>/:
  1. Load dev_preds.csv and keep explicit 0/1 labels only.
  2. Choose the decision threshold that maximises Youden's J on the dev set
     (frozen before touching the final-eval set).
  3. Aggregate final-eval probabilities per patient+study (mean).
  4. Compute AUROC / AP / sensitivity / specificity at the frozen threshold.

Writes outputs/evaluation/metrics.json, comparison.csv and roc_pr_comparison.png.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chexpert_uncertainty.metrics import (explicit_mask, threshold_by_youden,
                                          binary_metrics, aggregate_studies, curves)

STRATEGIES = ('uzeroes', 'uones', 'uignore')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', default='outputs')
    parser.add_argument('--strategies', nargs='+', default=list(STRATEGIES))
    args = parser.parse_args()

    eval_dir = Path(args.outputs) / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)

    summary, curve_data = {}, {}
    for strategy in args.strategies:
        sdir = Path(args.outputs) / strategy
        dev = pd.read_csv(sdir / 'dev_preds.csv', keep_default_na=False,
                          dtype={'pleural_effusion_raw': str})
        final = pd.read_csv(sdir / 'eval_preds.csv', keep_default_na=False,
                            dtype={'pleural_effusion_raw': str})

        mask = explicit_mask(dev['pleural_effusion_raw'].to_numpy())
        y_dev = dev.loc[mask, 'pleural_effusion_raw'].map({'0': 0, '1': 1}).to_numpy()
        p_dev = dev.loc[mask, 'prob'].to_numpy()
        threshold, youden = threshold_by_youden(y_dev, p_dev)

        dev_metrics = binary_metrics(y_dev, p_dev, threshold=threshold)
        final = final[explicit_mask(final['pleural_effusion_raw'].to_numpy())]
        studies = aggregate_studies(final)
        y_final = studies['label'].map({'0': 0, '1': 1}).to_numpy()
        p_final = studies['prob'].to_numpy()
        final_metrics = binary_metrics(y_final, p_final, threshold=threshold)
        curve_data[strategy] = {'dev': curves(y_dev, p_dev),
                                'final': curves(y_final, p_final)}
        summary[strategy] = {
            'threshold': threshold,
            'youden_dev': youden,
            'dev': dev_metrics,
            'final_study_level': final_metrics,
        }
        (sdir / 'threshold.json').write_text(
            json.dumps({'threshold': threshold, 'youden_dev': youden}, indent=2) + '\n')
        print(f'[{strategy}] threshold={threshold:.4f} '
              f'final AUROC={final_metrics["auroc"]:.4f} AP={final_metrics["ap"]:.4f} '
              f'sens={final_metrics["sensitivity"]:.4f} spec={final_metrics["specificity"]:.4f}')

    (eval_dir / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')

    rows = []
    for strategy, s in summary.items():
        m = s['final_study_level']
        rows.append({'strategy': strategy, 'threshold': round(s['threshold'], 4),
                     'auroc': round(m['auroc'], 4), 'ap': round(m['ap'], 4),
                     'sensitivity': round(m['sensitivity'], 4),
                     'specificity': round(m['specificity'], 4),
                     'tp': m['tp'], 'fp': m['fp'], 'tn': m['tn'], 'fn': m['fn']})
    pd.DataFrame(rows).to_csv(eval_dir / 'comparison.csv', index=False)

    # --- figures: ROC + PR curves for the three strategies on the final eval set ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for strategy in args.strategies:
        c = curve_data[strategy]['final']
        axes[0].plot(c['fpr'], c['tpr'],
                     label=f"{strategy} (AUROC={summary[strategy]['final_study_level']['auroc']:.3f})")
        axes[1].plot(c['recall'], c['precision'],
                     label=f"{strategy} (AP={summary[strategy]['final_study_level']['ap']:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
    axes[0].set_xlabel('1 - Specificity'); axes[0].set_ylabel('Sensitivity')
    axes[0].set_title('Final evaluation ROC (study level)'); axes[0].legend(fontsize=9)
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Final evaluation PR (study level)'); axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(eval_dir / 'roc_pr_comparison.png', dpi=200)
    print(f'\nEvaluation artifacts in {eval_dir}/')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
