# -*- coding: utf-8 -*-
"""Patient-level sampling and 80/20 split of the CheXpert training pool.

Usage:
  python scripts/sample_split.py \
      --pool data/manifests/train_pool.csv \
      --final data/manifests/final_eval.csv \
      --output-dir data/splits

Steps:
  1. Group the pool by patient. If it has more than --max-patients patients,
     keep a seeded random sample of exactly --max-patients patients.
  2. Split the sampled patients 80/20 (seed 6018). All images of one patient
     stay in the same split, so no patient appears in both.
  3. Re-check that no sampled patient appears in the final-eval manifest.
  4. Write train.csv / dev.csv (same columns as the pool) plus split.json.

The split is computed once and shared by all three label strategies; only the
meaning of -1 labels changes between strategies.
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd

FIELDS = ("image_path", "patient_id", "study_id", "pleural_effusion_raw", "split")
SEED = 6018


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', required=True, help='train_pool.csv from audit_chexpert.py')
    parser.add_argument('--final', required=True, help='final_eval.csv from audit_chexpert.py')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-patients', type=int, default=10000)
    parser.add_argument('--dev-frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pool = pd.read_csv(args.pool, keep_default_na=False)
    final = pd.read_csv(args.final, keep_default_na=False)
    rng = random.Random(args.seed)

    patients = sorted(pool['patient_id'].unique())
    if len(patients) > args.max_patients:
        patients = rng.sample(patients, args.max_patients)
    rng.shuffle(patients)
    n_dev = round(len(patients) * args.dev_frac)
    dev_patients = set(patients[:n_dev])
    train_patients = set(patients[n_dev:])

    pool['split'] = pool['patient_id'].map(
        lambda p: 'dev' if p in dev_patients else 'train')
    sampled = pool[pool['patient_id'].isin(set(patients))]

    final_patients = set(final['patient_id'].unique())
    overlap = set(patients) & final_patients
    if overlap:
        raise SystemExit(f'Patient overlap with final_eval: {sorted(overlap)[:10]} ...')

    for name, patient_set in (('train', train_patients), ('dev', dev_patients)):
        split_df = sampled[sampled['patient_id'].isin(patient_set)]
        split_df[list(FIELDS)].to_csv(out / f'{name}.csv', index=False)

    def dist(df):
        labels = df['pleural_effusion_raw'].value_counts().to_dict()
        return {k: labels.get(k, 0) for k in ('1', '0', '-1', '')}

    summary = {
        'seed': args.seed,
        'pool_images': len(pool),
        'pool_patients': len(pool['patient_id'].unique()),
        'sampled_patients': len(patients),
        'train_patients': len(train_patients),
        'dev_patients': len(dev_patients),
        'train_images': int((sampled['split'] == 'train').sum()),
        'dev_images': int((sampled['split'] == 'dev').sum()),
        'label_distribution_train': dist(sampled[sampled['split'] == 'train']),
        'label_distribution_dev': dist(sampled[sampled['split'] == 'dev']),
        'final_eval_patient_overlap': len(overlap),
    }
    (out / 'split.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
