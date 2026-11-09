# -*- coding: utf-8 -*-
"""Deterministic training loop with logging, checkpoints and predictions.

Artifacts per strategy, written to outputs/<strategy>/ :
  log.csv        one row per epoch: losses, dev AUROC/AP, learning rate
  best.pt        checkpoint with the highest dev AUROC
  last.pt        checkpoint after the final epoch
  run.json       configuration snapshot, sample counts, runtime, device
  dev_preds.csv  probabilities for every served dev row (metrics use explicit labels only)
  eval_preds.csv probabilities for the official final-evaluation rows
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import CheXpertDataset
from .metrics import explicit_mask, binary_metrics, threshold_by_youden
from .model import build_model


def set_seed(seed):
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict_probs(model, loader, device):
    """Run inference and return sigmoid probabilities in manifest order."""
    model.eval()
    probs = []
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
    return np.concatenate(probs)


def evaluate_dev(model, dev_loader, device, explicit_idx):
    """Dev AUROC and AP over explicit labels only (model-selection metrics)."""
    probs = predict_probs(model, dev_loader, device)
    metrics = binary_metrics(dev_loader.dataset.targets[explicit_idx],
                             probs[explicit_idx], threshold=0.5)
    return metrics['auroc'], metrics['ap']


def run(cfg):
    """Train one label strategy and export predictions.

    Args:
        cfg: validated configuration dict (see chexpert_uncertainty.config).

    Returns:
        Path of the strategy's output directory.
    """
    strategy = cfg['strategy']
    out_dir = os.path.join(cfg['outputs'], strategy)
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg['training']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[train] strategy={strategy} device={device}')

    data = cfg['data']
    train_ds = CheXpertDataset(data['train_manifest'], data['image_root'],
                               strategy, train=True, image_size=data['image_size'],
                               augment=cfg['training']['augment'])
    dev_ds = CheXpertDataset(data['dev_manifest'], data['image_root'],
                             strategy, train=False, image_size=data['image_size'],
                             augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'],
                              shuffle=True, num_workers=data['num_workers'])
    dev_loader = DataLoader(dev_ds, batch_size=cfg['training']['batch_size'],
                            shuffle=False, num_workers=data['num_workers'])

    model = build_model(cfg['model']).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'],
                                 weight_decay=cfg['training']['weight_decay'])

    dev_explicit = explicit_mask(dev_ds.raw)
    n_dev_explicit = int(dev_explicit.sum())
    print(f'[train] train images served={len(train_ds)}  '
          f'dev images={len(dev_ds)} (explicit {n_dev_explicit})')
    print(f'[train] train label counts: '
          f'{dict(zip(*np.unique(train_ds.targets, return_counts=True)))}')

    log_rows = []
    best_auroc, best_epoch, patience_left = -1.0, 0, cfg['training']['early_stopping_patience']
    started = time.time()

    for epoch in range(1, cfg['training']['epochs'] + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for images, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)).ravel(),
                             targets.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
            seen += len(targets)
        train_loss = total_loss / max(seen, 1)

        dev_auroc, dev_ap = evaluate_dev(model, dev_loader, device, dev_explicit)
        row = {'epoch': epoch, 'train_loss': round(train_loss, 6),
               'dev_auroc': round(dev_auroc, 6), 'dev_ap': round(dev_ap, 6),
               'lr': optimizer.param_groups[0]['lr'],
               'elapsed_s': round(time.time() - started, 1)}
        log_rows.append(row)
        print(f'[train] epoch {epoch}/{cfg["training"]["epochs"]} '
              f'loss={train_loss:.4f} dev_auroc={dev_auroc:.4f} dev_ap={dev_ap:.4f} '
              f'({time.time() - started:.0f}s)', flush=True)

        if dev_auroc > best_auroc:
            best_auroc, best_epoch = dev_auroc, epoch
            patience_left = cfg['training']['early_stopping_patience']
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
        else:
            patience_left -= 1
        torch.save(model.state_dict(), os.path.join(out_dir, 'last.pt'))
        if patience_left <= 0:
            print(f'[train] early stopping after {epoch} epochs (patience)')
            break

    # --- predictions with the best checkpoint ---
    model.load_state_dict(torch.load(os.path.join(out_dir, 'best.pt'),
                                     map_location=device))
    for name, loader, ds, tag in (('dev', dev_loader, dev_ds, 'dev'),
                                  ('eval', None, None, 'eval')):
        if name == 'eval':
            final_manifest = data.get('final_eval_manifest')
            if not final_manifest:
                continue
            final_ds = CheXpertDataset(final_manifest, data['image_root'], strategy,
                                       train=False, image_size=data['image_size'],
                                       augment=False)
            loader = DataLoader(final_ds, batch_size=cfg['training']['batch_size'],
                                shuffle=False, num_workers=data['num_workers'])
            ds = final_ds
        probs = predict_probs(model, loader, device)
        pred_df = ds.manifest.copy()
        pred_df['prob'] = probs
        pred_df[['image_path', 'patient_id', 'study_id', 'pleural_effusion_raw', 'prob']] \
            .to_csv(os.path.join(out_dir, f'{tag}_preds.csv'), index=False)
        print(f'[train] saved {tag}_preds.csv ({len(pred_df)} rows)')

    run_info = {
        'strategy': strategy,
        'device': device,
        'train_images_served': len(train_ds),
        'dev_images_served': len(dev_ds),
        'dev_explicit': n_dev_explicit,
        'best_epoch': best_epoch,
        'best_dev_auroc': best_auroc,
        'runtime_s': round(time.time() - started, 1),
        'config': {k: v for k, v in cfg.items() if k != 'outputs'},
    }
    with open(os.path.join(out_dir, 'run.json'), 'w') as f:
        json.dump(run_info, f, indent=2)
    with open(os.path.join(out_dir, 'log.csv'), 'w') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f'[train] done. best dev AUROC={best_auroc:.4f} at epoch {best_epoch}. '
          f'artifacts in {out_dir}/')
    return out_dir
