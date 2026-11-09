# -*- coding: utf-8 -*-
"""Manifest loading, label strategies and image transforms.

Manifest convention (see README): columns image_path, patient_id, study_id,
pleural_effusion_raw, split. Raw labels are 1 / 0 / -1 / blank.

Label strategies. The patient split is identical across strategies; only the
meaning of the -1 training labels changes:
  uzeroes : -1 -> 0 (reference)
  uones   : -1 -> 1
  uignore : -1 rows are excluded from training
Blank rows are always excluded from supervised training.

Preprocessing: aspect ratio is preserved by padding the shorter side to a
square and resizing to image_size x image_size; the single-channel radiograph
is replicated to three channels and normalised with ImageNet statistics so
the ImageNet-pretrained backbone can be used directly.
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

RAW_COL = 'pleural_effusion_raw'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_manifest(path):
    """Read a manifest CSV, keeping blank labels as empty strings (not NaN).

    The raw label column is forced to string dtype: a CSV whose labels are all
    numeric-looking would otherwise be inferred as int64 and break the
    string-based strategy masks.
    """
    return pd.read_csv(path, keep_default_na=False, dtype={RAW_COL: str})


def apply_strategy(labels, strategy):
    """Map raw labels to binary training targets for one strategy.

    Args:
        labels: raw pleural_effusion labels ('1', '0', '-1' or '').
        strategy: one of uzeroes, uones, uignore.

    Returns:
        keep: bool array, True for rows included in supervised training.
        binary: float array of 0/1 targets, aligned with the input labels.
    """
    labels = pd.Series(labels)
    if strategy == 'uignore':
        mask = labels.isin(('0', '1'))
    else:
        mask = labels.isin(('0', '1', '-1'))
    binary = labels.map({'0': 0.0, '1': 1.0,
                         '-1': 1.0 if strategy == 'uones' else 0.0})
    return mask.to_numpy(), binary.to_numpy()


class CheXpertDataset(torch.utils.data.Dataset):
    """Serves (image, target) pairs for one manifest and one strategy.

    Only rows kept by the strategy are served; dev metrics are computed on the
    explicit-label subset of those rows (see metrics.explicit_mask).
    """

    def __init__(self, manifest_path, image_root, strategy, train=False,
                 image_size=224, augment=True):
        self.manifest = load_manifest(manifest_path)
        self.image_root = image_root
        self.strategy = strategy
        self.train = train
        keep, targets = apply_strategy(self.manifest[RAW_COL], strategy)
        self.manifest = self.manifest[keep].reset_index(drop=True)
        self.targets = targets[keep].astype(np.float32)
        # Raw labels of the rows actually served, for metric filtering / logging.
        self.raw = self.manifest[RAW_COL].to_numpy()
        self.transform = make_transform(train and augment, image_size)
        print(f'[data] {manifest_path}: {len(self.manifest)} images served '
              f'(strategy={strategy}, train={train})')

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, i):
        row = self.manifest.iloc[i]
        path = f'{self.image_root}/{row.image_path}'
        with Image.open(path) as image:
            image = image.convert('L')               # CheXpert images are grayscale
            image = self.transform(image)
        return image, torch.tensor(self.targets[i])


class SquarePad:
    """Pad the shorter side so the image becomes a square (no resize).

    Applied before Resize(image_size), so the aspect ratio is preserved.
    """

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        side = max(w, h)
        pad_w, pad_h = side - w, side - h
        left, right = pad_w // 2, pad_w - pad_w // 2
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        return transforms.functional.pad(img, (left, top, right, bottom),
                                         fill=self.fill)


def make_transform(augment, image_size):
    """Build the preprocessing pipeline for an image size.

    Order: grayscale to 3 channels, optional horizontal flip, square pad,
    resize, tensor conversion, ImageNet normalisation. A horizontal flip is
    the only augmentation: flipping a chest radiograph left-right is safe for
    pleural effusion and acts as light regularisation.
    """
    steps = [transforms.Grayscale(num_output_channels=3)]
    if augment:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps += [
        SquarePad(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)
