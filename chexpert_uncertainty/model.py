# -*- coding: utf-8 -*-
"""Model: ImageNet-pretrained DenseNet-121 with one sigmoid output.

DenseNet-121 is the standard backbone for CheXpert benchmarks and provides a
good accuracy/compute trade-off on a single GPU. The final linear layer is
replaced by a single logit; the loss (BCEWithLogitsLoss) applies the sigmoid.
"""
import torch.nn as nn
from torchvision import models


def build_model(name='densenet121', pretrained=True):
    """Build the binary classifier.

    Args:
        name: backbone name; only densenet121 is supported.
        pretrained: load ImageNet weights (IMAGENET1K_V1).

    Returns:
        The model with an untrained 1-unit classifier head.
    """
    if name != 'densenet121':
        raise ValueError(f'unsupported model {name!r} (only densenet121)')
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1
                               if pretrained else None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    return model
