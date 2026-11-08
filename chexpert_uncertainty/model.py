# -*- coding: utf-8 -*-
"""Model: ImageNet-pretrained DenseNet-121 with one sigmoid output."""
import torch.nn as nn
from torchvision import models


def build_model(name='densenet121', pretrained=True):
    if name != 'densenet121':
        raise ValueError(f'unsupported model {name!r} (only densenet121)')
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1
                               if pretrained else None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    return model
