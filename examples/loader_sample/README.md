# CheXpert loader sample

Set the image root to this directory and read `manifest.csv`. The manifest
contains 32 frontal images from the training pool, with eight examples each
of positive, negative, uncertain, and unmentioned pleural effusion labels.
Keep the raw labels for loader checks and exclude blank labels before
supervised training. Use this sample for interface testing.

Source: [CheXpert-v1.0-small on Kaggle](https://www.kaggle.com/datasets/ashery/chexpert/data),
version 1. The Kaggle listing displays CC0. Images and labels were copied
from that package.
