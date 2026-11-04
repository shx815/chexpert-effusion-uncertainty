# Uncertain Label Handling for Pleural Effusion Classification

SDSC 6018 Health Data Analytics course project, Group 23.

We compare three ways to handle uncertain pleural effusion labels when training an ImageNet-pretrained DenseNet-121 on frontal CheXpert radiographs.

| Strategy | Treatment of uncertain training labels (`-1`) |
| --- | --- |
| U-Zeroes | Map to negative (`0`) |
| U-Ones | Map to positive (`1`) |
| U-Ignore | Exclude from training |

## Planned data and evaluation protocol

- Audit image files and summarize the original positive, negative, uncertain, and unmentioned labels. Exclude unmentioned labels consistently across strategies.
- Sample up to 10,000 patients from the official training data, subject to the GPU pilot. Use a fixed seed and an 80:20 patient-level training and development split shared across strategies.
- Use explicit positive and negative development labels for checkpoint and threshold selection. Reserve the official expert-labeled validation set for final study-level evaluation.
- Compare AUROC, average precision, sensitivity, and specificity.

## Data access

Obtain CheXpert through the [Stanford AIMI dataset page](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays). Keep radiographs, raw labels, and patient-level manifests in a local ignored directory such as `data/`. Share code and aggregate results in accordance with the dataset terms.
