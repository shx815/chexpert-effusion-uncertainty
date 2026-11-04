# Uncertain Label Handling for Pleural Effusion Classification on CheXpert

SDSC 6018 Health Data Analytics · Group 23

## Research question

How does the treatment of uncertain training labels affect pleural effusion classification on frontal chest radiographs when the model, patient split, and training protocol are held fixed?

CheXpert training labels are extracted from radiology reports. A report may describe a possible effusion, explicitly confirm or rule it out, or leave it unmentioned. Treating possible effusions as positive or negative changes the supervision the model receives. Excluding them reduces the number of training examples.

## Dataset and preparation

The project uses frontal radiographs from [CheXpert](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays) and the `Pleural Effusion` label. The original dataset contains 224,316 radiographs from 65,240 patients.

| Raw label | Meaning |
| --- | --- |
| `1` | Positive |
| `0` | Negative |
| `-1` | Uncertain |
| Blank | Unmentioned in the report |

1. Audit image paths, unreadable files, and duplicates. Summarize the four raw label categories before exclusions and remapping.
2. Draw a fixed, seeded sample of up to 10,000 patients from the official training data, with the final size determined by the GPU pilot. Retain eligible frontal images and split patients 80:20 into training and development sets before transforming labels. Use the same patient lists across strategies.
3. Reserve the official expert-labeled validation set for final evaluation. Verify that patients do not overlap across the three sets and report counts at the patient, study, and image levels.
4. Resize images while preserving aspect ratio, pad to 224 × 224, replicate the grayscale channel three times, and normalize for the pretrained model.

## Experimental design

The planned model is an ImageNet-pretrained DenseNet-121 with a single sigmoid output and binary cross-entropy loss. All strategies exclude blank training labels and share preprocessing, pretrained initialization, seed, and training budget.

| Strategy | Treatment of uncertain training labels |
| --- | --- |
| U-Zeroes (reference) | Map `-1` to negative (`0`) |
| U-Ones | Map `-1` to positive (`1`) |
| U-Ignore | Exclude examples labeled `-1` from training |

The proposed training setup uses Adam with a learning rate of 0.0001, a batch size of 32, and up to 10 epochs. A constant predictor based on prevalence among explicit positive and negative training labels serves as a sanity baseline. The comparison includes the effect of using fewer training examples under U-Ignore.

## Evaluation

- Use only explicit positive and negative development labels. Select each model's checkpoint by development AUROC and its decision threshold by maximizing Youden's J on the development set.
- Freeze the selected checkpoints and thresholds before evaluating on the official expert-labeled set. Average probabilities across frontal images from the same study for study-level evaluation.
- Report AUROC as the primary metric, with average precision (AP), sensitivity, and specificity as secondary metrics. Compare ROC and precision–recall curves, confusion matrices, and representative errors and disagreements.

## Repository status and data access

This repository currently documents the proposed protocol. The final sample size, training runs, and results are pending.

Obtain CheXpert through the [Stanford AIMI dataset page](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays). Store radiographs, raw labels, and patient-level manifests in an ignored local directory such as `data/`. Share code and aggregate results in accordance with the dataset terms.

## References

- Irvin et al. (2019). [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://doi.org/10.1609/aaai.v33i01.3301590).
- Huang et al. (2017). [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993).
