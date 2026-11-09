# Uncertain Label Handling for Pleural Effusion Classification on CheXpert

English | [简体中文](README.zh.md)

SDSC 6018 Health Data Analytics · Group 23

## Research question

How does the treatment of uncertain training labels affect pleural effusion classification on frontal chest radiographs when the model, patient split, and training protocol are held fixed?

CheXpert training labels are extracted from radiology reports. A report may describe a possible effusion, explicitly confirm or rule it out, or leave it unmentioned. Treating possible effusions as positive or negative changes the supervision the model receives. Excluding them reduces the number of training examples.

## Dataset and preparation

The project uses frontal radiographs from [CheXpert](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays) and the `Pleural Effusion` label. The original dataset contains 224,316 radiographs from 65,240 patients; this project uses the low-resolution release `CheXpert-v1.0-small`.

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

The data audit, loader sample, patient split, and the full training/evaluation pipeline are implemented and smoke-tested locally on the 32-image sample. Formal training runs are pending on GPU (Kaggle).

Obtain CheXpert through the [Stanford AIMI dataset page](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays) or the Kaggle dataset [ashery/chexpert](https://www.kaggle.com/datasets/ashery/chexpert) (CheXpert-v1.0-small, no registration needed). Store the full dataset, raw labels, and generated manifests in the ignored local `data/` directory. Share code and aggregate results in accordance with the dataset terms.

## References

- Irvin et al. (2019). [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://doi.org/10.1609/aaai.v33i01.3301590).
- Huang et al. (2017). [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993).

## Repository layout

```text
chexpert-effusion-uncertainty/
├── README.md                    # Project documentation (English, default)
├── README.zh.md                 # Simplified Chinese documentation
├── .gitignore                   # Keeps local data and training artifacts out of Git
├── configs/
│   ├── experiment.yaml          # Formal experiment configuration shared by all strategies
│   ├── kaggle.yaml              # Kaggle run configuration (dataset auto-mounted)
│   └── smoke.yaml               # 32-image local smoke test configuration
├── chexpert_uncertainty/        # Python package
│   ├── config.py                # Config loading with validated defaults
│   ├── data.py                  # Manifest loading, label strategies, image transforms
│   ├── model.py                 # DenseNet-121 with a single sigmoid output
│   ├── train.py                 # Training loop, checkpoints, prediction export
│   └── metrics.py               # Youden's J threshold, binary metrics, study-level aggregation
├── examples/loader_sample/      # 32 real images and their loader test manifest
├── notebooks/
│   └── kaggle_run.ipynb         # One-click formal run on Kaggle (generated, see scripts/build_kaggle_notebook.py)
├── scripts/
│   ├── audit_chexpert.py        # Screens frontal images and generates manifests
│   ├── sample_split.py          # Patient-level sampling and 80/20 split
│   ├── train.py                 # Training entry point for one strategy
│   ├── eval.py                  # Threshold selection and final evaluation
│   └── build_kaggle_notebook.py # Generates notebooks/kaggle_run.ipynb
└── data/                        # Local files ignored by Git
    ├── .download-venv/          # Local Python environment with KaggleHub and Pillow
    ├── datasets/ashery/chexpert/versions/1/
    │   ├── train/               # Training-source images organized by patient and study
    │   ├── train.csv            # Original report-derived CheXpert labels
    │   ├── valid/               # Official validation images reserved for final evaluation
    │   └── valid.csv            # Expert labels for the official validation set
    └── audit/
        ├── train_pool.csv       # Eligible frontal images before patient sampling and splitting
        ├── final_eval.csv       # Eligible frontal images for final evaluation
        ├── loader_sample.csv    # 32-image sample for checking paths and all four raw labels
        ├── label_distribution.csv # Image-level label counts and percentages by source
        ├── audit_summary.json   # Screening counts, cohort sizes, and patient overlap check
        └── issues.csv           # Row-level log of data quality issues
```

The dataset path shown is the local KaggleHub cache layout. If you unpack the
dataset elsewhere, pass the directory containing `train.csv` and `valid.csv`
as `--data-root`.

## Test the data loader

Use `examples/loader_sample/manifest.csv` as the manifest and
`examples/loader_sample/` as the image root. The sample has 32 frontal images
from the training pool, with eight per raw label state. See the
[sample instructions](examples/loader_sample/README.md).

### Manifest fields and DataLoader conventions

| Field | How to read it |
| --- | --- |
| `image_path` | Relative path. Join it to the configured image root. Use `examples/loader_sample/` for the loader test and the extracted CheXpert directory for the formal experiment. |
| `patient_id` | Patient identifier for patient-level splitting and overlap checks. |
| `study_id` | Study identifier. Use it together with `patient_id` to identify a study and aggregate predictions across its images. |
| `pleural_effusion_raw` | Raw label with values `1`, `0`, `-1`, or blank. Handle empty fields explicitly. Pandas may parse them as `NaN` by default. Exclude blank labels consistently from supervised training, then apply the selected strategy to `-1`. |
| `split` | `train_pool` in the loader sample indicates the training candidate pool. Formal manifests use `train` and `dev` for training and development, and `final_eval` for the official evaluation set. |

The DataLoader reads these fields and uses configuration to switch image roots. Formal manifests use the same column names and raw label encoding.

## Run the data audit

Install Pillow in your Python environment, then run from the repository root:

```sh
python scripts/audit_chexpert.py \
  --data-root data/datasets/ashery/chexpert/versions/1 \
  --output-dir data/audit
```

The audit checks frontal image paths and decodes each candidate JPEG. It writes
`train_pool.csv`, `final_eval.csv`, a data-quality log, and label counts.
Paths in the manifests are relative to `--data-root`. Use `patient_id` and
`study_id` together to identify a study. The raw target is encoded as
`1`, `0`, `-1`, or an empty CSV field.

`loader_sample.csv` is a fixed-seed, label-balanced interface fixture drawn
from the training pool. Use it to check path resolution and handling of all
four raw labels. Apply the shared blank-label exclusion before supervised
training. The fixture is separate from the formal patient sample and split.

## Patient split and formal training

Dependencies: Python 3.9+, PyTorch, torchvision, pandas, numpy, scikit-learn,
matplotlib, PyYAML.

1. Patient-level sampling and split. Sample up to 10,000 patients from the
   training pool with fixed seed 6018, split patients 80/20, and re-check that
   no patient overlaps with the final evaluation set:

   ```sh
   python scripts/sample_split.py \
     --pool data/audit/train_pool.csv \
     --final data/audit/final_eval.csv \
     --output-dir data/splits
   ```

2. Formal training. The three strategies share `configs/experiment.yaml`;
   only `image_root` needs to be adjusted. Each strategy writes to
   `outputs/<strategy>/`: `log.csv` (per-epoch loss and dev metrics),
   `best.pt` (selected by dev AUROC), `last.pt`, `run.json` (config snapshot
   and runtime), `dev_preds.csv` and `eval_preds.csv`:

   ```sh
   python scripts/train.py --config configs/experiment.yaml --strategy uzeroes
   python scripts/train.py --config configs/experiment.yaml --strategy uones
   python scripts/train.py --config configs/experiment.yaml --strategy uignore
   ```

3. Final evaluation. Freeze the decision threshold by maximizing Youden's J on
   the dev set explicit labels, then compute study-level metrics (mean
   probability over the frontal images of a study) on the official
   expert-labeled set. Results are written to `outputs/evaluation/`:

   ```sh
   python scripts/eval.py --outputs outputs
   ```

Local smoke test (no full dataset needed): `python scripts/train.py --config
configs/smoke.yaml` runs the complete pipeline on the 32-image sample for two
epochs and writes artifacts to `outputs_smoke/`.

For a one-click run on Kaggle, import `notebooks/kaggle_run.ipynb`, attach the
`ashery/chexpert` dataset, and run the cells top to bottom.
