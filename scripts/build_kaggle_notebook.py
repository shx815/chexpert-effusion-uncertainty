# -*- coding: utf-8 -*-
"""Build notebooks/kaggle_run.ipynb, the one-click formal run on Kaggle.

Edit the INTRO / CELLS strings below and re-run this script to regenerate
the notebook; commit the notebook together with the script.
Usage:  python scripts/build_kaggle_notebook.py
"""
from pathlib import Path

import nbformat as nbf

INTRO = """# Group 23 formal training (one-click run on Kaggle)

Three settings before running:

1. **GPU**: right-hand Settings → Accelerator → `GPU P100` (or T4 x2)
2. **Internet**: Settings → Internet → `On` (needed to clone the repo)
3. **Data**: right-hand Data → Add Input → search `ashery/chexpert`, click Add

The dataset is auto-mounted; the actual path may differ per account (Cell 1
prints it with `ls`, expecting `train.csv`, `valid.csv`, `train/`, `valid/`).
Run the cells top to bottom. Total runtime: about 2-3 hours."""

CELLS = [
    """# 1. Install dependencies and clone the repo
# If you see "Could not resolve host": the Internet toggle is on but the
# session was not restarted. Enable Internet, restart, and re-run this cell.
import os, sys
REPO = '/kaggle/working/chexpert-effusion-uncertainty'
if not os.path.isdir(REPO):
    !git clone --depth 1 https://github.com/shx815/chexpert-effusion-uncertainty {REPO}
assert os.path.isdir(REPO), 'Clone failed: set Settings→Internet to On, restart the session, then re-run this cell'
os.chdir(REPO)
!pip install -q pyyaml
!python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
!ls /kaggle/input/datasets/ashery/chexpert""",

    """# 2. Smoke test (optional, ~1 min): validate the environment on the 32-image sample
!python scripts/train.py --config configs/smoke.yaml --strategy uzeroes
!python scripts/eval.py --outputs outputs_smoke --strategies uzeroes""",

    """# 3. Data audit (~30-60 min): decode-check all frontal images and write manifests
!python scripts/audit_chexpert.py --data-root /kaggle/input/datasets/ashery/chexpert --output-dir data/audit --workers 4
import pandas as pd
display(pd.read_csv('data/audit/label_distribution.csv'))""",

    """# 4. Patient split: seeded sample of up to 10,000 patients, 80/20,
#    with an overlap check against the final evaluation set
!python scripts/sample_split.py \\
    --pool data/audit/train_pool.csv \\
    --final data/audit/final_eval.csv \\
    --output-dir data/splits""",

    """# 5. Formal training: three strategies (~20-40 min each). Same split;
#    only the meaning of -1 training labels differs between strategies.
!python scripts/train.py --config configs/kaggle.yaml --strategy uzeroes
!python scripts/train.py --config configs/kaggle.yaml --strategy uones
!python scripts/train.py --config configs/kaggle.yaml --strategy uignore""",

    """# 6. Final evaluation: Youden's J threshold frozen on dev explicit labels,
#    study-level metrics on the official expert-labelled set
!python scripts/eval.py --outputs outputs
import pandas as pd
from IPython.display import Image, display
display(pd.read_csv('outputs/evaluation/comparison.csv'))
display(Image('outputs/evaluation/roc_pr_comparison.png'))""",

    """# 7. Package artifacts (logs, checkpoints, predictions, evaluation plots)
!zip -r /kaggle/working/g23_outputs.zip outputs
print('Download g23_outputs.zip from the Output panel on the right')
print('Also recommended: Save Version (top right) so artifacts are kept')
!ls -lh outputs/uzeroes""",
]


def main():
    nb = nbf.v4.new_notebook()
    nb['cells'] = [nbf.v4.new_markdown_cell(INTRO)] + [
        nbf.v4.new_code_cell(c) for c in CELLS]
    nb['metadata'] = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                       'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10'},
    }
    out = Path(__file__).resolve().parent.parent / 'notebooks' / 'kaggle_run.ipynb'
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    print(f'wrote {out} ({len(nb.cells)} cells)')


if __name__ == '__main__':
    main()
