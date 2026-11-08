# -*- coding: utf-8 -*-
"""Build notebooks/kaggle_run.ipynb, the one-click formal run on Kaggle.

Usage:  python scripts/build_kaggle_notebook.py
"""
from pathlib import Path

import nbformat as nbf

INTRO = """# Group 23 正式训练(Kaggle 一键运行)

跑之前先做 3 个设置:

1. **GPU**:右侧 Settings → Accelerator 选 `GPU P100`
2. **联网**:Settings → Internet 选 `On`(下载代码需要)
3. **数据**:右侧 Data → Add Input → 搜索 `ashery/chexpert`,点 Add

数据集会自动挂载到 `/kaggle/input/chexpert`,不用下载 11GB。
然后从上到下依次运行每一格。总耗时约 2-3 小时。"""

CELLS = [
    """# 1. 依赖 + 下载代码
# 如果报"Could not resolve host":Internet 虽然开了,但要 Restart 重启会话才生效
import os, sys
REPO = '/kaggle/working/chexpert-effusion-uncertainty'
if not os.path.isdir(REPO):
    !git clone --depth 1 https://github.com/shx815/chexpert-effusion-uncertainty {REPO}
assert os.path.isdir(REPO), '克隆失败:确认 Settings→Internet 是 On,点 Restart 重启会话后再运行本格'
os.chdir(REPO)
!pip install -q pyyaml
!python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
!ls /kaggle/input/chexpert""",

    """# 2. 冒烟测试(可选,约 1 分钟):在 32 张样例上验证环境
!python scripts/train.py --config configs/smoke.yaml --strategy uzeroes
!python scripts/eval.py --outputs outputs_smoke --strategies uzeroes""",

    """# 3. 数据审计(约 30-60 分钟):解码检查 19 万张正位胸片,生成清单
!python scripts/audit_chexpert.py --data-root /kaggle/input/chexpert --output-dir data/audit --workers 4
import pandas as pd
display(pd.read_csv('data/audit/label_distribution.csv'))""",

    """# 4. 患者划分:种子抽样最多 10,000 患者 + 80:20,并检查与最终评估集无重叠
!python scripts/sample_split.py \\
    --pool data/audit/train_pool.csv \\
    --final data/audit/final_eval.csv \\
    --output-dir data/splits""",

    """# 5. 正式训练:三个策略(约 20-40 分钟/个,共用同一划分,只有 -1 标签含义不同)
!python scripts/train.py --config configs/kaggle.yaml --strategy uzeroes
!python scripts/train.py --config configs/kaggle.yaml --strategy uones
!python scripts/train.py --config configs/kaggle.yaml --strategy uignore""",

    """# 6. 最终评估:开发集 Youden's J 定阈值,官方专家标注集上按检查级算指标
!python scripts/eval.py --outputs outputs
import pandas as pd
from IPython.display import Image, display
display(pd.read_csv('outputs/evaluation/comparison.csv'))
display(Image('outputs/evaluation/roc_pr_comparison.png'))""",

    """# 7. 打包产物下载(训练日志、checkpoint、预测、评估图表)
!zip -r /kaggle/working/g23_outputs.zip outputs
print('去右侧 Output 面板下载 g23_outputs.zip')
print('另外建议:右上角 Save Version 保存一份,产物会保留')
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
