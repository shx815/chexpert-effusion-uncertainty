# CheXpert 胸腔积液分类中的不确定标签处理

简体中文 | [English](README.en.md)

SDSC 6018 Health Data Analytics · 第 23 组

## 研究问题

在模型、患者划分和训练流程相同的条件下，如何处理不确定训练标签，会对正位胸片的胸腔积液分类表现产生什么影响？

CheXpert 训练标签从放射科报告自动提取。报告可能明确描述有积液、明确否定、表述不确定，或者未提及积液。将不确定标签映射为阳性或阴性会改变模型接收的监督信号，排除这些标签则会减少训练样本。

## 数据与预处理

项目使用 CheXpert 的正位胸片和 `Pleural Effusion` 标签。原始数据集包含 65,240 名患者的 224,316 张胸片，本项目使用低分辨率版 `CheXpert-v1.0-small`。[Stanford 数据集说明](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays)

| 原始标签 | 含义 |
| --- | --- |
| `1` | 阳性 |
| `0` | 阴性 |
| `-1` | 不确定 |
| 空白 | 报告未提及 |

1. 检查图像路径、无法读取的文件和重复项。筛选正位胸片，在排除空白和转换标签前统计四类原始标签。
2. 根据 GPU pilot 确定正式规模，从官方训练来源用固定种子抽取最多 10,000 名患者。保留合格正位胸片，按患者以 80:20 划分训练集和开发集，再执行标签转换。三种策略共用患者划分。
3. 保留官方专家标注验证集用于最终评估。核查三组患者无重叠，并报告患者数、检查数和图像数。
4. 保持宽高比缩放图像，填充到 224 × 224，将灰度复制为三个通道，并按预训练模型要求归一化。

## 实验设计

计划使用 ImageNet 预训练的 DenseNet-121，配一个 Sigmoid 输出和二元交叉熵损失。三组统一排除空白训练标签，共用预处理、预训练初始化、随机种子和训练预算。

| 策略 | 不确定训练标签的处理 |
| --- | --- |
| U-Zeroes（对照） | 将 `-1` 映射为阴性 `0` |
| U-Ones | 将 `-1` 映射为阳性 `1` |
| U-Ignore | 排除标为 `-1` 的训练样本 |

计划使用 Adam，学习率 0.0001，批大小 32，最多训练 10 个 epoch。另用明确阳性和阴性训练样本中的阳性比例构建常数预测基线。比较结果也包含 U-Ignore 有效训练样本减少带来的影响。

## 评估

- 开发集仅使用明确的阳性和阴性标签。按开发集 AUROC 选择模型检查点，再在开发集上最大化 Youden's J 选择阈值。
- 固定检查点和阈值后，在官方专家标注集上做最终评估。同一次检查含多张正位胸片时，平均预测概率后计算检查级指标。
- 主要指标为 AUROC，次要指标为平均精确率（AP）、敏感度和特异度。比较 ROC 曲线、精确率–召回率曲线、混淆矩阵及代表性错例。

## 项目进度与数据访问

数据检查、试读样例、患者划分与训练评估流水线均已完成，正式训练待 GPU 环境确定后执行。全量数据、原始标签和生成的清单存放在 Git 忽略的本地 `data/` 目录。

## 目录结构

```text
chexpert-effusion-uncertainty/
├── README.md                     # 默认中文说明
├── README.en.md                  # 英文说明
├── .gitignore                    # 忽略本地数据和训练产物
├── configs/
│   ├── experiment.yaml           # 正式实验配置（三种策略共用）
│   └── smoke.yaml                # 32 张样例冒烟测试配置
├── chexpert_uncertainty/         # Python 包
│   ├── config.py                 # 配置加载与默认值
│   ├── data.py                   # 清单读取、标签策略、图像变换
│   ├── model.py                  # DenseNet-121 单 Sigmoid 输出
│   ├── train.py                  # 训练循环、检查点、预测导出
│   └── metrics.py                # Youden 阈值、二分类指标、检查级汇总
├── examples/loader_sample/       # 32 张真实胸片和试读清单
├── scripts/
│   ├── audit_chexpert.py         # 检查胸片并生成清单
│   ├── sample_split.py           # 患者级抽样与 80:20 划分
│   ├── train.py                  # 单个策略的训练入口
│   └── eval.py                   # 阈值选择与最终评估
└── data/                         # Git 忽略的本地文件
    ├── .download-venv/           # 安装 KaggleHub 和 Pillow 的本地环境
    ├── datasets/ashery/chexpert/versions/1/
    │   ├── train/                # 按患者和检查组织的训练来源胸片
    │   ├── train.csv             # 原始 CheXpert 训练标签
    │   ├── valid/                # 留作最终评估的官方验证胸片
    │   └── valid.csv             # 官方专家标签
    └── audit/
        ├── train_pool.csv        # 抽样划分前的合格正位胸片
        ├── final_eval.csv        # 最终评估清单
        ├── loader_sample.csv     # 本地生成的 32 张试读清单
        ├── label_distribution.csv # 图像级标签数量和比例
        ├── audit_summary.json    # 筛选计数、样本量和患者重叠检查
        └── issues.csv            # 逐行记录数据质量问题
```

## 克隆后试读样例

使用 `examples/loader_sample/manifest.csv` 作为清单，`examples/loader_sample/` 作为图像根目录。样例含 32 张真实正位胸片，四种原始标签各 8 张，用来验证 DataLoader 的路径解析、图像变换和标签读取。 [样例说明](examples/loader_sample/README.md)

### 清单字段与 DataLoader 约定

| 字段 | 读取规则 |
| --- | --- |
| `image_path` | 相对路径。与配置的图像根目录拼接，试读使用 `examples/loader_sample/`，正式实验使用 CheXpert 解压目录。 |
| `patient_id` | 患者标识，用于患者级划分和重叠检查。 |
| `study_id` | 检查标识。与 `patient_id` 一起唯一标识一次检查，用于汇总同次检查的图像预测。 |
| `pleural_effusion_raw` | 原始标签，取值为 `1`、`0`、`-1` 或空白。读取时显式识别空字段，例如 Pandas 默认可能将其读为 `NaN`。监督训练统一排除空白，再按实验策略处理 `-1`。 |
| `split` | 试读清单中的 `train_pool` 表示训练候选池。正式清单用 `train` 和 `dev` 区分训练集与开发集，官方评估清单使用 `final_eval`。 |

DataLoader 按这些字段读取清单，通过配置切换图像根目录。正式清单沿用相同的列名和原始标签格式。

## 运行数据检查

在 Python 环境中安装 Pillow 后，从仓库根目录运行下面的命令。如果数据解压在别处，把 `--data-root` 改为包含 `train.csv` 和 `valid.csv` 的目录。

```sh
python scripts/audit_chexpert.py \
  --data-root data/datasets/ashery/chexpert/versions/1 \
  --output-dir data/audit
```

脚本逐张解码候选正位 JPEG，并生成清单、检查记录和标签统计。输出清单中的路径相对于 `--data-root`，使用 `patient_id` 和 `study_id` 共同识别一次检查。原始标签保留为 `1`、`0`、`-1` 或空白字段。

## 患者划分与正式训练

依赖：Python 3.9+，PyTorch、torchvision、pandas、numpy、scikit-learn、matplotlib、PyYAML。

1. 患者级抽样与划分。用固定种子 6018 从训练池抽取最多 10,000 名患者，按患者 80:20 划分，并复查与最终评估集无患者重叠：

   ```sh
   python scripts/sample_split.py \
     --pool data/audit/train_pool.csv \
     --final data/audit/final_eval.csv \
     --output-dir data/splits
   ```

2. 正式训练。三种策略共用 `configs/experiment.yaml`，只需修改 `image_root`。每个策略写入 `outputs/<strategy>/`：`log.csv`（逐 epoch 损失与开发集指标）、`best.pt`（按开发集 AUROC 选择）、`last.pt`、`run.json`（配置快照与耗时）、`dev_preds.csv` 和 `eval_preds.csv`：

   ```sh
   python scripts/train.py --config configs/experiment.yaml --strategy uzeroes
   python scripts/train.py --config configs/experiment.yaml --strategy uones
   python scripts/train.py --config configs/experiment.yaml --strategy uignore
   ```

3. 最终评估。在开发集明确标签上以 Youden's J 选择阈值并冻结，在官方专家标注集上按检查级（同次检查多张胸片取平均概率）计算指标，输出 `outputs/evaluation/`：

   ```sh
   python scripts/eval.py --outputs outputs
   ```

本地冒烟测试（无需下载全量数据）：`python scripts/train.py --config configs/smoke.yaml`，在 32 张样例上以 2 个 epoch 跑通完整流程，产物写入 `outputs_smoke/`。

## 参考文献

- Irvin et al. (2019). [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://doi.org/10.1609/aaai.v33i01.3301590).
- Huang et al. (2017). [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993).
